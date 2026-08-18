from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterable
import urllib.error
import urllib.request

from .utils import utc_now


ROOT = Path(__file__).resolve().parents[2]
PACKAGED_PROFILE_DIR = Path(__file__).with_name("native")
DEVELOPMENT_PROFILE_DIR = ROOT / "work" / "gpt56_native_profile"
PROFILE_DIR = PACKAGED_PROFILE_DIR if PACKAGED_PROFILE_DIR.is_dir() else DEVELOPMENT_PROFILE_DIR
NODE_TOOL = PROFILE_DIR / "codex-native-transport.mjs"
RAW_PROFILE = PROFILE_DIR / "native-0.147.0.raw"
HISTORY_FIXTURE = PROFILE_DIR / "fixed_32k_history.json"
TERMINAL_EVENTS = {"response.completed", "response.incomplete", "response.failed"}


class TransportError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        elapsed_ms: int | None = None,
        exchange: dict[str, Any] | None = None,
        stage: str | None = None,
        category: str | None = None,
        retryable: bool | None = None,
        safe_message: str | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.elapsed_ms = elapsed_ms
        self.exchange = exchange or {}
        if category is None:
            if message == "invalid Responses SSE":
                category = "truncated_or_invalid_stream"
            elif status is not None:
                category = "upstream_http_error"
            elif "timeout" in message.casefold():
                category = "timeout"
            else:
                category = "connection_or_transport_error"
        if stage is None:
            stage = "response_stream" if category == "truncated_or_invalid_stream" else ("upstream_http" if status is not None else "transport")
        if retryable is None:
            retryable = status is None or status in {408, 429, 500, 502, 503, 504} or category == "truncated_or_invalid_stream"
        self.stage = stage
        self.category = category
        self.retryable = bool(retryable)
        self.safe_message = safe_message or (category + (f" (HTTP {status})" if status is not None else ""))

    def error_info(self, attempt: int) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "category": self.category,
            "retryable": self.retryable,
            "http_status": self.status,
            "attempt": int(attempt),
            "safe_message": self.safe_message,
        }


@dataclass
class TransportResult:
    response: dict[str, Any]
    answer: str
    meta: dict[str, Any]
    exchange: dict[str, Any]


def output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str):
        return direct.strip()
    chunks = []
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "".join(chunks).strip()


class _SSECollector:
    def __init__(self) -> None:
        self.data_lines: list[str] = []
        self.terminal: dict[str, Any] | None = None
        self.events = 0

    def _consume(self) -> None:
        if not self.data_lines:
            return
        payload = "\n".join(self.data_lines)
        self.data_lines.clear()
        if payload == "[DONE]":
            return
        event = json.loads(payload)
        if not isinstance(event, dict):
            raise ValueError("SSE event is not an object")
        self.events += 1
        if event.get("type") in TERMINAL_EVENTS and isinstance(event.get("response"), dict):
            self.terminal = event["response"]

    def feed(self, raw_line: bytes) -> bool:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            self._consume()
        elif line.startswith("data:"):
            self.data_lines.append(line[5:].lstrip(" "))
        return self.terminal is not None

    def finish(self) -> tuple[dict[str, Any], int]:
        self._consume()
        if self.terminal is None:
            raise ValueError("SSE ended without a terminal response")
        return self.terminal, self.events


def parse_sse(raw: bytes) -> tuple[dict[str, Any], int]:
    collector = _SSECollector()

    for raw_line in raw.splitlines(keepends=True):
        collector.feed(raw_line)
    return collector.finish()


def load_history() -> list[dict[str, str]]:
    value = json.loads(HISTORY_FIXTURE.read_text(encoding="utf-8"))
    return [{"role": item["role"], "content": item["text"]} for item in value]


def build_payload(model: str, messages: list[dict[str, str]], effort: str, context_mode: str, cache_key: str) -> dict[str, Any]:
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("the final probe message must use user role")
    input_items = list(messages[:-1])
    if context_mode == "fixed_32k_history":
        input_items.extend(load_history())
    input_items.append(messages[-1])
    return {
        "model": model,
        "input": input_items,
        "reasoning": {"effort": effort},
        "include": ["reasoning.encrypted_content"],
        "store": False,
        "stream": True,
        "prompt_cache_key": cache_key,
    }


class StreamingTransport:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 180.0, capture_exchange: bool = False):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.capture_exchange = capture_exchange

    def post(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        effort: str,
        request_format: str,
        context_mode: str,
        cache_key: str,
        job_id: str,
        session_id: str,
    ) -> TransportResult:
        if request_format == "normal":
            return self._post_normal(model, messages, effort, context_mode, cache_key, job_id, session_id)
        if request_format == "native_codex":
            return self._post_native(model, messages, effort, context_mode, job_id, session_id)
        raise ValueError(f"unsupported request format: {request_format}")

    def _post_normal(self, model: str, messages: list[dict[str, str]], effort: str, context_mode: str, cache_key: str, job_id: str, session_id: str) -> TransportResult:
        payload = build_payload(model, messages, effort, context_mode, cache_key)
        body_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        body = body_text.encode("utf-8")
        url = self.base_url + "/responses"
        request = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": os.environ.get("GPT56_USER_AGENT", "python-urllib/3"),
        })
        started_wall = utc_now()
        started = time.perf_counter()
        first_event_at = None
        first_ms = None
        raw = bytearray()
        collector = _SSECollector()
        stream_parse_error: Exception | None = None
        status = None
        headers: dict[str, str] = {}
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = int(response.status)
                headers = {str(key): str(value) for key, value in response.headers.items() if str(key).casefold() not in {"authorization", "set-cookie"}}
                while True:
                    line = response.readline()
                    if not line:
                        break
                    raw.extend(line)
                    if first_ms is None and line.strip():
                        first_ms = round((time.perf_counter() - started) * 1000)
                        first_event_at = utc_now()
                    try:
                        terminal_seen = collector.feed(line)
                    except Exception as exc:
                        stream_parse_error = exc
                        break
                    if terminal_seen:
                        break
                    if time.perf_counter() - started > self.timeout:
                        raise TimeoutError("stream exceeded hard deadline")
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            raw.extend(exc.read())
            raise TransportError("upstream HTTP error", status=status, elapsed_ms=round((time.perf_counter() - started) * 1000), exchange=self._exchange(session_id, job_id, started_wall, first_event_at, url, "normal", context_mode, model, effort, status, headers, body_text, raw.decode("utf-8", errors="replace"), None, first_ms, 0, "upstream HTTP error")) from exc
        except Exception as exc:
            raise TransportError(type(exc).__name__, status=status, elapsed_ms=round((time.perf_counter() - started) * 1000), exchange=self._exchange(session_id, job_id, started_wall, first_event_at, url, "normal", context_mode, model, effort, status, headers, body_text, raw.decode("utf-8", errors="replace"), None, first_ms, 0, type(exc).__name__)) from exc
        elapsed = round((time.perf_counter() - started) * 1000)
        try:
            if stream_parse_error is not None:
                raise stream_parse_error
            parsed, events = collector.finish()
        except Exception as exc:
            raise TransportError("invalid Responses SSE", status=status, elapsed_ms=elapsed, exchange=self._exchange(session_id, job_id, started_wall, first_event_at, url, "normal", context_mode, model, effort, status, headers, body_text, raw.decode("utf-8", errors="replace"), None, first_ms, 0, "invalid Responses SSE")) from exc
        exchange = self._exchange(session_id, job_id, started_wall, first_event_at, url, "normal", context_mode, model, effort, status, headers, body_text, raw.decode("utf-8"), parsed, first_ms, events, None)
        exchange["elapsed_ms"] = elapsed
        return TransportResult(parsed, output_text(parsed), {"http_status": status, "elapsed_ms": elapsed, "response_headers": headers, "streaming": True, "stream_event_count": events, "time_to_first_event_ms": first_ms}, exchange)

    def _post_native(self, model: str, messages: list[dict[str, str]], effort: str, context_mode: str, job_id: str, session_id: str) -> TransportResult:
        node = os.environ.get("GPT56_NODE") or shutil.which("node")
        if not node:
            raise TransportError("Node.js is required for native Codex requests")
        url = self.base_url + "/responses"
        command = [
            node, str(NODE_TOOL), "send-native", "--profile", str(RAW_PROFILE), "--url", url,
            "--model", model, "--timeout", str(round(self.timeout * 1000)), "--input-json", "true",
            "--effort", effort, "--capture-sanitized-request", "true" if self.capture_exchange else "false",
        ]
        if context_mode == "fixed_32k_history":
            command.extend(["--history-file", str(HISTORY_FIXTURE)])
        environment = dict(os.environ)
        environment["GPT56_NATIVE_AUTHORIZATION"] = self.api_key
        started_wall = utc_now()
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, input=json.dumps(messages, ensure_ascii=False, separators=(",", ":")), text=True, encoding="utf-8", capture_output=True, timeout=self.timeout + 10, env=environment, check=False)
        finally:
            environment.pop("GPT56_NATIVE_AUTHORIZATION", None)
        elapsed = round((time.perf_counter() - started) * 1000)
        if completed.returncode != 0:
            raise TransportError("native transport failed", elapsed_ms=elapsed)
        report = json.loads(completed.stdout)
        status = int(report["status"])
        stream_text = str(report.get("body", ""))
        request_text = ""
        if self.capture_exchange:
            encoded = report.get("request", {}).get("raw_without_auth_base64")
            if not isinstance(encoded, str):
                raise TransportError("native transport omitted sanitized request", status=status, elapsed_ms=elapsed)
            request_text = base64.b64decode(encoded).decode("utf-8")
        if not 200 <= status < 300:
            exchange = self._exchange(session_id, job_id, started_wall, None, url, "native_codex", context_mode, model, effort, status, report.get("headers", {}), request_text, stream_text, None, None, 0, "native upstream HTTP error")
            raise TransportError("native upstream HTTP error", status=status, elapsed_ms=elapsed, exchange=exchange)
        try:
            parsed, events = parse_sse(stream_text.encode("utf-8"))
        except Exception as exc:
            exchange = self._exchange(session_id, job_id, started_wall, None, url, "native_codex", context_mode, model, effort, status, report.get("headers", {}), request_text, stream_text, None, None, 0, "invalid Responses SSE")
            raise TransportError("invalid Responses SSE", status=status, elapsed_ms=elapsed, exchange=exchange) from exc
        exchange = self._exchange(session_id, job_id, started_wall, None, url, "native_codex", context_mode, model, effort, status, report.get("headers", {}), request_text, stream_text, parsed, None, events, None)
        exchange["elapsed_ms"] = elapsed
        return TransportResult(parsed, output_text(parsed), {"http_status": status, "elapsed_ms": elapsed, "response_headers": report.get("headers", {}), "streaming": True, "stream_event_count": events, "time_to_first_event_ms": None, "request_profile": report.get("request", {})}, exchange)

    @staticmethod
    def _exchange(session_id: str, job_id: str, started_at: str, first_event_at: str | None, url: str, request_format: str, context_mode: str, model: str, effort: str, status: int | None, headers: dict[str, Any], request_text: str, response_text: str, parsed: dict[str, Any] | None, first_ms: int | None, events: int, error: str | None) -> dict[str, Any]:
        return {
            "session_id": session_id, "job_id": job_id, "started_at": started_at,
            "first_event_at": first_event_at, "completed_at": utc_now(), "request_id": headers.get("x-request-id") or headers.get("X-Request-Id"),
            "sanitized_url": url, "request_format": request_format, "context_mode": context_mode,
            "model": model, "effort": effort, "http_status": status,
            "response_headers_without_auth": headers, "elapsed_ms": None, "time_to_first_event_ms": first_ms,
            "stream_event_count": events, "request_body_utf8_exact": request_text,
            "response_stream_utf8_exact": response_text, "parsed_response_json_if_available": parsed,
            "transport_error_if_any": error, "auth_header_omitted": True,
        }


__all__ = ["StreamingTransport", "TransportError", "TransportResult", "build_payload", "output_text", "parse_sse"]
