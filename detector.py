from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import math
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit
import uuid

from .catalog import build_coverage_messages, choose_juice_prompt, load_catalog
from .generator import probe_document, reference_baseline
from .juice import (
    EFFORTS,
    JuiceSession,
    classify_coverage,
    classify_juice_answer,
    classify_output_integrity,
)
from .normalizers import builtin_probability_normalizer, normalize_answer
from .presets import (
    DEFAULT_PROBABILITY_NORMALIZER,
    PROBABILITY_PROBES,
    custom_changes,
    normalize_config,
    probability_runtime_spec,
    selected_profiles,
)
from .probability_model import ProbabilityModel, load_baseline, verify_baseline
from .retention import RawRetention, RetentionWriteError
from .store import SQLiteStateStore
from .transport import StreamingTransport, TransportError
from .utils import deterministic_job_id, sha256_text, utc_now
from .verdict import build_overall_verdict


DEFAULT_BASELINE = Path(__file__).with_name("baselines") / "trusted_likelihood_v2.json"
TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}
KNOWN_COVERAGE_VALUES = {8, 12, 16, 20, 24, 32, 40, 48, 64, 84, 96, 128, 512, 768, 960, 4085, 40805, 40855, 40085}
HARD_TITLES = {"Juice混用", "仅概率探针混用"}


def _new_coverage_value() -> int:
    while True:
        value = 10000 + secrets.randbelow(90000)
        text = str(value)
        if value not in KNOWN_COVERAGE_VALUES and not text.startswith(("8", "16", "40")):
            return value


def _profile_parts(profile: str) -> tuple[str, str]:
    request_format, context_mode = profile.split("+", 1)
    return request_format, context_mode


def _job(fields: dict[str, Any]) -> dict[str, Any]:
    stable = {key: value for key, value in fields.items() if key != "job_id"}
    return {**stable, "job_id": deterministic_job_id(stable)}


def _safe_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))


def _messages_for_probe(probe_id: str, catalog: dict[str, Any], effort: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if probe_id.startswith("juice_") and probe_id != "juice_coverage":
        prompt, template_id = choose_juice_prompt(catalog, effort)
        return [{"role": "user", "content": prompt}], {"template_id": template_id}
    if probe_id == "output_luna_48":
        return [{"role": "user", "content": "Reply with exactly the two ASCII digits 48. Do not add any other text."}], {"expected": "48"}
    if probe_id == "output_terra_32":
        return [{"role": "user", "content": "Reply with exactly the two ASCII digits 32. Do not add any other text."}], {"expected": "32"}
    if probe_id == "output_integrity":
        raise ValueError("continuous output_integrity requires an explicit expected value")
    if probe_id == "juice_coverage":
        value = _new_coverage_value()
        return build_coverage_messages(catalog, value), {"synthetic_value": value}
    behavior = catalog["behavior_probes"].get(probe_id)
    if behavior:
        return [{"role": "user", "content": behavior["prompt"]}], {"normalizer": builtin_probability_normalizer(probe_id)}
    raise ValueError(f"unknown probe: {probe_id}")


def _custom_probe_messages(custom: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    custom = probe_document(custom)
    identity = custom.get("probe_identity") or {}
    prompts = custom.get("exact_prompts_and_hashes") or {}
    prompt = str(prompts.get("user_prompt") or "")
    if not identity.get("probe_id") or not prompt:
        raise ValueError("custom probe is missing identity or exact prompt")
    developer = str(prompts.get("developer_prompt") or "")
    messages = ([{"role": "developer", "content": developer}] if developer else []) + [{"role": "user", "content": prompt}]
    return messages, {"normalizer": custom.get("normalizer") or {"id": "exact_trimmed_casefold", "parameters": {}}}


def _custom_profiles(custom: dict[str, Any], selected: list[str]) -> list[str]:
    custom = probe_document(custom)
    available = set((custom.get("trusted_counts_by_model") or {}).keys())
    if not available:
        artifact = custom.get("baseline_artifact") or {}
        available = {
            key.split("|", 1)[1]
            for key in artifact.get("cells", {})
            if "|" in key
        }
        probe_id = str((custom.get("probe_identity") or {}).get("probe_id") or "")
        available.update(
            (((artifact.get("raw_counts") or {}).get(probe_id) or {}).get("profiles") or {}).keys()
        )
    return [profile for profile in selected if profile in available]


def build_single_jobs(config: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    config = normalize_config(config)
    profiles = selected_profiles(config)
    jobs: list[dict[str, Any]] = []
    for probe_id, probe in config["probes"].items():
        if not probe.get("enabled"):
            continue
        applicable = probe.get("profiles") or profiles
        for profile in applicable:
            request_format, context_mode = _profile_parts(profile)
            for repeat in range(1, int(probe.get("requests", 0)) + 1):
                effort = str(probe.get("effort", "high" if probe_id.startswith(("juice", "output")) else "low"))
                messages, extra = _messages_for_probe(probe_id, catalog, effort)
                jobs.append(_job({
                    "mode": "single",
                    "cycle": 0,
                    "probe_id": probe_id,
                    "repeat": repeat,
                    "request_format": request_format,
                    "context_mode": context_mode,
                    "effort": effort,
                    "messages": messages,
                    **extra,
                }))
    for custom in config.get("custom_probes", []):
        if custom.get("enabled", True) is False:
            continue
        document = probe_document(custom)
        messages, extra = _custom_probe_messages(custom)
        for profile in _custom_profiles(custom, profiles):
            request_format, context_mode = _profile_parts(profile)
            for repeat in range(1, int(custom.get("runtime_requests", 10)) + 1):
                jobs.append(_job({
                    "mode": "single",
                    "cycle": 0,
                    "probe_id": document["probe_identity"]["probe_id"],
                    "custom_probe": True,
                    "repeat": repeat,
                    "request_format": request_format,
                    "context_mode": context_mode,
                    "effort": str((document.get("profile") or {}).get("effort", "low")),
                    "messages": messages,
                    **extra,
                }))
    return jobs


def _behavior_category(probe_id: str, answer: str, normalizer: dict[str, Any] | None = None) -> str:
    if normalizer is not None:
        return normalize_answer(answer, normalizer)
    return normalize_answer(answer, DEFAULT_PROBABILITY_NORMALIZER)


def _error_info(exc: Exception, attempt: int) -> dict[str, Any]:
    status = getattr(exc, "status", None)
    if isinstance(exc, RetentionWriteError):
        return {
            "stage": "retention",
            "category": "retention_write_failed",
            "retryable": False,
            "http_status": status,
            "attempt": attempt,
            "safe_message": "完整请求/响应留存写入失败；检测已停止以避免产生不完整留存",
        }
    if isinstance(exc, TransportError):
        return exc.error_info(attempt)
    return {
        "stage": "local_processing",
        "category": type(exc).__name__,
        "retryable": False,
        "http_status": status,
        "attempt": attempt,
        "safe_message": f"本地处理失败：{type(exc).__name__}",
    }


class DetectorSession:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        config: dict[str, Any],
        directory: str | Path,
        baseline_path: str | Path = DEFAULT_BASELINE,
        retention_enabled: bool = False,
        retention_directory: str | Path | None = None,
    ):
        if not base_url.strip() or not model.strip() or not api_key:
            raise ValueError("candidate API address, model, and key are required")
        if model not in {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}:
            raise ValueError("claimed model must be gpt-5.6-sol, gpt-5.6-terra, or gpt-5.6-luna")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.config = normalize_config(config)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.store = SQLiteStateStore(self.directory / "state.sqlite3")
        requested_session = str(config.get("session_id") or "")
        self.session_id = requested_session or uuid.uuid4().hex
        self.catalog = load_catalog()
        self.baseline = load_baseline(baseline_path)
        full_runtime_spec = probability_runtime_spec(self.config)
        builtin_ids = set(PROBABILITY_PROBES)
        formal_components: list[tuple[ProbabilityModel, dict[str, Any], str]] = []
        reference_custom: list[tuple[ProbabilityModel, dict[str, Any], str]] = []
        if full_runtime_spec is not None:
            builtin_cells = {
                key: value for key, value in full_runtime_spec["cells"].items()
                if key.split("|", 1)[0] in builtin_ids
            }
            if builtin_cells:
                builtin_spec = {
                    "name": full_runtime_spec["name"] + ":builtin",
                    "cells": builtin_cells,
                    "contracts": {key: full_runtime_spec["contracts"][key] for key in builtin_cells},
                }
                formal_components.append((ProbabilityModel(self.baseline), builtin_spec, "builtin"))
        for custom in self.config.get("custom_probes", []):
            if custom.get("enabled", True) is False:
                continue
            document = probe_document(custom)
            probe_id = str((document.get("probe_identity") or {}).get("probe_id") or "")
            artifact = reference_baseline(document)
            if not probe_id or not artifact:
                continue
            verify_baseline(artifact)
            cells = {
                key: value for key, value in ((full_runtime_spec or {}).get("cells") or {}).items()
                if key.startswith(probe_id + "|")
            }
            contracts = {
                key: value for key, value in ((full_runtime_spec or {}).get("contracts") or {}).items()
                if key in cells
            }
            custom_spec = {"name": f"custom:{probe_id}", "cells": cells, "contracts": contracts}
            reference_custom.append((ProbabilityModel(artifact), custom_spec, probe_id))
        self.runtime_spec = full_runtime_spec
        self.formal_probability_models = formal_components
        self.reference_probability_models = reference_custom
        self.stop_event = threading.Event()
        self._running_lock = threading.RLock()
        self._closed = False
        self.store.create_session(
            session_id=self.session_id,
            kind="detector",
            status="running",
            config=self.config,
            config_hash=self.config["config_hash"],
            official=bool(self.config["official"]),
            claimed_model=self.model,
            safe_endpoint=_safe_endpoint(self.base_url),
        )
        prior = self.store.session(self.session_id)
        if prior and requested_session:
            self.store.update_session_status(self.session_id, "running", clear_stop=True)
        self.store.reconcile_incomplete_attempts(self.session_id, int(self.config["retries"]) + 1)
        self.retention: RawRetention | None = None
        if retention_enabled:
            if retention_directory is None:
                raise ValueError("retention directory is required when retention is enabled")
            self.retention = RawRetention(retention_directory, self.session_id)
        self.transport = StreamingTransport(
            self.base_url,
            self.api_key,
            timeout=300.0 if "native_codex" in self.config["request_formats"] or "fixed_32k_history" in self.config["context_modes"] else 180.0,
            capture_exchange=self.retention is not None,
        )

    def close(self) -> None:
        if self._closed:
            return
        self.api_key = ""
        self.transport.api_key = ""
        self.store.close()
        self._closed = True

    def stop(self) -> None:
        self.stop_event.set()
        self.store.request_stop(self.session_id)

    def progress_snapshot(self) -> dict[str, Any]:
        return self.store.progress(self.session_id)

    def run_single(self, requester: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
        with self._running_lock:
            frozen = self.store.frozen_jobs(self.session_id, 0)
            if frozen:
                jobs = frozen
            else:
                jobs = build_single_jobs(self.config, self.catalog)
                jobs = self.store.freeze_jobs(self.session_id, 0, jobs)
            self.store.append_event(self.session_id, "run_start", cycle=0, payload={"planned_jobs": len(jobs)})
            self._run_jobs(jobs, requester)
            if not self.stop_event.is_set():
                self.store.append_event(self.session_id, "run_end", cycle=0, payload=self.store.progress(self.session_id))
                self.store.update_session_status(self.session_id, "complete")
            else:
                self.store.update_session_status(self.session_id, "stopped")
            report = self.build_report()
            return report

    def run_continuous(
        self,
        *,
        max_cycles: int | None = None,
        sleep_between: bool = True,
        requester: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        with self._running_lock:
            cycles_run = 0
            while not self.stop_event.is_set() and (max_cycles is None or cycles_run < max_cycles):
                events = self.store.events(self.session_id)
                ended = {int(row["cycle"]) for row in events if row.get("event") == "cycle_end" and row.get("cycle") is not None}
                cycles = self.store.cycles(self.session_id)
                unfinished = [cycle for cycle in cycles if cycle not in ended]
                if unfinished:
                    cycle = max(unfinished)
                    jobs = self.store.frozen_jobs(self.session_id, cycle)
                    self.store.append_event(self.session_id, "cycle_resume", cycle=cycle, payload={"planned_jobs": len(jobs)})
                else:
                    cycle = max(cycles, default=0) + 1
                    generated = self._continuous_cycle_jobs(cycle)
                    jobs = self.store.freeze_jobs(self.session_id, cycle, generated)
                    self.store.append_event(self.session_id, "cycle_start", cycle=cycle, payload={"planned_jobs": len(jobs)})
                cycles_run += 1
                self._run_jobs(jobs, requester)
                if self.stop_event.is_set():
                    break
                self.store.append_event(self.session_id, "cycle_end", cycle=cycle, payload={"completed_jobs": len(jobs)})
                self.build_report()
                if max_cycles is not None and cycles_run >= max_cycles:
                    break
                minimum = int(self.config["min_interval_seconds"])
                maximum = int(self.config["max_interval_seconds"])
                delay = minimum if maximum <= minimum else minimum + secrets.randbelow(maximum - minimum + 1)
                self.store.append_event(self.session_id, "cycle_sleep", cycle=cycle, payload={"seconds": delay})
                if sleep_between:
                    self.stop_event.wait(delay)
            self.store.update_session_status(self.session_id, "stopped" if self.stop_event.is_set() else "complete")
            return self.build_report()

    def _continuous_cycle_jobs(self, cycle: int) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        profiles = selected_profiles(self.config)
        output_index = sum(
            row.get("probe_id") in {"output_luna_48", "output_terra_32"}
            for row in self.store.latest_results(self.session_id)
        )
        for slot in range(1, int(self.config["slots_per_cycle"]) + 1):
            for probe_id, probe in self.config["probes"].items():
                probability = int(probe.get("probability_percent", 0))
                if not probe.get("enabled") or secrets.randbelow(10000) >= probability * 100:
                    continue
                applicable = probe.get("profiles") or profiles
                selected_probe = probe_id
                expected = None
                if probe_id == "output_integrity":
                    expected = ("48", "32")[output_index % 2]
                    output_index += 1
                    selected_probe = "output_luna_48" if expected == "48" else "output_terra_32"
                for profile in applicable:
                    request_format, context_mode = _profile_parts(profile)
                    effort = str(probe.get("effort", "high" if probe_id.startswith(("juice", "output")) else "low"))
                    messages, extra = _messages_for_probe(selected_probe, self.catalog, effort)
                    jobs.append(_job({
                        "mode": "continuous",
                        "cycle": cycle,
                        "slot": slot,
                        "probe_id": selected_probe,
                        "request_format": request_format,
                        "context_mode": context_mode,
                        "effort": effort,
                        "messages": messages,
                        "expected": expected or extra.get("expected"),
                        **{key: value for key, value in extra.items() if key != "expected"},
                    }))
            for custom in self.config.get("custom_probes", []):
                if custom.get("enabled", True) is False:
                    continue
                document = probe_document(custom)
                probability = int(float(custom.get("probability_percent", 100)))
                if secrets.randbelow(10000) >= max(0, min(100, probability)) * 100:
                    continue
                messages, extra = _custom_probe_messages(custom)
                for profile in _custom_profiles(custom, profiles):
                    request_format, context_mode = _profile_parts(profile)
                    jobs.append(_job({
                        "mode": "continuous",
                        "cycle": cycle,
                        "slot": slot,
                        "probe_id": document["probe_identity"]["probe_id"],
                        "custom_probe": True,
                        "request_format": request_format,
                        "context_mode": context_mode,
                        "effort": str((document.get("profile") or {}).get("effort", "low")),
                        "messages": messages,
                        **extra,
                    }))
        return jobs

    def _request(self, job: dict[str, Any]) -> dict[str, Any]:
        result = self.transport.post(
            model=self.model,
            messages=job["messages"],
            effort=job["effort"],
            request_format=job["request_format"],
            context_mode=job["context_mode"],
            cache_key=f"{self.session_id}:{job['job_id']}",
            job_id=job["job_id"],
            session_id=self.session_id,
        )
        return {"answer": result.answer, "response": result.response, "meta": result.meta, "exchange": result.exchange}

    def _run_jobs(self, jobs: list[dict[str, Any]], requester: Callable[[dict[str, Any]], dict[str, Any]] | None) -> None:
        frozen_ids = {job["job_id"] for job in jobs}
        max_attempts = int(self.config["retries"]) + 1
        pending = [
            job for job in self.store.pending_jobs(self.session_id, max_attempts=max_attempts)
            if job["job_id"] in frozen_ids
        ]
        if not pending:
            return
        workers = max(1, min(int(self.config["workers"]), len(pending)))
        iterator = iter(pending)
        futures: dict[Any, dict[str, Any]] = {}
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gpt56-detector")

        def submit_next() -> bool:
            if self.stop_event.is_set():
                return False
            try:
                job = next(iterator)
            except StopIteration:
                return False
            futures[executor.submit(self._execute_job, job, requester)] = job
            return True

        try:
            for _ in range(workers):
                submit_next()
            while futures:
                done, _pending = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    future.result()
                    submit_next()
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            if self.stop_event.is_set():
                for job in self.store.pending_jobs(self.session_id, max_attempts=max_attempts):
                    if job["job_id"] not in frozen_ids:
                        continue
                    attempts_sent = self.store.next_attempt_number(self.session_id, job["job_id"]) - 1
                    if attempts_sent == 0:
                        self.store.record_cancelled(self.session_id, job["job_id"], {
                            **self._safe_job(job),
                            "time": utc_now(),
                            "status": "cancelled",
                            "attempts_sent": 0,
                            "stop_requested_at": self.store.session(self.session_id).get("stop_requested_at"),
                        })
                    else:
                        self.store.record_terminal_result(self.session_id, job["job_id"], "error", {
                            **self._safe_job(job),
                            "time": utc_now(),
                            "status": "error",
                            "attempts_sent": attempts_sent,
                            "error": {
                                "stage": "runtime",
                                "category": "stopped_before_retry",
                                "retryable": False,
                                "http_status": None,
                                "attempt": attempts_sent,
                                "safe_message": "用户停止了检测，未继续发送下一次重试",
                            },
                        })

    def _execute_job(self, job: dict[str, Any], requester: Callable[[dict[str, Any]], dict[str, Any]] | None) -> dict[str, Any]:
        if self.stop_event.is_set():
            row = {
                **self._safe_job(job),
                "time": utc_now(),
                "status": "cancelled",
                "attempts_sent": 0,
                "stop_requested_at": self.store.session(self.session_id).get("stop_requested_at"),
            }
            self.store.record_cancelled(self.session_id, job["job_id"], row)
            return row
        max_attempts = int(self.config["retries"]) + 1
        first_attempt = self.store.next_attempt_number(self.session_id, job["job_id"])
        for attempt_no in range(first_attempt, max_attempts + 1):
            attempt_id = self.store.start_attempt(self.session_id, job["job_id"], attempt_no, max_attempts=max_attempts)
            try:
                result = requester(job) if requester is not None else self._request(job)
                exchange = result.get("exchange") or self._fake_exchange(job, result)
                if self.retention is not None:
                    self.retention.write(exchange)
                row = self._classify_result(job, result, attempt_no)
                self.store.finish_attempt(
                    attempt_id=attempt_id,
                    status="ok",
                    stage="response",
                    category="ok",
                    retryable=False,
                    http_status=(result.get("meta") or {}).get("http_status", 200),
                    safe_message="ok",
                    final_result=row,
                    final_job_status="ok",
                )
                return row
            except Exception as exc:
                if isinstance(exc, TransportError) and self.retention is not None and exc.exchange:
                    try:
                        self.retention.write(exc.exchange)
                    except RetentionWriteError as retention_exc:
                        exc = retention_exc
                info = _error_info(exc, attempt_no)
                will_retry = bool(info["retryable"] and attempt_no < max_attempts and not self.stop_event.is_set())
                final = None
                if not will_retry:
                    final = {
                        **self._safe_job(job),
                        "time": utc_now(),
                        "status": "error",
                        "attempts_sent": attempt_no,
                        "error": info,
                    }
                self.store.finish_attempt(
                    attempt_id=attempt_id,
                    status="error",
                    stage=info["stage"],
                    category=info["category"],
                    retryable=will_retry,
                    http_status=info["http_status"],
                    safe_message=info["safe_message"],
                    final_result=final,
                    final_job_status="error" if final else None,
                )
                if not will_retry:
                    if isinstance(exc, RetentionWriteError):
                        self.stop_event.set()
                        self.store.request_stop(self.session_id)
                    return final or {}
                time.sleep(min(2.0, 0.25 * (2 ** (attempt_no - 1))))
        raise AssertionError("attempt loop exhausted")

    @staticmethod
    def _safe_job(job: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in job.items() if key not in {"messages", "normalizer"}}

    def _fake_exchange(self, job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "job_id": job["job_id"],
            "started_at": utc_now(),
            "completed_at": utc_now(),
            "sanitized_url": _safe_endpoint(self.base_url) + "/responses",
            "request_format": job["request_format"],
            "context_mode": job["context_mode"],
            "model": self.model,
            "effort": job["effort"],
            "http_status": (result.get("meta") or {}).get("http_status", 200),
            "request_body_utf8_exact": "",
            "response_stream_utf8_exact": "",
            "parsed_response_json_if_available": result.get("response"),
            "auth_header_omitted": True,
        }

    def _classify_result(self, job: dict[str, Any], result: dict[str, Any], attempts: int) -> dict[str, Any]:
        answer = str(result.get("answer", ""))
        row = {
            **self._safe_job(job),
            "time": utc_now(),
            "status": "ok",
            "attempts_sent": attempts,
            "answer_sha256": sha256_text(answer),
            "answer_length": len(answer),
        }
        probe_id = str(job["probe_id"])
        if probe_id.startswith("juice_") and probe_id != "juice_coverage":
            row.update(classify_juice_answer(job["effort"], answer, self.model))
        elif probe_id.startswith("output_"):
            row.update(classify_output_integrity(str(job["expected"]), answer))
        elif probe_id == "juice_coverage":
            row.update(classify_coverage(int(job["synthetic_value"]), answer))
        else:
            row["category"] = _behavior_category(probe_id, answer, job.get("normalizer"))
            row["classification"] = "category"
        meta = result.get("meta") or {}
        for key in ("http_status", "elapsed_ms", "streaming", "stream_event_count", "time_to_first_event_ms"):
            row[key] = meta.get(key)
        return row

    def _rolling_rows(self, rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool], valid_needed: int) -> list[dict[str, Any]]:
        matching = [row for row in rows if predicate(row)]
        selected: list[dict[str, Any]] = []
        valid = 0
        for row in reversed(matching):
            selected.append(row)
            if row.get("status") == "ok":
                valid += 1
            if valid >= valid_needed:
                break
        return list(reversed(selected))

    def _probability_rows(self, rows: list[dict[str, Any]], runtime_spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        states: dict[str, Any] = {}
        for key, needed in runtime_spec["cells"].items():
            probe_id, profile = key.split("|", 1)
            cell_rows = [
                row for row in rows
                if row.get("status") == "ok"
                and row.get("classification") == "category"
                and row.get("probe_id") == probe_id
                and f"{row.get('request_format')}+{row.get('context_mode')}" == profile
            ]
            if self.config["mode"] == "continuous":
                cell_rows = cell_rows[-int(needed):]
            selected.extend(cell_rows)
            states[key] = {
                "required": int(needed),
                "actual": len(cell_rows),
                "full": len(cell_rows) >= int(needed),
                "counts": dict(Counter(str(row.get("category")) for row in cell_rows)),
            }
        return selected, states

    def _juice_summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        configured = {
            str(probe.get(
                "effort",
                probe_id[len("juice_"):]
                if probe_id.startswith("juice_")
                else probe_id,
            ))
            for probe_id, probe in self.config["probes"].items()
            if probe.get("enabled") and probe_id.startswith("juice_") and probe_id != "juice_coverage"
        }
        efforts = tuple(effort for effort in EFFORTS if effort in configured)
        minimum_valid: dict[str, int] = {}
        juice_rows: list[dict[str, Any]] = []
        for effort in efforts:
            probe = self.config["probes"].get(f"juice_{effort}", {})
            needed = int(probe.get("requests" if self.config["mode"] == "single" else "window", 1))
            minimum_valid[effort] = needed
            matching = [
                row for row in rows
                if str(row.get("probe_id", "")).startswith("juice_")
                and row.get("probe_id") != "juice_coverage"
                and row.get("effort") == effort
            ]
            if self.config["mode"] == "continuous":
                matching = self._rolling_rows(matching, lambda _row: True, needed)
            juice_rows.extend(matching)
        session = JuiceSession(
            claimed_model=self.model,
            selected_efforts=efforts,
            minimum_valid_by_effort=minimum_valid,
        )
        for row in juice_rows:
            if row.get("status") != "ok":
                session.add_network_error(str(row.get("effort")), job_id=row.get("job_id"), time=row.get("time"))
            else:
                session.add(
                    str(row.get("effort")),
                    str(row.get("normalized_value") or ""),
                    job_id=row.get("job_id"),
                    time=row.get("time"),
                    request_format=row.get("request_format"),
                    context_mode=row.get("context_mode"),
                )
        summary = session.summary()
        summary["window_mode"] = "rolling" if self.config["mode"] == "continuous" else "single_run"
        return summary

    def build_report(self) -> dict[str, Any]:
        rows = self.store.latest_results(self.session_id)
        completed_rows = [row for row in rows if row.get("status") in {"ok", "error", "cancelled"}]
        juice_summary = self._juice_summary(completed_rows)
        output_rows = [row for row in completed_rows if str(row.get("probe_id", "")).startswith("output_") and row.get("status") == "ok"]
        output_summary = {
            "requests": len(output_rows),
            "exact": sum(row.get("exact") is True for row in output_rows),
            "hard_anomaly": any(row.get("hard_anomaly") for row in output_rows),
            "failures": [row for row in output_rows if row.get("hard_anomaly")],
        }
        coverage_rows = [row for row in completed_rows if row.get("probe_id") == "juice_coverage" and row.get("status") == "ok"]
        coverage_summary = {
            "requests": len(coverage_rows),
            "hard_anomaly": any(row.get("hard_anomaly") for row in coverage_rows),
            "classifications": dict(Counter(str(row.get("classification")) for row in coverage_rows)),
            "failures": [row for row in coverage_rows if row.get("hard_anomaly")],
        }
        probability_enabled = bool(self.formal_probability_models or self.reference_probability_models)
        probability: dict[str, Any] = {}
        window_states: dict[str, Any] = {}
        formal_results: list[dict[str, Any]] = []
        for formal_model, formal_spec, component_id in self.formal_probability_models:
            probability_rows, component_windows = self._probability_rows(completed_rows, formal_spec)
            result = formal_model.score(probability_rows, runtime_spec=formal_spec, claimed_model=self.model)
            result["component_id"] = component_id
            result["window_states"] = component_windows
            formal_results.append(result)
            window_states.update(component_windows)
        if formal_results:
            if len(formal_results) != 1:
                raise AssertionError("vNext formal path must use exactly one pre-calibrated built-in baseline")
            probability = dict(formal_results[0])
            probability["combination_policy"] = "single_builtin_precalibrated_baseline"
            probability["window_states"] = window_states
        reference_probability_results: list[dict[str, Any]] = []
        for reference_model, reference_spec, probe_id in self.reference_probability_models:
            reference_rows, reference_windows = self._probability_rows(completed_rows, reference_spec)
            result = reference_model.score(reference_rows, runtime_spec=reference_spec, claimed_model=self.model)
            result.update({
                "probe_id": probe_id,
                "reference_only": True,
                "formal_eligible": False,
                "probability_pass": False,
                "pure_model_alert": False,
                "mixture_alert": False,
                "evidence_insufficient": True,
                "window_states": reference_windows,
            })
            reasons = list(result.get("evidence_insufficient_reasons") or [])
            if "custom_baseline_reference_only" not in reasons:
                reasons.append("custom_baseline_reference_only")
            result["evidence_insufficient_reasons"] = reasons
            reference_probability_results.append(result)
        if not probability and reference_probability_results:
            probability = dict(reference_probability_results[0])
        changes = custom_changes(self.config, self.config.get("base_preset")) if not self.config["official"] else []
        verdict = build_overall_verdict(
            juice_summary=juice_summary,
            probability_summary=probability,
            output_integrity_summary=output_summary,
            coverage_summary=coverage_summary,
            probability_enabled=probability_enabled,
            preset=self.config["preset"],
            official=bool(self.config["official"]),
            custom_changes=changes,
            session_id=self.session_id,
            claimed_model=self.model,
        )
        if verdict["overall_verdict"] in HARD_TITLES:
            dedupe = sha256_text(json.dumps({"title": verdict["overall_verdict"], "failed": verdict["failed_items"]}, sort_keys=True, default=str))
            self.store.add_sticky_alert(self.session_id, dedupe, verdict["overall_verdict"], {"failed_items": verdict["failed_items"]})
        sticky = self.store.sticky_alerts(self.session_id)
        if sticky and verdict["overall_verdict"] not in HARD_TITLES:
            verdict["current_window_verdict"] = verdict["overall_verdict"]
            verdict["overall_verdict"] = sticky[0]["title"]
            verdict["title_cn"] = sticky[0]["title"] + ("（自定义参考）" if not self.config["official"] else "")
        progress = self.store.progress(self.session_id)
        attempt_details = self.store.attempt_details(self.session_id)
        error_details = []
        for attempt in attempt_details:
            if attempt["status"] != "error":
                continue
            manifest = json.loads(attempt.pop("manifest_json"))
            error_details.append({
                "probe_id": manifest.get("probe_id"),
                "profile": f"{manifest.get('request_format')}+{manifest.get('context_mode')}",
                "stage": attempt.get("stage"),
                "category": attempt.get("category"),
                "retryable": bool(attempt.get("retryable")),
                "http_status": attempt.get("http_status"),
                "attempt": attempt.get("attempt_no"),
                "safe_message": attempt.get("safe_message"),
                "sanitized_target": _safe_endpoint(self.base_url),
            })
        profiles: dict[str, Any] = {}
        for profile in selected_profiles(self.config):
            values = [row for row in completed_rows if f"{row.get('request_format')}+{row.get('context_mode')}" == profile]
            profiles[profile] = {
                "logical_tasks": len(values),
                "successful": sum(row.get("status") == "ok" for row in values),
                "final_errors": sum(row.get("status") == "error" for row in values),
                "cancelled": sum(row.get("status") == "cancelled" for row in values),
                "probability_windows": {key: value for key, value in window_states.items() if key.endswith("|" + profile)},
            }
        retention_manifest = self.retention.finalize() if self.retention is not None else None
        build_hash = hashlib.sha256()
        for name in ("detector.py", "juice.py", "probability_model.py", "store.py", "verdict.py"):
            build_hash.update((Path(__file__).with_name(name)).read_bytes())
        session = self.store.session(self.session_id) or {}
        report = {
            "schema_version": 2,
            "scoring_version": self.baseline["scoring_version"],
            "session_id": self.session_id,
            "mode": self.config["mode"],
            "preset": self.config["preset"],
            "official": bool(self.config["official"]),
            "normalized_config": self.config,
            "config_hash": self.config["config_hash"],
            "baseline_id": self.baseline["baseline_id"],
            "baseline_sha256": self.baseline["content_sha256"],
            "build_hash": build_hash.hexdigest(),
            "started_at": session.get("created_at"),
            "updated_at": utc_now(),
            "candidate_configuration_without_key": {"base_url": _safe_endpoint(self.base_url), "model": self.model},
            "request_profiles": selected_profiles(self.config),
            **verdict,
            "juice_summary": juice_summary,
            "output_integrity_summary": output_summary,
            "coverage_summary": coverage_summary,
            "probability_summary": probability,
            "reference_probability_results": reference_probability_results,
            "mixture_summary": probability.get("mixture", {}),
            "window_states": window_states,
            "profile_summary": profiles,
            "sticky_alerts": sticky,
            "network_summary": {
                "logical_tasks": progress["planned"],
                "logical_completed": progress["logical_completed"],
                "successful": progress["successful"],
                "final_errors": progress["errors"],
                "cancelled": progress["cancelled"],
                "http_attempts": progress["http_attempts"],
                "retries": progress["retries"],
                "in_flight": progress["in_flight"],
            },
            "network_error_details": error_details,
            "data_completeness": {
                "juice": {
                    "state": juice_summary["state"],
                    "insufficient_efforts": juice_summary["insufficient_valid_efforts"],
                    "missing_success_efforts": juice_summary["missing_current_success_efforts"],
                },
                "probability": {
                    "enabled": probability_enabled,
                    "formal_eligible": probability.get("formal_eligible") if probability_enabled else None,
                    "reasons": probability.get("evidence_insufficient_reasons", []) if probability_enabled else [],
                },
                "retention_complete": retention_manifest.get("complete", True) if retention_manifest else True,
            },
            "retention_enabled": self.retention is not None,
            "retention_directory_display": str(self.retention.directory) if self.retention is not None else None,
            "retention_manifest": retention_manifest,
            "run_stopped": self.stop_event.is_set(),
            "stop_requested_at": session.get("stop_requested_at"),
            "auth_values_persisted": False,
            "limitations": verdict["limitations"] + [
                "加密状态层不能区分 Sol/Terra/Luna",
                "Juice 是可伪造的辅助指纹",
                "综合通过不能排除透明代理、探针识别或普通请求差异化路由",
            ],
        }
        self.store.save_report(self.session_id, report)
        return report


__all__ = ["DetectorSession", "build_single_jobs"]
