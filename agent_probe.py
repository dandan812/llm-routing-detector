"""Deterministic Agent trajectory scoring and profile-isolated routing analysis.

This module deliberately does not infer model identity from self-reported model
fields or from an LLM judge.  It scores externally captured trajectories against
an enrolled, same-profile baseline.  Product tier, client, protocol, context,
and baseline version are part of the comparison key so Team/K12/Plus/Pro data
cannot silently contaminate one another.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Optional

from .utils import canonical_json, utc_now


SCHEMA_VERSION = 1
PROFILE_TIERS = ("team", "k12", "plus", "pro")
PROFILE_ALIASES = {
    "team": "team",
    "taem": "team",  # common typo in notes/configs
    "k12": "k12",
    "k-12": "k12",
    "k_12": "k12",
    "plus": "plus",
    "pro": "pro",
}
TRAJECTORY_STATUSES = ("match", "deviant", "insufficient")


def normalize_product_tier(value: Any) -> str:
    normalized = str(value or "").strip().casefold().replace(" ", "")
    tier = PROFILE_ALIASES.get(normalized)
    if tier is None:
        raise ValueError("product_tier must be one of team, k12, plus, or pro")
    return tier


@dataclass(frozen=True)
class AgentProfile:
    product_tier: str
    client: str = "unknown"
    protocol: str = "unknown"
    context_mode: str = "no_history"
    baseline_version: str = "unversioned"

    def __post_init__(self) -> None:
        object.__setattr__(self, "product_tier", normalize_product_tier(self.product_tier))
        for field_name in ("client", "protocol", "context_mode", "baseline_version"):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, value)

    @property
    def key(self) -> str:
        return "|".join((self.product_tier, self.client, self.protocol, self.context_mode, self.baseline_version))

    def to_dict(self) -> dict[str, str]:
        return {
            "product_tier": self.product_tier,
            "client": self.client,
            "protocol": self.protocol,
            "context_mode": self.context_mode,
            "baseline_version": self.baseline_version,
            "profile_key": self.key,
        }

    @classmethod
    def from_value(cls, value: Any) -> "AgentProfile":
        if isinstance(value, AgentProfile):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("profile must be an object")
        return cls(
            product_tier=value.get("product_tier", value.get("tier")),
            client=str(value.get("client", "unknown")),
            protocol=str(value.get("protocol", "unknown")),
            context_mode=str(value.get("context_mode", "no_history")),
            baseline_version=str(value.get("baseline_version", "unversioned")),
        )


def _tool_calls(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for event in trace.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        kind = str(event.get("type", event.get("event_type", ""))).casefold()
        if kind not in {"tool_call", "tool_use", "function_call"}:
            continue
        function = event.get("function")
        function_name = function.get("name") if isinstance(function, Mapping) else None
        name = event.get("name") or event.get("tool_name") or function_name
        if name:
            calls.append({"name": str(name), "arguments": event.get("arguments", event.get("input"))})
    return calls


def _multiset_f1(expected: list[str], actual: list[str]) -> float:
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    expected_counts, actual_counts = Counter(expected), Counter(actual)
    hits = sum((expected_counts & actual_counts).values())
    precision = hits / len(actual)
    recall = hits / len(expected)
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _lcs_length(left: list[str], right: list[str]) -> int:
    row = [0] * (len(right) + 1)
    for value in left:
        previous = 0
        for index, other in enumerate(right, 1):
            saved = row[index]
            if value == other:
                row[index] = previous + 1
            else:
                row[index] = max(row[index], row[index - 1])
            previous = saved
    return row[-1]


def _expected_state_match(task: Mapping[str, Any], trace: Mapping[str, Any]) -> Optional[bool]:
    expected_hash = task.get("expected_final_state_hash")
    if expected_hash:
        observed_hash = trace.get("final_state_hash")
        if not observed_hash and isinstance(trace.get("final_state"), Mapping):
            observed_hash = hashlib.sha256(canonical_json(trace["final_state"]).encode("utf-8")).hexdigest()
        return bool(observed_hash and str(observed_hash) == str(expected_hash))
    expected_state = task.get("expected_final_state")
    if isinstance(expected_state, Mapping):
        observed = trace.get("final_state")
        if not isinstance(observed, Mapping):
            return False
        return all(observed.get(key) == value for key, value in expected_state.items())
    return None


def score_trajectory(trace: Mapping[str, Any], task: Mapping[str, Any], profile: Any = None) -> dict[str, Any]:
    """Score one captured trajectory using exact, non-LLM checks."""
    if not isinstance(trace, Mapping) or not isinstance(task, Mapping):
        raise ValueError("trace and task must be objects")
    profile_obj = AgentProfile.from_value(profile or trace.get("profile"))
    if profile is not None and trace.get("profile") is not None:
        trace_profile = AgentProfile.from_value(trace.get("profile"))
        if trace_profile.key != profile_obj.key:
            raise ValueError("trace profile does not match supplied profile")
    task_id = str(task.get("task_id") or trace.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("task_id is required")
    expected_tools = [str(value) for value in (task.get("expected_tool_sequence") or task.get("required_tools") or [])]
    if not expected_tools:
        raise ValueError("task requires expected_tool_sequence or required_tools")
    actual_calls = _tool_calls(trace)
    actual_tools = [str(row["name"]) for row in actual_calls]
    allowed_tools = {str(value) for value in (task.get("allowed_tools") or expected_tools)}
    unexpected_tools = [name for name in actual_tools if name not in allowed_tools]
    tool_f1 = _multiset_f1(expected_tools, actual_tools)
    order_score = _lcs_length(expected_tools, actual_tools) / max(len(expected_tools), 1)
    state_match = _expected_state_match(task, trace)
    if state_match is None:
        raise ValueError("task requires expected_final_state_hash or expected_final_state")
    nonce_expected = task.get("expected_nonce_result")
    nonce_observed = trace.get("nonce_result")
    if nonce_expected is not None:
        nonce_match = nonce_observed == nonce_expected
    else:
        nonce_match = None
    components = [float(state_match), tool_f1, order_score]
    if nonce_match is not None:
        components.append(float(nonce_match))
    score = sum(components) / len(components)
    required_ok = all(expected_tools.count(name) <= actual_tools.count(name) for name in set(expected_tools))
    completed = bool(state_match and required_ok and not unexpected_tools)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "profile": profile_obj.to_dict(),
        "profile_key": profile_obj.key,
        "completed": completed,
        "status": "match" if completed else "deviant",
        "score": round(score, 6),
        "tool_f1": round(tool_f1, 6),
        "tool_order_score": round(order_score, 6),
        "state_match": bool(state_match),
        "nonce_match": nonce_match,
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "unexpected_tools": unexpected_tools,
        "tool_call_count": len(actual_tools),
        "trace_id": str(trace.get("trace_id") or ""),
        "scored_at": utc_now(),
    }


def build_agent_baseline(
    traces: Iterable[Mapping[str, Any]],
    tasks: Mapping[str, Mapping[str, Any]],
    profile: Any,
    *,
    reference_model: str,
    baseline_version: Optional[str] = None,
) -> dict[str, Any]:
    profile_obj = AgentProfile.from_value(profile)
    scored: list[dict[str, Any]] = []
    for trace in traces:
        task_id = str(trace.get("task_id") or "")
        if task_id not in tasks:
            raise ValueError(f"task is missing from baseline catalog: {task_id}")
        scored.append(score_trajectory(trace, tasks[task_id], profile_obj))
    if not scored:
        raise ValueError("at least one trajectory is required")
    def mean(key: str) -> float:
        return sum(float(row[key]) for row in scored) / len(scored)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": f"agent-{profile_obj.key}-{baseline_version or profile_obj.baseline_version}",
        "baseline_version": baseline_version or profile_obj.baseline_version,
        "reference_model": str(reference_model),
        "profile": profile_obj.to_dict(),
        "sample_count": len(scored),
        "task_ids": sorted({row["task_id"] for row in scored}),
        "success_rate": round(sum(bool(row["completed"]) for row in scored) / len(scored), 6),
        "mean_score": round(mean("score"), 6),
        "mean_tool_f1": round(mean("tool_f1"), 6),
        "mean_tool_order_score": round(mean("tool_order_score"), 6),
        "tolerance": {"score": 0.20, "tool_f1": 0.25, "order": 0.30},
    }
    artifact["content_sha256"] = hashlib.sha256(canonical_json(artifact).encode("utf-8")).hexdigest()
    return artifact


def compare_trajectory_batch(
    scores: Iterable[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    *,
    minimum_samples: int = 5,
) -> dict[str, Any]:
    rows = list(scores)
    if not rows:
        return {"status": "insufficient", "reason": "no_trajectory_scores", "sample_count": 0}
    expected_profile = AgentProfile.from_value(baseline.get("profile"))
    if any(str(row.get("profile_key")) != expected_profile.key for row in rows):
        raise ValueError("trajectory profile does not match baseline profile")
    if len(rows) < minimum_samples:
        return {"status": "insufficient", "reason": "minimum_trajectory_samples_not_reached", "sample_count": len(rows), "required": minimum_samples}
    def mean(key: str) -> float:
        return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
    deltas = {
        "score": mean("score") - float(baseline.get("mean_score", 0.0)),
        "tool_f1": mean("tool_f1") - float(baseline.get("mean_tool_f1", 0.0)),
        "order": mean("tool_order_score") - float(baseline.get("mean_tool_order_score", 0.0)),
    }
    tolerance = baseline.get("tolerance") or {}
    deviants = [name for name, delta in deltas.items() if delta < -float(tolerance.get(name, 0.2))]
    return {
        "status": "deviant" if deviants else "match",
        "profile": expected_profile.to_dict(),
        "profile_key": expected_profile.key,
        "sample_count": len(rows),
        "success_rate": round(sum(bool(row.get("completed")) for row in rows) / len(rows), 6),
        "mean_score": round(mean("score"), 6),
        "mean_tool_f1": round(mean("tool_f1"), 6),
        "mean_tool_order_score": round(mean("tool_order_score"), 6),
        "deltas": {key: round(value, 6) for key, value in deltas.items()},
        "deviant_metrics": deviants,
        "baseline_id": baseline.get("baseline_id"),
    }


def identify_trajectory_model(
    traces: Iterable[Mapping[str, Any]],
    tasks: Mapping[str, Mapping[str, Any]],
    baselines: Mapping[str, Mapping[str, Any]],
    *,
    minimum_samples: int = 5,
) -> dict[str, Any]:
    """Compare one profile's traces against several enrolled model baselines.

    A unique match is still only a behavioral attribution. Multiple matches are
    reported as ambiguous instead of being forced into one model.
    """
    trace_rows = list(traces)
    if not baselines:
        return {"status": "insufficient", "reason": "no_baselines", "candidates": {}}
    candidates: dict[str, Any] = {}
    for model, baseline in baselines.items():
        scores: list[dict[str, Any]] = []
        for trace in trace_rows:
            task_id = str(trace.get("task_id") or "")
            if task_id not in tasks:
                raise ValueError(f"task is missing from catalog: {task_id}")
            scores.append(score_trajectory(trace, tasks[task_id], baseline.get("profile")))
        candidates[str(model)] = compare_trajectory_batch(scores, baseline, minimum_samples=minimum_samples)
    matches = [model for model, result in candidates.items() if result.get("status") == "match"]
    if len(matches) == 1:
        status, winner = "match", matches[0]
    elif len(matches) > 1:
        status, winner = "ambiguous", None
    elif any(result.get("status") == "insufficient" for result in candidates.values()):
        status, winner = "insufficient", None
    else:
        status, winner = "unknown", None
    profile_keys = {result.get("profile_key") for result in candidates.values() if result.get("profile_key")}
    return {
        "status": status,
        "inferred_model": winner,
        "profile_key": next(iter(profile_keys)) if len(profile_keys) == 1 else None,
        "sample_count": len(trace_rows),
        "candidates": candidates,
        "note": "行为归因，不是服务端权重或路由表证明",
    }


def _js_divergence(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    keys = set(left) | set(right)
    left_total, right_total = max(sum(left.values()), 1), max(sum(right.values()), 1)
    value = 0.0
    for key in keys:
        p, q = left.get(key, 0) / left_total, right.get(key, 0) / right_total
        m = (p + q) / 2
        if p and m:
            value += 0.5 * p * math.log(p / m, 2)
        if q and m:
            value += 0.5 * q * math.log(q / m, 2)
    return value


def analyze_routing_drift(
    observations: Iterable[Mapping[str, Any]],
    *,
    claimed_model: str,
    minimum_samples: int = 10,
    secondary_share_threshold: float = 0.10,
    js_threshold: float = 0.15,
) -> dict[str, Any]:
    """Analyze route drift only within identical Agent profiles.

    Each observation should contain ``profile`` or ``profile_key`` and an
    ``inferred_model`` from a trusted layer.  Self-reported model metadata is
    intentionally ignored by this function.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    windows: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in observations:
        profile = AgentProfile.from_value(row.get("profile")) if row.get("profile") else None
        key = str(row.get("profile_key") or (profile.key if profile else ""))
        if not key:
            raise ValueError("routing observation requires profile or profile_key")
        grouped[key].append(row)
        window = str(row.get("window_id") or str(row.get("timestamp", "unknown"))[:10])
        windows[key][window].append(row)
    profiles: dict[str, Any] = {}
    alerts: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        counts = Counter(str(row.get("inferred_model") or "unknown") for row in rows)
        total = len(rows)
        secondary = 1.0 - counts.get(claimed_model, 0) / max(total, 1)
        window_rows = []
        ordered_windows = list(windows[key].items())
        previous_counts: Optional[Counter[str]] = None
        max_js = 0.0
        for window_id, values in ordered_windows:
            window_counts = Counter(str(row.get("inferred_model") or "unknown") for row in values)
            js = _js_divergence(previous_counts or window_counts, window_counts) if previous_counts is not None else 0.0
            max_js = max(max_js, js)
            previous_counts = window_counts
            window_rows.append({"window_id": window_id, "sample_count": len(values), "counts": dict(window_counts), "js_from_previous": round(js, 6)})
        reasons: list[str] = []
        if total >= minimum_samples and secondary >= secondary_share_threshold:
            reasons.append("secondary_model_share_above_threshold")
        if len(window_rows) >= 2 and max_js >= js_threshold:
            reasons.append("window_distribution_shift")
        status = "drift" if reasons else "stable" if total >= minimum_samples else "insufficient"
        profiles[key] = {
            "profile_key": key,
            "sample_count": total,
            "counts": dict(counts),
            "claimed_model_share": round(counts.get(claimed_model, 0) / max(total, 1), 6),
            "secondary_share": round(secondary, 6),
            "max_window_js": round(max_js, 6),
            "windows": window_rows,
            "status": status,
            "reasons": reasons,
        }
        if status == "drift":
            alerts.append({"profile_key": key, "reasons": reasons, "secondary_share": round(secondary, 6), "max_window_js": round(max_js, 6)})
    return {
        "schema_version": SCHEMA_VERSION,
        "claimed_model": claimed_model,
        "profile_count": len(profiles),
        "profiles": profiles,
        "alerts": alerts,
        "alert": bool(alerts),
        "profile_isolation": "product_tier|client|protocol|context_mode|baseline_version",
    }


def build_three_layer_report(
    *,
    api_fingerprint: Mapping[str, Any],
    agent_trajectory: Mapping[str, Any],
    routing_drift: Mapping[str, Any],
) -> dict[str, Any]:
    api_status = str(api_fingerprint.get("status", ""))
    if not api_status:
        verdict = str(api_fingerprint.get("overall_verdict") or api_fingerprint.get("title_cn") or "")
        if verdict == "通过":
            api_status = "match"
        elif any(marker in verdict for marker in ("混用", "可能非GPT")):
            api_status = "mismatch"
        else:
            api_status = "insufficient"
    agent_status = str(agent_trajectory.get("status", "insufficient"))
    drift_alert = bool(routing_drift.get("alert"))
    statuses = {api_status, agent_status}
    if drift_alert or "mismatch" in statuses or "deviant" in statuses:
        overall = "review_required"
    elif statuses == {"match"} and not routing_drift.get("alerts"):
        overall = "consistent_with_baseline"
    else:
        overall = "insufficient"
    return {
        "schema_version": SCHEMA_VERSION,
        "overall_status": overall,
        "layers": {
            "api_fingerprint": dict(api_fingerprint),
            "agent_trajectory": dict(agent_trajectory),
            "routing_drift": dict(routing_drift),
        },
        "limitations": [
            "三层结果都是同协议基线下的行为证据，不是服务端权重或路由表的密码学证明",
            "Team/K12/Plus/Pro 必须分别建立基线，不能跨产品档位直接比较",
            "Agent 任务成功可能来自工具和编排层，不能单独等同于模型身份",
        ],
    }


def task_catalog() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "nonce_roundtrip",
            "name": "随机 nonce 工具往返",
            "required_tools": ["probe.read_nonce", "probe.write_result"],
            "description": "读取本地随机 nonce，按给定规则变换后写入结果；最终由本地状态哈希校验。",
            "verification": "expected_final_state_hash 或 expected_final_state",
        },
        {
            "task_id": "two_step_state",
            "name": "跨轮状态恢复",
            "required_tools": ["probe.read_state", "probe.update_state", "probe.commit"],
            "description": "读取隐藏状态，完成一次更新，再提交最终状态；检查调用顺序和提交内容。",
            "verification": "expected_final_state_hash 或 expected_final_state",
        },
    ]


__all__ = [
    "AgentProfile",
    "PROFILE_TIERS",
    "build_agent_baseline",
    "build_three_layer_report",
    "compare_trajectory_batch",
    "identify_trajectory_model",
    "analyze_routing_drift",
    "normalize_product_tier",
    "score_trajectory",
    "task_catalog",
]
