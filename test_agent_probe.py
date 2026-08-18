import hashlib

from gpt56_vnext.agent_probe import (
    AgentProfile,
    analyze_routing_drift,
    build_agent_baseline,
    build_three_layer_report,
    compare_trajectory_batch,
    identify_trajectory_model,
    score_trajectory,
)
from gpt56_vnext.utils import canonical_json


def _task():
    final_state = {"value": "ok", "nonce_result": "42"}
    return {
        "task_id": "nonce_roundtrip",
        "required_tools": ["probe.read_nonce", "probe.write_result"],
        "expected_final_state": final_state,
        "expected_nonce_result": "42",
    }


def _trace(profile=None, *, nonce="42", tools=None):
    return {
        "trace_id": "trace-1",
        "task_id": "nonce_roundtrip",
        "profile": profile or {"product_tier": "taem", "client": "opencode", "protocol": "responses", "baseline_version": "2026-08"},
        "events": [
            {"type": "tool_call", "name": name, "arguments": {}}
            for name in (tools or ["probe.read_nonce", "probe.write_result"])
        ],
        "nonce_result": nonce,
        "final_state": {"value": "ok", "nonce_result": nonce},
    }


def test_profile_normalizes_taem_and_keeps_tiers_isolated():
    profile = AgentProfile.from_value({"product_tier": "taem", "client": "opencode"})
    assert profile.product_tier == "team"
    assert profile.key.startswith("team|opencode|")
    assert AgentProfile.from_value({"product_tier": "plus", "client": "opencode"}).key != profile.key


def test_explicit_profile_cannot_override_trace_profile():
    import pytest

    with pytest.raises(ValueError, match="trace profile"):
        score_trajectory(
            _trace({"product_tier": "plus", "client": "opencode"}),
            _task(),
            {"product_tier": "team", "client": "opencode"},
        )


def test_trajectory_score_uses_exact_state_and_tool_checks():
    result = score_trajectory(_trace(), _task())
    assert result["status"] == "match"
    assert result["completed"] is True
    assert result["tool_f1"] == 1.0
    assert result["state_match"] is True

    bad = score_trajectory(_trace(nonce="wrong"), _task())
    assert bad["status"] == "deviant"
    assert bad["completed"] is False


def test_baseline_compare_requires_same_profile_and_minimum_samples():
    profile = {"product_tier": "team", "client": "opencode", "protocol": "responses", "baseline_version": "2026-08"}
    traces = [_trace(profile) for _ in range(5)]
    baseline = build_agent_baseline(
        traces,
        {"nonce_roundtrip": _task()},
        profile,
        reference_model="gpt-5.6-sol",
    )
    scores = [score_trajectory(trace, _task()) for trace in traces]
    result = compare_trajectory_batch(scores, baseline, minimum_samples=5)
    assert result["status"] == "match"


def test_model_identification_does_not_force_ambiguous_result():
    profile = {"product_tier": "team", "client": "opencode", "protocol": "responses", "baseline_version": "2026-08"}
    traces = [_trace(profile) for _ in range(5)]
    baseline = build_agent_baseline(traces, {"nonce_roundtrip": _task()}, profile, reference_model="gpt-5.6-sol")
    result = identify_trajectory_model(
        traces,
        {"nonce_roundtrip": _task()},
        {"gpt-5.6-sol": baseline, "gpt-5.6-terra": dict(baseline, reference_model="gpt-5.6-terra")},
        minimum_samples=5,
    )
    assert result["status"] == "ambiguous"
    assert result["inferred_model"] is None


def test_routing_drift_is_computed_per_profile():
    observations = []
    for index in range(10):
        observations.append({
            "profile": {"product_tier": "team", "client": "opencode", "protocol": "responses", "baseline_version": "2026-08"},
            "window_id": "w1",
            "inferred_model": "gpt-5.6-sol" if index < 8 else "gpt-5.6-terra",
        })
    for index in range(10):
        observations.append({
            "profile": {"product_tier": "plus", "client": "opencode", "protocol": "responses", "baseline_version": "2026-08"},
            "window_id": "w1",
            "inferred_model": "gpt-5.6-sol",
        })
    result = analyze_routing_drift(observations, claimed_model="gpt-5.6-sol", minimum_samples=10)
    assert result["profile_count"] == 2
    assert result["alert"] is True
    assert result["profiles"]["team|opencode|responses|no_history|2026-08"]["status"] == "drift"
    assert result["profiles"]["plus|opencode|responses|no_history|2026-08"]["status"] == "stable"


def test_three_layer_report_escalates_only_on_evidence_or_drift():
    report = build_three_layer_report(
        api_fingerprint={"status": "match"},
        agent_trajectory={"status": "match"},
        routing_drift={"alert": False, "alerts": []},
    )
    assert report["overall_status"] == "consistent_with_baseline"
    report = build_three_layer_report(
        api_fingerprint={"status": "match"},
        agent_trajectory={"status": "match"},
        routing_drift={"alert": True, "alerts": [{"profile_key": "team|x"}]},
    )
    assert report["overall_status"] == "review_required"

    report = build_three_layer_report(
        api_fingerprint={"overall_verdict": "Juice混用"},
        agent_trajectory={"status": "match"},
        routing_drift={"alert": False, "alerts": []},
    )
    assert report["overall_status"] == "review_required"
