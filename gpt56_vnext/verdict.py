from __future__ import annotations

from typing import Any

from .juice import TARGET_MODELS


VERDICT_TITLES = (
    "可能非GPT",
    "Juice混用",
    "仅概率探针混用",
    "Juice通过但概率探针证据不足",
    "通过",
)

MODEL_LABELS = {
    "gpt-5.6-sol": "Sol",
    "gpt-5.6-terra": "Terra",
    "gpt-5.6-luna": "Luna",
    "gpt-5.5": "GPT-5.5",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4 mini",
}


def _probability_rows(probability: dict[str, Any]) -> list[dict[str, Any]]:
    values = probability.get("conditional_relative_probability") or {}
    if not values:
        return []
    return sorted(
        [
            {
                "model": model,
                "label_cn": MODEL_LABELS.get(model, model),
                "probability": float(values.get(model, 0.0)),
                "score": probability.get("pure_scores", {}).get(model),
            }
            for model in TARGET_MODELS
        ],
        key=lambda row: row["probability"],
        reverse=True,
    )


def build_overall_verdict(
    *,
    juice_summary: dict[str, Any],
    probability_summary: dict[str, Any] | None,
    output_integrity_summary: dict[str, Any] | None = None,
    coverage_summary: dict[str, Any] | None = None,
    probability_enabled: bool = True,
    preset: str = "low",
    official: bool | None = None,
    custom_changes: list[str] | None = None,
    session_id: str | None = None,
    claimed_model: str = "gpt-5.6-sol",
) -> dict[str, Any]:
    probability = probability_summary or {}
    output = output_integrity_summary or {}
    coverage = coverage_summary or {}
    deterministic_anomaly = bool(output.get("hard_anomaly") or coverage.get("hard_anomaly"))
    juice_mixed = bool(juice_summary.get("juice_mixed"))
    juice_pass = bool(juice_summary.get("juice_pass"))
    juice_all_unsuccessful = bool(juice_summary.get("juice_all_unsuccessful"))
    probability_alert = bool(probability.get("pure_model_alert") or probability.get("mixture_alert"))
    probability_pass = bool(probability.get("probability_pass"))

    if juice_all_unsuccessful:
        title = "可能非GPT"
    elif juice_mixed or deterministic_anomaly:
        title = "Juice混用"
    elif not juice_pass:
        title = None
    elif juice_pass and probability_enabled and probability_alert:
        title = "仅概率探针混用"
    elif juice_pass and (not probability_enabled or probability_pass):
        title = "通过"
    else:
        title = "Juice通过但概率探针证据不足"

    failed_items: list[dict[str, Any]] = []
    for event in juice_summary.get("sticky_events", []):
        failed_items.append({"layer": "juice", "reason": event.get("reason"), "evidence": event.get("evidence")})
    for row in output.get("failures", []):
        failed_items.append({"layer": "output_integrity", "reason": "non_exact_literal", "evidence": row})
    for row in coverage.get("failures", []):
        failed_items.append({"layer": "juice_coverage", "reason": row.get("classification"), "evidence": row})
    if probability_alert:
        failed_items.append({
            "layer": "probability",
            "reason": "mixture" if probability.get("mixture_alert") else "pure_replacement",
            "winner": probability.get("winner"),
            "margin": probability.get("score_margin"),
            "mixture": probability.get("mixture"),
        })
    if probability.get("evidence_insufficient"):
        failed_items.extend({"layer": "probability", "reason": reason} for reason in probability.get("evidence_insufficient_reasons", []))
    if juice_summary.get("data_insufficient"):
        failed_items.append({
            "layer": "juice",
            "reason": "data_incomplete",
            "missing_current_success_efforts": juice_summary.get("missing_current_success_efforts", []),
            "insufficient_valid_efforts": juice_summary.get("insufficient_valid_efforts", []),
        })

    common_causes: list[str] = []
    if title == "可能非GPT":
        common_causes.extend(["非 GPT 模型", "响应被统一改写", "参数未透传", "探针被拦截但返回了有效非指纹答案"])
    if juice_mixed:
        common_causes.append("同一端点返回了其他 GPT-5.6 或旧 GPT 型号的确定性 Juice 指纹")
    if output.get("hard_anomaly"):
        common_causes.append("输出端改写了要求精确返回的 32/48 控制值")
    if coverage.get("hard_anomaly"):
        common_causes.append("隐藏 system/developer 规则可能覆盖了显式 Juice 定义")
    if probability.get("pure_model_alert"):
        common_causes.append("概率行为整体更接近另一可信纯模型基线")
    if probability.get("mixture_alert"):
        common_causes.append("混合分布拟合显著优于任一可信纯模型")

    is_official = bool(official if official is not None else preset != "custom")
    custom = not is_official
    probability_rows = _probability_rows(probability)
    subtitle = None
    if title == "通过" and not probability_enabled:
        subtitle = "未启用概率探针，仅完成Juice与确定性检查"
    elif title == "Juice通过但概率探针证据不足":
        subtitle = "未达到正式概率门禁；具体原因见数据完整性与逐探针详情"
    elif title is None:
        subtitle = "Juice 数据不足，未形成冻结五状态中的正式结论；网络错误与未知输出不算通过"
    if custom:
        subtitle = ((subtitle + "；") if subtitle else "") + "自定义档位测试结果仅供参考"

    result = {
        "overall_verdict": title,
        "verdict_available": title is not None,
        "operational_status": "complete" if title is not None else "juice_data_insufficient",
        "title_cn": (title + ("（自定义参考）" if custom else "")) if title is not None else "未形成正式结论",
        "subtitle_cn": subtitle,
        "preset": preset,
        "custom_preset": custom,
        "official_grade": is_official,
        "trust_scope": "official_preset" if is_official else "advisory_only",
        "custom_changes": list(custom_changes or []),
        "session_id": session_id,
        "claimed_model": claimed_model,
        "failed_items": failed_items,
        "common_causes": sorted(set(common_causes)),
        "possible_models": probability_rows,
        "probability_details": probability_rows,
        "probability_hard_alert": probability_alert,
        "juice_state": juice_summary.get("state"),
        "juice_quality_warning": bool(juice_summary.get("quality_warning")),
        "quality_note": "Juice 身份判定通过，但有效命中质量偏低" if juice_summary.get("quality_warning") else None,
        "limitations": [
            "条件相对概率只比较已导入的 Sol/Terra/Luna 可信基线，不是 API 身份的密码学证明",
            "自定义档位测试结果仅供参考" if custom else "网络错误不计入 Juice 有效响应",
        ],
    }
    if result["overall_verdict"] is not None and result["overall_verdict"] not in VERDICT_TITLES:
        raise AssertionError("verdict title escaped the frozen five-state set")
    return result


__all__ = ["VERDICT_TITLES", "build_overall_verdict"]
