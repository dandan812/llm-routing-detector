from gpt56_vnext.verdict import build_overall_verdict


def _juice(**overrides):
    value = {
        "juice_pass": True,
        "juice_mixed": False,
        "juice_all_unsuccessful": False,
        "data_insufficient": False,
        "sticky_events": [],
    }
    value.update(overrides)
    return value


def test_probability_evidence_shortfall_is_not_reported_as_pass():
    result = build_overall_verdict(
        juice_summary=_juice(),
        probability_summary={
            "formal_eligible": False,
            "probability_pass": False,
            "pure_model_alert": False,
            "mixture_alert": False,
            "evidence_insufficient": True,
            "evidence_insufficient_reasons": ["candidate_samples_incomplete:rand_country"],
        },
        probability_enabled=True,
    )
    assert result["overall_verdict"] == "Juice通过但概率探针证据不足"
    assert result["verdict_available"] is True


def test_juice_mixing_has_priority_over_probability_pass():
    result = build_overall_verdict(
        juice_summary=_juice(juice_pass=False, juice_mixed=True),
        probability_summary={"probability_pass": True},
        probability_enabled=True,
    )
    assert result["overall_verdict"] == "Juice混用"
