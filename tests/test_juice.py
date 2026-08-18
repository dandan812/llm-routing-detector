from gpt56_vnext.juice import JuiceSession, classify_juice_answer


def test_shared_juice_value_is_not_mixing():
    # 8 is a valid Sol value and is also shared by an older model family in
    # the compatibility table.  A shared value must not be called mixing.
    result = classify_juice_answer("low", "8", "gpt-5.6-sol")
    assert result["classification"] == "current_success"
    assert result["mixed_models"] == []
    assert result["shared_with_models"]


def test_juice_pass_requires_minimum_valid_samples():
    session = JuiceSession(
        claimed_model="gpt-5.6-sol",
        selected_efforts=("high", "low"),
        minimum_valid_by_effort={"high": 2, "low": 2},
    )
    session.add("high", "40")
    session.add("low", "8")

    summary = session.summary()
    assert summary["state"] == "data_insufficient"
    assert summary["juice_pass"] is False
    assert summary["insufficient_valid_efforts"] == ["high", "low"]


def test_juice_pass_after_each_effort_reaches_gate():
    session = JuiceSession(
        claimed_model="gpt-5.6-sol",
        selected_efforts=("high", "low"),
        minimum_valid_by_effort={"high": 2, "low": 2},
    )
    for _ in range(2):
        session.add("high", "40")
        session.add("low", "8")

    summary = session.summary()
    assert summary["state"] == "juice_pass"
    assert summary["juice_pass"] is True
