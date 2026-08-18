from gpt56_vnext.transport import build_payload


def test_payload_keeps_declared_model_and_request_contract():
    payload = build_payload(
        "gpt-5.6-sol",
        [{"role": "user", "content": "probe"}],
        "high",
        "no_history",
        "cache-key",
    )
    assert payload["model"] == "gpt-5.6-sol"
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["stream"] is True
    assert payload["store"] is False
    assert payload["input"] == [{"role": "user", "content": "probe"}]


def test_history_is_added_before_final_probe_message():
    payload = build_payload(
        "gpt-5.6-terra",
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "probe"},
        ],
        "low",
        "fixed_32k_history",
        "cache-key",
    )
    assert payload["model"] == "gpt-5.6-terra"
    assert payload["input"][0] == {"role": "system", "content": "system"}
    assert payload["input"][-1] == {"role": "user", "content": "probe"}
    assert len(payload["input"]) > 2
