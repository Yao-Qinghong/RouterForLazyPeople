from router import benchmark


def test_benchmark_prompt_defaults_to_no_think():
    prompt = benchmark._benchmark_prompt("answer briefly", "no_think")

    assert "/no_think" in prompt
    assert "Answer directly" in prompt


def test_benchmark_prompt_can_request_thinking_or_raw_default():
    thinking = benchmark._benchmark_prompt("answer briefly", "think")
    raw = benchmark._benchmark_prompt("answer briefly", "default")

    assert "/think" in thinking
    assert "/no_think" not in raw
    assert "/think" not in raw


def test_format_results_shows_thinking_mode():
    output = benchmark.format_results([
        {
            "backend_key": "fast",
            "thinking_mode": "no_think",
            "pp_tok_s": 100.0,
            "tg_tok_s": 50.0,
            "ttft_ms": 123.0,
            "tier_assigned": "fast",
            "tier_measured": "fast",
            "tier_mismatch": False,
        }
    ])

    assert "Think" in output
    assert "no_think" in output
