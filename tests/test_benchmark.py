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


def test_format_results_pp_only_shows_pp_not_error():
    """When TG fails but PP succeeds, show PP data instead of ERROR."""
    output = benchmark.format_results([
        {
            "backend_key": "big-model",
            "thinking_mode": "no_think",
            "pp_tok_s": 220.4,
            "tg_tok_s": None,
            "ttft_ms": 1800.0,
            "tier_assigned": "mid",
            "tier_measured": None,
            "tier_mismatch": False,
            "tg_error": "Read timed out",
            "error": None,
        }
    ])

    assert "220" in output
    assert "ERROR" not in output
    assert "PP only" in output
    assert "TG failed" in output


def test_format_results_total_failure_shows_error():
    """When both PP and TG fail, show ERROR."""
    output = benchmark.format_results([
        {
            "backend_key": "broken",
            "thinking_mode": "no_think",
            "pp_tok_s": None,
            "tg_tok_s": None,
            "ttft_ms": None,
            "tier_assigned": "fast",
            "tier_measured": None,
            "tier_mismatch": False,
            "pp_error": "Connection refused",
            "tg_error": "Connection refused",
            "error": "PP: Connection refused; TG: Connection refused",
        }
    ])

    assert "ERROR" in output


def test_format_results_legacy_error_format():
    """Old benchmark results with only 'error' field still work."""
    output = benchmark.format_results([
        {
            "backend_key": "legacy",
            "thinking_mode": "no_think",
            "pp_tok_s": None,
            "tg_tok_s": None,
            "ttft_ms": None,
            "tier_assigned": "fast",
            "tier_measured": None,
            "tier_mismatch": False,
            "error": "Connection refused",
        }
    ])

    assert "ERROR" in output
    assert "Connection refused" in output
