"""Tests for router/anthropic_compat.py — Anthropic ↔ OpenAI translation."""

import json
import pytest
from router.anthropic_compat import (
    anthropic_to_openai,
    openai_to_anthropic,
    model_to_backend,
    _finish_to_stop_reason,
    _anthropic_tools_to_openai,
    _openai_tool_calls_to_anthropic,
)


# ── model_to_backend ──────────────────────────────────────────

class TestModelToBackend:
    def test_haiku_fast(self):
        assert model_to_backend("claude-3-haiku-20240307") == "fast"
        assert model_to_backend("claude-3-5-haiku-latest") == "fast"

    def test_sonnet_mid(self):
        assert model_to_backend("claude-3-5-sonnet-20241022") == "mid"
        assert model_to_backend("claude-3-sonnet-20240229") == "mid"

    def test_opus_deep(self):
        assert model_to_backend("claude-3-opus-20240229") == "deep"
        assert model_to_backend("claude-3-5-opus-latest") == "deep"

    def test_claude4_deep(self):
        assert model_to_backend("claude-4-sonnet-latest") == "deep"

    def test_catchall_mid(self):
        assert model_to_backend("claude-unknown-model") == "mid"

    def test_unknown_model_none(self):
        assert model_to_backend("gpt-4") is None
        assert model_to_backend("gemini-pro") is None


# ── anthropic_to_openai ──────────────────────────────────────

class TestAnthropicToOpenai:
    def test_basic_text(self):
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_openai(payload)
        assert result["model"] == "local"
        assert result["max_tokens"] == 1024
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Hello"

    def test_system_prompt(self):
        payload = {
            "system": "You are helpful",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_openai(payload)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful"
        assert result["messages"][1]["content"] == "Hi"

    def test_system_as_list(self):
        payload = {
            "system": [{"type": "text", "text": "Be concise"}],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_openai(payload)
        assert result["messages"][0]["content"] == "Be concise"

    def test_content_blocks(self):
        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "text", "text": "Explain."},
                ],
            }],
        }
        result = anthropic_to_openai(payload)
        assert "What is this?" in result["messages"][0]["content"]
        assert "Explain." in result["messages"][0]["content"]

    def test_image_block(self):
        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBOR...",
                    }},
                ],
            }],
        }
        result = anthropic_to_openai(payload)
        content = result["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[1]["type"] == "image_url"

    def test_tool_use_translation(self):
        payload = {
            "messages": [
                {"role": "user", "content": "Get the weather"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "toolu_123", "name": "get_weather",
                     "input": {"city": "SF"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_123",
                     "content": "72°F and sunny"},
                ]},
            ],
        }
        result = anthropic_to_openai(payload)
        # User message
        assert result["messages"][0]["content"] == "Get the weather"
        # Assistant with tool_calls
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(result["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {"city": "SF"}
        # Tool result
        assert result["messages"][2]["role"] == "tool"
        assert result["messages"][2]["content"] == "72°F and sunny"

    def test_tool_definitions(self):
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{
                "name": "calculator",
                "description": "Do math",
                "input_schema": {"type": "object", "properties": {"expr": {"type": "string"}}},
            }],
        }
        result = anthropic_to_openai(payload)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "calculator"

    def test_tool_choice_auto(self):
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "auto"},
        }
        result = anthropic_to_openai(payload)
        assert result["tool_choice"] == "auto"

    def test_tool_choice_any(self):
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "any"},
        }
        result = anthropic_to_openai(payload)
        assert result["tool_choice"] == "required"

    def test_tool_choice_specific(self):
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "tool", "name": "calculator"},
        }
        result = anthropic_to_openai(payload)
        assert result["tool_choice"]["function"]["name"] == "calculator"

    def test_stop_sequences(self):
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "stop_sequences": ["\n\nHuman:"],
        }
        result = anthropic_to_openai(payload)
        assert result["stop"] == ["\n\nHuman:"]

    def test_temperature_and_top_p(self):
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
        }
        result = anthropic_to_openai(payload)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9


# ── openai_to_anthropic ──────────────────────────────────────

class TestOpenaiToAnthropic:
    def test_basic_response(self):
        oai = {
            "choices": [{
                "message": {"content": "Hello there!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = openai_to_anthropic(oai, "claude-3-5-sonnet-20241022")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello there!"
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_calls_response(self):
        oai = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        result = openai_to_anthropic(oai, "claude-3-5-sonnet-20241022")
        assert result["stop_reason"] == "tool_use"
        tool_block = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_block) == 1
        assert tool_block[0]["name"] == "get_weather"
        assert tool_block[0]["input"] == {"city": "SF"}

    def test_empty_response(self):
        result = openai_to_anthropic({}, "claude-3-5-sonnet-20241022")
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == ""


# ── _finish_to_stop_reason ────────────────────────────────────

class TestFinishToStopReason:
    def test_stop(self):
        assert _finish_to_stop_reason("stop") == "end_turn"

    def test_length(self):
        assert _finish_to_stop_reason("length") == "max_tokens"

    def test_tool_calls(self):
        assert _finish_to_stop_reason("tool_calls") == "tool_use"

    def test_unknown(self):
        assert _finish_to_stop_reason("unknown_reason") == "end_turn"


# ── Tool helpers ──────────────────────────────────────────────

class TestToolHelpers:
    def test_anthropic_tools_to_openai(self):
        tools = [{
            "name": "search",
            "description": "Search the web",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }]
        result = _anthropic_tools_to_openai(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"

    def test_openai_tool_calls_to_anthropic(self):
        tool_calls = [{
            "id": "call_123",
            "function": {"name": "search", "arguments": '{"q": "test"}'},
        }]
        result = _openai_tool_calls_to_anthropic(tool_calls)
        assert len(result) == 1
        assert result[0]["type"] == "tool_use"
        assert result[0]["name"] == "search"
        assert result[0]["input"] == {"q": "test"}

    def test_openai_tool_calls_bad_json(self):
        tool_calls = [{
            "id": "call_123",
            "function": {"name": "search", "arguments": "not json"},
        }]
        result = _openai_tool_calls_to_anthropic(tool_calls)
        assert result[0]["input"] == {"raw": "not json"}
