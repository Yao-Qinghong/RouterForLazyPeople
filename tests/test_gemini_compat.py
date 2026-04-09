"""Tests for router/gemini_compat.py — Gemini ↔ OpenAI translation."""

import json
import pytest
from router.gemini_compat import (
    gemini_to_openai,
    openai_to_gemini,
    gemini_model_to_backend,
)


class TestGeminiModelToBackend:
    def test_flash_fast(self):
        assert gemini_model_to_backend("gemini-2.0-flash-latest") == "fast"
        assert gemini_model_to_backend("gemini-1.5-flash-001") == "fast"

    def test_pro_mid(self):
        assert gemini_model_to_backend("gemini-1.5-pro-latest") == "mid"

    def test_pro2_deep(self):
        assert gemini_model_to_backend("gemini-2.0-pro-latest") == "deep"

    def test_unknown(self):
        assert gemini_model_to_backend("gpt-4") is None


class TestGeminiToOpenai:
    def test_basic(self):
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
            ],
        }
        result = gemini_to_openai(payload)
        assert result["model"] == "local"
        assert result["messages"][0]["content"] == "Hello"

    def test_system_instruction(self):
        payload = {
            "systemInstruction": {"parts": [{"text": "Be brief"}]},
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        }
        result = gemini_to_openai(payload)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "Be brief"

    def test_model_role_mapping(self):
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "Q"}]},
                {"role": "model", "parts": [{"text": "A"}]},
            ],
        }
        result = gemini_to_openai(payload)
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_generation_config(self):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "generationConfig": {
                "temperature": 0.5,
                "topP": 0.9,
                "maxOutputTokens": 256,
                "stopSequences": ["\n"],
            },
        }
        result = gemini_to_openai(payload)
        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.9
        assert result["max_tokens"] == 256
        assert result["stop"] == ["\n"]

    def test_function_call(self):
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "weather"}]},
                {"role": "model", "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"city": "SF"}}},
                ]},
                {"role": "user", "parts": [
                    {"functionResponse": {"name": "get_weather", "response": {"temp": 72}}},
                ]},
            ],
        }
        result = gemini_to_openai(payload)
        # Assistant with tool call
        assert result["messages"][1]["tool_calls"][0]["function"]["name"] == "get_weather"
        # Tool response
        assert result["messages"][2]["role"] == "tool"

    def test_tools(self):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "tools": [{
                "functionDeclarations": [{
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object"},
                }],
            }],
        }
        result = gemini_to_openai(payload)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "search"


class TestOpenaiToGemini:
    def test_basic(self):
        oai = {
            "choices": [{"message": {"content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        result = openai_to_gemini(oai, "gemini-1.5-pro")
        assert result["candidates"][0]["content"]["parts"][0]["text"] == "Hi!"
        assert result["candidates"][0]["finishReason"] == "STOP"
        assert result["usageMetadata"]["promptTokenCount"] == 5

    def test_tool_call_response(self):
        oai = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "search", "arguments": '{"q":"test"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
        result = openai_to_gemini(oai, "gemini-1.5-pro")
        parts = result["candidates"][0]["content"]["parts"]
        func_call = [p for p in parts if "functionCall" in p]
        assert len(func_call) == 1
        assert func_call[0]["functionCall"]["name"] == "search"

    def test_empty_response(self):
        result = openai_to_gemini({}, "gemini-1.5-pro")
        assert result["candidates"][0]["content"]["parts"][0]["text"] == ""
