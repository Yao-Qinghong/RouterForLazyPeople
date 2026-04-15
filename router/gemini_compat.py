"""
router/gemini_compat.py — Google Gemini API compatibility layer

Translates between Google Gemini API format and OpenAI Chat Completions
format so that any tool using the Gemini SDK can use local models.

Endpoint:  POST /gemini/v1beta/models/{model}:generateContent
           POST /gemini/v1beta/models/{model}:streamGenerateContent

Request translation:  Gemini → OpenAI (sent to local backend)
Response translation: OpenAI → Gemini (returned to client)

Gemini model → backend mapping:
  gemini-2.0-flash-*  → fast
  gemini-1.5-flash-*  → fast
  gemini-1.5-pro-*    → mid
  gemini-2.0-pro-*    → deep
  gemini-*            → mid  (default)
"""

import json
import logging
import uuid
import time
from typing import AsyncIterator, Optional

from router.sse import sse_events

logger = logging.getLogger("llm-router.gemini")


# ─────────────────────────────────────────────────────────────
# Model → backend mapping
# ─────────────────────────────────────────────────────────────

GEMINI_MODEL_TO_BACKEND: list[tuple[str, str]] = [
    ("gemini-2.0-flash",  "fast"),
    ("gemini-1.5-flash",  "fast"),
    ("gemini-2.0-pro",    "deep"),
    ("gemini-1.5-pro",    "mid"),
    ("gemini-exp",        "deep"),
    ("gemini",            "mid"),
]


def gemini_model_to_backend(model: str) -> Optional[str]:
    model_lower = model.lower()
    for prefix, backend in GEMINI_MODEL_TO_BACKEND:
        if model_lower.startswith(prefix):
            return backend
    return None


# ─────────────────────────────────────────────────────────────
# Request translation: Gemini → OpenAI
# ─────────────────────────────────────────────────────────────

def gemini_to_openai(payload: dict, is_stream: bool = False) -> dict:
    """
    Convert a Gemini generateContent request to OpenAI Chat Completions format.

    Gemini fields handled:
      contents, systemInstruction, generationConfig, tools
    """
    messages = []

    # System instruction → system message
    sys_instr = payload.get("systemInstruction", {})
    if sys_instr:
        parts = sys_instr.get("parts", [])
        text = " ".join(p.get("text", "") for p in parts if "text" in p)
        if text.strip():
            messages.append({"role": "system", "content": text})

    # Convert contents
    for content in payload.get("contents", []):
        role = content.get("role", "user")
        # Gemini uses "model" for assistant
        oai_role = "assistant" if role == "model" else role

        parts = content.get("parts", [])
        text_parts = []
        image_parts = []
        func_call_parts = []
        func_response_parts = []

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "inlineData" in part:
                inline = part["inlineData"]
                image_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{inline['mimeType']};base64,{inline['data']}"
                    },
                })
            elif "functionCall" in part:
                func_call_parts.append(part["functionCall"])
            elif "functionResponse" in part:
                func_response_parts.append(part["functionResponse"])

        # Function responses → tool messages
        if func_response_parts:
            for fr in func_response_parts:
                messages.append({
                    "role": "tool",
                    "tool_call_id": fr.get("name", ""),
                    "content": json.dumps(fr.get("response", {})),
                })
            continue

        # Function calls → assistant with tool_calls
        if func_call_parts and oai_role == "assistant":
            oai_msg: dict = {"role": "assistant", "content": None}
            oai_msg["tool_calls"] = []
            for fc in func_call_parts:
                oai_msg["tool_calls"].append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {})),
                    },
                })
            if text_parts:
                oai_msg["content"] = " ".join(text_parts)
            messages.append(oai_msg)
            continue

        # Regular text/image content
        if image_parts:
            content_parts = [{"type": "text", "text": t} for t in text_parts] + image_parts
            messages.append({"role": oai_role, "content": content_parts})
        else:
            messages.append({"role": oai_role, "content": " ".join(text_parts)})

    oai: dict = {
        "model": "local",
        "messages": messages,
        "stream": is_stream,
    }

    # Generation config
    gen_config = payload.get("generationConfig", {})
    if "temperature" in gen_config:
        oai["temperature"] = gen_config["temperature"]
    if "topP" in gen_config:
        oai["top_p"] = gen_config["topP"]
    if "maxOutputTokens" in gen_config:
        oai["max_tokens"] = gen_config["maxOutputTokens"]
    if "stopSequences" in gen_config:
        oai["stop"] = gen_config["stopSequences"]

    # Tools → OpenAI function tools
    if "tools" in payload:
        oai_tools = []
        for tool in payload["tools"]:
            for fd in tool.get("functionDeclarations", []):
                oai_tools.append({
                    "type": "function",
                    "function": {
                        "name": fd.get("name", ""),
                        "description": fd.get("description", ""),
                        "parameters": fd.get("parameters", {}),
                    },
                })
        if oai_tools:
            oai["tools"] = oai_tools

    return oai


# ─────────────────────────────────────────────────────────────
# Response translation: OpenAI → Gemini
# ─────────────────────────────────────────────────────────────

def openai_to_gemini(oai_response: dict, original_model: str) -> dict:
    """Convert an OpenAI Chat Completions response to Gemini format."""
    choices = oai_response.get("choices", [])
    parts = []
    finish_reason = "STOP"

    if choices:
        choice = choices[0]
        msg = choice.get("message", {})
        finish = choice.get("finish_reason", "stop")
        finish_reason = _finish_to_gemini_reason(finish)

        text = msg.get("content", "") or ""
        if text:
            parts.append({"text": text})

        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args = {}
            parts.append({
                "functionCall": {
                    "name": func.get("name", ""),
                    "args": args,
                },
            })

    if not parts:
        parts.append({"text": ""})

    usage = oai_response.get("usage", {})

    return {
        "candidates": [{
            "content": {
                "parts": parts,
                "role": "model",
            },
            "finishReason": finish_reason,
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
        },
        "modelVersion": original_model,
    }


def _finish_to_gemini_reason(finish: str) -> str:
    mapping = {
        "stop": "STOP",
        "length": "MAX_TOKENS",
        "tool_calls": "STOP",
        "content_filter": "SAFETY",
    }
    return mapping.get(finish, "STOP")


# ─────────────────────────────────────────────────────────────
# Streaming translation: OpenAI SSE → Gemini SSE
# ─────────────────────────────────────────────────────────────

def stream_openai_to_gemini(original_model: str):
    """
    Returns an async generator factory that wraps an OpenAI SSE stream
    and re-emits Gemini-format SSE events.
    """

    async def convert(openai_stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        async for data_str in sse_events(openai_stream):
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError as e:
                logger.debug("Malformed SSE: %s: %s", data_str[:100], e)
                continue

            choices = data.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            finish = choices[0].get("finish_reason")

            parts = []
            text = delta.get("content", "")
            if text:
                parts.append({"text": text})

            tool_calls = delta.get("tool_calls", [])
            for tc in tool_calls:
                func = tc.get("function", {})
                if func.get("name"):
                    args_str = func.get("arguments", "")
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        args = {}
                    parts.append({
                        "functionCall": {
                            "name": func["name"],
                            "args": args,
                        },
                    })

            if parts:
                gemini_chunk = {
                    "candidates": [{
                        "content": {
                            "parts": parts,
                            "role": "model",
                        },
                        "finishReason": _finish_to_gemini_reason(finish) if finish else None,
                        "index": 0,
                    }],
                }
                yield f"data: {json.dumps(gemini_chunk)}\n\n".encode()

    return convert
