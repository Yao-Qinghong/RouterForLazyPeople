"""
router/anthropic_compat.py — Anthropic Messages API compatibility layer

Translates between Anthropic Messages API format and OpenAI Chat Completions
format so that any tool using the Anthropic SDK can point at this router
and use local models transparently.

Endpoint:  POST /anthropic/v1/messages
           POST /v1/messages  (alias)

Works with: Claude Code, @anthropic-ai/sdk, any Anthropic-SDK-based app.

Request translation:  Anthropic → OpenAI (sent to local backend)
Response translation: OpenAI   → Anthropic (returned to client)

Anthropic model → backend mapping (override via ?backend= or [route:key]):
  claude-3-haiku-*   → fast
  claude-3-sonnet-*  → mid
  claude-3-5-sonnet-* → mid
  claude-3-opus-*    → deep
  claude-*           → mid  (default for unknown Claude models)
  *                  → classifier (keyword routing)
"""

import json
import time
import uuid
from typing import AsyncIterator, Optional


# ─────────────────────────────────────────────────────────────
# Model → backend mapping
# ─────────────────────────────────────────────────────────────

# Maps Anthropic model name prefixes to backend keys.
# Checked in order — first match wins.
MODEL_TO_BACKEND: list[tuple[str, str]] = [
    ("claude-3-haiku",     "fast"),
    ("claude-haiku",       "fast"),
    ("claude-3-5-haiku",   "fast"),
    ("claude-3-5-sonnet",  "mid"),
    ("claude-3-sonnet",    "mid"),
    ("claude-sonnet",      "mid"),
    ("claude-3-opus",      "deep"),
    ("claude-opus",        "deep"),
    ("claude-3-5-opus",    "deep"),
    ("claude-4",           "deep"),
    ("claude",             "mid"),   # catch-all for any Claude model
]


def model_to_backend(model: str) -> Optional[str]:
    """Map an Anthropic model name to a local backend key. Returns None if no match."""
    model_lower = model.lower()
    for prefix, backend in MODEL_TO_BACKEND:
        if model_lower.startswith(prefix):
            return backend
    return None


# ─────────────────────────────────────────────────────────────
# Request translation: Anthropic → OpenAI
# ─────────────────────────────────────────────────────────────

def anthropic_to_openai(payload: dict) -> dict:
    """
    Convert an Anthropic Messages API request body to OpenAI Chat Completions format.

    Anthropic fields handled:
      model, max_tokens, messages, system, stream, temperature, top_p,
      stop_sequences, metadata (ignored)

    Content formats handled:
      - string content
      - list of content blocks: {"type": "text", "text": "..."}
      - image blocks (passed through as base64 OpenAI image_url)
    """
    messages = []

    # system prompt → system message
    if payload.get("system"):
        system = payload["system"]
        if isinstance(system, list):
            # Anthropic allows system as list of content blocks
            text = " ".join(
                b.get("text", "") for b in system if b.get("type") == "text"
            )
        else:
            text = str(system)
        if text.strip():
            messages.append({"role": "system", "content": text})

    # Convert each message
    for msg in payload.get("messages", []):
        role    = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Flatten content blocks
            parts = []
            for block in content:
                btype = block.get("type", "text")
                if btype == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})
                elif btype == "image":
                    src = block.get("source", {})
                    if src.get("type") == "base64":
                        parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{src['media_type']};base64,{src['data']}"
                            },
                        })
                    elif src.get("type") == "url":
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": src["url"]},
                        })
                # tool_use / tool_result blocks are ignored for now
            # If only text parts, flatten to string for broader compatibility
            if all(p["type"] == "text" for p in parts):
                messages.append({"role": role, "content": " ".join(p["text"] for p in parts)})
            else:
                messages.append({"role": role, "content": parts})
        else:
            messages.append({"role": role, "content": str(content)})

    oai: dict = {
        "messages": messages,
        "stream":   payload.get("stream", False),
    }

    # Map fields with different names
    if "max_tokens" in payload:
        oai["max_tokens"] = payload["max_tokens"]
    if "temperature" in payload:
        oai["temperature"] = payload["temperature"]
    if "top_p" in payload:
        oai["top_p"] = payload["top_p"]
    if "stop_sequences" in payload:
        oai["stop"] = payload["stop_sequences"]

    # Use a neutral model name so local backends don't complain
    oai["model"] = "local"

    return oai


# ─────────────────────────────────────────────────────────────
# Response translation: OpenAI → Anthropic
# ─────────────────────────────────────────────────────────────

def openai_to_anthropic(oai_response: dict, original_model: str) -> dict:
    """
    Convert an OpenAI Chat Completions response to Anthropic Messages format.
    """
    choices = oai_response.get("choices", [])
    content_text = ""
    stop_reason  = "end_turn"

    if choices:
        choice = choices[0]
        msg = choice.get("message", {})
        content_text = msg.get("content", "") or ""
        finish = choice.get("finish_reason", "stop")
        stop_reason = _finish_to_stop_reason(finish)

    usage = oai_response.get("usage", {})

    return {
        "id":      f"msg_{uuid.uuid4().hex[:24]}",
        "type":    "message",
        "role":    "assistant",
        "content": [{"type": "text", "text": content_text}],
        "model":   original_model,
        "stop_reason":    stop_reason,
        "stop_sequence":  None,
        "usage": {
            "input_tokens":  usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _finish_to_stop_reason(finish: str) -> str:
    mapping = {
        "stop":          "end_turn",
        "length":        "max_tokens",
        "tool_calls":    "tool_use",
        "content_filter": "stop_sequence",
    }
    return mapping.get(finish, "end_turn")


# ─────────────────────────────────────────────────────────────
# Streaming translation: OpenAI SSE → Anthropic SSE
# ─────────────────────────────────────────────────────────────

def stream_openai_to_anthropic(original_model: str):
    """
    Returns an async generator factory that wraps an OpenAI SSE stream
    and re-emits Anthropic SSE events.

    Usage:
        converter = stream_openai_to_anthropic(model)
        async for chunk in converter(openai_sse_stream):
            yield chunk
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    async def convert(openai_stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        # Opening events
        yield _sse({
            "type":  "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "content": [], "model": original_model,
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })
        yield _sse({"type": "content_block_start", "index": 0,
                    "content_block": {"type": "text", "text": ""}})
        yield _sse({"type": "ping"})

        input_tokens  = 0
        output_tokens = 0
        stop_reason   = "end_turn"

        async for raw_chunk in openai_stream:
            for line in raw_chunk.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except Exception:
                    continue

                # Extract delta text
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    text  = delta.get("content", "")
                    finish = choices[0].get("finish_reason")

                    if text:
                        yield _sse({
                            "type":  "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": text},
                        })

                    if finish:
                        stop_reason = _finish_to_stop_reason(finish)

                # Capture usage if provided mid-stream
                if "usage" in data:
                    u = data["usage"]
                    input_tokens  = u.get("prompt_tokens", input_tokens)
                    output_tokens = u.get("completion_tokens", output_tokens)

        # Closing events
        yield _sse({"type": "content_block_stop", "index": 0})
        yield _sse({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        })
        yield _sse({
            "type": "message_stop",
            "amazon-bedrock-invocationMetrics": None,
        })

    return convert


def _sse(data: dict) -> bytes:
    """Encode a dict as a Server-Sent Events frame."""
    return f"data: {json.dumps(data)}\n\n".encode()
