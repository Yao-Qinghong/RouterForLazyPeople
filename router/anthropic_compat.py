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

Supported features:
  - Text messages, system prompts, images
  - Tool use / function calling (bidirectional)
  - Streaming (SSE event translation)

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
# Tool definition translation
# ─────────────────────────────────────────────────────────────

def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI function-calling format."""
    oai_tools = []
    for tool in tools:
        if tool.get("type") != "function" and tool.get("type") != "custom":
            # Standard Anthropic tool definition
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        elif tool.get("type") == "function":
            # Already in function format (some SDKs send this)
            oai_tools.append(tool)
    return oai_tools


def _openai_tool_calls_to_anthropic(tool_calls: list[dict]) -> list[dict]:
    """Convert OpenAI tool_calls in a response to Anthropic tool_use content blocks."""
    blocks = []
    for tc in tool_calls:
        func = tc.get("function", {})
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            args = {"raw": args_str}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": func.get("name", ""),
            "input": args,
        })
    return blocks


# ─────────────────────────────────────────────────────────────
# Request translation: Anthropic → OpenAI
# ─────────────────────────────────────────────────────────────

def anthropic_to_openai(payload: dict) -> dict:
    """
    Convert an Anthropic Messages API request body to OpenAI Chat Completions format.

    Anthropic fields handled:
      model, max_tokens, messages, system, stream, temperature, top_p,
      stop_sequences, tools, tool_choice, metadata (ignored)

    Content formats handled:
      - string content
      - list of content blocks: {"type": "text", "text": "..."}
      - image blocks (passed through as base64 OpenAI image_url)
      - tool_use blocks (→ OpenAI tool_calls)
      - tool_result blocks (→ OpenAI tool messages)
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
            # Separate tool_use, tool_result, and content blocks
            text_parts = []
            image_parts = []
            tool_use_blocks = []
            tool_result_blocks = []

            for block in content:
                btype = block.get("type", "text")
                if btype == "text":
                    text_parts.append({"type": "text", "text": block.get("text", "")})
                elif btype == "image":
                    src = block.get("source", {})
                    if src.get("type") == "base64":
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{src['media_type']};base64,{src['data']}"
                            },
                        })
                    elif src.get("type") == "url":
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {"url": src["url"]},
                        })
                elif btype == "tool_use":
                    tool_use_blocks.append(block)
                elif btype == "tool_result":
                    tool_result_blocks.append(block)

            # Handle tool_result blocks → OpenAI tool messages
            if tool_result_blocks:
                for tr in tool_result_blocks:
                    tr_content = tr.get("content", "")
                    if isinstance(tr_content, list):
                        tr_content = " ".join(
                            b.get("text", "") for b in tr_content
                            if b.get("type") == "text"
                        )
                    elif not isinstance(tr_content, str):
                        tr_content = str(tr_content)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": tr_content,
                    })
                continue

            # Handle assistant messages with tool_use → OpenAI assistant with tool_calls
            if tool_use_blocks and role == "assistant":
                oai_msg: dict = {"role": "assistant"}
                # Include text content if present
                if text_parts:
                    if len(text_parts) == 1:
                        oai_msg["content"] = text_parts[0]["text"]
                    else:
                        oai_msg["content"] = " ".join(p["text"] for p in text_parts)
                else:
                    oai_msg["content"] = None
                # Convert tool_use blocks to tool_calls
                oai_msg["tool_calls"] = []
                for tu in tool_use_blocks:
                    input_data = tu.get("input", {})
                    oai_msg["tool_calls"].append({
                        "id": tu.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": tu.get("name", ""),
                            "arguments": json.dumps(input_data) if isinstance(input_data, dict) else str(input_data),
                        },
                    })
                messages.append(oai_msg)
                continue

            # Regular content (text + images)
            all_parts = text_parts + image_parts
            if all(p["type"] == "text" for p in all_parts) and all_parts:
                messages.append({"role": role, "content": " ".join(p["text"] for p in all_parts)})
            elif all_parts:
                messages.append({"role": role, "content": all_parts})
            else:
                messages.append({"role": role, "content": ""})
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

    # Tool definitions
    if "tools" in payload:
        oai["tools"] = _anthropic_tools_to_openai(payload["tools"])

    # Tool choice
    if "tool_choice" in payload:
        tc = payload["tool_choice"]
        if isinstance(tc, dict):
            tc_type = tc.get("type", "auto")
            if tc_type == "auto":
                oai["tool_choice"] = "auto"
            elif tc_type == "any":
                oai["tool_choice"] = "required"
            elif tc_type == "tool":
                oai["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tc.get("name", "")},
                }
            elif tc_type == "none":
                oai["tool_choice"] = "none"
        elif isinstance(tc, str):
            oai["tool_choice"] = tc

    # Use a neutral model name so local backends don't complain
    oai["model"] = "local"

    return oai


# ─────────────────────────────────────────────────────────────
# Response translation: OpenAI → Anthropic
# ─────────────────────────────────────────────────────────────

def openai_to_anthropic(oai_response: dict, original_model: str) -> dict:
    """
    Convert an OpenAI Chat Completions response to Anthropic Messages format.
    Handles both text responses and tool_calls.
    """
    choices = oai_response.get("choices", [])
    content_blocks = []
    stop_reason = "end_turn"

    if choices:
        choice = choices[0]
        msg = choice.get("message", {})
        finish = choice.get("finish_reason", "stop")
        stop_reason = _finish_to_stop_reason(finish)

        # Text content
        text = msg.get("content", "") or ""
        if text:
            content_blocks.append({"type": "text", "text": text})

        # Tool calls → tool_use content blocks
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            content_blocks.extend(_openai_tool_calls_to_anthropic(tool_calls))
            if not content_blocks or all(b["type"] == "tool_use" for b in content_blocks):
                # Ensure there's at least an empty text block before tool_use
                pass
            stop_reason = "tool_use"

    # Fallback: ensure at least one content block
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = oai_response.get("usage", {})

    return {
        "id":      f"msg_{uuid.uuid4().hex[:24]}",
        "type":    "message",
        "role":    "assistant",
        "content": content_blocks,
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

    Supports both text content and tool_calls streaming.

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

        input_tokens  = 0
        output_tokens = 0
        stop_reason   = "end_turn"

        # Track content blocks: text block at index 0, tool_use blocks at index 1+
        text_block_started = False
        tool_call_index_map: dict[int, int] = {}  # oai tool index → anthropic block index
        next_block_index = 0
        # Accumulate tool call data for building complete blocks
        tool_calls_acc: dict[int, dict] = {}  # oai tool index → accumulated data

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

                choices = data.get("choices", [])
                if not choices:
                    # Capture usage if provided outside choices
                    if "usage" in data:
                        u = data["usage"]
                        input_tokens = u.get("prompt_tokens", input_tokens)
                        output_tokens = u.get("completion_tokens", output_tokens)
                    continue

                delta = choices[0].get("delta", {})
                finish = choices[0].get("finish_reason")

                # Text content delta
                text = delta.get("content", "")
                if text:
                    if not text_block_started:
                        yield _sse({"type": "content_block_start", "index": next_block_index,
                                    "content_block": {"type": "text", "text": ""}})
                        yield _sse({"type": "ping"})
                        text_block_started = True
                        next_block_index += 1

                    yield _sse({
                        "type":  "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                    })

                # Tool calls delta
                tool_calls = delta.get("tool_calls", [])
                for tc in tool_calls:
                    tc_index = tc.get("index", 0)
                    func = tc.get("function", {})

                    if tc_index not in tool_call_index_map:
                        # Close text block if it was open and this is the first tool
                        if text_block_started and not tool_call_index_map:
                            yield _sse({"type": "content_block_stop", "index": 0})

                        # Start a new tool_use content block
                        block_idx = next_block_index
                        tool_call_index_map[tc_index] = block_idx
                        next_block_index += 1

                        tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                        tool_name = func.get("name", "")
                        tool_calls_acc[tc_index] = {
                            "id": tool_id, "name": tool_name, "arguments": ""
                        }

                        yield _sse({
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": tool_name,
                                "input": {},
                            },
                        })

                    # Accumulate argument fragments
                    arg_chunk = func.get("arguments", "")
                    if arg_chunk:
                        tool_calls_acc[tc_index]["arguments"] += arg_chunk
                        block_idx = tool_call_index_map[tc_index]
                        yield _sse({
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": arg_chunk,
                            },
                        })

                if finish:
                    stop_reason = _finish_to_stop_reason(finish)

                # Capture usage if provided mid-stream
                if "usage" in data:
                    u = data["usage"]
                    input_tokens  = u.get("prompt_tokens", input_tokens)
                    output_tokens = u.get("completion_tokens", output_tokens)

        # Close any open content blocks
        if text_block_started and not tool_call_index_map:
            yield _sse({"type": "content_block_stop", "index": 0})
        for tc_idx, block_idx in tool_call_index_map.items():
            yield _sse({"type": "content_block_stop", "index": block_idx})

        # If no blocks were started at all, emit an empty text block
        if not text_block_started and not tool_call_index_map:
            yield _sse({"type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""}})
            yield _sse({"type": "content_block_stop", "index": 0})

        # Closing events
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
