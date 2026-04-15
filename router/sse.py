"""SSE line-buffering utility for chunked byte streams."""

import json
import logging
from typing import AsyncIterator

logger = logging.getLogger("llm-router.sse")


async def sse_events(byte_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """Yield complete SSE data payloads from a chunked byte stream.

    Handles chunk boundaries that split lines mid-JSON.
    Yields the string after 'data: ' for each complete SSE data line.
    """
    buf = ""
    async for chunk in byte_stream:
        buf += chunk.decode("utf-8", errors="replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    yield payload
    # Flush remaining buffer
    if buf.strip():
        line = buf.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                yield payload
