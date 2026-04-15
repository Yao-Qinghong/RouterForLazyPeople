"""Tests for router/sse.py — SSE line buffering."""
import pytest
from router.sse import sse_events


async def _chunks(*parts):
    for p in parts:
        yield p.encode() if isinstance(p, str) else p


class TestSseEvents:
    @pytest.mark.asyncio
    async def test_simple_events(self):
        stream = _chunks('data: {"id": 1}\n\n', 'data: [DONE]\n\n')
        events = [e async for e in sse_events(stream)]
        assert events == ['{"id": 1}', '[DONE]']

    @pytest.mark.asyncio
    async def test_chunk_boundary_splits_json(self):
        # JSON split across two chunks
        stream = _chunks('data: {"ke', 'y": "value"}\n\ndata: [DONE]\n\n')
        events = [e async for e in sse_events(stream)]
        assert events == ['{"key": "value"}', '[DONE]']

    @pytest.mark.asyncio
    async def test_multiple_events_in_one_chunk(self):
        stream = _chunks('data: {"a": 1}\ndata: {"b": 2}\n\n')
        events = [e async for e in sse_events(stream)]
        assert '{"a": 1}' in events
        assert '{"b": 2}' in events

    @pytest.mark.asyncio
    async def test_non_data_lines_skipped(self):
        stream = _chunks('event: ping\ndata: {"ok": true}\n\n')
        events = [e async for e in sse_events(stream)]
        assert events == ['{"ok": true}']

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        stream = _chunks()
        events = [e async for e in sse_events(stream)]
        assert events == []

    @pytest.mark.asyncio
    async def test_trailing_data_without_newline(self):
        # Data at end of stream with no trailing newline — should still be flushed
        stream = _chunks('data: {"flush": true}')
        events = [e async for e in sse_events(stream)]
        assert events == ['{"flush": true}']

    @pytest.mark.asyncio
    async def test_blank_lines_ignored(self):
        stream = _chunks('\n\n\ndata: {"x": 1}\n\n\n')
        events = [e async for e in sse_events(stream)]
        assert events == ['{"x": 1}']
