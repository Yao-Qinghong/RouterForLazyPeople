"""
router/metrics.py — Request benchmarking and metrics store

Tracks per-request performance stats (TTFT, latency, tokens/sec)
and persists them to daily JSONL files for analysis.

Usage in proxy.py:
    store = MetricsStore(config)
    rec = RequestRecord(...)   # fill in timing after request completes
    store.record(rec)

Endpoints:
    GET /metrics            → store.summary()
    GET /metrics/export     → store.export_csv(path)
    GET /metrics/prometheus → store.prometheus()
"""

import asyncio
import csv
import json
import logging
import statistics
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from router.config import AppConfig

logger = logging.getLogger("llm-router.metrics")


@dataclass
class RequestRecord:
    """One row of benchmark data per inference request."""
    request_id: str
    timestamp_utc: str           # ISO 8601
    backend_key: str
    engine: str
    model_path: str
    endpoint: str                # e.g. "chat/completions"
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float               # time-to-first-token in ms (= total for non-streaming)
    total_latency_ms: float      # wall time: request start → last byte
    tokens_per_sec: float        # completion_tokens / (total_latency_ms / 1000)
    status_code: int
    error: Optional[str] = None

    @classmethod
    def make_id(cls) -> str:
        return uuid.uuid4().hex[:8]


class MetricsStore:
    """
    In-memory ring buffer + JSONL file persistence for request metrics.

    Thread-safe for asyncio (uses asyncio.Lock).
    Does NOT use any external database — just plain files.
    """

    RING_SIZE = 1000   # keep last N records in memory for fast summary

    def __init__(self, config: "AppConfig"):
        self.config = config
        self.persist_dir = config.metrics.persist_dir
        self.flush_interval = config.metrics.flush_interval_sec
        self._ring: deque[RequestRecord] = deque(maxlen=self.RING_SIZE)
        self._pending: list[RequestRecord] = []   # buffer waiting to be flushed
        self._lock = asyncio.Lock()
        self._enabled = config.metrics.enabled
        # Global counters for Prometheus (never reset)
        self._total_requests = 0
        self._total_errors = 0
        self._total_tokens = 0

    def record(self, rec: RequestRecord):
        """Add a record to the ring buffer and the flush-pending list."""
        if not self._enabled:
            return
        self._ring.append(rec)
        self._pending.append(rec)
        self._total_requests += 1
        if rec.error or rec.status_code >= 400:
            self._total_errors += 1
        self._total_tokens += rec.prompt_tokens + rec.completion_tokens

    async def flush(self):
        """Write pending records to today's JSONL file."""
        if not self._pending:
            return
        async with self._lock:
            to_write = self._pending[:]
            self._pending.clear()

        if not to_write:
            return

        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_file = self.persist_dir / f"{today}.jsonl"
            with open(log_file, "a") as f:
                for rec in to_write:
                    f.write(json.dumps(asdict(rec)) + "\n")
            logger.debug(f"Flushed {len(to_write)} metric records to {log_file}")
        except Exception as e:
            logger.warning(f"Metrics flush failed: {e}")

    async def flush_loop(self):
        """Background task: flush metrics every flush_interval_sec."""
        while True:
            await asyncio.sleep(self.flush_interval)
            await self.flush()

    def summary(self) -> dict:
        """
        Return aggregated per-backend stats from the in-memory ring buffer.
        Fast path — no file I/O.
        """
        by_backend: dict[str, list[RequestRecord]] = {}
        for rec in self._ring:
            by_backend.setdefault(rec.backend_key, []).append(rec)

        result = {}
        now = time.time()

        for key, records in by_backend.items():
            errors = [r for r in records if r.error is not None or r.status_code >= 400]
            successes = [r for r in records if r.error is None and r.status_code < 400]

            ttfts = [r.ttft_ms for r in successes] if successes else []
            latencies = [r.total_latency_ms for r in successes] if successes else []
            tps_list = [r.tokens_per_sec for r in successes if r.tokens_per_sec > 0]

            # Last 24 hours
            cutoff = now - 86400
            recent = [r for r in records if _ts_to_epoch(r.timestamp_utc) > cutoff]

            result[key] = {
                "request_count":        len(records),
                "success_count":        len(successes),
                "error_count":          len(errors),
                "error_rate":           round(len(errors) / len(records), 3) if records else 0.0,
                "last_24h_count":       len(recent),
                "avg_ttft_ms":          round(statistics.mean(ttfts), 1) if ttfts else None,
                "p50_ttft_ms":          round(_percentile(ttfts, 50), 1) if ttfts else None,
                "p95_ttft_ms":          round(_percentile(ttfts, 95), 1) if ttfts else None,
                "avg_total_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
                "avg_tokens_per_sec":   round(statistics.mean(tps_list), 1) if tps_list else None,
            }

        return result

    def prometheus(self) -> str:
        """
        Return metrics in Prometheus text exposition format.
        Scrapeable by Prometheus, VictoriaMetrics, Grafana Agent, etc.
        """
        lines = []
        lines.append("# HELP llm_router_requests_total Total number of inference requests")
        lines.append("# TYPE llm_router_requests_total counter")
        lines.append(f"llm_router_requests_total {self._total_requests}")

        lines.append("# HELP llm_router_errors_total Total number of failed requests")
        lines.append("# TYPE llm_router_errors_total counter")
        lines.append(f"llm_router_errors_total {self._total_errors}")

        lines.append("# HELP llm_router_tokens_total Total tokens processed (prompt + completion)")
        lines.append("# TYPE llm_router_tokens_total counter")
        lines.append(f"llm_router_tokens_total {self._total_tokens}")

        # Per-backend gauges from ring buffer
        summary = self.summary()
        for backend, stats in summary.items():
            labels = f'backend="{backend}"'

            lines.append(f"# HELP llm_router_backend_requests Backend request count (ring buffer)")
            lines.append(f"# TYPE llm_router_backend_requests gauge")
            lines.append(f'llm_router_backend_requests{{{labels}}} {stats["request_count"]}')

            lines.append(f"# HELP llm_router_backend_errors Backend error count (ring buffer)")
            lines.append(f"# TYPE llm_router_backend_errors gauge")
            lines.append(f'llm_router_backend_errors{{{labels}}} {stats["error_count"]}')

            if stats["avg_ttft_ms"] is not None:
                lines.append(f"# HELP llm_router_ttft_avg_ms Average time to first token (ms)")
                lines.append(f"# TYPE llm_router_ttft_avg_ms gauge")
                lines.append(f'llm_router_ttft_avg_ms{{{labels}}} {stats["avg_ttft_ms"]}')

            if stats["p95_ttft_ms"] is not None:
                lines.append(f"# HELP llm_router_ttft_p95_ms P95 time to first token (ms)")
                lines.append(f"# TYPE llm_router_ttft_p95_ms gauge")
                lines.append(f'llm_router_ttft_p95_ms{{{labels}}} {stats["p95_ttft_ms"]}')

            if stats["avg_total_latency_ms"] is not None:
                lines.append(f"# HELP llm_router_latency_avg_ms Average total latency (ms)")
                lines.append(f"# TYPE llm_router_latency_avg_ms gauge")
                lines.append(f'llm_router_latency_avg_ms{{{labels}}} {stats["avg_total_latency_ms"]}')

            if stats["avg_tokens_per_sec"] is not None:
                lines.append(f"# HELP llm_router_tokens_per_sec Average tokens per second")
                lines.append(f"# TYPE llm_router_tokens_per_sec gauge")
                lines.append(f'llm_router_tokens_per_sec{{{labels}}} {stats["avg_tokens_per_sec"]}')

        return "\n".join(lines) + "\n"

    def export_csv(self, output_path: Path):
        """Write all history (JSONL files) to a single CSV file."""
        records = self.load_history(days=365)
        if not records:
            output_path.write_text("")
            return

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
            writer.writeheader()
            for rec in records:
                writer.writerow(asdict(rec))

    def load_history(self, days: int = 7) -> list[RequestRecord]:
        """Read JSONL files from the last N days into a list."""
        results = []
        if not self.persist_dir.exists():
            return results

        from datetime import timedelta
        today = datetime.now(timezone.utc).date()

        for i in range(days):
            date = today - timedelta(days=i)
            log_file = self.persist_dir / f"{date}.jsonl"
            if not log_file.exists():
                continue
            try:
                with open(log_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        results.append(RequestRecord(**data))
            except Exception as e:
                logger.warning(f"Could not read metrics file {log_file}: {e}")

        results.sort(key=lambda r: r.timestamp_utc)
        return results


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * pct / 100
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def _ts_to_epoch(ts: str) -> float:
    """Parse ISO 8601 timestamp to Unix epoch float."""
    try:
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return 0.0


def extract_token_counts(response_body: dict) -> tuple[int, int]:
    """
    Extract prompt and completion token counts from an inference response.
    Falls back to word-count estimate if usage data is unavailable.
    """
    usage = response_body.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    if completion_tokens == 0:
        # Rough estimate from response text
        choices = response_body.get("choices", [])
        text = ""
        for choice in choices:
            msg = choice.get("message", {})
            text += msg.get("content", "")
            text += choice.get("text", "")
        completion_tokens = max(1, int(len(text.split()) * 1.3))

    return prompt_tokens, completion_tokens
