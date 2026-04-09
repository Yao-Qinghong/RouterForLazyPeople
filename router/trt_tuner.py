from __future__ import annotations

"""
router/trt_tuner.py — TensorRT-LLM memory auto-tuner

Searches for the largest context window that fits in GPU memory
by trying progressively smaller configs until one passes the health check.
Saves successful configs to disk so future startups skip the search.
"""

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from router.config import AppConfig


class TRTLLMTuner:
    """
    Auto-tunes TRT-LLM backend memory configuration.

    Search space is tried in order from largest to smallest context.
    The first config that passes the health check is saved and reused.
    """

    SEARCH_SPACE = [
        {"max_input_len": 131072, "kv_cache_dtype": "int8",  "chunked_context": True, "enable_mtp": False, "gpu_memory_fraction": 0.92},
        {"max_input_len": 65536,  "kv_cache_dtype": "int8",  "chunked_context": True, "enable_mtp": False, "gpu_memory_fraction": 0.92},
        {"max_input_len": 32768,  "kv_cache_dtype": "int8",  "chunked_context": True, "enable_mtp": False, "gpu_memory_fraction": 0.90},
        {"max_input_len": 16384,  "kv_cache_dtype": "int8",  "chunked_context": True, "enable_mtp": False, "gpu_memory_fraction": 0.85},
        {"max_input_len": 16384,  "kv_cache_dtype": "fp8",   "chunked_context": True, "enable_mtp": False, "gpu_memory_fraction": 0.85},
        {"max_input_len": 8192,   "kv_cache_dtype": "int8",  "chunked_context": True, "enable_mtp": False, "gpu_memory_fraction": 0.80},
    ]

    OOM_SIGNALS = [
        "out of memory", "oom", "cuda error", "cuda out of memory",
        "cannot allocate", "memory allocation failed", "insufficient memory",
        "std::bad_alloc", "cublas_status_alloc_failed",
    ]

    def __init__(self, key: str, config: "AppConfig"):
        self.key = key
        self.tune_dir = config.data_dir / "tuning"
        self.tune_dir.mkdir(parents=True, exist_ok=True)
        self.tune_file = self.tune_dir / f"{key}.json"

    def load_saved(self) -> dict | None:
        if self.tune_file.exists():
            try:
                with open(self.tune_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def save(self, config: dict):
        with open(self.tune_file, "w") as f:
            json.dump(config, f, indent=2)

    def clear(self):
        if self.tune_file.exists():
            self.tune_file.unlink()

    def is_oom(self, log_path: str) -> bool:
        """Scan the tail of a log file for OOM error signatures."""
        try:
            with open(log_path) as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 5000))
                tail = f.read().lower()
            return any(sig in tail for sig in self.OOM_SIGNALS)
        except Exception:
            return False

    def get_search_space(self, base_config: dict) -> list[dict]:
        """Build the list of configs to try, starting with base_config if provided."""
        configs = []
        if base_config:
            configs.append(base_config)
        for trial in self.SEARCH_SPACE:
            merged = {**base_config, **trial} if base_config else trial
            if merged not in configs:
                configs.append(merged)
        return configs
