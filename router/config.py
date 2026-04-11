"""
router/config.py — Configuration loader

Reads config/settings.yaml and config/backends.yaml (or user-specified paths)
and returns typed dataclass objects used throughout the router.

Search order for settings.yaml:
  1. Explicit path passed to load_config()
  2. $LLM_ROUTER_CONFIG environment variable
  3. ./config/settings.yaml  (project-local, next to cli.py)
  4. ~/.llm-router/settings.yaml  (user override)
"""
from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class ConfigError(Exception):
    """Raised when a required config value is missing or invalid."""


# ─────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────

@dataclass
class RouterSettings:
    host: str = "0.0.0.0"
    port: int = 9001
    log_level: str = "INFO"


@dataclass
class LoggingSettings:
    log_dir: Path = Path("~/.llm-router/logs")
    log_max_bytes: int = 10_485_760   # 10 MB
    log_backup_count: int = 5
    json_format: bool = False         # structured JSON logging


@dataclass
class ScanDirs:
    gguf: list[Path] = field(default_factory=list)
    hf: list[Path] = field(default_factory=list)
    trtllm: list[Path] = field(default_factory=list)


@dataclass
class DiscoverySettings:
    port_start: int = 8100
    port_end: int = 8299
    probe_ports: list[int] = field(default_factory=lambda: [8080, 8000, 8001, 8002, 30000])


@dataclass
class TierThresholds:
    fast: float = 15.0
    mid: float = 40.0


@dataclass
class IdleTimeouts:
    fast: int = 300
    mid: int = 180
    deep: int = 600


@dataclass
class RoutingConfig:
    token_threshold_deep: int = 4000
    token_threshold_mid: int = 500
    deep_keywords: list[str] = field(default_factory=list)
    mid_keywords: list[str] = field(default_factory=list)


@dataclass
class ProxyConfig:
    timeout_sec: int = 300
    max_concurrent_requests: int = 20
    queue_timeout_sec: int = 30
    retry_attempts: int = 2           # retries on transient errors (0 = no retry)
    retry_backoff_sec: float = 1.0    # base backoff between retries
    retry_on_status: list[int] = field(default_factory=lambda: [502, 503, 504])


@dataclass
class MetricsConfig:
    enabled: bool = True
    persist_dir: Path = Path("~/.llm-router/metrics")
    flush_interval_sec: int = 60


@dataclass
class AuthConfig:
    enabled: bool = False
    api_keys: list[dict] = field(default_factory=list)
    # Each key: {"key": "sk-...", "name": "dev", "scope": "all"}


@dataclass
class CORSConfig:
    enabled: bool = True
    allow_origins: list[str] = field(default_factory=lambda: ["*"])
    allow_methods: list[str] = field(default_factory=lambda: ["*"])
    allow_headers: list[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = False


@dataclass
class AuditConfig:
    enabled: bool = False
    log_dir: Path = Path("~/.llm-router/audit")
    log_request_body: bool = False    # log full request body (privacy risk)
    log_response_body: bool = False   # log full response body (privacy risk)
    redact_content: bool = True       # strip message content, keep metadata


@dataclass
class RateLimitConfig:
    enabled: bool = False
    default_rpm: int = 60             # requests per minute per key
    default_tpm: int = 100_000        # tokens per minute per key
    per_key_overrides: dict = field(default_factory=dict)
    # e.g. {"sk-admin": {"rpm": 0, "tpm": 0}}  (0 = unlimited)


@dataclass
class TRTLLMDockerSettings:
    enabled: bool = True
    image: str = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc7"
    container_port: int = 8000
    hf_cache_dir: Path = Path("~/.cache/huggingface")
    log_dir: Path = Path("~/.llm-router/trtllm-logs")
    env: dict = field(default_factory=lambda: {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    serve_defaults: dict = field(default_factory=lambda: {
        "max_seq_len": 65536,
        "max_num_tokens": 16384,
        "max_batch_size": 4,
        "kv_cache_free_gpu_memory_fraction": 0.75,
    })


@dataclass
class BackendCapabilities:
    supports_tools: bool = False
    supports_json_schema: bool = False
    max_context: int = 32768
    code_quality: str = "good"        # "weak" | "good" | "strong"


@dataclass
class BackendConfig:
    engine: str = "llama.cpp"
    port: int = 8080
    model: str = ""
    model_dir: str = ""
    log: str = ""
    tier: str = ""
    size_gb: float | None = None
    vram_estimate_gb: float | None = None
    ctx_size: int = 32768
    gpu_layers: int = 999
    flash_attn: bool = True
    reasoning: bool = False
    reasoning_budget: int | None = None
    idle_timeout: int = 300
    startup_wait: int = 30
    auto_discovered: bool = False
    description: str = ""
    dtype: str = "auto"
    gpu_memory_fraction: float = 0.90
    trust_remote_code: bool = False
    tensor_parallel_size: int = 1
    quantization: str | None = None
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    tokenizer: str = ""
    wrapper_script: str = ""
    model_type: str = ""
    trt_config: dict = field(default_factory=dict)
    docker_config: dict = field(default_factory=dict)
    extra_args: list = field(default_factory=list)
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)

    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


def _infer_capabilities(engine: str, size_gb: float | None, name: str = "") -> BackendCapabilities:
    """Infer backend capabilities from engine type and model size/name."""
    size = size_gb or 0.0
    name_lower = name.lower()
    if size >= 25 or any(kw in name_lower for kw in ("70b", "72b", "65b", "32b", "34b")):
        return BackendCapabilities(
            supports_tools=True, supports_json_schema=True,
            max_context=65536, code_quality="strong",
        )
    elif size >= 8 or any(kw in name_lower for kw in ("14b", "13b", "27b")):
        return BackendCapabilities(
            supports_tools=True, supports_json_schema=True,
            max_context=32768, code_quality="good",
        )
    return BackendCapabilities(
        supports_tools=False, supports_json_schema=False,
        max_context=32768, code_quality="weak",
    )


def _estimate_vram(engine: str, size_gb: float | None) -> float | None:
    """Estimate VRAM needed. GGUF × 1.15 for KV cache; HF/vLLM × 1.3."""
    if size_gb is None:
        return None
    if engine == "llama.cpp":
        return round(size_gb * 1.15, 2)
    elif engine in ("vllm", "sglang", "huggingface", "trt-llm", "trt-llm-docker"):
        return round(size_gb * 1.3, 2)
    return None


@dataclass
class AppConfig:
    router: RouterSettings
    logging: LoggingSettings
    scan_dirs: ScanDirs
    discovery: DiscoverySettings
    tier_thresholds: TierThresholds
    idle_timeouts: IdleTimeouts
    routing: RoutingConfig
    proxy: ProxyConfig
    metrics: MetricsConfig
    auth: AuthConfig
    cors: CORSConfig
    audit: AuditConfig
    rate_limit: RateLimitConfig
    trtllm_docker: TRTLLMDockerSettings
    llama_bin: Path
    data_dir: Path
    settings_file: Path
    backends_file: Path
    model_aliases: dict[str, str] = field(default_factory=dict)
    preload: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────

def _expand(p) -> Path:
    """Expand ~ and return a Path."""
    return Path(os.path.expanduser(str(p)))


def _find_settings_file(explicit: Optional[Path] = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("LLM_ROUTER_CONFIG")
    if env:
        candidates.append(Path(env))
    # Project-local (next to cli.py / this file's package root)
    here = Path(__file__).parent.parent / "config" / "settings.yaml"
    candidates.append(here)
    candidates.append(_expand("~/.llm-router/settings.yaml"))

    for p in candidates:
        if p.exists():
            return p
    raise ConfigError(
        "Could not find settings.yaml. Tried:\n" +
        "\n".join(f"  {p}" for p in candidates)
    )


def _find_backends_file(settings_path: Path, explicit: Optional[Path] = None) -> Path:
    if explicit and Path(explicit).exists():
        return Path(explicit)
    # Same directory as settings.yaml
    sibling = settings_path.parent / "backends.yaml"
    if sibling.exists():
        return sibling
    fallback = _expand("~/.llm-router/backends.yaml")
    if fallback.exists():
        return fallback
    raise ConfigError(
        f"Could not find backends.yaml. Expected it next to {settings_path}"
    )


def load_config(
    settings_path: Optional[Path] = None,
    backends_path: Optional[Path] = None,
) -> "AppConfig":
    """
    Load settings.yaml (and locate backends.yaml) into an AppConfig.
    Call this once at startup; pass the result into all other modules.
    """
    sf = _find_settings_file(settings_path)
    with open(sf) as f:
        raw = yaml.safe_load(f) or {}

    # ── Router ────────────────────────────────────────────────
    r = raw.get("router", {})
    router = RouterSettings(
        host=r.get("host", "0.0.0.0"),
        port=int(r.get("port", 9001)),
        log_level=r.get("log_level", "INFO").upper(),
    )

    # ── Logging ───────────────────────────────────────────────
    lg = raw.get("logging", {})
    logging_cfg = LoggingSettings(
        log_dir=_expand(lg.get("log_dir", "~/.llm-router/logs")),
        log_max_bytes=int(lg.get("log_max_bytes", 10_485_760)),
        log_backup_count=int(lg.get("log_backup_count", 5)),
        json_format=bool(lg.get("json_format", False)),
    )

    # ── Paths ─────────────────────────────────────────────────
    llama_bin = _expand(raw.get("llama_bin", "~/llama.cpp/build/bin/llama-server"))
    data_dir  = _expand(raw.get("data_dir",  "~/.llm-router"))

    # ── Scan dirs ─────────────────────────────────────────────
    sd = raw.get("scan_dirs", {})
    scan_dirs = ScanDirs(
        gguf=[_expand(p) for p in sd.get("gguf", [])],
        hf=[_expand(p) for p in sd.get("hf", [])],
        trtllm=[_expand(p) for p in sd.get("trtllm", [])],
    )

    # ── Discovery ─────────────────────────────────────────────
    disc = raw.get("discovery", {})
    discovery = DiscoverySettings(
        port_start=int(disc.get("port_start", 8100)),
        port_end=int(disc.get("port_end", 8299)),
        probe_ports=[int(p) for p in disc.get("probe_ports", [8080, 8000, 8001, 8002, 30000])],
    )

    # ── Tier thresholds ───────────────────────────────────────
    tt = raw.get("tier_thresholds_gb", {})
    tier_thresholds = TierThresholds(
        fast=float(tt.get("fast", 15)),
        mid=float(tt.get("mid", 40)),
    )

    # ── Idle timeouts ─────────────────────────────────────────
    it = raw.get("idle_timeouts_sec", {})
    idle_timeouts = IdleTimeouts(
        fast=int(it.get("fast", 300)),
        mid=int(it.get("mid", 180)),
        deep=int(it.get("deep", 600)),
    )

    # ── Routing ───────────────────────────────────────────────
    rt = raw.get("routing", {})
    routing = RoutingConfig(
        token_threshold_deep=int(rt.get("token_threshold_deep", 4000)),
        token_threshold_mid=int(rt.get("token_threshold_mid", 500)),
        deep_keywords=rt.get("deep_keywords", []),
        mid_keywords=rt.get("mid_keywords", []),
    )

    # ── Proxy ─────────────────────────────────────────────────
    px = raw.get("proxy", {})
    proxy = ProxyConfig(
        timeout_sec=int(px.get("timeout_sec", 300)),
        max_concurrent_requests=int(px.get("max_concurrent_requests", 20)),
        queue_timeout_sec=int(px.get("queue_timeout_sec", 30)),
        retry_attempts=int(px.get("retry_attempts", 2)),
        retry_backoff_sec=float(px.get("retry_backoff_sec", 1.0)),
        retry_on_status=[int(s) for s in px.get("retry_on_status", [502, 503, 504])],
    )

    # ── Metrics ───────────────────────────────────────────────
    mx = raw.get("metrics", {})
    metrics = MetricsConfig(
        enabled=bool(mx.get("enabled", True)),
        persist_dir=_expand(mx.get("persist_dir", "~/.llm-router/metrics")),
        flush_interval_sec=int(mx.get("flush_interval_sec", 60)),
    )

    # ── Auth ──────────────────────────────────────────────────
    au = raw.get("auth", {})
    auth = AuthConfig(
        enabled=bool(au.get("enabled", False)),
        api_keys=au.get("api_keys", []),
    )

    # ── CORS ──────────────────────────────────────────────────
    co = raw.get("cors", {})
    cors = CORSConfig(
        enabled=bool(co.get("enabled", True)),
        allow_origins=co.get("allow_origins", ["*"]),
        allow_methods=co.get("allow_methods", ["*"]),
        allow_headers=co.get("allow_headers", ["*"]),
        allow_credentials=bool(co.get("allow_credentials", False)),
    )

    # ── Audit logging ─────────────────────────────────────────
    ad = raw.get("audit", {})
    audit = AuditConfig(
        enabled=bool(ad.get("enabled", False)),
        log_dir=_expand(ad.get("log_dir", "~/.llm-router/audit")),
        log_request_body=bool(ad.get("log_request_body", False)),
        log_response_body=bool(ad.get("log_response_body", False)),
        redact_content=bool(ad.get("redact_content", True)),
    )

    # ── Rate limiting ─────────────────────────────────────────
    rl = raw.get("rate_limit", {})
    rate_limit = RateLimitConfig(
        enabled=bool(rl.get("enabled", False)),
        default_rpm=int(rl.get("default_rpm", 60)),
        default_tpm=int(rl.get("default_tpm", 100_000)),
        per_key_overrides=rl.get("per_key_overrides", {}),
    )

    # ── TensorRT-LLM Docker fallback ─────────────────────────
    td = raw.get("trtllm_docker", {})
    trtllm_docker = TRTLLMDockerSettings(
        enabled=bool(td.get("enabled", True)),
        image=str(td.get("image", TRTLLMDockerSettings.image)),
        container_port=int(td.get("container_port", 8000)),
        hf_cache_dir=_expand(td.get("hf_cache_dir", "~/.cache/huggingface")),
        log_dir=_expand(td.get("log_dir", "~/.llm-router/trtllm-logs")),
        env=dict(td.get("env", {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})),
        serve_defaults=dict(td.get("serve_defaults", {
            "max_seq_len": 65536,
            "max_num_tokens": 16384,
            "max_batch_size": 4,
            "kv_cache_free_gpu_memory_fraction": 0.75,
        })),
    )

    # ── Model aliases ─────────────────────────────────────────
    model_aliases = raw.get("model_aliases", {})

    # ── Preload ───────────────────────────────────────────────
    preload = raw.get("preload", [])

    bf = _find_backends_file(sf, backends_path)

    return AppConfig(
        router=router,
        logging=logging_cfg,
        scan_dirs=scan_dirs,
        discovery=discovery,
        tier_thresholds=tier_thresholds,
        idle_timeouts=idle_timeouts,
        routing=routing,
        proxy=proxy,
        metrics=metrics,
        auth=auth,
        cors=cors,
        audit=audit,
        rate_limit=rate_limit,
        trtllm_docker=trtllm_docker,
        llama_bin=llama_bin,
        data_dir=data_dir,
        settings_file=sf,
        backends_file=bf,
        model_aliases=model_aliases,
        preload=preload,
    )


def load_backends(config: AppConfig) -> dict[str, BackendConfig]:
    """
    Parse backends.yaml into a dict of typed backend configs.
    Each entry is validated for required fields.
    Model/model_dir paths are ~ -expanded.
    Returns dict[slug → BackendConfig].
    """
    with open(config.backends_file) as f:
        raw = yaml.safe_load(f) or {}

    entries = raw.get("backends") or {}
    if not isinstance(entries, dict):
        raise ConfigError(
            f"backends.yaml must have a top-level 'backends:' mapping, got {type(entries)}"
        )

    result: dict[str, BackendConfig] = {}
    known_fields = {f.name for f in BackendConfig.__dataclass_fields__.values()}

    for slug, cfg in entries.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"backends.yaml: entry '{slug}' must be a mapping")

        # Required fields
        for field_name in ("engine", "port"):
            if field_name not in cfg:
                raise ConfigError(
                    f"backends.yaml: '{slug}' is missing required field '{field_name}'"
                )

        # Expand ~ in path fields
        for path_key in ("model", "model_dir", "tokenizer", "wrapper_script"):
            if path_key in cfg and cfg[path_key]:
                cfg[path_key] = os.path.expanduser(str(cfg[path_key]))
        docker_cfg = cfg.get("docker_config")
        if isinstance(docker_cfg, dict):
            for path_key in ("hf_cache_dir", "log_dir", "launcher_script"):
                if path_key in docker_cfg and docker_cfg[path_key]:
                    docker_cfg[path_key] = os.path.expanduser(str(docker_cfg[path_key]))

        # Provide defaults that the rest of the code expects
        cfg.setdefault("port", 8080)
        cfg.setdefault("gpu_layers", 999)
        cfg.setdefault("flash_attn", True)
        cfg.setdefault("reasoning", False)
        cfg.setdefault("idle_timeout", 300)
        cfg.setdefault("startup_wait", 30)
        cfg.setdefault("auto_discovered", False)
        cfg.setdefault("description", slug)

        # Build a log path if not provided
        if "log" not in cfg:
            cfg["log"] = str(config.data_dir / "logs" / f"backend-{slug}.log")

        # Construct typed BackendConfig, dropping unknown YAML keys
        filtered = {k: v for k, v in cfg.items() if k in known_fields and k != "capabilities"}
        backend = BackendConfig(**filtered)

        # Capabilities: use explicit YAML overrides if present, else infer
        caps_raw = cfg.get("capabilities")
        if isinstance(caps_raw, dict):
            inferred = _infer_capabilities(backend.engine, backend.size_gb, backend.description)
            known_cap_fields = {f.name for f in BackendCapabilities.__dataclass_fields__.values()}
            cap_values = {k: v for k, v in caps_raw.items() if k in known_cap_fields}
            # Merge: explicit YAML values override inferred defaults
            from dataclasses import asdict
            merged = {**asdict(inferred), **cap_values}
            backend.capabilities = BackendCapabilities(**merged)
        else:
            backend.capabilities = _infer_capabilities(backend.engine, backend.size_gb, backend.description)

        backend.vram_estimate_gb = _estimate_vram(backend.engine, backend.size_gb)
        result[slug] = backend

    return result
