#!/usr/bin/env python3
from __future__ import annotations

"""
cli.py — LLM Router command-line interface

Replaces start_router.sh with a cleaner, Python-native tool.

Usage:
    python cli.py start [--update]     Start the router (optionally update llama.cpp first)
    python cli.py stop                 Stop the running router
    python cli.py status               Show backend run-state
    python cli.py benchmark            Show performance stats
    python cli.py benchmark --export ./report.csv
    python cli.py update [--restart]   Update llama.cpp + pip deps
    python cli.py rescan               Trigger live model rescan (router must be running)
    python cli.py logs                 Tail the router log
"""

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Paths (resolved relative to this file so cli.py is portable)
# ─────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.resolve()
VENV_DIR    = PROJECT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
VENV_PIP    = VENV_DIR / "bin" / "pip"
VENV_UVICORN = VENV_DIR / "bin" / "uvicorn"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"
PID_FILE    = Path.home() / ".llm-router" / "router.pid"
LOG_DIR     = Path.home() / ".llm-router" / "logs"
ROUTER_LOG  = LOG_DIR / "router.log"
MIN_PYTHON = (3, 10)


def _python_version_text(version_info=None) -> str:
    version_info = sys.version_info if version_info is None else version_info
    return f"{version_info[0]}.{version_info[1]}.{version_info[2]}"


def _ensure_supported_python(version_info=None):
    """Fail before creating a venv from an interpreter the router cannot run on."""
    version_info = sys.version_info if version_info is None else version_info
    if version_info >= MIN_PYTHON:
        return

    required = ".".join(str(part) for part in MIN_PYTHON)
    current = _python_version_text(version_info)
    print(
        f"RouterForLazyPeople requires Python {required}+; current interpreter is {current}.",
        file=sys.stderr,
    )
    print("Re-run with python3.10+ or install a newer Python, then delete any old .venv.", file=sys.stderr)
    sys.exit(2)

# Read port from config if available, otherwise use default
def _router_port() -> int:
    config_path = PROJECT_DIR / "config" / "settings.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
            return int(raw.get("router", {}).get("port", 9001))
        except Exception:
            pass
    return 9001


def _router_url() -> str:
    return f"http://localhost:{_router_port()}"


def _uvicorn_cmd() -> list[str]:
    """Command that runs the FastAPI router. Used in foreground and daemon modes."""
    return [
        str(VENV_UVICORN),
        "router.main:create_app",
        "--factory",
        "--host", "0.0.0.0",
        "--port", str(_router_port()),
        "--app-dir", str(PROJECT_DIR),
    ]


def _local_python() -> str:
    """Prefer the project venv when present; otherwise use the current interpreter."""
    return str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))


def _warn_firewall(port: int, lan: str):
    """
    Print a firewall warning + fix command if the port is likely blocked.
    Checks UFW on Linux. Skips silently on macOS or if ufw is not installed.
    """
    if platform.system() != "Linux":
        return
    try:
        result = subprocess.run(["ufw", "status"], capture_output=True, text=True)
        if "Status: active" not in result.stdout:
            return   # UFW inactive — no block
        # Port allowed if the port number appears in the status output
        if str(port) in result.stdout:
            return   # already open
        print(f"  NOTE: UFW firewall is active and port {port} is not open.")
        print(f"  Other PCs cannot connect until you run:")
        print(f"    sudo ufw allow {port}/tcp")
        print(f"  After that, they can use: {lan}:{port}/v1")
        print()
    except FileNotFoundError:
        pass   # ufw not installed — nothing to check


def _lan_ip() -> str:
    """Return the machine's LAN IP (no traffic sent — just resolves local interface)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def _wait_router_ready(timeout_s: float = 20.0) -> tuple[bool, str | None]:
    """Poll /health so user-facing commands do not report success before HTTP is ready."""
    import urllib.request

    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{_router_url()}/health", timeout=1) as response:
                if 200 <= getattr(response, "status", 200) < 500:
                    return True, None
        except Exception as e:
            last_error = str(e)
        time.sleep(0.5)
    return False, last_error


# ─────────────────────────────────────────────────────────────
# Virtual environment helpers
# ─────────────────────────────────────────────────────────────

def ensure_venv():
    """Create venv and install requirements if not already set up."""
    _ensure_supported_python()

    if not VENV_DIR.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)

    print("Installing / verifying dependencies...")
    subprocess.run(
        [str(VENV_PIP), "install", "-q", "-r", str(REQUIREMENTS)],
        check=True,
    )


def update_pip():
    """Reinstall requirements (picks up any version changes in requirements.txt)."""
    print("Updating Python dependencies...")
    subprocess.run(
        [str(VENV_PIP), "install", "-q", "-r", str(REQUIREMENTS)],
        check=True,
    )
    print("Python deps up to date.")


# ─────────────────────────────────────────────────────────────
# llama.cpp update
# ─────────────────────────────────────────────────────────────

def _llama_bin() -> Path:
    """Read the configured llama-server binary path, even if it does not exist yet."""
    config_path = PROJECT_DIR / "config" / "settings.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
            return Path(os.path.expanduser(raw.get("llama_bin", "~/llama.cpp/build/bin/llama-server")))
        except Exception:
            pass
    return Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"

def _llama_dir() -> Path | None:
    """Try to read llama_bin from config and derive the repo dir."""
    llama_bin = _llama_bin()
    # Walk up to find the repo root (has a CMakeLists.txt)
    for parent in llama_bin.parents:
        if (parent / "CMakeLists.txt").exists():
            return parent
    default = Path.home() / "llama.cpp"
    return default if default.exists() else None


def _cuda_compiler() -> str | None:
    """Return a usable nvcc path when a CUDA toolchain is available."""
    candidates = [
        os.environ.get("CUDACXX"),
        str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc") if os.environ.get("CUDA_HOME") else None,
        str(Path(os.environ["CUDA_PATH"]) / "bin" / "nvcc") if os.environ.get("CUDA_PATH") else None,
        shutil.which("nvcc"),
        "/usr/local/cuda/bin/nvcc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _llama_build_config() -> tuple[str, list[str]]:
    """Choose a llama.cpp build profile that matches the local environment."""
    system = platform.system()
    if system == "Darwin":
        return ("Metal", ["cmake", "-B", "build", "-DGGML_METAL=ON"])
    if _cuda_compiler():
        return ("CUDA", ["cmake", "-B", "build", "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=native"])
    return ("CPU", ["cmake", "-B", "build"])


def update_llama():
    """Pull and rebuild llama.cpp, rolling back the checkout/binary on build failure."""
    llama_dir = _llama_dir()
    if not llama_dir or not llama_dir.exists():
        print("llama.cpp directory not found — skipping update.")
        return

    if not shutil.which("git"):
        print("git not found — skipping llama.cpp update.")
        return

    print(f"Checking llama.cpp for updates ({llama_dir})...")

    try:
        local = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=llama_dir, text=True
        ).strip()
        subprocess.run(["git", "fetch", "--quiet"], cwd=llama_dir, check=True)
        remote = subprocess.check_output(
            ["git", "rev-parse", "@{u}"], cwd=llama_dir, text=True
        ).strip()
    except subprocess.CalledProcessError as e:
        print(f"git error: {e} — skipping llama.cpp update.")
        return

    if local == remote:
        try:
            tag = subprocess.check_output(
                ["git", "describe", "--tags", "--always"], cwd=llama_dir, text=True
            ).strip()
        except Exception:
            tag = local[:8]
        print(f"llama.cpp already up to date ({tag})")
        return

    nproc = os.cpu_count() or 4
    build_mode, configure_cmd = _llama_build_config()
    llama_bin = _llama_bin()
    backup_dir = None
    binary_backup = None

    # The Git checkout and runnable llama-server are separate. Keep a binary
    # copy so a failed compile does not replace the user's last working server.
    if llama_bin.exists():
        backup_dir = Path(tempfile.mkdtemp(prefix="router-llama-update-"))
        binary_backup = backup_dir / llama_bin.name
        shutil.copy2(llama_bin, binary_backup)

    try:
        print("Pulling latest changes...")
        subprocess.run(["git", "pull", "--ff-only", "--quiet"], cwd=llama_dir, check=True)

        print(f"Rebuilding for {build_mode} (using {nproc} cores — this may take a few minutes)...")
        subprocess.run(
            configure_cmd,
            cwd=llama_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["cmake", "--build", "build", "--config", "Release", f"-j{nproc}"],
            cwd=llama_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print("llama.cpp update failed — rolling source checkout back to the previous commit...")
        subprocess.run(["git", "checkout", "--quiet", local], cwd=llama_dir, check=False)
        if binary_backup and llama_bin.parent.exists():
            shutil.copy2(binary_backup, llama_bin)
            print(f"Restored previous llama-server binary: {llama_bin}")
        print("Rollback complete. Fix the build error, then run update again.")
        raise
    finally:
        if backup_dir:
            shutil.rmtree(backup_dir, ignore_errors=True)

    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--always"], cwd=llama_dir, text=True
        ).strip()
    except Exception:
        tag = "updated"
    print(f"llama.cpp updated to {tag}")


# ─────────────────────────────────────────────────────────────
# Process management
# ─────────────────────────────────────────────────────────────

def _kill_port(port: int):
    """Kill whatever process is listening on a given port."""
    if not shutil.which("lsof"):
        return
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid:
                subprocess.run(["kill", pid], capture_output=True)
                print(f"Stopped existing process on port {port} (PID {pid})")
        if pids:
            time.sleep(1)
    except Exception:
        pass


def _save_pid(pid: int):
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid))


def _read_pid() -> int | None:
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except Exception:
            pass
    return None


def _stop_saved_router_pid():
    """Stop a router that was started by `./router-start start` and recorded in PID_FILE."""
    pid = _read_pid()
    if not pid:
        return
    subprocess.run(["kill", str(pid)], capture_output=True)
    PID_FILE.unlink(missing_ok=True)
    print(f"Stopped existing daemonized router (PID {pid})")


# ─────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────

def cmd_start(args):
    if getattr(args, "update", False):
        update_llama()

    ensure_venv()

    port = _router_port()
    _kill_port(port)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting LLM Router on port {port}...")
    log_handle = open(ROUTER_LOG, "a")

    proc = subprocess.Popen(
        _uvicorn_cmd(),
        stdout=log_handle,
        stderr=log_handle,
    )

    _save_pid(proc.pid)
    ready, error = _wait_router_ready()
    if not ready:
        print(f"\nRouter process started with PID {proc.pid}, but HTTP is not ready: {error}")
        print(f"Logs: {ROUTER_LOG}")
        sys.exit(1)

    base    = _router_url()
    lan     = _lan_ip()
    same_pc = f"http://localhost:{port}/v1"
    other_pc = f"http://{lan}:{port}/v1"

    print(f"\nStarted with PID {proc.pid}\n")

    w = max(len(same_pc), len(other_pc)) + 24
    def _row(label, value):
        pad = w - 4 - len(label) - len(value)
        print(f"│  {label}{value}" + " " * max(pad, 1) + "│")

    print("┌─ Connect your LLM client ─" + "─" * (w - 28) + "┐")
    _row("On this machine  : ", same_pc)
    _row("From another PC  : ", other_pc)
    _row("API key          : ", "any string  (no auth by default)")
    print("└" + "─" * (w - 1) + "┘")
    print()

    _warn_firewall(port, lan)

    print("  Works with: Open WebUI · LM Studio · Cursor · Continue · Jan")
    print("              OpenClaw · anything OpenAI-compatible")
    print()
    print("  Anthropic SDK (Claude Code, claude.ai/code):")
    print(f"    base_url = {base}/anthropic    api_key = any string")
    print()
    print("  Logs : ./router-start logs")
    print("  Stop : ./router-start stop")


def cmd_serve(args):
    """Run the router in the foreground. Intended for systemd/launchd service managers."""
    ensure_venv()
    os.execv(str(VENV_UVICORN), _uvicorn_cmd())


def cmd_stop(args):
    port = _router_port()
    _stop_saved_router_pid()
    _kill_port(port)


def _status_bench_label(info: dict) -> str:
    tg = info.get("bench_tg_tok_s")
    pp = info.get("bench_pp_tok_s")
    if tg is None and pp is None:
        return "not benchmarked"

    parts = []
    if tg is not None:
        parts.append(f"TG {tg:g} tok/s")
    if pp is not None:
        parts.append(f"PP {pp:g} tok/s")
    if info.get("bench_tier_measured"):
        parts.append(f"measured tier {info['bench_tier_measured']}")
    if info.get("bench_mismatch"):
        parts.append("tier mismatch")
    return " | ".join(parts)


def _print_empty_status():
    print("Router is reachable, but no backends/models are registered.")
    print()
    print("Next checks:")
    print("  ./router-start sysinfo")
    print("  open http://localhost:9001/v1/models")
    print("  edit config/settings.yaml scan_dirs, then run ./router-start rescan")


def _print_status(data: dict):
    if not data:
        _print_empty_status()
        return

    running_count = sum(1 for info in data.values() if info.get("running"))
    measured_count = sum(1 for info in data.values() if info.get("bench_tg_tok_s") is not None)
    print(f"Router: {_router_url()}")
    print(f"Registered backends: {len(data)} | running: {running_count} | benchmarked: {measured_count}")
    print()

    print(f"{'Tier':<6} {'Run':<3} {'Engine':<10} {'Port':<5} Backend key")
    print(f"{'-' * 6} {'-' * 3} {'-' * 10} {'-' * 5} {'-' * 42}")

    # Group by tier for clarity
    tier_order = ["fast", "mid", "deep", None]
    by_tier: dict = {t: [] for t in tier_order}
    for key, info in data.items():
        tier = info.get("tier")
        by_tier.setdefault(tier, []).append((key, info))

    for tier in tier_order:
        entries = sorted(by_tier.get(tier, []))
        if not entries:
            continue
        tier_label = tier or "—"
        for key, info in entries:
            running = "yes" if info.get("running") else "no"
            engine = info.get("engine", "")
            port = info.get("port", "")
            desc = info.get("description", key)
            print(f"{tier_label:<6} {running:<3} {_shorten(engine, 10):<10} {str(port):<5} {key}")
            print(f"{'':6} {'':3} {'Model':<10} {'':5} {_shorten(desc, 92)}")
            print(f"{'':6} {'':3} {'Bench':<10} {'':5} {_status_bench_label(info)}")
        print()

    if running_count:
        print("Benchmark currently running backend(s):")
        print("  ./router-start bench")
    else:
        print("No model backend is running. That is OK: the router lazy-loads one model per request.")
        print("DGX Spark safe benchmark:")
        print("  ./router-start bench --backend <backend-key> --start-stopped")

    print()
    print("If the list is wrong: edit config/settings.yaml scan_dirs, then run ./router-start rescan")


def cmd_status(args):
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(f"{_router_url()}/status", timeout=5) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"Router not reachable: {e}")
        sys.exit(1)

    _print_status(data)


def cmd_benchmark(args):
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(f"{_router_url()}/metrics", timeout=5) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"Router not reachable: {e}")
        sys.exit(1)

    if not data:
        print("No metrics recorded yet. Send some requests first.")
        return

    if args.export:
        import urllib.request
        export_path = Path(args.export)
        with urllib.request.urlopen(f"{_router_url()}/metrics/export", timeout=30) as r:
            export_path.write_bytes(r.read())
        print(f"Exported to {export_path}")
        return

    print(f"{'Backend':<20} {'Reqs':>6} {'Errors':>7} {'TTFT p50':>10} {'TTFT p95':>10} {'Tok/s':>8} {'24h':>5}")
    print("─" * 75)
    for key, stats in data.items():
        reqs   = stats.get("request_count", 0)
        errors = stats.get("error_count", 0)
        p50    = f"{stats['p50_ttft_ms']:.0f}ms" if stats.get("p50_ttft_ms") else "-"
        p95    = f"{stats['p95_ttft_ms']:.0f}ms" if stats.get("p95_ttft_ms") else "-"
        tps    = f"{stats['avg_tokens_per_sec']:.1f}" if stats.get("avg_tokens_per_sec") else "-"
        recent = stats.get("last_24h_count", 0)
        print(f"{key:<20} {reqs:>6} {errors:>7} {p50:>10} {p95:>10} {tps:>8} {recent:>5}")


def cmd_update(args):
    update_llama()
    update_pip()

    if args.restart:
        print("Restarting router...")
        cmd_stop(args)
        time.sleep(2)
        cmd_start(argparse.Namespace(update=False))


def cmd_rescan(args):
    import urllib.request, urllib.error
    try:
        req = urllib.request.Request(
            f"{_router_url()}/rescan",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=b"{}",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        print(f"Rescan complete: {data['total']} backends ({data['discovered']} discovered)")
        print("Backends:", ", ".join(data["backends"]))
    except Exception as e:
        print(f"Rescan failed: {e}")
        sys.exit(1)


def _bench_model_label(cfg: dict) -> str:
    """Readable model identity for benchmark progress output."""
    model = cfg.get("model")
    if not model:
        return str(cfg.get("description") or "model not specified")

    path = Path(str(model))
    if path.name:
        return path.name
    return str(model)


def _shorten(text, width: int) -> str:
    """Keep terminal tables readable without hiding the start of model names."""
    text = str(text)
    if len(text) <= width:
        return text
    if width <= 1:
        return "…"[:width]
    return text[:width - 1] + "…"


def _print_bench_plan(backends: dict, runnable: list[str]):
    """Print the benchmark target list before slow model startup begins."""
    print("Benchmark plan:")
    print(f"{'#':>2}  {'Backend':<26} {'Engine':<10} {'Tier':<6} {'Port':<6} {'Model'}")
    print(f"{'--':>2}  {'-' * 26} {'-' * 10} {'-' * 6} {'-' * 6} {'-' * 32}")
    for index, key in enumerate(runnable, start=1):
        cfg = backends[key]
        print(
            f"{index:>2}  "
            f"{_shorten(key, 26):<26} "
            f"{_shorten(cfg.get('engine', 'unknown'), 10):<10} "
            f"{_shorten(cfg.get('tier', 'unknown'), 6):<6} "
            f"{str(cfg.get('port', '')):<6} "
            f"{_shorten(_bench_model_label(cfg), 48)}"
        )
    print()


def _suggest_bench_keys(backends: dict, limit: int = 3) -> list[str]:
    """Prefer fast llama.cpp backends for beginner-safe one-model benchmark suggestions."""
    priority = {"fast": 0, "mid": 1, "deep": 2}

    def score(key: str) -> tuple:
        cfg = backends[key]
        engine_score = 0 if cfg.get("engine") == "llama.cpp" else 1
        size = cfg.get("size_gb")
        size_score = size if isinstance(size, (int, float)) else 9999
        return (
            priority.get(cfg.get("tier"), 3),
            engine_score,
            size_score,
            key,
        )

    return sorted(backends, key=score)[:limit]


def _print_no_running_bench_help(backends: dict):
    print("No model is running, so nothing was benchmarked.")
    print()
    print("Safest next step: benchmark ONE model. Copy one command:")
    for key in _suggest_bench_keys(backends):
        cfg = backends[key]
        print()
        print(f"  # {cfg.get('tier', 'unknown')} | {cfg.get('engine', 'unknown')} | {_shorten(_bench_model_label(cfg), 70)}")
        print(f"  ./router-start bench --backend {key} --start-stopped")
    print()
    print("Need the full backend list?")
    print("  ./router-start status")
    print("  ./router-start bench --list")


def _speed_label(value, unit: str) -> str:
    return f"{value:g} {unit}" if isinstance(value, (int, float)) else "not measured"


def _benchmark_result_sort_key(result: dict) -> tuple:
    tier_order = {"fast": 0, "mid": 1, "deep": 2}
    measured = result.get("tier_measured") or "unmeasured"
    tg = result.get("tg_tok_s")
    return (
        tier_order.get(measured, 3),
        -(tg if isinstance(tg, (int, float)) else -1),
        result.get("backend_key", ""),
    )


def _best_benchmark_result(measured: list[dict]) -> dict | None:
    """Return the fastest successful result from the already-sorted leaderboard list."""
    return measured[0] if measured else None


def _print_benchmark_usage_help(measured: list[dict]):
    best = _best_benchmark_result(measured)
    if not best:
        return

    best_key = best.get("backend_key", "")
    mismatches = [
        r for r in measured
        if r.get("tier_assigned") and r.get("tier_measured") != r.get("tier_assigned")
    ]

    print()
    print("Use this result")
    print("  Auto-routing: cached TG speed ranks backends inside the same configured tier.")
    print("  Direct use: put a backend key in your OpenAI-compatible app's model field.")
    print(f"  Fastest measured key: {best_key}")
    print("  One-message override:")
    print(f"    [route:{best_key}] Say hello in one sentence. /no_think")
    if mismatches:
        print("  Tier mismatch to review:")
        for result in mismatches[:3]:
            print(
                f"    {result.get('backend_key', 'unknown')}: "
                f"configured={result.get('tier_assigned')}  "
                f"measured={result.get('tier_measured')}"
            )
        print("    Auto-routing still uses the configured tier. Edit config/backends.yaml if you want to move it.")


def _print_benchmark_leaderboard(results_by_key: dict):
    """Print cached active-benchmark results as a speed tier list."""
    results = sorted(results_by_key.values(), key=_benchmark_result_sort_key)
    measured = [r for r in results if r.get("validated") and not r.get("error")]
    failures = [r for r in results if r.get("error")]

    if not measured and not failures:
        print("No cached benchmark results yet.")
        print("Benchmark one model:")
        print("  ./router-start bench --backend <backend-key> --start-stopped")
        return

    print("Benchmark results (cached)")
    print("Sorted by measured tier, then token-generation speed.")
    print()

    current_tier = None
    for result in measured:
        tier = result.get("tier_measured") or "unmeasured"
        if tier != current_tier:
            current_tier = tier
            print(f"{tier.upper()}")
        key = result.get("backend_key", "unknown")
        print(f"  {key}")
        print(
            f"    TG={_speed_label(result.get('tg_tok_s'), 'tok/s')}  "
            f"PP={_speed_label(result.get('pp_tok_s'), 'tok/s')}  "
            f"TTFT={_speed_label(result.get('ttft_ms'), 'ms')}  "
            f"think={result.get('thinking_mode', '—')}  "
            f"engine={result.get('engine', '—')}"
            f"{'  configured=' + result['tier_assigned'] if result.get('tier_assigned') else ''}"
        )
        print(f"    {result.get('description', '')}")

    _print_benchmark_usage_help(measured)

    if failures:
        print()
        print("FAILED / INCOMPLETE")
        for result in failures:
            print(f"  {result.get('backend_key', 'unknown')}: {result.get('error')}")


def _fetch_benchmark_results() -> dict:
    import urllib.request
    with urllib.request.urlopen(f"{_router_url()}/benchmarks", timeout=5) as r:
        return json.loads(r.read())


def _bench_thinking_mode(args) -> str:
    if getattr(args, "thinking", False):
        return "think"
    if getattr(args, "default_thinking", False):
        return "default"
    return "no_think"


def _running_backend_keys(status: dict) -> set[str]:
    return {key for key, info in status.items() if info.get("running")}


def _fetch_router_status() -> dict:
    import urllib.request
    with urllib.request.urlopen(f"{_router_url()}/status", timeout=5) as r:
        return json.loads(r.read())


def _post_router(path: str, timeout: int = 30):
    import urllib.request
    req = urllib.request.Request(
        f"{_router_url()}{path}",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=b"{}",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _router_exception_text(exc: Exception) -> str:
    """Return a useful one-line error, including FastAPI JSON detail/log fields when present."""
    body = None
    if hasattr(exc, "read"):
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = None

    if body:
        try:
            payload = json.loads(body)
        except Exception:
            return f"{exc}: {body[:300]}"

        detail = payload.get("detail")
        error = payload.get("error")
        log = payload.get("log")
        parts = [str(value) for value in [detail, error] if value]
        if log:
            parts.append(f"log: {log}")
        if parts:
            return " | ".join(parts)

    return str(exc)


def _extract_log_path(text: str) -> str | None:
    match = re.search(r"(?:log:\s*|Check\s+)(\S+\.log)", text)
    return match.group(1) if match else None


def _read_tail(path: str, max_lines: int = 80) -> str:
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-max_lines:])


def _diagnose_backend_failure(text: str, engine: str = "") -> list[str]:
    """Classify common backend startup failures from the router error plus backend log tail."""
    lower = text.lower()
    hints = []
    if re.search(r"out of memory|cuda.*oom|oom|cumemalloc|memory allocation", lower):
        hints.append("Likely GPU memory/OOM. Do not bulk-retry; stop backends, check nvidia-smi, and power-restart the Spark if GPU memory is wedged.")
    if "address already in use" in lower or "port is already in use" in lower or "errno 98" in lower:
        hints.append("Port conflict. Stop the process using this backend port or change the port in config/backends.yaml.")
    if "no module named" in lower or "modulenotfounderror" in lower:
        hints.append("Missing Python dependency for this engine. Run ./router-start sysinfo and install the engine shown as missing.")
    if "trust_remote_code" in lower or "trust-remote-code" in lower:
        hints.append("Model requires remote custom code. Add trust_remote_code: true to that backend, then rescan.")
    if "unsupported" in lower or "not supported" in lower or "unrecognized configuration" in lower:
        hints.append("Engine/model compatibility failure. Try a llama.cpp GGUF backend first, or install a vLLM version that supports this model/quantization.")
    if "no such file" in lower or "does not exist" in lower or "not found" in lower:
        hints.append("Path/config problem. Confirm the model path exists or remove stale scan_dirs entries.")
    if engine == "trt-llm-docker":
        hints.append("Managed Docker TRT-LLM needs a working Docker daemon plus access to the configured TensorRT-LLM image. Test with 'docker ps' and check Docker permissions/image pull errors in the backend log.")
    if "failed to start" in lower and not hints:
        hints.append("Backend did not become healthy. Inspect the backend log shown below for the exact engine error.")
    return hints[:4]


def _print_backend_failure_help(
    key: str,
    error_text: str,
    log_path: str | None = None,
    engine: str = "",
):
    log_path = log_path or _extract_log_path(error_text)
    log_tail = _read_tail(log_path) if log_path else ""
    hints = _diagnose_backend_failure(error_text + "\n" + log_tail, engine=engine)

    if hints:
        print("      diagnosis:")
        for hint in hints:
            print(f"        - {hint}")
    if log_path:
        print(f"      log: {log_path}")
        print(f"      inspect: tail -n 120 {log_path}")
    if log_tail and not hints:
        last_lines = [line for line in log_tail.splitlines() if line.strip()][-8:]
        if last_lines:
            print("      backend log tail:")
            for line in last_lines:
                print(f"        {_shorten(line, 120)}")


def cmd_bench(args):
    """
    Run PP and TG speed benchmarks against registered backends.
    Requires the router to be running (./router-start first).
    Results are cached to ~/.llm-router/benchmarks/ and shown in status.
    """
    import urllib.request

    if getattr(args, "results", False):
        try:
            _print_benchmark_leaderboard(_fetch_benchmark_results())
        except Exception as e:
            print(f"Could not fetch benchmark results: {_router_exception_text(e)}")
            print("Start the router first with: ./router-start service install")
            sys.exit(1)
        return

    # Fetch backend list from router
    try:
        with urllib.request.urlopen(f"{_router_url()}/backends", timeout=5) as r:
            backends = json.loads(r.read())
        status = _fetch_router_status()
        for key, info in status.items():
            if key in backends and info.get("log"):
                backends[key]["log"] = info["log"]
    except Exception as e:
        print(f"Router not reachable: {e}")
        print("Start it first with: ./router-start")
        sys.exit(1)

    running_before = _running_backend_keys(status)
    if getattr(args, "all", False):
        targets = list(backends.keys())
    else:
        targets = [k for k in backends if k in running_before]

    if getattr(args, "backend", None):
        if args.backend not in backends:
            print(f"Unknown backend '{args.backend}'.")
            print("Run ./router-start status to copy a backend key.")
            sys.exit(1)
        targets = [args.backend]

    if getattr(args, "list", False):
        print("Registered benchmarkable backends:")
        _print_bench_plan(backends, list(backends.keys()))
        print("Benchmark exactly one:")
        print("  ./router-start bench --backend <backend-key> --start-stopped")
        return

    stopped_targets = [k for k in targets if k not in running_before]
    if stopped_targets and not getattr(args, "start_stopped", False):
        print("Benchmark did not start a stopped backend.")
        print("DGX Spark safety: loading the wrong large model can exhaust memory and require a hardware restart.")
        print(f"Requested stopped backend(s): {', '.join(stopped_targets)}")
        print()
        print(f"Safe one-model command:")
        print(f"  ./router-start bench --backend {stopped_targets[0]} --start-stopped")
        print()
        print("If you intentionally want to benchmark every backend one by one:")
        print("  ./router-start bench --all --start-stopped")
        return

    if not targets and not getattr(args, "backend", None):
        _print_no_running_bench_help(backends)
        return

    # Skip external servers (LM Studio, Ollama managed externally)
    skippable = {"openai", "ollama"}
    runnable  = [k for k in targets if backends[k].get("engine") not in skippable]
    skipped   = [k for k in targets if backends[k].get("engine") in skippable]

    if skipped:
        print(f"Skipping external servers (not managed by router): {', '.join(skipped)}")

    if not runnable:
        print("No benchmarkable backends found.")
        return

    print(f"Benchmarking {len(runnable)} router backend(s) — this takes ~30s per backend.")
    if getattr(args, "start_stopped", False):
        print("Stopped targets may be started one at a time. Anything bench starts is stopped after measurement.")
    print("This tests the local backend servers directly; it is not testing Open WebUI, Cursor, or another client app.\n")
    thinking_mode = _bench_thinking_mode(args)
    print(f"Thinking mode for this benchmark: {thinking_mode}")
    if thinking_mode == "no_think":
        print("  Adds /no_think to the benchmark prompt so speed reflects direct-answer mode.")
    elif thinking_mode == "think":
        print("  Adds /think to the benchmark prompt. Expect lower tok/s on reasoning models.")
    else:
        print("  Sends the benchmark prompt without a /think or /no_think directive.")
    print()
    _print_bench_plan(backends, runnable)

    import asyncio

    async def _run_all():
        # Dynamic import inside async context to avoid circular imports
        sys.path.insert(0, str(PROJECT_DIR))
        from router.benchmark import measure_backend, save_result, format_results
        from router.config import load_config
        config = load_config()

        results = []
        saved_count = 0
        for index, key in enumerate(runnable, start=1):
            cfg = backends[key]
            engine = cfg.get("engine", "unknown")
            tier = cfg.get("tier", "unknown")
            port = cfg.get("port", "unknown")
            print(f"[{index}/{len(runnable)}] {key}")
            print(f"      engine={engine}  tier={tier}  port={port}")
            print(f"      model={_bench_model_label(cfg)}")
            started_by_bench = key not in running_before
            attempted_start = False
            start_ok = False

            # Ensure backend is running first
            try:
                if started_by_bench:
                    print("      start...", end=" ", flush=True)
                    attempted_start = True
                    _post_router(f"/start/{key}", timeout=120).read()
                    start_ok = True
                    print("up", end="  ", flush=True)
                else:
                    start_ok = True
                    print("      start... already running  ", end="", flush=True)

                print("benchmark...", end=" ", flush=True)
                r = await measure_backend(key, cfg, config, thinking_mode=thinking_mode)
                save_result(r, config)
                saved_count += 1
                results.append(r)

                if r.get("error"):
                    print(f"ERROR: {r['error']}")
                elif r.get("tier_mismatch"):
                    print(f"done  ⚠  TG={r['tg_tok_s']:.0f} tok/s  (tier mismatch)")
                else:
                    print(f"done  TG={r['tg_tok_s']:.0f} tok/s  PP={r['pp_tok_s']:.0f} tok/s")
            except Exception as e:
                error_text = _router_exception_text(e)
                print(f"failed: {error_text}")
                _print_backend_failure_help(
                    key,
                    error_text,
                    cfg.get("log"),
                    engine=engine,
                )
                results.append({"backend_key": key, "thinking_mode": thinking_mode, "error": error_text})
            finally:
                if started_by_bench and attempted_start and not getattr(args, "keep_running", False):
                    action = "stop" if start_ok else "cleanup"
                    print(f"      {action}...", end=" ", flush=True)
                    try:
                        _post_router(f"/stop/{key}", timeout=30).read()
                        if start_ok:
                            print("stopped")
                        else:
                            print("cleanup requested")
                    except Exception as e:
                        print(f"{action} failed: {_router_exception_text(e)}")

        print()
        print(format_results(results))

        if saved_count:
            # The router reads benchmark cache at startup and during rescan. Refresh
            # after saving so speed-informed routing is active immediately.
            try:
                _post_router("/rescan", timeout=30).read()
                print(f"\nBenchmark cache saved ({saved_count}) and router routing data refreshed.")
            except Exception as e:
                print(f"\nBenchmark cache saved ({saved_count}). Run './router-start rescan' before relying on it for routing ({_router_exception_text(e)}).")
        else:
            print("\nNo benchmark result was saved. Fix the start/health error above, then rerun the one-backend bench command.")

        # Suggest fixes for mismatches
        mismatches = [r for r in results if r.get("tier_mismatch")]
        if mismatches:
            print("\nTo fix tier assignments, edit config/backends.yaml and set tier: explicitly.")
            for r in mismatches:
                print(f"  {r['backend_key']}: assigned={r['tier_assigned']}  "
                      f"measured speed suggests tier={r['tier_measured']}")

    asyncio.run(_run_all())


def cmd_logs(args):
    if not ROUTER_LOG.exists():
        print(f"Log file not found: {ROUTER_LOG}")
        sys.exit(1)
    try:
        subprocess.run(["tail", "-f", str(ROUTER_LOG)])
    except KeyboardInterrupt:
        pass


def cmd_sysinfo(args):
    """Show hardware info, engine versions, and install recommendations."""
    # Try live router first; fall back to running detection directly
    data = None
    try:
        import urllib.request
        with urllib.request.urlopen(f"{_router_url()}/sysinfo", timeout=3) as r:
            data = json.loads(r.read())
    except Exception:
        # Router not running — detect directly via venv Python
        try:
            result = subprocess.run(
                [_local_python(), "-c",
                 "import sys; sys.path.insert(0, '.'); "
                 "from router.sysinfo import detect_system; "
                 "import json; print(json.dumps(detect_system()))"],
                capture_output=True, text=True, cwd=str(PROJECT_DIR),
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
        except Exception as e:
            print(f"Could not detect system info: {e}")
            sys.exit(1)

    if not data:
        print("No data available.")
        sys.exit(1)

    w = 60

    # ── Platform ──────────────────────────────────────────────
    plat = data.get("platform", {})
    cpu  = data.get("cpu", {})
    ram  = data.get("ram", {})
    _section("System Info")
    print(f"  OS      : {plat.get('os', '?')} {plat.get('arch', '')}  ({plat.get('os_version', '')})")
    cpu_label = cpu.get('model') or cpu.get('arch', '?')
    print(f"  CPU     : {cpu_label}  ({cpu.get('cores', '?')} cores)")
    ram_gb = ram.get('total_gb')
    print(f"  RAM     : {f'{ram_gb} GB' if ram_gb else 'unknown'}")
    print(f"  Python  : {plat.get('python', '?')}")

    # ── GPU ───────────────────────────────────────────────────
    gpu  = data.get("gpu", {})
    cuda = data.get("cuda", {})
    _section("GPU")
    if gpu.get("available"):
        for i, dev in enumerate(gpu.get("devices", [])):
            print(f"  GPU {i}   : {dev['name']}")
            total = dev.get('vram_total_gb')
            free  = dev.get('vram_free_gb')
            if total is not None:
                print(f"  VRAM    : {total} GB total  |  {free} GB free")
        print(f"  Driver  : {gpu.get('driver_version', 'unknown')}")
        print(f"  CUDA    : {cuda.get('version', 'unknown') if cuda.get('available') else 'not detected'}")
    else:
        print("  No NVIDIA GPU detected  (CPU-only mode)")

    # ── Local LLMs ────────────────────────────────────────────
    _section("Local LLMs (registered backends)")
    backends_data = None
    try:
        import urllib.request
        with urllib.request.urlopen(f"{_router_url()}/backends", timeout=3) as r:
            backends_data = json.loads(r.read())
    except Exception:
        pass

    if backends_data:
        shown = 0
        for key, info in backends_data.items():
            size = f"{info['size_gb']:.1f} GB" if info.get("size_gb") else ""
            tag  = "[auto]" if info.get("auto_discovered") else "[manual]"
            desc = info.get("description", "")[:40]
            print(f"  {key:<16} {size:<10} port {info['port']}  {tag}")
            shown += 1
            if not args.all and shown >= 5 and len(backends_data) > 5:
                print(f"  ... {len(backends_data) - shown} more — run: python cli.py sysinfo --all")
                break
    else:
        print("  (router not running — start with: python cli.py start)")

    # ── Engine versions & recommendations ─────────────────────
    _section("Engine Versions & Recommendations")
    versions = data.get("engine_versions", {})
    recs     = data.get("recommendations", {})

    engines_display = [
        ("llama.cpp",   "llama.cpp"),
        ("vLLM",        "vllm"),
        ("SGLang",      "sglang"),
        ("TRT-LLM",     "trt-llm"),
        ("HuggingFace", "huggingface"),
    ]
    for label, key in engines_display:
        installed = versions.get(key)
        rec       = recs.get(key, {})
        rec_ver   = rec.get("recommended", "?")
        compatible = rec.get("compatible", True)
        reason     = rec.get("reason", "")
        install    = rec.get("install_cmd", "")

        if installed:
            compat_str = "✓ compatible" if compatible else f"✗ {reason}"
            print(f"  {label:<14} installed: {installed:<12} recommended: {rec_ver:<10} {compat_str}")
        else:
            compat_str = "✗ incompatible" if not compatible else ""
            install_short = install.splitlines()[0][:50] if install else ""
            print(f"  {label:<14} NOT installed        recommended: {rec_ver:<10} {compat_str}")
            if install_short and compatible:
                print(f"  {'':14}   install: {install_short}")

    # ── Client connection snippets ────────────────────────────
    base = _router_url()
    _section("Quick Connect")
    print(f"  OpenAI SDK / Open WebUI / Jan:")
    print(f"    base_url = {base}/v1")
    print(f"    api_key  = anything")
    print()
    print(f"  OpenClaw (openclaw.json):")
    print(f'    "baseUrl": "{base}/v1",')
    print(f'    "api": "openai-completions"')
    print()
    print(f"  Anthropic SDK / Claude Code:")
    print(f"    base_url = {base}/anthropic")
    print(f"    api_key  = anything")

    # ── Port conflicts ────────────────────────────────────────
    _section("Port Conflicts")
    conflicts = data.get("conflict_processes", [])
    if conflicts:
        for c in conflicts:
            cmd_short = c.get("cmdline", "")[:55]
            print(f"  Port {c['port']}  PID {c['pid']}  {cmd_short}")
        print()
        print("  Tip: stop conflicting processes before starting backends,")
        print("       or change ports in config/backends.yaml")
    else:
        print("  No conflicts detected.")


def cmd_service(args):
    """Install / uninstall the router as a system service (auto-start on boot)."""
    system = platform.system()
    if system == "Darwin":
        _service_macos(args.action)
    elif system == "Linux":
        _service_linux(args.action)
    else:
        print(f"Service management is not supported on {system}.")
        print("Start the router manually with: ./router-start")
        sys.exit(1)


def _print_service_next_steps():
    """Show the shortest useful post-install path for people who skip the README."""
    print("  Next steps:")
    print("    ./router-start status   # confirm backends/models are visible")
    print("    ./router-start bench --backend <key> --start-stopped")
    print("                         # benchmark one model safely; get <key> from status")


def _require_service_health(log_hint: str):
    ready, error = _wait_router_ready()
    if ready:
        return
    print("Service was registered, but router HTTP is not reachable yet.")
    print(f"  Last health-check error: {error}")
    print(f"  Logs: {log_hint}")
    print("  Fix the logged error, then run: ./router-start service install")
    sys.exit(1)


def _require_launchd_running(log_hint: str):
    result = subprocess.run(["launchctl", "list", _LAUNCHD_LABEL], capture_output=True)
    if result.returncode == 0:
        return
    print("Service was registered, but launchd is not keeping it running.")
    print(f"  Logs: {log_hint}")
    sys.exit(1)


# ── macOS — launchd ───────────────────────────────────────────

_LAUNCHD_LABEL = "com.llm-router"
_LAUNCHD_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{_LAUNCHD_LABEL}.plist"
_ROUTER_START  = PROJECT_DIR / "router-start"


def _service_macos(action: str):
    port    = _router_port()
    log_out = str(LOG_DIR / "service-stdout.log")
    log_err = str(LOG_DIR / "service-stderr.log")

    if action == "install":
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ensure_venv()

        plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>             <string>{_LAUNCHD_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{_ROUTER_START}</string>
    <string>serve</string>
  </array>
  <key>WorkingDirectory</key>  <string>{PROJECT_DIR}</string>
  <key>RunAtLoad</key>         <true/>
  <key>KeepAlive</key>         <true/>
  <key>StandardOutPath</key>   <string>{log_out}</string>
  <key>StandardErrorPath</key> <string>{log_err}</string>
</dict>
</plist>
"""
        _LAUNCHD_PLIST.parent.mkdir(parents=True, exist_ok=True)
        _LAUNCHD_PLIST.write_text(plist)

        # Unload first in case an old version is registered
        subprocess.run(["launchctl", "unload", str(_LAUNCHD_PLIST)],
                       capture_output=True)
        _stop_saved_router_pid()
        _kill_port(port)
        result = subprocess.run(["launchctl", "load", "-w", str(_LAUNCHD_PLIST)],
                                capture_output=True, text=True)
        if result.returncode == 0:
            _require_service_health(log_out)
            _require_launchd_running(log_out)
            lan = _lan_ip()
            print(f"Service installed and started.")
            print(f"  Auto-starts at every login.")
            print(f"  On this machine  : http://localhost:{port}/v1")
            print(f"  From another PC  : http://{lan}:{port}/v1")
            print(f"  Logs: {log_out}")
            _print_service_next_steps()
            print(f"  To remove: ./router-start service uninstall")
        else:
            print(f"launchctl load failed: {result.stderr.strip()}")
            sys.exit(1)

    elif action == "uninstall":
        if not _LAUNCHD_PLIST.exists():
            print("Service is not installed.")
            return
        subprocess.run(["launchctl", "unload", "-w", str(_LAUNCHD_PLIST)],
                       capture_output=True)
        _LAUNCHD_PLIST.unlink(missing_ok=True)
        print("Service uninstalled. Router will no longer auto-start.")

    elif action == "status":
        if not _LAUNCHD_PLIST.exists():
            print("Service: not installed")
            return
        result = subprocess.run(
            ["launchctl", "list", _LAUNCHD_LABEL],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("Service: installed and running")
            print(result.stdout.strip())
        else:
            print("Service: installed but not running")
            print(f"  Start with: launchctl load -w {_LAUNCHD_PLIST}")


# ── Linux — systemd user service ─────────────────────────────

_SYSTEMD_UNIT = "llm-router.service"
_SYSTEMD_DIR  = Path.home() / ".config" / "systemd" / "user"
_SYSTEMD_FILE = _SYSTEMD_DIR / _SYSTEMD_UNIT


def _service_linux(action: str):
    port = _router_port()

    if action == "install":
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ensure_venv()

        unit = f"""[Unit]
Description=LLM Router — local LLM proxy
After=network.target

[Service]
Type=simple
WorkingDirectory={PROJECT_DIR}
ExecStart={_ROUTER_START} serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""
        _SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
        _SYSTEMD_FILE.write_text(unit)

        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "stop", _SYSTEMD_UNIT], capture_output=True)
        _stop_saved_router_pid()
        _kill_port(port)
        subprocess.run(["systemctl", "--user", "enable", "--now", _SYSTEMD_UNIT], check=True)
        log_hint = f"journalctl --user -u {_SYSTEMD_UNIT} -f"
        _require_service_health(log_hint)
        subprocess.run(["systemctl", "--user", "is-active", "--quiet", _SYSTEMD_UNIT], check=True)
        lan = _lan_ip()
        print(f"Service installed and started.")
        print(f"  Auto-starts at every login.")
        print(f"  On this machine  : http://localhost:{port}/v1")
        print(f"  From another PC  : http://{lan}:{port}/v1")
        print(f"  Logs: {log_hint}")
        _warn_firewall(port, lan)
        _print_service_next_steps()
        print(f"  To remove: ./router-start service uninstall")

    elif action == "uninstall":
        if not _SYSTEMD_FILE.exists():
            print("Service is not installed.")
            return
        subprocess.run(["systemctl", "--user", "disable", "--now", _SYSTEMD_UNIT],
                       capture_output=True)
        _SYSTEMD_FILE.unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        print("Service uninstalled. Router will no longer auto-start.")

    elif action == "status":
        result = subprocess.run(
            ["systemctl", "--user", "status", _SYSTEMD_UNIT],
            capture_output=True, text=True,
        )
        print(result.stdout.strip() or result.stderr.strip())


def _section(title: str):
    print()
    print(f"── {title} " + "─" * max(0, 50 - len(title)))


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    _ensure_supported_python()

    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description="LLM Router — local LLM proxy for lazy people",
    )
    sub = parser.add_subparsers(dest="command")

    # start
    p_start = sub.add_parser("start", help="Start the router")
    p_start.add_argument("--update", action="store_true", help="Update llama.cpp before starting")
    p_start.set_defaults(func=cmd_start)

    # stop
    p_stop = sub.add_parser("stop", help="Stop the router")
    p_stop.set_defaults(func=cmd_stop)

    # status
    p_status = sub.add_parser("status", help="Show backend run-state")
    p_status.set_defaults(func=cmd_status)

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Show performance metrics")
    p_bench.add_argument("--export", metavar="FILE", help="Export metrics to CSV file")
    p_bench.set_defaults(func=cmd_benchmark)

    # update
    p_update = sub.add_parser("update", help="Update llama.cpp and Python deps")
    p_update.add_argument("--restart", action="store_true", help="Restart router after update")
    p_update.set_defaults(func=cmd_update)

    # rescan
    p_rescan = sub.add_parser("rescan", help="Re-scan for new models (router must be running)")
    p_rescan.set_defaults(func=cmd_rescan)

    # logs
    p_logs = sub.add_parser("logs", help="Tail the router log")
    p_logs.set_defaults(func=cmd_logs)

    # serve
    p_serve = sub.add_parser("serve", help=argparse.SUPPRESS)
    p_serve.set_defaults(func=cmd_serve)

    # bench
    p_bench2 = sub.add_parser("bench", help="Benchmark PP and TG speed for each backend")
    p_bench2.add_argument("--backend", metavar="KEY", help="Benchmark a single backend only")
    p_bench2.add_argument("--start-stopped", action="store_true",
                          help="Allow bench to start stopped backend(s); stopped again by default")
    p_bench2.add_argument("--keep-running", action="store_true",
                          help="Leave backend(s) running after bench started them")
    p_bench2.add_argument("--all", action="store_true",
                          help="Benchmark all router-managed backends; combine with --start-stopped for stopped models")
    p_bench2.add_argument("--list", action="store_true",
                          help="List backend keys that can be passed to --backend")
    p_bench2.add_argument("--results", action="store_true",
                          help="Show cached benchmark tier list without starting models")
    thinking_group = p_bench2.add_mutually_exclusive_group()
    thinking_group.add_argument("--thinking", action="store_true",
                                help="Add /think to benchmark prompts and measure reasoning-mode speed")
    thinking_group.add_argument("--default-thinking", action="store_true",
                                help="Do not add /think or /no_think to benchmark prompts")
    p_bench2.set_defaults(func=cmd_bench)

    # sysinfo
    p_sys = sub.add_parser("sysinfo", help="Show hardware, engine versions, and install recommendations")
    p_sys.add_argument("--all", action="store_true", help="Show all discovered backends (not just first 5)")
    p_sys.set_defaults(func=cmd_sysinfo)

    # service
    p_svc = sub.add_parser("service", help="Install or remove the router as a system service (auto-start on boot)")
    p_svc.add_argument("action", choices=["install", "uninstall", "status"],
                       help="install: register service  |  uninstall: remove it  |  status: show state")
    p_svc.set_defaults(func=cmd_service)

    args = parser.parse_args()

    # Default: no subcommand → start
    if args.command is None:
        cmd_start(argparse.Namespace(update=False))
        return

    args.func(args)


if __name__ == "__main__":
    main()
