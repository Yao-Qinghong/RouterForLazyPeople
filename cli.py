#!/usr/bin/env python3
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
import shutil
import subprocess
import sys
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


# ─────────────────────────────────────────────────────────────
# Virtual environment helpers
# ─────────────────────────────────────────────────────────────

def ensure_venv():
    """Create venv and install requirements if not already set up."""
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

def _llama_dir() -> Path | None:
    """Try to read llama_bin from config and derive the repo dir."""
    config_path = PROJECT_DIR / "config" / "settings.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
            llama_bin = Path(os.path.expanduser(raw.get("llama_bin", "~/llama.cpp/build/bin/llama-server")))
            # Walk up to find the repo root (has a CMakeLists.txt)
            for parent in llama_bin.parents:
                if (parent / "CMakeLists.txt").exists():
                    return parent
        except Exception:
            pass
    default = Path.home() / "llama.cpp"
    return default if default.exists() else None


def update_llama():
    """Pull and rebuild llama.cpp if there are new commits upstream."""
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

    print("Pulling latest changes...")
    subprocess.run(["git", "pull", "--quiet"], cwd=llama_dir, check=True)

    nproc = os.cpu_count() or 4
    print(f"Rebuilding with CUDA (using {nproc} cores — this may take a few minutes)...")
    subprocess.run(
        ["cmake", "-B", "build", "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=native"],
        cwd=llama_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["cmake", "--build", "build", "--config", "Release", f"-j{nproc}"],
        cwd=llama_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

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


# ─────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────

def cmd_start(args):
    if args.update:
        update_llama()

    ensure_venv()

    port = _router_port()
    _kill_port(port)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting LLM Router on port {port}...")
    log_handle = open(ROUTER_LOG, "a")

    proc = subprocess.Popen(
        [
            str(VENV_UVICORN),
            "router.main:create_app",
            "--factory",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--app-dir", str(PROJECT_DIR),
        ],
        stdout=log_handle,
        stderr=log_handle,
    )

    _save_pid(proc.pid)
    print(f"Started with PID {proc.pid}")
    print()
    print(f"  Status:    curl {_router_url()}/status")
    print(f"  Backends:  curl {_router_url()}/backends")
    print(f"  Metrics:   curl {_router_url()}/metrics")
    print(f"  Logs:      python cli.py logs")
    print(f"  Stop:      python cli.py stop")


def cmd_stop(args):
    port = _router_port()
    pid = _read_pid()
    if pid:
        try:
            subprocess.run(["kill", str(pid)], check=True)
            print(f"Stopped router (PID {pid})")
            PID_FILE.unlink(missing_ok=True)
        except subprocess.CalledProcessError:
            print(f"PID {pid} not found — trying port kill...")
    _kill_port(port)


def cmd_status(args):
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(f"{_router_url()}/status", timeout=5) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"Router not reachable: {e}")
        sys.exit(1)

    print(f"{'Backend':<20} {'Engine':<12} {'Running':<8} {'PID':<8} {'Idle (s)':<10} Description")
    print("─" * 80)
    for key, info in data.items():
        running = "yes" if info["running"] else "no"
        pid     = str(info.get("pid") or "")
        idle    = str(info.get("idle_seconds") or "")
        desc    = info.get("description", "")[:35]
        engine  = info.get("engine", "")
        print(f"{key:<20} {engine:<12} {running:<8} {pid:<8} {idle:<10} {desc}")


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
        cmd_start(args)


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


def cmd_logs(args):
    if not ROUTER_LOG.exists():
        print(f"Log file not found: {ROUTER_LOG}")
        sys.exit(1)
    try:
        subprocess.run(["tail", "-f", str(ROUTER_LOG)])
    except KeyboardInterrupt:
        pass


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description="LLM Router — local LLM proxy for lazy people",
    )
    sub = parser.add_subparsers(dest="command", required=True)

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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
