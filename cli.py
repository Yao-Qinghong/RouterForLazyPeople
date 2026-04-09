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
import shutil
import socket
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
    build_mode, configure_cmd = _llama_build_config()
    print(f"Rebuilding for {build_mode} (using {nproc} cores — this may take a few minutes)...")
    subprocess.run(
        configure_cmd,
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
    if getattr(args, "update", False):
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
        result = subprocess.run(["launchctl", "load", "-w", str(_LAUNCHD_PLIST)],
                                capture_output=True, text=True)
        if result.returncode == 0:
            lan = _lan_ip()
            print(f"Service installed and started.")
            print(f"  Auto-starts at every login.")
            print(f"  On this machine  : http://localhost:{port}/v1")
            print(f"  From another PC  : http://{lan}:{port}/v1")
            print(f"  Logs: {log_out}")
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
ExecStart={_ROUTER_START}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""
        _SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
        _SYSTEMD_FILE.write_text(unit)

        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "--now", _SYSTEMD_UNIT], check=True)
        lan = _lan_ip()
        print(f"Service installed and started.")
        print(f"  Auto-starts at every login.")
        print(f"  On this machine  : http://localhost:{port}/v1")
        print(f"  From another PC  : http://{lan}:{port}/v1")
        print(f"  Logs: journalctl --user -u {_SYSTEMD_UNIT} -f")
        _warn_firewall(port, lan)
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
