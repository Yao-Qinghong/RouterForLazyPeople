"""
router/sysinfo.py — System detection and diagnostics

Detects hardware (GPU, CUDA, CPU, RAM), installed engine versions,
and checks compatibility with recommended stable versions.

No new pip dependencies — uses only stdlib + CLI tools that are
already present on a standard Linux/macOS machine with CUDA.

All functions degrade gracefully: missing tools → None, never raises.
"""

import os
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Compatibility table
# Update this when new stable versions ship or requirements change.
# ─────────────────────────────────────────────────────────────
COMPATIBILITY = {
    "llama.cpp": {
        "min_cuda": None,          # also works CPU-only
        "recommended": "b4500+",
        "install_cmd": (
            "git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp\n"
            "cd ~/llama.cpp\n"
            "cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native\n"
            "cmake --build build --config Release -j$(nproc)"
        ),
        "notes": "GPU acceleration requires CUDA 11.8+. CPU-only works anywhere.",
    },
    "vllm": {
        "min_cuda": "12.1",
        "recommended": "0.6.4",
        "install_cmd": "pip install vllm==0.6.4",
        "notes": "Requires Ampere GPU (sm_80+) or newer. CUDA 12.1+ required.",
    },
    "sglang": {
        "min_cuda": "12.1",
        "recommended": "0.4.1",
        "install_cmd": "pip install sglang[all]==0.4.1",
        "notes": "Requires CUDA 12.1+. Works with Turing (sm_75) and newer.",
    },
    "trt-llm": {
        "min_cuda": "12.2",
        "recommended": "0.14.0",
        "install_cmd": "pip install tensorrt-llm==0.14.0",
        "notes": "Requires CUDA 12.2+. Best on Ampere/Hopper (A100/H100/DGX).",
    },
    "huggingface": {
        "min_cuda": None,
        "recommended": "4.45.0",
        "install_cmd": "pip install transformers>=4.45.0 accelerate",
        "notes": "CPU-compatible. GPU via CUDA optional.",
    },
}

# Ports commonly used by local LLM tools — scanned for conflicts
COMMON_LLM_PORTS = list(range(8080, 8100)) + [11434, 1234, 5000, 8000, 8888]


# ─────────────────────────────────────────────────────────────
# Master detector
# ─────────────────────────────────────────────────────────────

def detect_system(llama_bin: Optional[Path] = None) -> dict:
    """
    Run all detectors and return a single dict.
    Safe to call at any time — never raises.
    llama_bin: path to llama-server binary (for version detection).
    """
    gpu_info  = _detect_gpu()
    cuda_info = _detect_cuda()
    cuda_ver  = cuda_info.get("version")

    return {
        "platform":        _detect_platform(),
        "cpu":             _detect_cpu(),
        "ram":             _detect_ram(),
        "gpu":             gpu_info,
        "cuda":            cuda_info,
        "engine_versions": _detect_engine_versions(llama_bin),
        "recommendations": _build_recommendations(cuda_ver),
        "conflict_processes": detect_existing_llm_processes(),
    }


# ─────────────────────────────────────────────────────────────
# Sub-detectors
# ─────────────────────────────────────────────────────────────

def _detect_platform() -> dict:
    return {
        "os":         platform.system(),                  # "Linux" | "Darwin" | "Windows"
        "arch":       platform.machine(),                 # "x86_64" | "arm64" | "aarch64"
        "os_version": platform.release(),
        "python":     platform.python_version(),
    }


def _detect_cpu() -> dict:
    arch  = platform.machine()
    cores = os.cpu_count()
    model = _cpu_model()
    return {"model": model, "arch": arch, "cores": cores}


def _cpu_model() -> Optional[str]:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            if out:
                return out.strip()
    except Exception:
        pass
    return None


def _detect_ram() -> dict:
    total_gb = None
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        total_gb = round(kb / (1024 ** 2), 1)
                        break
        elif system == "Darwin":
            out = _run(["sysctl", "-n", "hw.memsize"])
            if out:
                total_gb = round(int(out.strip()) / (1024 ** 3), 1)
    except Exception:
        pass
    return {"total_gb": total_gb}


def _detect_gpu() -> dict:
    """Query nvidia-smi for GPU count, names, VRAM, and driver version."""
    if not shutil.which("nvidia-smi"):
        return {"available": False, "driver_version": None, "devices": []}

    try:
        out = _run([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        if not out:
            return {"available": False, "driver_version": None, "devices": []}

        devices = []
        driver_version = None
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            name, drv, mem_total, mem_free = parts[0], parts[1], parts[2], parts[3]
            driver_version = drv
            try:
                devices.append({
                    "name":          name,
                    "vram_total_gb": round(int(mem_total) / 1024, 1),
                    "vram_free_gb":  round(int(mem_free)  / 1024, 1),
                })
            except ValueError:
                devices.append({"name": name, "vram_total_gb": None, "vram_free_gb": None})

        return {
            "available":      True,
            "driver_version": driver_version,
            "devices":        devices,
        }
    except Exception:
        return {"available": False, "driver_version": None, "devices": []}


def query_free_vram():
    """
    Query current GPU VRAM: returns (free_gb, total_gb) summed across devices.
    Returns None if no GPU or nvidia-smi unavailable. ~50ms per call.
    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = _run([
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ])
        if not out:
            return None
        free_total = 0.0
        total_total = 0.0
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            free_total += int(parts[0]) / 1024
            total_total += int(parts[1]) / 1024
        return (round(free_total, 1), round(total_total, 1))
    except Exception:
        return None


def _detect_cuda() -> dict:
    """Detect CUDA version from nvcc, nvidia-smi, or version files."""
    version = None

    # 1. nvcc --version
    if shutil.which("nvcc"):
        out = _run(["nvcc", "--version"])
        if out:
            m = re.search(r"release\s+(\d+\.\d+)", out)
            if m:
                version = m.group(1)

    # 2. nvidia-smi last line (e.g. "CUDA Version: 12.4")
    if version is None and shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi"])
        if out:
            m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
            if m:
                version = m.group(1)

    # 3. /usr/local/cuda/version.txt
    if version is None:
        for path in ["/usr/local/cuda/version.txt", "/usr/local/cuda/version.json"]:
            try:
                text = Path(path).read_text()
                m = re.search(r"(\d+\.\d+)", text)
                if m:
                    version = m.group(1)
                    break
            except Exception:
                pass

    return {"available": version is not None, "version": version}


def _detect_engine_versions(llama_bin: Optional[Path] = None) -> dict:
    """Get installed version strings for each engine. Returns None when not installed."""
    versions = {}

    # llama.cpp — try --version flag, fall back to git describe
    versions["llama.cpp"] = _llama_version(llama_bin)

    # Python-module engines
    for engine, module, attr in [
        ("vllm",        "vllm",           "__version__"),
        ("sglang",      "sglang",         "__version__"),
        ("trt-llm",     "tensorrt_llm",   "__version__"),
        ("huggingface", "transformers",   "__version__"),
    ]:
        versions[engine] = _python_version(module, attr)

    return versions


def _llama_version(llama_bin: Optional[Path]) -> Optional[str]:
    # Try the binary's --version flag
    if llama_bin and Path(llama_bin).exists():
        out = _run([str(llama_bin), "--version"], timeout=5)
        if out:
            # Output is typically "version: b4286 (abc1234)"
            m = re.search(r"b(\d+)", out)
            if m:
                return f"b{m.group(1)}"
            return out.strip().splitlines()[0][:20]

    # Fall back to git describe in the repo directory
    for candidate in [Path.home() / "llama.cpp", Path("/opt/llama.cpp")]:
        if (candidate / ".git").exists():
            out = _run(["git", "describe", "--tags", "--always"], cwd=str(candidate))
            if out:
                return out.strip()

    return None


def _python_version(module: str, attr: str = "__version__") -> Optional[str]:
    try:
        result = subprocess.run(
            ["python3", "-c", f"import {module}; print({module}.{attr})"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────
# Recommendations
# ─────────────────────────────────────────────────────────────

def _build_recommendations(cuda_version: Optional[str]) -> dict:
    """
    For each engine, return recommended version + install command + compatibility flag.
    compatible=False means the detected CUDA is too old for this engine.
    """
    cuda_tuple = _version_tuple(cuda_version) if cuda_version else None
    result = {}

    for engine, info in COMPATIBILITY.items():
        min_cuda = info.get("min_cuda")
        compatible = True
        reason = None

        if min_cuda and cuda_tuple is not None:
            if cuda_tuple < _version_tuple(min_cuda):
                compatible = False
                reason = f"Requires CUDA {min_cuda}+, detected {cuda_version}"
        elif min_cuda and cuda_tuple is None:
            compatible = False
            reason = f"Requires CUDA {min_cuda}+ (CUDA not detected)"

        result[engine] = {
            "recommended":   info.get("recommended"),
            "install_cmd":   info.get("install_cmd"),
            "notes":         info.get("notes"),
            "compatible":    compatible,
            "reason":        reason,
        }

    return result


def _version_tuple(version_str: str) -> tuple:
    """Convert "12.4" → (12, 4) for comparison."""
    try:
        return tuple(int(x) for x in version_str.split(".")[:2])
    except Exception:
        return (0, 0)


# ─────────────────────────────────────────────────────────────
# Port conflict detection
# ─────────────────────────────────────────────────────────────

def detect_existing_llm_processes(ports: Optional[list[int]] = None) -> list[dict]:
    """
    Scan for processes already listening on common LLM ports.
    Uses lsof (macOS/Linux). Returns [] if lsof is unavailable.
    """
    if not shutil.which("lsof"):
        return []

    check_ports = ports or COMMON_LLM_PORTS
    conflicts = []

    for port in check_ports:
        try:
            out = _run(["lsof", "-ti", f":{port}"], timeout=3)
            if not out:
                continue
            for pid_str in out.strip().splitlines():
                pid = pid_str.strip()
                if not pid:
                    continue
                cmdline = _process_cmdline(pid)
                conflicts.append({
                    "port":    port,
                    "pid":     int(pid),
                    "cmdline": cmdline,
                })
        except Exception:
            pass

    return conflicts


def _process_cmdline(pid: str) -> str:
    """Get a short cmdline string for a PID."""
    try:
        if platform.system() == "Linux":
            return Path(f"/proc/{pid}/cmdline").read_text().replace("\x00", " ").strip()[:80]
        else:
            out = _run(["ps", "-p", pid, "-o", "command="], timeout=3)
            return (out or "").strip()[:80]
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────
# LAN IP
# ─────────────────────────────────────────────────────────────

def lan_ip() -> str:
    """Return the machine's LAN IP address. Falls back to 'localhost'."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))   # no traffic sent — just resolves local interface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


# ─────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 10, cwd: Optional[str] = None) -> Optional[str]:
    """Run a subprocess and return stdout as a string, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return result.stdout if result.returncode == 0 else None
    except Exception:
        return None
