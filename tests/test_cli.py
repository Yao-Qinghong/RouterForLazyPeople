"""Tests for cli.py command helpers and regression-prone flows."""

import argparse
import json
import subprocess
import sys
import urllib.error
from types import SimpleNamespace

import cli


class TestLlamaBuildConfig:
    def test_darwin_uses_metal(self, monkeypatch):
        monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(cli, "_cuda_compiler", lambda: None)

        mode, cmd = cli._llama_build_config()

        assert mode == "Metal"
        assert "-DGGML_METAL=ON" in cmd

    def test_linux_with_cuda_uses_cuda_flags(self, monkeypatch):
        monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli, "_cuda_compiler", lambda: "/usr/local/cuda/bin/nvcc")

        mode, cmd = cli._llama_build_config()

        assert mode == "CUDA"
        assert "-DGGML_CUDA=ON" in cmd
        assert "-DCMAKE_CUDA_ARCHITECTURES=native" in cmd

    def test_cpu_only_falls_back_to_plain_cmake(self, monkeypatch):
        monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli, "_cuda_compiler", lambda: None)

        mode, cmd = cli._llama_build_config()

        assert mode == "CPU"
        assert cmd == ["cmake", "-B", "build"]


class TestUpdateAndSysinfo:
    def test_status_print_keeps_long_backend_key_copyable(self, capsys):
        long_key = "hf-nvidia-nvidia-nemotron-3-super-1-b6f6"
        cli._print_status({
            long_key: {
                "tier": "mid",
                "running": False,
                "engine": "vllm",
                "port": 8123,
                "description": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B long description",
                "bench_tg_tok_s": 12.5,
                "bench_pp_tok_s": 222.0,
                "bench_tier_measured": "mid",
            },
        })

        out = capsys.readouterr().out
        assert "Registered backends: 1 | running: 0 | benchmarked: 1" in out
        assert f"8123  {long_key}" in out
        assert "Model" in out
        assert "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B" in out
        assert "Bench" in out
        assert "TG 12.5 tok/s" in out
        assert "./router-start bench --backend <backend-key> --start-stopped" in out

    def test_bench_plan_prints_backend_table(self, capsys):
        backends = {
            "qwen3-5-35b-a3b-ud-q4-k-xl": {
                "engine": "llama.cpp",
                "tier": "fast",
                "port": 8080,
                "model": "/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
            }
        }

        cli._print_bench_plan(backends, list(backends))

        out = capsys.readouterr().out
        assert "Benchmark plan:" in out
        assert "Backend" in out
        assert "Engine" in out
        assert "Tier" in out
        assert "Port" in out
        assert "qwen3-5-35b-a3b-ud-q4-k-xl" in out
        assert "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf" in out
        assert cli._shorten("abcdefghijklmnopqrstuvwxyz", 8) == "abcdefg…"

    def test_no_running_bench_help_shows_copyable_commands(self, capsys):
        backends = {
            "slow-deep": {
                "engine": "llama.cpp",
                "tier": "deep",
                "size_gb": 80.0,
                "description": "Slow deep model",
            },
            "fast-choice": {
                "engine": "llama.cpp",
                "tier": "fast",
                "size_gb": 20.0,
                "description": "Fast choice",
            },
            "hf-choice": {
                "engine": "vllm",
                "tier": "fast",
                "size_gb": 18.0,
                "description": "HF choice",
            },
        }

        cli._print_no_running_bench_help(backends)

        out = capsys.readouterr().out
        assert "No model is running" in out
        assert "./router-start bench --backend fast-choice --start-stopped" in out
        assert "./router-start bench --backend hf-choice --start-stopped" in out
        assert "bench --list" in out
        assert out.index("fast-choice") < out.index("hf-choice")

    def test_router_exception_text_decodes_json_detail_and_log(self):
        class FakeHTTPError(Exception):
            def read(self):
                return json.dumps({
                    "detail": "backend failed health check",
                    "log": "/tmp/backend.log",
                }).encode()

            def __str__(self):
                return "HTTP Error 503"

        text = cli._router_exception_text(FakeHTTPError())

        assert "backend failed health check" in text
        assert "log: /tmp/backend.log" in text
        assert "HTTP Error 503" not in text

    def test_router_exception_text_falls_back_to_http_body(self):
        class FakeHTTPError(Exception):
            def read(self):
                return b"plain failure"

            def __str__(self):
                return "HTTP Error 503"

        assert cli._router_exception_text(FakeHTTPError()) == "HTTP Error 503: plain failure"

    def test_benchmark_leaderboard_prints_cached_tier_list(self, capsys):
        cli._print_benchmark_leaderboard({
            "slow": {
                "backend_key": "slow",
                "validated": True,
                "engine": "llama.cpp",
                "thinking_mode": "no_think",
                "tier_measured": "deep",
                "tg_tok_s": 8.0,
                "pp_tok_s": 80.0,
                "ttft_ms": 900.0,
                "description": "slow model",
            },
            "quick": {
                "backend_key": "quick",
                "validated": True,
                "engine": "llama.cpp",
                "thinking_mode": "no_think",
                "tier_assigned": "deep",
                "tier_measured": "fast",
                "tg_tok_s": 61.0,
                "pp_tok_s": 750.0,
                "ttft_ms": 573.0,
                "description": "quick model",
            },
        })

        out = capsys.readouterr().out
        assert "Benchmark results (cached)" in out
        assert "FAST" in out
        assert "DEEP" in out
        assert "quick" in out
        assert "TG=61 tok/s" in out
        assert "Use this result" in out
        assert "Fastest measured key: quick" in out
        assert "[route:quick] Say hello" in out
        assert "configured=deep" in out
        assert "measured=fast" in out
        assert "Auto-routing still uses the configured tier" in out
        assert out.index("quick") < out.index("slow")

    def test_backend_failure_help_prints_log_and_oom_hint(self, tmp_path, capsys):
        log_path = tmp_path / "backend.log"
        log_path.write_text("INFO loading\nCUDA out of memory while allocating KV cache\n")

        cli._print_backend_failure_help("deep", "Backend 'deep' failed to start", str(log_path))

        out = capsys.readouterr().out
        assert "diagnosis:" in out
        assert "GPU memory/OOM" in out
        assert str(log_path) in out
        assert "tail -n 120" in out

    def test_backend_failure_help_classifies_vllm_unsupported_model(self, capsys):
        cli._print_backend_failure_help(
            "hf-test",
            "ValueError: model architecture is not supported by vLLM",
            None,
        )

        out = capsys.readouterr().out
        assert "compatibility failure" in out
        assert "vLLM version" in out

    def test_backend_failure_help_adds_docker_hint_for_managed_trt(self, capsys):
        cli._print_backend_failure_help(
            "hf-test",
            "Backend 'hf-test' failed to start",
            None,
            engine="trt-llm-docker",
        )

        out = capsys.readouterr().out
        assert "Managed Docker TRT-LLM" in out

    def test_bench_thinking_mode_defaults_to_no_think(self):
        assert cli._bench_thinking_mode(argparse.Namespace()) == "no_think"
        assert cli._bench_thinking_mode(argparse.Namespace(thinking=True)) == "think"
        assert cli._bench_thinking_mode(argparse.Namespace(default_thinking=True)) == "default"

    def test_running_backend_keys(self):
        status = {
            "fast": {"running": True},
            "mid": {"running": False},
            "deep": {},
        }

        assert cli._running_backend_keys(status) == {"fast"}

    def test_python_guard_exits_before_creating_unsupported_venv(self, capsys):
        try:
            cli._ensure_supported_python((3, 9, 6))
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("old Python should be rejected")

        err = capsys.readouterr().err
        assert "requires Python 3.10+" in err
        assert "delete any old .venv" in err

    def test_wait_router_ready_polls_health_endpoint(self, monkeypatch):
        calls = []

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_urlopen(url, **kwargs):
            calls.append((url, kwargs))
            if len(calls) == 1:
                raise urllib.error.URLError("booting")
            return FakeResponse()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        monkeypatch.setattr(cli.time, "sleep", lambda _: None)

        ready, error = cli._wait_router_ready(timeout_s=2)

        assert ready is True
        assert error is None
        assert calls[0][0].endswith("/health")

    def test_update_restart_uses_clean_start_args(self, monkeypatch):
        calls = []

        monkeypatch.setattr(cli, "update_llama", lambda: calls.append("llama"))
        monkeypatch.setattr(cli, "update_pip", lambda: calls.append("pip"))
        monkeypatch.setattr(cli, "cmd_stop", lambda args: calls.append(("stop", args.restart)))
        monkeypatch.setattr(cli.time, "sleep", lambda _: calls.append("sleep"))

        def fake_start(args):
            calls.append(("start", args.update))

        monkeypatch.setattr(cli, "cmd_start", fake_start)

        cli.cmd_update(argparse.Namespace(restart=True))

        assert calls == [
            "llama",
            "pip",
            ("stop", True),
            "sleep",
            ("start", False),
        ]

    def test_sysinfo_uses_current_python_when_venv_is_missing(self, monkeypatch, tmp_path, capsys):
        payload = {
            "platform": {"os": "Linux", "arch": "x86_64", "os_version": "6.8.0", "python": "3.11.9"},
            "cpu": {"model": "Test CPU", "cores": 8},
            "ram": {"total_gb": 32.0},
            "gpu": {"available": False, "devices": []},
            "cuda": {"available": False, "version": None},
            "engine_versions": {},
            "recommendations": {},
            "conflict_processes": [],
        }
        commands = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

        def fake_urlopen(*args, **kwargs):
            raise RuntimeError("router not running")

        monkeypatch.setattr(cli, "VENV_PYTHON", tmp_path / "missing-python")
        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        cli.cmd_sysinfo(argparse.Namespace(all=False))

        out = capsys.readouterr().out
        assert commands
        assert commands[0][0] == sys.executable
        assert "System Info" in out
        assert "router not running" in out.lower()

    def test_llama_update_rolls_back_checkout_and_binary_on_build_failure(self, monkeypatch, tmp_path):
        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()
        llama_bin = llama_dir / "build" / "bin" / "llama-server"
        llama_bin.parent.mkdir(parents=True)
        llama_bin.write_text("old-binary")
        run_calls = []

        def fake_check_output(cmd, **kwargs):
            if cmd == ["git", "rev-parse", "HEAD"]:
                return "oldsha\n"
            if cmd == ["git", "rev-parse", "@{u}"]:
                return "newsha\n"
            if cmd[:3] == ["git", "describe", "--tags"]:
                return "b4567\n"
            raise AssertionError(f"unexpected check_output command: {cmd}")

        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            if cmd == ["cmake", "--build", "build", "--config", "Release", f"-j{cli.os.cpu_count() or 4}"]:
                llama_bin.write_text("partial-new-binary")
                raise subprocess.CalledProcessError(1, cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(cli, "_llama_dir", lambda: llama_dir)
        monkeypatch.setattr(cli, "_llama_bin", lambda: llama_bin)
        monkeypatch.setattr(cli, "_llama_build_config", lambda: ("CPU", ["cmake", "-B", "build"]))
        monkeypatch.setattr(cli.subprocess, "check_output", fake_check_output)
        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")

        try:
            cli.update_llama()
        except subprocess.CalledProcessError:
            pass
        else:
            raise AssertionError("build failure should propagate")

        assert ["git", "checkout", "--quiet", "b4567"] in run_calls
        assert ["git", "checkout", "--quiet", "oldsha"] in run_calls
        assert llama_bin.read_text() == "old-binary"

    def test_bench_treats_trtllm_docker_as_managed_backend(self, monkeypatch, capsys):
        backends = {
            "docker-trt": {
                "engine": "trt-llm-docker",
                "tier": "fast",
                "port": 8111,
                "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
                "log": "/tmp/backend.log",
            }
        }
        status = {"docker-trt": {"running": True, "log": "/tmp/backend.log"}}

        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(self._payload).encode()

        class FakePostResponse:
            def read(self):
                return b"{}"

        async def fake_measure_backend(key, cfg, config, thinking_mode="no_think"):
            return {
                "backend_key": key,
                "thinking_mode": thinking_mode,
                "engine": cfg["engine"],
                "tier_assigned": cfg["tier"],
                "tg_tok_s": 42.0,
                "pp_tok_s": 300.0,
                "ttft_ms": 500.0,
                "validated": True,
            }

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda url, timeout=5: FakeResponse(backends),
        )
        monkeypatch.setattr(cli, "_fetch_router_status", lambda: status)
        monkeypatch.setattr("router.benchmark.measure_backend", fake_measure_backend)
        monkeypatch.setattr("router.benchmark.save_result", lambda result, config: None)
        monkeypatch.setattr("router.benchmark.format_results", lambda results: "formatted")
        monkeypatch.setattr("router.config.load_config", lambda: SimpleNamespace())
        monkeypatch.setattr(cli, "_post_router", lambda path, timeout=30: FakePostResponse())

        cli.cmd_bench(
            argparse.Namespace(
                results=False,
                all=False,
                backend="docker-trt",
                start_stopped=False,
                keep_running=False,
                thinking=False,
                default_thinking=False,
                list=False,
            )
        )

        out = capsys.readouterr().out
        assert "Skipping external servers" not in out
        assert "[1/1] docker-trt" in out
        assert "formatted" in out
