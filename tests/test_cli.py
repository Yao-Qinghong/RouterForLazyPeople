"""Tests for cli.py command helpers and regression-prone flows."""

import argparse
import json
import subprocess
import sys
import urllib.error

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

        assert ["git", "pull", "--ff-only", "--quiet"] in run_calls
        assert ["git", "checkout", "--quiet", "oldsha"] in run_calls
        assert llama_bin.read_text() == "old-binary"
