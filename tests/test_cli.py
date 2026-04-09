"""Tests for cli.py command helpers and regression-prone flows."""

import argparse
import json
import subprocess
import sys

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
