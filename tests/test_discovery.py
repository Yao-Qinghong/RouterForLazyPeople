from types import SimpleNamespace

from router import discovery


def _config(scan_dir, *, gguf=None, hf=None, trtllm=None):
    return SimpleNamespace(
        scan_dirs=SimpleNamespace(
            gguf=[scan_dir] if gguf is None else gguf,
            hf=[] if hf is None else hf,
            trtllm=[] if trtllm is None else trtllm,
        ),
        discovery=SimpleNamespace(port_end=8200),
        data_dir=scan_dir,
        tier_thresholds=SimpleNamespace(fast=10, mid=40),
        idle_timeouts=SimpleNamespace(fast=300, mid=300, deep=300),
    )


def test_auxiliary_mmproj_gguf_files_are_not_discovered(monkeypatch, tmp_path):
    model = tmp_path / "qwen3-vision-30b-q4_k_m.gguf"
    projector = tmp_path / "mmproj-qwen3-vision-bf16.gguf"
    model.write_text("fake model")
    projector.write_text("fake projector")

    monkeypatch.setattr(discovery, "_file_size_gb", lambda path: 2.0)

    found = discovery.discover_gguf_models(_config(tmp_path), [8100])

    assert list(found) == ["qwen3-vision-30b-q4-k-m"]
    assert found["qwen3-vision-30b-q4-k-m"]["model"] == str(model)


def test_auxiliary_gguf_name_detection():
    assert discovery._is_auxiliary_gguf("mmproj-f32.gguf")
    assert discovery._is_auxiliary_gguf("mmproj_Qwen3-VL.gguf")
    assert discovery._is_auxiliary_gguf("Qwen3-VL-mmproj-bf16.gguf")
    assert not discovery._is_auxiliary_gguf("Qwen3-30B-A3B-Q4_K_M.gguf")


def test_discover_hf_nvfp4_prefers_trtllm(monkeypatch, tmp_path):
    model_dir = tmp_path / "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"architectures":["NemotronHForCausalLM"],"model_type":"nemotron_h"}'
    )
    (model_dir / "model.safetensors").write_text("fake")

    monkeypatch.setattr(discovery, "_hf_model_size_gb", lambda path: 18.0)
    monkeypatch.setattr(
        discovery,
        "is_engine_available",
        lambda engine, config: engine in {"trt-llm", "vllm"},
    )

    found = discovery.discover_hf_models(_config(tmp_path, gguf=[], hf=[tmp_path]), [8100])

    assert len(found) == 1
    backend = next(iter(found.values()))
    assert backend["engine"] == "trt-llm"
    assert backend["model"] == "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
    assert backend["model_dir"] == str(model_dir)
    assert backend["tokenizer"] == str(model_dir)


def test_discover_hf_non_nvfp4_keeps_vllm_preference(monkeypatch, tmp_path):
    model_dir = tmp_path / "Qwen3.5-27B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2"}'
    )
    (model_dir / "model.safetensors").write_text("fake")

    monkeypatch.setattr(discovery, "_hf_model_size_gb", lambda path: 26.0)
    monkeypatch.setattr(
        discovery,
        "is_engine_available",
        lambda engine, config: engine in {"trt-llm", "vllm"},
    )

    found = discovery.discover_hf_models(_config(tmp_path, gguf=[], hf=[tmp_path]), [8100])

    backend = found["hf-qwen3-5-27b"]
    assert backend["engine"] == "vllm"
    assert backend["model"] == str(model_dir)
