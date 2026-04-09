from types import SimpleNamespace

from router import discovery


def _config(scan_dir):
    return SimpleNamespace(
        scan_dirs=SimpleNamespace(gguf=[scan_dir]),
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
