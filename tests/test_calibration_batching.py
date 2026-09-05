"""CPU unit integration of full calibration coverage, not production benchmarks."""

from __future__ import annotations

import copy

import torch

from asgcn_unet import engine
from asgcn_unet.model import ASGCNUNet
from asgcn_unet.utils import atomic_torch_save
from tests.fixtures import make_eventhdr
from tests.test_p0_engine import _eval_config, _model_config


def _debug_fixture(tmp_path):
    root = tmp_path / "hdr"
    make_eventhdr(root / "scene_a")
    make_eventhdr(root / "scene_b")
    model_config = _model_config()
    model = ASGCNUNet(**model_config)
    state = model.state_dict()
    source = tmp_path / "debug-ann.pt"
    atomic_torch_save(
        {
            "checkpoint_type": "training",
            "epoch": 1,
            "model": state,
            "model_state_sha256": engine._model_state_sha256(state),
            "model_config": model_config,
            "paper_core_version": engine.PAPER_CORE_VERSION,
        },
        source,
    )
    config = _eval_config(root, tmp_path / "unused-eval")
    config["calibration"] = {"batch_size": 2, "num_workers": 0, "persistent_workers": False}
    dataset = engine.build_dataset(config["dataset"], split="calibration")
    try:
        total = len(dataset)
    finally:
        dataset.close()
    return config, source, total


def test_full_calibration_uses_physical_batches_and_seals_actual_frame_count(tmp_path, monkeypatch):
    config, source, total = _debug_fixture(tmp_path)
    observed = []
    original = ASGCNUNet.calibrate_batch

    def tracked(self, batch):
        assert not torch.is_autocast_enabled("cpu")
        assert batch.targets is None
        assert all("target" not in sample for sample in batch)
        observed.append(len(batch))
        return original(self, batch)

    def serial_forbidden(*args, **kwargs):
        raise AssertionError("Full calibration must use the packed batch path")

    monkeypatch.setattr(ASGCNUNet, "calibrate_batch", tracked)
    monkeypatch.setattr(ASGCNUNet, "calibrate_sample", serial_forbidden)
    output = tmp_path / "debug-batched-snn.pt"
    with torch.autocast("cpu", dtype=torch.bfloat16):
        engine.calibrate(config, source, output, allow_unsealed_calibration=True)
        assert torch.is_autocast_enabled("cpu"), (
            "Calibration must restore the caller's precision scope"
        )
    result = torch.load(output, map_location="cpu", weights_only=False)

    assert max(observed) == 2
    assert sum(observed) == total
    assert result["snn_calibration_samples"] == total
    assert result["snn_calibration_summary"]["minimum_valid_samples"] == total
    assert int(result["model"]["calibration_attempts"]) == total
    report = result["execution_report"]
    assert report["precision"]["effective"] == "fp32"
    assert report["precision"]["tf32"] is False
    assert report["batching"]["physical_batch_size"] == 2
    assert report["data"]["used_samples"] == total
    assert report["data"]["used_ratio"] == 1
    assert report["data"]["graph_statistics"]["frames"] == total
    assert result["calibration_performance"]["frames"] == total
    assert result["calibration_performance"]["frames_per_second"] > 0
    assert result["report_eligible"] is False
    engine.load_model_checkpoint(output, torch.device("cpu"), config["model"])


def test_diagnostic_calibration_samples_do_not_leak_into_final_maxima_or_counts(
    tmp_path, monkeypatch
):
    config, source, total = _debug_fixture(tmp_path)
    reference_config = copy.deepcopy(config)
    reference_config["calibration"]["batch_size"] = 1
    reference_path = tmp_path / "debug-reference-snn.pt"
    engine.calibrate(reference_config, source, reference_path, allow_unsealed_calibration=True)
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)

    config["calibration"].update(
        {
            "batch_size": "auto",
            "num_workers": "auto",
            "batch_candidates": [1, 2],
            "worker_candidates": [0],
            "profile_steps": 1,
            "profile_warmup": 0,
            "profile_debug_cpu": True,
        }
    )
    attempts = []
    original = ASGCNUNet.calibrate_batch

    def tracked(self, batch):
        assert batch.targets is None
        assert all("target" not in sample for sample in batch)
        attempts.append(len(batch))
        return original(self, batch)

    monkeypatch.setattr(ASGCNUNet, "calibrate_batch", tracked)
    output = tmp_path / "debug-auto-snn.pt"
    engine.calibrate(config, source, output, allow_unsealed_calibration=True)
    result = torch.load(output, map_location="cpu", weights_only=False)

    assert sum(attempts) > total
    assert int(result["model"]["calibration_attempts"]) == total
    assert result["snn_calibration_samples"] == total
    assert result["snn_calibration_summary"]["minimum_valid_samples"] == total
    assert result["calibration_batch_profile"]["debug_cpu"] is True
    assert result["calibration_batch_profile"]["report_eligible"] is False
    assert result["calibration_batch_profile"]["cuda_measured"] is False
    for name, tensor in reference["model"].items():
        if name.endswith(("calibration_activation_max", "normalization_scale")):
            torch.testing.assert_close(result["model"][name], tensor, rtol=1e-5, atol=1e-6)
