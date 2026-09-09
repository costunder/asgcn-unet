"""Synthetic CPU integration only; no production configuration or GPU run."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from asgcn_unet import engine
from asgcn_unet.batching import pack_samples
from asgcn_unet.model import ASGCNUNet
from tests.test_stream_preflight import SyntheticStreams
from tests.test_stream_preflight import _config as stream_config
from tests.test_training_batch import _config as fixture_config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_stream_train_checkpoint_calibrate_evaluate_benchmark_cpu_smoke(tmp_path):
    config = fixture_config(tmp_path, batch_size=2)
    config["model"].update({
        "architecture_version": 3, "graph_execution": "event_driven", "graph_layers": 2,
        "stream_config": {**stream_config()["model"]["stream_config"],
                          "window_seconds": 0.004, "time_scale_seconds": 0.01},
    })
    config["dataset"].update({
        "max_events": None, "event_time_contract": "physical_seconds_v1",
        "timestamp_scale_to_seconds": 1.0, "interval_timestamp_scale_to_seconds": 1.0,
    })
    config["train"]["validation_context_frames"] = None
    config["train"]["validation"] = {"batch_size": 2, "num_workers": 0}
    ann = engine.train(config)
    assert ann == Path(config["output"]["run_dir"]) / "best.pt"
    loaded, metadata = engine.load_model_checkpoint(ann, torch.device("cpu"), config["model"])
    assert loaded.graph_execution == "event_driven" and loaded.architecture_version == 3
    assert metadata["checkpoint_type"] == "ann_inference" and metadata["epoch"] == 1
    config["calibration"] = {"batch_size": 2, "num_workers": 0, "persistent_workers": False}
    snn = tmp_path / "synthetic-stream-snn.pt"
    engine.calibrate(config, ann, snn, allow_unsealed_calibration=True)
    _, converted = engine.load_model_checkpoint(snn, torch.device("cpu"), config["model"])
    assert converted["snn_calibration_samples"] == 7
    evaluation = copy.deepcopy(config)
    evaluation["dataset"]["root"] = evaluation["dataset"].pop("val_root")
    evaluation["dataset"].pop("split_manifest")
    evaluation["eval"] = {
        "batch_size": 2, "num_workers": 0, "max_samples": None, "save_predictions": 1,
        "precision": "fp32", "tf32": False, "recurrent_context_frames": None,
        "output_dir": str(tmp_path / "synthetic-stream-eval"),
    }
    for checkpoint, mode in ((ann, "ann"), (snn, "snn")):
        quality = engine.evaluate(evaluation, checkpoint, inference_mode=mode, simulation_steps=4,
                                  allow_unsealed_checkpoint_for_non_reporting=True)
        assert quality["quality"]["frames"] == 2 and quality["report_eligible"] is False
        assert quality["performance"]["stream_execution"]["frames"] == 2
        timing = engine.benchmark(evaluation, checkpoint, inference_mode=mode, simulation_steps=4,
                                  warmup=0, steps=2, allow_unsealed_checkpoint_for_non_reporting=True)
        assert timing["frames"] == 2 and timing["report_eligible"] is False
        assert timing["recurrent_context_policy"] == "full_group_prefix"
        assert timing["stream_execution"]["frames"] == 2
        assert timing["stream_execution"]["arrival_updates"] > 0


def test_validation_auto_profile_reconstructs_full_prefix_and_preserves_live_state(monkeypatch):
    dataset = SyntheticStreams(frames=3)
    model = ASGCNUNet(**stream_config()["model"]).eval()
    loader = DataLoader(dataset, batch_size=1, collate_fn=list)
    seen = []

    def profile(data, net, device, *, run_batch, **kwargs):
        targets = pack_samples([data[2], data[5]])
        predictions, diagnostics = run_batch(targets)
        assert predictions.shape == (2, 1, 16, 16)
        assert all(detail["recurrent_state"].sequence_index == 2 for detail in diagnostics)
        seen.append(run_batch.report)
        return {"batch_size": 2, "num_workers": 0, "report": {}}

    monkeypatch.setattr(engine, "profile_inference_batches", profile)
    result = engine.validate(model, loader, torch.device("cpu"),
                             batching_section={"batch_size": "auto", "num_workers": 0})
    assert result["frames"] == 6
    report = result["execution"]["batch_profile"]
    assert report["stream_bootstrap"] == seen[0]
    assert report["stream_bootstrap"]["full_prefix_frames"] == 4
    assert report["stream_bootstrap"]["steady_state_throughput"] is False


@pytest.mark.parametrize("gate", [None, {"schema": "asgcn_preflight_verification_v1"},
                                 {"status": "bypassed", "report_eligible": False}])
def test_direct_cuda_training_rejects_missing_or_static_preflight_before_output(tmp_path, monkeypatch, gate):
    config = stream_config()
    config["device"] = "cuda"
    config["preflight_gate"] = gate
    config["output"]["run_dir"] = str(tmp_path / "must-not-be-created")
    monkeypatch.setattr(engine, "resolve_device", lambda value: torch.device("cuda"))
    with pytest.raises(ValueError, match="verified stateful physical-batch preflight"):
        engine._train(config, None, None, 60.0)
    assert not Path(config["output"]["run_dir"]).exists()
