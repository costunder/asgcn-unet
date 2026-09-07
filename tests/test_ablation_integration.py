"""Synthetic CPU diagnostics only; production configs/data are never reduced here."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from asgcn_unet import engine, preflight
from asgcn_unet.batching import pack_samples
from asgcn_unet.losses import ReconstructionLoss
from asgcn_unet.model import ASGCNUNet
from tests.test_training_batch import _config


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def architecture(config, encoder, decoder):
    config["encoder_kind"] = encoder
    config["decoder_kind"] = decoder
    if decoder == "transformer":
        config["transformer_config"] = {
            "depths": [1, 1, 2, 1, 1], "heads": [1, 2, 4, 2, 1],
            "window_size": 2, "mlp_ratio": 4.0,
        }
    return config


@pytest.mark.parametrize("encoder,decoder", [
    ("identity", "unet"), ("pointwise", "unet"),
    ("graph", "unet"), ("graph", "transformer"),
])
def test_synthetic_train_calibrate_evaluate_and_benchmark(tmp_path, encoder, decoder):
    # This fixture is explicitly a one-epoch CPU smoke test. The checked-in
    # research configs retain their full 40 epochs and original dataset scope.
    config = _config(tmp_path, batch_size=2)
    architecture(config["model"], encoder, decoder)
    engine.train(config)
    ann = Path(config["output"]["run_dir"]) / "best.pt"
    loaded, metadata = engine.load_model_checkpoint(ann, torch.device("cpu"), config["model"])
    assert metadata["checkpoint_type"] == "ann_inference"
    assert loaded.encoder_kind == encoder
    assert loaded.decoder_kind == decoder
    assert loaded.architecture_description()["spiking_supported"] == (encoder != "identity")
    result_config = copy.deepcopy(config)
    result_config["dataset"]["root"] = config["dataset"]["val_root"]
    result_config["dataset"].pop("val_root")
    result_config["dataset"].pop("split_manifest")
    result_config["eval"] = {
        "output_dir": str(tmp_path / "debug-evaluation"), "batch_size": 2,
        "num_workers": 0, "max_samples": None, "save_predictions": 1,
        "precision": "fp32", "tf32": False,
    }
    result = engine.evaluate(
        result_config, ann, allow_unsealed_checkpoint_for_non_reporting=True,
    )
    assert result["quality"]["frames"] == 2
    assert result["report_eligible"] is False
    timing = engine.benchmark(
        result_config, ann, warmup=0, steps=2,
        allow_unsealed_checkpoint_for_non_reporting=True,
    )
    assert timing["frames"] == 2
    assert timing["report_eligible"] is False
    if encoder == "identity":
        with pytest.raises(ValueError, match="no-SNN|identity|Identity"):
            loaded.fold_batch_norm()
        return
    config["calibration"] = {"batch_size": 2, "num_workers": 0, "persistent_workers": False}
    snn = tmp_path / "debug-snn.pt"
    engine.calibrate(config, ann, snn, allow_unsealed_calibration=True)
    converted, seal = engine.load_model_checkpoint(snn, torch.device("cpu"), config["model"])
    assert seal["snn_calibration_samples"] == 7
    assert converted.supports_snn
    snn_result = engine.evaluate(
        result_config, snn, inference_mode="snn", simulation_steps=4,
        allow_unsealed_checkpoint_for_non_reporting=True,
    )
    assert snn_result["quality"]["frames"] == 2
    assert snn_result["report_eligible"] is False


def samples():
    torch.manual_seed(31)
    output = []
    for index in range(2):
        events = torch.rand(36, 4)
        events[:, :2] *= 30
        events[:, 2] = torch.arange(36)
        output.append({
            "events": events, "target": torch.rand(1, 31, 31),
            "sensor_size": (31, 31), "sample_id": f"debug-{index}/0",
            "metadata": {"scene": f"debug-{index}", "sequence_index": 0},
        })
    return output


@pytest.mark.parametrize("encoder", ["identity", "pointwise"])
def test_no_graph_forward_preflight_and_cache_identity(monkeypatch, encoder):
    def forbidden(*args, **kwargs):
        raise AssertionError("Declared no-graph architecture must not build radius graphs")
    monkeypatch.setattr("asgcn_unet.model.build_event_graph", forbidden)
    monkeypatch.setattr("asgcn_unet.model.build_event_graph_batch", forbidden)
    monkeypatch.setattr(preflight, "radius_graph_topology", forbidden)
    config = {"encoder_kind": encoder, "spline_backend": "torch"}
    model = ASGCNUNet(**config).train()
    batch = samples()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    prediction, details = model.forward_training_batch(pack_samples(batch))
    target = torch.stack([sample["target"] for sample in batch])
    loss, _ = ReconstructionLoss()(prediction, target)
    loss.backward()
    missing = [name for name, value in model.named_parameters() if value.grad is None]
    assert missing == []
    engine._centralize_gradients(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.step()
    assert any(not torch.equal(value, before[name]) for name, value in model.named_parameters())
    assert all(int(item["edges"]) == 0 and item["edge_feature"] is None for item in details)
    record = preflight._sample_topology(batch[0], config, 0)
    assert record["topology_kind"] == "no_graph"
    assert record["actual_directed_edges"] == record["candidate_directed_edges"] == 0
    preflight._validate_topology_records([record], 1, config, complete=True)
    with pytest.raises(ValueError, match="cannot be reused"):
        preflight._validate_topology_records([record], 1, {}, complete=True)
    assert preflight._topology_input_config({"model": config}) != preflight._topology_input_config({})
    model.eval()
    with torch.no_grad():
        packed, _ = model.forward_batch(pack_samples(batch))
        singles = torch.cat([model.forward_sample(sample)[0] for sample in batch])
    torch.testing.assert_close(packed, singles, atol=2e-6, rtol=2e-5)


def test_original_implicit_graph_architecture_is_bitwise_unchanged():
    torch.manual_seed(71)
    old = ASGCNUNet(hidden_dim=4, graph_layers=1, decoder_channels=4)
    torch.manual_seed(71)
    explicit = ASGCNUNet(
        hidden_dim=4, graph_layers=1, decoder_channels=4,
        encoder_kind="graph", decoder_kind="unet",
    )
    assert old.state_dict().keys() == explicit.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])
    old.eval()
    explicit.eval()
    with torch.no_grad():
        assert torch.equal(old.forward_batch(samples())[0], explicit.forward_batch(samples())[0])
