"""Transformer-only ablation: synthetic CPU integration, never production results.

The one-epoch HDF5 fixture below is a named smoke test. Checked-in research
settings are only read/constructed, never trained or overwritten by this file.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from collections import Counter
from pathlib import Path

import pytest
import torch

from asgcn_unet import engine
from asgcn_unet.ablation_encoders import IdentityEventEncoder
from asgcn_unet.batching import pack_samples
from asgcn_unet.losses import ReconstructionLoss
from asgcn_unet.model import ASGCNUNet
from asgcn_unet.transformer_decoder import RecurrentTransformerDecoder, WindowTransformerBlock
from tests.test_training_batch import _config

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def bounded_synthetic_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def forbid_graph_and_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Transformer-only must not create a GNN/radius graph or initialize CUDA"
        )

    for name in ("build_radius_graph", "build_event_graph", "build_event_graph_batch"):
        monkeypatch.setattr("asgcn_unet.graph." + name, forbidden)
    monkeypatch.setattr("asgcn_unet.model.build_event_graph", forbidden)
    monkeypatch.setattr("asgcn_unet.model.build_event_graph_batch", forbidden)
    monkeypatch.setattr("asgcn_unet.graph.ASGCNEncoder.__init__", forbidden)
    monkeypatch.setattr("asgcn_unet.graph.PaperSplineConv.__init__", forbidden)
    monkeypatch.setattr("asgcn_unet.preflight.radius_graph_topology", forbidden)
    monkeypatch.setattr("torch.cuda._lazy_init", forbidden)


def research_config(kind, split="train"):
    return json.loads((PROJECT / "configs" / "ablations" / f"{kind}-{split}.json").read_text())


def synthetic_samples(counts=(36, 41), *, height=35, width=39):
    generator = torch.Generator().manual_seed(812)
    output = []
    for index, count in enumerate(counts):
        events = torch.rand(count, 4, generator=generator)
        events[:, 0] *= width - 1
        events[:, 1] *= height - 1
        events[:, 2] = torch.arange(count) * 13 + index * 1000
        events[:, 3] = torch.where(events[:, 3] > 0.5, 1.0, -1.0)
        output.append(
            {
                "events": events,
                "target": torch.rand(1, height, width, generator=generator),
                "sensor_size": (height, width),
                "sample_id": f"synthetic-transformer-only-{index}/0",
                "metadata": {"scene": f"synthetic-transformer-only-{index}", "sequence_index": 0},
            }
        )
    return output


def manual_raw_mean_raster(samples, factor, downsample):
    """Independent slow CPU test reference; not production rasterization code."""
    height, width = samples[0]["sensor_size"]
    gh, gw = math.ceil(height / downsample), math.ceil(width / downsample)
    raster = torch.zeros(len(samples), 4, gh, gw)
    counts = torch.zeros(len(samples), 1, gh, gw)
    for batch_index, sample in enumerate(samples):
        selected = sample["events"][::factor]
        if not len(selected):
            continue
        first, last = selected[0, 2], selected[-1, 2]
        duration = (last - first).abs().clamp_min(1e-6)
        for event in selected:
            x = event[0] / max(width - 1, 1)
            y = event[1] / max(height - 1, 1)
            t = (event[2] - first) / duration
            polarity = torch.tensor(1.0 if event[3] > 0 else -1.0)
            column = min(max(int(x * width / downsample), 0), gw - 1)
            row = min(max(int(y * height / downsample), 0), gh - 1)
            raster[batch_index, :, row, column] += torch.stack((x, y, t, polarity))
            counts[batch_index, :, row, column] += 1
    return raster / counts.clamp_min(1)


@pytest.mark.parametrize("split", ["train", "hdr", "aid"])
def test_checked_in_e_is_transformer_only_with_same_a_data_and_input_contract(split, monkeypatch):
    forbid_graph_and_cuda(monkeypatch)
    a, e = research_config("unet", split), research_config("transformer", split)
    assert e["dataset"] == a["dataset"]
    assert e["model"]["encoder_kind"] == a["model"]["encoder_kind"] == "identity"
    assert e["model"]["decoder_kind"] == "transformer"
    for key in (
        "event_sampling_factor",
        "raster_downsample",
        "decoder_channels",
        "output_channels",
        "recurrent",
    ):
        assert e["model"][key] == a["model"][key]
    assert e["model"]["transformer_config"] == {
        "depths": [1, 1, 2, 1, 1],
        "heads": [3, 6, 12, 6, 3],
        "window_size": 8,
        "mlp_ratio": 4.0,
    }
    if split == "train":
        assert e["train"]["epochs"] == a["train"]["epochs"] == 40
        assert e["train"]["batch_size"] == a["train"]["batch_size"] == 16
        assert e["train"]["max_train_samples"] is e["train"]["max_val_samples"] is None
    model = ASGCNUNet(**e["model"])
    assert isinstance(model.encoder, IdentityEventEncoder)
    assert isinstance(model.decoder, RecurrentTransformerDecoder)
    assert list(model.encoder.parameters()) == []
    assert model.decoder.stem.in_features == 4
    assert model.decoder.base_channels == 48
    assert sum(isinstance(layer, WindowTransformerBlock) for layer in model.modules()) == 6
    assert not model.supports_snn
    assert model.architecture_description()["topology_kind"] == "no_graph"
    assert model.architecture_description()["encoder_layers"] == 0
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_380_443


@pytest.mark.parametrize("factor", [1, 3, 5])
def test_a_and_e_receive_identical_four_channel_raw_mean_raster(factor, monkeypatch):
    forbid_graph_and_cuda(monkeypatch)
    batch = synthetic_samples((17, 0, 29, 1), height=17, width=19)
    captured = {}
    expected = manual_raw_mean_raster(batch, factor, 4)
    for name in ("unet", "transformer"):
        model_config = research_config(name)["model"]
        # Test-only sampling variant, never a checked-in research config change.
        model_config["event_sampling_factor"] = factor
        model = ASGCNUNet(**model_config).eval()

        def capture(_module, args, key=name):
            captured[key] = args[0].detach().clone()

        hook = model.decoder.register_forward_pre_hook(capture)
        try:
            with torch.no_grad():
                prediction, details = model.forward_batch(pack_samples(batch))
        finally:
            hook.remove()
        assert prediction.shape == (4, 1, 17, 19)
        assert all(int(detail["edges"]) == 0 for detail in details)
        assert [detail["nodes"] for detail in details] == [
            math.ceil(count / factor) for count in (17, 0, 29, 1)
        ]
        assert captured[name].shape == (4, 4, 5, 5)
        torch.testing.assert_close(captured[name], expected, rtol=0, atol=0)
    torch.testing.assert_close(captured["unet"], captured["transformer"], rtol=0, atol=0)


def test_full_e_forward_loss_backward_optimizer_and_recurrent_path_use_all_parameters(monkeypatch):
    forbid_graph_and_cuda(monkeypatch)
    torch.manual_seed(921)
    model = ASGCNUNet(**research_config("transformer")["model"]).train()
    batch = synthetic_samples()
    state = [torch.randn(1, 192, 3, 3, requires_grad=True) for _ in batch]
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    predictions, details = model.forward_training_batch(pack_samples(batch), state)
    target = torch.stack([sample["target"] for sample in batch])
    loss, metrics = ReconstructionLoss()(predictions, target)
    assert torch.isfinite(loss)
    assert metrics
    loss.backward()
    parameters = dict(model.named_parameters())
    assert parameters and all(name.startswith("decoder.") for name in parameters)
    assert {id(parameter) for parameter in parameters.values()} == {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in state)
    optimizer.step()
    assert all(not torch.equal(parameter, before[name]) for name, parameter in parameters.items())
    assert all(int(detail["edges"]) == 0 and detail["edge_feature"] is None for detail in details)
    assert all(detail["recurrent_state"].shape == (1, 192, 3, 3) for detail in details)


def test_synthetic_train_best_ann_load_evaluate_benchmark_and_snn_rejection(tmp_path, monkeypatch):
    forbid_graph_and_cuda(monkeypatch)
    # This is a separate synthetic CPU smoke profile: 7 H5 training frames,
    # 2 validation frames, one epoch. No research config or real result is edited.
    config = _config(tmp_path, batch_size=2)
    config["model"].update(
        {
            "encoder_kind": "identity",
            "decoder_kind": "transformer",
            "transformer_config": {
                "depths": [1, 1, 2, 1, 1],
                "heads": [1, 2, 4, 2, 1],
                "window_size": 2,
                "mlp_ratio": 4.0,
            },
        }
    )
    original_training = ASGCNUNet.forward_training_batch
    seen, physical_batches = Counter(), []

    def observed_training(model, samples, recurrent_states=None, *, timing=None):
        assert model.encoder_kind == "identity" and model.decoder_kind == "transformer"
        physical_batches.append(len(samples))
        seen.update(sample["sample_id"] for sample in samples)
        return original_training(model, samples, recurrent_states, timing=timing)

    monkeypatch.setattr(ASGCNUNet, "forward_training_batch", observed_training)
    engine.train(config)
    assert sum(seen.values()) == 7 and len(seen) == 7
    assert max(physical_batches) == 2
    ann = Path(config["output"]["run_dir"]) / "best.pt"
    model, checkpoint = engine.load_model_checkpoint(ann, torch.device("cpu"), config["model"])
    assert checkpoint["checkpoint_type"] == "ann_inference"
    assert checkpoint["epoch"] == 1
    assert model.encoder_kind == "identity" and model.decoder_kind == "transformer"
    assert model.decoder.stem.in_features == 4
    assert not model.supports_snn
    evaluation = copy.deepcopy(config)
    evaluation["dataset"]["root"] = config["dataset"]["val_root"]
    evaluation["dataset"].pop("val_root")
    evaluation["dataset"].pop("split_manifest")
    evaluation["eval"] = {
        "output_dir": str(tmp_path / "synthetic-evaluation"),
        "batch_size": 2,
        "num_workers": 0,
        "max_samples": None,
        "save_predictions": 2,
        "precision": "fp32",
        "tf32": False,
    }
    quality = engine.evaluate(evaluation, ann, allow_unsealed_checkpoint_for_non_reporting=True)
    assert quality["quality"]["frames"] == 2
    assert quality["report_eligible"] is False
    assert quality["graph_edge_guard"]["topology_kind"] == "no_graph"
    assert quality["graph_edge_guard"]["edge_guard_applicable"] is False
    output = Path(evaluation["eval"]["output_dir"]) / "ann"
    with (output / "frames.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and all(int(row["edges"]) == 0 for row in rows)
    assert len(list((output / "predictions").glob("*_pred.png"))) == 2
    timing = engine.benchmark(
        evaluation, ann, warmup=0, steps=2, allow_unsealed_checkpoint_for_non_reporting=True
    )
    assert timing["frames"] == 2 and timing["report_eligible"] is False
    assert timing["graph_edge_guard"]["topology_kind"] == "no_graph"
    assert (output / "benchmark.json").is_file()
    # SNN/calibration must not silently become ANN or manufacture a checkpoint.
    snn_output = tmp_path / "forbidden-snn.pt"
    with pytest.raises(ValueError, match="SNN|snn|identity|Identity"):
        engine.calibrate(config, ann, snn_output, allow_unsealed_calibration=True)
    assert not snn_output.exists()
    for runner in (engine.evaluate, engine.benchmark):
        with pytest.raises(ValueError, match="SNN|snn"):
            runner(
                evaluation,
                ann,
                inference_mode="snn",
                simulation_steps=4,
                allow_unsealed_checkpoint_for_non_reporting=True,
            )


@pytest.mark.parametrize(
    "operation",
    [
        "sample",
        "batch",
        "calibrate_sample",
        "calibrate_batch",
        "fold_batch_norm",
        "reset_activation_maxima",
    ],
)
def test_transformer_only_rejects_all_snn_and_calibration_entrypoints(operation, monkeypatch):
    forbid_graph_and_cuda(monkeypatch)
    model = ASGCNUNet(**research_config("transformer")["model"])
    batch = synthetic_samples((8, 9), height=17, width=19)
    with pytest.raises(ValueError, match="SNN|snn|identity|Identity"):
        if operation == "sample":
            model.forward_sample(batch[0], inference_mode="snn", simulation_steps=4)
        elif operation == "batch":
            model.forward_batch(pack_samples(batch), inference_mode="snn", simulation_steps=4)
        elif operation == "calibrate_sample":
            model.calibrate_sample(batch[0])
        elif operation == "calibrate_batch":
            model.calibrate_batch(pack_samples(batch))
        else:
            getattr(model, operation)()
