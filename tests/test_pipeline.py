from __future__ import annotations

import json

import pytest
import torch

from asgcn_recon.data import EventAidRZipDataset, EventHDRDataset
from asgcn_recon.data.factory import build_dataset
from asgcn_recon.engine import _data_loader, benchmark, train
from asgcn_recon.graph import build_causal_graph, prepare_event_nodes
from asgcn_recon.losses import ReconstructionLoss
from asgcn_recon.model import ASGCNReconstructor
from asgcn_recon.smoke import create_eventaid_smoke, create_eventhdr_smoke
from asgcn_recon.utils import (
    load_json,
    resolve_experiment_paths,
)


def test_eventhdr_loader(tmp_path):
    create_eventhdr_smoke(tmp_path / "hdr")
    dataset = EventHDRDataset(tmp_path / "hdr", max_events=32)
    sample = dataset[0]
    assert sample["events"].shape == (32, 4)
    assert sample["target"].shape == (1, 32, 48)
    assert sample["events"][:, 2].min() >= 0
    assert sample["events"][:, 2].max() <= 1
    assert dataset[1]["metadata"]["dt_us"] == 2_000


def test_eventhdr_stride_aggregates_intervals(tmp_path):
    create_eventhdr_smoke(tmp_path / "hdr")
    dataset = EventHDRDataset(tmp_path / "hdr", max_events=None, frame_stride=2)
    assert len(dataset) == 2
    assert dataset.samples[1]["end_idx"] - dataset.samples[1]["start_idx"] == 192
    assert dataset[1]["metadata"]["dt_us"] == 4_000


def test_eventaid_next_frame_alignment(tmp_path):
    create_eventaid_smoke(tmp_path / "eventaid")
    dataset = EventAidRZipDataset(tmp_path / "eventaid", max_events=32)
    assert len(dataset) == 3
    assert dataset.samples[0]["frame_id"] == 1
    assert dataset.samples[0]["target_name"].endswith("000002_img.png")
    assert dataset[0]["metadata"]["dt_us"] == 10_000


def test_causal_graph_has_no_future_sources():
    events = torch.tensor([[i, i, i, i % 2] for i in range(12)], dtype=torch.float32)
    _, positions = prepare_event_nodes(events, (16, 16))
    edge_index, _ = build_causal_graph(
        positions, candidates=4, spatial_radius=1.0, temporal_radius=1.0
    )
    assert torch.all(edge_index[0] <= edge_index[1])


def test_singleton_graph_can_train():
    sample = {
        "events": torch.empty((0, 4), dtype=torch.float32),
        "target": torch.zeros((1, 8, 8), dtype=torch.float32),
        "sensor_size": (8, 8),
        "sample_id": "empty/0",
        "metadata": {},
    }
    model = ASGCNReconstructor(
        hidden_dim=4,
        graph_layers=1,
        causal_candidates=2,
        spatial_radius=1.0,
        temporal_radius=1.0,
        raster_downsample=4,
        decoder_channels=4,
    )
    prediction, _ = model.forward_sample(sample)
    prediction.mean().backward()
    assert torch.isfinite(prediction).all()


def test_model_forward_backward(tmp_path):
    create_eventhdr_smoke(tmp_path / "hdr")
    sample = EventHDRDataset(tmp_path / "hdr", max_events=32)[0]
    model = ASGCNReconstructor(
        hidden_dim=8,
        graph_layers=2,
        causal_candidates=4,
        spatial_radius=1.0,
        temporal_radius=1.0,
        raster_downsample=4,
        decoder_channels=4,
    )
    prediction, diagnostics = model.forward_sample(sample)
    loss, _ = ReconstructionLoss()(prediction, sample["target"].unsqueeze(0))
    loss.backward()
    assert prediction.shape == (1, 1, 32, 48)
    assert diagnostics["edges"] >= diagnostics["nodes"]
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_bn_folding_and_snn_rate_path(tmp_path):
    create_eventhdr_smoke(tmp_path / "hdr")
    sample = EventHDRDataset(tmp_path / "hdr", max_events=32)[0]
    model = ASGCNReconstructor(
        hidden_dim=8,
        graph_layers=2,
        causal_candidates=4,
        spatial_radius=1.0,
        temporal_radius=1.0,
        raster_downsample=4,
        decoder_channels=4,
        recurrent=False,
    ).eval()
    with torch.no_grad():
        ann_before, _ = model.forward_sample(sample)
        model.fold_batch_norm()
        ann_after, _ = model.forward_sample(sample)
        restored = ASGCNReconstructor(
            hidden_dim=8,
            graph_layers=2,
            causal_candidates=4,
            spatial_radius=1.0,
            temporal_radius=1.0,
            raster_downsample=4,
            decoder_channels=4,
            recurrent=False,
        ).eval()
        restored.load_state_dict(model.state_dict())
        ann_restored, _ = restored.forward_sample(sample)
        model.encoder.reset_thresholds()
        model.calibrate_sample(sample)
        snn_output, diagnostics = model.forward_sample(
            sample, inference_mode="snn", simulation_steps=8
        )
    assert torch.allclose(ann_before, ann_after, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ann_after, ann_restored, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(snn_output).all()
    assert len(diagnostics["firing_rates"]) == 2


def test_cpu_autocast_keeps_raster_dtypes_compatible(tmp_path):
    create_eventhdr_smoke(tmp_path / "hdr")
    sample = EventHDRDataset(tmp_path / "hdr", max_events=16)[0]
    model = ASGCNReconstructor(
        hidden_dim=4,
        graph_layers=1,
        causal_candidates=2,
        spatial_radius=1.0,
        temporal_radius=1.0,
        raster_downsample=4,
        decoder_channels=4,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction, _ = model.forward_sample(sample)
    assert prediction.dtype == torch.bfloat16
    assert torch.isfinite(prediction).all()


def test_config_paths_are_independent_of_shell_cwd(tmp_path):
    repository = tmp_path / "checkout"
    config_dir = repository / "configs"
    config_dir.mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    config_path = config_dir / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "root": "data/train",
                    "split_manifest": "manifests/split.json",
                },
                "output": {"run_dir": "runs/example"},
            }
        ),
        encoding="utf-8",
    )
    config = resolve_experiment_paths(load_json(config_path), config_path)
    assert config["dataset"]["root"] == str((repository / "data/train").resolve())
    assert config["dataset"]["split_manifest"] == str(
        (repository / "manifests/split.json").resolve()
    )
    assert config["output"]["run_dir"] == str((repository / "runs/example").resolve())


def test_eventhdr_split_names_all_missing_files(tmp_path):
    data_root = tmp_path / "hdr"
    create_eventhdr_smoke(data_root)
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        json.dumps({"train_files": ["1.h5", "2.h5"], "val_files": ["3.h5"]}),
        encoding="utf-8",
    )
    config = {
        "type": "eventhdr",
        "root": str(data_root),
        "split_manifest": str(manifest_path),
    }
    with pytest.raises(FileNotFoundError, match=r"1\.h5, 2\.h5"):
        build_dataset(config, split="train")


def _tiny_training_config(tmp_path, data_root):
    return {
        "seed": 17,
        "device": "cpu",
        "dataset": {
            "type": "eventhdr",
            "root": str(data_root),
            "target_channels": 1,
            "max_events": 16,
            "crop_size": [16, 16],
            "tone_map": "log",
        },
        "model": {
            "hidden_dim": 4,
            "graph_layers": 1,
            "causal_candidates": 2,
            "spatial_radius": 1.0,
            "temporal_radius": 1.0,
            "raster_downsample": 4,
            "decoder_channels": 4,
            "output_channels": 1,
            "recurrent": True,
        },
        "train": {
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "amp": False,
            "max_train_samples": 1,
            "max_val_samples": 1,
            "log_every": 100,
        },
        "output": {"run_dir": str(tmp_path / "run")},
    }


def test_training_checkpoint_can_resume_optimizer_and_epoch(tmp_path):
    data_root = tmp_path / "hdr"
    create_eventhdr_smoke(data_root)
    config = _tiny_training_config(tmp_path, data_root)
    train(config)
    first = torch.load(tmp_path / "run/last.pt", map_location="cpu", weights_only=False)
    assert first["epoch"] == 1
    assert "optimizer" in first and "scaler" in first and "rng_state" in first

    config["train"]["epochs"] = 2
    train(config, resume_from=tmp_path / "run/last.pt")
    resumed = torch.load(tmp_path / "run/last.pt", map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2
    assert [entry["epoch"] for entry in resumed["history"]] == [1, 2]


def test_benchmark_rejects_empty_measurement(tmp_path):
    with pytest.raises(ValueError, match="steps must be at least 1"):
        benchmark({}, tmp_path / "unused.pt", steps=0)


def test_hdf5_and_zip_loaders_are_multiprocess_safe(tmp_path):
    hdr = tmp_path / "hdr"
    eventaid = tmp_path / "eventaid"
    create_eventhdr_smoke(hdr)
    create_eventaid_smoke(eventaid)
    datasets = [
        EventHDRDataset(hdr, max_events=8),
        EventAidRZipDataset(eventaid, max_events=8),
    ]
    for dataset in datasets:
        loader = _data_loader(
            dataset,
            batch_size=1,
            num_workers=2,
            device=torch.device("cpu"),
            persistent_workers=False,
        )
        sample = next(iter(loader))[0]
        assert sample["events"].shape == (8, 4)
