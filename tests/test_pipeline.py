from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from asgcn_recon.cli import inspect_dataset
from asgcn_recon.data import EventAidRZipDataset, EventHDRDataset
from asgcn_recon.data.common import stratified_subsample, uniform_cap_ratio
from asgcn_recon.data.factory import build_dataset
from asgcn_recon.engine import _data_loader, _model_state_sha256, benchmark, train
from asgcn_recon.graph import build_radius_graph, prepare_event_nodes
from asgcn_recon.losses import ReconstructionLoss
from asgcn_recon.model import ASGCNReconstructor
from asgcn_recon.utils import (
    load_json,
    resolve_experiment_paths,
)
from tests.fixtures import make_eventaid, make_eventhdr


def _paper_model_config(
    hidden_dim: int = 4,
    graph_layers: int = 1,
    *,
    recurrent: bool = True,
) -> dict:
    return {
        "architecture_version": 2,
        "graph_operator": "spline",
        "spline_backend": "torch",
        "spline_pseudo": "distance_over_radius",
        "spline_is_open": True,
        "hidden_dim": hidden_dim,
        "graph_layers": graph_layers,
        "event_sampling_factor": 1,
        "graph_radius": 2.0,
        "graph_position_dims": 3,
        "graph_chunk_size": 16,
        "spline_kernel_size": 3,
        "spline_degree": 1,
        "spline_root_weight": True,
        "raster_downsample": 4,
        "decoder_channels": 4,
        "output_channels": 1,
        "recurrent": recurrent,
    }


def test_eventhdr_loader(tmp_path):
    make_eventhdr(tmp_path / "hdr")
    dataset = EventHDRDataset(tmp_path / "hdr", max_events=32)
    sample = dataset[0]
    assert sample["events"].shape == (32, 4)
    assert sample["target"].shape == (1, 32, 48)
    assert sample["events"][:, 2].min() >= 0
    assert sample["events"][:, 2].max() <= 1
    assert sample["metadata"]["raw_event_count"] == 96
    assert sample["metadata"]["cropped_event_count"] == 96
    assert sample["metadata"]["retained_event_count"] == 32
    assert sample["metadata"]["dataset_sampling_ratio"] == 3.0
    assert dataset[1]["metadata"]["dt_us"] == 2_000


def test_eventhdr_stride_aggregates_intervals(tmp_path):
    make_eventhdr(tmp_path / "hdr")
    dataset = EventHDRDataset(tmp_path / "hdr", max_events=None, frame_stride=2)
    assert len(dataset) == 2
    assert dataset.samples[1]["end_idx"] - dataset.samples[1]["start_idx"] == 192
    assert dataset[1]["metadata"]["dt_us"] == 4_000
    assert dataset[1]["metadata"]["raw_event_count"] == 192
    assert dataset[1]["metadata"]["cropped_event_count"] == 192
    assert dataset[1]["metadata"]["retained_event_count"] == 192
    assert dataset[1]["metadata"]["dataset_sampling_ratio"] == 1.0


def test_eventhdr_preserves_zero_event_target_intervals(tmp_path):
    path = make_eventhdr(tmp_path / "hdr")
    with h5py.File(path, "r+") as h5:
        first_end = int(h5["images/image000000000"].attrs["event_idx"])
        h5["images/image000000001"].attrs["event_idx"] = first_end

    dataset = EventHDRDataset(path.parent, max_events=None)
    assert len(dataset) == 4
    assert dataset.zero_event_intervals == 1
    empty = dataset[1]
    assert empty["events"].shape == (0, 4)
    assert empty["metadata"]["zero_event_interval"] is True
    assert empty["metadata"]["raw_event_count"] == 0
    assert empty["metadata"]["sequence_index"] == 1
    assert empty["metadata"]["dt_us"] == 2_000


def test_eventaid_next_frame_alignment(tmp_path):
    make_eventaid(tmp_path / "eventaid")
    dataset = EventAidRZipDataset(tmp_path / "eventaid", max_events=32)
    assert len(dataset) == 3
    assert dataset.samples[0]["frame_id"] == 1
    assert dataset.samples[0]["target_name"].endswith("000002_img.png")
    assert dataset[0]["metadata"]["dt_us"] == 10_000
    assert dataset[0]["metadata"]["raw_event_count"] == 80
    assert dataset[0]["metadata"]["cropped_event_count"] == 80
    assert dataset[0]["metadata"]["retained_event_count"] == 32
    assert dataset[0]["metadata"]["dataset_sampling_ratio"] == 2.5


def test_max_events_uses_exact_size_uniform_sampling() -> None:
    events = np.arange(13 * 4, dtype=np.float32).reshape(13, 4)
    assert uniform_cap_ratio(len(events), max_events=5) == 2.6
    retained = stratified_subsample(events, max_events=5)
    np.testing.assert_array_equal(retained, events[[0, 3, 6, 9, 12]])
    assert uniform_cap_ratio(len(events), max_events=None) == 1.0
    np.testing.assert_array_equal(stratified_subsample(events, None), events)


def test_max_events_has_no_one_event_boundary_collapse() -> None:
    at_cap = np.arange(8192 * 4, dtype=np.float32).reshape(8192, 4)
    over_cap = np.arange(8193 * 4, dtype=np.float32).reshape(8193, 4)
    assert len(stratified_subsample(at_cap, 8192)) == 8192
    assert len(stratified_subsample(over_cap, 8192)) == 8192
    assert uniform_cap_ratio(len(over_cap), 8192) == pytest.approx(8193 / 8192)


@pytest.mark.parametrize("dataset_name", ["eventhdr", "eventaid"])
def test_random_crop_is_deterministic_and_sequence_aligned(tmp_path, dataset_name):
    root = tmp_path / dataset_name
    if dataset_name == "eventhdr":
        make_eventhdr(root)
        dataset_class = EventHDRDataset
    else:
        make_eventaid(root)
        dataset_class = EventAidRZipDataset

    arguments = {
        "max_events": None,
        "crop_size": [8, 8],
        "random_crop": True,
        "seed": 41,
    }
    first = dataset_class(root, **arguments)
    first_samples = [first[index] for index in range(len(first))]
    first_crops = [
        (sample["metadata"]["crop"]["top"], sample["metadata"]["crop"]["left"])
        for sample in first_samples
    ]
    assert all(
        sample["metadata"]["raw_event_count"]
        >= sample["metadata"]["cropped_event_count"]
        == sample["metadata"]["retained_event_count"]
        and sample["metadata"]["dataset_sampling_ratio"] == 1.0
        for sample in first_samples
    )
    assert any(
        sample["metadata"]["cropped_event_count"] < sample["metadata"]["raw_event_count"]
        for sample in first_samples
    )
    repeated_crop = first[0]["metadata"]["crop"]
    first.close()

    reopened = dataset_class(root, **arguments)
    reopened_crops = [
        (sample["metadata"]["crop"]["top"], sample["metadata"]["crop"]["left"])
        for sample in (reopened[index] for index in range(len(reopened)))
    ]
    assert reopened[0]["metadata"]["crop"] == repeated_crop
    reopened.close()

    assert reopened_crops == first_crops
    assert len(set(first_crops)) == 1


def test_event_graph_is_undirected_instead_of_causal():
    events = torch.tensor([[i, i, i, i % 2] for i in range(12)], dtype=torch.float32)
    _, positions = prepare_event_nodes(events, (16, 16))
    edge_index, edge_attr = build_radius_graph(positions, radius=2.0, position_dims=3, chunk_size=4)
    pairs = set(map(tuple, edge_index.transpose(0, 1).tolist()))
    assert len(pairs) == len(events) * (len(events) - 1)
    assert all(source != destination for source, destination in pairs)
    assert all((destination, source) in pairs for source, destination in pairs)
    assert edge_attr.shape == (len(pairs), 1)


def test_empty_event_interval_uses_zero_node_graph():
    sample = {
        "events": torch.empty((0, 4), dtype=torch.float32),
        "target": torch.zeros((1, 8, 8), dtype=torch.float32),
        "sensor_size": (8, 8),
        "sample_id": "empty/0",
        "metadata": {},
    }
    model = ASGCNReconstructor(**_paper_model_config())
    prediction, diagnostics = model.forward_sample(sample)
    prediction.mean().backward()
    assert torch.isfinite(prediction).all()
    assert diagnostics["nodes"] == 0
    assert diagnostics["edges"] == 0


def test_model_forward_backward(tmp_path):
    make_eventhdr(tmp_path / "hdr")
    sample = EventHDRDataset(tmp_path / "hdr", max_events=32)[0]
    model = ASGCNReconstructor(**_paper_model_config(hidden_dim=8, graph_layers=2))
    prediction, diagnostics = model.forward_sample(sample)
    loss, _ = ReconstructionLoss()(prediction, sample["target"].unsqueeze(0))
    loss.backward()
    assert prediction.shape == (1, 1, 32, 48)
    assert diagnostics["edges"] == diagnostics["nodes"] * (diagnostics["nodes"] - 1)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_bn_folding_and_explicit_snn_path(tmp_path):
    make_eventhdr(tmp_path / "hdr")
    sample = EventHDRDataset(tmp_path / "hdr", max_events=32)[0]
    model_config = _paper_model_config(hidden_dim=8, graph_layers=2, recurrent=False)
    model = ASGCNReconstructor(**model_config).eval()
    with torch.no_grad():
        ann_before, _ = model.forward_sample(sample)
        model.fold_batch_norm()
        ann_after, _ = model.forward_sample(sample)
        restored = ASGCNReconstructor(**model_config).eval()
        restored.load_state_dict(model.state_dict())
        ann_restored, _ = restored.forward_sample(sample)
        model.reset_activation_maxima()
        model.calibrate_sample(sample)
        model.apply_parameter_normalization()
        snn_output, diagnostics = model.forward_sample(
            sample, inference_mode="snn", simulation_steps=8
        )
    assert torch.allclose(ann_before, ann_after, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ann_after, ann_restored, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(snn_output).all()
    assert len(diagnostics["firing_rates"]) == 2


def test_cpu_autocast_keeps_raster_dtypes_compatible(tmp_path):
    make_eventhdr(tmp_path / "hdr")
    sample = EventHDRDataset(tmp_path / "hdr", max_events=16)[0]
    model = ASGCNReconstructor(**_paper_model_config())
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
    make_eventhdr(data_root)
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


def test_eventhdr_manifest_accepts_nested_relative_paths(tmp_path):
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root / "scene")
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train_files": ["scene/test.h5"],
                "val_files": ["unused.h5"],
            }
        ),
        encoding="utf-8",
    )
    dataset = build_dataset(
        {
            "type": "eventhdr",
            "root": str(data_root),
            "split_manifest": str(manifest_path),
        },
        split="train",
    )
    assert len(dataset) == 4
    assert dataset[0]["metadata"]["scene"] == "scene/test.h5"


def test_factory_uses_val_root_for_validation_split(tmp_path):
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    train_path = make_eventhdr(train_root, frames=2)
    val_path = make_eventhdr(val_root, frames=4)
    train_path.rename(train_root / "train.h5")
    val_path.rename(val_root / "val.h5")
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train_files": ["train.h5"],
                "val_files": ["val.h5"],
            }
        ),
        encoding="utf-8",
    )
    dataset = build_dataset(
        {
            "type": "eventhdr",
            "root": str(train_root),
            "val_root": str(val_root),
            "split_manifest": str(manifest_path),
        },
        split="val",
    )
    assert len(dataset) == 4
    assert dataset[0]["metadata"]["source"].endswith("val.h5")


def test_inspect_training_config_validates_both_manifest_splits(tmp_path):
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root)
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train_files": ["test.h5"],
                "val_files": ["missing_validation.h5"],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "dataset": {
            "type": "eventhdr",
            "root": str(data_root),
            "split_manifest": str(manifest_path),
        }
    }
    with pytest.raises(FileNotFoundError, match=r"missing_validation\.h5"):
        inspect_dataset(config, samples=1)


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
        "model": _paper_model_config(),
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
    make_eventhdr(data_root)
    config = _tiny_training_config(tmp_path, data_root)
    train(config)
    first = torch.load(tmp_path / "run/last.pt", map_location="cpu", weights_only=False)
    assert first["epoch"] == 1
    assert all(key in first for key in ("optimizer", "scheduler", "scaler", "rng_state"))
    assert (tmp_path / "run/.data_hash_cache.json").is_file()
    protocol_text = json.dumps(first["validation_protocol"])
    assert str(data_root) not in protocol_text
    assert "mtime_ns" not in protocol_text
    best = torch.load(tmp_path / "run/best.pt", map_location="cpu", weights_only=False)
    assert best["checkpoint_type"] == "ann_inference"
    assert first["checkpoint_type"] == "training"
    assert first["model_state_sha256"] == _model_state_sha256(first["model"])
    assert len(first["best_model_state_sha256"]) == 64
    assert best["model_state_sha256"] == first["best_model_state_sha256"]
    for training_key in (
        "optimizer",
        "scheduler",
        "scaler",
        "history",
        "rng_state",
        "config",
    ):
        assert training_key not in best

    config["train"]["epochs"] = 2
    train(config, resume_from=tmp_path / "run/last.pt")
    resumed = torch.load(tmp_path / "run/last.pt", map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2
    assert [entry["epoch"] for entry in resumed["history"]] == [1, 2]

    config["train"]["epochs"] = 3
    config["seed"] = 18
    with pytest.raises(ValueError, match="validation protocol differs"):
        train(config, resume_from=tmp_path / "run/last.pt")


def test_null_validation_interval_scores_only_the_single_final_epoch(tmp_path) -> None:
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root)
    config = _tiny_training_config(tmp_path, data_root)
    config["train"].update({"epochs": 2, "validate_every": None})

    train(config)

    checkpoint = torch.load(
        tmp_path / "run/last.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["history"][0]["val"] == {}
    assert checkpoint["history"][1]["val"]["frames"] == 1
    assert checkpoint["checkpoint_selection"] == "single_final_epoch"
    assert checkpoint["validation_protocol"]["selection_metric"] == (
        "single_final_epoch_macro_ssim"
    )


def test_training_rejects_resume_into_a_different_run_directory(tmp_path):
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root)
    config = _tiny_training_config(tmp_path, data_root)
    train(config)
    source = tmp_path / "run/last.pt"

    config["train"]["epochs"] = 2
    config["output"]["run_dir"] = str(tmp_path / "other-run")
    with pytest.raises(ValueError, match="inside the configured run_dir"):
        train(config, resume_from=source)


def test_exact_resume_rejects_missing_state_and_tampered_historical_best(tmp_path):
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root)

    for missing_key in ("scaler", "rng_state"):
        run_root = tmp_path / missing_key
        config = _tiny_training_config(run_root, data_root)
        train(config)
        last_path = run_root / "run/last.pt"
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        checkpoint.pop(missing_key)
        torch.save(checkpoint, last_path)
        config["train"]["epochs"] = 2
        expected_message = "GradScaler state" if missing_key == "scaler" else "RNG state"
        with pytest.raises(ValueError, match=expected_message):
            train(config, resume_from=last_path)

    digest_root = tmp_path / "digest"
    config = _tiny_training_config(digest_root, data_root)
    train(config)
    best_path = digest_root / "run/best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    tensor_name = next(name for name, value in best["model"].items() if value.is_floating_point())
    best["model"][tensor_name] = best["model"][tensor_name].clone()
    best["model"][tensor_name].view(-1)[0] += 1
    torch.save(best, best_path)
    config["train"]["epochs"] = 2
    with pytest.raises(ValueError, match="does not match tensor bytes"):
        train(config, resume_from=digest_root / "run/last.pt")

    last_digest_root = tmp_path / "last-digest"
    config = _tiny_training_config(last_digest_root, data_root)
    train(config)
    last_path = last_digest_root / "run/last.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    tensor_name = next(name for name, value in last["model"].items() if value.is_floating_point())
    last["model"][tensor_name] = last["model"][tensor_name].clone()
    last["model"][tensor_name].view(-1)[0] += 1
    torch.save(last, last_path)
    config["train"]["epochs"] = 2
    with pytest.raises(ValueError, match="does not match tensor bytes"):
        train(config, resume_from=last_path)


def test_training_can_resume_before_first_validation_checkpoint(tmp_path):
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root)
    config = _tiny_training_config(tmp_path, data_root)
    train(config)

    last_path = tmp_path / "run/last.pt"
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    checkpoint["best_ssim"] = float("-inf")
    checkpoint["val"] = {}
    checkpoint["history"] = [
        {"epoch": 1, "train_loss": checkpoint["history"][0]["train_loss"], "val": {}}
    ]
    torch.save(checkpoint, last_path)
    (tmp_path / "run/best.pt").unlink()

    config["train"]["epochs"] = 2
    train(config, resume_from=last_path)
    resumed = torch.load(last_path, map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2
    assert (tmp_path / "run/best.pt").is_file()


def test_benchmark_rejects_empty_measurement(tmp_path):
    with pytest.raises(ValueError, match="steps must be at least 1"):
        benchmark({}, tmp_path / "unused.pt", steps=0)


def test_hdf5_and_zip_loaders_are_multiprocess_safe(tmp_path):
    hdr = tmp_path / "hdr"
    eventaid = tmp_path / "eventaid"
    make_eventhdr(hdr)
    make_eventaid(eventaid)
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
