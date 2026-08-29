from __future__ import annotations

import json
import shutil
from collections import Counter

import pytest
import torch

from asgcn_recon.data import EventHDRDataset, load_eventhdr_split_manifest
from asgcn_recon.engine import (
    _balanced_contiguous_indices,
    _balanced_sample_indices,
    _continues_sequence,
    _dataset_content_fingerprint,
    _enforce_training_split_status,
    _macro_ssim,
    _prefix_context_schedule,
    _representative_schedule,
    _resume_best_macro_ssim,
    _sampling_summary,
    _validate_resume_best_pair,
    _validate_snn_request,
    benchmark,
    calibrate,
    evaluate,
)
from asgcn_recon.model import ASGCNReconstructor
from asgcn_recon.utils import atomic_torch_save
from tests.fixtures import make_eventhdr


class _GroupedIndexDataset:
    def __init__(self) -> None:
        self.samples = [
            *({"scene": "long"} for _ in range(8)),
            *({"scene": "medium"} for _ in range(4)),
            *({"scene": "short"} for _ in range(2)),
        ]

    def __len__(self) -> int:
        return len(self.samples)


def _model_config() -> dict:
    return {
        "hidden_dim": 4,
        "graph_layers": 1,
        "causal_candidates": 2,
        "spatial_radius": 1.0,
        "temporal_radius": 1.0,
        "raster_downsample": 4,
        "decoder_channels": 4,
        "output_channels": 1,
        "recurrent": False,
    }


def _eval_config(root, output_dir) -> dict:
    return {
        "seed": 19,
        "device": "cpu",
        "dataset": {
            "type": "eventhdr",
            "root": str(root),
            "target_channels": 1,
            "max_events": 16,
            "crop_size": None,
            "tone_map": "log",
        },
        "model": _model_config(),
        "eval": {
            "num_workers": 0,
            "max_samples": 1,
            "save_predictions": 0,
            "output_dir": str(output_dir),
        },
    }


def test_balanced_indices_cover_groups_before_repeating() -> None:
    dataset = _GroupedIndexDataset()
    indices = _balanced_sample_indices(dataset, limit=6, seed=3)
    counts = Counter(dataset.samples[index]["scene"] for index in indices)
    assert counts == {"long": 2, "medium": 2, "short": 2}
    assert indices == sorted(indices)
    summary = _sampling_summary(dataset, indices)
    assert summary["selected_samples"] == 6
    assert summary["selected_groups"] == summary["available_groups"] == 3
    long_indices = [index for index in indices if dataset.samples[index]["scene"] == "long"]
    assert long_indices == [0, 7]


def test_representative_schedule_has_exact_length_and_balances_groups() -> None:
    dataset = _GroupedIndexDataset()
    schedule = _representative_schedule(dataset, count=6, seed=3)
    assert len(schedule) == 6
    counts = Counter(dataset.samples[index]["scene"] for index in schedule)
    assert counts == {"long": 2, "medium": 2, "short": 2}


def test_contiguous_sampler_balances_groups_without_state_gaps() -> None:
    dataset = _GroupedIndexDataset()
    indices = _balanced_contiguous_indices(
        dataset, limit=6, seed=3, require_all_groups=True
    )
    grouped = {
        scene: [index for index in indices if dataset.samples[index]["scene"] == scene]
        for scene in ("long", "medium", "short")
    }
    assert {scene: len(values) for scene, values in grouped.items()} == {
        "long": 2,
        "medium": 2,
        "short": 2,
    }
    assert all(values[1] == values[0] + 1 for values in grouped.values())

    with pytest.raises(ValueError, match="every validation group"):
        _balanced_contiguous_indices(
            dataset, limit=2, seed=3, require_all_groups=True
        )


def test_prefix_context_replays_unscored_predecessors() -> None:
    dataset = _GroupedIndexDataset()
    schedule, score_positions = _prefix_context_schedule(dataset, [6, 7, 12])
    assert schedule == [0, 1, 2, 3, 4, 5, 6, 7, 12]
    assert score_positions == {6, 7, 8}
    bounded, bounded_scores = _prefix_context_schedule(
        dataset, [6, 7], max_context_frames=2
    )
    assert bounded == [4, 5, 6, 7]
    assert bounded_scores == {2, 3}
    with pytest.raises(ValueError, match="non-negative"):
        _prefix_context_schedule(dataset, [6], max_context_frames=-1)


def test_content_fingerprint_is_path_independent_and_detects_changes(tmp_path) -> None:
    source = make_eventhdr(tmp_path / "one")
    destination_root = tmp_path / "two"
    destination_root.mkdir()
    destination = destination_root / source.name
    shutil.copy2(source, destination)
    first = EventHDRDataset(source.parent, max_events=8)
    second = EventHDRDataset(destination.parent, max_events=8)

    assert _dataset_content_fingerprint(first) == _dataset_content_fingerprint(second)

    with destination.open("ab") as handle:
        handle.write(b"changed")
    assert _dataset_content_fingerprint(first) != _dataset_content_fingerprint(second)
    first.close()
    second.close()


def test_macro_ssim_is_the_checkpoint_selection_score() -> None:
    validation = {
        "micro": {"ssim": 0.95},
        "macro": {"ssim": 0.61},
        "per_scene": {},
    }
    assert _macro_ssim(validation) == pytest.approx(0.61)
    assert _macro_ssim({"ssim": 0.72}) == pytest.approx(0.72)


def test_resume_rejects_legacy_micro_best_score() -> None:
    with pytest.raises(ValueError, match="predates macro-SSIM"):
        _resume_best_macro_ssim({"best_ssim": 0.9, "val": {"ssim": 0.9}})
    assert _resume_best_macro_ssim(
        {"best_metric": "macro_ssim", "best_ssim": 0.7}
    ) == pytest.approx(0.7)


def test_resume_rejects_unrelated_historical_best_checkpoint() -> None:
    resume = {
        "epoch": 2,
        "model_config": {"hidden_dim": 4},
        "validation_protocol": {"version": 2},
        "best_metric": "macro_ssim",
        "best_ssim": 0.7,
        "val": {},
    }
    best = {
        **resume,
        "epoch": 1,
        "val": {"macro": {"ssim": 0.7}},
    }
    _validate_resume_best_pair(resume, best)

    best["validation_protocol"] = {"version": 99}
    with pytest.raises(ValueError, match="different validation protocol"):
        _validate_resume_best_pair(resume, best)


def test_training_rejects_nonfinal_split_without_explicit_override(tmp_path) -> None:
    manifest = tmp_path / "split.json"
    split = {
        "status": "provisional",
        "train_files": ["train.h5"],
        "val_files": ["val.h5"],
    }
    manifest.write_text(json.dumps(split), encoding="utf-8")
    config = {
        "dataset": {"split_manifest": str(manifest)},
        "train": {"allow_provisional_split": False},
    }
    with pytest.raises(ValueError, match="allow_provisional_split"):
        _enforce_training_split_status(config)
    config["train"]["allow_provisional_split"] = True
    _enforce_training_split_status(config)

    split["status"] = "final"
    manifest.write_text(json.dumps(split), encoding="utf-8")
    config["train"]["allow_provisional_split"] = False
    _enforce_training_split_status(config)


def test_split_manifest_rejects_exact_train_val_overlap(tmp_path) -> None:
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "final",
                "train_files": ["shared.h5"],
                "val_files": ["shared.h5"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="leaks files across train/val"):
        load_eventhdr_split_manifest(manifest)


def test_sequence_continuity_requires_adjacent_index_and_matching_shape() -> None:
    assert _continues_sequence("scene", 8, (32, 48), "scene", 7, (32, 48))
    assert not _continues_sequence("scene", 9, (32, 48), "scene", 7, (32, 48))
    assert not _continues_sequence("scene", 8, (16, 48), "scene", 7, (32, 48))
    assert not _continues_sequence("other", 8, (32, 48), "scene", 7, (32, 48))


def test_snn_request_requires_steps_and_calibration_metadata() -> None:
    with pytest.raises(ValueError, match="simulation_steps"):
        _validate_snn_request("snn", 0)
    with pytest.raises(ValueError, match="calibrated checkpoint"):
        _validate_snn_request("snn", 4, {"model": {}}, "ann.pt")
    _validate_snn_request(
        "snn",
        4,
        {"batch_norm_folded": True, "snn_calibration_samples": 1},
        "snn.pt",
    )


def test_calibration_is_balanced_and_writes_clean_inference_checkpoint(tmp_path) -> None:
    root = tmp_path / "hdr"
    first = make_eventhdr(root / "scene_a")
    second = make_eventhdr(root / "scene_b")
    first.rename(first.with_name("a.h5"))
    second.rename(second.with_name("b.h5"))

    model_config = _model_config()
    source = tmp_path / "training.pt"
    model = ASGCNReconstructor(**model_config)
    atomic_torch_save(
        {
            "epoch": 7,
            "model": model.state_dict(),
            "model_config": model_config,
            "optimizer": {"large": "training-only"},
            "scaler": {"training-only": True},
            "history": [1, 2, 3],
            "rng_state": {"training-only": True},
        },
        source,
    )
    config = _eval_config(root, tmp_path / "eval")
    output = tmp_path / "snn.pt"
    calibrate(config, source, output, samples=2)

    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_type"] == "snn_inference"
    assert checkpoint["batch_norm_folded"] is True
    assert checkpoint["snn_calibration_samples"] == 2
    assert checkpoint["snn_calibration_sampling"]["selected_groups"] == 2
    assert set(checkpoint["snn_calibration_sampling"]["per_group"].values()) == {1}
    for training_key in ("optimizer", "scaler", "history", "rng_state", "config", "val"):
        assert training_key not in checkpoint

    config["eval"]["max_samples"] = 2
    result = evaluate(config, output, inference_mode="snn", simulation_steps=2)
    assert result["quality"]["frames"] == 2
    assert result["quality"]["micro"]["temporal_l1"] >= 0
    assert result["quality"]["per_scene"]["scene_a/a.h5"]["temporal_l1_frames"] == 1
    assert result["gpu_memory"] == {
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
    }


def test_public_snn_paths_reject_invalid_requests(tmp_path) -> None:
    with pytest.raises(ValueError, match="simulation_steps"):
        evaluate({}, tmp_path / "unused.pt", inference_mode="snn", simulation_steps=0)
    with pytest.raises(ValueError, match="calibration samples"):
        calibrate({}, tmp_path / "unused.pt", tmp_path / "out.pt", samples=0)

    root = tmp_path / "hdr"
    make_eventhdr(root)
    model_config = _model_config()
    uncalibrated = tmp_path / "ann.pt"
    atomic_torch_save(
        {
            "epoch": 1,
            "model": ASGCNReconstructor(**model_config).state_dict(),
            "model_config": model_config,
        },
        uncalibrated,
    )
    config = _eval_config(root, tmp_path / "eval")
    with pytest.raises(ValueError, match="calibrated checkpoint"):
        evaluate(config, uncalibrated, inference_mode="snn", simulation_steps=2)
    with pytest.raises(ValueError, match="calibrated checkpoint"):
        benchmark(
            config,
            uncalibrated,
            warmup=0,
            steps=1,
            inference_mode="snn",
            simulation_steps=2,
        )


def test_balanced_sampler_uses_eventhdr_files(tmp_path) -> None:
    root = tmp_path / "hdr"
    first = make_eventhdr(root / "a", frames=4)
    second = make_eventhdr(root / "b", frames=2)
    first.rename(first.with_name("a.h5"))
    second.rename(second.with_name("b.h5"))
    dataset = EventHDRDataset(root, max_events=8)
    indices = _balanced_sample_indices(dataset, limit=4, seed=5)
    paths = Counter(str(dataset.samples[index]["path"]) for index in indices)
    assert sorted(paths.values()) == [2, 2]
    dataset.close()
