from __future__ import annotations

import copy
import json
import shutil
from collections import Counter

import pytest
import torch

import asgcn_unet.engine as engine_module
import asgcn_unet.resources as resources_module
from asgcn_unet.data import EventHDRDataset, load_eventhdr_split_manifest
from asgcn_unet.engine import (
    _balanced_contiguous_indices,
    _balanced_sample_indices,
    _centralize_gradients,
    _continues_sequence,
    _dataset_content_fingerprint,
    _enforce_training_split_status,
    _macro_ssim,
    _model_state_sha256,
    _prefix_context_schedule,
    _representative_schedule,
    _resume_best_macro_ssim,
    _sample_event_counts,
    _sampling_summary,
    _set_inference_max_graph_edges,
    _validate_resume_best_pair,
    _validate_snn_request,
    benchmark,
    calibrate,
    evaluate,
    load_model_checkpoint,
)
from asgcn_unet.graph import PaperSplineConv
from asgcn_unet.model import ASGCNUNet
from asgcn_unet.utils import atomic_torch_save
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
        "architecture_version": 2,
        "graph_operator": "spline",
        "spline_backend": "torch",
        "spline_pseudo": "distance_over_radius",
        "spline_is_open": True,
        "hidden_dim": 4,
        "graph_layers": 1,
        "event_sampling_factor": 1,
        "graph_radius": 1.0,
        "graph_position_dims": 3,
        "graph_chunk_size": 16,
        "spline_kernel_size": 3,
        "spline_degree": 1,
        "spline_root_weight": True,
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
            "batch_size": 1,
            "num_workers": 0,
            "max_samples": 1,
            "save_predictions": 0,
            "output_dir": str(output_dir),
        },
    }


def _verified_preflight_gate() -> dict[str, object]:
    digest = "b" * 64
    return {
        "schema": "asgcn_preflight_verification_v1",
        "status": "verified",
        "report_eligible": True,
        "report": "runs/profile.json",
        "report_sha256": digest,
        "measurement_scope": {
            "name": "selected_top_density_training_steps",
            "topology_scope": "complete_eventhdr_training_split",
            "absolute_vram_guarantee": False,
            "statement": "Empirical GPU preflight for the recorded runtime.",
        },
        "config_sha256": digest,
        "data_sha256": digest,
        "source_tree_sha256": digest,
        "gpu": {"name": "synthetic-test-gpu"},
        "measured_steps": 1,
    }


def _recommit_calibration_metadata(checkpoint: dict) -> None:
    """Update the unsigned tensor commitment so semantic seal checks still run."""
    commitment = engine_module._calibration_commitment_sha256(
        checkpoint["calibration_protocol"],
        checkpoint["snn_calibration_samples"],
        checkpoint["snn_calibration_sampling"],
        engine_module._calibration_summary_commitment_core(
            checkpoint["snn_calibration_summary"]
        ),
    )
    checkpoint["model"]["calibration_commitment_digest"] = torch.tensor(
        list(bytes.fromhex(commitment)), dtype=torch.uint8
    )
    checkpoint["model"]["calibration_commitment_sealed"] = torch.tensor(True)
    checkpoint["snn_calibration_summary"][
        "calibration_commitment_sha256"
    ] = commitment
    checkpoint["snn_calibration_summary"]["commitment_sealed"] = True
    checkpoint["model_state_sha256"] = _model_state_sha256(checkpoint["model"])


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
    indices = _balanced_contiguous_indices(dataset, limit=6, seed=3, require_all_groups=True)
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
        _balanced_contiguous_indices(dataset, limit=2, seed=3, require_all_groups=True)


def test_prefix_context_replays_unscored_predecessors() -> None:
    dataset = _GroupedIndexDataset()
    schedule, score_positions = _prefix_context_schedule(dataset, [6, 7, 12])
    assert schedule == [0, 1, 2, 3, 4, 5, 6, 7, 12]
    assert score_positions == {6, 7, 8}
    bounded, bounded_scores = _prefix_context_schedule(dataset, [6, 7], max_context_frames=2)
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


def test_checkpoint_loader_rejects_legacy_graph_architecture(tmp_path) -> None:
    legacy = tmp_path / "legacy.pt"
    atomic_torch_save(
        {
            "model": {},
            "model_config": {"hidden_dim": 4, "graph_layers": 1},
        },
        legacy,
    )
    with pytest.raises(ValueError, match="architecture_version"):
        load_model_checkpoint(legacy, torch.device("cpu"), _model_config())


def test_resume_rejects_unrelated_historical_best_checkpoint() -> None:
    model_digest = "a" * 64
    resume = {
        "epoch": 2,
        "model_config": _model_config(),
        "validation_protocol": {"version": 2},
        "training_protocol": {"version": 1},
        "paper_core_version": 2,
        "best_metric": "macro_ssim",
        "best_ssim": 0.7,
        "best_model_state_sha256": model_digest,
        "val": {},
    }
    best = {
        **resume,
        "epoch": 1,
        "model_state_sha256": model_digest,
        "val": {"macro": {"ssim": 0.7}},
    }
    _validate_resume_best_pair(resume, best)

    best["validation_protocol"] = {"version": 99}
    with pytest.raises(ValueError, match="different validation protocol"):
        _validate_resume_best_pair(resume, best)


def test_training_rejects_nonfinal_split(tmp_path) -> None:
    manifest = tmp_path / "split.json"
    split = {
        "status": "provisional",
        "train_files": ["train.h5"],
        "val_files": ["val.h5"],
    }
    manifest.write_text(json.dumps(split), encoding="utf-8")
    config = {
        "dataset": {"split_manifest": str(manifest)},
        "train": {},
    }
    with pytest.raises(ValueError, match="status='final'"):
        _enforce_training_split_status(config)

    split["status"] = "final"
    manifest.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match="split_schema"):
        _enforce_training_split_status(config)

    split = {
        "status": "final",
        "split_schema": "official_separate_roots_v1",
        "group_semantics": "h5_sequence_file_not_physical_scene",
        "train_files": ["train.h5"],
        "val_files": ["val.h5"],
    }
    manifest.write_text(json.dumps(split), encoding="utf-8")
    _enforce_training_split_status(config)


def test_split_manifest_allows_same_names_in_distinct_official_roots(tmp_path) -> None:
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "final",
                "split_schema": "official_separate_roots_v1",
                "group_semantics": "h5_sequence_file_not_physical_scene",
                "train_files": ["shared.h5"],
                "val_files": ["shared.h5"],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_eventhdr_split_manifest(manifest)
    assert loaded["train_file_to_group"]["shared.h5"] == "official-train-h5::shared.h5"
    assert loaded["val_file_to_group"]["shared.h5"] == "official-eval-h5::shared.h5"


def test_sequence_continuity_requires_adjacent_index_and_matching_shape() -> None:
    assert _continues_sequence("scene", 8, (32, 48), "scene", 7, (32, 48))
    assert not _continues_sequence("scene", 9, (32, 48), "scene", 7, (32, 48))
    assert not _continues_sequence("scene", 8, (16, 48), "scene", 7, (32, 48))
    assert not _continues_sequence("other", 8, (32, 48), "scene", 7, (32, 48))


def test_event_count_fallback_uses_retained_tensor_for_custom_datasets() -> None:
    sample = {"events": torch.zeros((7, 4))}
    assert _sample_event_counts(sample) == (7, 7)
    sample["metadata"] = None
    assert _sample_event_counts(sample) == (7, 7)
    sample["metadata"] = {"raw_event_count": "12"}
    assert _sample_event_counts(sample) == (12, 7)
    sample["metadata"] = {"raw_event_count": 3}
    assert _sample_event_counts(sample) == (7, 7)


def test_gradient_centralization_respects_spline_output_axis() -> None:
    class MixedWeights(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spline = PaperSplineConv(2, 3, kernel_size=2)
            self.linear = torch.nn.Linear(4, 3, bias=False)

    model = MixedWeights()
    model.spline.weight.grad = torch.arange(
        model.spline.weight.numel(), dtype=torch.float32
    ).reshape_as(model.spline.weight)
    model.spline.root.grad = torch.arange(
        model.spline.root.numel(), dtype=torch.float32
    ).reshape_as(model.spline.root)
    model.linear.weight.grad = torch.arange(
        model.linear.weight.numel(), dtype=torch.float32
    ).reshape_as(model.linear.weight)

    _centralize_gradients(model)

    torch.testing.assert_close(
        model.spline.weight.grad.mean(dim=(0, 1)),
        torch.zeros(model.spline.out_channels),
    )
    torch.testing.assert_close(
        model.spline.root.grad.mean(dim=0),
        torch.zeros(model.spline.out_channels),
    )
    torch.testing.assert_close(
        model.linear.weight.grad.mean(dim=1),
        torch.zeros(model.linear.out_features),
    )


def test_snn_request_requires_steps_and_calibration_metadata() -> None:
    with pytest.raises(ValueError, match="simulation_steps"):
        _validate_snn_request("snn", 0)
    with pytest.raises(ValueError, match="integer"):
        _validate_snn_request("snn", 1.5)
    with pytest.raises(ValueError, match="calibrated checkpoint"):
        _validate_snn_request("snn", 4, {"model": {}}, "ann.pt")
    _validate_snn_request(
        "snn",
        4,
        {
            "checkpoint_type": "snn_inference",
            "batch_norm_folded": True,
            "snn_calibration_samples": 1,
            "snn_calibration_valid_samples": 1,
            "paper_core_version": 2,
            "parameter_normalized": True,
        },
        "snn.pt",
    )


def test_ann_request_rejects_parameter_normalized_snn_checkpoint() -> None:
    _validate_snn_request("ann", 16, {"checkpoint_type": "ann_inference"}, "ann.pt")
    with pytest.raises(ValueError, match="ANN checkpoint"):
        _validate_snn_request(
            "ann",
            16,
            {"checkpoint_type": "snn_inference"},
            "snn.pt",
        )
    with pytest.raises(ValueError, match=r"Eq\. \(6\)-normalized"):
        _validate_snn_request("ann", 16, {"parameter_normalized": True}, "snn.pt")


def test_inference_edge_guard_override_only_raises_runtime_limit() -> None:
    model = ASGCNUNet(**_model_config())
    assert _set_inference_max_graph_edges(model, None) == {
        "configured_max_graph_edges": 2_000_000,
        "requested_max_graph_edges_override": None,
        "effective_max_graph_edges": 2_000_000,
    }
    assert _set_inference_max_graph_edges(model, 3_000_000) == {
        "configured_max_graph_edges": 2_000_000,
        "requested_max_graph_edges_override": 3_000_000,
        "effective_max_graph_edges": 3_000_000,
    }
    assert model.max_graph_edges == 3_000_000

    for invalid in (True, 0, -1, 2_000_000.0):
        fresh = ASGCNUNet(**_model_config())
        with pytest.raises(ValueError, match="positive integer"):
            _set_inference_max_graph_edges(fresh, invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="greater than or equal"):
        _set_inference_max_graph_edges(ASGCNUNet(**_model_config()), 1_999_999)

    unbounded_config = _model_config()
    unbounded_config["max_graph_edges"] = None
    with pytest.raises(ValueError, match="unbounded configured guard"):
        _set_inference_max_graph_edges(ASGCNUNet(**unbounded_config), 3_000_000)


def test_calibration_zero_wall_interval_reports_unavailable_throughput(
    tmp_path, monkeypatch
) -> None:
    """CPU debug calibration must tolerate two snapshots in one clock tick."""
    root = tmp_path / "hdr"
    make_eventhdr(root)
    config = _eval_config(root, tmp_path / "unused-eval")
    config["calibration"] = {"batch_size": 2, "num_workers": 0}
    model_state = ASGCNUNet(**config["model"]).state_dict()
    source = tmp_path / "debug-ann.pt"
    atomic_torch_save(
        {
            "checkpoint_type": "training",
            "epoch": 1,
            "model": model_state,
            "model_state_sha256": _model_state_sha256(model_state),
            "model_config": config["model"],
            "paper_core_version": engine_module.PAPER_CORE_VERSION,
        },
        source,
    )
    snapshots = []
    collect_snapshot = resources_module.collect_runtime_resources

    def same_clock_tick(**kwargs):
        snapshot = collect_snapshot(**kwargs)
        snapshot["monotonic_seconds"] = 760261.375
        snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(resources_module, "collect_runtime_resources", same_clock_tick)
    monkeypatch.setattr(engine_module, "collect_runtime_resources", same_clock_tick)
    output = tmp_path / "debug-snn.pt"
    calibrate(config, source, output, allow_unsealed_calibration=True)
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)

    assert len(snapshots) == 2
    report = checkpoint["execution_report"]
    performance = checkpoint["calibration_performance"]
    assert report["resources"]["monotonic_seconds"] == 760261.375
    assert performance["resources_after"]["monotonic_seconds"] == 760261.375
    assert performance["wall_seconds"] == 0.0
    assert performance["frames_per_second"] is None
    assert performance["process_cpu_percent"] is None
    assert performance["process_cpu_allocation_percent"] is None
    assert performance["process_cpu_utilization_status"] == "unavailable_zero_wall_interval"
    assert "clock resolution" in performance["process_cpu_utilization_note"]
    assert performance["frames"] == checkpoint["snn_calibration_samples"]
    assert performance["frames"] == report["data"]["dataset_size"] > 0
    assert report["data"]["used_ratio"] == 1.0
    assert checkpoint["report_eligible"] is False
    json.dumps(performance, allow_nan=False)


def test_calibration_is_balanced_and_writes_clean_inference_checkpoint(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "hdr"
    first = make_eventhdr(root / "scene_a")
    second = make_eventhdr(root / "scene_b")
    first.rename(first.with_name("a.h5"))
    second.rename(second.with_name("b.h5"))

    model_config = _model_config()
    source = tmp_path / "training.pt"
    model = ASGCNUNet(**model_config)
    with torch.no_grad():
        dead_layer = model.encoder.layers[0]
        dead_layer.weight[..., 0].zero_()
        if dead_layer.root is not None:
            dead_layer.root[:, 0].zero_()
        if dead_layer.bias is not None:
            dead_layer.bias[0] = -1.0
        dead_layer.norm.running_mean[0] = 0.0
        dead_layer.norm.running_var[0] = 1.0
        dead_layer.norm.weight[0] = 1.0
        dead_layer.norm.bias[0] = 0.0
    model_state = model.state_dict()
    # Exercise compatibility with the architecture-v2 ANN state written before
    # raw maxima, effective scales, and dead masks became separate buffers.
    model_state["encoder.layers.0.activation_max"] = model_state.pop(
        "encoder.layers.0.calibration_activation_max"
    )
    model_state.pop("encoder.layers.0.normalization_scale")
    model_state.pop("encoder.layers.0.dead_channel_mask")
    atomic_torch_save(
        {
            "checkpoint_type": "training",
            "epoch": 7,
            "model": model_state,
            "model_state_sha256": _model_state_sha256(model_state),
            "model_config": model_config,
            "paper_core_version": 2,
            "optimizer": {"large": "training-only"},
            "scaler": {"training-only": True},
            "history": [1, 2, 3],
            "rng_state": {"training-only": True},
        },
        source,
    )
    config = _eval_config(root, tmp_path / "eval")
    config["train"] = {
        "batch_size": 1,
        "num_workers": 0,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }
    loader_calls = []
    original_data_loader = engine_module._data_loader

    def tracked_data_loader(dataset, *args, **kwargs):
        loader_calls.append((list(dataset.indices), args, kwargs))
        return original_data_loader(dataset, *args, **kwargs)

    monkeypatch.setattr(engine_module, "_data_loader", tracked_data_loader)
    output = tmp_path / "snn.pt"
    with pytest.raises(ValueError, match="not clean ann_inference"):
        calibrate(config, source, output, samples=2)
    calibrate(
        config,
        source,
        output,
        samples=2,
        allow_unsealed_calibration=True,
    )
    assert len(loader_calls) == 1
    selected_indices, positional, loader_options = loader_calls[0]
    assert selected_indices == sorted(selected_indices)
    assert positional == ()
    assert loader_options["batch_size"] == 1
    assert loader_options["num_workers"] == 0
    assert loader_options["shuffle"] is False
    assert loader_options["persistent_workers"] is False
    assert loader_options["prefetch_factor"] == 2
    with pytest.raises(FileExistsError, match="already exists"):
        calibrate(config, source, output, samples=1)
    monkeypatch.setattr(engine_module, "_data_loader", original_data_loader)

    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_type"] == "snn_inference"
    assert checkpoint["model_state_sha256"] == _model_state_sha256(checkpoint["model"])
    assert checkpoint["batch_norm_folded"] is True
    assert checkpoint["paper_core_version"] == 2
    assert checkpoint["parameter_normalized"] is True
    assert checkpoint["snn_calibration_samples"] == 2
    assert checkpoint["snn_calibration_valid_samples"] == 2
    assert checkpoint["snn_calibration_summary"]["minimum_valid_samples"] == 2
    assert checkpoint["snn_calibration_summary"]["valid_samples_per_layer"] == [2]
    raw_max = checkpoint["model"]["encoder.layers.0.calibration_activation_max"]
    normalization_scale = checkpoint["model"]["encoder.layers.0.normalization_scale"]
    dead_mask = checkpoint["model"]["encoder.layers.0.dead_channel_mask"]
    assert raw_max[0].item() == 0.0
    assert normalization_scale[0].item() == 1.0
    assert dead_mask[0].item() is True
    assert checkpoint["snn_calibration_summary"]["dead_channels_per_layer"] == [
        int(dead_mask.sum().item())
    ]
    restored, restored_metadata = load_model_checkpoint(
        output,
        torch.device("cpu"),
        model_config,
    )
    assert restored_metadata["checkpoint_type"] == "snn_inference"
    assert restored.encoder.calibration_summary()["dead_channels_per_layer"] == [
        int(dead_mask.sum().item())
    ]
    torch.testing.assert_close(
        restored.encoder.layers[0].normalization_scale,
        normalization_scale,
    )
    assert checkpoint["snn_calibration_sampling"]["selected_groups"] == 2
    assert set(checkpoint["snn_calibration_sampling"]["per_group"].values()) == {1}
    assert checkpoint["calibration_protocol"]["sealed"] is False
    assert "source checkpoint is not clean ann_inference" in checkpoint[
        "calibration_protocol"
    ]["unsealed_reasons"]
    assert checkpoint["source_checkpoint"] == "$EXTERNAL/training.pt"
    for training_key in ("optimizer", "scaler", "history", "rng_state", "config", "val"):
        assert training_key not in checkpoint

    config["eval"]["max_samples"] = 2
    with pytest.raises(ValueError, match="Checkpoint reporting protocol is not sealed"):
        evaluate(config, output, inference_mode="snn", simulation_steps=2)
    with pytest.raises(ValueError, match="Checkpoint reporting protocol is not sealed"):
        benchmark(
            config,
            output,
            warmup=0,
            steps=2,
            inference_mode="snn",
            simulation_steps=2,
        )
    result = evaluate(
        config,
        output,
        inference_mode="snn",
        simulation_steps=2,
        allow_unsealed_checkpoint_for_non_reporting=True,
    )
    assert result["report_eligible"] is False
    assert result["quality"]["frames"] == 2
    assert result["quality"]["micro"]["temporal_l1"] >= 0
    assert result["quality"]["per_scene"]["scene_a/a.h5"]["temporal_l1_frames"] == 1
    assert result["gpu_memory"] == {
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
    }
    assert result["snn_dynamics"] == "literal_eq15"
    assert 0.0 <= result["graph_topology"]["isolate_ratio"] <= 1.0
    timing = benchmark(
        config,
        output,
        warmup=0,
        steps=2,
        inference_mode="snn",
        simulation_steps=2,
        max_graph_edges_override=3_000_000,
        allow_unsealed_checkpoint_for_non_reporting=True,
    )
    assert timing["snn_dynamics"] == "literal_eq15"
    assert len(timing["layer_firing_rates"]) == 1
    assert timing["mean_firing_rate"] == pytest.approx(timing["layer_firing_rates"][0])
    benchmark_guard = timing["benchmark_protocol"]["execution"]["contract"][
        "graph_edge_guard"
    ]
    assert benchmark_guard == {
        "configured_max_graph_edges": 2_000_000,
        "requested_max_graph_edges_override": 3_000_000,
        "effective_max_graph_edges": 3_000_000,
    }
    assert timing["graph_edge_guard"] == benchmark_guard
    standard_timing = benchmark(
        config,
        output,
        warmup=0,
        steps=1,
        inference_mode="snn",
        simulation_steps=2,
        snn_dynamics="standard_if",
        allow_unsealed_checkpoint_for_non_reporting=True,
    )
    assert standard_timing["snn_dynamics"] == "standard_if"

    tampered = torch.load(output, map_location="cpu", weights_only=False)
    tampered["model"]["encoder.layers.0.calibration_activation_max"][0] = float("nan")
    tampered["model_state_sha256"] = _model_state_sha256(tampered["model"])
    nonfinite_path = tmp_path / "snn_nonfinite.pt"
    torch.save(tampered, nonfinite_path)
    with pytest.raises(ValueError, match="non-finite state"):
        evaluate(config, nonfinite_path, inference_mode="snn", simulation_steps=2)

    changed_dead_mask = torch.load(output, map_location="cpu", weights_only=False)
    changed_dead_mask["model"]["encoder.layers.0.dead_channel_mask"][0] = False
    changed_dead_mask["model_state_sha256"] = _model_state_sha256(
        changed_dead_mask["model"]
    )
    changed_dead_mask_path = tmp_path / "snn_changed_dead_mask.pt"
    atomic_torch_save(changed_dead_mask, changed_dead_mask_path)
    with pytest.raises(ValueError, match="dead-channel mask"):
        evaluate(
            config,
            changed_dead_mask_path,
            inference_mode="snn",
            simulation_steps=2,
            allow_unsealed_checkpoint_for_non_reporting=True,
        )

    changed_dead_summary = torch.load(output, map_location="cpu", weights_only=False)
    changed_dead_summary["snn_calibration_summary"][
        "dead_channels_per_layer"
    ][0] += 1
    _recommit_calibration_metadata(changed_dead_summary)
    changed_dead_summary_path = tmp_path / "snn_changed_dead_summary.pt"
    atomic_torch_save(changed_dead_summary, changed_dead_summary_path)
    with pytest.raises(ValueError, match="summary differs from layer state"):
        evaluate(
            config,
            changed_dead_summary_path,
            inference_mode="snn",
            simulation_steps=2,
            allow_unsealed_checkpoint_for_non_reporting=True,
        )

    inconsistent = torch.load(output, map_location="cpu", weights_only=False)
    inconsistent["snn_calibration_summary"]["valid_samples_per_layer"] = [999]
    inconsistent_path = tmp_path / "snn_inconsistent.pt"
    torch.save(inconsistent, inconsistent_path)
    with pytest.raises(ValueError, match="invalid calibration summary"):
        evaluate(config, inconsistent_path, inference_mode="snn", simulation_steps=2)

    wrong_threshold = torch.load(output, map_location="cpu", weights_only=False)
    wrong_threshold["model"]["encoder.layers.0.threshold"][0] = 0.5
    wrong_threshold["model_state_sha256"] = _model_state_sha256(wrong_threshold["model"])
    wrong_threshold_path = tmp_path / "snn_wrong_threshold.pt"
    torch.save(wrong_threshold, wrong_threshold_path)
    with pytest.raises(ValueError, match="unit threshold"):
        evaluate(config, wrong_threshold_path, inference_mode="snn", simulation_steps=2)
    with pytest.raises(ValueError, match="ANN checkpoint"):
        evaluate(config, output, inference_mode="ann")


def test_calibration_seals_ann_training_data_transform_manifest_and_source(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    val_root = tmp_path / "hdr-val"
    make_eventhdr(val_root)
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "final",
                "split_schema": "official_separate_roots_v1",
                "group_semantics": "h5_sequence_file_not_physical_scene",
                "train_files": ["test.h5"],
                "val_files": ["test.h5"],
            }
        ),
        encoding="utf-8",
    )
    config = _eval_config(root, tmp_path / "eval")
    config["eval"]["max_samples"] = None
    config["dataset"]["expected_file_count"] = 1
    config["dataset"]["val_root"] = str(val_root)
    config["dataset"]["split_manifest"] = str(manifest_path)
    config["train"] = {
        "batch_size": 1,
        "num_workers": 0,
        "max_train_samples": None,
        "max_val_samples": None,
        "validation_context_frames": 0,
        "validate_every": None,
        "epochs": 7,
    }
    dataset = engine_module.build_dataset(config["dataset"], split="train")
    val_dataset = engine_module.build_dataset(config["dataset"], split="val")
    try:
        train_content = _dataset_content_fingerprint(dataset)
        calibration_sample_count = len(dataset)
        validation_sample_count = len(val_dataset)
        validation_sampling = engine_module._sampling_summary(
            val_dataset, list(range(validation_sample_count))
        )
        validation_sampling.update(
            {
                "context_policy": "none_non_recurrent",
                "max_context_frames_per_group": 0,
                "context_samples": 0,
                "forward_samples": validation_sample_count,
            }
        )
        validation_protocol = engine_module._validation_protocol(
            config,
            validation_sampling,
            dataset,
            val_dataset,
            {},
        )
    finally:
        dataset.close()
        val_dataset.close()

    model_config = _model_config()
    model_state = ASGCNUNet(**model_config).state_dict()
    training_config = engine_module._public_config(config)
    training_protocol = engine_module._training_protocol(
        config, torch.device("cpu")
    )
    preflight_gate = _verified_preflight_gate()
    preflight_gate["config_sha256"] = engine_module._canonical_sha256(training_config)
    preflight_gate["data_sha256"] = train_content["sha256"]
    preflight_gate["source_tree_sha256"] = training_protocol["source"][
        "source_tree_sha256"
    ]
    source = tmp_path / "best.pt"
    atomic_torch_save(
        {
            "checkpoint_type": "ann_inference",
            "epoch": 7,
            "model": model_state,
            "model_state_sha256": _model_state_sha256(model_state),
            "model_config": model_config,
            "paper_core_version": 2,
            "preflight_gate": preflight_gate,
            "training_config": training_config,
            "val": {
                "frames": validation_sample_count,
                "macro": {"ssim": 0.75},
            },
            "best_ssim": 0.75,
            "best_metric": "macro_ssim",
            "checkpoint_selection": "single_final_epoch",
            "validation_protocol": validation_protocol,
            "training_protocol": training_protocol,
            "terminal_validation_state": {
                "planned_epoch": 7,
                "completed": True,
                "completed_epoch": 7,
            },
        },
        source,
    )

    repeated_validation_ann = torch.load(
        source, map_location="cpu", weights_only=False
    )
    repeated_validation_ann["training_protocol"]["terminal_validation"] = None
    repeated_validation_ann["training_protocol"]["validate_every"] = 1
    repeated_validation_ann["training_protocol"][
        "checkpoint_selection"
    ] = "best_validation_macro_ssim"
    repeated_validation_ann["terminal_validation_state"] = None
    repeated_validation_ann["checkpoint_selection"] = "best_validation_macro_ssim"
    repeated_validation_ann["validation_protocol"]["selection_metric"] = "macro_ssim"
    repeated_validation_ann["training_config"]["train"]["validate_every"] = 1
    repeated_validation_ann["preflight_gate"]["config_sha256"] = (
        engine_module._canonical_sha256(repeated_validation_ann["training_config"])
    )
    repeated_validation_path = tmp_path / "repeated-eventhdr-validation-ann.pt"
    atomic_torch_save(repeated_validation_ann, repeated_validation_path)
    with pytest.raises(ValueError, match="EventHDR reporting requires"):
        evaluate(config, repeated_validation_path, inference_mode="ann")

    inconsistent_validation_config = torch.load(
        source, map_location="cpu", weights_only=False
    )
    inconsistent_validation_config["training_config"]["train"][
        "validate_every"
    ] = 1
    inconsistent_validation_config["preflight_gate"]["config_sha256"] = (
        engine_module._canonical_sha256(
            inconsistent_validation_config["training_config"]
        )
    )
    inconsistent_validation_path = tmp_path / "inconsistent-validation-config.pt"
    atomic_torch_save(inconsistent_validation_config, inconsistent_validation_path)
    with pytest.raises(ValueError, match="validate_every differs"):
        evaluate(config, inconsistent_validation_path, inference_mode="ann")

    missing_manifest_ann = torch.load(source, map_location="cpu", weights_only=False)
    missing_manifest_ann["validation_protocol"]["split_manifest"] = None
    missing_manifest_ann_path = tmp_path / "missing-manifest-ann.pt"
    atomic_torch_save(missing_manifest_ann, missing_manifest_ann_path)
    with pytest.raises(ValueError, match="split manifest"):
        evaluate(config, missing_manifest_ann_path, inference_mode="ann")

    partial_validation_ann = torch.load(source, map_location="cpu", weights_only=False)
    partial_validation_ann["validation_protocol"]["sampling"] = {}
    partial_validation_ann_path = tmp_path / "partial-validation-ann.pt"
    atomic_torch_save(partial_validation_ann, partial_validation_ann_path)
    with pytest.raises(ValueError, match="full validation sampling"):
        evaluate(config, partial_validation_ann_path, inference_mode="ann")

    capped_validation_ann = torch.load(source, map_location="cpu", weights_only=False)
    capped_validation_ann["training_config"]["train"]["max_val_samples"] = 1
    capped_validation_ann_path = tmp_path / "capped-validation-ann.pt"
    atomic_torch_save(capped_validation_ann, capped_validation_ann_path)
    with pytest.raises(ValueError, match="partial validation sample limit"):
        evaluate(config, capped_validation_ann_path, inference_mode="ann")

    with pytest.raises(ValueError, match="every EventHDR training sample exactly once"):
        calibrate(config, source, tmp_path / "partial-rejected.pt", samples=1)
    partial_path = tmp_path / "partial-non-reporting.pt"
    calibrate(
        config,
        source,
        partial_path,
        samples=1,
        allow_unsealed_calibration=True,
    )
    partial = torch.load(partial_path, map_location="cpu", weights_only=False)
    assert partial["report_eligible"] is False
    assert any(
        "every EventHDR training sample exactly once" in reason
        for reason in partial["report_ineligible_reasons"]
    )
    with pytest.raises(ValueError, match="Checkpoint reporting protocol is not sealed"):
        evaluate(config, partial_path, inference_mode="snn", simulation_steps=2)

    relabeled_partial = torch.load(partial_path, map_location="cpu", weights_only=False)
    relabeled_protocol = relabeled_partial["calibration_protocol"]
    relabeled_protocol["sealed"] = True
    relabeled_protocol["unsealed_reasons"] = []
    relabeled_partial["report_eligible"] = True
    relabeled_partial["report_ineligible_reasons"] = []
    relabeled_identity = copy.deepcopy(relabeled_protocol["selected_sample_ids"][0])
    relabeled_identity["dataset_index"] = 0
    relabeled_identity["sequence_index"] = 0
    relabeled_protocol["selected_sample_ids"] = [relabeled_identity]
    relabeled_sampling = relabeled_partial["snn_calibration_sampling"]
    relabeled_sampling["selected"] = [copy.deepcopy(relabeled_identity)]
    relabeled_sampling["available_per_group"] = copy.deepcopy(
        relabeled_sampling["per_group"]
    )
    relabeled_sampling["available_groups"] = relabeled_sampling["selected_groups"]
    relabeled_path = tmp_path / "relabeled-partial.pt"
    atomic_torch_save(relabeled_partial, relabeled_path)
    with pytest.raises(ValueError, match="persistent tensor commitment"):
        evaluate(config, relabeled_path, inference_mode="snn", simulation_steps=2)

    forced_unsealed_path = tmp_path / "forced-unsealed-full.pt"
    calibrate(
        config,
        source,
        forced_unsealed_path,
        allow_unsealed_calibration=True,
    )
    forced_unsealed = torch.load(
        forced_unsealed_path, map_location="cpu", weights_only=False
    )
    assert forced_unsealed["report_eligible"] is False
    assert forced_unsealed["calibration_protocol"]["sealed"] is False
    assert forced_unsealed["report_ineligible_reasons"][0] == (
        "explicit unsealed calibration override requested"
    )

    output = tmp_path / "best_snn.pt"
    calibrate(config, source, output)
    sealed = torch.load(output, map_location="cpu", weights_only=False)[
        "calibration_protocol"
    ]
    assert sealed["sealed"] is True
    assert sealed["unsealed_reasons"] == []
    assert sealed["dataset_content"] == train_content
    assert sealed["dataset_content_sha256"] == train_content["sha256"]
    assert len(sealed["source_ann_model_sha256"]) == 64
    assert len(sealed["source_ann_checkpoint_sha256"]) == 64
    assert len(sealed["source_ann_training_protocol_sha256"]) == 64
    assert len(sealed["source_ann_validation_protocol_sha256"]) == 64
    assert len(sealed["source_ann_reporting_contract_sha256"]) == 64
    assert sealed["source_ann_training_protocol"]["sha256"] == sealed[
        "source_ann_training_protocol_sha256"
    ]
    assert sealed["source_ann_validation_protocol"]["sha256"] == sealed[
        "source_ann_validation_protocol_sha256"
    ]
    assert sealed["source_ann_reporting_contract"]["sha256"] == sealed[
        "source_ann_reporting_contract_sha256"
    ]
    assert sealed["source_preflight_gate"]["contract"]["report_eligible"] is True
    assert len(sealed["selected_sample_ids"]) == calibration_sample_count
    assert [item["dataset_index"] for item in sealed["selected_sample_ids"]] == list(
        range(calibration_sample_count)
    )
    assert str(tmp_path) not in json.dumps(sealed)

    full_checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    transplanted_metadata = copy.deepcopy(full_checkpoint)
    transplanted_metadata["model"] = copy.deepcopy(partial["model"])
    transplanted_metadata["model_state_sha256"] = partial["model_state_sha256"]
    transplanted_path = tmp_path / "full-metadata-partial-tensors.pt"
    atomic_torch_save(transplanted_metadata, transplanted_path)
    with pytest.raises(ValueError, match="persistent attempted calibration state"):
        evaluate(
            config,
            transplanted_path,
            inference_mode="snn",
            simulation_steps=2,
        )

    changed_snn_epoch = torch.load(output, map_location="cpu", weights_only=False)
    changed_snn_epoch["epoch"] = 6
    changed_snn_epoch_path = tmp_path / "changed-snn-epoch.pt"
    atomic_torch_save(changed_snn_epoch, changed_snn_epoch_path)
    with pytest.raises(ValueError, match="SNN checkpoint epoch differs"):
        evaluate(
            config,
            changed_snn_epoch_path,
            inference_mode="snn",
            simulation_steps=2,
        )

    changed_snn_model = torch.load(output, map_location="cpu", weights_only=False)
    changed_snn_model["model_config"]["graph_radius"] = 0.75
    changed_snn_model_path = tmp_path / "changed-snn-model-config.pt"
    atomic_torch_save(changed_snn_model, changed_snn_model_path)
    changed_snn_eval_config = copy.deepcopy(config)
    changed_snn_eval_config["model"]["graph_radius"] = 0.75
    with pytest.raises(ValueError, match="SNN checkpoint model config differs"):
        evaluate(
            changed_snn_eval_config,
            changed_snn_model_path,
            inference_mode="snn",
            simulation_steps=2,
        )

    changed_dead_summary = torch.load(output, map_location="cpu", weights_only=False)
    changed_dead_summary["snn_calibration_summary"][
        "dead_channels_per_layer"
    ][0] += 1
    _recommit_calibration_metadata(changed_dead_summary)
    changed_dead_summary_path = tmp_path / "changed-dead-summary.pt"
    atomic_torch_save(changed_dead_summary, changed_dead_summary_path)
    with pytest.raises(ValueError, match="summary differs from layer state"):
        evaluate(
            config,
            changed_dead_summary_path,
            inference_mode="snn",
            simulation_steps=2,
        )

    partial_quality_config = copy.deepcopy(config)
    partial_quality_config["eval"]["max_samples"] = 1
    partial_quality_config["eval"]["output_dir"] = str(tmp_path / "partial-quality")
    with pytest.raises(ValueError, match="eval.max_samples=null"):
        evaluate(
            partial_quality_config,
            output,
            inference_mode="snn",
            simulation_steps=2,
        )
    partial_quality_result = evaluate(
        partial_quality_config,
        output,
        inference_mode="snn",
        simulation_steps=2,
        allow_unsealed_checkpoint_for_non_reporting=True,
    )
    assert partial_quality_result["report_eligible"] is False
    assert any(
        "quality evaluation uses eval.max_samples" in reason
        for reason in partial_quality_result["report_ineligible_reasons"]
    )

    evaluation_source = copy.deepcopy(engine_module._current_source_contract())
    evaluation_source["source_tree_sha256"] = "e" * 64
    monkeypatch.setattr(
        engine_module, "_current_source_contract", lambda: evaluation_source
    )
    reporting_result = evaluate(
        config,
        output,
        inference_mode="snn",
        simulation_steps=2,
        max_graph_edges_override=3_000_000,
    )
    assert reporting_result["report_eligible"] is True
    assert reporting_result["evaluation_protocol"]["source"]["contract"] == evaluation_source
    lineage = reporting_result["evaluation_protocol"]["checkpoint"]
    assert lineage["calibration_protocol"]["sha256"] == engine_module._canonical_sha256(
        sealed
    )
    assert lineage["preflight_gate"]["contract"]["status"] == "verified"
    execution = reporting_result["evaluation_protocol"]["execution"]["contract"]
    assert execution == {
        "inference_mode": "snn",
        "simulation_steps": 2,
        "snn_dynamics": "literal_eq15",
        "graph_edge_guard": {
            "configured_max_graph_edges": 2_000_000,
            "requested_max_graph_edges_override": 3_000_000,
            "effective_max_graph_edges": 3_000_000,
        },
        "scope": "full_dataset_quality_evaluation",
        "batching": {
            "policy": "single_sequence_reference",
            "physical_batch_size_limit": 1,
            "num_workers": 0,
            "full_coverage": True,
            "latency_scope": "physical_batch_completion_not_amortized",
        },
    }
    assert reporting_result["graph_edge_guard"] == execution["graph_edge_guard"]

    for field, message in (
        ("dataset_transform", "calibration dataset transform"),
        ("split_manifest", "calibration split manifest"),
        ("selected_sample_ids", "calibration sample identities"),
        ("runtime", "calibration runtime identity"),
    ):
        tampered = torch.load(output, map_location="cpu", weights_only=False)
        tampered["calibration_protocol"].pop(field)
        _recommit_calibration_metadata(tampered)
        tampered_path = tmp_path / f"missing-{field}.pt"
        atomic_torch_save(tampered, tampered_path)
        with pytest.raises(ValueError, match=message):
            evaluate(config, tampered_path, inference_mode="snn", simulation_steps=2)

    changed_source = torch.load(output, map_location="cpu", weights_only=False)
    changed_source["calibration_protocol"]["calibration_source"] = copy.deepcopy(
        changed_source["calibration_protocol"]["training_source"]
    )
    changed_source["calibration_protocol"]["calibration_source"][
        "source_tree_sha256"
    ] = "f" * 64
    _recommit_calibration_metadata(changed_source)
    changed_source_path = tmp_path / "changed-calibration-source.pt"
    atomic_torch_save(changed_source, changed_source_path)
    with pytest.raises(ValueError, match="calibration source differs"):
        evaluate(config, changed_source_path, inference_mode="snn", simulation_steps=2)

    changed_content = torch.load(output, map_location="cpu", weights_only=False)
    changed_content["calibration_protocol"]["dataset_content"]["sha256"] = "f" * 64
    changed_content["calibration_protocol"]["dataset_content_sha256"] = "f" * 64
    _recommit_calibration_metadata(changed_content)
    changed_content_path = tmp_path / "changed-calibration-content.pt"
    atomic_torch_save(changed_content, changed_content_path)
    with pytest.raises(ValueError, match="source ANN training data"):
        evaluate(config, changed_content_path, inference_mode="snn", simulation_steps=2)

    changed_source_manifest = torch.load(output, map_location="cpu", weights_only=False)
    protocol = changed_source_manifest["calibration_protocol"]
    source_validation = protocol["source_ann_validation_protocol"]
    source_validation["contract"]["split_manifest"] = None
    source_validation["sha256"] = engine_module._canonical_sha256(
        source_validation["contract"]
    )
    protocol["source_ann_validation_protocol_sha256"] = source_validation["sha256"]
    _recommit_calibration_metadata(changed_source_manifest)
    changed_source_manifest_path = tmp_path / "changed-source-manifest.pt"
    atomic_torch_save(changed_source_manifest, changed_source_manifest_path)
    with pytest.raises(ValueError, match="source ANN split manifest"):
        evaluate(
            config,
            changed_source_manifest_path,
            inference_mode="snn",
            simulation_steps=2,
        )

    changed_selection = torch.load(output, map_location="cpu", weights_only=False)
    protocol = changed_selection["calibration_protocol"]
    reporting_contract = protocol["source_ann_reporting_contract"]
    reporting_contract["contract"]["checkpoint_selection"] = "unverified-selection"
    reporting_contract["sha256"] = engine_module._canonical_sha256(
        reporting_contract["contract"]
    )
    protocol["source_ann_reporting_contract_sha256"] = reporting_contract["sha256"]
    _recommit_calibration_metadata(changed_selection)
    changed_selection_path = tmp_path / "changed-source-selection.pt"
    atomic_torch_save(changed_selection, changed_selection_path)
    with pytest.raises(ValueError, match="source ANN reporting contract"):
        evaluate(
            config,
            changed_selection_path,
            inference_mode="snn",
            simulation_steps=2,
        )

    tampered_gate_checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    tampered_gate_checkpoint["preflight_gate"]["status"] = "bypassed_non_reporting"
    tampered_gate_path = tmp_path / "tampered-gate-snn.pt"
    atomic_torch_save(tampered_gate_checkpoint, tampered_gate_path)
    with pytest.raises(ValueError, match="preflight gate"):
        evaluate(
            config,
            tampered_gate_path,
            inference_mode="snn",
            simulation_steps=2,
        )

    with next(root.rglob("*.h5")).open("ab") as handle:
        handle.write(b"changed-after-training")
    with pytest.raises(ValueError, match="differs from the training dataset content"):
        calibrate(config, source, tmp_path / "changed_snn.pt")


def test_public_snn_paths_reject_invalid_requests(tmp_path) -> None:
    with pytest.raises(ValueError, match="simulation_steps"):
        evaluate({}, tmp_path / "unused.pt", inference_mode="snn", simulation_steps=0)
    with pytest.raises(ValueError, match="calibration samples"):
        calibrate({}, tmp_path / "unused.pt", tmp_path / "out.pt", samples=0)
    same_path = tmp_path / "same.pt"
    same_path.touch()
    with pytest.raises(ValueError, match="must be different"):
        calibrate({}, same_path, same_path, samples=1, overwrite=True)

    root = tmp_path / "hdr"
    make_eventhdr(root)
    model_config = _model_config()
    uncalibrated = tmp_path / "ann.pt"
    uncalibrated_state = ASGCNUNet(**model_config).state_dict()
    atomic_torch_save(
        {
            "checkpoint_type": "ann_inference",
            "epoch": 1,
            "model": uncalibrated_state,
            "model_state_sha256": _model_state_sha256(uncalibrated_state),
            "model_config": model_config,
            "paper_core_version": 2,
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
