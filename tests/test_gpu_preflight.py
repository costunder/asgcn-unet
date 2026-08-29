from __future__ import annotations

import json

import pytest
import torch

import asgcn_unet.cli as cli_module
import asgcn_unet.preflight as preflight_module
from asgcn_unet.graph import build_radius_graph, radius_graph_topology
from asgcn_unet.preflight import training_preflight, verify_training_preflight
from asgcn_unet.utils import save_json
from tests.fixtures import make_eventhdr


def _config(root, *, max_graph_edges: int = 10_000, radius: float = 0.45) -> dict:
    return {
        "seed": 31,
        "device": "cpu",
        "dataset": {
            "type": "eventhdr",
            "root": str(root),
            "target_channels": 1,
            "max_events": 32,
            "crop_size": None,
            "frame_stride": 1,
            "target_normalization": {"mode": "integer_dtype_max"},
            "tone_map": "log",
            "tone_map_mu": 5000.0,
        },
        "model": {
            "architecture_version": 2,
            "graph_operator": "spline",
            "spline_backend": "torch",
            "spline_pseudo": "distance_over_radius",
            "spline_is_open": True,
            "hidden_dim": 4,
            "graph_layers": 1,
            "event_sampling_factor": 2,
            "graph_radius": radius,
            "graph_position_dims": 3,
            "graph_chunk_size": 8,
            "max_graph_edges": max_graph_edges,
            "spline_kernel_size": 3,
            "spline_degree": 1,
            "spline_root_weight": True,
            "spline_chunk_size": 32,
            "raster_downsample": 4,
            "decoder_channels": 4,
            "output_channels": 1,
            "recurrent": False,
        },
        "train": {
            "epochs": 2,
            "batch_size": 1,
            "num_workers": 0,
            "optimizer": "adam_gc",
            "learning_rate": 1e-3,
            "weight_decay": 5e-3,
            "grad_clip": 1.0,
            "amp": False,
            "validate_every": 1,
            "loss_weights": {
                "charbonnier": 1.0,
                "ssim": 0.2,
                "gradient": 0.1,
                "temporal": 0.2,
            },
        },
        "output": {"run_dir": str(root.parent / "train")},
    }


def test_topology_counter_matches_materialized_directed_graph() -> None:
    generator = torch.Generator().manual_seed(8)
    positions = torch.rand((37, 4), generator=generator)
    edge_index, _ = build_radius_graph(
        positions,
        0.3,
        position_dims=3,
        chunk_size=7,
    )
    topology = radius_graph_topology(
        positions,
        0.3,
        position_dims=3,
        chunk_size=7,
    )

    degree = torch.bincount(edge_index[1], minlength=len(positions))
    assert topology["actual_directed_edges"] == edge_index.shape[1]
    assert topology["candidate_directed_edges"] >= topology["actual_directed_edges"]
    assert topology["max_degree"] == int(degree.max())
    assert topology["isolated_nodes"] == int((degree == 0).sum())


def test_cpu_diagnostic_scans_every_sample_and_runs_configured_backward(tmp_path) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    output = tmp_path / "preflight.json"

    report = training_preflight(
        _config(root),
        output,
        profile_samples=2,
        top_density_count=3,
        require_cuda=False,
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == report
    assert report["status"] == "diagnostic_passed"
    assert report["passed"] is True
    assert report["report_eligible"] is False
    assert report["measurement_scope"]["name"] == "selected_top_density_training_steps"
    assert report["measurement_scope"]["absolute_vram_guarantee"] is False
    assert report["checks"] == {
        "cuda_available": False,
        "complete_topology_scan": True,
        "edge_guard": True,
        "forward_backward": True,
        "cuda_oom_free": None,
    }
    topology = report["topology"]
    assert topology["scan_scope"] == "complete_eventhdr_training_split"
    assert topology["samples_scanned"] == topology["dataset_samples"] == 4
    assert len(topology["samples"]) == 4
    assert len(topology["top_density_samples"]) == 3
    assert all(item["raw_events"] == 96 for item in topology["samples"])
    assert all(item["retained_events"] == 32 for item in topology["samples"])
    assert all(item["model_sampled_events"] == 16 for item in topology["samples"])
    assert all(
        item["candidate_directed_edges"] >= item["actual_directed_edges"]
        for item in topology["samples"]
    )
    assert report["training_probe"]["completed_samples"] == 2
    assert len(report["training_probe"]["steps"]) == 2
    assert all(step["gradient_norm"] >= 0 for step in report["training_probe"]["steps"])
    assert report["data_provenance"]["content"]["files"] == 1
    assert len(report["data_provenance"]["content"]["sha256"]) == 64
    assert report["config_provenance"]["config"]["dataset"]["root"] == "$EXTERNAL/hdr"


def test_edge_guard_failure_is_measured_exactly_and_written_atomically(tmp_path) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    output = tmp_path / "edge-failure.json"

    report = training_preflight(
        _config(root, max_graph_edges=5, radius=2.0),
        output,
        profile_samples=1,
        top_density_count=2,
        require_cuda=False,
    )

    assert report["status"] == "failed"
    assert report["passed"] is False
    assert report["checks"]["complete_topology_scan"] is True
    assert report["checks"]["edge_guard"] is False
    assert report["training_probe"]["failure_category"] == "edge_guard_exceeded"
    assert report["training_probe"]["completed_samples"] == 0
    assert report["topology"]["edge_guard_exceeded_samples"] == 4
    assert all(
        item["actual_directed_edges"] == 16 * 15
        for item in report["topology"]["samples"]
    )
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False


def test_report_mode_requires_real_cuda_before_scanning(tmp_path, monkeypatch) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    output = tmp_path / "no-cuda.json"
    monkeypatch.setattr(preflight_module.torch.cuda, "is_available", lambda: False)

    report = training_preflight(_config(root), output, require_cuda=True)

    assert report["status"] == "failed"
    assert report["passed"] is False
    assert report["checks"]["cuda_available"] is False
    assert report["checks"]["complete_topology_scan"] is False
    assert report["training_probe"]["failure_category"] == "cuda_required_but_unavailable"
    with pytest.raises(FileExistsError, match="already exists"):
        training_preflight(_config(root), output, require_cuda=True)


def test_profile_sample_count_cannot_exceed_recorded_density_ranking(tmp_path) -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        training_preflight(
            _config(tmp_path / "unused"),
            tmp_path / "unused.json",
            profile_samples=4,
            top_density_count=3,
            require_cuda=False,
        )


def test_verifier_rebinds_report_to_current_data_source_config_and_runtime(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    output = tmp_path / "profile.json"
    config = _config(root)
    report = training_preflight(
        config,
        output,
        profile_samples=1,
        top_density_count=1,
        require_cuda=False,
    )
    source = {
        "source_tree_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "git_source_dirty": False,
    }
    runtime = {
        "python": "test",
        "platform": "Linux",
        "torch": "test",
        "requested_device": "cuda",
        "cuda_available": True,
        "cuda_runtime": "test",
        "cudnn": 1,
        "gpu": {
            "index": 0,
            "name": "test-gpu",
            "compute_capability": [9, 0],
            "total_memory_mib": 81920.0,
            "multiprocessors": 1,
        },
    }
    protocol = {"version": "test-protocol"}
    report["status"] = "passed"
    report["passed"] = True
    report["report_eligible"] = True
    report["request"]["require_cuda"] = True
    report["checks"]["cuda_available"] = True
    report["checks"]["cuda_oom_free"] = True
    report["source_provenance"] = source
    report["runtime_provenance"] = runtime
    report["training_probe"]["training_protocol"] = protocol
    report["training_probe"]["steps"][0]["peak_allocated_mib"] = 100.0
    report["training_probe"]["steps"][0]["peak_reserved_mib"] = 120.0
    save_json(output, report)

    monkeypatch.setattr(preflight_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(preflight_module, "resolve_device", lambda value: torch.device("cuda"))
    monkeypatch.setattr(preflight_module, "_current_source_contract", lambda: source)
    monkeypatch.setattr(preflight_module, "_runtime_provenance", lambda device: runtime)
    monkeypatch.setattr(preflight_module, "_training_protocol", lambda config, device: protocol)

    verification = verify_training_preflight(config, output)
    assert verification["status"] == "verified"
    assert verification["report_eligible"] is True
    assert verification["measured_steps"] == 1
    assert verification["gpu"]["name"] == "test-gpu"
    assert len(verification["report_sha256"]) == 64

    report["measurement_scope"]["absolute_vram_guarantee"] = True
    save_json(output, report)
    with pytest.raises(ValueError, match="overstates"):
        verify_training_preflight(config, output)


def test_cli_nonreporting_bypass_is_warned_and_embedded_in_training_config(
    tmp_path, monkeypatch, capsys
) -> None:
    config = _config(tmp_path / "unused")
    config_path = tmp_path / "config.json"
    save_json(config_path, config)
    captured: dict = {}

    def fake_train(resolved_config, resume_from=None):
        captured.update(resolved_config)
        assert resume_from is None
        return tmp_path / "best.pt"

    monkeypatch.setattr(cli_module, "train", fake_train)
    cli_module.main(
        [
            "train",
            "--config",
            str(config_path),
            "--allow-unverified-preflight",
        ]
    )

    assert captured["preflight_gate"]["status"] == "bypassed_non_reporting"
    assert captured["preflight_gate"]["report_eligible"] is False
    gate_path = tmp_path / "train" / "preflight_gate.json"
    assert json.loads(gate_path.read_text(encoding="utf-8"))["status"] == (
        "bypassed_non_reporting"
    )
    assert "WARNING" in capsys.readouterr().err
