from __future__ import annotations

import json

import pytest
import torch

from asgcn_unet.engine import _model_state_sha256
from asgcn_unet.model import ASGCNUNet
from asgcn_unet.sample_probe import probe_evaluation_sample, save_probe_result
from tests.fixtures import make_eventhdr


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
        "graph_radius": 0.4,
        "graph_position_dims": 3,
        "graph_chunk_size": 16,
        "max_graph_edges": 10_000,
        "spline_kernel_size": 3,
        "spline_degree": 1,
        "spline_root_weight": True,
        "raster_downsample": 4,
        "decoder_channels": 4,
        "output_channels": 1,
        "recurrent": True,
    }


def _config(root) -> dict:
    return {
        "seed": 7,
        "device": "cpu",
        "dataset": {
            "type": "eventhdr",
            "root": str(root),
            "target_channels": 1,
            "max_events": 32,
            "crop_size": None,
            "tone_map": "log",
        },
        "model": _model_config(),
        "eval": {"batch_size": 1, "precision": "fp32", "tf32": False},
    }


def _checkpoint(path, model_config: dict) -> None:
    model = ASGCNUNet(**model_config)
    state = model.state_dict()
    torch.save(
        {
            "model_config": model_config,
            "model": state,
            "model_state_sha256": _model_state_sha256(state),
            "checkpoint_type": "ann_inference",
            "epoch": 3,
        },
        path,
    )


def test_probe_runs_exactly_one_sample_and_reports_topology(tmp_path) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    config = _config(root)
    checkpoint = tmp_path / "best.pt"
    _checkpoint(checkpoint, config["model"])

    result = probe_evaluation_sample(
        config,
        checkpoint,
        sample_index=1,
        max_graph_edges=20_000,
    )

    assert result["schema"] == "asgcn_eval_sample_probe_v1"
    assert result["report_eligible"] is False
    assert result["sample"]["dataset_index"] == 1
    assert result["sample"]["retained_events"] == 32
    assert result["graph_topology"]["exact"] is True
    assert result["graph_topology"]["nodes"] == 32
    assert result["graph_topology"]["actual_directed_edges"] >= 0
    assert result["graph_edge_guard"] == {
        "configured_max_graph_edges": 10_000,
        "requested_max_graph_edges_override": 20_000,
        "effective_max_graph_edges": 20_000,
    }
    assert result["gpu_memory"] == {
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
    }
    assert result["inference"]["recurrent_state"] == "reset"


def test_probe_rejects_index_guard_and_checkpoint_config_mismatch(tmp_path) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    config = _config(root)
    checkpoint = tmp_path / "best.pt"
    _checkpoint(checkpoint, config["model"])

    with pytest.raises(IndexError, match="outside evaluation dataset size"):
        probe_evaluation_sample(
            config,
            checkpoint,
            sample_index=99,
            max_graph_edges=20_000,
        )
    with pytest.raises(ValueError, match="greater than or equal"):
        probe_evaluation_sample(
            config,
            checkpoint,
            sample_index=0,
            max_graph_edges=9_999,
        )
    mismatched = _config(root)
    mismatched["model"]["graph_radius"] = 0.3
    with pytest.raises(ValueError, match="model_config differs"):
        probe_evaluation_sample(
            mismatched,
            checkpoint,
            sample_index=0,
            max_graph_edges=20_000,
        )


def test_save_probe_result_never_overwrites(tmp_path) -> None:
    output = tmp_path / "probe.json"
    save_probe_result(output, {"value": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}

    with pytest.raises(FileExistsError, match="already exists"):
        save_probe_result(output, {"value": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
