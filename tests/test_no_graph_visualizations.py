"""Synthetic CPU no-graph architecture diagnostics; not learned image results."""

from __future__ import annotations

import copy
import csv
import json
import re

import pytest
import torch

from asgcn_unet.ablation_encoders import prepare_event_container
from asgcn_unet.diagnostic_graph import build_diagnostic_graph
from asgcn_unet.graph_preview import build_graph_preview
from asgcn_unet.offline_viewer import OfflineViewerError
from tests.test_result_visualization import mock_resource_gate, run_generate, snapshot


def _config(kind, factor=1):
    return {
        "encoder_kind": kind,
        "event_sampling_factor": factor,
        "graph_radius": 0.8,
        "graph_position_dims": 3,
        "graph_chunk_size": 16,
        "max_graph_edges": 20_000,
    }


def _sample(count):
    generator = torch.Generator().manual_seed(92)
    events = torch.rand((count, 4), generator=generator)
    events[:, :2] *= 7
    events[:, 2] = torch.arange(count)
    events[:, 3] = torch.where(events[:, 3] >= 0.5, 1.0, -1.0)
    return {"events": events, "sensor_size": (8, 8)}


def _forbidden(*args, **kwargs):
    raise AssertionError("No-graph diagnostics must not build radius edges or use CUDA")


@pytest.mark.parametrize("kind", ["identity", "pointwise"])
@pytest.mark.parametrize("factor", [1, 3])
@pytest.mark.parametrize("count", [0, 1, 29])
def test_no_graph_preserves_actual_model_nodes_without_any_pair_computation(
    kind, factor, count, monkeypatch
):
    sample, config = _sample(count), _config(kind, factor)
    reference = prepare_event_container(sample["events"], (8, 8), event_sampling_factor=factor)
    original_events = sample["events"].clone()
    monkeypatch.setattr("asgcn_unet.diagnostic_graph._pair_tiles", _forbidden)
    monkeypatch.setattr("asgcn_unet.graph_preview.build_event_graph", _forbidden)
    monkeypatch.setattr("torch.cuda._lazy_init", _forbidden)
    diagnostic = build_diagnostic_graph(sample, config, memory_budget_bytes=2 << 20)
    preview = build_graph_preview(sample, config, max_graph_edges=20_000)
    nodes = reference.node_features.tolist()
    for payload in (diagnostic, preview.payload):
        assert payload["nodes"] == nodes
        assert payload["edges"] == []
        assert payload["topology_kind"] == "no_graph"
        assert payload["encoder_kind"] == kind
        assert payload["radius"] is payload["position_dims"] is None
        assert payload["configured_radius"] == 0.8
        assert payload["configured_position_dims"] == 3
        assert payload["statistics"] == {
            "nodes": len(nodes),
            "actual_directed_edges": 0,
            "displayed_edges": 0,
            "isolated_nodes": len(nodes),
            "max_degree": 0,
        }
        assert "no-graph" in payload["provenance_note"] or "no graph" in payload["provenance_note"]
        json.dumps(payload, allow_nan=False)
    assert diagnostic["degrees"] == [0] * len(nodes)
    assert diagnostic["memory_plan"]["estimated_tile_scratch_bytes"] == 0
    assert diagnostic["memory_plan"]["maximum_pair_tile"] == 0
    for node in range(len(nodes)):
        assert preview.neighbors(node) == {"node": node, "neighbors": [], "degree": 0}
    torch.testing.assert_close(sample["events"], original_events)


@pytest.mark.parametrize("kind", ["identity", "pointwise"])
def test_no_graph_node_memory_guard_still_precedes_allocation(kind, monkeypatch):
    monkeypatch.setattr("asgcn_unet.diagnostic_graph.prepare_event_nodes", _forbidden)
    with pytest.raises(MemoryError, match="memory budget"):
        build_diagnostic_graph(_sample(4096), _config(kind), memory_budget_bytes=2 << 20)


@pytest.mark.parametrize("kind", ["typo", None, False, ["identity"]])
def test_unknown_encoder_is_never_interpreted_as_no_graph(kind):
    with pytest.raises(ValueError, match="encoder_kind"):
        build_diagnostic_graph(_sample(2), _config(kind), memory_budget_bytes=2 << 20)
    with pytest.raises(ValueError, match="encoder_kind"):
        build_graph_preview(_sample(2), _config(kind), max_graph_edges=20_000)


def test_missing_encoder_remains_radius_graph_and_uses_unchanged_guard():
    config = _config("graph")
    implicit = copy.deepcopy(config)
    implicit.pop("encoder_kind")
    sample = _sample(9)
    first = build_diagnostic_graph(sample, config, memory_budget_bytes=2 << 20)
    second = build_diagnostic_graph(sample, implicit, memory_budget_bytes=2 << 20)
    assert first == second
    assert first["topology_kind"] == "radius_graph"
    assert first["statistics"]["actual_directed_edges"] > 0
    with pytest.raises(ValueError, match="cannot be below"):
        build_graph_preview(sample, config, max_graph_edges=1)


def _no_graph_fixture(base, kind):
    from asgcn_unet.data import build_dataset
    from asgcn_unet.engine import _dataset_sample_identity, _prediction_artifact_stem
    from asgcn_unet.utils import save_image
    from tests.fixtures import make_eventhdr
    from tests.test_viewer_protocol import make_viewer_report

    data_root = base / "synthetic-no-graph-data"
    make_eventhdr(data_root, frames=4)
    config = {
        "device": "cpu",
        "dataset": {
            "type": "eventhdr",
            "root": str(data_root),
            "target_channels": 1,
            "max_events": 32,
            "crop_size": None,
            "frame_stride": 1,
            "tone_map": "log",
            "tone_map_mu": 5000.0,
            "target_normalization": {"mode": "integer_dtype_max"},
        },
        "model": _config(kind),
        "eval": {"precision": "fp32"},
    }
    config_path = base / "synthetic-no-graph-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    root = base / "synthetic-no-graph-eval"
    run = root / "hdr" / "ann"
    run.mkdir(parents=True)
    dataset = build_dataset(config["dataset"], split="eval")
    try:
        rows = []
        for index in range(len(dataset)):
            sample = dataset[index]
            nodes = prepare_event_container(
                sample["events"], sample["sensor_size"], event_sampling_factor=1
            )
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "scene": "synthetic-only",
                    "psnr": 11.0,
                    "ssim": 0.6,
                    "rmse": 0.3,
                    "nodes": nodes.node_features.shape[0],
                    "edges": 0,
                }
            )
            if index < 2:
                stem = _prediction_artifact_stem(sample["sample_id"], index)
                save_image(run / "predictions" / (stem + "_gt.png"), sample["target"])
                # Explicitly synthetic UI fixture; never labelled as model inference.
                save_image(run / "predictions" / (stem + "_pred.png"), sample["target"] * 0.8)
        with (run / "frames.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        report = make_viewer_report(
            config, [_dataset_sample_identity(dataset, i) for i in range(len(dataset))]
        )
        (run / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    finally:
        dataset.close()
    return root, config_path


@pytest.mark.parametrize("kind", ["identity", "pointwise"])
def test_offline_generation_binds_no_graph_source_png_and_zero_edge_counts(
    kind, tmp_path, monkeypatch
):
    root, config = _no_graph_fixture(tmp_path, kind)
    mock_resource_gate(monkeypatch)
    before = snapshot(root)
    monkeypatch.setattr("asgcn_unet.diagnostic_graph._pair_tiles", _forbidden)
    monkeypatch.setattr("asgcn_unet.graph_preview.build_event_graph", _forbidden)
    monkeypatch.setattr("torch.cuda._lazy_init", _forbidden)
    output = tmp_path / "generated"
    summary = run_generate(root, config, output)
    assert summary["complete"] is True
    assert summary["graph_reconstruction"] is False
    assert summary["input_topology_visualization"] is True
    assert summary["topology_kinds"] == ["no_graph"]
    assert summary["generated_frames"] == 2
    assert snapshot(root) == before
    for index in (0, 1):
        directory = output / "hdr" / f"{index:08d}"
        graph = json.loads((directory / "graph.json").read_text())
        assert graph["topology_kind"] == "no_graph"
        assert graph["statistics"]["nodes"] == 32
        assert graph["statistics"]["actual_directed_edges"] == 0
        assert graph["offline_neighbor_scope"] == "complete_no_graph"
        assert "no radius connections" in graph["provenance_note"]
        assert "saved GT pixels" in graph["source_binding_checks"]
        assert (directory / "events-xy.png").is_file()
        assert (directory / "graph-xyt.png").is_file()
    html = (output / "index.html").read_text(encoding="utf-8")
    embedded = json.loads(
        re.search(
            r'<script id="offline-data" type="application/json">(.*?)</script>', html, re.DOTALL
        )[1]
    )
    assert embedded["export"]["graph_reconstruction"] is False
    assert embedded["datasets"][0]["frames"][0]["graph"]["topology_kind"] == "no_graph"


def test_no_graph_does_not_bypass_saved_nonzero_edge_mismatch(tmp_path, monkeypatch):
    root, config = _no_graph_fixture(tmp_path, "identity")
    path = root / "hdr" / "ann" / "frames.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["edges"] = "1"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    mock_resource_gate(monkeypatch)
    output = tmp_path / "mismatch"
    with pytest.raises(OfflineViewerError, match="Graph edges differs"):
        run_generate(root, config, output)
    assert not (output / "index.html").exists()
    assert json.loads((output / "generation.failed.json").read_text())["complete"] is False


def test_no_graph_legacy_readonly_viewer_retains_empty_neighbor_semantics(tmp_path, monkeypatch):
    from asgcn_unet.result_viewer import ResultViewer

    root, config = _no_graph_fixture(tmp_path, "pointwise")
    before = snapshot(root)
    monkeypatch.setattr("asgcn_unet.graph_preview.build_event_graph", _forbidden)
    monkeypatch.setattr("torch.cuda._lazy_init", _forbidden)
    app = ResultViewer(root, configs={"hdr": config})
    try:
        graph = app.graph("hdr", 0)
        assert graph["topology_kind"] == "no_graph"
        assert graph["statistics"]["actual_directed_edges"] == 0
        assert app.neighbors("hdr", 0, 0) == {"node": 0, "neighbors": [], "degree": 0}
    finally:
        app.close()
    assert snapshot(root) == before
