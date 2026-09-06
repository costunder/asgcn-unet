"""Explicit CPU smoke fixtures for display-only graph preview contracts."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from asgcn_unet.graph import build_event_graph
from asgcn_unet.graph_preview import build_graph_preview


@pytest.fixture
def smoke_config():
    return {
        "event_sampling_factor": 1,
        "graph_radius": 0.8,
        "graph_position_dims": 3,
        "graph_chunk_size": 3,
        "max_graph_edges": 100,
        # A graph-only preview must not load this CUDA-only encoder backend.
        "spline_backend": "triton",
    }


@pytest.fixture
def smoke_sample():
    return {
        "events": torch.tensor([
            [0, 0, 10, -1], [1, 1, 10.5, 1], [2, 1, 11, -1],
            [2, 2, 11.5, 1], [3, 3, 12, -1], [7, 7, 13, 1],
        ], dtype=torch.float64),
        "target": torch.arange(64, dtype=torch.float32).reshape(1, 8, 8),
        "sensor_size": (8, 8),
        "sample_id": "smoke/000001",
        "metadata": {"raw_event_count": 9, "t0_us": 10, "t1_us": 13},
    }


def _reference(sample, config, guard=100):
    return build_event_graph(
        sample["events"], sample["sensor_size"],
        event_sampling_factor=config["event_sampling_factor"],
        graph_radius=config["graph_radius"],
        graph_position_dims=config["graph_position_dims"],
        graph_chunk_size=config["graph_chunk_size"],
        max_graph_edges=guard,
    )


@pytest.mark.parametrize("factor", [1, 2, 3])
def test_cpu_smoke_matches_actual_graph_without_extra_node_sampling(smoke_sample, smoke_config, factor):
    smoke_config["event_sampling_factor"] = factor
    reference = _reference(smoke_sample, smoke_config)
    preview = build_graph_preview(smoke_sample, smoke_config, max_graph_edges=100, display_edges=100)
    assert preview.payload["nodes"] == reference.node_features.tolist()
    assert preview.payload["edges"] == reference.edge_index.t().tolist()
    assert preview.payload["statistics"]["nodes"] == reference.node_features.shape[0]
    assert preview.payload["statistics"]["actual_directed_edges"] == reference.edge_index.shape[1]
    assert preview.payload["metadata"]["retained_events"] == 6
    assert preview.payload["metadata"]["raw_events"] == 9
    assert preview.payload["metadata"]["t0_us"] == 10
    assert preview.payload["metadata"]["t1_us"] == 13
    json.dumps(preview.payload, allow_nan=False)


@pytest.mark.parametrize("display_edges", [0, 1, 3, 5000])
def test_cpu_smoke_display_subset_keeps_full_neighbors(smoke_sample, smoke_config, display_edges):
    reference = _reference(smoke_sample, smoke_config)
    preview = build_graph_preview(
        smoke_sample, smoke_config, max_graph_edges=100, display_edges=display_edges,
    )
    repeated = build_graph_preview(
        smoke_sample, smoke_config, max_graph_edges=100, display_edges=display_edges,
    )
    full_edges = {tuple(edge) for edge in reference.edge_index.t().tolist()}
    shown = [tuple(edge) for edge in preview.payload["edges"]]
    assert len(shown) == min(display_edges, len(full_edges))
    assert len(set(shown)) == len(shown)
    assert set(shown) <= full_edges
    assert preview.payload == repeated.payload
    assert preview.payload["statistics"]["actual_directed_edges"] == len(full_edges)
    assert "display-only limit" in preview.payload["provenance_note"]
    for node in range(reference.node_features.shape[0]):
        expected = reference.edge_index[1, reference.edge_index[0] == node].tolist()
        assert preview.neighbors(node) == {"node": node, "neighbors": expected, "degree": len(expected)}


@pytest.mark.parametrize("count", [0, 1, 4])
def test_cpu_smoke_empty_singleton_and_coincident_nodes(smoke_config, count):
    sample = {"events": torch.zeros((count, 4)), "sensor_size": (1, 1)}
    preview = build_graph_preview(sample, smoke_config, max_graph_edges=100)
    assert preview.payload["statistics"] == {
        "nodes": count,
        "actual_directed_edges": count * max(count - 1, 0),
        "displayed_edges": count * max(count - 1, 0),
        "isolated_nodes": count if count < 2 else 0,
        "max_degree": max(count - 1, 0),
    }
    assert preview.payload["metadata"]["t0_us"] is None
    assert preview.payload["metadata"]["t1_us"] is None
    for node in range(count):
        assert preview.neighbors(node)["neighbors"] == [other for other in range(count) if other != node]
    with pytest.raises(IndexError):
        preview.neighbors(count)


def test_cpu_smoke_strict_radius_boundary_and_isolated_node(smoke_config):
    smoke_config.update(graph_radius=0.5, graph_position_dims=1)
    sample = {
        "events": torch.tensor([[0, 0, 0, 1], [1, 0, 1, 1], [2, 0, 2, -1]]),
        "sensor_size": (1, 3),
    }
    preview = build_graph_preview(sample, smoke_config, max_graph_edges=100)
    assert preview.payload["edges"] == []
    assert preview.payload["statistics"]["isolated_nodes"] == 3
    assert preview.neighbors(1) == {"node": 1, "neighbors": [], "degree": 0}


def test_cpu_smoke_does_not_mutate_inputs_or_initialize_cuda(smoke_sample, smoke_config, monkeypatch):
    before_sample, before_config = copy.deepcopy(smoke_sample), copy.deepcopy(smoke_config)

    def forbidden(*args, **kwargs):
        raise AssertionError("Graph-only CPU preview must not initialize/query CUDA or run a model")

    import asgcn_unet.model as model_module
    import asgcn_unet.ops as ops_module

    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    monkeypatch.setattr(torch.cuda, "current_stream", forbidden)
    monkeypatch.setattr(ops_module, "_triton_ops", forbidden)
    monkeypatch.setattr(model_module.ASGCNUNet, "forward_sample", forbidden)
    monkeypatch.setattr(model_module.ASGCNUNet, "forward_batch", forbidden)
    preview = build_graph_preview(smoke_sample, smoke_config, max_graph_edges=100)
    assert preview._edge_index.device.type == "cpu"
    assert smoke_config == before_config
    for name in ("events", "target"):
        torch.testing.assert_close(smoke_sample[name], before_sample[name], rtol=0, atol=0)
    for name in ("sensor_size", "metadata", "sample_id"):
        assert smoke_sample[name] == before_sample[name]


def test_cpu_smoke_guard_failure_does_not_truncate_graph(smoke_config):
    smoke_config["max_graph_edges"] = 2
    sample = {"events": torch.zeros((4, 4)), "sensor_size": (1, 1)}
    with pytest.raises(RuntimeError, match="exceeded max_graph_edges"):
        build_graph_preview(sample, smoke_config, max_graph_edges=2, display_edges=1)
    preview = build_graph_preview(sample, smoke_config, max_graph_edges=12, display_edges=1)
    assert preview.payload["statistics"]["actual_directed_edges"] == 12


@pytest.mark.parametrize("node", [-1, True, 1.5, "0"])
def test_cpu_smoke_rejects_invalid_node_indices(smoke_sample, smoke_config, node):
    preview = build_graph_preview(smoke_sample, smoke_config, max_graph_edges=100)
    with pytest.raises(ValueError, match="node"):
        preview.neighbors(node)


@pytest.mark.parametrize("settings", [
    {"max_graph_edges": 0}, {"max_graph_edges": True}, {"max_graph_edges": 2.5},
    {"max_graph_edges": 99}, {"display_edges": -1}, {"display_edges": False},
    {"display_edges": 1.5},
])
def test_cpu_smoke_rejects_invalid_guards_and_display_budgets(smoke_sample, smoke_config, settings):
    kwargs = {"max_graph_edges": 100, **settings}
    with pytest.raises(ValueError):
        build_graph_preview(smoke_sample, smoke_config, **kwargs)


@pytest.mark.parametrize("name,value", [
    ("event_sampling_factor", 0), ("event_sampling_factor", 1.5),
    ("graph_radius", float("nan")), ("graph_radius", float("inf")), ("graph_radius", 0),
    ("graph_position_dims", 5), ("graph_position_dims", False), ("graph_chunk_size", 0),
])
def test_cpu_smoke_rejects_invalid_model_settings(smoke_sample, smoke_config, name, value):
    smoke_config[name] = value
    with pytest.raises(ValueError):
        build_graph_preview(smoke_sample, smoke_config, max_graph_edges=100)


def test_cpu_smoke_requires_explicit_model_settings(smoke_sample, smoke_config):
    del smoke_config["graph_radius"]
    with pytest.raises(ValueError, match="Missing model graph settings"):
        build_graph_preview(smoke_sample, smoke_config, max_graph_edges=100)


@pytest.mark.parametrize("changes", [
    {"sensor_size": (0, 3)}, {"sensor_size": (3,)},
    {"events": torch.zeros((3, 3))}, {"events": torch.zeros((3, 4), dtype=torch.bool)},
    {"events": torch.empty((1, 4), device="meta")},
    {"metadata": {"raw_event_count": 1}}, {"metadata": {"t0_us": float("nan")}},
    {"metadata": {"t0_us": 2, "t1_us": 1}},
])
def test_cpu_smoke_rejects_invalid_sample_without_gpu_execution(smoke_sample, smoke_config, changes):
    smoke_sample.update(changes)
    with pytest.raises((ValueError, TypeError)):
        build_graph_preview(smoke_sample, smoke_config, max_graph_edges=100)
