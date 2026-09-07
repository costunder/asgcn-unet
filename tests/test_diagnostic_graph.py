"""Small, explicitly synthetic CPU unit fixtures; not model quality results."""

from __future__ import annotations

import copy
import json

import pytest
import torch

import asgcn_unet.diagnostic_graph as diagnostic
from asgcn_unet.graph import build_event_graph


@pytest.fixture
def smoke_config():
    return {
        "event_sampling_factor": 1,
        "graph_radius": 0.55,
        "graph_position_dims": 3,
        "graph_chunk_size": 7,
        "max_graph_edges": 2,
        "spline_backend": "triton",
    }


@pytest.fixture
def smoke_sample():
    generator = torch.Generator().manual_seed(319)
    events = torch.rand((29, 4), generator=generator, dtype=torch.float64)
    events[:, :2] *= 7
    events[:, 2] = torch.arange(29, dtype=torch.float64)
    events[:, 3] = (events[:, 3] >= 0.5).double()
    return {
        "events": events, "sensor_size": (8, 8),
        "metadata": {"raw_event_count": 40, "t0_us": 100, "t1_us": 200},
    }


def _reference(sample, config):
    return build_event_graph(
        sample["events"], sample["sensor_size"],
        event_sampling_factor=config["event_sampling_factor"],
        graph_radius=config["graph_radius"],
        graph_position_dims=config["graph_position_dims"],
        graph_chunk_size=config["graph_chunk_size"],
        max_graph_edges=None,
    )


def _budget(sample, config, display_edges, pair_capacity):
    node_count = (len(sample["events"]) + config["event_sampling_factor"] - 1) // (
        config["event_sampling_factor"]
    )
    return (
        diagnostic._FIXED_BYTES + sample["events"].untyped_storage().nbytes()
        + node_count * diagnostic._BYTES_PER_NODE
        + min(display_edges, node_count * max(0, node_count - 1))
        * diagnostic._BYTES_PER_DISPLAY_EDGE
        + pair_capacity * diagnostic._BYTES_PER_PAIR
    )


@pytest.mark.parametrize("dims", [1, 2, 3, 4])
@pytest.mark.parametrize("factor", [1, 2, 3])
@pytest.mark.parametrize("display_edges", [0, 1, 13, 1000])
def test_smoke_exact_model_topology_and_source_major_display(
    smoke_sample, smoke_config, dims, factor, display_edges,
):
    smoke_config.update(graph_position_dims=dims, event_sampling_factor=factor)
    reference = _reference(smoke_sample, smoke_config)
    payload = diagnostic.build_diagnostic_graph(
        smoke_sample, smoke_config,
        memory_budget_bytes=_budget(smoke_sample, smoke_config, display_edges, 23),
        display_edges=display_edges,
    )
    count = reference.edge_index.shape[1]
    displayed = min(display_edges, count)
    chosen = torch.arange(displayed) * max(0, count - 1) // max(displayed - 1, 1)
    assert payload["nodes"] == reference.node_features.tolist()
    assert payload["edges"] == reference.edge_index[:, chosen].t().tolist()
    assert payload["degrees"] == reference.in_degree.tolist()
    assert payload["statistics"] == {
        "nodes": reference.node_features.shape[0],
        "actual_directed_edges": count,
        "displayed_edges": displayed,
        "isolated_nodes": int((reference.in_degree == 0).sum()),
        "max_degree": int(reference.in_degree.max()),
    }
    assert payload["metadata"]["raw_events"] == 40
    assert payload["metadata"]["retained_events"] == 29
    assert payload["metadata"]["t0_us"] == 100
    assert "display-only limit" in payload["provenance_note"]
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("count", [0, 1, 4, 256])
def test_smoke_dense_complete_graph_has_no_topology_cap(smoke_config, count):
    sample = {"events": torch.zeros((count, 4)), "sensor_size": (1, 1)}
    payload = diagnostic.build_diagnostic_graph(
        sample, smoke_config, memory_budget_bytes=4 << 20, display_edges=11,
    )
    assert payload["statistics"]["actual_directed_edges"] == count * max(count - 1, 0)
    assert payload["statistics"]["nodes"] == count
    assert payload["statistics"]["displayed_edges"] == min(11, count * max(count - 1, 0))
    assert payload["degrees"] == [max(count - 1, 0)] * count
    assert len(payload["nodes"]) == count


def test_smoke_strict_boundary_float32_and_coincident_nodes(smoke_config):
    smoke_config.update(graph_radius=0.5, graph_position_dims=1)
    half = torch.tensor(0.5)
    sample = {
        "events": torch.tensor([
            [0, 0, 0, 1], [0, 0, 0, -1],
            [torch.nextafter(half, torch.tensor(0.0)), 0, 1, 1],
            [0.5, 0, 2, 1],
            [torch.nextafter(half, torch.tensor(1.0)), 0, 3, 1],
        ]),
        "sensor_size": (1, 2),
    }
    payload = diagnostic.build_diagnostic_graph(
        sample, smoke_config, memory_budget_bytes=2 << 20, display_edges=100,
    )
    reference = _reference(sample, smoke_config)
    assert payload["edges"] == reference.edge_index.t().tolist()
    assert [0, 1] in payload["edges"]
    assert [0, 2] in payload["edges"]
    assert [0, 3] not in payload["edges"]
    assert [0, 4] not in payload["edges"]
    assert all(source != destination for source, destination in payload["edges"])


@pytest.mark.parametrize("pair_capacity", [1, 2, 7, 31])
def test_smoke_every_pair_allocation_obeys_explicit_budget(
    smoke_sample, smoke_config, monkeypatch, pair_capacity,
):
    original_norm = torch.linalg.vector_norm
    shapes = []

    def checked_norm(values, *args, **kwargs):
        assert values.ndim == 3
        assert values.shape[0] * values.shape[1] <= pair_capacity
        shapes.append(values.shape)
        return original_norm(values, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "vector_norm", checked_norm)
    budget = _budget(smoke_sample, smoke_config, 5, pair_capacity)
    payload = diagnostic.build_diagnostic_graph(
        smoke_sample, smoke_config, memory_budget_bytes=budget, display_edges=5,
    )
    assert shapes
    assert payload["memory_plan"]["maximum_pair_tile"] <= pair_capacity
    assert payload["memory_plan"]["estimated_working_set_bytes"] <= budget
    assert payload["memory_plan"]["source_tile_nodes"] <= smoke_config["graph_chunk_size"]


def test_smoke_budget_failure_precedes_node_or_pair_allocation(
    smoke_sample, smoke_config, monkeypatch,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("budget must be checked before node normalization or pair allocation")

    monkeypatch.setattr(diagnostic, "prepare_event_nodes", forbidden)
    monkeypatch.setattr(torch.linalg, "vector_norm", forbidden)
    with pytest.raises(MemoryError, match="No topology was truncated"):
        diagnostic.build_diagnostic_graph(smoke_sample, smoke_config, memory_budget_bytes=1)


def test_smoke_does_not_materialize_edges_touch_gpu_or_mutate_inputs(
    smoke_sample, smoke_config, monkeypatch,
):
    import asgcn_unet.graph as graph_module
    import asgcn_unet.model as model_module

    before_sample, before_config = copy.deepcopy(smoke_sample), copy.deepcopy(smoke_config)

    def forbidden(*args, **kwargs):
        raise AssertionError("offline graph diagnostics must not initialize GPU/model/full edge list")

    for name in ("init", "is_available", "device_count", "current_stream"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(graph_module, "build_event_graph", forbidden)
    monkeypatch.setattr(graph_module, "build_radius_graph", forbidden)
    monkeypatch.setattr(graph_module, "_radius_graph_candidate_chunks", forbidden)
    monkeypatch.setattr(model_module.ASGCNUNet, "forward_sample", forbidden)
    monkeypatch.setattr(model_module.ASGCNUNet, "forward_batch", forbidden)
    payload = diagnostic.build_diagnostic_graph(
        smoke_sample, smoke_config, memory_budget_bytes=2 << 20,
    )
    assert payload["statistics"]["nodes"] == 29
    torch.testing.assert_close(smoke_sample["events"], before_sample["events"], rtol=0, atol=0)
    assert smoke_sample["metadata"] == before_sample["metadata"]
    assert smoke_config == before_config


@pytest.mark.parametrize("field,value", [
    ("memory_budget_bytes", 0), ("memory_budget_bytes", True), ("memory_budget_bytes", 2.5),
    ("display_edges", -1), ("display_edges", False), ("display_edges", 0.5),
])
def test_smoke_rejects_invalid_memory_or_display_settings(smoke_sample, smoke_config, field, value):
    settings = {"memory_budget_bytes": 2 << 20, field: value}
    with pytest.raises(ValueError):
        diagnostic.build_diagnostic_graph(smoke_sample, smoke_config, **settings)


@pytest.mark.parametrize("field,value", [
    ("graph_radius", None), ("graph_radius", True), ("graph_radius", float("nan")),
    ("graph_radius", 0), ("graph_radius", float("inf")),
    ("graph_position_dims", 5), ("graph_position_dims", False),
    ("event_sampling_factor", 0), ("event_sampling_factor", 1.5),
    ("graph_chunk_size", 0), ("graph_chunk_size", True),
])
def test_smoke_rejects_invalid_model_rules(smoke_sample, smoke_config, field, value):
    smoke_config[field] = value
    with pytest.raises(ValueError):
        diagnostic.build_diagnostic_graph(smoke_sample, smoke_config, memory_budget_bytes=2 << 20)


@pytest.mark.parametrize("changes", [
    {"events": torch.zeros((3, 3))}, {"events": torch.zeros((3, 4), dtype=torch.bool)},
    {"events": torch.zeros((3, 4), dtype=torch.complex64)},
    {"events": torch.empty((1, 4), device="meta")},
    {"events": torch.tensor([[0, 0, 1, 1], [0, 0, 0, 1]])},
    {"events": torch.tensor([[0, 0, float("nan"), 1]])},
    {"events": torch.tensor([[10, 0, 0, 1]])},
    {"sensor_size": (0, 8)}, {"sensor_size": (8,)},
    {"metadata": []}, {"metadata": {"raw_event_count": 1}},
    {"metadata": {"t0_us": float("nan")}}, {"metadata": {"t0_us": 2, "t1_us": 1}},
])
def test_smoke_rejects_invalid_samples(smoke_sample, smoke_config, changes):
    smoke_sample.update(changes)
    with pytest.raises((ValueError, TypeError)):
        diagnostic.build_diagnostic_graph(smoke_sample, smoke_config, memory_budget_bytes=2 << 20)


def test_smoke_budget_is_required(smoke_sample, smoke_config):
    with pytest.raises(TypeError, match="memory_budget_bytes"):
        diagnostic.build_diagnostic_graph(smoke_sample, smoke_config)


def test_smoke_explicit_model_rules_are_required(smoke_sample, smoke_config):
    del smoke_config["graph_radius"]
    with pytest.raises(ValueError, match="Missing model graph settings"):
        diagnostic.build_diagnostic_graph(smoke_sample, smoke_config, memory_budget_bytes=2 << 20)
