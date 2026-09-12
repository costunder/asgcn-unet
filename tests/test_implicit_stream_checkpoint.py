"""CPU synthetic implicit state/checkpoint tests, not performance measurements."""

from __future__ import annotations

import copy
import io
from dataclasses import replace

import pytest
import torch

from asgcn_unet.checkpoint import capture_training_state, restore_training_state
from asgcn_unet.implicit_radius import (
    ImplicitRadiusGraph,
    ImplicitRadiusIndex,
    build_implicit_radius_graph,
)
from asgcn_unet.stream_graph import StreamGraph
from asgcn_unet.stream_state import StreamingReconstructionState, restore_stream_training_state
from asgcn_unet.training import TrainingState


@pytest.fixture(autouse=True)
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _state(*, empty=False):
    positions = torch.tensor([[0.1, 0.1, 0.1, 0], [0.2, 0.1, 0.1, 1],
                              [0.8, 0.8, 0.2, 1]], dtype=torch.float64)
    features = positions.float().clone().requires_grad_()
    timestamps = torch.tensor([0.1, 0.1, 0.2], dtype=torch.float64)
    if empty:
        positions, features, timestamps = positions[:0], features[:0], timestamps[:0]
    batch = torch.zeros(len(positions), dtype=torch.long)
    graph = build_implicit_radius_graph(features, positions, batch, batch_size=1, radius=0.3,
                                        position_dims=3, chunk_size=2, candidate_pair_budget=3)
    return StreamingReconstructionState(
        StreamGraph(graph, batch, timestamps), None, torch.ones(1, 4, 2, 2, requires_grad=True),
        0.0, 0.3, 2, ("synthetic-a", "source.h5"), (0, 2), "a" * 64,
    )


def _store(state):
    store = TrainingState(independent_sequences=True)
    key = state.sequence_identity
    store.values[key] = (state.sequence_index, (8, 8), state,
                         torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8))
    store.last_key = key
    return store


def _same(actual, expected):
    assert isinstance(actual.graph.graph, ImplicitRadiusGraph)
    for name in ("node_features", "positions", "in_degree", "edge_counts"):
        torch.testing.assert_close(getattr(actual.graph.graph, name), getattr(expected.graph.graph, name), rtol=0, atol=0)
    torch.testing.assert_close(actual.graph.node_batch, expected.graph.node_batch, rtol=0, atol=0)
    torch.testing.assert_close(actual.graph.timestamps, expected.graph.timestamps, rtol=0, atol=0)
    torch.testing.assert_close(actual.decoder, expected.decoder, rtol=0, atol=0)
    assert actual.contract == expected.contract and actual.sequence_identity == expected.sequence_identity
    assert actual.graph.graph.edge_count == expected.graph.graph.edge_count


@pytest.mark.parametrize("operation", ["detach", "clone", "to"])
def test_implicit_map_operations_preserve_geometry_without_enumerating_edges(monkeypatch, operation):
    original = _state()
    monkeypatch.setattr(ImplicitRadiusIndex, "iter_directed_neighbors", lambda *args, **kwargs: pytest.fail("mapping must not enumerate edges"))
    actual = original.to("cpu", copy=True) if operation == "to" else getattr(original, operation)()
    _same(actual, original)
    assert actual.graph.graph._index is None
    assert not hasattr(actual.graph.graph, "edge_index") and not hasattr(actual.graph.graph, "edge_attr")
    if operation == "detach":
        assert not actual.graph.graph.node_features.requires_grad and not actual.decoder.requires_grad
    else:
        assert actual.graph.graph.positions.data_ptr() != original.graph.graph.positions.data_ptr()
        assert actual.graph.graph.in_degree.data_ptr() != original.graph.graph.in_degree.data_ptr()


@pytest.mark.parametrize("empty", [False, True])
def test_v2_safe_serialization_and_strict_restore_preserve_all_states(empty):
    original = _state(empty=empty)
    snapshot = capture_training_state(_store(original))
    raw = snapshot["entries"][0]["recurrent"]
    assert raw["schema"] == "asgcn_stream_training_state_v2"
    assert set(raw["graph"]) == {"node_features", "positions", "in_degree", "edge_counts"}
    assert raw["geometry"]["representation"] == "implicit_radius_v1"
    output = io.BytesIO()
    torch.save(snapshot, output)
    output.seek(0)
    restored = restore_training_state(torch.load(output, weights_only=True), independent_sequences=True, device="cpu")
    actual = restored.values[restored.last_key][2]
    _same(actual, original)
    assert actual.graph.graph.positions.data_ptr() != original.graph.graph.positions.data_ptr() or empty
    assert not actual.graph.graph.node_features.requires_grad and not actual.decoder.requires_grad


def test_capture_checks_trusted_live_state_without_recounting_but_external_load_recounts(monkeypatch):
    original = _state()
    method = ImplicitRadiusIndex.iter_directed_neighbors
    monkeypatch.setattr(ImplicitRadiusIndex, "iter_directed_neighbors", lambda *args, **kwargs: pytest.fail("capture must be O(N), not O(E)"))
    payload = capture_training_state(_store(original))
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(self.positions.shape[0])
        yield from method(self, *args, **kwargs)

    monkeypatch.setattr(ImplicitRadiusIndex, "iter_directed_neighbors", counted)
    restore_training_state(payload, independent_sequences=True, device="cpu")
    assert calls == [3]


def test_external_payload_cannot_request_trusted_live_instance_skip():
    original = _state()
    payload = capture_training_state(_store(original))
    payload["entries"][0]["recurrent"] = original
    with pytest.raises(ValueError, match="serialized streaming state"):
        restore_training_state(payload, independent_sequences=True, device="cpu")


def test_capture_detects_mutated_runtime_degree_versions_and_rejects_encoder_cache():
    original = _state()
    original.graph.graph.in_degree.zero_()
    original.graph.graph.edge_counts.zero_()
    with pytest.raises(RuntimeError, match="modified"):
        capture_training_state(_store(original))
    with pytest.raises(ValueError, match="raw-graph ANN"):
        capture_training_state(_store(replace(_state(), encoder=object())))


def test_geometrically_wrong_but_cardinality_consistent_degree_is_rejected():
    raw = copy.deepcopy(_state().detach().training_payload())
    raw["graph"]["in_degree"].zero_()
    raw["graph"]["edge_counts"].zero_()
    with pytest.raises(ValueError, match="exact radius geometry"):
        restore_stream_training_state(raw)
    raw = copy.deepcopy(_state().detach().training_payload())
    raw["graph"]["in_degree"] = torch.tensor([1, 0, 1])
    with pytest.raises(ValueError, match="exact radius geometry"):
        restore_stream_training_state(raw)


@pytest.mark.parametrize("field,value", [
    ("radius", 0.0), ("radius", float("nan")), ("position_dims", 0),
    ("chunk_size", 0), ("candidate_pair_budget", 0), ("batch_size", 2),
    ("representation", "materialized"),
])
def test_invalid_geometry_contract_is_rejected(field, value):
    raw = copy.deepcopy(_state().detach().training_payload())
    raw["geometry"][field] = value
    with pytest.raises((ValueError, TypeError)):
        restore_stream_training_state(raw)


@pytest.mark.parametrize("field,value", [
    ("node_features", torch.zeros(3, 3)), ("positions", torch.zeros(3, 4)),
    ("positions", torch.full((3, 4), float("nan"), dtype=torch.float64)),
    ("in_degree", torch.tensor([1, 1, -1])), ("edge_counts", torch.tensor([3])),
    ("edge_counts", torch.tensor([2.0])),
])
def test_invalid_graph_tensor_contract_is_rejected(field, value):
    raw = copy.deepcopy(_state().detach().training_payload())
    raw["graph"][field] = value
    with pytest.raises((ValueError, TypeError)):
        restore_stream_training_state(raw)


@pytest.mark.parametrize("field,value", [
    ("timestamps", torch.tensor([0.2, 0.1, 0.1], dtype=torch.float64)),
    ("watermark_seconds", 0.05), ("origin_seconds", float("nan")),
    ("node_batch", torch.tensor([0, 1, 0])), ("sequence_index", True),
    ("sequence_identity", ("", "source.h5")), ("last_event_id", None),
    ("contract", "g" * 64), ("decoder", torch.full((1, 4, 2, 2), float("nan"))),
])
def test_invalid_clock_identity_or_decoder_rejected(field, value):
    raw = copy.deepcopy(_state().detach().training_payload())
    raw[field] = value
    with pytest.raises((ValueError, TypeError)):
        restore_stream_training_state(raw)


def test_no_materialized_edge_placeholder_or_cross_schema_fallback():
    raw = copy.deepcopy(_state().detach().training_payload())
    raw["graph"]["edge_index"] = torch.empty(2, 0, dtype=torch.long)
    with pytest.raises(ValueError, match="placeholders"):
        restore_stream_training_state(raw)
    raw = copy.deepcopy(_state().detach().training_payload())
    raw["schema"] = "asgcn_stream_training_state_v1"
    with pytest.raises(ValueError, match="fields"):
        restore_stream_training_state(raw)


def test_finite_checks_positions_and_features_without_edge_attributes():
    original = _state()
    assert bool(original.finite())
    original.graph.graph.node_features.detach()[0, 0] = float("nan")
    assert not bool(original.finite())
    original = _state()
    original.graph.graph.positions[0, 0] = float("inf")
    assert not bool(original.finite())
