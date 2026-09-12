"""Tiny one-thread CPU state/schema tests, not model or performance validation."""

from __future__ import annotations

import copy
import io
from dataclasses import dataclass, field, replace

import pytest
import torch

from asgcn_unet.checkpoint import capture_training_state, restore_training_state
from asgcn_unet.graph import EventGraph
from asgcn_unet.hierarchy import HierarchyState
from asgcn_unet.implicit_radius import ImplicitRadiusIndex, build_implicit_radius_graph
from asgcn_unet.stream_encoder import StreamEncoderState
from asgcn_unet.stream_graph import StreamGraph
from asgcn_unet.stream_state import StreamingReconstructionState, restore_stream_training_state
from asgcn_unet.training import TrainingState


@pytest.fixture(autouse=True, scope="module")
def _one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _state(storage="materialized", *, sampling_offset=0):
    positions = torch.tensor([[0.1, 0.1, 0.1, 0], [0.2, 0.1, 0.1, 1],
                              [0.8, 0.8, 0.2, 1]], dtype=torch.float64)
    features = positions.float().clone().requires_grad_()
    node_batch = torch.zeros(3, dtype=torch.long)
    if storage == "implicit":
        graph = build_implicit_radius_graph(
            features, positions, node_batch, batch_size=1, radius=0.3,
            position_dims=3, chunk_size=2, candidate_pair_budget=3,
        )
    else:
        graph = EventGraph(features, positions, torch.tensor([[0, 1], [1, 0]]),
                           torch.full((2, 1), 1 / 3, dtype=torch.float64))
    # The original nine positional arguments remain source-compatible.
    state = StreamingReconstructionState(
        StreamGraph(graph, node_batch, torch.tensor([0.1, 0.1, 0.2], dtype=torch.float64)),
        None, torch.ones(1, 4, 2, 2, requires_grad=True),
        0.0, 0.3, 2, ("synthetic-a", "source.h5"), (0, 2), "a" * 64,
    )
    return replace(state, sampling_offset=sampling_offset)


def _store(state):
    store = TrainingState(independent_sequences=True)
    key = state.sequence_identity
    store.values[key] = (state.sequence_index, (4, 4), state,
                         torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4))
    store.last_key = key
    return store


@dataclass(frozen=True)
class _SyntheticPoolCache:
    """Minimal pool protocol fixture; no pooling result or model is simulated."""

    raw_graph: StreamGraph
    graph: StreamGraph
    feature_sums: torch.Tensor
    counts: torch.Tensor
    work: dict = field(default_factory=dict)


def _with_caches(state):
    raw = state.graph
    coarse = StreamGraph(
        EventGraph(torch.ones(2, 4, requires_grad=True),
                   raw.graph.positions[:2].clone(), torch.tensor([[0, 1], [1, 0]]),
                   torch.full((2, 1), 0.25, dtype=torch.float64)),
        torch.zeros(2, dtype=torch.long), raw.timestamps[:2].clone(),
    )
    prefix = StreamEncoderState(
        raw, torch.ones(3, 4, requires_grad=True),
        (torch.ones(3, 4, requires_grad=True),),
        local_ticks=(torch.ones(3, dtype=torch.long),),
    )
    suffix = StreamEncoderState(
        coarse, torch.ones(2, 4, requires_grad=True),
        (torch.ones(2, 4, requires_grad=True),),
        local_ticks=(torch.ones(2, dtype=torch.long),),
    )
    pool = _SyntheticPoolCache(raw, coarse, torch.ones(2, 4, requires_grad=True),
                               torch.tensor([2, 1]))
    return replace(state, encoder=prefix, hierarchy=HierarchyState(pool, suffix))


@pytest.mark.parametrize("storage,version", [("materialized", 1), ("implicit", 2)])
def test_zero_sampling_offset_keeps_legacy_payload_fields_and_defaults(storage, version):
    original = _state(storage)
    assert original.sampling_offset == 0 and original.hierarchy is None
    payload = original.training_payload()
    assert payload["schema"] == f"asgcn_stream_training_state_v{version}"
    assert "sampling_offset" not in payload and "raw_state" not in payload
    restored = restore_stream_training_state(payload)
    assert restored.sampling_offset == 0 and restored.hierarchy is None
    assert set(restored.training_payload()) == set(payload)


@pytest.mark.parametrize("storage", ["materialized", "implicit"])
@pytest.mark.parametrize("operation", ["clone", "detach", "to"])
def test_mapping_preserves_counter_and_maps_raw_prefix_and_coarse_suffix_separately(
    storage, operation, monkeypatch,
):
    original = _with_caches(_state(storage, sampling_offset=2**70 + 7))
    monkeypatch.setattr(ImplicitRadiusIndex, "iter_directed_neighbors",
                        lambda *args, **kwargs: pytest.fail("mapping must not enumerate edges"))
    mapped = original.to("cpu", copy=True) if operation == "to" else getattr(original, operation)()
    assert mapped.sampling_offset == original.sampling_offset
    assert type(mapped.sampling_offset) is int
    assert mapped.encoder.graph is mapped.graph
    assert mapped.hierarchy.pool.raw_graph is mapped.graph
    assert mapped.hierarchy.suffix.graph is mapped.hierarchy.pool.graph
    assert len(mapped.encoder.outputs) == 3 and len(mapped.hierarchy.suffix.outputs) == 2
    assert mapped.hierarchy is not original.hierarchy
    before = tuple(original.hierarchy.tensors())
    after = tuple(mapped.hierarchy.tensors())
    for source, destination in zip(before, after, strict=True):
        torch.testing.assert_close(source, destination, rtol=0, atol=0)
        assert destination.device.type == "cpu"
        if operation == "detach":
            assert not destination.requires_grad
        else:
            assert source.data_ptr() != destination.data_ptr()
    assert bool(mapped.finite())


@pytest.mark.parametrize("location", ["pool_sum", "coarse_position", "coarse_edge",
                                      "suffix_output", "suffix_layer", "prefix_layer"])
def test_finite_includes_both_hierarchy_and_raw_prefix_caches(location):
    state = _with_caches(_state(sampling_offset=17))
    tensors = {
        "pool_sum": state.hierarchy.pool.feature_sums,
        "coarse_position": state.hierarchy.pool.graph.graph.positions,
        "coarse_edge": state.hierarchy.pool.graph.graph.edge_attr,
        "suffix_output": state.hierarchy.suffix.outputs,
        "suffix_layer": state.hierarchy.suffix.layer_outputs[0],
        "prefix_layer": state.encoder.layer_outputs[0],
    }
    assert bool(state.finite())
    with torch.no_grad():
        tensors[location].reshape(-1)[0] = float("nan")
    assert not bool(state.finite())


@pytest.mark.parametrize("field", ["encoder", "hierarchy"])
def test_training_capture_rejects_either_learned_cache(field):
    state = replace(_state(sampling_offset=5), **{field: object()})
    with pytest.raises(ValueError, match="raw-graph ANN"):
        state.training_payload()
    with pytest.raises(ValueError, match="raw-graph ANN"):
        capture_training_state(_store(state))


@pytest.mark.parametrize("storage,version", [("materialized", 1), ("implicit", 2)])
def test_v3_safe_serialization_round_trip_preserves_exact_counter_and_contract(storage, version):
    original = _state(storage, sampling_offset=2**70 + 7)
    payload = capture_training_state(_store(original))
    assert payload["version"] == 2  # Independent outer training-context protocol.
    wrapper = payload["entries"][0]["recurrent"]
    assert set(wrapper) == {"schema", "sampling_offset", "raw_state"}
    assert wrapper["schema"] == "asgcn_stream_training_state_v3"
    assert wrapper["sampling_offset"] == original.sampling_offset
    assert wrapper["raw_state"]["schema"] == f"asgcn_stream_training_state_v{version}"
    output = io.BytesIO()
    torch.save(payload, output)
    output.seek(0)
    loaded = torch.load(output, weights_only=True)
    restored = restore_training_state(loaded, independent_sequences=True, device="cpu")
    actual = restored.values[restored.last_key][2]
    assert actual.sampling_offset == original.sampling_offset
    assert actual.contract == original.contract
    assert actual.sequence_identity == original.sequence_identity
    assert actual.encoder is None and actual.hierarchy is None
    assert not actual.decoder.requires_grad
    saved_features = wrapper["raw_state"]["graph"]["node_features"]
    assert len({original.graph.graph.node_features.data_ptr(), saved_features.data_ptr(),
                actual.graph.graph.node_features.data_ptr()}) == 3
    torch.testing.assert_close(actual.graph.graph.node_features,
                               original.graph.graph.node_features, rtol=0, atol=0)


@pytest.mark.parametrize("offset", [-1, True, False, 1.0, "1", None, float("inf"), torch.tensor(1)])
def test_invalid_counter_rejected_in_capture_and_before_restore_transfer(offset, monkeypatch):
    original = _state(sampling_offset=1)
    payload = capture_training_state(_store(original))
    with pytest.raises(ValueError, match="sampling_offset"):
        replace(original, sampling_offset=offset).training_payload()
    with pytest.raises(ValueError, match="sampling_offset"):
        capture_training_state(_store(replace(original, sampling_offset=offset)))
    payload["entries"][0]["recurrent"]["sampling_offset"] = offset
    monkeypatch.setattr(torch.Tensor, "to", lambda *args, **kwargs:
                        pytest.fail("invalid sampling counter must fail before transfer"))
    with pytest.raises(ValueError, match="sampling_offset"):
        restore_training_state(payload, independent_sequences=True, device="cpu")


@pytest.mark.parametrize("corruption", ["extra", "missing", "nested", "unknown_raw",
                                      "nonstring_raw", "live_raw"])
def test_v3_wrapper_strict_fields_and_no_recursive_or_live_raw_state(corruption):
    original = _state(sampling_offset=7)
    payload = copy.deepcopy(original.training_payload())
    if corruption == "extra":
        payload["hierarchy"] = None
    elif corruption == "missing":
        del payload["sampling_offset"]
    elif corruption == "nested":
        payload["raw_state"] = copy.deepcopy(payload)
    elif corruption == "unknown_raw":
        payload["raw_state"]["schema"] = "unknown"
    elif corruption == "nonstring_raw":
        payload["raw_state"]["schema"] = []
    else:
        payload["raw_state"] = original
    with pytest.raises(ValueError, match="fields|v1/v2"):
        restore_stream_training_state(payload)


@pytest.mark.parametrize("storage", ["materialized", "implicit"])
@pytest.mark.parametrize("corruption", ["contract", "raw_extra", "degree", "clock"])
def test_v3_wrapper_does_not_bypass_legacy_raw_state_validation(storage, corruption):
    payload = copy.deepcopy(_state(storage, sampling_offset=7).training_payload())
    raw = payload["raw_state"]
    if corruption == "contract":
        raw["contract"] = "g" * 64
    elif corruption == "raw_extra":
        raw["sampling_offset"] = 7
    elif corruption == "degree":
        raw["graph"]["in_degree"].zero_()
        if storage == "implicit":
            raw["graph"]["edge_counts"].zero_()
    else:
        raw["watermark_seconds"] = 0.05
    with pytest.raises(ValueError):
        restore_stream_training_state(payload)


def test_v3_implicit_trusted_capture_preserves_counter_without_recount_external_restore_recounts(
    monkeypatch,
):
    original = _state("implicit", sampling_offset=37)
    method = ImplicitRadiusIndex.iter_directed_neighbors
    monkeypatch.setattr(ImplicitRadiusIndex, "iter_directed_neighbors", lambda *args, **kwargs:
                        pytest.fail("trusted capture must not enumerate implicit edges"))
    payload = capture_training_state(_store(original))
    assert payload["entries"][0]["recurrent"]["sampling_offset"] == 37
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(len(self.positions))
        yield from method(self, *args, **kwargs)

    monkeypatch.setattr(ImplicitRadiusIndex, "iter_directed_neighbors", counted)
    restored = restore_training_state(payload, independent_sequences=True, device="cpu")
    assert calls == [3]
    assert restored.values[restored.last_key][2].sampling_offset == 37


def test_v3_trusted_capture_still_rejects_mutated_implicit_graph():
    original = _state("implicit", sampling_offset=37)
    original.graph.graph.in_degree.zero_()
    original.graph.graph.edge_counts.zero_()
    with pytest.raises(RuntimeError, match="modified"):
        capture_training_state(_store(original))


def test_explicit_zero_v3_wrapper_restores_but_canonical_write_keeps_legacy_schema():
    raw = _state().training_payload()
    restored = restore_stream_training_state({"schema": "asgcn_stream_training_state_v3",
                                             "raw_state": raw, "sampling_offset": 0})
    assert restored.sampling_offset == 0
    assert restored.training_payload()["schema"] == "asgcn_stream_training_state_v1"
