from __future__ import annotations

import math

import pytest
import torch

from asgcn_unet import graph


@pytest.fixture(autouse=True)
def clear_hash_cache():
    graph._spatial_hash_constants.cache_clear()
    yield
    graph._spatial_hash_constants.cache_clear()


def test_lookup_batches_scalar_transfers_without_merging_candidate_chunks(monkeypatch):
    positions = torch.rand((31, 4), generator=torch.Generator().manual_seed(31))
    original = torch.Tensor.tolist
    transfers = []

    def record_transfer(tensor):
        transfers.append((tensor.dtype, tensor.shape))
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", record_transfer)
    chunks = list(graph._radius_graph_candidate_chunks(positions, 0.3, chunk_size=2))
    # Sixteen original source chunks share two count transfers. Public finite and
    # range validation still happens, in a separate single paired flag transfer.
    assert transfers == [(torch.bool, (2,)), (torch.long, (8,)), (torch.long, (8,))]
    assert len(chunks) == 16
    for index, (sources, destinations, distances) in enumerate(chunks):
        start = index * 2
        assert torch.all((sources >= start) & (sources < min(start + 2, 31)))
        assert sources.shape == destinations.shape == distances.shape


def test_lookup_tables_stay_bounded_when_requested_chunk_is_tiny(monkeypatch):
    positions = torch.rand((129, 4), generator=torch.Generator().manual_seed(91))
    original = torch.searchsorted
    lookup_source_counts = []

    def record_lookup(sorted_sequence, values, **kwargs):
        lookup_source_counts.append(values.numel() // 27)
        return original(sorted_sequence, values, **kwargs)

    monkeypatch.setattr(torch, "searchsorted", record_lookup)
    list(graph._radius_graph_candidate_chunks(positions, 0.08, chunk_size=1))
    assert len(lookup_source_counts) == math.ceil(129 / 8) * 2
    assert max(lookup_source_counts) == 8


def test_worst_case_candidate_cap_remains_four_million(monkeypatch):
    positions = torch.full((9000, 4), 0.5)
    allocations = []

    class AllocationObserved(Exception):
        pass

    def record_without_allocating(_groups, _counts, *, output_size):
        allocations.append(output_size)
        raise AllocationObserved

    monkeypatch.setattr(torch, "repeat_interleave", record_without_allocating)
    candidates = graph._radius_graph_candidate_chunks(positions, 0.08, chunk_size=9000)
    with pytest.raises(AllocationObserved):
        next(candidates)
    assert allocations == [(4_000_000 // 9000) * 9000]
    assert allocations[0] <= 4_000_000


def test_edge_guard_stops_after_first_materialized_chunk(monkeypatch):
    positions = torch.full((31, 4), 0.5)
    original = torch.linalg.vector_norm
    materializations = []

    def record_norm(values, **kwargs):
        materializations.append(values.shape)
        return original(values, **kwargs)

    monkeypatch.setattr(torch.linalg, "vector_norm", record_norm)
    with pytest.raises(RuntimeError, match="max_graph_edges=1"):
        graph.build_radius_graph(positions, 0.08, chunk_size=2, max_edges=1)
    assert materializations == [(2 * 31, 3)]


def test_position_gradients_match_dense_oracle_across_lookup_blocks():
    initial = torch.rand((37, 4), generator=torch.Generator().manual_seed(86),
                         dtype=torch.float64)
    positions = initial.clone().requires_grad_()
    reference = initial.clone().requires_grad_()
    edges, attributes = graph.build_radius_graph(positions, 0.5, chunk_size=2)
    distances = torch.linalg.vector_norm(reference[:, None, :3] - reference[None, :, :3],
                                         dim=-1)
    valid = (distances < 0.5) & ~torch.eye(37, dtype=torch.bool)
    assert torch.equal(edges, valid.nonzero().t())
    expected = (distances[valid] / 0.5).unsqueeze(-1)
    assert torch.equal(attributes, expected)
    weights = torch.linspace(0.1, 0.9, len(attributes), dtype=torch.float64)[:, None]
    actual_gradient = torch.autograd.grad((attributes * weights).sum(), positions)[0]
    expected_gradient = torch.autograd.grad((expected * weights).sum(), reference)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12)


def test_cached_tables_reused_and_not_mutated_by_graph_queries(monkeypatch):
    original = torch.cartesian_prod
    creations = []

    def record_cartesian(*arrays):
        creations.append(len(arrays))
        return original(*arrays)

    monkeypatch.setattr(torch, "cartesian_prod", record_cartesian)
    positions = torch.rand((19, 4), generator=torch.Generator().manual_seed(222))
    graph.build_radius_graph(positions, 0.08, chunk_size=2)
    strides, offsets = graph._spatial_hash_constants(14, 3, "cpu", None)
    original_strides, original_offsets = strides.clone(), offsets.clone()
    graph.build_radius_graph(positions.flip(0), 0.08, chunk_size=5)
    graph.radius_graph_topology(positions, 0.08, chunk_size=7)
    next_strides, next_offsets = graph._spatial_hash_constants(14, 3, "cpu", None)
    assert next_strides is strides and next_offsets is offsets
    assert torch.equal(strides, original_strides)
    assert torch.equal(offsets, original_offsets)
    assert creations == [3]
    assert graph._spatial_hash_constants.cache_info().currsize == 1


def test_cache_is_bounded_and_higher_dimensional_queries_are_not_retained():
    for cells_per_axis in range(2, 50):
        graph._spatial_hash_constants(cells_per_axis, 3, "cpu", None)
    assert graph._spatial_hash_constants.cache_info().maxsize == 32
    assert graph._spatial_hash_constants.cache_info().currsize == 32
    graph._spatial_hash_constants.cache_clear()
    positions = torch.zeros((2, 5))
    edges, attributes = graph.build_radius_graph(positions, 0.4, position_dims=5)
    assert edges.shape == (2, 2)
    assert torch.count_nonzero(attributes) == 0
    assert graph._spatial_hash_constants.cache_info().currsize == 0


@pytest.mark.parametrize("operation", [graph.build_radius_graph, graph.radius_graph_topology])
def test_combined_coordinate_validation_keeps_error_priority(operation):
    positions = torch.tensor([[float("nan"), 2.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="Graph coordinates must be finite"):
        operation(positions, 0.1)


def test_combined_event_validation_keeps_error_priority_and_one_transfer(monkeypatch):
    events = torch.tensor([[float("nan"), 0, 1, 1], [0, 0, 0, 1]])
    original = torch.Tensor.tolist
    transfers = []

    def record_transfer(tensor):
        transfers.append((tensor.dtype, tensor.shape))
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", record_transfer)
    with pytest.raises(ValueError, match="must be finite"):
        graph.prepare_event_nodes(events, (8, 8))
    assert transfers == [(torch.bool, (2,))]
    events[0, 0] = 0
    with pytest.raises(ValueError, match="monotonically non-decreasing"):
        graph.prepare_event_nodes(events, (8, 8))


@pytest.mark.parametrize("backend", ["torch", "torch_fused", "triton"])
def test_encoder_forwards_explicit_backend_without_changing_state_dict(backend):
    default = graph.ASGCNEncoder(hidden_dim=4, graph_layers=2)
    selected = graph.ASGCNEncoder(hidden_dim=4, graph_layers=2, spline_backend=backend)
    assert all(layer.spline_backend == backend for layer in selected.layers)
    selected.load_state_dict(default.state_dict(), strict=True)
    assert selected.state_dict().keys() == default.state_dict().keys()


def test_invalid_backend_is_rejected_at_construction():
    with pytest.raises(ValueError, match="spline_backend"):
        graph.PaperSplineConv(4, 4, spline_backend="automatic-fallback")


def test_explicit_triton_backend_rejects_cpu_even_for_empty_graph():
    layer = graph.PaperSplineConv(4, 4, spline_backend="triton")
    with pytest.raises(ValueError, match="requires CUDA"):
        layer.spline_aggregate(torch.empty((0, 4)), torch.empty((2, 0), dtype=torch.long),
                               torch.empty((0, 1)))


@pytest.mark.parametrize("mode", ["ann", "snn"])
def test_encoder_backend_validation_cannot_be_bypassed_with_empty_graph(mode):
    encoder = graph.ASGCNEncoder(hidden_dim=4, graph_layers=1, spline_backend="triton")
    encoder.layers[0]._snn_is_normalized = True
    empty_graph = graph.EventGraph(torch.empty((0, 4)), torch.empty((0, 4)),
                                   torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1)))
    with pytest.raises(ValueError, match="requires CUDA"):
        getattr(encoder, f"forward_{mode}")(empty_graph)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA accelerator unavailable")
def test_hash_cache_isolated_between_cuda_streams():
    positions = torch.rand((31, 4), generator=torch.Generator().manual_seed(12)).cuda()
    stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
    stream_a.wait_stream(torch.cuda.current_stream())
    stream_b.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream_a):
        edges_a, attr_a = graph.build_radius_graph(positions, 0.3, chunk_size=2)
        constants_a = graph._spatial_hash_constants(5, 3, str(positions.device),
                                                  stream_a.cuda_stream)
    with torch.cuda.stream(stream_b):
        edges_b, attr_b = graph.build_radius_graph(positions, 0.3, chunk_size=2)
        constants_b = graph._spatial_hash_constants(5, 3, str(positions.device),
                                                  stream_b.cuda_stream)
    torch.cuda.current_stream().wait_stream(stream_a)
    torch.cuda.current_stream().wait_stream(stream_b)
    assert constants_a[0].data_ptr() != constants_b[0].data_ptr()
    assert constants_a[1].data_ptr() != constants_b[1].data_ptr()
    assert torch.equal(edges_a, edges_b)
    assert torch.equal(attr_a, attr_b)
    assert graph._spatial_hash_constants.cache_info().currsize == 2
