"""Bounded synthetic CPU count-kernel tests, not research performance results."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from asgcn_unet import stream_graph, stream_topology
from asgcn_unet.stream_graph import evolve_stream_graph
from asgcn_unet.stream_topology import count_stream_topology, count_stream_topology_update


@pytest.fixture(autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _count(points, batches=None, mask=None, *, batch_size=1, **kwargs):
    positions = torch.as_tensor(points, dtype=torch.float64).reshape(-1, 4)
    count = positions.shape[0]
    node_batch = torch.tensor([0] * count if batches is None else batches, dtype=torch.long)
    readout = torch.tensor([True] * count if mask is None else mask, dtype=torch.bool)
    return count_stream_topology(positions, node_batch, readout, batch_size=batch_size,
                                 radius=kwargs.pop("radius", 1.0), **kwargs)


def _assert_reference(positions, batches, mask, result, batch_size, radius=1.0, position_dims=3):
    coordinates = positions[:, :position_dims].double()
    # Only this tiny CPU test oracle materializes the N*N distance matrix.
    distances = torch.linalg.vector_norm((coordinates[:, None] - coordinates[None, :]) / radius, dim=-1)
    valid = (distances < 1) & (batches[:, None] == batches[None, :])
    valid.fill_diagonal_(False)
    sources = torch.nonzero(valid, as_tuple=True)[0]
    kept_sources = torch.nonzero(valid & mask[:, None] & mask[None, :], as_tuple=True)[0]
    torch.testing.assert_close(result.union_nodes, torch.bincount(batches, minlength=batch_size))
    torch.testing.assert_close(result.readout_nodes, torch.bincount(batches[mask], minlength=batch_size))
    torch.testing.assert_close(result.union_directed_edges, torch.bincount(batches[sources], minlength=batch_size))
    torch.testing.assert_close(result.readout_directed_edges, torch.bincount(batches[kept_sources], minlength=batch_size))


@pytest.mark.parametrize("position_dims", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("chunk_size,budget", [(1, 7), (6, 19), (512, 1_048_576)])
def test_packed_counts_match_dense_reference_and_real_updater(position_dims, dtype, chunk_size, budget):
    generator = torch.Generator().manual_seed(912)
    positions = torch.rand((27, 4), generator=generator, dtype=dtype) * 4 - 2
    batches = torch.randint(0, 3, (27,), generator=generator)
    mask = torch.rand(27, generator=generator) > 0.4
    result = count_stream_topology(positions, batches, mask, batch_size=4, radius=1.4,
                                   position_dims=position_dims, chunk_size=chunk_size,
                                   candidate_pair_budget=budget)
    _assert_reference(positions, batches, mask, result, 4, 1.4, position_dims)
    update = evolve_stream_graph(
        None, torch.zeros((27, 4)), positions, torch.zeros(27, dtype=torch.float64), batches,
        torch.full((4,), -1.0, dtype=torch.float64), radius=1.4, position_dims=position_dims,
        chunk_size=chunk_size, max_graph_edges=None,
    )
    edges = update.state.graph.edge_index
    torch.testing.assert_close(result.union_directed_edges, torch.bincount(batches[edges[0]], minlength=4))
    kept = mask[edges[0]] & mask[edges[1]]
    torch.testing.assert_close(result.readout_directed_edges, torch.bincount(batches[edges[0, kept]], minlength=4))
    assert result.peak_candidate_pairs <= budget
    assert result.peak_query_cells <= chunk_size * 3**position_dims


@pytest.mark.parametrize("budget", [1, 5, 17])
def test_single_dense_cell_chunks_candidate_pairs_without_truncation(monkeypatch, budget):
    count = 23
    positions = torch.zeros((count, 4), dtype=torch.float64)
    batches = torch.zeros(count, dtype=torch.long)
    mask = torch.arange(count) % 3 == 0
    observed_norm_rows = []
    original_norm = torch.linalg.vector_norm

    def bounded_norm(values, *args, **kwargs):
        observed_norm_rows.append(values.shape[0])
        assert values.shape[0] <= budget
        return original_norm(values, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Count-only diagnostics must not construct or guard a graph")

    monkeypatch.setattr(torch.linalg, "vector_norm", bounded_norm)
    monkeypatch.setattr(torch, "repeat_interleave", forbidden)
    monkeypatch.setattr(stream_graph, "evolve_stream_graph", forbidden)
    monkeypatch.setattr(stream_graph, "_check_guard", forbidden)
    monkeypatch.setattr(stream_graph.EventGraph, "__init__", forbidden)
    result = count_stream_topology(positions, batches, mask, batch_size=1, radius=1,
                                   chunk_size=512, candidate_pair_budget=budget)
    kept = int(mask.sum())
    assert result.union_directed_edges.tolist() == [count * (count - 1)]
    assert result.readout_directed_edges.tolist() == [kept * (kept - 1)]
    assert result.candidate_pairs_visited == count**2
    assert result.peak_candidate_pairs == budget
    assert result.candidate_chunks == (count**2 + budget - 1) // budget
    assert observed_norm_rows and max(observed_norm_rows) <= budget
    assert all(getattr(result, field.name).shape == (1,) for field in fields(result)
               if isinstance(getattr(result, field.name), torch.Tensor))


def test_strict_float64_boundary_and_readout_induced_subgraph():
    inside = torch.nextafter(torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0)).item()
    outside = torch.nextafter(torch.tensor(1.0, dtype=torch.float64), torch.tensor(2.0)).item()
    points = [[0, 0, 0, 0], [inside, 0, 0, 0], [1, 0, 0, 0], [outside, 0, 0, 0]]
    for index, expected in [(1, 2), (2, 0), (3, 0)]:
        result = _count(points, mask=[i in (0, index) for i in range(4)], candidate_pair_budget=2)
        assert result.readout_directed_edges.tolist() == [expected]
        assert result.union_directed_edges.tolist() == [8]


def test_negative_large_coordinates_and_identical_points_never_connect_streams():
    points = [[-1e15, -2, -1e12, 0], [1e15, 4, 1e12, 0],
              [-1e15 + 0.5, -2, -1e12, 0], [1e15 + 0.5, 4, 1e12, 0]]
    result = _count(points, [0, 2, 0, 2], [True, True, False, True], batch_size=4, candidate_pair_budget=3)
    assert result.union_nodes.tolist() == [2, 0, 2, 0]
    assert result.union_directed_edges.tolist() == [2, 0, 2, 0]
    assert result.readout_directed_edges.tolist() == [0, 0, 2, 0]
    result = _count([[0, 0, 0, 0]] * 6, [0, 1, 0, 1, 0, 1], batch_size=2)
    assert result.union_directed_edges.tolist() == [6, 6]


@pytest.mark.parametrize("radius,coordinate", [(2e200, 1e200), (2e-200, 1e-200)])
def test_distance_is_scaled_before_norm_without_overflow_or_underflow(radius, coordinate):
    result = _count([[0, 0, 0, 0], [coordinate, 0, 0, 0]], radius=radius, candidate_pair_budget=1)
    assert result.union_directed_edges.tolist() == [2]


@pytest.mark.parametrize("batch_size", [0, 3])
def test_empty_input_needs_no_cell_lookup(monkeypatch, batch_size):
    monkeypatch.setattr(stream_topology, "_occupied_cells", lambda *args: pytest.fail("empty input needs no cell index"))
    result = _count([], batch_size=batch_size)
    assert result.union_nodes.tolist() == result.union_directed_edges.tolist() == [0] * batch_size
    assert result.readout_nodes.tolist() == result.readout_directed_edges.tolist() == [0] * batch_size
    assert result.peak_candidate_pairs == result.candidate_pairs_visited == result.query_chunks == 0


def test_no_edges_or_readout_nodes_are_not_substituted():
    result = _count([[0, 0, 0, 0], [5, 5, 5, 0]], mask=[False, False], candidate_pair_budget=1)
    assert result.union_nodes.tolist() == [2]
    assert result.union_directed_edges.tolist() == result.readout_nodes.tolist() == result.readout_directed_edges.tolist() == [0]
    assert result.candidate_pairs_visited == 2  # Self candidates are examined, never edges.


def test_inputs_are_unchanged_and_no_autograd_history_is_retained():
    positions = torch.tensor([[0., 0, 0, 0], [0.5, 0, 0, 0]], dtype=torch.float64, requires_grad=True)
    batches = torch.zeros(2, dtype=torch.long)
    mask = torch.tensor([True, False])
    before = [value.clone() for value in (positions, batches, mask)]
    result = count_stream_topology(positions, batches, mask, batch_size=1, radius=1)
    for actual, expected in zip((positions, batches, mask), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(not getattr(result, field.name).requires_grad for field in fields(result)
               if isinstance(getattr(result, field.name), torch.Tensor))


@pytest.mark.parametrize("keyword,value", [
    ("radius", 0), ("radius", True), ("radius", float("nan")), ("radius", float("inf")),
    ("position_dims", 0), ("position_dims", 5), ("position_dims", True),
    ("chunk_size", 0), ("chunk_size", True), ("candidate_pair_budget", 0),
    ("candidate_pair_budget", 1.5), ("candidate_pair_budget", True),
    ("batch_size", -1), ("batch_size", True),
])
def test_invalid_configuration_is_not_repaired(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        _count([], **{keyword: value})


@pytest.mark.parametrize("positions,batches,mask,message", [
    (torch.zeros(2, 3), torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool), "positions"),
    (torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool), "positions"),
    (torch.zeros(2, 4), torch.zeros(2), torch.ones(2, dtype=torch.bool), "node_batch"),
    (torch.zeros(2, 4), torch.zeros(2, dtype=torch.long), torch.ones(2), "readout_mask"),
    (torch.zeros(2, 4), torch.tensor([0, 1]), torch.ones(2, dtype=torch.bool), "valid batch"),
    (torch.zeros(2, 4), torch.tensor([0, -1]), torch.ones(2, dtype=torch.bool), "valid batch"),
    (torch.full((2, 4), float("nan")), torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool), "finite"),
])
def test_invalid_input_is_rejected(positions, batches, mask, message):
    with pytest.raises(ValueError, match=message):
        count_stream_topology(positions, batches, mask, batch_size=1, radius=1)


def test_numeric_cell_address_overflow_fails_without_truncation():
    with pytest.raises(ValueError, match="float64 cell addressing"):
        _count([[-1e20, 0, 0, 0], [1e20, 0, 0, 0]])


def _assert_update_matches_full(positions, batches, union_mask, readout_mask, arrivals, cached,
                                *, batch_size, radius=1.0, position_dims=3, budget=7, chunk_size=4):
    result = count_stream_topology_update(
        positions, batches, union_mask, readout_mask, arrivals, cached,
        batch_size=batch_size, radius=radius, position_dims=position_dims,
        chunk_size=chunk_size, candidate_pair_budget=budget,
    )
    expected = count_stream_topology(
        positions[union_mask], batches[union_mask], readout_mask[union_mask],
        batch_size=batch_size, radius=radius, position_dims=position_dims,
        chunk_size=chunk_size, candidate_pair_budget=budget,
    )
    for field in ("union_nodes", "union_directed_edges", "readout_nodes", "readout_directed_edges"):
        torch.testing.assert_close(getattr(result, field), getattr(expected, field), rtol=0, atol=0)
    assert result.peak_candidate_pairs <= budget
    assert result.peak_query_cells <= chunk_size * 3**position_dims
    return result


@pytest.mark.parametrize("position_dims", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_incremental_multiple_packed_frames_gaps_expirations_and_late_arrivals(position_dims, dtype):
    generator = torch.Generator().manual_seed(2291)
    positions = torch.empty((0, 4), dtype=dtype)
    batches = torch.empty(0, dtype=torch.long)
    times = torch.empty(0, dtype=torch.float64)
    cached = torch.zeros(4, dtype=torch.long)
    ends = torch.zeros(4, dtype=torch.float64)
    for step in range(6):
        starts = ends + torch.tensor([0.0, 0.5 if step % 2 else 0.0, 2.0 if step % 2 else 0.0, 0.0])
        ends = starts + torch.tensor([0.4, 0.3, 0.7, 0.2])
        new_batches = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])  # Lane 3 remains empty.
        new_times = torch.cat((starts[:3] - 1.3, starts[:3] + 0.1, ends[:3]))
        new_positions = torch.rand((9, 4), generator=generator, dtype=dtype)
        new_positions[:, 2] = (new_times / 10).to(dtype)
        arrivals = torch.cat((torch.zeros(len(times), dtype=torch.bool), torch.ones(9, dtype=torch.bool)))
        positions = torch.cat((positions, new_positions))
        times = torch.cat((times, new_times))
        batches = torch.cat((batches, new_batches))
        union_mask = times >= (starts - 1.0)[batches]
        readout_mask = times >= (ends - 1.0)[batches]
        result = _assert_update_matches_full(
            positions, batches, union_mask, readout_mask, arrivals, cached,
            batch_size=4, radius=0.8, position_dims=position_dims, budget=11,
        )
        positions, batches, times = positions[readout_mask], batches[readout_mask], times[readout_mask]
        cached = result.readout_directed_edges


@pytest.mark.parametrize("budget", [1, 5, 17])
def test_incremental_dense_queries_only_changed_endpoints_and_respects_budget(monkeypatch, budget):
    count, old_count = 23, 21
    positions = torch.zeros((count, 4), dtype=torch.float64)
    batches = torch.zeros(count, dtype=torch.long)
    arrivals = torch.arange(count) >= old_count
    union_mask = torch.ones(count, dtype=torch.bool)
    union_mask[0] = union_mask[-1] = False  # Old expiry and an already-expired new arrival.
    readout_mask = union_mask.clone()
    readout_mask[1] = False
    cached = torch.tensor([old_count * (old_count - 1)])
    selected_count = int((arrivals | ~readout_mask).sum())
    observed = []
    original_norm = torch.linalg.vector_norm

    def bounded_norm(values, *args, **kwargs):
        observed.append(len(values))
        assert len(values) <= budget
        return original_norm(values, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg, "vector_norm", bounded_norm)
        result = count_stream_topology_update(
            positions, batches, union_mask, readout_mask, arrivals, cached,
            batch_size=1, radius=1, chunk_size=512, candidate_pair_budget=budget,
        )
    assert result.union_directed_edges.tolist() == [21 * 20]
    assert result.readout_directed_edges.tolist() == [20 * 19]
    assert result.candidate_pairs_visited == selected_count * count < count**2
    expected_incident_pairs = selected_count * (count - selected_count) + selected_count * (selected_count - 1) // 2
    assert sum(observed) == expected_incident_pairs
    assert result.peak_candidate_pairs == budget
    assert cached.tolist() == [old_count * (old_count - 1)]


@pytest.mark.parametrize("keep_union", [False, True])
def test_expiration_only_and_all_old_nodes_expired(keep_union):
    positions = torch.zeros((6, 4), dtype=torch.float64)
    batches = torch.tensor([0, 1, 0, 1, 0, 1])
    union_mask = torch.full((6,), keep_union, dtype=torch.bool)
    readout_mask = torch.zeros(6, dtype=torch.bool)
    result = _assert_update_matches_full(
        positions, batches, union_mask, readout_mask, torch.zeros(6, dtype=torch.bool),
        torch.tensor([6, 6]), batch_size=2,
    )
    assert result.readout_directed_edges.tolist() == [0, 0]
    assert result.union_directed_edges.tolist() == ([6, 6] if keep_union else [0, 0])


def test_incremental_preserves_equal_cutoff_ties_and_strict_radius_boundary():
    inside = torch.nextafter(torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0)).item()
    positions = torch.tensor([[0., 0, 0, 0], [inside, 0, 0, 0], [1., 0, 0, 0], [0., 0, 0, 0]])
    positions = positions.double()
    positions[1, 0] = inside  # Keep the one-ULP distinction in float64.
    batches = torch.zeros(4, dtype=torch.long)
    times = torch.tensor([1.0, 1.0, 2.0, 0.999], dtype=torch.float64)
    union_mask, readout_mask = times >= 0.5, times >= 1.0
    result = _assert_update_matches_full(
        positions, batches, union_mask, readout_mask, torch.tensor([False, True, True, True]),
        torch.zeros(1, dtype=torch.long), batch_size=1, position_dims=1, budget=2,
    )
    assert result.readout_nodes.tolist() == [3]
    assert result.readout_directed_edges.tolist() == [4]


def test_interleaved_arrival_identity_handles_selected_target_larger_than_source():
    positions = torch.zeros((6, 4), dtype=torch.float64)
    batches = torch.tensor([0, 1, 0, 1, 0, 1])
    arrivals = torch.tensor([True, False, False, True, False, False])
    union_mask = torch.tensor([True, True, False, True, True, True])
    readout_mask = torch.tensor([True, False, False, True, True, True])
    _assert_update_matches_full(positions, batches, union_mask, readout_mask, arrivals,
                                torch.tensor([2, 2]), batch_size=2, budget=1)


def test_unchanged_old_stream_does_not_build_index_or_recompute_distances(monkeypatch):
    positions = torch.zeros((4, 4), dtype=torch.float64)
    batches = torch.tensor([0, 1, 0, 1])
    mask = torch.ones(4, dtype=torch.bool)
    cached = torch.tensor([2, 2])
    def forbidden(*args, **kwargs):
        raise AssertionError("Unchanged old pairs must use cached counts without spatial work")
    monkeypatch.setattr(stream_topology, "_occupied_cells", forbidden)
    monkeypatch.setattr(torch.linalg, "vector_norm", forbidden)
    result = count_stream_topology_update(positions, batches, mask, mask, ~mask, cached,
                                          batch_size=2, radius=1)
    assert result.union_directed_edges.tolist() == result.readout_directed_edges.tolist() == [2, 2]
    assert result.query_chunks == result.candidate_pairs_visited == 0
    assert result.union_directed_edges.data_ptr() != cached.data_ptr()
    assert result.readout_directed_edges.data_ptr() != cached.data_ptr()


def test_incremental_empty_batch_and_first_frame_all_arrivals():
    empty_positions = torch.empty((0, 4), dtype=torch.float64)
    empty_batches, empty_mask = torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool)
    result = count_stream_topology_update(empty_positions, empty_batches, empty_mask, empty_mask,
                                          empty_mask, empty_batches, batch_size=0, radius=1)
    assert result.union_directed_edges.numel() == result.readout_directed_edges.numel() == 0
    positions = torch.zeros((5, 4), dtype=torch.float64)
    mask = torch.ones(5, dtype=torch.bool)
    _assert_update_matches_full(positions, torch.zeros(5, dtype=torch.long), mask, mask, mask,
                                torch.zeros(1, dtype=torch.long), batch_size=1)


@pytest.mark.parametrize("cache", [torch.tensor([-2]), torch.tensor([3]), torch.tensor([8]),
                                    torch.tensor([2.0]), torch.tensor([2, 2])])
def test_incremental_invalid_cache_rejected(cache):
    positions = torch.zeros((2, 4), dtype=torch.float64)
    mask = torch.ones(2, dtype=torch.bool)
    with pytest.raises(ValueError, match="previous_edge_counts"):
        count_stream_topology_update(positions, torch.zeros(2, dtype=torch.long), mask, mask, ~mask,
                                     cache, batch_size=1, radius=1)


@pytest.mark.parametrize("field", ["union_mask", "is_arrival"])
def test_incremental_invalid_masks_rejected(field):
    positions = torch.zeros((2, 4), dtype=torch.float64)
    mask = torch.ones(2, dtype=torch.bool)
    options = {"union_mask": mask, "readout_mask": mask, "is_arrival": ~mask}
    options[field] = torch.ones(2)
    with pytest.raises(ValueError, match=field):
        count_stream_topology_update(positions, torch.zeros(2, dtype=torch.long),
                                     previous_edge_counts=torch.tensor([2]), batch_size=1, radius=1, **options)


def test_incremental_readout_outside_union_is_not_repaired():
    positions = torch.zeros((2, 4), dtype=torch.float64)
    mask = torch.ones(2, dtype=torch.bool)
    with pytest.raises(ValueError, match="subset"):
        count_stream_topology_update(positions, torch.zeros(2, dtype=torch.long), ~mask, mask, ~mask,
                                     torch.tensor([2]), batch_size=1, radius=1)


def test_inconsistent_cache_failure_does_not_mutate_any_inputs():
    positions = torch.zeros((2, 4), dtype=torch.float64, requires_grad=True)
    batches = torch.zeros(2, dtype=torch.long)
    mask = torch.zeros(2, dtype=torch.bool)
    cached = torch.tensor([0])  # Feasible by cardinality, but wrong for these connected old nodes.
    snapshots = [value.clone() for value in (positions, batches, mask, cached)]
    with pytest.raises(ValueError, match="cached previous_edge_counts disagree"):
        count_stream_topology_update(positions, batches, mask, mask, mask, cached, batch_size=1, radius=1)
    for value, snapshot in zip((positions, batches, mask, cached), snapshots, strict=True):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
