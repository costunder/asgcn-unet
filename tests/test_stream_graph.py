"""CPU synthetic topology tests; these are not reconstruction quality results."""

from __future__ import annotations

import pytest
import torch

from asgcn_unet.stream_graph import StreamGraph, evolve_stream_graph


def _arrival(points, times=None, batches=None, *, requires_grad=False):
    positions = torch.as_tensor(points, dtype=torch.float64).reshape(-1, 4)
    features = positions.float().clone().requires_grad_(requires_grad)
    count = positions.shape[0]
    timestamps = torch.tensor(times if times is not None else [0.0] * count, dtype=torch.float64)
    node_batch = torch.tensor(batches if batches is not None else [0] * count, dtype=torch.long)
    return features, positions, timestamps, node_batch


def _evolve(previous, points, times=None, batches=None, *, cutoffs=(-1.0,), radius=1.0,
            max_graph_edges=None, chunk_size=2, position_dims=3):
    return evolve_stream_graph(
        previous, *_arrival(points, times, batches), torch.tensor(cutoffs, dtype=torch.float64),
        radius=radius, max_graph_edges=max_graph_edges, chunk_size=chunk_size,
        position_dims=position_dims,
    )


def _reference(state: StreamGraph, radius: float, position_dims: int = 3):
    graph = state.graph
    coordinates = graph.positions[:, :position_dims].double()
    distances = torch.linalg.vector_norm(
        (coordinates[:, None] - coordinates[None, :]) / radius, dim=-1,
    )
    valid = (distances < 1.0) & (state.node_batch[:, None] == state.node_batch[None, :])
    valid.fill_diagonal_(False)
    edges = torch.nonzero(valid, as_tuple=False).T
    return edges, distances[valid].to(graph.positions.dtype).unsqueeze(1)


def _assert_reference(state, radius=1.0, position_dims=3):
    graph = state.graph
    expected, attr = _reference(state, radius, position_dims)
    count = graph.node_features.shape[0]
    order = torch.argsort(graph.edge_index[0] * count + graph.edge_index[1])
    torch.testing.assert_close(graph.edge_index[:, order], expected)
    torch.testing.assert_close(graph.edge_attr[order], attr, rtol=0, atol=0)
    torch.testing.assert_close(graph.in_degree, torch.bincount(expected[1], minlength=count))


def _snapshot(state):
    return [value.clone() for value in (
        state.graph.node_features, state.graph.positions, state.graph.edge_index,
        state.graph.edge_attr, state.graph.in_degree, state.node_batch, state.timestamps,
    )]


def _assert_unchanged(state, snapshot):
    for actual, expected in zip((
        state.graph.node_features, state.graph.positions, state.graph.edge_index,
        state.graph.edge_attr, state.graph.in_degree, state.node_batch, state.timestamps,
    ), snapshot, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("chunk_size", [1, 2, 512])
def test_append_reuses_old_edges_and_remaps_cache_indices(chunk_size):
    first = _evolve(None, [[0, 0, 0, 0], [0.25, 0, 0, 0], [4, 0, 0, 0]])
    snapshot = _snapshot(first.state)
    second = _evolve(first.state, [[0.5, 0, 0, 0], [5, 0, 0, 0]], chunk_size=chunk_size)
    _assert_reference(second.state)
    _assert_unchanged(first.state, snapshot)
    torch.testing.assert_close(second.old_indices, torch.tensor([0, 1, 2, -1, -1]))
    torch.testing.assert_close(second.changed_nodes, torch.tensor([True, True, False, True, True]))
    old_edges = first.state.graph.edge_index.shape[1]
    torch.testing.assert_close(second.state.graph.edge_index[:, :old_edges], first.state.graph.edge_index)
    torch.testing.assert_close(second.state.graph.edge_attr[:old_edges], first.state.graph.edge_attr)


def test_expiration_marks_lost_neighbors_and_keeps_cutoff_ties():
    first = _evolve(None, [[0, 0, 0, 0], [0.2, 0, 0, 0], [4, 0, 0, 0]], [0, 1, 1])
    update = _evolve(first.state, [], cutoffs=(1.0,))
    torch.testing.assert_close(update.old_indices, torch.tensor([1, 2]))
    torch.testing.assert_close(update.changed_nodes, torch.tensor([True, False]))
    torch.testing.assert_close(update.state.timestamps, torch.tensor([1.0, 1.0], dtype=torch.float64))
    _assert_reference(update.state)


def test_neighbor_replacement_marks_changed_even_when_degree_is_unchanged():
    first = _evolve(None, [[0, 0, 0, 0], [0.5, 0, 0, 0]], [0, 1])
    update = _evolve(first.state, [[1, 0, 0, 0]], [2], cutoffs=(1.0,))
    torch.testing.assert_close(update.state.graph.in_degree, torch.tensor([1, 1]))
    assert bool(update.changed_nodes.all())
    _assert_reference(update.state)


def test_no_arrivals_does_not_query_cells(monkeypatch):
    first = _evolve(None, [[0, 0, 0, 0], [0.5, 0, 0, 0]])
    def forbidden(*args, **kwargs):
        raise AssertionError("No cell query is needed without arrivals")
    monkeypatch.setattr("asgcn_unet.stream_graph._occupied_cells", forbidden)
    update = _evolve(first.state, [])
    assert not bool(update.changed_nodes.any())
    _assert_reference(update.state)


def test_only_arrivals_issue_queries_and_full_graph_builder_is_never_used(monkeypatch):
    first = _evolve(None, [[float(i) / 20, 0, 0, 0] for i in range(20)])
    from asgcn_unet import stream_graph
    lookup = stream_graph._cell_lookup
    queried = []
    def record(rows, queries):
        queried.append(queries.shape[0])
        return lookup(rows, queries)
    def forbidden(*args, **kwargs):
        raise AssertionError("A stream update must not rebuild a full radius graph")
    monkeypatch.setattr(stream_graph, "_cell_lookup", record)
    monkeypatch.setattr("asgcn_unet.graph.build_radius_graph", forbidden)
    monkeypatch.setattr("asgcn_unet.graph.build_event_graph", forbidden)
    update = _evolve(first.state, [[0.7, 0, 0, 0], [0.9, 0, 0, 0]])
    assert sum(queried) == 2 * 27
    _assert_reference(update.state)


@pytest.mark.parametrize("position_dims", [1, 2, 3, 4])
def test_reference_for_all_supported_position_dimensions(position_dims):
    points = [[-2, 0, 0, 0], [-1.8, 0.2, 0.2, 1], [-1.7, 0.2, 0.2, 0]]
    update = _evolve(None, points, position_dims=position_dims)
    _assert_reference(update.state, position_dims=position_dims)


def test_coincident_zero_feature_nodes_have_topological_degree():
    update = _evolve(None, [[0, 0, 0, 0]] * 4)
    assert not bool(update.state.graph.node_features.any())
    torch.testing.assert_close(update.state.graph.in_degree, torch.full((4,), 3))
    assert update.state.graph.edge_index.shape == (2, 12)
    assert not bool(update.state.graph.edge_attr.any())
    _assert_reference(update.state)


def test_independent_batches_and_permutation():
    points = [[0, 0, 0, 0], [0.25, 0, 0, 0], [0, 0, 0, 0], [0.25, 0, 0, 0]]
    original = _evolve(None, points, batches=[0, 0, 1, 1], cutoffs=(-1.0, -1.0))
    permutation = [3, 0, 2, 1]
    shuffled = _evolve(None, [points[i] for i in permutation], batches=[1, 0, 1, 0],
                       cutoffs=(-1.0, -1.0))
    _assert_reference(original.state)
    _assert_reference(shuffled.state)
    assert original.state.graph.edge_index.shape == shuffled.state.graph.edge_index.shape == (2, 4)
    torch.testing.assert_close(shuffled.state.graph.in_degree, torch.ones(4, dtype=torch.long))


def test_per_batch_cutoffs_and_already_expired_arrivals():
    first = _evolve(None, [[0, 0, 0, 0]] * 4, [0, 1, 0, 1], [0, 0, 1, 1],
                    cutoffs=(-1.0, -1.0))
    update = _evolve(first.state, [[0, 0, 0, 0]] * 3, [0, 2, 1], [0, 0, 1],
                     cutoffs=(1.0, 2.0))
    torch.testing.assert_close(update.old_indices, torch.tensor([1, -1]))
    torch.testing.assert_close(update.state.node_batch, torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(update.state.timestamps, torch.tensor([1.0, 2.0], dtype=torch.float64))
    _assert_reference(update.state)


@pytest.mark.parametrize("previous_nonempty", [False, True])
def test_empty_and_all_expired_states(previous_nonempty):
    first = _evolve(None, [[0, 0, 0, 0]] if previous_nonempty else [])
    update = _evolve(first.state, [], cutoffs=(1.0,))
    assert update.state.graph.node_features.shape == (0, 4)
    assert update.state.graph.edge_index.shape == (2, 0)
    assert update.state.graph.edge_attr.shape == (0, 1)
    assert update.old_indices.numel() == update.changed_nodes.numel() == 0
    again = _evolve(update.state, [[0, 0, 0, 0]], [2], cutoffs=(1.0,))
    assert again.old_indices.tolist() == [-1]
    _assert_reference(again.state)


def test_exact_radius_boundary_excludes_equal_distance():
    inside = torch.nextafter(torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0)).item()
    outside = torch.nextafter(torch.tensor(1.0, dtype=torch.float64), torch.tensor(2.0)).item()
    update = _evolve(None, [[0, 0, 0, 0], [inside, 0, 0, 0], [1, 0, 0, 0], [outside, 0, 0, 0]])
    _assert_reference(update.state)
    neighbors = set(update.state.graph.edge_index[1, update.state.graph.edge_index[0] == 0].tolist())
    assert neighbors == {1}


def test_negative_large_absolute_coordinates_have_no_hash_alias():
    update = _evolve(None, [[-1e15, -2, -1e12, 0], [-1e15 + 0.5, -2, -1e12, 0],
                           [1e15, 4, 1e12, 0], [1e15 + 0.5, 4, 1e12, 0]],
                     batches=[0, 0, 1, 1], cutoffs=(-1.0, -1.0))
    _assert_reference(update.state)
    assert update.state.graph.edge_index.shape[1] == 4


def test_numeric_address_overflow_fails_explicitly():
    with pytest.raises(ValueError, match="float64 cell addressing"):
        _evolve(None, [[-1e20, 0, 0, 0], [1e20, 0, 0, 0]])


def test_large_finite_distance_does_not_overflow_during_radius_filter():
    positions = torch.tensor([[0, 0, 0, 0], [1e200, 0, 0, 0]], dtype=torch.float64)
    update = evolve_stream_graph(
        None, torch.zeros((2, 4)), positions, torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.long), torch.tensor([-1.0], dtype=torch.float64),
        radius=2e200, max_graph_edges=None,
    )
    _assert_reference(update.state, 2e200)
    assert update.state.graph.edge_index.shape[1] == 2


def test_guard_is_per_stream_and_failure_does_not_mutate_previous():
    first = _evolve(None, [[0, 0, 0, 0], [0.2, 0, 0, 0]], max_graph_edges=2)
    before = _snapshot(first.state)
    with pytest.raises(RuntimeError, match="max_graph_edges=2"):
        _evolve(first.state, [[0.4, 0, 0, 0]], max_graph_edges=2)
    _assert_unchanged(first.state, before)
    update = _evolve(None, [[0, 0, 0, 0]] * 4, batches=[0, 0, 1, 1],
                     cutoffs=(-1.0, -1.0), max_graph_edges=2)
    assert update.state.graph.edge_index.shape[1] == 4
    with pytest.raises(RuntimeError, match="max_graph_edges=1"):
        _evolve(first.state, [], max_graph_edges=1)


def test_feature_gradients_flow_through_retention_and_append():
    first_values = _arrival([[0, 0, 0, 0], [0.5, 0, 0, 0]], [0, 1], requires_grad=True)
    first = evolve_stream_graph(None, *first_values, torch.tensor([-1.0], dtype=torch.float64),
                                radius=1.0, max_graph_edges=None)
    second_values = _arrival([[0.75, 0, 0, 0]], [2], requires_grad=True)
    second = evolve_stream_graph(first.state, *second_values, torch.tensor([1.0], dtype=torch.float64),
                                 radius=1.0, max_graph_edges=None)
    second.state.graph.node_features.sum().backward()
    torch.testing.assert_close(first_values[0].grad, torch.tensor([[0.] * 4, [1.] * 4]))
    torch.testing.assert_close(second_values[0].grad, torch.ones((1, 4)))


def test_multiple_updates_match_brute_force_with_mixed_expiration():
    generator = torch.Generator().manual_seed(20260910)
    state = None
    for step in range(6):
        positions = torch.rand((11, 4), generator=generator, dtype=torch.float64) * 6 - 3
        features = positions.float()
        batches = torch.randint(0, 3, (11,), generator=generator)
        times = torch.full((11,), float(step), dtype=torch.float64)
        update = evolve_stream_graph(
            state, features, positions, times, batches,
            torch.tensor([step - 2.0, step - 1.0, step], dtype=torch.float64),
            radius=1.75, max_graph_edges=None, chunk_size=3,
        )
        _assert_reference(update.state, 1.75)
        state = update.state


@pytest.mark.parametrize("keyword,value,message", [
    ("radius", 0, "radius"), ("radius", float("nan"), "radius"),
    ("position_dims", 0, "position_dims"), ("position_dims", True, "position_dims"),
    ("chunk_size", 0, "chunk_size"), ("max_graph_edges", 0, "max_graph_edges"),
])
def test_invalid_configuration_is_not_silently_repaired(keyword, value, message):
    with pytest.raises(ValueError, match=message):
        _evolve(None, [], **{keyword: value})


def test_invalid_node_identity_or_nonfinite_input_is_rejected():
    with pytest.raises(ValueError, match="valid batch"):
        _evolve(None, [[0, 0, 0, 0]], batches=[1])
    with pytest.raises(ValueError, match="finite"):
        _evolve(None, [[float("nan"), 0, 0, 0]])
    with pytest.raises(ValueError, match="float64"):
        values = _arrival([])
        evolve_stream_graph(None, *values, torch.tensor([-1.0]), radius=1.0, max_graph_edges=None)
