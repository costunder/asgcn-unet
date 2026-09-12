"""Small CPU append/expire topology integration checks, not dataset measurements."""

import pytest
import torch

from asgcn_unet.implicit_radius import ImplicitRadiusGraph, ImplicitRadiusIndex
from asgcn_unet.stream_graph import evolve_stream_graph


@pytest.fixture(autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _update(previous, positions, times, batches, cutoffs, *, storage, radius=1., dims=3, guard=None):
    positions = torch.as_tensor(positions, dtype=torch.float64).reshape(-1, 4)
    return evolve_stream_graph(
        previous, positions.float().clone(), positions, torch.tensor(times, dtype=torch.float64),
        torch.tensor(batches, dtype=torch.long), torch.tensor(cutoffs, dtype=torch.float64),
        radius=radius, position_dims=dims, max_graph_edges=guard, chunk_size=3, graph_storage=storage,
    )


def _assert_same(actual, expected):
    graph, reference = actual.state.graph, expected.state.graph
    assert isinstance(graph, ImplicitRadiusGraph)
    assert "edge_index" not in vars(graph) and "edge_attr" not in vars(graph)
    for name in ("node_features", "positions", "in_degree"):
        torch.testing.assert_close(getattr(graph, name), getattr(reference, name), rtol=0, atol=0)
    for name in ("node_batch", "timestamps"):
        torch.testing.assert_close(getattr(actual.state, name), getattr(expected.state, name), rtol=0, atol=0)
    torch.testing.assert_close(actual.old_indices, expected.old_indices, rtol=0, atol=0)
    torch.testing.assert_close(actual.changed_nodes, expected.changed_nodes, rtol=0, atol=0)
    chunks = list(graph.iter_directed_neighbors())  # Tiny test oracle only.
    if chunks:
        source, destination, pseudo = (torch.cat(values) for values in zip(*chunks))
        edges = torch.stack((source, destination))
    else:
        edges = graph.node_batch.new_empty((2, 0))
        pseudo = graph.positions.new_empty((0, 1))
    count = len(graph.node_features)
    order = torch.argsort(edges[0] * count + edges[1])
    reference_order = torch.argsort(reference.edge_index[0] * count + reference.edge_index[1])
    torch.testing.assert_close(edges[:, order], reference.edge_index[:, reference_order], rtol=0, atol=0)
    torch.testing.assert_close(pseudo[order], reference.edge_attr[reference_order], rtol=0, atol=0)
    assert graph.edge_count == reference.edge_index.shape[1]


@pytest.mark.parametrize("dims", [1, 2, 3, 4])
def test_real_updater_matches_materialized_on_arrivals_expiry_gaps_and_empty_lanes(dims):
    # Coincident nodes, replaced neighbors with unchanged net degree, mixed lane
    # expiry, cutoff ties, rejected already-expired arrivals, all-empty, restart.
    frames = [
        ([[0., 0, 0, 0], [.5, 0, 0, 0], [0., 0, 0, 0], [.5, 0, 0, 0]],
         [0, 1, 0, 1], [0, 0, 1, 1], [-1, -1, -1]),
        ([[1., 0, 0, 0], [0., 0, 0, 0]], [2, 0], [0, 1], [1, 1, -1]),
        ([], [], [], [1, 2, 0]),
        ([[1., 0, 0, 0], [1., 0, 0, 0], [.5, 0, 0, 0]], [3, 3, 3], [2, 2, 0], [2, 2, 3]),
        ([], [], [], [10, 10, 10]),
        ([[0., 0, 0, 0], [.25, 0, 0, 0]], [11, 11], [1, 1], [10, 10, 10]),
    ]
    states = [None, None]
    for points, times, batches, cutoffs in frames:
        updates = [_update(state, points, times, batches, cutoffs, storage=storage, dims=dims)
                   for state, storage in zip(states, ("implicit_radius", "materialized"))]
        _assert_same(*updates)
        states = [update.state for update in updates]


def test_incremental_queries_only_expiries_and_arrivals_not_unchanged_old_pairs(monkeypatch):
    first = _update(None, [[0., 0, 0, 0]] * 9, [0] + [1] * 8, [0] * 9, [-1], storage="implicit_radius")
    before_degree = first.state.graph.in_degree.clone()
    original = ImplicitRadiusIndex.iter_directed_neighbors
    queried = []

    def neighbors(index, destinations=None, **kwargs):
        assert destinations is not None, "The updater must not query every old-old edge"
        queried.append(destinations.numel())
        return original(index, destinations, **kwargs)

    monkeypatch.setattr(ImplicitRadiusIndex, "iter_directed_neighbors", neighbors)
    second = _update(first.state, [[.25, 0, 0, 0]], [2], [0], [1], storage="implicit_radius")
    assert queried == [1, 1]
    torch.testing.assert_close(first.state.graph.in_degree, before_degree, rtol=0, atol=0)
    assert second.state.graph.edge_count == 9 * 8
    assert bool(second.changed_nodes.all())


def test_graph_guard_and_storage_contract_failure_preserve_previous():
    states = [_update(None, [[0., 0, 0, 0], [.25, 0, 0, 0]], [0, 0], [0, 0], [-1], storage=storage).state
              for storage in ("implicit_radius", "materialized")]
    before_degree = states[0].graph.in_degree.clone()
    with pytest.raises(RuntimeError, match="max_graph_edges"):
        _update(states[0], [[.5, 0, 0, 0]], [1], [0], [-1], storage="implicit_radius", guard=2)
    torch.testing.assert_close(states[0].graph.in_degree, before_degree, rtol=0, atol=0)
    for state, wrong_storage in zip(states, ("materialized", "implicit_radius")):
        with pytest.raises(ValueError, match="storage"):
            _update(state, [], [], [], [-1], storage=wrong_storage)
