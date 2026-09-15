"""Tiny synthetic CPU namespace regressions, not data/model performance checks."""

import pytest
import torch

from asgcn_unet.stream_graph import StreamGraph, evolve_stream_graph


@pytest.fixture(autouse=True)
def _one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _update(previous, positions, timestamps, batches, cutoffs, storage):
    positions = torch.as_tensor(positions, dtype=torch.float64).reshape(-1, 4)
    return evolve_stream_graph(
        previous, positions.float(), positions,
        torch.tensor(timestamps, dtype=torch.float64),
        torch.tensor(batches, dtype=torch.long),
        torch.tensor(cutoffs, dtype=torch.float64),
        radius=1.0, position_dims=3, max_graph_edges=None, chunk_size=2,
        graph_storage=storage,
    )


@pytest.mark.parametrize("storage", ["implicit_radius", "materialized"])
def test_previous_namespace_mismatch_cannot_reuse_cross_stream_degrees(storage):
    original = _update(
        None, [[0, 0, 0, 0], [.2, 0, 0, 0], [10, 0, 0, 0], [10.2, 0, 0, 0]],
        [0] * 4, [0] * 4, [-1, -1], storage,
    ).state
    # Splitting each nearby pair across lanes leaves no valid edges. Stale
    # degrees nevertheless pass nonnegative/even/per-lane cardinality checks.
    wrong_batch = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    malformed = StreamGraph(original.graph, wrong_batch, original.timestamps)
    before_degree = original.graph.in_degree.clone()
    with pytest.raises(ValueError, match="namespace|independent stream"):
        _update(malformed, [], [], [], [-1, -1], storage)
    assert torch.equal(original.graph.in_degree, before_degree)
    assert torch.equal(original.node_batch, torch.zeros(4, dtype=torch.long))
    assert torch.equal(malformed.node_batch, wrong_batch)


@pytest.mark.parametrize("storage", ["implicit_radius", "materialized"])
def test_valid_independent_stream_updates_match_full_distance_oracle(storage):
    state = None
    frames = [
        ([[0, 0, 0, 0], [.2, 0, 0, 0], [0, 0, 0, 0], [.2, 0, 0, 0]],
         [0, 0, 0, 0], [0, 0, 1, 1], [-1, -1, -1]),
        ([[.4, 0, 0, 0], [.4, 0, 0, 0]], [1, 1], [0, 1], [0, 0, -1]),
        ([[.6, 0, 0, 0], [.6, 0, 0, 0]], [2, 2], [0, 2], [1, 0, 2]),
        ([], [], [], [1, 3, 2]),
        ([], [], [], [9, 9, 9]),
    ]
    for positions, timestamps, batches, cutoffs in frames:
        state = _update(state, positions, timestamps, batches, cutoffs, storage).state
        graph = state.graph
        points = graph.positions[:, :3]
        distances = torch.linalg.vector_norm(points[:, None] - points[None, :], dim=2)
        expected = (distances < 1) & (state.node_batch[:, None] == state.node_batch[None, :])
        expected.fill_diagonal_(False)
        assert torch.equal(graph.in_degree, expected.sum(0))
        if storage == "implicit_radius":
            actual = torch.zeros_like(expected)
            for source, destination, _distance in graph.iter_directed_neighbors():
                actual[source, destination] = True
            assert graph.edge_count == int(expected.sum())
        else:
            actual = torch.zeros_like(expected)
            actual[graph.edge_index[0], graph.edge_index[1]] = True
            assert graph.edge_index.shape[1] == int(expected.sum())
        assert torch.equal(actual, expected)
