"""Small synthetic CPU checks; no actual-data/CUDA performance claims."""

import pytest
import torch

from asgcn_unet import stream_graph
from asgcn_unet.implicit_radius import ImplicitRadiusGraph, build_implicit_radius_graph


@pytest.fixture(autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _build(positions, batches=None, *, batch_size=3, **kwargs):
    count = positions.shape[0]
    if batches is None:
        batches = torch.zeros(count, dtype=torch.long)
    return build_implicit_radius_graph(torch.zeros((count, 4), dtype=positions.dtype), positions, batches,
                                       batch_size=batch_size, radius=kwargs.pop("radius", 1.), **kwargs)


def _edges(graph, selected=None, stats=None):
    chunks = list(graph.iter_directed_neighbors(selected, stats=stats))
    if not chunks:
        return torch.empty((2, 0), dtype=torch.long), graph.positions.new_empty((0, 1))
    source, destination, pseudo = (torch.cat(parts) for parts in zip(*chunks))
    order = torch.argsort(source * graph.positions.shape[0] + destination)
    return torch.stack((source[order], destination[order])), pseudo[order]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("dims", [1, 2, 3, 4])
@pytest.mark.parametrize("budget", [3, 1_048_576])
def test_exact_packed_topology_matches_materialized_reference(dtype, dims, budget):
    generator = torch.Generator().manual_seed(43)
    positions = torch.rand((19, 4), generator=generator, dtype=dtype) * 4 - 2
    batches = torch.randint(0, 3, (19,), generator=generator)
    graph = _build(positions, batches, batch_size=4, radius=1.5, position_dims=dims,
                   chunk_size=4, candidate_pair_budget=budget)
    edges, pseudo = _edges(graph)
    # This test-only tiny oracle may allocate full E; the implementation may not.
    reference = stream_graph.evolve_stream_graph(
        None, graph.node_features, positions, torch.zeros(19, dtype=torch.float64), batches,
        torch.full((4,), -1., dtype=torch.float64), radius=1.5, position_dims=dims,
        max_graph_edges=None,
    ).state.graph
    order = torch.argsort(reference.edge_index[0] * 19 + reference.edge_index[1])
    torch.testing.assert_close(edges, reference.edge_index[:, order])
    torch.testing.assert_close(pseudo, reference.edge_attr[order], rtol=0, atol=0)
    torch.testing.assert_close(graph.in_degree, torch.bincount(edges[1], minlength=19))
    torch.testing.assert_close(graph.edge_counts, torch.bincount(batches[edges[0]], minlength=4))
    assert graph.edge_count == edges.shape[1]
    assert isinstance(graph.edge_count, int)
    assert not hasattr(graph, "edge_index") and not hasattr(graph, "edge_attr")


@pytest.mark.parametrize("budget", [1, 5, 17])
def test_dense_cell_candidate_scratch_is_bounded_and_edges_never_truncated(budget, monkeypatch):
    positions = torch.zeros((13, 4), dtype=torch.float64)
    observed = []
    original = torch.linalg.vector_norm

    def norm(values, *args, **kwargs):
        observed.append(values.shape[0])
        assert values.shape[0] <= budget
        return original(values, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "vector_norm", norm)
    graph = _build(positions, candidate_pair_budget=budget)
    stats = {}
    edges, _ = _edges(graph, stats=stats)
    assert edges.shape[1] == graph.edge_count == 13 * 12
    assert stats["peak_candidate_pairs"] == budget
    assert stats["candidate_pairs_visited"] == 13**2
    assert observed


def test_selected_destinations_include_all_sources_and_no_cross_stream_edges():
    positions = torch.zeros((6, 4), dtype=torch.float64)
    batches = torch.tensor([0, 1, 0, 1, 0, 1])
    graph = _build(positions, batches)
    selected = torch.tensor([4, 1])
    edges, _ = _edges(graph, selected)
    assert set(map(tuple, edges.t().tolist())) == {(0, 4), (2, 4), (3, 1), (5, 1)}
    assert graph.in_degree.tolist() == [2] * 6
    assert _edges(graph, selected[:0])[0].shape == (2, 0)


def test_strict_boundary_and_large_translated_coordinates():
    positions = torch.zeros((4, 4), dtype=torch.float64)
    positions[:, 0] = torch.tensor([0., torch.nextafter(torch.tensor(1., dtype=torch.float64),
                                                      torch.tensor(0., dtype=torch.float64)), 1., 2.])
    graph = _build(positions, position_dims=1)
    pairs = set(map(tuple, _edges(graph)[0].t().tolist()))
    assert (0, 1) in pairs and (0, 2) not in pairs and (2, 3) not in pairs
    shifted = positions.clone()
    shifted[:, 0] = torch.tensor([-1e12, -1e12 + .25, -1e12 + 1., -1e12 + 3.], dtype=torch.float64)
    shifted_graph = _build(shifted, position_dims=1)
    assert set(map(tuple, _edges(shifted_graph)[0].t().tolist())) == {(0, 1), (1, 0), (1, 2), (2, 1)}


def test_index_is_cached_and_counted_factory_is_lazy_without_recount(monkeypatch):
    original = stream_graph._occupied_cells
    calls = []

    def occupied(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(stream_graph, "_occupied_cells", occupied)
    positions = torch.zeros((5, 4), dtype=torch.float64)
    graph = _build(positions)
    assert len(calls) == 1
    _edges(graph)
    _edges(graph, torch.tensor([2]))
    assert len(calls) == 1
    restored = ImplicitRadiusGraph.from_counted_nodes(
        graph.node_features, positions, graph.node_batch, graph.in_degree,
        batch_size=3, radius=1., edge_counts=graph.edge_counts,
    )
    assert restored._index is None and len(calls) == 1
    assert restored.edge_count == 20
    _edges(restored)
    assert len(calls) == 2
    cached = ImplicitRadiusGraph.from_counted_nodes(
        graph.node_features, positions, graph.node_batch, graph.in_degree,
        batch_size=3, radius=1., index=graph.index,
    )
    assert cached.index is graph.index and len(calls) == 2


@pytest.mark.parametrize("batch_size", [0, 3])
def test_empty_graph_is_explicit_and_valid(batch_size):
    graph = _build(torch.empty((0, 4), dtype=torch.float64), batch_size=batch_size)
    assert graph.edge_count == 0 and graph.edge_counts.tolist() == [0] * batch_size
    assert _edges(graph)[0].shape == (2, 0)


def test_inference_mode_tensors_work_without_version_counter():
    with torch.inference_mode():
        graph = _build(torch.zeros((3, 4), dtype=torch.float64))
        assert graph.edge_count == 6
        assert _edges(graph)[0].shape == (2, 6)


@pytest.mark.parametrize("field", ["positions", "node_batch", "in_degree", "edge_counts"])
def test_mutation_fails_before_using_stale_index_or_degrees(field):
    graph = _build(torch.zeros((3, 4), dtype=torch.float64))
    getattr(graph, field).add_(0)
    with pytest.raises(RuntimeError, match="modified"):
        _edges(graph)


@pytest.mark.parametrize("kwargs", [{"radius": 0}, {"radius": float("nan")}, {"radius": True},
                                    {"position_dims": 0}, {"position_dims": True},
                                    {"chunk_size": 0}, {"candidate_pair_budget": 0}, {"batch_size": -1}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        _build(torch.zeros((3, 4)), **kwargs)


def test_invalid_geometry_selection_and_counts_fail_explicitly():
    positions = torch.zeros((3, 4), dtype=torch.float64)
    with pytest.raises(ValueError, match="position gradients"):
        _build(positions.clone().requires_grad_())
    with pytest.raises(ValueError, match="independent stream"):
        _build(positions, torch.tensor([0, 0, 3]))
    graph = _build(positions)
    for selected in (torch.tensor([0, 0]), torch.tensor([-1]), torch.tensor([3]), torch.tensor([0.] )):
        with pytest.raises(ValueError):
            _edges(graph, selected)
    for degrees in (torch.tensor([-1, 1, 2]), torch.tensor([3, 1, 2]), torch.tensor([1, 1, 1])):
        with pytest.raises(ValueError, match="degrees"):
            ImplicitRadiusGraph.from_counted_nodes(graph.node_features, positions, graph.node_batch,
                                                   degrees, batch_size=3, radius=1.)
    with pytest.raises(RuntimeError, match="no edges were truncated"):
        _build(positions, max_graph_edges=5)
    with pytest.raises(ValueError, match="coordinate span"):
        _build(torch.tensor([[0., 0, 0, 0], [1e30, 0, 0, 0]], dtype=torch.float64))
