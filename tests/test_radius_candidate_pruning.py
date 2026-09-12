"""Small synthetic CPU oracles; no dataset/CUDA throughput certification."""

import pytest
import torch

from asgcn_unet import implicit_radius
from asgcn_unet.implicit_radius import ImplicitRadiusIndex, build_implicit_radius_graph


@pytest.fixture(autouse=True)
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _pairs(index, selection=None, active=None):
    stats = {}
    chunks = list(index.iter_directed_neighbors(selection, active_sources=active, stats=stats))
    pairs = set()
    for source, destination, _ in chunks:
        pairs.update(zip(source.tolist(), destination.tolist(), strict=True))
    assert len(pairs) == sum(len(source) for source, _, _ in chunks)
    return pairs, stats


@pytest.mark.parametrize("dims", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("radius", [0.08, 1.0, 2e-200, 2e200])
def test_pruned_and_source_filtered_queries_match_independent_dense_oracle(dims, dtype, radius):
    # Very large/small positions require float64; fp32 cases still test the
    # same scales with representable zero geometry instead of overflow input.
    generator = torch.Generator().manual_seed(256)
    scale = radius if dtype == torch.float64 or radius == 1.0 or radius == .08 else 0.
    positions = (torch.rand((23, 4), generator=generator, dtype=dtype) * 6 - 3) * scale
    batches = torch.arange(23) % 3
    index = ImplicitRadiusIndex(positions, batches, batch_size=4, radius=radius,
                                position_dims=dims, chunk_size=5, candidate_pair_budget=7)
    selected = torch.tensor([21, 1, 16, 2, 8])
    active = torch.arange(23) % 4 == 1
    actual, stats = _pairs(index, selected, active)
    distances = torch.linalg.vector_norm(
        (positions[:, None, :dims].double() - positions[None, :, :dims].double()) / radius,
        dim=-1,
    )
    selected_mask = torch.zeros(23, dtype=torch.bool)
    selected_mask[selected] = True
    keep = ((distances < 1) & (batches[:, None] == batches[None, :])
            & active[:, None] & selected_mask[None, :])
    keep.fill_diagonal_(False)
    expected = set(map(tuple, torch.nonzero(keep).tolist()))
    assert actual == expected
    assert stats["peak_candidate_pairs"] <= 7


def test_bounds_prune_work_without_changing_edges(monkeypatch):
    generator = torch.Generator().manual_seed(762)
    positions = torch.rand((512, 4), generator=generator, dtype=torch.float64)
    batches = torch.arange(512) % 4
    index = ImplicitRadiusIndex(positions, batches, batch_size=4, radius=.08)
    actual, pruned = _pairs(index)

    def unpruned(queries, ids, boundaries, bounds, radius):
        return torch.where(ids >= 0, boundaries[1, ids.clamp_min(0)], 0)

    monkeypatch.setattr(implicit_radius, "prune_cell_counts", unpruned)
    expected, original = _pairs(index)
    assert actual == expected
    assert pruned["candidate_pairs_visited"] < original["candidate_pairs_visited"] // 2


def test_zero_spikes_are_excluded_before_distance_work_but_full_degree_is_unchanged():
    positions = torch.zeros((128, 4), dtype=torch.float64)
    graph = build_implicit_radius_graph(positions.clone(), positions, torch.zeros(128, dtype=torch.long),
                                        batch_size=1, radius=1., candidate_pair_budget=11)
    original_degrees = graph.in_degree.clone()
    active = torch.zeros(128, dtype=torch.bool)
    active[7] = True
    pairs, stats = _pairs(graph.index, active=active)
    assert pairs == {(7, target) for target in range(128) if target != 7}
    assert stats["candidate_pairs_visited"] == 128  # Not 128*128 then a post-filter.
    torch.testing.assert_close(graph.in_degree, original_degrees)
    assert graph.edge_count == 128 * 127
    pairs, stats = _pairs(graph.index, active=torch.zeros_like(active))
    assert not pairs and stats["candidate_pairs_visited"] == stats["query_chunks"] == 0


def test_source_mask_is_validated_even_on_empty_graph():
    index = ImplicitRadiusIndex(torch.empty((0, 4)), torch.empty(0, dtype=torch.long),
                                batch_size=0, radius=1.)
    for active in (torch.empty(0), torch.ones(1, dtype=torch.bool), [False]):
        with pytest.raises(ValueError, match="active_sources"):
            list(index.iter_directed_neighbors(active_sources=active))
