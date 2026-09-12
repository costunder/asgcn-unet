"""Tiny CPU exact-count oracles, never real-data throughput certification."""

from __future__ import annotations

import pytest
import torch

from asgcn_unet import stream_topology as topology


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _brute(positions, batches, mask, *, radius, dimensions, batch_size):
    coordinates = positions[:, :dimensions].double()
    distance = torch.linalg.vector_norm((coordinates[:, None] - coordinates[None, :]) / radius, dim=-1)
    edges = ((distance < 1) & (batches[:, None] == batches[None, :])
             & mask[:, None] & mask[None, :])
    edges.fill_diagonal_(False)
    source = edges.nonzero(as_tuple=True)[0]
    return torch.bincount(batches[source], minlength=batch_size)


@pytest.mark.parametrize("dimensions", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", range(8))
def test_incremental_bulk_mixed_cells_lanes_and_masks_matches_pairwise_oracle(dimensions, seed):
    generator = torch.Generator().manual_seed(seed + 902)
    # Interleaved old/arrival IDs and independent streams invalidate any shortcut
    # that assumes old nodes form a prefix, or cells consist of one ownership set.
    count, batch_size = 37, 4
    cluster = torch.randint(0, 4, (count,), generator=generator)
    positions = torch.rand((count, 4), generator=generator, dtype=torch.float64) * 0.08
    positions[:, 0] += cluster.double() * 1.23
    batches = torch.randint(0, batch_size - 1, (count,), generator=generator)
    arrival = torch.rand(count, generator=generator) > 0.65
    union = torch.rand(count, generator=generator) > 0.15
    readout = union & (torch.rand(count, generator=generator) > 0.25)
    options = {"radius": 0.5, "position_dims": dimensions, "batch_size": batch_size,
               "chunk_size": 3, "candidate_pair_budget": 5}
    cached = _brute(positions, batches, ~arrival, radius=0.5, dimensions=dimensions, batch_size=batch_size)
    result = topology.count_stream_topology_update(positions, batches, union, readout, arrival, cached, **options)
    torch.testing.assert_close(result.union_directed_edges,
                               _brute(positions, batches, union, radius=0.5, dimensions=dimensions, batch_size=batch_size))
    torch.testing.assert_close(result.readout_directed_edges,
                               _brute(positions, batches, readout, radius=0.5, dimensions=dimensions, batch_size=batch_size))
    assert result.bulk_query_blocks > 0
    assert result.peak_candidate_pairs <= 5


@pytest.mark.parametrize("dimensions", [1, 2, 3, 4])
def test_bulk_and_forced_exact_fallback_have_identical_counts(monkeypatch, dimensions):
    generator = torch.Generator().manual_seed(239)
    positions = torch.rand((73, 4), generator=generator, dtype=torch.float64) * 2.2
    # Dense subclusters plus uncertain border cells exercise both paths together.
    positions[:24] *= 0.01
    batches = torch.arange(73) % 3
    mask = torch.arange(73) % 4 != 0
    args = {"batch_size": 3, "radius": 1.0, "position_dims": dimensions,
            "chunk_size": 7, "candidate_pair_budget": 11}
    fast = topology.count_stream_topology(positions, batches, mask, **args)
    with monkeypatch.context() as patch:
        patch.setattr(topology._BulkCellCounts, "certified_inside",
                      lambda self, positions, sources, cell_ids, radius, position_dims: torch.zeros_like(cell_ids, dtype=torch.bool))
        baseline = topology.count_stream_topology(positions, batches, mask, **args)
    torch.testing.assert_close(fast.union_directed_edges, baseline.union_directed_edges)
    torch.testing.assert_close(fast.readout_directed_edges, baseline.readout_directed_edges)
    assert fast.candidate_pairs_visited < baseline.candidate_pairs_visited
    assert fast.bulk_pairwise_evaluations_avoided > 0
    assert baseline.bulk_pairwise_evaluations_avoided == 0


def test_large_dense_complete_cell_counts_all_edges_without_pair_expansion(monkeypatch):
    # Deliberately more edges than signed int32, but only 50,000 stored nodes.
    count = 50_000
    positions = torch.zeros((count, 4), dtype=torch.float64)
    batches = torch.zeros(count, dtype=torch.long)
    mask = torch.arange(count) % 2 == 0

    def forbidden(*args, **kwargs):
        raise AssertionError("A certified complete cell must not evaluate pair distances")

    monkeypatch.setattr(torch.linalg, "vector_norm", forbidden)
    result = topology.count_stream_topology(positions, batches, mask, batch_size=1, radius=1,
                                            candidate_pair_budget=1)
    assert result.union_directed_edges.tolist() == [count * (count - 1)]
    assert result.readout_directed_edges.tolist() == [(count // 2) * (count // 2 - 1)]
    assert result.candidate_pairs_visited == result.candidate_chunks == result.peak_candidate_pairs == 0
    assert result.bulk_query_blocks == count
    assert result.bulk_pairwise_evaluations_avoided == count * (count - 1) // 2


def test_uncertain_one_ulp_radius_boundary_is_not_bulk_certified():
    inside = torch.nextafter(torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0)).item()
    outside = torch.nextafter(torch.tensor(1.0, dtype=torch.float64), torch.tensor(2.0)).item()
    positions = torch.tensor([[0, 0, 0, 0], [inside, 0, 0, 0],
                              [1, 0, 0, 0], [outside, 0, 0, 0]], dtype=torch.float64)
    batches = torch.zeros(4, dtype=torch.long)
    mask = torch.ones(4, dtype=torch.bool)
    result = topology.count_stream_topology(positions, batches, mask, batch_size=1, radius=1,
                                            candidate_pair_budget=2)
    torch.testing.assert_close(result.union_directed_edges,
                               _brute(positions, batches, mask, radius=1, dimensions=3, batch_size=1))
    assert result.candidate_pairs_visited > 0


@pytest.mark.parametrize("radius,origin", [(2e200, 1e200), (2e-200, 1e-200)])
def test_bulk_bounds_preserve_extreme_finite_coordinate_scales(radius, origin):
    positions = torch.tensor([[origin, 0, 0, 0], [origin * 1.1, 0, 0, 0],
                              [origin * 1.2, 0, 0, 0]], dtype=torch.float64)
    batches = torch.zeros(3, dtype=torch.long)
    mask = torch.ones(3, dtype=torch.bool)
    result = topology.count_stream_topology(positions, batches, mask, batch_size=1, radius=radius)
    torch.testing.assert_close(result.union_directed_edges,
                               _brute(positions, batches, mask, radius=radius, dimensions=3, batch_size=1))
    assert result.bulk_pairwise_evaluations_avoided == 3
