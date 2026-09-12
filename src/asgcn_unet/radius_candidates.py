"""Conservative cell bounds for exact radius queries, not a graph sparsifier.

Bounds come from actual member coordinates, never rounded cell corners. They
only reject blocks which cannot contain a radius edge. The final float64 strict
distance predicate remains authoritative. All storage is node/cell sized.
"""

from __future__ import annotations

import torch


def coordinate_bounds(positions, sorted_nodes, boundaries, position_dims):
    """Return actual float64 [occupied cells, dimensions] minima and maxima."""
    count = boundaries.shape[1]
    lower = torch.full((count, position_dims), float("inf"),
                       device=positions.device, dtype=torch.float64)
    upper = torch.full_like(lower, -float("inf"))
    if not sorted_nodes.numel():
        return lower, upper
    ranks = torch.arange(sorted_nodes.numel(), device=positions.device)
    cell_ids = torch.searchsorted(boundaries[0].contiguous(), ranks, right=True) - 1
    coordinates = positions[sorted_nodes, :position_dims].double()
    index = cell_ids[:, None].expand_as(coordinates)
    lower.scatter_reduce_(0, index, coordinates, reduce="amin", include_self=True)
    upper.scatter_reduce_(0, index, coordinates, reduce="amax", include_self=True)
    return lower, upper


def prune_cell_counts(query_positions, cell_ids, boundaries, bounds, radius):
    """Exclude only provably distant [query, neighboring cell] blocks.

    The minimum box distance is a lower bound for every member distance. A
    float64 margin makes near-boundary classification conservative even when
    the norm reduction's rounding differs. It admits extra candidates, not
    fewer edges. Scaled distances avoid radius-squared overflow.
    """
    if not boundaries.shape[1]:
        return torch.zeros_like(cell_ids)
    lower, upper = bounds
    safe = cell_ids.clamp_min(0)
    queries = query_positions[:, None, :lower.shape[1]].double()
    gap = torch.maximum(lower[safe] - queries, queries - upper[safe]).clamp_min(0)
    squared_lower_bound = (gap / radius).square().sum(dim=-1)
    margin = 64 * torch.finfo(torch.float64).eps
    possible = (cell_ids >= 0) & (squared_lower_bound <= 1.0 + margin)
    return torch.where(possible, boundaries[1, safe], 0)
