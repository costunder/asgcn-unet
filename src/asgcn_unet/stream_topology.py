"""Exact, count-only packed stream topology with bounded candidate-pair scratch.

This diagnostic does not construct an edge index, impose the model edge guard,
change the radius, or drop nodes. A supplied mask identifies the readout-induced
subgraph of the complete input union; both graphs are counted in one traversal.
Occupied-cell storage is O(N), while each candidate-pair work tensor contains at
most the explicit ``candidate_pair_budget`` entries, even for one dense cell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch

from .stream_graph import _cell_lookup, _occupied_cells


@dataclass(frozen=True)
class StreamTopologyCounts:
    union_nodes: torch.Tensor
    union_directed_edges: torch.Tensor
    readout_nodes: torch.Tensor
    readout_directed_edges: torch.Tensor
    candidate_pair_budget: int
    peak_candidate_pairs: int
    candidate_pairs_visited: int
    query_chunks: int
    candidate_chunks: int
    peak_query_cells: int


@dataclass
class _Scratch:
    candidate_pair_budget: int
    peak_candidate_pairs: int = 0
    candidate_pairs_visited: int = 0
    query_chunks: int = 0
    candidate_chunks: int = 0
    peak_query_cells: int = 0

    def counts(self, union_nodes, union_edges, readout_nodes, readout_edges):
        return StreamTopologyCounts(
            union_nodes, union_edges, readout_nodes, readout_edges,
            self.candidate_pair_budget, self.peak_candidate_pairs, self.candidate_pairs_visited,
            self.query_chunks, self.candidate_chunks, self.peak_query_cells,
        )


def _validate_options(radius, position_dims, chunk_size, candidate_pair_budget) -> float:
    if (isinstance(radius, bool) or not isinstance(radius, Real)
            or not math.isfinite(float(radius)) or radius <= 0):
        raise ValueError("Stream topology radius must be finite and positive")
    if isinstance(position_dims, bool) or not isinstance(position_dims, int) or not 1 <= position_dims <= 4:
        raise ValueError("Stream topology position_dims must be an integer from 1 to 4")
    for name, value in (("chunk_size", chunk_size), ("candidate_pair_budget", candidate_pair_budget)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Stream topology {name} must be a positive integer")
    return float(radius)


def _validate_mask(mask, positions, name):
    if (not isinstance(mask, torch.Tensor) or mask.layout != torch.strided
            or mask.shape != (positions.shape[0],) or mask.dtype != torch.bool):
        raise ValueError(f"Stream topology {name} must have shape [N] and dtype bool")
    if mask.device != positions.device:
        raise ValueError("All stream topology tensors must share a device")


def _validate_inputs(positions, node_batch, readout_mask, batch_size) -> None:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 0:
        raise ValueError("Stream topology batch_size must be a nonnegative integer")
    if (not isinstance(positions, torch.Tensor) or positions.layout != torch.strided
            or positions.ndim != 2 or positions.shape[1] != 4 or not positions.is_floating_point()):
        raise ValueError("Stream topology positions must be a floating strided tensor with shape [N,4]")
    count = positions.shape[0]
    if (not isinstance(node_batch, torch.Tensor) or node_batch.layout != torch.strided
            or node_batch.shape != (count,) or node_batch.dtype != torch.long):
        raise ValueError("Stream topology node_batch must have shape [N] and dtype long")
    _validate_mask(readout_mask, positions, "readout_mask")
    if node_batch.device != positions.device:
        raise ValueError("All stream topology tensors must share a device")
    if not bool(torch.stack((
        torch.isfinite(positions).all(),
        ((node_batch >= 0) & (node_batch < batch_size)).all(),
    )).all()):
        raise ValueError("Stream topology positions must be finite and have valid batch identifiers")


def _selected_edge_pairs(positions, node_batch, query_mask, *, batch_size, radius,
                         position_dims, chunk_size, scratch):
    """Yield bounded strict-radius pairs touching the selected endpoint set.

    Pairs are unordered and yielded once: when both endpoints are selected, the
    larger input index owns the query; otherwise the selected endpoint owns it.
    No pairs between two unselected nodes are expanded for distance evaluation.
    """
    query_nodes = torch.nonzero(query_mask, as_tuple=True)[0]
    if not query_nodes.numel():
        return
    rows, occupied, sorted_nodes, boundaries = _occupied_cells(
        positions, node_batch, batch_size, radius, position_dims,
    )
    axis = torch.tensor((-1, 0, 1), device=positions.device, dtype=torch.long)
    offsets = torch.cartesian_prod(*([axis] * position_dims)).reshape(-1, position_dims)
    cells_per_query = offsets.shape[0]
    for start in range(0, query_nodes.numel(), chunk_size):
        sources = query_nodes[start:start + chunk_size]
        query_rows = rows[sources]
        query_cells = query_rows[:, None, 1:] + offsets[None, :, :]
        query_batches = query_rows[:, None, :1].expand(-1, cells_per_query, -1)
        queries = torch.cat((query_batches, query_cells), dim=2).flatten(0, 1)
        cell_ids = _cell_lookup(occupied, queries)
        cell_counts = boundaries[1, cell_ids.clamp_min(0)].masked_fill(cell_ids < 0, 0)
        cell_starts = boundaries[0, cell_ids.clamp_min(0)]
        candidate_ends = cell_counts.cumsum(0)
        candidate_starts = candidate_ends - cell_counts
        candidate_count = int(candidate_ends[-1])
        scratch.query_chunks += 1
        scratch.peak_query_cells = max(scratch.peak_query_cells, queries.shape[0])
        # Address bounded slices rather than expanding repeat_interleave(counts):
        # even a single occupied cell can contain more than the scratch budget.
        for pair_start in range(0, candidate_count, scratch.candidate_pair_budget):
            pair_stop = min(pair_start + scratch.candidate_pair_budget, candidate_count)
            flat = torch.arange(pair_start, pair_stop, device=positions.device)
            groups = torch.searchsorted(candidate_ends, flat, right=True)
            source = sources[groups.div(cells_per_query, rounding_mode="floor")]
            target = sorted_nodes[cell_starts[groups] + flat - candidate_starts[groups]]
            scratch.peak_candidate_pairs = max(scratch.peak_candidate_pairs, pair_stop - pair_start)
            scratch.candidate_pairs_visited += pair_stop - pair_start
            scratch.candidate_chunks += 1
            pair = (source != target) & (~query_mask[target] | (target < source))
            source, target = source[pair], target[pair]
            # This exact float64 strict-radius predicate is shared by both
            # counters and matches the existing evolve_stream_graph updater.
            normalized_distances = torch.linalg.vector_norm(
                (positions[source, :position_dims].double()
                 - positions[target, :position_dims].double()) / radius, dim=1,
            )
            valid = normalized_distances < 1.0
            yield source[valid], target[valid]


@torch.no_grad()
def count_stream_topology(
    positions: torch.Tensor,
    node_batch: torch.Tensor,
    readout_mask: torch.Tensor,
    *,
    batch_size: int,
    radius: float,
    position_dims: int = 3,
    chunk_size: int = 512,
    candidate_pair_budget: int = 1_048_576,
) -> StreamTopologyCounts:
    """Count every strict-radius directed edge in a packed union and its readout.

    Input positions follow ``evolve_stream_graph``'s [N,4] fixed-coordinate
    contract. Distance is evaluated in float64 after division by ``radius``;
    equality to the radius and self edges are excluded. Batch IDs are part of the
    collision-free occupied-cell key, so independent streams cannot connect.

    ``chunk_size`` bounds source queries; ``candidate_pair_budget`` separately
    bounds expanded candidate pairs. It is a scratch budget, never an edge cap.
    Counts remain device tensors. One candidate-total scalar per query chunk is
    used to schedule bounded chunks; there is no per-edge/node host transfer.
    Scratch evidence counts entries, not total allocated bytes or measured RSS.
    """
    radius = _validate_options(radius, position_dims, chunk_size, candidate_pair_budget)
    _validate_inputs(positions, node_batch, readout_mask, batch_size)
    union_nodes = torch.bincount(node_batch, minlength=batch_size)
    readout_nodes = torch.bincount(node_batch[readout_mask], minlength=batch_size)
    union_edges = torch.zeros_like(union_nodes)
    readout_edges = torch.zeros_like(union_nodes)
    scratch = _Scratch(candidate_pair_budget)
    for source, target in _selected_edge_pairs(
        positions, node_batch, torch.ones_like(readout_mask), batch_size=batch_size, radius=radius,
        position_dims=position_dims, chunk_size=chunk_size, scratch=scratch,
    ):
        union_edges.add_(2 * torch.bincount(node_batch[source], minlength=batch_size))
        kept = readout_mask[source] & readout_mask[target]
        readout_edges.add_(2 * torch.bincount(node_batch[source[kept]], minlength=batch_size))
    return scratch.counts(union_nodes, union_edges, readout_nodes, readout_edges)


@torch.no_grad()
def count_stream_topology_update(
    positions: torch.Tensor,
    node_batch: torch.Tensor,
    union_mask: torch.Tensor,
    readout_mask: torch.Tensor,
    is_arrival: torch.Tensor,
    previous_edge_counts: torch.Tensor,
    *,
    batch_size: int,
    radius: float,
    position_dims: int = 3,
    chunk_size: int = 512,
    candidate_pair_budget: int = 1_048_576,
) -> StreamTopologyCounts:
    """Reuse exact old edge counts and count only new/expired incident pairs.

    Inputs contain *all* nodes from the previous readout plus all raw arrivals;
    old nodes must not be removed before this call. ``previous_edge_counts`` is
    the exact directed count of those old nodes under the unchanged coordinate,
    radius and dimension contract. This cache provenance belongs to the caller;
    verifying every old edge here would defeat incremental counting.

    The caller supplies inclusive timestamp-cutoff masks and verifies arrivals
    are not beyond the current readout watermark. This API has no timestamp or
    watermark arguments and does not infer them from coordinates. The readout
    must be a subset of the union; even arrivals already outside either mask are
    allowed and never silently substituted. An old pair contributes -2 when it
    leaves a graph, and a pair touching any arrival contributes +2 when retained.

    Only arrivals or nodes excluded from readout issue cell queries. Unchanged
    old-old survivor distances are not recalculated; dense incident work can
    still be quadratic. Scratch and strict-distance rules match the full counter.
    """
    radius = _validate_options(radius, position_dims, chunk_size, candidate_pair_budget)
    _validate_inputs(positions, node_batch, readout_mask, batch_size)
    _validate_mask(union_mask, positions, "union_mask")
    _validate_mask(is_arrival, positions, "is_arrival")
    if (not isinstance(previous_edge_counts, torch.Tensor) or previous_edge_counts.layout != torch.strided
            or previous_edge_counts.shape != (batch_size,) or previous_edge_counts.dtype != torch.long):
        raise ValueError("Stream topology previous_edge_counts must have shape [B] and dtype long")
    if previous_edge_counts.device != positions.device:
        raise ValueError("All stream topology tensors must share a device")
    old_nodes = torch.bincount(node_batch[~is_arrival], minlength=batch_size)
    if not bool(torch.stack((
        (~readout_mask | union_mask).all(),
        (previous_edge_counts >= 0).all(), (previous_edge_counts % 2 == 0).all(),
        (previous_edge_counts <= old_nodes * (old_nodes - 1)).all(),
    )).all()):
        raise ValueError(
            "Stream topology requires readout_mask subset of union_mask and feasible "
            "nonnegative even previous_edge_counts for every complete old-node stream"
        )
    union_nodes = torch.bincount(node_batch[union_mask], minlength=batch_size)
    readout_nodes = torch.bincount(node_batch[readout_mask], minlength=batch_size)
    union_edges, readout_edges = previous_edge_counts.clone(), previous_edge_counts.clone()
    scratch = _Scratch(candidate_pair_budget)
    query_mask = is_arrival | ~readout_mask
    for source, target in _selected_edge_pairs(
        positions, node_batch, query_mask, batch_size=batch_size, radius=radius,
        position_dims=position_dims, chunk_size=chunk_size, scratch=scratch,
    ):
        new_pair = is_arrival[source] | is_arrival[target]
        for mask, edges in ((union_mask, union_edges), (readout_mask, readout_edges)):
            both = mask[source] & mask[target]
            delta = 2 * ((new_pair & both).long() - (~new_pair & ~both).long())
            edges.index_add_(0, node_batch[source], delta)
    if not bool(torch.stack((
        (readout_edges >= 0).all(), (readout_edges <= union_edges).all(),
        (union_edges <= union_nodes * (union_nodes - 1)).all(),
        (readout_edges <= readout_nodes * (readout_nodes - 1)).all(),
    )).all()):
        raise ValueError("Stream topology cached previous_edge_counts disagree with the supplied old nodes/masks")
    return scratch.counts(union_nodes, union_edges, readout_nodes, readout_edges)
