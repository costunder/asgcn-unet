"""Exact append/expire topology updates for packed, independent event streams.

This module does not normalize events, choose a time window, subsample events, or
advance a decoder. Callers supply fixed-coordinate nodes and explicit per-stream
cutoffs. A state must only be reused with the same radius/coordinate contract.
Surviving old edges (including their attributes) are reused; radius queries are
issued only for arrivals. No full radius graph builder is called.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .graph import EventGraph


@dataclass(frozen=True)
class StreamGraph:
    graph: EventGraph
    node_batch: torch.Tensor
    timestamps: torch.Tensor


@dataclass(frozen=True)
class GraphUpdate:
    state: StreamGraph
    old_indices: torch.Tensor
    changed_nodes: torch.Tensor


def _validate_nodes(
    features: torch.Tensor,
    positions: torch.Tensor,
    timestamps: torch.Tensor,
    node_batch: torch.Tensor,
    cutoffs: torch.Tensor,
) -> None:
    count = features.shape[0] if features.ndim else -1
    if features.shape != (count, 4) or not features.is_floating_point():
        raise ValueError("Stream features must be a floating tensor with shape [N,4]")
    if positions.shape != (count, 4) or not positions.is_floating_point():
        raise ValueError("Stream positions must be a floating tensor with shape [N,4]")
    if timestamps.shape != (count,) or timestamps.dtype != torch.float64:
        raise ValueError("Stream timestamps must have shape [N] and dtype float64")
    if node_batch.shape != (count,) or node_batch.dtype != torch.long:
        raise ValueError("Stream node_batch must have shape [N] and dtype long")
    if any(value.device != features.device for value in (positions, timestamps, node_batch, cutoffs)):
        raise ValueError("All stream tensors and cutoffs must share a device")
    if not bool(torch.stack((
        torch.isfinite(features).all(), torch.isfinite(positions).all(),
        torch.isfinite(timestamps).all(),
        ((node_batch >= 0) & (node_batch < cutoffs.numel())).all(),
    )).all()):
        raise ValueError("Stream nodes must be finite and have valid batch identifiers")


def _check_guard(edge_counts: torch.Tensor, max_graph_edges: int | None) -> None:
    if max_graph_edges is not None and bool((edge_counts > max_graph_edges).any()):
        raise RuntimeError(
            f"Stream radius graph exceeded max_graph_edges={max_graph_edges:,} "
            "in at least one independent stream. No edges were truncated and the "
            "previous state was not changed. Measure topology and accelerator memory "
            "before changing the explicit per-stream guard."
        )


def _cell_lookup(sorted_rows: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """Collision-free, vectorized lower_bound of lexicographically sorted rows.

    Tuple coordinates are never flattened into a finite-width radix hash. The
    loop is over logarithmic search levels, not nodes, graphs, or candidate edges.
    Returns -1 for an unoccupied cell.
    """
    count = sorted_rows.shape[0]
    low = torch.zeros(queries.shape[0], dtype=torch.long, device=queries.device)
    high = torch.full_like(low, count)
    for _ in range(count.bit_length()):
        middle = (low + high) // 2
        candidate = sorted_rows[middle.clamp_max(count - 1)]
        equal = candidate == queries
        prefix_equal = torch.cat((
            torch.ones_like(equal[:, :1]), equal[:, :-1].cumprod(dim=1).bool(),
        ), dim=1)
        less = ((candidate < queries) & prefix_equal).any(dim=1)
        active = low < high
        low = torch.where(active & less, middle + 1, low)
        high = torch.where(active & ~less, middle, high)
    found = (low < count) & (sorted_rows[low.clamp_max(count - 1)] == queries).all(dim=1)
    return torch.where(found, low, -1)


def _occupied_cells(
    positions: torch.Tensor, node_batch: torch.Tensor, batch_size: int,
    radius: float, position_dims: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a collision-free occupied-cell index, not a graph or distance matrix.

    Per-stream translation permits negative and large absolute coordinates. Cells
    have width 2*radius, so adjacent-cell search has a full half-cell of numerical
    slack. Explicit float64 representability checks prevent silent cell aliasing.
    """
    coordinates = positions[:, :position_dims].double()
    origins = coordinates.new_full((batch_size, position_dims), float("inf"))
    origins.scatter_reduce_(
        0, node_batch[:, None].expand_as(coordinates), coordinates,
        reduce="amin", include_self=True,
    )
    scaled = ((coordinates - origins[node_batch]) / radius) * 0.5
    # This is a numeric-address guard, not a node/edge/time-window truncation.
    # Beyond this range float64 cannot reliably preserve adjacent-cell geometry.
    if not bool((torch.isfinite(scaled) & (scaled.abs() <= 2**48)).all()):
        raise ValueError(
            "Stream coordinate span/radius exceeds exact float64 cell addressing. "
            "Use an explicitly justified coordinate scale/window; no graph was truncated."
        )
    cells = torch.floor(scaled).to(torch.long)
    rows = torch.cat((node_batch[:, None], cells), dim=1)
    occupied, inverse, counts = torch.unique(
        rows, dim=0, sorted=True, return_inverse=True, return_counts=True,
    )
    sorted_nodes = torch.argsort(inverse, stable=True)
    starts = counts.cumsum(0) - counts
    return rows, occupied, sorted_nodes, torch.stack((starts, counts))


def evolve_stream_graph(
    previous: StreamGraph | None,
    features: torch.Tensor,
    positions: torch.Tensor,
    timestamps: torch.Tensor,
    node_batch: torch.Tensor,
    cutoffs: torch.Tensor,
    *,
    radius: float,
    position_dims: int = 3,
    max_graph_edges: int | None,
    chunk_size: int = 512,
) -> GraphUpdate:
    """Expire timestamps < cutoff and append arrivals, preserving all valid edges.

    Nodes are retained in previous order, followed by nonexpired arrivals in input
    order. ``old_indices`` maps each output node to its previous offset, or -1 for
    an arrival. ``changed_nodes`` marks arrivals and surviving endpoints of added
    or removed edges, even when their final degree happens to be unchanged.

    Every edge is directed both ways; self edges are excluded and distance must
    be strictly less than radius. ``edge_attr`` is distance/radius, matching the
    spline graph contract. Edge order is not globally sorted. ``max_graph_edges``
    is an explicit per-stream guard, never a batch-wide cap or truncation.
    The previous tensors are never mutated, including on failure.
    """
    if isinstance(radius, bool) or not math.isfinite(float(radius)) or radius <= 0:
        raise ValueError("Stream graph radius must be finite and positive")
    radius = float(radius)
    if isinstance(position_dims, bool) or not isinstance(position_dims, int) or not 1 <= position_dims <= 4:
        raise ValueError("Stream position_dims must be an integer from 1 to 4")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("Stream chunk_size must be a positive integer")
    if max_graph_edges is not None and (
        isinstance(max_graph_edges, bool) or not isinstance(max_graph_edges, int) or max_graph_edges < 1
    ):
        raise ValueError("max_graph_edges must be a positive integer or None")
    if cutoffs.ndim != 1 or cutoffs.dtype != torch.float64 or not bool(torch.isfinite(cutoffs).all()):
        raise ValueError("Stream cutoffs must be a finite float64 tensor with shape [B]")
    _validate_nodes(features, positions, timestamps, node_batch, cutoffs)
    device = features.device
    old_count = 0
    if previous is not None:
        old_graph = previous.graph
        _validate_nodes(
            old_graph.node_features, old_graph.positions, previous.timestamps,
            previous.node_batch, cutoffs,
        )
        if old_graph.node_features.dtype != features.dtype or old_graph.positions.dtype != positions.dtype:
            raise ValueError("Previous and arriving stream feature/position dtypes must match")
        old_count = old_graph.node_features.shape[0]
        if (
            old_graph.edge_index.device != device or old_graph.edge_attr.device != device
            or old_graph.edge_attr.shape != (old_graph.edge_index.shape[1], 1)
            or old_graph.edge_attr.dtype != positions.dtype
        ):
            raise ValueError("Previous stream edges have incompatible attributes/device")
        if not bool(torch.stack((
            ((old_graph.edge_index >= 0) & (old_graph.edge_index < old_count)).all(),
            torch.isfinite(old_graph.edge_attr).all(),
        )).all()):
            raise ValueError("Previous stream edges contain invalid indices or attributes")
        retained = torch.nonzero(
            previous.timestamps >= cutoffs[previous.node_batch], as_tuple=True,
        )[0]
    else:
        retained = torch.empty(0, dtype=torch.long, device=device)
    arriving = torch.nonzero(timestamps >= cutoffs[node_batch], as_tuple=True)[0]
    retained_count = retained.numel()
    parts_features = [features[arriving]]
    parts_positions = [positions[arriving]]
    parts_timestamps = [timestamps[arriving]]
    parts_batch = [node_batch[arriving]]
    if previous is not None:
        parts_features.insert(0, previous.graph.node_features[retained])
        parts_positions.insert(0, previous.graph.positions[retained])
        parts_timestamps.insert(0, previous.timestamps[retained])
        parts_batch.insert(0, previous.node_batch[retained])
    output_features = torch.cat(parts_features)
    output_positions = torch.cat(parts_positions)
    output_timestamps = torch.cat(parts_timestamps)
    output_batch = torch.cat(parts_batch)
    count = output_features.shape[0]
    old_indices = torch.cat((retained, retained.new_full((arriving.numel(),), -1)))
    changed = torch.arange(count, device=device) >= retained_count
    old_to_new = torch.full((old_count,), -1, dtype=torch.long, device=device)
    old_to_new[retained] = torch.arange(retained_count, device=device)
    if previous is None:
        surviving_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        surviving_attr = positions.new_empty((0, 1))
    else:
        remapped = old_to_new[previous.graph.edge_index]
        survives = (remapped >= 0).all(dim=0)
        removed_endpoints = remapped[:, ~survives].flatten()
        changed[removed_endpoints[removed_endpoints >= 0]] = True
        surviving_edges = remapped[:, survives]
        surviving_attr = previous.graph.edge_attr[survives]
    edge_counts = torch.bincount(
        output_batch[surviving_edges[0]], minlength=cutoffs.numel(),
    )
    _check_guard(edge_counts, max_graph_edges)
    edge_parts = [surviving_edges]
    attr_parts = [surviving_attr]
    if arriving.numel():
        rows, occupied, sorted_nodes, boundaries = _occupied_cells(
            output_positions, output_batch, cutoffs.numel(), radius, position_dims,
        )
        axis = torch.tensor((-1, 0, 1), device=device, dtype=torch.long)
        offsets = torch.cartesian_prod(*([axis] * position_dims)).reshape(-1, position_dims)
        cells_per_query = offsets.shape[0]
        # Chunking bounds scratch only, never nodes, edges, or selected arrivals.
        effective_chunk = min(chunk_size, max(1, 1_048_576 // max(count, 1)))
        for start in range(retained_count, count, effective_chunk):
            stop = min(start + effective_chunk, count)
            query_cells = rows[start:stop, None, 1:] + offsets[None, :, :]
            query_batches = rows[start:stop, None, :1].expand(-1, cells_per_query, -1)
            queries = torch.cat((query_batches, query_cells), dim=2).flatten(0, 1)
            cell_ids = _cell_lookup(occupied, queries)
            cell_counts = boundaries[1, cell_ids.clamp_min(0)].masked_fill(cell_ids < 0, 0)
            cell_starts = boundaries[0, cell_ids.clamp_min(0)]
            candidate_count = int(cell_counts.sum())
            groups = torch.repeat_interleave(
                torch.arange(queries.shape[0], device=device), cell_counts,
                output_size=candidate_count,
            )
            source = groups.div(cells_per_query, rounding_mode="floor") + start
            starts = cell_counts.cumsum(0) - cell_counts
            target = sorted_nodes[
                (cell_starts - starts)[groups] + torch.arange(candidate_count, device=device)
            ]
            # Each unordered pair is discovered by its newer endpoint exactly once.
            pair = target < source
            source, target = source[pair], target[pair]
            normalized_distances = torch.linalg.vector_norm(
                (output_positions[source, :position_dims].double()
                 - output_positions[target, :position_dims].double()) / radius, dim=1,
            )
            valid = normalized_distances < 1.0
            source, target = source[valid], target[valid]
            normalized_distances = normalized_distances[valid]
            edge_counts = edge_counts + 2 * torch.bincount(
                output_batch[source], minlength=cutoffs.numel(),
            )
            _check_guard(edge_counts, max_graph_edges)
            changed[source] = True
            changed[target] = True
            edge_parts.append(torch.stack((torch.cat((source, target)), torch.cat((target, source)))))
            attr = normalized_distances.to(positions.dtype).unsqueeze(1)
            attr_parts.append(torch.cat((attr, attr)))
    graph = EventGraph(
        output_features, output_positions, torch.cat(edge_parts, dim=1), torch.cat(attr_parts),
    )
    return GraphUpdate(StreamGraph(graph, output_batch, output_timestamps), old_indices, changed)
