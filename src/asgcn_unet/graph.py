from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate

import torch
from torch import nn
from torch.nn import functional as F

from .ops import require_spline_backend, weighted_spline_sum

PAPER_CORE_VERSION = 2


@dataclass
class EventGraph:
    node_features: torch.Tensor
    positions: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    in_degree: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Cache topology-only normalization shared by every layer and timestep.

        ``in_degree`` is optional so callers that construct the original four-field
        graph remain source-compatible.  Materialized event graphs calculate it once
        here instead of rebuilding the same destination histogram in every spline
        layer (and every SNN timestep).
        """
        node_count = int(self.node_features.shape[0])
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if self.edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long indices")
        if self.in_degree is None:
            destination = self.edge_index[1]
            degree = torch.bincount(destination, minlength=node_count)
            if degree.shape != (node_count,):
                raise ValueError("edge_index contains a destination outside the graph")
            self.in_degree = degree
        assert self.in_degree is not None
        if self.in_degree.shape != (node_count,):
            raise ValueError("in_degree must contain one value per graph node")
        if self.in_degree.device != self.node_features.device:
            raise ValueError("in_degree and node_features must share a device")
        if self.in_degree.dtype != torch.long:
            raise TypeError("in_degree must use torch.long counts")


@dataclass
class PackedEventGraph:
    """A disjoint event graph with sample identities, including empty samples."""

    graph: EventGraph
    node_counts: tuple[int, ...]
    node_batch: torch.Tensor
    edge_counts: torch.Tensor


def _safe_batch_norm(norm: nn.BatchNorm1d, values: torch.Tensor) -> torch.Tensor:
    """Use running statistics when a graph has fewer than two events."""
    if norm.training and values.shape[0] < 2:
        return F.batch_norm(
            values,
            norm.running_mean,
            norm.running_var,
            norm.weight,
            norm.bias,
            training=False,
            momentum=0.0,
            eps=norm.eps,
        )
    return norm(values)


def uniformly_sample_events(events: torch.Tensor, factor: int = 1) -> torch.Tensor:
    """Apply the paper's deterministic event sampling factor R."""
    if isinstance(factor, bool) or int(factor) != factor:
        raise ValueError("event_sampling_factor must be an integer")
    factor = int(factor)
    if factor < 1:
        raise ValueError("event_sampling_factor must be at least 1")
    return events[::factor]


def prepare_event_nodes(
    events: torch.Tensor, sensor_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize Eq. (9)-(10) event nodes while retaining temporal order.

    The paper does not specify coordinate normalization. This implementation records
    the explicit assumption x,y,t in [0,1] and polarity in {-1,+1}. Graph distances
    use x,y,t by default; polarity remains a node feature.
    """
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("Events must have shape [N,4] with x,y,t,p columns")
    height, width = (int(value) for value in sensor_size)
    if height < 1 or width < 1:
        raise ValueError("sensor_size must contain positive height and width")
    if events.numel() == 0:
        return (
            torch.empty((0, 4), device=events.device, dtype=torch.float32),
            torch.empty((0, 4), device=events.device, dtype=torch.float32),
        )
    events = events.float()
    finite, unordered = torch.stack((
        torch.isfinite(events).all(), (events[1:, 2] < events[:-1, 2]).any()
    )).tolist()
    if not finite:
        raise ValueError("Event coordinates, timestamps, and polarities must be finite")
    if unordered:
        raise ValueError("Event timestamps must be monotonically non-decreasing")
    x = events[:, 0] / max(width - 1, 1)
    y = events[:, 1] / max(height - 1, 1)
    t = events[:, 2]
    t = (t - t[0]) / (t[-1] - t[0]).abs().clamp_min(1e-6)
    polarity = torch.where(events[:, 3] > 0, 1.0, -1.0)
    polarity_position = (polarity + 1.0) * 0.5
    positions = torch.stack((x, y, t, polarity_position), dim=-1)
    node_features = torch.stack((x, y, t, polarity), dim=-1)
    return node_features, positions


@lru_cache(maxsize=32)
def _spatial_hash_constants(
    cells_per_axis: int, position_dims: int, device_name: str, stream_handle: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache immutable, tiny hash tables, never event-dependent graph data.

    The caller only caches the model's one-to-four dimensional tables. Larger
    public radius-query dimensions use the uncached factory to avoid retaining
    an arbitrarily large Cartesian product. The CUDA stream is part of the cache
    key: asynchronously initialized tables are never consumed on another stream.
    Do not mutate the returned tensors.
    """
    device = torch.device(device_name)
    strides = torch.tensor(
        [cells_per_axis**dimension for dimension in range(position_dims)],
        device=device,
        dtype=torch.long,
    )
    offset_axis = torch.tensor((-1, 0, 1), device=device, dtype=torch.long)
    offsets = torch.cartesian_prod(*([offset_axis] * position_dims)).reshape(-1, position_dims)
    return strides, offsets


def _radius_graph_candidate_chunks(
    positions: torch.Tensor,
    radius: float,
    *,
    position_dims: int = 3,
    chunk_size: int = 512,
    node_batch: torch.Tensor | None = None,
    largest_graph_nodes: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield bounded adjacent-cell candidate chunks for an exact radius search."""
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("graph_radius must be positive")
    position_dims = int(position_dims)
    if position_dims < 1 or position_dims > positions.shape[1]:
        raise ValueError("graph_position_dims must select available position columns")
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("graph_chunk_size must be at least 1")

    count = int(positions.shape[0])
    device = positions.device
    if count == 0:
        return

    coordinates = positions[:, :position_dims]
    finite, outside = torch.stack((
        torch.isfinite(coordinates).all(), ((coordinates < 0) | (coordinates > 1)).any()
    )).tolist()
    if not finite:
        raise ValueError("Graph coordinates must be finite")
    if outside:
        raise ValueError("Normalized graph coordinates must lie in [0,1]")

    cells_per_axis = max(2, math.ceil(1.0 / radius) + 1)
    if cells_per_axis**position_dims >= torch.iinfo(torch.long).max:
        raise ValueError("graph_radius is too small for integer spatial hashing")
    factory = _spatial_hash_constants if position_dims <= 4 else _spatial_hash_constants.__wrapped__
    stream_handle = torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None
    strides, offsets = factory(cells_per_axis, position_dims, str(device), stream_handle)
    cells = torch.floor(coordinates / radius).to(torch.long)
    cells = cells.clamp_(0, cells_per_axis - 1)
    cell_hash = (cells * strides).sum(dim=1)
    # Namespace the spatial hash, rather than filtering cross-sample pairs after
    # allocating them. Samples with coincident coordinates remain independent.
    hash_volume = cells_per_axis**position_dims
    if node_batch is not None:
        cell_hash = cell_hash + node_batch * hash_volume
    sorted_hash, sorted_nodes = torch.sort(cell_hash)
    # Bound worst-case candidate materialization even if every event occupies one cell.
    candidate_node_bound = count if largest_graph_nodes is None else largest_graph_nodes
    effective_chunk_size = min(chunk_size, max(1, 4_000_000 // candidate_node_bound))
    # Lookup several small source/cell tables together, then copy their scalar
    # allocation counts once. Candidate distances are still materialized one
    # original bounded chunk at a time; this does not enlarge the edge scratch.
    lookup_chunks = max(1, min(8, 4096 // effective_chunk_size))
    lookup_size = effective_chunk_size * lookup_chunks
    cells_per_source = offsets.shape[0]
    for lookup_start in range(0, count, lookup_size):
        lookup_stop = min(lookup_start + lookup_size, count)
        neighbor_cells = cells[lookup_start:lookup_stop, None, :] + offsets[None, :, :]
        valid_cells = ((neighbor_cells >= 0) & (neighbor_cells < cells_per_axis)).all(dim=2)
        neighbor_hashes = (neighbor_cells * strides).sum(dim=2)
        if node_batch is not None:
            neighbor_hashes = neighbor_hashes + (
                node_batch[lookup_start:lookup_stop, None] * hash_volume
            )
        neighbor_hashes = neighbor_hashes.flatten()
        left_all = torch.searchsorted(sorted_hash, neighbor_hashes, right=False)
        right = torch.searchsorted(sorted_hash, neighbor_hashes, right=True)
        # Keep a fixed-size source/cell table: repeat_interleave already accepts
        # zero counts. Compacting valid/nonempty cells first adds five boolean
        # index operations (and their CUDA synchronization) without changing any
        # candidates. Mask invalid cells because their hashes may alias valid ones.
        counts_all = (right - left_all).masked_fill_(~valid_cells.flatten(), 0)
        ranges = [
            (start, min(start + effective_chunk_size, lookup_stop))
            for start in range(lookup_start, lookup_stop, effective_chunk_size)
        ]
        cell_ranges = [
            ((start - lookup_start) * cells_per_source,
             (stop - lookup_start) * cells_per_source)
            for start, stop in ranges
        ]
        candidate_counts = torch.stack([
            counts_all[start:stop].sum() for start, stop in cell_ranges
        ]).tolist()
        del neighbor_cells, valid_cells, neighbor_hashes, right
        for (start, _stop), (cell_start, cell_stop), candidate_count in zip(
            ranges, cell_ranges, candidate_counts, strict=True
        ):
            left = left_all[cell_start:cell_stop]
            counts = counts_all[cell_start:cell_stop]
            starts = counts.cumsum(0) - counts
            groups = torch.repeat_interleave(
                torch.arange(counts.numel(), device=device), counts, output_size=candidate_count
            )
            expanded_sources = groups.div(cells_per_source, rounding_mode="floor") + start
            # Expand the group identity once rather than repeating sources, left
            # boundaries and cumulative counts into three candidate-sized tensors.
            destination_offsets = (left - starts)[groups] + torch.arange(
                candidate_count, device=device
            )
            candidate_destinations = sorted_nodes[destination_offsets]
            # A generator retains its locals while the caller filters/materializes
            # edges. Release candidate-sized integer scratch before yielding.
            del groups, destination_offsets
            candidate_distances = torch.linalg.vector_norm(
                coordinates[expanded_sources] - coordinates[candidate_destinations], dim=1
            )
            yield expanded_sources, candidate_destinations, candidate_distances


def build_radius_graph(
    positions: torch.Tensor,
    radius: float,
    *,
    position_dims: int = 3,
    chunk_size: int = 512,
    max_edges: int | None = None,
    node_batch: torch.Tensor | None = None,
    node_counts: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the paper's exact radius graph with a uniform-cell candidate search.

    Every ordered edge direction is materialized so source-to-target aggregation is
    equivalent to an undirected graph.  Cells have width ``radius``; therefore only
    the 3^d adjacent cells can contain a valid neighbor.  Exact Euclidean filtering
    after candidate generation preserves the brute-force graph while avoiding the
    O(N^2) distance matrix on sparse event volumes. Optional ``node_batch`` and
    host-side ``node_counts`` enable one packed search over independent graphs;
    ``max_edges`` then remains a per-graph guard, never an aggregate batch cap.
    """
    radius = float(radius)
    if max_edges is not None:
        if isinstance(max_edges, bool) or int(max_edges) != max_edges:
            raise ValueError("max_graph_edges must be an integer or null")
        max_edges = int(max_edges)
        if max_edges < 1:
            raise ValueError("max_graph_edges must be at least 1 or null")

    count = int(positions.shape[0])
    device = positions.device
    if (node_batch is None) != (node_counts is None):
        raise ValueError("node_batch and node_counts must be supplied together")
    batch_edges: torch.Tensor | None = None
    if node_batch is not None:
        assert node_counts is not None
        _validate_packed_counts(node_counts, count)
        if node_batch.shape != (count,) or node_batch.dtype != torch.long:
            raise ValueError("node_batch must be a long tensor with one id per node")
        if node_batch.device != device:
            raise ValueError("node_batch and positions must share a device")
        if node_batch.numel() and bool(
            ((node_batch < 0) | (node_batch >= len(node_counts))).any()
        ):
            raise ValueError("node_batch contains an invalid sample id")
        actual_counts = torch.bincount(node_batch, minlength=len(node_counts))
        if not torch.equal(actual_counts, node_batch.new_tensor(node_counts)):
            raise ValueError("node_counts does not match node_batch")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("graph_radius must be positive")
        cells_per_axis = max(2, math.ceil(1.0 / radius) + 1)
        if cells_per_axis**int(position_dims) * max(1, len(node_counts)) >= (
            torch.iinfo(torch.long).max
        ):
            raise ValueError("graph_radius is too small for packed integer spatial hashing")
        batch_edges = torch.zeros(len(node_counts), device=device, dtype=torch.long)
    sources: list[torch.Tensor] = []
    destination_chunks: list[torch.Tensor] = []
    distances_kept: list[torch.Tensor] = []
    retained_edge_count = 0
    for expanded_sources, candidate_destinations, candidate_distances in (
        _radius_graph_candidate_chunks(
            positions,
            radius,
            position_dims=position_dims,
            chunk_size=chunk_size,
            node_batch=node_batch,
            largest_graph_nodes=max(node_counts, default=1) if node_counts is not None else None,
        )
    ):
        within_radius = (expanded_sources != candidate_destinations) & (
            candidate_distances < radius
        )
        # One compaction serves all three tensors. Tensor shape metadata supplies
        # the retained count without an additional scalar reduction/.item().
        kept = torch.nonzero(within_radius, as_tuple=True)[0]
        chunk_edge_count = kept.numel()
        kept_sources = expanded_sources.index_select(0, kept)
        if batch_edges is not None:
            assert node_batch is not None
            batch_edges.add_(torch.bincount(
                node_batch[kept_sources], minlength=batch_edges.numel()
            ))
            if max_edges is not None and bool((batch_edges > max_edges).any()):
                exceeded = torch.nonzero(batch_edges > max_edges).flatten().tolist()
                raise RuntimeError(
                    f"Radius graph exceeded max_graph_edges={max_edges:,} "
                    f"in batch sample(s) {exceeded}. Measure the exact topology and "
                    "accelerator memory before adjusting the explicit per-graph guard; "
                    "preserve graph_radius and event sampling."
                )
        elif (
            max_edges is not None
            and retained_edge_count + chunk_edge_count > max_edges
        ):
            raise RuntimeError(
                "Radius graph exceeded max_graph_edges="
                f"{max_edges:,} while processing {count:,} nodes. Measure the exact "
                "topology and accelerator memory before adjusting the explicit "
                "guard; preserve graph_radius and event sampling."
            )
        if chunk_edge_count == 0:
            continue
        retained_edge_count += chunk_edge_count
        sources.append(kept_sources)
        destination_chunks.append(candidate_destinations.index_select(0, kept))
        distances_kept.append(candidate_distances.index_select(0, kept))

    if not sources:
        return (
            torch.empty((2, 0), device=device, dtype=torch.long),
            torch.empty((0, 1), device=device, dtype=positions.dtype),
        )
    source = torch.cat(sources)
    destination = torch.cat(destination_chunks)
    distance = torch.cat(distances_kept)
    order = torch.argsort(source * count + destination)
    source = source[order]
    destination = destination[order]
    distance = distance[order]
    edge_index = torch.stack((source, destination), dim=0)
    edge_attr = (distance / radius).clamp(0.0, 1.0).unsqueeze(-1)
    return edge_index, edge_attr


def radius_graph_topology(
    positions: torch.Tensor,
    radius: float,
    *,
    position_dims: int = 3,
    chunk_size: int = 512,
) -> dict[str, int | float]:
    """Count the exact directed topology without materializing the complete edge list.

    ``candidate_directed_edges`` excludes self-pairs and counts adjacent-cell pairs
    before the exact radius predicate. No edge cap is accepted here: a pre-training
    scan must measure an over-limit graph rather than truncate or abort halfway.
    """
    count = int(positions.shape[0])
    in_degree = torch.zeros(count, device=positions.device, dtype=torch.long)
    candidate_count = 0
    radius = float(radius)
    for expanded_sources, candidate_destinations, candidate_distances in (
        _radius_graph_candidate_chunks(
            positions,
            radius,
            position_dims=position_dims,
            chunk_size=chunk_size,
        )
    ):
        nonself = expanded_sources != candidate_destinations
        # Every source appears in exactly one chunk and its own cell contributes
        # exactly one self-pair, including coincident but distinct event nodes.
        # Deduct those N pairs once after summing allocation sizes on the host.
        candidate_count += candidate_destinations.numel()
        within_radius = nonself & (candidate_distances < radius)
        # Fixed-shape integer scatter avoids per-chunk nonzero compaction,
        # bincount output-size discovery and scalar synchronizations on CUDA.
        in_degree.index_add_(0, candidate_destinations, within_radius.to(torch.long))

    if count:
        edge_count, max_degree, isolated_nodes = torch.stack(
            (in_degree.sum(), in_degree.max(), (in_degree == 0).sum())
        ).tolist()
    else:
        edge_count = max_degree = isolated_nodes = 0
    return {
        "nodes": count,
        "candidate_directed_edges": candidate_count - count,
        "actual_directed_edges": edge_count,
        "max_degree": max_degree,
        "isolated_nodes": isolated_nodes,
        "isolate_ratio": isolated_nodes / count if count else 0.0,
    }


def build_event_graph(
    events: torch.Tensor,
    sensor_size: tuple[int, int],
    *,
    event_sampling_factor: int,
    graph_radius: float,
    graph_position_dims: int,
    graph_chunk_size: int,
    max_graph_edges: int | None = None,
) -> EventGraph:
    sampled = uniformly_sample_events(events, event_sampling_factor)
    node_features, positions = prepare_event_nodes(sampled, sensor_size)
    edge_index, edge_attr = build_radius_graph(
        positions,
        graph_radius,
        position_dims=graph_position_dims,
        chunk_size=graph_chunk_size,
        max_edges=max_graph_edges,
    )
    return EventGraph(node_features, positions, edge_index, edge_attr)


def _validate_packed_counts(counts: tuple[int, ...], total: int) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("Packed event/node counts must be non-negative integers")
    if sum(counts) != total:
        raise ValueError("Packed event/node counts must sum to the tensor row count")


def build_event_graph_batch(
    events: torch.Tensor,
    event_counts: tuple[int, ...],
    sensor_size: tuple[int, int],
    *,
    event_sampling_factor: int,
    graph_radius: float,
    graph_position_dims: int,
    graph_chunk_size: int,
    max_graph_edges: int | None = None,
) -> PackedEventGraph:
    """Build exact independent graphs in one packed device operation stream.

    Counts come from CPU collation, so sampling offsets and allocation sizes need
    no device-to-host count discovery. Temporal normalization and deterministic
    factor-R sampling restart at each sample, exactly as ``build_event_graph``.
    No Python loop performs per-sample graph construction or device transfers.
    """
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("Events must have shape [N,4] with x,y,t,p columns")
    _validate_packed_counts(event_counts, int(events.shape[0]))
    # Share validation with the single-graph API without sampling this packed
    # tensor globally, which would shift the phase after uneven sample lengths.
    uniformly_sample_events(events[:0], event_sampling_factor)
    factor = int(event_sampling_factor)
    height, width = (int(value) for value in sensor_size)
    if height < 1 or width < 1:
        raise ValueError("sensor_size must contain positive height and width")
    node_counts = tuple((value + factor - 1) // factor for value in event_counts)
    node_total = sum(node_counts)
    counts_tensor = torch.tensor(node_counts, device=events.device, dtype=torch.long)
    node_batch = torch.repeat_interleave(
        torch.arange(len(node_counts), device=events.device), counts_tensor,
        output_size=node_total,
    )
    if node_total:
        node_starts = counts_tensor.cumsum(0) - counts_tensor
        if factor == 1:
            # The flat input already has exactly the required sample order.
            # Avoid an identity gather and its event-sized index/copy buffers.
            sampled = events.float()
        else:
            event_starts = torch.tensor(
                (0, *accumulate(event_counts)), device=events.device, dtype=torch.long
            )
            local_index = torch.arange(node_total, device=events.device) - node_starts[node_batch]
            source_index = event_starts[node_batch] + local_index * factor
            sampled = events[source_index].float()
        adjacent_same_sample = node_batch[1:] == node_batch[:-1]
        finite, unordered = torch.stack((
            torch.isfinite(sampled).all(),
            ((sampled[1:, 2] < sampled[:-1, 2]) & adjacent_same_sample).any(),
        )).tolist()
        if not finite:
            raise ValueError("Event coordinates, timestamps, and polarities must be finite")
        if unordered:
            raise ValueError("Event timestamps must be monotonically non-decreasing")
        first = sampled[node_starts[node_batch], 2]
        last = sampled[(node_starts + counts_tensor - 1)[node_batch], 2]
        x = sampled[:, 0] / max(width - 1, 1)
        y = sampled[:, 1] / max(height - 1, 1)
        t = (sampled[:, 2] - first) / (last - first).abs().clamp_min(1e-6)
        polarity = torch.where(sampled[:, 3] > 0, 1.0, -1.0)
        node_features = torch.stack((x, y, t, polarity), dim=-1)
        positions = torch.stack((x, y, t, (polarity + 1.0) * 0.5), dim=-1)
    else:
        node_features = torch.empty((0, 4), device=events.device, dtype=torch.float32)
        positions = torch.empty_like(node_features)
    edge_index, edge_attr = build_radius_graph(
        positions, graph_radius, position_dims=graph_position_dims,
        chunk_size=graph_chunk_size, max_edges=max_graph_edges,
        node_batch=node_batch, node_counts=node_counts,
    )
    edge_counts = torch.bincount(node_batch[edge_index[0]], minlength=len(node_counts))
    return PackedEventGraph(
        EventGraph(node_features, positions, edge_index, edge_attr),
        node_counts, node_batch, edge_counts,
    )


def linear_open_bspline_basis(
    pseudo: torch.Tensor, kernel_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return indices and weights for an open degree-1 B-spline basis.

    For each scalar pseudo-coordinate this yields the two non-zero basis terms.
    The weights form a partition of unity, including exact endpoints.
    """
    kernel_size = int(kernel_size)
    if kernel_size < 2:
        raise ValueError("spline_kernel_size must be at least 2")
    if pseudo.ndim != 2 or pseudo.shape[1] != 1:
        raise ValueError("ASGCN edge pseudo-coordinates must have shape [E,1]")
    if not torch.isfinite(pseudo).all():
        raise ValueError("ASGCN edge pseudo-coordinates must be finite")
    if bool(((pseudo < 0) | (pseudo > 1)).any()):
        raise ValueError("ASGCN edge pseudo-coordinates must lie in [0,1]")

    scaled = pseudo[:, 0] * float(kernel_size - 1)
    cell = torch.floor(scaled)
    left = cell.to(torch.long).remainder(kernel_size)
    right = (left + 1).remainder(kernel_size)
    right_weight = scaled - cell
    left_weight = 1.0 - right_weight
    indices = torch.stack((left, right), dim=-1)
    weights = torch.stack((left_weight, right_weight), dim=-1)
    return indices, weights


class PaperSplineConv(nn.Module):
    """Eq. (11) weighted open B-spline convolution followed by BN.

    ASGCN names a PyG weighted B-spline tensor-product kernel but omits its
    hyperparameters. The reconstruction adaptation therefore exposes them in every
    config. Degree 1 is implemented directly in PyTorch and matches the public
    SplineCNN operator definition for scalar pseudo-coordinates.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 5,
        degree: int = 1,
        root_weight: bool = True,
        bias: bool = True,
        edge_chunk_size: int | None = 65_536,
        spline_backend: str = "torch",
    ) -> None:
        super().__init__()
        if int(degree) != 1:
            raise ValueError("Only the configured open degree-1 B-spline is supported")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.degree = int(degree)
        if self.kernel_size < 2:
            raise ValueError("spline_kernel_size must be at least 2")
        if edge_chunk_size is not None:
            if (
                isinstance(edge_chunk_size, bool)
                or int(edge_chunk_size) != edge_chunk_size
                or int(edge_chunk_size) < 1
            ):
                raise ValueError("spline_chunk_size must be a positive integer or null")
            edge_chunk_size = int(edge_chunk_size)
        self.edge_chunk_size = edge_chunk_size
        if spline_backend not in {"torch", "torch_fused", "triton"}:
            raise ValueError("spline_backend must be torch, torch_fused, or triton")
        self.spline_backend = spline_backend
        self.weight = nn.Parameter(
            torch.empty(self.kernel_size, self.in_channels, self.out_channels)
        )
        self.root = (
            nn.Parameter(torch.empty(self.in_channels, self.out_channels)) if root_weight else None
        )
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bias else None
        self.norm = nn.BatchNorm1d(self.out_channels)
        self.register_buffer("bn_bypassed", torch.tensor(False), persistent=True)
        self.register_buffer("snn_normalized", torch.tensor(False), persistent=True)
        self.register_buffer(
            "calibration_activation_max",
            torch.ones(self.out_channels),
            persistent=True,
        )
        self.register_buffer(
            "normalization_scale",
            torch.ones(self.out_channels),
            persistent=True,
        )
        self.register_buffer(
            "dead_channel_mask",
            torch.zeros(self.out_channels, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer("threshold", torch.ones(self.out_channels), persistent=True)
        self._bn_is_folded = False
        self._snn_is_normalized = False
        self.reset_parameters()

    @property
    def activation_max(self) -> torch.Tensor:
        """Compatibility alias for the raw, observed calibration maximum."""
        return self.calibration_activation_max

    def reset_parameters(self) -> None:
        weight_bound = 1.0 / math.sqrt(self.kernel_size * self.in_channels)
        nn.init.uniform_(self.weight, -weight_bound, weight_bound)
        if self.root is not None:
            root_bound = 1.0 / math.sqrt(self.in_channels)
            nn.init.uniform_(self.root, -root_bound, root_bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.norm.reset_parameters()
        self.bn_bypassed.fill_(False)
        self.snn_normalized.fill_(False)
        self.calibration_activation_max.fill_(1.0)
        self.normalization_scale.fill_(1.0)
        self.dead_channel_mask.fill_(False)
        self.threshold.fill_(1.0)
        self._bn_is_folded = False
        self._snn_is_normalized = False

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # Architecture-v2 checkpoints written before the calibration-state split
        # stored either the raw ANN maximum (unconverted checkpoints) or the
        # effective Eq. (6) scale (converted checkpoints) under ``activation_max``.
        # The no-dead-channel case is exactly recoverable. A legacy converted
        # checkpoint that reported dead channels remains rejected later because
        # their channel identities were not present in its tensor state.
        legacy_key = prefix + "activation_max"
        split_keys = (
            prefix + "calibration_activation_max",
            prefix + "normalization_scale",
            prefix + "dead_channel_mask",
        )
        if legacy_key in state_dict and all(key not in state_dict for key in split_keys):
            legacy_max = state_dict.get(legacy_key)
            normalized_flag = state_dict.get(prefix + "snn_normalized")
            if isinstance(legacy_max, torch.Tensor):
                state_dict.pop(legacy_key)
                state_dict[split_keys[0]] = legacy_max
                is_normalized = (
                    isinstance(normalized_flag, torch.Tensor)
                    and normalized_flag.numel() == 1
                    and bool(normalized_flag.item())
                )
                state_dict[split_keys[1]] = (
                    legacy_max.clone()
                    if is_normalized
                    else torch.ones_like(legacy_max)
                )
                state_dict[split_keys[2]] = legacy_max <= 0
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._bn_is_folded = bool(self.bn_bypassed.item())
        self._snn_is_normalized = bool(self.snn_normalized.item())

    def spline_aggregate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        basis_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        in_degree: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.spline_backend == "triton":
            require_spline_backend(self.spline_backend, x.device)
        source, destination = edge_index
        output = torch.zeros((x.shape[0], self.out_channels), device=x.device, dtype=x.dtype)
        if source.numel() > 0:
            indices, basis = (
                basis_cache
                if basis_cache is not None
                else linear_open_bspline_basis(edge_attr, self.kernel_size)
            )
            # Project nodes once for every control point, then gather only the two
            # degree-1 basis terms that are active on each edge.
            projected = torch.einsum("ni,kio->nko", x, self.weight)
            edge_count = int(source.numel())
            chunk_size = edge_count if self.edge_chunk_size is None else self.edge_chunk_size
            output = weighted_spline_sum(
                projected, source, destination, indices, basis, chunk_size, x.dtype,
                backend=self.spline_backend,
            )
            if in_degree is None:
                in_degree = torch.bincount(destination, minlength=x.shape[0])
            if in_degree.shape != (x.shape[0],):
                raise ValueError("in_degree must contain one value per graph node")
            degree = in_degree.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            output = output / degree.clamp_min(1.0)
        return output

    def affine(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        basis_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        in_degree: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.spline_aggregate(x, edge_index, edge_attr, basis_cache, in_degree)
        if self.root is not None:
            output = output + x @ self.root
        if self.bias is not None:
            output = output + self.bias
        return output

    def preactivation(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        basis_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        in_degree: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.affine(x, edge_index, edge_attr, basis_cache, in_degree)
        return output if self._bn_is_folded else _safe_batch_norm(self.norm, output)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        basis_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        in_degree: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preactivation = self.preactivation(
            x, edge_index, edge_attr, basis_cache, in_degree
        )
        return torch.relu(preactivation), preactivation

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        """Fold Eq. (13) into every output term of Eq. (11), yielding Eq. (14)."""
        if self._bn_is_folded:
            return
        if self.training:
            raise RuntimeError("Call eval() before folding BatchNorm")
        scale = self.norm.weight / torch.sqrt(self.norm.running_var + self.norm.eps)
        self.weight.mul_(scale.view(1, 1, -1))
        if self.root is not None:
            self.root.mul_(scale.view(1, -1))
        if self.bias is None:
            raise RuntimeError("BN folding requires an affine convolution bias")
        self.bias.copy_((self.bias - self.norm.running_mean) * scale + self.norm.bias)
        self.bn_bypassed.fill_(True)
        self._bn_is_folded = True

    @torch.no_grad()
    def apply_parameter_normalization(
        self, input_scale: torch.Tensor, output_scale: torch.Tensor
    ) -> None:
        """Apply Eq. (6): W_l <- W_l lambda_(l-1)/lambda_l, b_l <- b_l/lambda_l."""
        if not self._bn_is_folded:
            raise RuntimeError("Fold BatchNorm before ANN-to-SNN parameter normalization")
        if self._snn_is_normalized:
            raise RuntimeError("ANN-to-SNN parameter normalization was already applied")
        input_scale = input_scale.to(device=self.weight.device, dtype=self.weight.dtype)
        output_scale = output_scale.to(device=self.weight.device, dtype=self.weight.dtype)
        if input_scale.shape != (self.in_channels,):
            raise ValueError("Input activation scale does not match spline input channels")
        if output_scale.shape != (self.out_channels,):
            raise ValueError("Output activation scale does not match spline output channels")
        input_scale = input_scale.clamp_min(1e-6)
        output_scale = output_scale.clamp_min(1e-6)
        raw_max = self.calibration_activation_max.to(
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        if not bool(torch.isfinite(raw_max).all()) or bool((raw_max < 0).any()):
            raise ValueError("Calibration activation maximum must be finite and non-negative")
        dead_mask = raw_max <= 0
        expected_output_scale = torch.where(
            dead_mask,
            torch.ones_like(raw_max),
            raw_max,
        ).clamp_min(1e-6)
        if not torch.equal(output_scale, expected_output_scale):
            raise ValueError(
                "Output activation scale must be the effective scale derived from "
                "the raw calibration maximum"
            )
        self.weight.mul_(input_scale.view(1, -1, 1))
        self.weight.div_(output_scale.view(1, 1, -1))
        if self.root is not None:
            self.root.mul_(input_scale.view(-1, 1))
            self.root.div_(output_scale.view(1, -1))
        if self.bias is not None:
            self.bias.div_(output_scale)
        self.normalization_scale.copy_(output_scale)
        self.dead_channel_mask.copy_(dead_mask)
        self.threshold.fill_(1.0)
        self.snn_normalized.fill_(True)
        self._snn_is_normalized = True


class ASGCNEncoder(nn.Module):
    """Public-equation-derived static ASGCN graph core for reconstruction."""

    def __init__(
        self,
        hidden_dim: int = 64,
        graph_layers: int = 6,
        *,
        spline_kernel_size: int = 5,
        spline_degree: int = 1,
        spline_root_weight: bool = True,
        spline_chunk_size: int | None = 65_536,
        spline_backend: str = "torch",
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        graph_layers = int(graph_layers)
        if graph_layers < 1:
            raise ValueError("graph_layers must be at least 1 for ASGCN")
        self.hidden_dim = hidden_dim
        self.spline_backend = spline_backend
        channels = [4] + [hidden_dim] * graph_layers
        self.layers = nn.ModuleList(
            [
                PaperSplineConv(
                    channels[index],
                    channels[index + 1],
                    kernel_size=spline_kernel_size,
                    degree=spline_degree,
                    root_weight=spline_root_weight,
                    bias=True,
                    edge_chunk_size=spline_chunk_size,
                    spline_backend=spline_backend,
                )
                for index in range(graph_layers)
            ]
        )
        self.register_buffer(
            "calibration_samples_seen",
            torch.zeros(graph_layers, dtype=torch.long),
            persistent=True,
        )

    def _basis_cache(self, graph: EventGraph) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Compute the fixed edge basis once for all layers and IF timesteps."""
        if graph.edge_attr.shape[0] == 0:
            return None
        return linear_open_bspline_basis(
            graph.edge_attr,
            self.layers[0].kernel_size,
        )

    def forward_ann(
        self, graph: EventGraph, return_activations: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        require_spline_backend(self.spline_backend, graph.node_features.device)
        hidden = graph.node_features
        activations: list[torch.Tensor] = []
        basis_cache = self._basis_cache(graph)
        for layer in self.layers:
            hidden, _preactivation = layer(
                hidden,
                graph.edge_index,
                graph.edge_attr,
                basis_cache,
                graph.in_degree,
            )
            if return_activations:
                # ``hidden`` is already ReLU(preactivation); retain that tensor
                # instead of launching and storing an identical second ReLU.
                activations.append(hidden)
        return hidden, activations

    def forward_snn(
        self,
        graph: EventGraph,
        simulation_steps: int = 16,
        dynamics: str = "literal_eq15",
        *,
        node_batch: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run explicit IF timesteps using literal Eq. (15) or a standard-IF control."""
        require_spline_backend(self.spline_backend, graph.node_features.device)
        if isinstance(simulation_steps, bool) or int(simulation_steps) != simulation_steps:
            raise ValueError("simulation_steps must be an integer")
        simulation_steps = int(simulation_steps)
        if simulation_steps < 1:
            raise ValueError("simulation_steps must be at least 1")
        if dynamics not in {"literal_eq15", "standard_if"}:
            raise ValueError("snn_dynamics must be 'literal_eq15' or 'standard_if'")
        if any(not layer._snn_is_normalized for layer in self.layers):
            raise RuntimeError("SNN inference requires Eq. (6) parameter normalization")
        node_count = int(graph.node_features.shape[0])
        if (node_batch is None) != (batch_size is None):
            raise ValueError("node_batch and batch_size must be supplied together")
        nodes_per_sample: torch.Tensor | None = None
        if node_batch is not None:
            if (
                isinstance(batch_size, bool) or not isinstance(batch_size, int)
                or batch_size < 1
            ):
                raise ValueError("batch_size must be a positive integer")
            if node_batch.shape != (node_count,) or node_batch.dtype != torch.long:
                raise ValueError("node_batch must be a long tensor with one id per node")
            if node_batch.device != graph.node_features.device:
                raise ValueError("node_batch and node features must share a device")
            if node_count and bool(((node_batch < 0) | (node_batch >= batch_size)).any()):
                raise ValueError("node_batch contains an invalid sample id")
            nodes_per_sample = torch.bincount(node_batch, minlength=batch_size).to(
                graph.node_features.dtype
            )
        statistic_shape = () if node_batch is None else (batch_size,)
        if node_count == 0:
            empty = graph.node_features.new_empty((0, self.hidden_dim))
            zeros = [graph.node_features.new_zeros(statistic_shape) for _ in self.layers]
            return empty, zeros

        membranes = [
            layer.threshold.to(graph.node_features).expand(node_count, -1).clone() * 0.5
            for layer in self.layers
        ]
        previous_spikes = (
            [
                graph.node_features.new_zeros((node_count, layer.out_channels))
                for layer in self.layers
            ]
            if dynamics == "literal_eq15"
            else None
        )
        output_spike_sum = graph.node_features.new_zeros(
            (node_count, self.layers[-1].out_channels)
        )
        active_counts = [graph.node_features.new_zeros(statistic_shape) for _ in self.layers]
        basis_cache = self._basis_cache(graph)
        # The analog event features and topology are constant for all IF steps.
        # ``affine`` has no BatchNorm/dropout state updates, so its first-layer
        # current can be reused without changing the recurrence or autograd.
        first_current = self.layers[0].affine(
            graph.node_features,
            graph.edge_index,
            graph.edge_attr,
            basis_cache,
            graph.in_degree,
        )
        # Resolve conversions against each layer's actual integration dtype, not
        # the input dtype: autocast plus a float32 bias can promote the current.
        thresholds: list[torch.Tensor | None] = [None for _ in self.layers]

        for _ in range(simulation_steps):
            hidden = graph.node_features
            for index, layer in enumerate(self.layers):
                current = (
                    first_current
                    if index == 0
                    else layer.affine(
                        hidden,
                        graph.edge_index,
                        graph.edge_attr,
                        basis_cache,
                        graph.in_degree,
                    )
                )
                integrated = membranes[index] + current
                if previous_spikes is not None:
                    # This is the paper's written +h_i^l(t-1) recurrence. It is
                    # intentionally separate from the standard rate-conversion IF
                    # control because the paper does not resolve their mismatch.
                    integrated = integrated + previous_spikes[index]
                threshold = thresholds[index]
                if threshold is None:
                    threshold = layer.threshold.to(integrated).expand_as(integrated)
                    thresholds[index] = threshold
                # A scalar zero avoids allocating a full node/channel tensor on
                # every layer/step while preserving threshold-valued spikes.
                spikes = torch.where(integrated >= threshold, threshold, 0.0)
                membranes[index] = integrated - spikes
                if previous_spikes is not None:
                    previous_spikes[index] = spikes
                if index == len(self.layers) - 1:
                    output_spike_sum = output_spike_sum + spikes
                # Counts are non-differentiable statistics, unlike membrane and
                # spike tensors; updating only these scalar accumulators is safe.
                if node_batch is None:
                    active_counts[index].add_((spikes != 0).sum())
                else:
                    # Reduce exact integer counts before casting once per step,
                    # matching the single-graph accumulator even at low precision.
                    per_sample_counts = torch.zeros(
                        batch_size, device=node_batch.device, dtype=torch.long
                    )
                    per_sample_counts.index_add_(0, node_batch, (spikes != 0).sum(dim=1))
                    active_counts[index].add_(per_sample_counts)
                hidden = spikes

        firing_rates = [
            active.to(graph.node_features.dtype)
            / (
                float(simulation_steps * max(1, node_count * layer.out_channels))
                if nodes_per_sample is None
                else (nodes_per_sample * layer.out_channels).clamp_min(1) * simulation_steps
            )
            for active, layer in zip(active_counts, self.layers, strict=True)
        ]
        return output_spike_sum / float(simulation_steps), firing_rates

    @torch.no_grad()
    def update_activation_maxima(
        self, activations: list[torch.Tensor], *, sample_count: int = 1
    ) -> None:
        """Reduce maxima over packed nodes, counting original nonempty samples."""
        if (
            isinstance(sample_count, bool) or not isinstance(sample_count, int)
            or sample_count < 0
        ):
            raise ValueError("sample_count must be a non-negative integer")
        if len(activations) != len(self.layers):
            raise ValueError("Activation count does not match graph layer count")
        for index, (layer, activation) in enumerate(zip(self.layers, activations, strict=True)):
            if activation.numel() == 0:
                continue
            if sample_count == 0:
                raise ValueError("Nonempty calibration activations require sample_count > 0")
            if activation.ndim != 2 or activation.shape[1] != layer.out_channels:
                raise ValueError("Calibration activation shape does not match graph layer")
            if not bool(torch.isfinite(activation).all()):
                raise FloatingPointError("Non-finite activation encountered during calibration")
            maxima = activation.amax(dim=0)
            layer.calibration_activation_max.copy_(
                torch.maximum(layer.calibration_activation_max, maxima)
            )
            layer.dead_channel_mask.copy_(layer.calibration_activation_max <= 0)
            self.calibration_samples_seen[index].add_(sample_count)

    @torch.no_grad()
    def reset_activation_maxima(self) -> None:
        for layer in self.layers:
            layer.calibration_activation_max.zero_()
            layer.normalization_scale.fill_(1.0)
            layer.dead_channel_mask.fill_(True)
        self.calibration_samples_seen.zero_()

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        self.eval()
        for layer in self.layers:
            layer.fold_batch_norm()

    @torch.no_grad()
    def apply_parameter_normalization(self) -> None:
        missing = torch.nonzero(self.calibration_samples_seen == 0).flatten().tolist()
        if missing:
            raise RuntimeError(
                "Cannot apply Eq. (6): no non-empty calibration activation for layer(s) "
                + ", ".join(str(index) for index in missing)
            )
        previous_scale = self.layers[0].weight.new_ones(self.layers[0].in_channels)
        for layer in self.layers:
            measured = layer.calibration_activation_max.detach().clone()
            # A ReLU channel that stayed identically zero has no usable lambda.
            # Keep it at unit scale instead of dividing its parameters by epsilon.
            output_scale = torch.where(
                measured > 0,
                measured,
                torch.ones_like(measured),
            ).clamp_min(1e-6)
            layer.apply_parameter_normalization(previous_scale, output_scale)
            previous_scale = layer.normalization_scale.detach().clone()

    def calibration_summary(self) -> dict[str, list[int] | int]:
        dead_channels = [
            int((layer.calibration_activation_max <= 0).sum().item())
            for layer in self.layers
        ]
        valid_samples = [int(value) for value in self.calibration_samples_seen.tolist()]
        return {
            "valid_samples_per_layer": valid_samples,
            "minimum_valid_samples": min(valid_samples, default=0),
            "dead_channels_per_layer": dead_channels,
        }

    def output_activation_scale(self, reference: torch.Tensor) -> torch.Tensor:
        """Return lambda_L used to express spikes in the analog decoder's units."""
        if not self.layers[-1]._snn_is_normalized:
            raise RuntimeError("Output activation scale is available after Eq. (6) conversion")
        return self.layers[-1].normalization_scale.to(reference)
