"""Exact radius adjacency stored as O(N) nodes and a cached occupied-cell index.

There is deliberately no ``edge_index`` placeholder: consumers must explicitly
support this representation. Neighbors are regenerated in bounded chunks, without
changing the strict float64 radius predicate, node set, or independent streams.
The index is immutable geometry, not a differentiable coordinate operator.
"""

from __future__ import annotations

import math
from numbers import Real

import torch

from .radius_candidates import coordinate_bounds, prune_cell_counts


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_geometry(positions, node_batch, batch_size, radius, position_dims,
                       chunk_size, candidate_pair_budget):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 0:
        raise ValueError("batch_size must be a nonnegative integer")
    if isinstance(radius, bool) or not isinstance(radius, Real) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if isinstance(position_dims, bool) or not isinstance(position_dims, int) or not 1 <= position_dims <= 4:
        raise ValueError("position_dims must be an integer from 1 to 4")
    _positive_integer(chunk_size, "chunk_size")
    _positive_integer(candidate_pair_budget, "candidate_pair_budget")
    if (not isinstance(positions, torch.Tensor) or positions.layout != torch.strided
            or positions.ndim != 2 or positions.shape[1] != 4 or not positions.is_floating_point()):
        raise ValueError("positions must be a floating strided tensor with shape [N,4]")
    if positions.requires_grad:
        raise ValueError("Implicit radius positions are fixed input geometry; position gradients are unsupported")
    if (not isinstance(node_batch, torch.Tensor) or node_batch.layout != torch.strided
            or node_batch.shape != (positions.shape[0],) or node_batch.dtype != torch.long
            or node_batch.device != positions.device):
        raise ValueError("node_batch must be a long [N] tensor on the positions device")
    if not bool(torch.stack((torch.isfinite(positions).all(),
                            ((node_batch >= 0) & (node_batch < batch_size)).all())).all()):
        raise ValueError("positions must be finite and node_batch must identify an independent stream")


def _version(tensor):
    # Inference-mode tensors intentionally have no PyTorch version counter. They
    # are still immutable by contract; ordinary/autograd tensors are checked.
    return None if torch.is_inference(tensor) else tensor._version


class ImplicitRadiusIndex:
    """An index-only builder, reusable across layers, ticks, and backward.

    Building the index does not enumerate edges. The caller owns the tensors and
    must not mutate geometry while the index is alive (including inference mode).
    All streams share a tensor index; no Python per-stream graph loop is used.
    """

    def __init__(self, positions, node_batch, *, batch_size, radius, position_dims=3,
                 chunk_size=512, candidate_pair_budget=1_048_576):
        _validate_geometry(positions, node_batch, batch_size, radius, position_dims,
                           chunk_size, candidate_pair_budget)
        self.positions, self.node_batch = positions, node_batch
        self.batch_size, self.radius = batch_size, float(radius)
        self.position_dims, self.chunk_size = position_dims, chunk_size
        self.candidate_pair_budget = candidate_pair_budget
        self._versions = (_version(positions), _version(node_batch))
        if positions.shape[0]:
            # Lazy import avoids graph/stream_graph dispatch import cycles.
            from .stream_graph import _occupied_cells

            self._rows, self._occupied, self._sorted_nodes, self._boundaries = _occupied_cells(
                positions, node_batch, batch_size, self.radius, position_dims,
            )
        else:
            self._rows = node_batch.new_empty((0, position_dims + 1))
            self._occupied = self._rows
            self._sorted_nodes = node_batch.new_empty(0)
            self._boundaries = node_batch.new_empty((2, 0))
        axis = node_batch.new_tensor((-1, 0, 1))
        self._offsets = torch.cartesian_prod(*([axis] * position_dims)).reshape(-1, position_dims)
        self._bounds = coordinate_bounds(positions, self._sorted_nodes, self._boundaries, position_dims)

    def validate_integrity(self):
        if self._versions != (_version(self.positions), _version(self.node_batch)):
            raise RuntimeError("Implicit radius geometry was modified after its index was built")

    def validate_destinations(self, destinations):
        if (not isinstance(destinations, torch.Tensor) or destinations.layout != torch.strided
                or destinations.ndim != 1 or destinations.dtype != torch.long
                or destinations.device != self.positions.device):
            raise ValueError("destinations must be a long vector on the graph device")
        if not bool(((destinations >= 0) & (destinations < self.positions.shape[0])).all()):
            raise ValueError("destinations contain an invalid node index")
        if destinations.unique().numel() != destinations.numel():
            raise ValueError("destinations must be unique")

    def iter_directed_neighbors(self, destinations=None, *, stats=None, active_sources=None):
        """Yield source, destination, distance/radius for all incoming neighbors.

        Destination selection never restricts sources or the full degree. Each
        directed pair appears exactly once. Candidate scratch is bounded even for
        a dense cell; chunking does not cap/truncate edges. A scalar candidate
        count is required per query chunk by this portable implementation.
        An optional source mask restricts MESSAGE queries, not graph degrees.
        Inactive sources are excluded before pair expansion/distance work.
        """
        from .stream_graph import _cell_lookup

        self.validate_integrity()
        if destinations is None:
            destinations = torch.arange(self.positions.shape[0], device=self.positions.device)
        else:
            self.validate_destinations(destinations)
        if stats is not None:
            for name in ("query_chunks", "candidate_chunks", "candidate_pairs_visited", "peak_candidate_pairs"):
                stats.setdefault(name, 0)
        sorted_nodes, boundaries = self._sorted_nodes, self._boundaries
        if active_sources is not None:
            if (not isinstance(active_sources, torch.Tensor) or active_sources.layout != torch.strided
                    or active_sources.shape != (self.positions.shape[0],)
                    or active_sources.dtype != torch.bool or active_sources.device != self.positions.device):
                raise ValueError("active_sources must be a bool [N] tensor on the graph device")
            active = active_sources[sorted_nodes]
            prefix = torch.cat((self.node_batch.new_zeros(1), active.long().cumsum(0)))
            starts = prefix[boundaries[0]]
            counts = prefix[boundaries[0] + boundaries[1]] - starts
            boundaries = torch.stack((starts, counts))
            sorted_nodes = sorted_nodes[active]
        if not sorted_nodes.numel():
            return
        cells_per_query = self._offsets.shape[0]
        for query_start in range(0, destinations.numel(), self.chunk_size):
            selected = destinations[query_start:query_start + self.chunk_size]
            query_rows = self._rows[selected, None, :].expand(-1, cells_per_query, -1).clone()
            query_rows[:, :, 1:] += self._offsets
            cell_ids = _cell_lookup(self._occupied, query_rows.reshape(-1, self.position_dims + 1))
            safe_ids = cell_ids.clamp_min(0)
            counts = prune_cell_counts(
                self.positions[selected], cell_ids.reshape(-1, cells_per_query),
                boundaries, self._bounds, self.radius,
            ).flatten()
            cell_starts = boundaries[0, safe_ids]
            ends = counts.cumsum(0)
            starts = ends - counts
            candidates = int(ends[-1])
            if stats is not None:
                stats["query_chunks"] += 1
                stats["candidate_pairs_visited"] += candidates
            for start in range(0, candidates, self.candidate_pair_budget):
                stop = min(start + self.candidate_pair_budget, candidates)
                flat = torch.arange(start, stop, device=self.positions.device)
                groups = torch.searchsorted(ends, flat, right=True)
                destination = selected[groups // cells_per_query]
                source = sorted_nodes[cell_starts[groups] + flat - starts[groups]]
                nonself = source != destination
                source, destination = source[nonself], destination[nonself]
                distances = torch.linalg.vector_norm(
                    (self.positions[source, :self.position_dims].double()
                     - self.positions[destination, :self.position_dims].double()) / self.radius, dim=1,
                )
                keep = distances < 1
                if stats is not None:
                    stats["candidate_chunks"] += 1
                    stats["peak_candidate_pairs"] = max(stats["peak_candidate_pairs"], stop - start)
                yield source[keep], destination[keep], distances[keep].to(self.positions.dtype)[:, None]


class ImplicitRadiusGraph:
    """Explicit adjacency representation with exact cached node degrees/counts.

    ``from_counted_nodes`` trusts the caller's geometric degree calculation after
    checking cardinality consistency. Untrusted checkpoint restoration must also
    reconstruct/verify the geometry. The occupied-cell index is built lazily so
    pack/split/clone operations do not enumerate edges or unnecessarily index them.
    """

    @classmethod
    def from_counted_nodes(cls, node_features, positions, node_batch, in_degree, *,
                           batch_size, radius, position_dims=3, chunk_size=512,
                           candidate_pair_budget=1_048_576, edge_counts=None,
                           index=None, max_graph_edges=None):
        _validate_geometry(positions, node_batch, batch_size, radius, position_dims,
                           chunk_size, candidate_pair_budget)
        count = positions.shape[0]
        if (not isinstance(node_features, torch.Tensor) or node_features.layout != torch.strided
                or node_features.shape != (count, 4) or not node_features.is_floating_point()
                or node_features.device != positions.device):
            raise ValueError("node_features must be a floating [N,4] tensor on the graph device")
        if (not isinstance(in_degree, torch.Tensor) or in_degree.layout != torch.strided
                or in_degree.shape != (count,) or in_degree.dtype != torch.long
                or in_degree.device != positions.device):
            raise ValueError("in_degree must be a long [N] tensor on the graph device")
        calculated = node_batch.new_zeros(batch_size).index_add_(0, node_batch, in_degree)
        if edge_counts is None:
            edge_counts = calculated
        if (not isinstance(edge_counts, torch.Tensor) or edge_counts.layout != torch.strided
                or edge_counts.shape != (batch_size,) or edge_counts.dtype != torch.long
                or edge_counts.device != positions.device):
            raise ValueError("edge_counts must be a long [B] tensor on the graph device")
        node_counts = torch.bincount(node_batch, minlength=batch_size)
        if not bool(torch.stack((
            torch.isfinite(node_features).all(), (in_degree >= 0).all(),
            (in_degree < node_counts[node_batch]).all(), (edge_counts >= 0).all(),
            (edge_counts % 2 == 0).all(), (calculated == edge_counts).all(),
        )).all()):
            raise ValueError("Invalid implicit graph features, node degrees, or directed edge counts")
        if max_graph_edges is not None:
            _positive_integer(max_graph_edges, "max_graph_edges")
            if bool((edge_counts > max_graph_edges).any()):
                raise RuntimeError(f"Stream radius graph exceeded max_graph_edges={max_graph_edges:,}; no edges were truncated")
        if index is not None:
            if (not isinstance(index, ImplicitRadiusIndex) or index.positions is not positions
                    or index.node_batch is not node_batch or index.batch_size != batch_size
                    or index.radius != float(radius) or index.position_dims != position_dims
                    or index.chunk_size != chunk_size or index.candidate_pair_budget != candidate_pair_budget):
                raise ValueError("Provided implicit radius index does not match graph geometry/configuration")
            index.validate_integrity()
        result = cls()
        result.node_features, result.positions, result.node_batch = node_features, positions, node_batch
        result.in_degree, result.edge_counts = in_degree, edge_counts
        result.edge_count = int(edge_counts.sum())
        result.batch_size, result.radius, result.position_dims = batch_size, float(radius), position_dims
        result.chunk_size, result.candidate_pair_budget = chunk_size, candidate_pair_budget
        result._index = index
        result._versions = (_version(positions), _version(node_batch), _version(in_degree), _version(edge_counts))
        return result

    def validate_integrity(self):
        if self._versions != tuple(_version(value) for value in (
                self.positions, self.node_batch, self.in_degree, self.edge_counts)):
            raise RuntimeError("Implicit radius geometry/degrees were modified after graph creation")

    @property
    def index(self):
        self.validate_integrity()
        if self._index is None:
            self._index = ImplicitRadiusIndex(
                self.positions, self.node_batch, batch_size=self.batch_size, radius=self.radius,
                position_dims=self.position_dims, chunk_size=self.chunk_size,
                candidate_pair_budget=self.candidate_pair_budget,
            )
        return self._index

    def iter_directed_neighbors(self, destinations=None, *, stats=None, active_sources=None):
        return self.index.iter_directed_neighbors(destinations, stats=stats, active_sources=active_sources)


def build_implicit_radius_graph(node_features, positions, node_batch, *, batch_size,
                                radius, position_dims=3, chunk_size=512,
                                candidate_pair_budget=1_048_576, max_graph_edges=None):
    """Count exact degrees once without retaining any edge-size tensor."""
    index = ImplicitRadiusIndex(
        positions, node_batch, batch_size=batch_size, radius=radius, position_dims=position_dims,
        chunk_size=chunk_size, candidate_pair_budget=candidate_pair_budget,
    )
    degree = node_batch.new_zeros(positions.shape[0])
    for _, destination, _ in index.iter_directed_neighbors():
        degree.index_add_(0, destination, torch.ones_like(destination))
    return ImplicitRadiusGraph.from_counted_nodes(
        node_features, positions, node_batch, degree, batch_size=batch_size, radius=radius,
        position_dims=position_dims, chunk_size=chunk_size,
        candidate_pair_budget=candidate_pair_budget, index=index, max_graph_edges=max_graph_edges,
    )
