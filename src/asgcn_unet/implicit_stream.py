"""Append/expire exact implicit radius graphs without a materialized edge list."""

from __future__ import annotations

import torch

from .implicit_radius import ImplicitRadiusGraph, ImplicitRadiusIndex


def evolve(previous, features, positions, timestamps, node_batch, cutoffs, *,
           radius, position_dims, max_graph_edges, chunk_size):
    from .stream_graph import GraphUpdate, StreamGraph, _check_guard, _validate_nodes

    if previous is not None and not isinstance(previous.graph, ImplicitRadiusGraph):
        raise ValueError("Cannot change graph storage inside an existing stream")
    arriving = torch.nonzero(timestamps >= cutoffs[node_batch], as_tuple=True)[0]
    if previous is None:
        retained = arriving.new_empty(0)
        removed = arriving.new_empty(0)
        old_count = 0
    else:
        _validate_nodes(previous.graph.node_features, previous.graph.positions,
                        previous.timestamps, previous.node_batch, cutoffs)
        old_graph = previous.graph
        if (old_graph.radius != radius or old_graph.position_dims != position_dims
                or old_graph.node_features.dtype != features.dtype
                or old_graph.positions.dtype != positions.dtype):
            raise ValueError("Implicit stream geometry/dtype changed")
        old_count = len(previous.timestamps)
        survives = previous.timestamps >= cutoffs[previous.node_batch]
        retained = torch.nonzero(survives, as_tuple=True)[0]
        removed = torch.nonzero(~survives, as_tuple=True)[0]
    def join(name, incoming):
        values = [incoming[arriving]]
        if previous is not None:
            old = getattr(previous, name) if name in {"timestamps", "node_batch"} else getattr(previous.graph, name)
            values.insert(0, old[retained])
        return torch.cat(values)
    new_features = join("node_features", features)
    new_positions = join("positions", positions)
    new_times = join("timestamps", timestamps)
    new_batch = join("node_batch", node_batch)
    n, retained_count = len(new_times), len(retained)
    old_indices = torch.cat((retained, arriving.new_full((len(arriving),), -1)))
    changed = torch.arange(n, device=features.device) >= retained_count
    old_to_new = arriving.new_full((old_count,), -1)
    old_to_new[retained] = torch.arange(retained_count, device=features.device)
    degree = arriving.new_zeros(n)
    if previous is not None:
        degree[:retained_count] = previous.graph.in_degree[retained]
        # Query only removed destinations. Each surviving source loses exactly
        # one incoming neighbor per returned (source, removed) pair.
        for source, _destination, _distance in previous.graph.iter_directed_neighbors(removed):
            surviving_source = old_to_new[source]
            surviving_source = surviving_source[surviving_source >= 0]
            degree.index_add_(0, surviving_source, -torch.ones_like(surviving_source))
            changed[surviving_source] = True
    index = None
    if len(arriving):
        index = ImplicitRadiusIndex(new_positions, new_batch, batch_size=len(cutoffs),
                                    radius=radius, position_dims=position_dims, chunk_size=chunk_size)
        destinations = torch.arange(retained_count, n, device=features.device)
        for source, destination, _distance in index.iter_directed_neighbors(destinations):
            degree.index_add_(0, destination, torch.ones_like(destination))
            # Both directions between two arrivals are already enumerated by
            # the destination query. Only old sources need the reverse update.
            old_sources = source[source < retained_count]
            degree.index_add_(0, old_sources, torch.ones_like(old_sources))
            changed[old_sources] = True
    edge_counts = degree.new_zeros(len(cutoffs))
    edge_counts.index_add_(0, new_batch, degree)
    _check_guard(edge_counts, max_graph_edges)
    graph = ImplicitRadiusGraph.from_counted_nodes(
        new_features, new_positions, new_batch, degree, batch_size=len(cutoffs),
        radius=radius, position_dims=position_dims, chunk_size=chunk_size,
        edge_counts=edge_counts, index=index,
    )
    return GraphUpdate(StreamGraph(graph, new_batch, new_times), old_indices, changed)


class ImplicitIncidence:
    """Bounded incident-neighbor queries; no E-sized sort or CSR expansion."""

    def __init__(self, graph):
        self.graph = graph

    def dependants(self, graph, sources):
        selected = torch.zeros(len(graph.node_features), device=sources.device, dtype=torch.bool)
        selected[sources] = True
        # The graph is symmetric: querying incoming neighbors of changed sources
        # returns exactly their outgoing dependants, without constructing E.
        for neighbor, _destination, _distance in graph.iter_directed_neighbors(sources):
            selected[neighbor] = True
        return torch.nonzero(selected, as_tuple=True)[0]


def split_state(graph, cache, batch_size):
    """One packed node permutation followed by lane-owned views and metadata."""
    from dataclasses import fields, replace

    from .stream_graph import StreamGraph

    order = torch.argsort(graph.node_batch, stable=True)
    node_counts = torch.bincount(graph.node_batch, minlength=batch_size)
    counts = torch.stack((node_counts, graph.graph.edge_counts)).cpu().tolist()
    features = graph.graph.node_features[order]
    positions = graph.graph.positions[order]
    degrees = graph.graph.in_degree[order]
    timestamps = graph.timestamps[order]
    cached = {}
    if cache is not None:
        for item in fields(cache):
            value = getattr(cache, item.name)
            if isinstance(value, torch.Tensor):
                cached[item.name] = value[order]
            elif isinstance(value, tuple):
                cached[item.name] = tuple(tensor[order] for tensor in value)
    result, start = [], 0
    for lane_id, (count, _edge_count) in enumerate(zip(*counts, strict=True)):
        stop = start + count
        batch = graph.node_batch.new_zeros(count)
        raw = ImplicitRadiusGraph.from_counted_nodes(
            features[start:stop], positions[start:stop], batch, degrees[start:stop],
            batch_size=1, radius=graph.graph.radius, position_dims=graph.graph.position_dims,
            chunk_size=graph.graph.chunk_size, candidate_pair_budget=graph.graph.candidate_pair_budget,
            edge_counts=graph.graph.edge_counts[lane_id:lane_id + 1],
        )
        lane = StreamGraph(raw, batch, timestamps[start:stop])
        lane_cache = None
        if cache is not None:
            values = {key: value[start:stop] if isinstance(value, torch.Tensor)
                      else tuple(tensor[start:stop] for tensor in value)
                      for key, value in cached.items()}
            lane_cache = replace(cache, graph=lane, work={}, **values)
        result.append((lane, lane_cache))
        start = stop
    return result
