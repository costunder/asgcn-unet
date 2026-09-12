"""Fixed-cell mean pooling and exact quotient remapping of existing raw edges.

Training builds a differentiable full snapshot. Inference keeps cluster sums and
quotient contributor counts, querying only added/removed incident raw edges on
the implicit backend. The materialized compatibility backend must scan its edge
array to select incident edges; that O(E) cost is explicitly counted in ``work``.
No radius graph is constructed from cluster centroids and no edge is clipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Real

import torch

from .graph import EventGraph
from .implicit_radius import ImplicitRadiusGraph
from .stream_graph import GraphUpdate, StreamGraph


@dataclass(frozen=True)
class PoolState:
    raw_graph: StreamGraph
    raw_features: torch.Tensor
    raw_to_cluster: torch.Tensor
    cluster_keys: torch.Tensor
    counts: torch.Tensor
    feature_sums: torch.Tensor
    position_sums: torch.Tensor
    timestamp_sums: torch.Tensor
    graph: StreamGraph
    edge_refcounts: torch.Tensor
    edge_pseudo_sums: torch.Tensor
    config: dict
    sensor_size: tuple[int, int]
    time_scale_seconds: float
    work: dict


def _configuration(config, sensor_size, time_scale_seconds):
    required = {"spatial_cell_pixels", "temporal_cell_seconds"}
    if not isinstance(config, dict) or not required <= config.keys() or config.keys() - required - {"edge_chunk_size"}:
        raise ValueError("Pool config requires explicit spatial_cell_pixels and temporal_cell_seconds")
    values = {}
    for name in required:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Pool {name} must be finite and positive")
        values[name] = float(value)
    chunk = config.get("edge_chunk_size", 65_536)
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("Pool edge_chunk_size must be a positive integer scratch bound")
    values["edge_chunk_size"] = chunk
    if (not isinstance(sensor_size, (tuple, list)) or len(sensor_size) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in sensor_size)):
        raise ValueError("Pool sensor_size must contain positive integer height and width")
    if (isinstance(time_scale_seconds, bool) or not isinstance(time_scale_seconds, Real)
            or not math.isfinite(time_scale_seconds) or time_scale_seconds <= 0):
        raise ValueError("Pool time_scale_seconds must be finite and positive")
    return values, tuple(sensor_size), float(time_scale_seconds)


def _validate_raw(raw, features):
    if not isinstance(raw, StreamGraph) or not isinstance(raw.graph, (EventGraph, ImplicitRadiusGraph)):
        raise TypeError("Pooling requires a raw StreamGraph with explicit materialized or implicit storage")
    graph, positions = raw.graph, raw.graph.positions
    count = positions.shape[0]
    if (not isinstance(features, torch.Tensor) or features.layout != torch.strided
            or features.ndim != 2 or features.shape[0] != count or features.shape[1] < 1
            or not features.is_floating_point() or features.device != positions.device):
        raise ValueError("Pool features must be a floating [N,C] tensor on the raw graph device")
    if (positions.shape != (count, 4) or not positions.is_floating_point()
            or raw.node_batch.shape != (count,) or raw.node_batch.dtype != torch.long
            or raw.timestamps.shape != (count,) or raw.timestamps.dtype != torch.float64
            or raw.node_batch.device != positions.device or raw.timestamps.device != positions.device):
        raise ValueError("Invalid raw pooling geometry, timestamps, or batch identifiers")
    checks = [torch.isfinite(features).all(), torch.isfinite(positions).all(),
              torch.isfinite(raw.timestamps).all(), (raw.node_batch >= 0).all()]
    if isinstance(graph, ImplicitRadiusGraph):
        graph.validate_integrity()
        checks.append((raw.node_batch == graph.node_batch).all())
    else:
        edges, pseudo = graph.edge_index, graph.edge_attr
        if (edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long
                or edges.device != positions.device or pseudo.shape != (edges.shape[1], 1)
                or pseudo.device != positions.device or not pseudo.is_floating_point()):
            raise ValueError("Invalid materialized raw edge shape/dtype/device")
        valid_ids = (edges >= 0) & (edges < count)
        if not bool(valid_ids.all()):
            raise ValueError("Raw pooling edge contains an invalid node index")
        checks.extend((torch.isfinite(pseudo).all(), ((pseudo >= 0) & (pseudo <= 1)).all(),
                       (raw.node_batch[edges[0]] == raw.node_batch[edges[1]]).all()))
    if not bool(torch.stack(checks).all()):
        raise ValueError("Raw pooling inputs must be finite, with bounded pseudo coordinates and disjoint streams")


def _keys(raw, nodes, config, sensor_size, time_scale_seconds):
    positions = raw.graph.positions[nodes, :3].double()
    height, width = sensor_size
    scales = positions.new_tensor((max(width - 1, 1) / config["spatial_cell_pixels"],
                                   max(height - 1, 1) / config["spatial_cell_pixels"],
                                   time_scale_seconds / config["temporal_cell_seconds"]))
    # t is already measured from each stream's FIXED origin. Never subtract a
    # moving window minimum or frame boundary when assigning cluster identities.
    scaled = positions * scales
    if not bool((torch.isfinite(scaled) & (scaled.abs() <= 2**48)).all()):
        raise ValueError("Pool cell addressing exceeds the exact float64 range; no cells were clipped")
    return torch.cat((raw.node_batch[nodes, None], torch.floor(scaled).to(torch.long)), dim=1)


def _new_work(raw):
    return {"raw_edges_visited": 0, "raw_query_nodes": 0, "materialized_edges_scanned": 0,
            "materialized_validation_edges": raw.graph.edge_index.shape[1] if isinstance(raw.graph, EventGraph) else 0,
            "feature_rows_updated": 0, "whole_raw_edge_reenumeration": False}


def _edge_chunks(raw, chunk_size, work, incident=None):
    """Visit each stored directed edge once, or only selected incident edges.

    Implicit radius graphs are symmetric. Selected incoming queries plus reverse
    directions whose source is not selected cover an incident edge exactly once.
    Arbitrary materialized graphs are never reinterpreted as radius graphs.
    """
    graph = raw.graph
    if incident is not None and not incident.numel():
        return
    if isinstance(graph, ImplicitRadiusGraph):
        if incident is not None:
            work["raw_query_nodes"] += incident.numel()
            selected = torch.zeros(len(raw.timestamps), dtype=torch.bool, device=raw.timestamps.device)
            selected[incident] = True
        for source, destination, pseudo in graph.iter_directed_neighbors(incident):
            for start in range(0, source.numel(), chunk_size):
                src, dst, value = source[start:start + chunk_size], destination[start:start + chunk_size], pseudo[start:start + chunk_size]
                work["raw_edges_visited"] += src.numel()
                yield src, dst, value
                if incident is not None:
                    reverse = ~selected[src]
                    reverse_source, reverse_destination = dst[reverse], src[reverse]
                    work["raw_edges_visited"] += reverse_source.numel()
                    yield reverse_source, reverse_destination, value[reverse]
    else:
        if incident is not None:
            selected = torch.zeros(len(raw.timestamps), dtype=torch.bool, device=raw.timestamps.device)
            selected[incident] = True
            work["raw_query_nodes"] += incident.numel()
        work["materialized_edges_scanned"] += graph.edge_index.shape[1]
        work["whole_raw_edge_reenumeration"] = True
        for start in range(0, graph.edge_index.shape[1], chunk_size):
            source, destination = graph.edge_index[:, start:start + chunk_size]
            pseudo = graph.edge_attr[start:start + chunk_size]
            if incident is not None:
                keep = selected[source] | selected[destination]
                source, destination, pseudo = source[keep], destination[keep], pseudo[keep]
            work["raw_edges_visited"] += source.numel()
            yield source, destination, pseudo


def _empty_edges(reference):
    return (reference.new_empty((0, 2), dtype=torch.long), reference.new_empty(0, dtype=torch.long),
            reference.new_empty((0, 1), dtype=torch.float64))


def _contributions(raw, raw_to_cluster, pairs, refcounts, sums, config, work, *, incident=None, sign=1):
    if incident is not None and not incident.numel():
        return pairs, refcounts, sums
    from .quotient_accumulator import QuotientAccumulator
    accumulator = QuotientAccumulator(pairs, refcounts, sums)
    for source, destination, pseudo in _edge_chunks(raw, config["edge_chunk_size"], work, incident):
        src, dst = raw_to_cluster[source], raw_to_cluster[destination]
        nonself = src != dst
        new_pairs = torch.stack((src[nonself], dst[nonself]), dim=1)
        new_counts = torch.full((len(new_pairs),), sign, dtype=torch.long, device=src.device)
        accumulator.add(new_pairs, new_counts, pseudo[nonself].double() * sign)
    return accumulator.finish(work)


def _finish(raw, features, raw_to_cluster, keys, counts, feature_sums, position_sums,
            timestamp_sums, pairs, refs, pseudo_sums, config, sensor_size, time_scale_seconds, work):
    if bool((counts <= 0).any()) or bool((refs <= 0).any()):
        raise ValueError("Pooled cluster/contributor counts must be positive; raw update must be append/expire consistent")
    means = (feature_sums / counts[:, None]).to(features.dtype)
    positions = (position_sums / counts[:, None]).to(raw.graph.positions.dtype)
    times = timestamp_sums / counts
    pseudo = pseudo_sums / refs[:, None]
    if not bool(torch.stack((torch.isfinite(means).all(), torch.isfinite(positions).all(),
                            torch.isfinite(times).all(), torch.isfinite(pseudo).all(),
                            ((pseudo >= 0) & (pseudo <= 1)).all())).all()):
        raise FloatingPointError("Nonfinite or out-of-range pooled mean; no edge pseudo coordinate was clipped")
    edges = pairs.t().contiguous()
    graph = EventGraph(means, positions, edges, pseudo.to(raw.graph.positions.dtype))
    stream = StreamGraph(graph, keys[:, 0], times)
    work.update({"pooled_nodes": len(keys), "pooled_edges": len(pairs)})
    work["topology_changed_nodes"] = torch.ones(len(keys), dtype=torch.bool, device=keys.device)
    work["input_changed_sources"] = torch.zeros(len(keys), dtype=torch.bool, device=keys.device)
    return PoolState(raw, features, raw_to_cluster, keys, counts, feature_sums, position_sums,
                     timestamp_sums, stream, refs, pseudo_sums, config, sensor_size, time_scale_seconds, work)


def pool_graph(raw_stream_graph, features, config, sensor_size, time_scale_seconds):
    """Fresh Eq.18 means and Eq.19 quotient snapshot, differentiable in features.

    Float64 sums reduce long-lived subtraction error; feature means are returned
    in the input feature dtype. Edge pseudo is the arithmetic mean of contributor
    raw distance/radius values, not a centroid radius query or recomputed distance.
    """
    config, sensor_size, time_scale_seconds = _configuration(config, sensor_size, time_scale_seconds)
    _validate_raw(raw_stream_graph, features)
    nodes = torch.arange(len(features), device=features.device)
    keys, inverse, counts = torch.unique(_keys(raw_stream_graph, nodes, config, sensor_size, time_scale_seconds),
                                         dim=0, sorted=True, return_inverse=True, return_counts=True)
    feature_sums = features.new_zeros((len(keys), features.shape[1]), dtype=torch.float64).index_add(0, inverse, features.double())
    position_sums = features.new_zeros((len(keys), 4), dtype=torch.float64).index_add(0, inverse, raw_stream_graph.graph.positions.double())
    timestamp_sums = features.new_zeros(len(keys), dtype=torch.float64).index_add(0, inverse, raw_stream_graph.timestamps)
    work = _new_work(raw_stream_graph)
    work["feature_rows_updated"] = len(features)
    pairs, refs, sums = _contributions(raw_stream_graph, inverse, *_empty_edges(features), config, work)
    return _finish(raw_stream_graph, features, inverse, keys, counts, feature_sums, position_sums,
                   timestamp_sums, pairs, refs, sums, config, sensor_size, time_scale_seconds, work)


def _active_mask(active_sources, count, device):
    mask = torch.zeros(count, dtype=torch.bool, device=device)
    if active_sources is None:
        return mask
    if not isinstance(active_sources, torch.Tensor) or active_sources.device != device or active_sources.ndim != 1:
        raise ValueError("active_sources must be a same-device node mask or unique long node IDs")
    if active_sources.dtype == torch.bool and active_sources.shape == (count,):
        return active_sources
    if active_sources.dtype != torch.long or not bool(((active_sources >= 0) & (active_sources < count)).all()):
        raise ValueError("active_sources contains invalid node IDs")
    if active_sources.unique().numel() != active_sources.numel():
        raise ValueError("active_sources node IDs must be unique")
    mask[active_sources] = True
    return mask


def _validate_materialized_retained_edges(previous, current, retained, retained_old):
    """Compatibility-only O(E) validation; arbitrary E is not a radius oracle."""
    old_graph, graph = previous.raw_graph.graph, current.graph
    mapping = retained_old.new_full((len(previous.raw_features),), -1)
    mapping[retained_old] = retained
    remapped = mapping[old_graph.edge_index]
    old_keep = (remapped >= 0).all(dim=0)
    is_old = torch.zeros(len(current.timestamps), dtype=torch.bool, device=retained.device)
    is_old[retained] = True
    new_keep = is_old[graph.edge_index].all(dim=0)
    old_edges, old_pseudo = remapped[:, old_keep], old_graph.edge_attr[old_keep]
    new_edges, new_pseudo = graph.edge_index[:, new_keep], graph.edge_attr[new_keep]
    if old_edges.shape != new_edges.shape:
        raise ValueError("Retained materialized raw edges changed; pooling requires append/expire updates")

    def ordered(edges, pseudo):
        order = torch.arange(edges.shape[1], device=edges.device)
        # Constant three-column lexicographic sort, never Python per-edge work.
        for value in (pseudo[:, 0], edges[1], edges[0]):
            order = order[torch.argsort(value[order], stable=True)]
        return edges[:, order], pseudo[order]

    old_edges, old_pseudo = ordered(old_edges, old_pseudo)
    new_edges, new_pseudo = ordered(new_edges, new_pseudo)
    if not torch.equal(old_edges, new_edges) or not torch.equal(old_pseudo, new_pseudo):
        raise ValueError("Retained materialized raw edge attributes changed; pooling requires append/expire updates")
    return old_edges.shape[1] + new_edges.shape[1]


def _changed_destinations(previous, state, old_cluster_indices, active_raw):
    topology_changed = old_cluster_indices < 0
    old_to_new = state.raw_to_cluster.new_full((len(previous.counts),), -1)
    retained = torch.nonzero(old_cluster_indices >= 0, as_tuple=True)[0]
    old_to_new[old_cluster_indices[retained]] = retained
    input_changed = torch.zeros_like(topology_changed)
    input_changed[retained] = (state.graph.graph.node_features[retained]
                               != previous.graph.graph.node_features[old_cluster_indices[retained]]).any(dim=1)
    input_changed[state.raw_to_cluster[active_raw]] = True
    topology_changed[retained] |= (state.graph.graph.positions[retained]
                                   != previous.graph.graph.positions[old_cluster_indices[retained]]).any(dim=1)
    # Compare actual quotient edge presence/mean, not just contributor counts.
    old_pairs = old_to_new[previous.graph.graph.edge_index].t()
    old_survives = (old_pairs >= 0).all(dim=1)
    removed_endpoints = old_pairs[~old_survives].flatten()
    topology_changed[removed_endpoints[removed_endpoints >= 0]] = True
    old_pairs = old_pairs[old_survives]
    new_pairs = state.graph.graph.edge_index.t()
    pairs, inverse = torch.unique(torch.cat((old_pairs, new_pairs)), dim=0, sorted=True, return_inverse=True)
    old_n = len(old_pairs)
    presence = state.counts.new_zeros((len(pairs), 2))
    presence[inverse[:old_n], 0] = 1
    presence[inverse[old_n:], 1] = 1
    values = state.position_sums.new_zeros((len(pairs), 2))
    values[inverse[:old_n], 0] = previous.graph.graph.edge_attr[old_survives, 0].double()
    values[inverse[old_n:], 1] = state.graph.graph.edge_attr[:, 0].double()
    edge_changed = (presence[:, 0] != presence[:, 1]) | (values[:, 0] != values[:, 1])
    topology_changed[pairs[edge_changed].flatten()] = True
    # These causes must remain separate for event-local SNN clocks: persistent
    # structural seeds are not interchangeable with a one-sweep input/pulse.
    state.work["topology_changed_nodes"] = topology_changed
    state.work["input_changed_sources"] = input_changed
    changed = topology_changed | input_changed
    # Features/pulses affect the root transform and all outgoing quotient edges.
    if new_pairs.numel():
        changed[new_pairs[input_changed[new_pairs[:, 0]], 1]] = True
    return changed


def update_pool(raw_update, features, previous_pool, config, sensor_size, time_scale_seconds, active_sources=None):
    """Exact append/expire pool update and downstream GraphUpdate seeds.

    Retained raw coordinates/topology must be unchanged; prefix features may
    change. ``active_sources`` identifies current raw pulse rows and forces a
    cluster/root/neighbor update even when repeated pulse values compare equal.
    ``state.work`` exposes separate same-device bool masks:
    ``topology_changed_nodes`` persists structural seeds; ``input_changed_sources``
    is a one-sweep feature/pulse cause. The returned GraphUpdate keeps the union
    with outgoing input dependants for compatibility; local-clock SNN consumers
    must use the separated masks instead of treating that union as topology.
    Previous state tensors are never modified. Fresh training must use pool_graph
    so no learned prefix from an earlier optimizer step is reused.
    """
    if not isinstance(raw_update, GraphUpdate):
        raise TypeError("update_pool requires the actual raw GraphUpdate")
    raw = raw_update.state
    config, sensor_size, time_scale_seconds = _configuration(config, sensor_size, time_scale_seconds)
    _validate_raw(raw, features)
    active = _active_mask(active_sources, len(features), features.device)
    if previous_pool is None:
        state = pool_graph(raw, features, config, sensor_size, time_scale_seconds)
        state.work["input_changed_sources"][state.raw_to_cluster[active]] = True
        old_indices = state.counts.new_full((len(state.counts),), -1)
        return state, GraphUpdate(state.graph, old_indices, torch.ones_like(old_indices, dtype=torch.bool))
    if not isinstance(previous_pool, PoolState):
        raise TypeError("previous_pool must be a PoolState or None")
    previous = previous_pool
    if (previous.config != config or previous.sensor_size != sensor_size
            or previous.time_scale_seconds != time_scale_seconds
            or previous.raw_features.shape[1:] != features.shape[1:]
            or previous.raw_features.dtype != features.dtype or previous.raw_features.device != features.device
            or type(previous.raw_graph.graph) is not type(raw.graph)
            or previous.raw_graph.graph.positions.dtype != raw.graph.positions.dtype):
        raise ValueError("Pool configuration, raw representation, or feature contract changed")
    if isinstance(raw.graph, ImplicitRadiusGraph) and (
            raw.graph.radius != previous.raw_graph.graph.radius
            or raw.graph.position_dims != previous.raw_graph.graph.position_dims):
        raise ValueError("Raw implicit radius geometry contract changed during pooling")
    old = raw_update.old_indices
    old_count = len(previous.raw_features)
    if (old.shape != (len(features),) or old.dtype != torch.long or old.device != features.device
            or not bool(((old >= -1) & (old < old_count)).all())):
        raise ValueError("Invalid raw old_indices mapping for pool update")
    retained = torch.nonzero(old >= 0, as_tuple=True)[0]
    arriving = torch.nonzero(old < 0, as_tuple=True)[0]
    retained_old = old[retained]
    if retained_old.unique().numel() != retained_old.numel():
        raise ValueError("A retained raw node was duplicated in the pool update")
    if not bool(torch.stack((
        (raw.graph.positions[retained] == previous.raw_graph.graph.positions[retained_old]).all(),
        (raw.timestamps[retained] == previous.raw_graph.timestamps[retained_old]).all(),
        (raw.node_batch[retained] == previous.raw_graph.node_batch[retained_old]).all(),
    )).all()):
        raise ValueError("Retained raw geometry or stream identity changed; pool update is not append/expire")
    if raw.graph is previous.raw_graph.graph and torch.equal(old, torch.arange(len(features), device=old.device)):
        # Local SNN sweeps change pulses, not topology. Keep the exact existing
        # quotient and cell IDs rather than sorting/re-hashing Q for each tick.
        changed_rows = torch.nonzero(features.ne(previous.raw_features).any(dim=1), as_tuple=True)[0]
        feature_sums = previous.feature_sums.index_add(
            0, previous.raw_to_cluster[changed_rows],
            features[changed_rows].double() - previous.raw_features[changed_rows].double())
        means = (feature_sums / previous.counts[:, None]).to(features.dtype)
        coarse = previous.graph.graph
        graph = StreamGraph(EventGraph(means, coarse.positions, coarse.edge_index, coarse.edge_attr, coarse.in_degree),
                            previous.graph.node_batch, previous.graph.timestamps)
        topology = torch.zeros(len(previous.counts), device=old.device, dtype=torch.bool)
        input_changed = means.ne(coarse.node_features).any(dim=1)
        input_changed[previous.raw_to_cluster[active]] = True
        changed = input_changed.clone()
        source, destination = coarse.edge_index
        changed[destination[input_changed[source]]] = True
        work = _new_work(raw)
        work.update(feature_rows_updated=len(changed_rows), reused_quotient_topology=True,
                    pooled_nodes=len(previous.counts), pooled_edges=coarse.edge_index.shape[1],
                    topology_changed_nodes=topology, input_changed_sources=input_changed)
        state = replace(previous, raw_graph=raw, raw_features=features, feature_sums=feature_sums, graph=graph, work=work)
        return state, GraphUpdate(graph, torch.arange(len(previous.counts), device=old.device), changed)
    checked_edges = (_validate_materialized_retained_edges(previous, raw, retained, retained_old)
                     if isinstance(raw.graph, EventGraph) else 0)
    removed_mask = torch.ones(old_count, dtype=torch.bool, device=features.device)
    removed_mask[retained_old] = False
    removed = torch.nonzero(removed_mask, as_tuple=True)[0]
    keys, union_inverse = torch.unique(torch.cat((previous.cluster_keys,
        _keys(raw, arriving, config, sensor_size, time_scale_seconds))), dim=0, sorted=True, return_inverse=True)
    old_clusters = union_inverse[:len(previous.counts)]
    arrival_clusters = union_inverse[len(previous.counts):]
    raw_clusters = old.new_empty(len(features))
    raw_clusters[retained] = old_clusters[previous.raw_to_cluster[retained_old]]
    raw_clusters[arriving] = arrival_clusters
    previous_raw_clusters = old_clusters[previous.raw_to_cluster]
    counts = old.new_zeros(len(keys)).index_add(0, old_clusters, previous.counts)
    counts.index_add_(0, previous_raw_clusters[removed], -torch.ones_like(removed))
    counts.index_add_(0, arrival_clusters, torch.ones_like(arriving))
    feature_sums = previous.feature_sums.new_zeros((len(keys), features.shape[1])).index_add(0, old_clusters, previous.feature_sums)
    position_sums = previous.position_sums.new_zeros((len(keys), 4)).index_add(0, old_clusters, previous.position_sums)
    timestamp_sums = previous.timestamp_sums.new_zeros(len(keys)).index_add(0, old_clusters, previous.timestamp_sums)
    feature_sums = feature_sums.index_add(0, previous_raw_clusters[removed], -previous.raw_features[removed].double())
    feature_sums = feature_sums.index_add(0, arrival_clusters, features[arriving].double())
    changed_rows = (features[retained] != previous.raw_features[retained_old]).any(dim=1)
    changed_raw = retained[changed_rows]
    feature_sums = feature_sums.index_add(0, raw_clusters[changed_raw],
                                         features[changed_raw].double() - previous.raw_features[old[changed_raw]].double())
    position_sums = position_sums.index_add(0, previous_raw_clusters[removed], -previous.raw_graph.graph.positions[removed].double())
    position_sums = position_sums.index_add(0, arrival_clusters, raw.graph.positions[arriving].double())
    timestamp_sums = timestamp_sums.index_add(0, previous_raw_clusters[removed], -previous.raw_graph.timestamps[removed])
    timestamp_sums = timestamp_sums.index_add(0, arrival_clusters, raw.timestamps[arriving])
    work = _new_work(raw)
    work["materialized_retained_edges_checked"] = checked_edges
    work["feature_rows_updated"] = len(removed) + len(arriving) + len(changed_raw)
    pairs = old_clusters[previous.graph.graph.edge_index].t()
    pairs, refs, sums = _contributions(previous.raw_graph, previous_raw_clusters, pairs,
        previous.edge_refcounts, previous.edge_pseudo_sums, config, work, incident=removed, sign=-1)
    pairs, refs, sums = _contributions(raw, raw_clusters, pairs, refs, sums, config, work, incident=arriving)
    alive = counts > 0
    if bool((counts < 0).any()) or (pairs.numel() and not bool(alive[pairs].all())):
        raise ValueError("Pool empty-cluster/edge reference counts are inconsistent")
    remap = old.new_full((len(keys),), -1)
    alive_indices = torch.nonzero(alive, as_tuple=True)[0]
    remap[alive_indices] = torch.arange(alive_indices.numel(), device=features.device)
    old_by_union = old.new_full((len(keys),), -1)
    old_by_union[old_clusters] = torch.arange(len(previous.counts), device=features.device)
    state = _finish(raw, features, remap[raw_clusters], keys[alive], counts[alive], feature_sums[alive],
                    position_sums[alive], timestamp_sums[alive], remap[pairs], refs, sums,
                    config, sensor_size, time_scale_seconds, work)
    old_cluster_indices = old_by_union[alive]
    changed = _changed_destinations(previous, state, old_cluster_indices, active)
    return state, GraphUpdate(state.graph, old_cluster_indices, changed)
