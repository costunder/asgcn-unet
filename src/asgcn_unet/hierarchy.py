"""ASGCN Eq18/19 in the trained and event-local inference paths.

Clustering parameters/placement are explicit reconstruction design choices,
not undisclosed author hyperparameters. No weights are copied or bypassed.
"""

from dataclasses import dataclass, fields, replace
from types import SimpleNamespace

import torch

from .encoder_stage import EncoderStage
from .hierarchy_contract import (
    validate_hierarchy_config,  # noqa: F401 - public compatibility re-export
)
from .stream_encoder import update_encoder
from .stream_graph import GraphUpdate
from .stream_state import map_cache, map_graph


def pool_config(model):
    return {key: model.hierarchy_config[key] for key in ("spatial_cell_pixels", "temporal_cell_seconds")}


def stages(model):
    boundary = model.hierarchy_config["after_layer"]
    return (EncoderStage(model.encoder, 0, boundary),
            EncoderStage(model.encoder, boundary, len(model.encoder.layers), input_is_spiking=True))


def forward_snapshot(model, graph, sensor_size, *, calibration=False):
    from .graph_pool import pool_graph
    prefix, suffix = stages(model)
    hidden, before = prefix.forward_ann(graph.graph, calibration)
    pool = pool_graph(graph, hidden, pool_config(model), sensor_size=sensor_size,
                      time_scale_seconds=model.stream_config["time_scale_seconds"])
    output, after = suffix.forward_ann(pool.graph.graph, calibration)
    return output, before + after, pool.graph


@dataclass(frozen=True)
class HierarchyState:
    pool: object
    suffix: object

    def map(self, fn, raw_graph):
        coarse = map_graph(self.pool.graph, fn)
        values = {field.name: fn(getattr(self.pool, field.name))
                  for field in fields(self.pool) if isinstance(getattr(self.pool, field.name), torch.Tensor)}
        pool = replace(self.pool, raw_graph=raw_graph, graph=coarse, work={}, **values)
        return HierarchyState(pool, map_cache(self.suffix, coarse, fn))

    def tensors(self):
        for field in fields(self.pool):
            value = getattr(self.pool, field.name)
            if isinstance(value, torch.Tensor):
                yield value
        yield self.pool.graph.graph.node_features
        yield self.pool.graph.graph.positions
        yield self.pool.graph.graph.edge_attr
        if self.suffix is not None:
            for field in fields(self.suffix):
                value = getattr(self.suffix, field.name)
                if isinstance(value, torch.Tensor):
                    yield value
                elif isinstance(value, tuple):
                    yield from (item for item in value if isinstance(item, torch.Tensor))


def update_hierarchy(model, update, prefix_cache, previous, sensor_size, *, mode, simulation_steps,
                     active_graphs=None):
    from .graph_pool import update_pool
    prefix, suffix = stages(model)
    pool = None if previous is None else previous.pool
    suffix_cache = None if previous is None else previous.suffix
    work = []
    coarse_seed = None
    # Dependent local ticks must be interleaved across the pool, not executed
    # as all prefix ticks followed by pooled cumulative rates and suffix ticks.
    for tick in range(simulation_steps if mode == "snn" else 1):
        raw_update = update if tick == 0 else GraphUpdate(
            update.state, torch.arange(len(update.old_indices), device=update.old_indices.device), update.changed_nodes)
        prefix_cache = update_encoder(prefix, raw_update, prefix_cache, mode=mode, simulation_steps=1,
                                      dynamics=model.snn_dynamics, active_graphs=active_graphs)
        features = prefix_cache.last_pulses[-1] if mode == "snn" else prefix_cache.outputs
        active_sources = None
        if mode == "snn":
            active_sources = features.ne(0).any(dim=1)
            if active_graphs is not None:
                active_sources = active_sources & active_graphs[update.state.node_batch]
        pool, coarse_update = update_pool(
            raw_update, features, pool, pool_config(model), sensor_size=sensor_size,
            time_scale_seconds=model.stream_config["time_scale_seconds"], active_sources=active_sources)
        if tick == 0:
            coarse_seed = pool.work["topology_changed_nodes"]
        coarse_update = replace(coarse_update, changed_nodes=pool.work["topology_changed_nodes"] | coarse_seed)
        suffix_cache = update_encoder(suffix, coarse_update, suffix_cache, mode=mode, simulation_steps=1,
                                      dynamics=model.snn_dynamics, active_graphs=active_graphs,
                                      input_changed_sources=pool.work["input_changed_sources"])
        pool_work = {key: value for key, value in pool.work.items() if not isinstance(value, torch.Tensor)}
        work.append(({**prefix_cache.work, "pooling": pool_work}, suffix_cache.work))
    return prefix_cache, HierarchyState(pool, suffix_cache), work


def pack_hierarchy(model, states, raw_graph, sensor_size):
    from .graph_pool import pool_graph
    from .stream_model import _empty, _pack_previous
    existing = [state.hierarchy if state is not None else None for state in states]
    if not any(item is not None for item in existing):
        return None
    if any(state is not None and len(state.graph.timestamps) and state.hierarchy is None for state in states):
        raise ValueError("Training/raw-only state cannot be reused as hierarchical inference state")
    pools = []
    for item in existing:
        pools.append(item.pool if item is not None else pool_graph(
            _empty(raw_graph.timestamps.device, model),
            raw_graph.graph.node_features.new_empty((0, model.encoder.hidden_dim)), pool_config(model),
            sensor_size=sensor_size, time_scale_seconds=model.stream_config["time_scale_seconds"]))
    coarse, suffix = _pack_previous([
        SimpleNamespace(graph=pool.graph, encoder=None if item is None else item.suffix)
        for pool, item in zip(pools, existing)], raw_graph.timestamps.device)
    values = {}
    for key in ("raw_features", "counts", "feature_sums", "position_sums", "timestamp_sums",
                "edge_refcounts", "edge_pseudo_sums"):
        values[key] = torch.cat([getattr(pool, key) for pool in pools])
    keys, assignments, offset = [], [], 0
    for lane, pool in enumerate(pools):
        key = pool.cluster_keys.clone()
        key[:, 0] = lane
        keys.append(key)
        assignments.append(pool.raw_to_cluster + offset)
        offset += len(pool.counts)
    values.update(cluster_keys=torch.cat(keys), raw_to_cluster=torch.cat(assignments))
    return HierarchyState(replace(pools[0], raw_graph=raw_graph, graph=coarse, work={}, **values), suffix)


def split_hierarchy(value, raw_lanes, batch_size):
    from .stream_model import _split_state
    coarse_lanes = _split_state(value.pool.graph, value.suffix, batch_size)
    result = []
    pool = value.pool
    for lane, ((raw, _prefix), (coarse, suffix)) in enumerate(zip(raw_lanes, coarse_lanes)):
        raw_nodes = torch.nonzero(pool.raw_graph.node_batch == lane, as_tuple=True)[0]
        nodes = torch.nonzero(pool.graph.node_batch == lane, as_tuple=True)[0]
        edge_ids = torch.nonzero(pool.graph.node_batch[pool.graph.graph.edge_index[0]] == lane, as_tuple=True)[0]
        inverse = torch.full_like(pool.counts, -1)
        inverse[nodes] = torch.arange(len(nodes), device=nodes.device)
        values = {key: getattr(pool, key)[nodes] for key in
                  ("counts", "feature_sums", "position_sums", "timestamp_sums")}
        values.update({key: getattr(pool, key)[edge_ids] for key in ("edge_refcounts", "edge_pseudo_sums")})
        keys = pool.cluster_keys[nodes].clone()
        keys[:, 0] = 0
        result.append(HierarchyState(replace(
            pool, raw_graph=raw, graph=coarse, raw_features=pool.raw_features[raw_nodes],
            raw_to_cluster=inverse[pool.raw_to_cluster[raw_nodes]], cluster_keys=keys, work={}, **values), suffix))
    return result
