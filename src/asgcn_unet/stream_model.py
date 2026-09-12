"""Stateful event-arrival ASGCN reconstruction, sharing the trained spline layers.

ANN learning recomputes the current causal window with pooled-node BN. Inference
uses event-local incremental updates; it is not static rate-T equivalence. Frame
readout invokes the analog recurrent decoder once, not once per sensor event.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from dataclasses import fields, replace

import torch

from .batching import pack_samples, sequence_key
from .graph import EventGraph
from .stream_encoder import update_encoder
from .stream_graph import StreamGraph, evolve_stream_graph
from .stream_state import StreamingReconstructionState


def validate_stream_config(value):
    expected = {"window_seconds", "time_scale_seconds", "node_time_feature", "clock", "arrival_policy"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("stream_config requires explicit window_seconds, time_scale_seconds, "
                         "node_time_feature, clock and arrival_policy")
    for name in ("window_seconds", "time_scale_seconds"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0:
            raise ValueError(f"stream_config.{name} must be an explicit finite positive number")
    if value["node_time_feature"] != "physical_frame_offset":
        raise ValueError("Only the declared causal physical_frame_offset feature is supported")
    if value["clock"] != "event_local_pending_off_v1":
        raise ValueError("The streaming IF update clock must be explicitly acknowledged")
    if value["arrival_policy"] != "simultaneous_equal_timestamp":
        raise ValueError("Arrival grouping must preserve every distinct timestamp")
    return dict(value)


def stream_contract(model):
    # This is a structural contract, not a substitute for checkpoint weight hashes.
    values = {"version": getattr(model, "architecture_version", 3), "stream": model.stream_config, "radius": model.graph_radius,
              "position_dims": model.graph_position_dims, "sampling": model.event_sampling_factor,
              "width": model.encoder.hidden_dim, "depth": len(model.encoder.layers),
              "dynamics": model.snn_dynamics, "decoder": model.decoder_kind,
              "raster_downsample": model.raster_downsample}
    storage = getattr(model, "graph_storage", "materialized")
    if storage != "materialized":
        values["graph_storage"] = storage
    if getattr(model, "hierarchy_config", None) is not None:
        values["hierarchy"] = model.hierarchy_config
        values["sampling_phase"] = "persistent_sequence_ordinal"
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _empty(device, model=None):
    if getattr(model, "graph_storage", "materialized") == "implicit_radius":
        from .implicit_radius import ImplicitRadiusGraph
        batch = torch.empty(0, device=device, dtype=torch.long)
        graph = ImplicitRadiusGraph.from_counted_nodes(
            torch.empty((0, 4), device=device), torch.empty((0, 4), device=device, dtype=torch.float64),
            batch, batch, batch_size=1, radius=model.graph_radius,
            position_dims=model.graph_position_dims, chunk_size=model.graph_chunk_size,
        )
        return StreamGraph(graph, batch, torch.empty(0, device=device, dtype=torch.float64))
    return StreamGraph(EventGraph(
        torch.empty((0, 4), device=device), torch.empty((0, 4), device=device, dtype=torch.float64),
        torch.empty((2, 0), device=device, dtype=torch.long), torch.empty((0, 1), device=device, dtype=torch.float64),
    ), torch.empty(0, device=device, dtype=torch.long),
        torch.empty(0, device=device, dtype=torch.float64))


def _pack_previous(states, device, model=None):
    graphs = [state.graph if state is not None else _empty(device, model) for state in states]
    counts = [len(value.timestamps) for value in graphs]
    offsets, cursor = [], 0
    for count in counts:
        offsets.append(cursor)
        cursor += count
    from .implicit_radius import ImplicitRadiusGraph
    implicit = isinstance(graphs[0].graph, ImplicitRadiusGraph)
    if any(isinstance(value.graph, ImplicitRadiusGraph) != implicit for value in graphs):
        raise ValueError("Cannot pack streams with different graph storage")
    packed_batch = torch.cat([torch.full((count,), lane, device=device, dtype=torch.long)
                              for lane, count in enumerate(counts)])
    if implicit:
        reference_graph = graphs[0].graph
        raw = ImplicitRadiusGraph.from_counted_nodes(
            torch.cat([value.graph.node_features for value in graphs]),
            torch.cat([value.graph.positions for value in graphs]), packed_batch,
            torch.cat([value.graph.in_degree for value in graphs]), batch_size=len(states),
            radius=reference_graph.radius, position_dims=reference_graph.position_dims,
            chunk_size=reference_graph.chunk_size, candidate_pair_budget=reference_graph.candidate_pair_budget,
            edge_counts=torch.cat([value.graph.edge_counts for value in graphs]),
        )
        graph = StreamGraph(raw, packed_batch, torch.cat([value.timestamps for value in graphs]))
    else:
        graph = StreamGraph(EventGraph(
            torch.cat([value.graph.node_features for value in graphs]),
            torch.cat([value.graph.positions for value in graphs]),
            torch.cat([value.graph.edge_index + offset for value, offset in zip(graphs, offsets)], dim=1),
            torch.cat([value.graph.edge_attr for value in graphs]),
            torch.cat([value.graph.in_degree for value in graphs]),
        ), packed_batch, torch.cat([value.timestamps for value in graphs]))
    present = [state.encoder for state in states if state is not None and state.encoder is not None]
    if not present:
        return graph, None
    reference = present[0]
    if any(cache.mode != reference.mode or cache.dynamics != reference.dynamics for cache in present):
        raise ValueError("Cannot batch different streaming encoder modes")
    if any(state is not None and count and state.encoder is None
           for state, count in zip(states, counts)):
        raise ValueError("Training graph state cannot be reused as calibrated inference state")
    values = {"graph": graph, "work": {}}
    for field in fields(reference):
        if field.name in values:
            continue
        source = getattr(reference, field.name)
        if isinstance(source, torch.Tensor):
            values[field.name] = torch.cat([
                getattr(state.encoder, field.name) if state is not None and state.encoder is not None
                else source.new_empty((0, *source.shape[1:])) for state in states])
        elif isinstance(source, tuple):
            values[field.name] = tuple(torch.cat([
                getattr(state.encoder, field.name)[index]
                if state is not None and state.encoder is not None else item.new_empty((0, *item.shape[1:]))
                for state in states]) for index, item in enumerate(source))
        else:
            values[field.name] = source
    return graph, replace(reference, **values)


def _split_state(graph, cache, batch_size):
    """One batched node/edge permutation, then per-lane state views only."""
    from .implicit_radius import ImplicitRadiusGraph
    if isinstance(graph.graph, ImplicitRadiusGraph):
        from .implicit_stream import split_state
        return split_state(graph, cache, batch_size)
    node_order = torch.argsort(graph.node_batch, stable=True)
    inverse = torch.empty_like(node_order)
    inverse[node_order] = torch.arange(node_order.numel(), device=node_order.device)
    remapped_edges = inverse[graph.graph.edge_index]
    edge_batch = graph.node_batch[graph.graph.edge_index[0]]
    edge_order = torch.argsort(edge_batch, stable=True)
    node_counts = torch.bincount(graph.node_batch, minlength=batch_size)
    edge_counts = torch.bincount(edge_batch, minlength=batch_size)
    # One device->host transfer for boundary metadata, not a synchronization per node/event.
    counts = torch.stack((node_counts, edge_counts)).cpu().tolist()
    node_values = [graph.graph.node_features[node_order], graph.graph.positions[node_order],
                   graph.graph.in_degree[node_order], graph.timestamps[node_order]]
    edge_values = [remapped_edges[:, edge_order], graph.graph.edge_attr[edge_order]]
    cached = {}
    if cache is not None:
        for field in fields(cache):
            value = getattr(cache, field.name)
            if isinstance(value, torch.Tensor):
                cached[field.name] = value[node_order]
            elif isinstance(value, tuple):
                cached[field.name] = tuple(item[node_order] for item in value)
    result, start, edge_start = [], 0, 0
    for count, edge_count in zip(*counts):
        stop, edge_stop = start + count, edge_start + edge_count
        lane = StreamGraph(EventGraph(
            node_values[0][start:stop], node_values[1][start:stop],
            edge_values[0][:, edge_start:edge_stop] - start,
            edge_values[1][edge_start:edge_stop], node_values[2][start:stop],
        ), graph.node_batch.new_zeros(count), node_values[3][start:stop])
        lane_cache = None
        if cache is not None:
            values = {name: (value[start:stop] if isinstance(value, torch.Tensor)
                             else tuple(item[start:stop] for item in value))
                      for name, value in cached.items()}
            # Work counters describe this whole physical batch and are consumed
            # before splitting. Do not retain another lane's counter tensors in
            # a persistent per-stream state (including across device moves).
            lane_cache = replace(cache, graph=lane, work={}, **values)
        result.append((lane, lane_cache))
        start, edge_start = stop, edge_stop
    return result


def _metadata(model, samples, states):
    contract = stream_contract(model)
    records = []
    for sample, previous in zip(samples, states):
        metadata = sample.get("metadata", {})
        timing = metadata.get("stream_time")
        if not isinstance(timing, dict) or timing.get("schema") != "physical_seconds_v1":
            raise ValueError("Event-driven ASGCN requires physical_seconds_v1 input, not normalized frames")
        if sample["events"].dtype != torch.float64 or "event_ids" not in sample:
            raise ValueError("Streaming events require float64 physical seconds and stable event_ids")
        identity = sequence_key(sample)
        index = metadata.get("sequence_index")
        if type(index) is not int or index < 0:
            raise ValueError("Streaming samples require chronological sequence_index")
        keys = ("interval_start_seconds", "interval_end_seconds", "sequence_origin_seconds")
        for name in keys:
            item = timing.get(name)
            if isinstance(item, bool) or not isinstance(item, (float, int)) or not math.isfinite(item):
                raise ValueError(f"Missing or invalid physical stream clock: {name}")
        t0, t1, origin = (float(timing[name]) for name in keys)
        if t1 < t0:
            raise ValueError("Streaming readout interval is reversed")
        if origin > t0:
            raise ValueError("Streaming origin must be known by the interval start")
        groups = timing.get("arrival_group_counts")
        if (not isinstance(groups, (list, tuple))
                or any(type(item) is not int or item < 1 for item in groups)
                or sum(groups) != len(sample["events"])):
            raise ValueError("Streaming arrival_group_counts must cover every input event exactly once")
        if previous is not None:
            if not isinstance(previous, StreamingReconstructionState):
                raise TypeError("Static decoder state cannot be reused for event-driven ASGCN")
            if (previous.contract != contract or previous.sequence_identity != identity
                    or previous.sequence_index + 1 != index or previous.origin_seconds != origin
                    or previous.watermark_seconds > t1):
                raise ValueError("Streaming state/config/clock/sequence continuity mismatch")
        records.append((identity, index, t0, t1, origin, tuple(groups)))
    if len({record[0] for record in records}) != len(records):
        raise ValueError("Causally dependent frames of one stream cannot share a physical batch")
    return records, contract


def _prepared(model, packed, records):
    device = packed.events.device
    counts = torch.tensor(packed.event_counts, device=device)
    node_batch = torch.repeat_interleave(torch.arange(len(packed), device=device), counts)
    height, width = packed.sensor_size
    events = packed.events
    if not bool(torch.stack((torch.isfinite(events).all(),
                            ((events[:, 0] >= 0) & (events[:, 0] < width)).all(),
                            ((events[:, 1] >= 0) & (events[:, 1] < height)).all(),
                            ((events[:, 3] == 1) | (events[:, 3] == -1)).all())).all()):
        raise ValueError("Invalid physical sensor event values")
    # Validate the claimed arrival grouping once on the whole physical batch.
    # Metadata only determines scheduling; it cannot authorize coalescing two
    # distinct timestamps or silently reversing their order.
    groups = []
    group_offset = 0
    for record in records:
        for index, count in enumerate(record[5]):
            groups.extend([group_offset + index] * count)
        group_offset += len(record[5])
    group_ids = torch.tensor(groups, device=device, dtype=torch.long)
    if len(events) > 1:
        same_stream = node_batch[1:] == node_batch[:-1]
        delta = events[1:, 2] - events[:-1, 2]
        same_group = group_ids[1:] == group_ids[:-1]
        ids = packed.event_ids
        increasing_id = (ids[1:, 0] > ids[:-1, 0]) | (
            (ids[1:, 0] == ids[:-1, 0]) & (ids[1:, 1] > ids[:-1, 1]))
        if not bool(((~same_stream) | ((delta >= 0) & (same_group == (delta == 0)) & increasing_id)).all()):
            raise ValueError("Physical event order, identity or equal-timestamp grouping is invalid")
    x, y = events[:, 0] / max(width - 1, 1), events[:, 1] / max(height - 1, 1)
    origin = events.new_tensor([record[4] for record in records])[node_batch]
    interval_start = events.new_tensor([record[2] for record in records])[node_batch]
    interval_end = events.new_tensor([record[3] for record in records])[node_batch]
    if not bool(((events[:, 2] >= origin) & (events[:, 2] <= interval_end)).all()):
        raise ValueError("Physical event timestamp is before its origin or after its readout (future leakage)")
    scale = model.stream_config["time_scale_seconds"]
    polarity = torch.where(events[:, 3] > 0, 1.0, -1.0)
    features = torch.stack((x, y, (events[:, 2] - interval_start) / scale, polarity), dim=1).float()
    positions = torch.stack((x, y, (events[:, 2] - origin) / scale, (polarity + 1) / 2), dim=1)
    return features, positions, events[:, 2], node_batch


def _update(model, previous, features, positions, timestamps, node_batch, cutoffs):
    return evolve_stream_graph(previous, features, positions, timestamps, node_batch, cutoffs,
                               radius=model.graph_radius, position_dims=model.graph_position_dims,
                               max_graph_edges=model.max_graph_edges, chunk_size=model.graph_chunk_size,
                               graph_storage=getattr(model, "graph_storage", "materialized"))


def _decoder(model, raster, sensor_size, states):
    reference = next((state.decoder for state in states if state is not None and state.decoder is not None), None)
    state_batch = None
    if reference is not None:
        state_batch = torch.cat([torch.zeros_like(reference) if state is None or state.decoder is None
                                 else state.decoder for state in states])
    return model.decoder(raster, sensor_size, state_batch)


def stream_forward_batch(model, samples, recurrent_states=None, *, inference_mode="ann",
                         simulation_steps=16, timing=None, calibration=False):
    from .model import rasterize_batch

    if not samples or inference_mode not in {"ann", "snn"}:
        raise ValueError("Streaming forward requires a nonempty ANN or SNN batch")
    if type(simulation_steps) is not int or simulation_steps < 1:
        raise ValueError("Streaming simulation_steps must be a positive integer")
    if model.training and inference_mode != "ann":
        raise ValueError("ASGCN conversion trains the ANN path; SNN is inference-only")
    hierarchical = model.architecture_version == 4
    if not hierarchical and model.event_sampling_factor != 1:
        raise ValueError("The approved physical-stream contract retains all events (R=1)")
    states = [None] * len(samples) if recurrent_states is None else recurrent_states
    if len(states) != len(samples):
        raise ValueError("One explicit streaming state is required per sequence")
    if hierarchical and (model.training or calibration) and any(
            state is not None and (state.encoder is not None or state.hierarchy is not None) for state in states):
        raise ValueError("Hierarchical training/calibration requires raw-only training state, not learned inference caches")
    packed = pack_samples(samples)
    device = packed.events.device
    records, contract = _metadata(model, packed, states)
    id_endpoints, offset = [], 0
    for count in packed.event_counts:
        if count:
            id_endpoints.extend((offset, offset + count - 1))
        offset += count
    control_ids = iter(packed.event_ids[id_endpoints].cpu().tolist())
    last_ids = []
    for count, previous in zip(packed.event_counts, states):
        if count:
            first_id, last_id = tuple(next(control_ids)), tuple(next(control_ids))
            if previous is not None and previous.last_event_id is not None and first_id <= previous.last_event_id:
                raise ValueError("An event identity was repeated or reordered across streaming frames")
        else:
            last_id = None if previous is None else previous.last_event_id
        last_ids.append(last_id)
    graph, cache = _pack_previous(states, device, model)
    if cache is not None and (cache.mode != inference_mode or
                             (inference_mode == "snn" and cache.dynamics != model.snn_dynamics)):
        raise ValueError("Streaming inference dynamics cannot change inside a sequence")
    features, positions, timestamps, node_batch = _prepared(model, packed, records)
    raw_counts = packed.event_counts
    sampling_offsets = [0 if state is None else state.sampling_offset for state in states]
    hierarchy = None
    if hierarchical:
        from .hierarchy import forward_snapshot, pack_hierarchy, split_hierarchy, update_hierarchy
        from .stream_sampling import replace_record_groups, sample_stream_batch
        sampled = sample_stream_batch(packed, sampling_offsets, factor=model.event_sampling_factor)
        packed = sampled.packed
        records = replace_record_groups(records, sampled.arrival_group_counts)
        sampling_offsets = sampled.next_offsets
        features, positions, timestamps, node_batch = (
            value[sampled.keep_mask] for value in (features, positions, timestamps, node_batch))
        hierarchy = pack_hierarchy(model, states, graph, packed.sensor_size)
    window = model.stream_config["window_seconds"]
    readouts = timestamps.new_tensor([record[3] for record in records])
    watermarks = timestamps.new_tensor([state.watermark_seconds if state is not None else record[2]
                                       for state, record in zip(states, records)])
    operation_totals = {"arrival_updates": 0, "readout_updates": 0,
                        "incoming_events": len(timestamps), "training_dense_snapshot": model.training or calibration}
    emitted_totals = [features.new_zeros(len(packed)) for _ in model.encoder.layers]
    tick_totals = [node_batch.new_zeros(len(packed)) for _ in model.encoder.layers]

    def record_work(value):
        for key in ("updated_nodes_per_layer", "message_edges_per_layer", "projected_sources_per_layer"):
            previous = operation_totals.setdefault(key, [0] * len(model.encoder.layers))
            operation_totals[key] = [a + b for a, b in zip(
                previous, value.work.get(key, [0] * len(model.encoder.layers)), strict=True)]
        operation_totals["topology_indexed_edges"] = (operation_totals.get("topology_indexed_edges", 0)
                                                       + value.work.get("topology_indexed_edges", 0))
        if inference_mode == "snn":
            for destination, source in zip(emitted_totals, value.work["emitted_spikes_per_graph_per_layer"]):
                destination[:len(source)].add_(source)
            for destination, source in zip(tick_totals, value.work["neuron_ticks_per_graph_per_layer"]):
                destination[:len(source)].add_(source)

    def scope(name):
        return timing.scope(name, gpu=device.type == "cuda") if timing is not None else nullcontext()

    def encode_update(update, active_graphs=None):
        nonlocal cache, hierarchy
        if not hierarchical:
            cache = update_encoder(model.encoder, update, cache, mode=inference_mode,
                                   simulation_steps=simulation_steps, dynamics=model.snn_dynamics,
                                   active_graphs=active_graphs)
            record_work(cache)
            return
        from types import SimpleNamespace
        cache, hierarchy, work = update_hierarchy(
            model, update, cache, hierarchy, packed.sensor_size, mode=inference_mode,
            simulation_steps=simulation_steps, active_graphs=active_graphs)
        for prefix, suffix in work:
            pool = prefix["pooling"]
            totals = operation_totals.setdefault("pooling", {"updates": 0, "reused_quotient_updates": 0})
            totals["updates"] += 1
            totals["reused_quotient_updates"] += int(pool.get("reused_quotient_topology", False))
            for key in ("raw_edges_visited", "raw_query_nodes", "feature_rows_updated", "materialized_edges_scanned",
                        "quotient_chunks", "quotient_rehashes", "quotient_probe_rounds", "quotient_final_sorts"):
                totals[key] = totals.get(key, 0) + pool.get(key, 0)
            totals["peak_table_capacity"] = max(totals.get("peak_table_capacity", 0),
                                                 pool.get("quotient_peak_table_capacity", 0))
            merged = {key: prefix[key] + suffix[key] for key in (
                "updated_nodes_per_layer", "message_edges_per_layer", "projected_sources_per_layer",
                "emitted_spikes_per_graph_per_layer", "neuron_ticks_per_graph_per_layer")}
            merged["topology_indexed_edges"] = prefix["topology_indexed_edges"] + suffix["topology_indexed_edges"]
            record_work(SimpleNamespace(work=merged))

    if model.training or calibration:
        # ANN training/calibration observes the same fixed-coordinate sliding
        # graph as the inference reference, but does not reuse learned caches
        # across optimizer updates or apply local BN statistics.
        keep = timestamps >= (readouts[node_batch] - window)
        with scope("graph"):
            update = _update(model, graph, features[keep], positions[keep], timestamps[keep],
                             node_batch[keep], readouts - window)
            graph = update.state
        with scope("encoder"):
            if hierarchical:
                outputs, activations, readout_graph = forward_snapshot(
                    model, graph, packed.sensor_size, calibration=calibration)
            else:
                outputs, activations = model.encoder.forward_ann(graph.graph, return_activations=calibration)
                readout_graph = graph
        cache = None
    else:
        # Equal-timestamp events form one simultaneous arrival. Wave r contains
        # the r-th arrival of every independent stream: no causally dependent
        # timestamps are coalesced to manufacture throughput.
        wave_counts = [0] * max((len(record[5]) for record in records), default=0)
        wave_ids = []
        for record in records:
            for wave, count in enumerate(record[5]):
                wave_counts[wave] += count
                wave_ids.extend([wave] * count)
        order = torch.argsort(torch.tensor(wave_ids, device=device, dtype=torch.long), stable=True)
        start = 0
        for count in wave_counts:
            indices = order[start:start + count]
            start += count
            arrived_batch = node_batch[indices]
            active_graphs = torch.bincount(arrived_batch, minlength=len(packed)) > 0
            watermarks = watermarks.scatter_reduce(0, arrived_batch, timestamps[indices],
                                                   reduce="amax", include_self=True)
            with scope("graph"):
                update = _update(model, graph, features[indices], positions[indices], timestamps[indices],
                                 arrived_batch, watermarks - window)
                graph = update.state
            with scope("encoder"):
                encode_update(update, active_graphs)
            operation_totals["arrival_updates"] += 1
        # Readout is an explicit clock boundary: expire old nodes and deliver
        # pending pulse-off effects using the same local-sweep rule.
        with scope("graph"):
            update = _update(model, graph, features[:0], positions[:0], timestamps[:0],
                             node_batch[:0], readouts - window)
            graph = update.state
        with scope("encoder"):
            encode_update(update)
        operation_totals["readout_updates"] = 1
        outputs, activations = (hierarchy.suffix.outputs if hierarchical else cache.outputs), []
        readout_graph = hierarchy.pool.graph if hierarchical else graph
    if inference_mode == "snn":
        outputs = outputs * model.encoder.output_activation_scale(outputs)
    with scope("decoder"):
        if calibration:
            predictions, next_decoder = None, None
        else:
            raster = rasterize_batch(outputs, readout_graph.graph, readout_graph.node_batch, len(packed),
                                     packed.sensor_size, model.raster_downsample)
            predictions, next_decoder = _decoder(model, raster, packed.sensor_size, states)
    lanes = _split_state(graph, cache, len(packed))
    hierarchy_lanes = (split_hierarchy(hierarchy, lanes, len(packed))
                       if hierarchy is not None else [None] * len(packed))
    diagnostics = []
    for lane_index, ((lane, lane_cache), record, previous) in enumerate(zip(lanes, records, states)):
        last_id = last_ids[lane_index]
        state = StreamingReconstructionState(
            lane, lane_cache, None if next_decoder is None else next_decoder[lane_index:lane_index + 1],
            record[4], record[3], record[1], record[0], last_id, contract,
            sampling_offset=sampling_offsets[lane_index] if hierarchical else 0,
            hierarchy=hierarchy_lanes[lane_index],
        )
        degree = lane.graph.in_degree
        count = len(lane.timestamps)
        isolated = (degree == 0).sum()
        if inference_mode == "snn":
            assert lane_cache is not None
            denominators = [value[lane_index] for value in tick_totals]
            spikes = [value[lane_index] for value in emitted_totals]
            rates = [spikes_count / denominator.clamp_min(1)
                     for spikes_count, denominator in zip(spikes, denominators)]
        else:
            rates, denominators, spikes = [], [], []
        diagnostics.append({
            "architecture": model.architecture_description(), "paper_core_version": 2,
            "nodes": count, "edges": (lane.graph.edge_count if model.graph_storage == "implicit_radius"
                                    else lane.graph.edge_index.shape[1]), "isolated_nodes": isolated,
            "isolate_ratio": isolated.float() / max(count, 1),
            "max_degree": degree.max() if count else degree.new_zeros(()),
            "edge_feature": "fixed_physical_scalar_distance", "event_sampling_factor": model.event_sampling_factor,
            "dataset_sampling_ratio": 1.0, "effective_sampling_ratio": 1.0 / model.event_sampling_factor,
            "raw_incoming_events": raw_counts[lane_index], "sampled_incoming_events": packed.event_counts[lane_index],
            "sampling_offset": sampling_offsets[lane_index] if hierarchical else 0,
            "hierarchy_nodes": (readout_graph.node_batch == lane_index).sum() if hierarchical else None,
            "hierarchy_edges": (readout_graph.node_batch[readout_graph.graph.edge_index[0]] == lane_index).sum()
            if hierarchical else None,
            "snn_dynamics": model.snn_dynamics if inference_mode == "snn" else None,
            "decoder_input_lambda_applied": inference_mode == "snn", "firing_rates": rates,
            "firing_rate_denominators": denominators, "spike_counts": spikes,
            "activations": activations if calibration else [], "recurrent_state": state,
            "stream_execution": dict(operation_totals),
            "firing_statistics_scope": "current_frame_actual_local_updates_not_cumulative_window_history",
        })
    if calibration:
        model.encoder.update_activation_maxima(activations,
                                               sample_count=sum(len(lane.timestamps) > 0 for lane, _ in lanes))
        model.calibration_attempts.add_(len(packed))
    return predictions, diagnostics
