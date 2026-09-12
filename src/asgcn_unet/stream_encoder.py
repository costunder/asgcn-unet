"""Incremental paper-core encoder with explicitly event-local IF clocks.

This is NOT the static encoder's global-T-clock execution: only topology seeds
and their causal feature/pulse dependants advance. A stream update contains T
local sweeps. Pending one-sweep pulse endings are delivered on the next sweep,
including the first sweep of the next update. Neurons outside this causal set
keep their membrane, previous-local-tick spike, and cumulative readout unchanged.

Topology indexing visits the complete edge list once per update. Learned
projection and message aggregation visit only incident edges of affected
destinations; SNN projection additionally excludes zero-valued sources. Indexing
cost is reported separately, not disguised as constant-time incremental work.
All node/sample work is tensor-batched; only dependent layers/ticks are iterated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .graph import ASGCNEncoder, EventGraph, PaperSplineConv, linear_open_bspline_basis
from .ops import require_spline_backend, weighted_spline_sum
from .stream_graph import GraphUpdate, StreamGraph


@dataclass(frozen=True)
class StreamEncoderState:
    """Remappable per-node state; the function never changes a supplied state.

    ``layer_outputs`` are ANN activations or SNN cumulative local spike rates.
    ``previous_spikes`` means the previous tick of each neuron's *own* clock;
    ``last_pulses`` instead means the previous update/sweep's transmitted pulse.
    Their distinction is required when an inactive neuron retains local state
    while its one-tick message pulse ends. ``local_ticks`` is [N] for each layer.
    ``spike_sums`` stores threshold-valued spikes (not binary firing counts).
    """

    graph: StreamGraph
    outputs: torch.Tensor
    layer_outputs: tuple[torch.Tensor, ...]
    membranes: tuple[torch.Tensor, ...] = ()
    previous_spikes: tuple[torch.Tensor, ...] = ()
    spike_sums: tuple[torch.Tensor, ...] = ()
    local_ticks: tuple[torch.Tensor, ...] = ()
    last_pulses: tuple[torch.Tensor, ...] = ()
    mode: str = "ann"
    dynamics: str | None = None
    work: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Incidence:
    incoming_order: torch.Tensor
    incoming_ptr: torch.Tensor
    outgoing_order: torch.Tensor
    outgoing_ptr: torch.Tensor

    @classmethod
    def build(cls, graph: EventGraph) -> _Incidence:
        n = graph.node_features.shape[0]
        source, destination = graph.edge_index

        def index(values):
            # Stable within each node preserves the original per-destination
            # accumulation order in the reference torch spline backend.
            order = torch.argsort(values, stable=True)
            counts = torch.bincount(values, minlength=n)
            ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            return order, ptr

        incoming_order, incoming_ptr = index(destination)
        outgoing_order, outgoing_ptr = index(source)
        return cls(incoming_order, incoming_ptr, outgoing_order, outgoing_ptr)

    @staticmethod
    def gather(order: torch.Tensor, ptr: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """Vectorized ragged CSR ranges, without an E-length membership mask."""
        starts = ptr[nodes]
        counts = ptr[nodes + 1] - starts
        repeated_starts = torch.repeat_interleave(starts, counts)
        offsets = torch.repeat_interleave(counts.cumsum(0) - counts, counts)
        locations = repeated_starts + torch.arange(
            repeated_starts.numel(), device=nodes.device
        ) - offsets
        return order[locations]

    def incoming(self, nodes: torch.Tensor) -> torch.Tensor:
        return self.gather(self.incoming_order, self.incoming_ptr, nodes)

    def dependants(self, graph: EventGraph, sources: torch.Tensor) -> torch.Tensor:
        edges = self.gather(self.outgoing_order, self.outgoing_ptr, sources)
        # Self is needed for the root affine even in a graph without self edges.
        return torch.unique(torch.cat((sources, graph.edge_index[1, edges])), sorted=True)


def _remap(old, old_indices, shape, reference, *, fill=None, dtype=None):
    result = reference.new_zeros(shape, dtype=dtype)
    if fill is not None:
        result.copy_(fill.expand_as(result))
    if old is not None:
        retained = torch.nonzero(old_indices >= 0, as_tuple=False).flatten()
        result[retained] = old[old_indices[retained]]
    return result


def _affected(seed, changed_sources, graph, incidence):
    if not changed_sources.numel():
        return seed
    return torch.unique(torch.cat((seed, incidence.dependants(graph, changed_sources))), sorted=True)


def _partial_affine(
    layer: PaperSplineConv,
    x: torch.Tensor,
    graph: EventGraph,
    destinations: torch.Tensor,
    incidence: _Incidence | None,
    *,
    omit_zero_sources: bool,
) -> tuple[torch.Tensor, int, int]:
    """Compute complete incoming means at selected destinations, never globally.

    Compact source IDs and compact destination IDs deliberately occupy separate
    namespaces. Padding the projection to max(source_count, destination_count)
    satisfies the existing backend's output allocation contract without
    projecting the padding or any unaffected node.
    """
    if not destinations.numel():
        return x.new_zeros((0, layer.out_channels)), 0, 0
    from .implicit_radius import ImplicitRadiusGraph
    if isinstance(graph, ImplicitRadiusGraph):
        from .implicit_model import affine
        values, sources, edges = affine(layer, x, graph, destinations,
                                        omit_zero_sources=omit_zero_sources)
        return values, edges, sources
    if incidence is None:
        raise RuntimeError("active destinations require a topology incidence index")
    selected_edges = incidence.incoming(destinations)
    source = graph.edge_index[0, selected_edges]
    destination = graph.edge_index[1, selected_edges]
    sources, inverse = torch.unique(source, sorted=True, return_inverse=True)
    if omit_zero_sources:
        active = x[sources].ne(0).any(dim=1)
        selected_edges = selected_edges[active[inverse]]
        source = graph.edge_index[0, selected_edges]
        destination = graph.edge_index[1, selected_edges]
        sources, inverse = torch.unique(source, sorted=True, return_inverse=True)
    count = destinations.numel()
    output = x.new_zeros((count, layer.out_channels))
    if selected_edges.numel():
        projected = torch.einsum("ni,kio->nko", x[sources], layer.weight)
        if sources.numel() < count:
            projected = torch.cat((
                projected,
                projected.new_zeros((count - sources.numel(), layer.kernel_size, layer.out_channels)),
            ))
        local_destination = torch.searchsorted(destinations, destination)
        indices, basis = linear_open_bspline_basis(graph.edge_attr[selected_edges], layer.kernel_size)
        chunk_size = selected_edges.numel() if layer.edge_chunk_size is None else layer.edge_chunk_size
        output = weighted_spline_sum(
            projected, inverse, local_destination, indices, basis, chunk_size, x.dtype,
            backend=layer.spline_backend,
        )[:count]
        # Degree is the FULL live graph degree, including zero-spike sources.
        output = output / graph.in_degree[destinations].to(x).unsqueeze(-1).clamp_min(1)
    if layer.root is not None:
        output = output + x[destinations] @ layer.root
    if layer.bias is not None:
        output = output + layer.bias
    return output, selected_edges.numel(), sources.numel()


def _validate(encoder, update, previous, mode, simulation_steps, dynamics):
    from .encoder_stage import EncoderStage
    if not isinstance(encoder, (ASGCNEncoder, EncoderStage)):
        raise TypeError("stream encoder requires the existing ASGCNEncoder")
    if encoder.training or any(layer.norm.training for layer in encoder.layers):
        raise ValueError("incremental encoder is eval-only; use full-snapshot ANN training with global BN")
    if mode not in {"ann", "snn"}:
        raise ValueError("mode must be ann or snn")
    if isinstance(simulation_steps, bool) or not isinstance(simulation_steps, int) or simulation_steps < 1:
        raise ValueError("simulation_steps must be a positive integer")
    if dynamics not in {"literal_eq15", "standard_if"}:
        raise ValueError("dynamics must be literal_eq15 or standard_if")
    graph = update.state.graph
    n = graph.node_features.shape[0]
    old = update.old_indices
    if old.shape != (n,) or old.dtype != torch.long or old.device != graph.node_features.device:
        raise ValueError("old_indices must be a device-local long tensor with one entry per live node")
    if update.changed_nodes.shape != (n,) or update.changed_nodes.dtype != torch.bool:
        raise ValueError("changed_nodes must be a bool tensor with one entry per live node")
    if update.changed_nodes.device != old.device:
        raise ValueError("changed_nodes and node state must share a device")
    if bool((old < -1).any()):
        raise ValueError("old_indices accepts only -1 arrivals or retained indices")
    if bool(((old < 0) & ~update.changed_nodes).any()):
        raise ValueError("every arriving node must be marked changed")
    if previous is None:
        if bool((old >= 0).any()):
            raise ValueError("retained indices require previous encoder state")
    else:
        if previous.mode != mode or previous.dynamics != (dynamics if mode == "snn" else None):
            raise ValueError("stream mode/dynamics changed; begin a separate encoder state")
        retained = old[old >= 0]
        if bool((retained >= previous.outputs.shape[0]).any()) or retained.unique().numel() != retained.numel():
            raise ValueError("retained encoder indices are invalid or duplicated")
        fields = ("layer_outputs",) if mode == "ann" else (
            "layer_outputs", "membranes", "previous_spikes", "spike_sums", "local_ticks", "last_pulses"
        )
        for name in fields:
            values = getattr(previous, name)
            if len(values) != len(encoder.layers):
                raise ValueError(f"previous {name} does not match encoder depth")
            for layer, value in zip(encoder.layers, values, strict=True):
                shape = (previous.outputs.shape[0],) if name == "local_ticks" else (
                    previous.outputs.shape[0], layer.out_channels
                )
                if value.shape != shape or value.device != old.device:
                    raise ValueError(f"previous {name} has incompatible shape or device")
    if mode == "snn" and any(not layer._snn_is_normalized for layer in encoder.layers):
        raise RuntimeError("stream SNN requires the existing calibrated parameter normalization")
    require_spline_backend(encoder.spline_backend, graph.node_features.device)


@torch.no_grad()
def update_encoder(
    encoder: ASGCNEncoder,
    update: GraphUpdate,
    previous: StreamEncoderState | None = None,
    *,
    mode: str = "ann",
    simulation_steps: int = 16,
    dynamics: str = "literal_eq15",
    active_graphs: torch.Tensor | None = None,
    input_changed_sources: torch.Tensor | None = None,
) -> StreamEncoderState:
    """Apply one explicit arrival/expiry/readout-time advance, retry-safe.

    Calling again with the same ``previous`` recomputes the same transition;
    feeding the result back as ``previous`` instead advances another local
    update, including pending pulse endings. No wall-clock/bias ticks occur in
    untouched neurons. Simulation steps are not input-event timestamps.

    ``input_changed_sources`` identifies changed/emitted/ending external input
    pulses. Their dependants enter the FIRST layer only; they are not topology
    changes broadcast to every layer. Hierarchical SNN callers provide one
    interleaved tick at a time and recompute this source mask for each tick.
    """
    _validate(encoder, update, previous, mode, simulation_steps, dynamics)
    graph = update.state.graph
    reference = graph.node_features
    n = reference.shape[0]
    if active_graphs is None:
        active_nodes = torch.ones(n, device=reference.device, dtype=torch.bool)
    else:
        if (active_graphs.ndim != 1 or active_graphs.dtype != torch.bool
                or active_graphs.device != reference.device):
            raise ValueError("active_graphs must be a same-device boolean vector")
        active_nodes = active_graphs[update.state.node_batch]
        if bool((update.changed_nodes & ~active_nodes).any()):
            raise ValueError("A topology change cannot be hidden in an idle graph")
    old = update.old_indices
    seed = torch.nonzero(update.changed_nodes, as_tuple=False).flatten()
    input_sources = old.new_empty(0)
    if input_changed_sources is not None:
        if (input_changed_sources.shape != (n,) or input_changed_sources.dtype != torch.bool
                or input_changed_sources.device != reference.device):
            raise ValueError("input_changed_sources must be a device-local bool vector over live nodes")
        if bool((input_changed_sources & ~active_nodes).any()):
            raise ValueError("An input pulse change cannot be hidden in an idle graph")
        input_sources = torch.nonzero(input_changed_sources, as_tuple=True)[0]
    # An ANN expiry of isolated nodes or a readout-only advance can have no
    # changed destinations at all. It requires remapping, not an edge sort.
    from .implicit_radius import ImplicitRadiusGraph
    implicit = isinstance(graph, ImplicitRadiusGraph)
    if mode == "ann" and not seed.numel() and not input_sources.numel():
        incidence = None
    elif implicit:
        from .implicit_stream import ImplicitIncidence
        incidence = ImplicitIncidence(graph)
    else:
        incidence = _Incidence.build(graph)
    first_destinations = _affected(seed, input_sources, graph, incidence)
    outputs, membranes, previous_spikes, spike_sums, local_ticks, last_pulses = [], [], [], [], [], []
    for i, layer in enumerate(encoder.layers):
        shape = (n, layer.out_channels)
        outputs.append(_remap(None if previous is None else previous.layer_outputs[i], old, shape, reference))
        if mode == "snn":
            membranes.append(_remap(
                None if previous is None else previous.membranes[i], old, shape, reference,
                fill=layer.threshold.to(reference)[None] * 0.5,
            ))
            previous_spikes.append(_remap(None if previous is None else previous.previous_spikes[i], old, shape, reference))
            spike_sums.append(_remap(None if previous is None else previous.spike_sums[i], old, shape, reference))
            local_ticks.append(_remap(None if previous is None else previous.local_ticks[i], old, (n,), reference, dtype=torch.long))
            last_pulses.append(_remap(None if previous is None else previous.last_pulses[i], old, shape, reference))
    updated_nodes = [0] * len(encoder.layers)
    message_edges = [0] * len(encoder.layers)
    projected_sources = [0] * len(encoder.layers)
    batch_counts = torch.bincount(update.state.node_batch)
    emitted_per_graph = [reference.new_zeros(batch_counts.shape) for _ in encoder.layers]
    neuron_ticks_per_graph = [batch_counts.new_zeros(batch_counts.shape) for _ in encoder.layers]
    if mode == "ann":
        x = reference
        changed = seed
        for i, layer in enumerate(encoder.layers):
            destinations = first_destinations if i == 0 else _affected(seed, changed, graph, incidence)
            values, edges, sources = _partial_affine(
                layer, x, graph, destinations, incidence, omit_zero_sources=False
            )
            values = torch.relu(values if layer._bn_is_folded else layer.norm(values))
            changed = destinations[values.ne(outputs[i][destinations]).any(dim=1)]
            outputs[i][destinations] = values
            x = outputs[i]
            updated_nodes[i] += destinations.numel()
            message_edges[i] += edges
            projected_sources[i] += sources
    else:
        # Analog event features and the update's topology stay constant during
        # all local sweeps. Reuse the exact first-layer current, not its spikes.
        first_current, first_edges, first_sources = _partial_affine(
            encoder.layers[0], reference, graph, first_destinations, incidence,
            omit_zero_sources=getattr(encoder, "input_is_spiking", False),
        )
        message_edges[0] = first_edges
        projected_sources[0] = first_sources
        previous_supports = [
            torch.nonzero(p.ne(0).any(dim=1), as_tuple=False).flatten() for p in last_pulses
        ]
        for _ in range(simulation_steps):
            pulses = []
            pulse_supports = []
            for i, layer in enumerate(encoder.layers):
                if i == 0:
                    destinations = first_destinations
                    x = reference
                else:
                    x = pulses[i - 1]
                    # Both an emitted pulse and an ending pulse can alter the
                    # downstream current. Do not hold an old spike as input.
                    changed = torch.unique(torch.cat((
                        pulse_supports[i - 1], previous_supports[i - 1],
                    )), sorted=True)
                    destinations = _affected(seed, changed, graph, incidence)
                    destinations = destinations[active_nodes[destinations]]
                if i == 0:
                    current, edges, sources = first_current, 0, 0
                else:
                    current, edges, sources = _partial_affine(
                        layer, x, graph, destinations, incidence, omit_zero_sources=True
                    )
                integrated = membranes[i][destinations] + current
                if dynamics == "literal_eq15":
                    integrated = integrated + previous_spikes[i][destinations]
                threshold = layer.threshold.to(integrated)[None]
                spikes = torch.where(integrated >= threshold, threshold, 0.0)
                emitted_per_graph[i].index_add_(0, update.state.node_batch[destinations],
                                               spikes.ne(0).sum(dim=1).to(reference.dtype))
                neuron_ticks_per_graph[i].index_add_(
                    0, update.state.node_batch[destinations],
                    torch.full_like(destinations, layer.out_channels),
                )
                membranes[i][destinations] = integrated - spikes
                previous_spikes[i][destinations] = spikes
                spike_sums[i][destinations] += spikes
                local_ticks[i][destinations] += 1
                outputs[i][destinations] = spike_sums[i][destinations] / local_ticks[i][destinations, None]
                emitted = reference.new_zeros((n, layer.out_channels))
                emitted[destinations] = spikes
                # Other streams' arrivals do not advance this stream's clock,
                # including its pending pulse-ending state.
                emitted[~active_nodes] = last_pulses[i][~active_nodes]
                pulses.append(emitted)
                pulse_supports.append(destinations[spikes.ne(0).any(dim=1)])
                updated_nodes[i] += destinations.numel()
                message_edges[i] += edges
                projected_sources[i] += sources
            last_pulses = pulses
            previous_supports = pulse_supports
    return StreamEncoderState(
        graph=update.state, outputs=outputs[-1], layer_outputs=tuple(outputs),
        membranes=tuple(membranes), previous_spikes=tuple(previous_spikes),
        spike_sums=tuple(spike_sums), local_ticks=tuple(local_ticks), last_pulses=tuple(last_pulses),
        mode=mode, dynamics=dynamics if mode == "snn" else None,
        work={
            "clock_policy": "event_local_causal_pulses_v1" if mode == "snn" else "incremental_frozen_bn_ann_v1",
            "update_reason": "explicit_arrival_expiry_or_readout_advance",
            "local_sweeps": simulation_steps if mode == "snn" else 1,
            "live_nodes": n, "live_edges": graph.edge_count if implicit else graph.edge_index.shape[1],
            "topology_indexed_edges": graph.edge_index.shape[1] if incidence is not None and not implicit else 0,
            "updated_nodes_per_layer": updated_nodes,
            "message_edges_per_layer": message_edges,
            "projected_sources_per_layer": projected_sources,
            "emitted_spikes_per_graph_per_layer": emitted_per_graph,
            "neuron_ticks_per_graph_per_layer": neuron_ticks_per_graph,
            "full_graph_timestep_message_passing": False,
            "global_clock_snapshot_equivalent": False if mode == "snn" else None,
        },
    )
