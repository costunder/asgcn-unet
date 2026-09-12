"""Independent tiny CPU clock equations; not throughput or model-quality evidence.

The oracle uses explicit root-only affine equations and handwritten cluster
membership. It never calls the encoder, pooling, or hierarchy update helpers.
This isolates pulse timing from the separate quotient-edge tests.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from asgcn_unet.graph import ASGCNEncoder, EventGraph
from asgcn_unet.hierarchy import (
    forward_snapshot,
    pack_hierarchy,
    split_hierarchy,
    stages,
    update_hierarchy,
)
from asgcn_unet.stream_graph import GraphUpdate, StreamGraph
from asgcn_unet.stream_model import _pack_previous, _split_state


@pytest.fixture(autouse=True, scope="module")
def _one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model(coefficients=(0.6, 0.4), biases=None, *, boundary=1, dynamics="standard_if"):
    # Explicit synthetic width, no production model/config is changed.
    with torch.random.fork_rng():
        torch.manual_seed(409)
        encoder = ASGCNEncoder(1, len(coefficients), spline_kernel_size=2).eval()
    biases = [0.0] * len(coefficients) if biases is None else biases
    with torch.no_grad():
        for layer, coefficient, bias in zip(encoder.layers, coefficients, biases, strict=True):
            layer.weight.zero_()
            layer.root.zero_()
            layer.root[0, 0] = coefficient
            layer.bias.fill_(bias)
            layer.norm.weight.copy_(torch.sqrt(layer.norm.running_var + layer.norm.eps))
            layer.norm.bias.zero_()
            layer.norm.running_mean.zero_()
            # A controlled unit conversion is part of this numerical fixture.
            layer.calibration_activation_max.fill_(1.0)
        encoder.calibration_samples_seen.fill_(1)
        encoder.fold_batch_norm()
        encoder.apply_parameter_normalization()
    return SimpleNamespace(
        encoder=encoder, snn_dynamics=dynamics,
        hierarchy_config={"after_layer": boundary, "spatial_cell_pixels": 4,
                          "temporal_cell_seconds": 1,
                          "edge_pseudo": "mean_fine_distance_over_radius"},
        stream_config={"time_scale_seconds": 1},
    )


def _graph(*, lanes=1):
    features = torch.tensor([[1.0, 0, 0, 0], [0.35, 0, 0, 0]]).repeat(lanes, 1)
    positions = torch.tensor([[0.1, 0.1, 0.1, 0], [0.2, 0.1, 0.1, 1]],
                             dtype=torch.float64).repeat(lanes, 1)
    # Both raw nodes of each lane belong to one declared fixed cell. No edges
    # means the independent oracle can use x @ root + bias directly.
    graph = EventGraph(features, positions, torch.empty(2, 0, dtype=torch.long),
                       torch.empty(0, 1, dtype=torch.float64))
    return StreamGraph(graph, torch.arange(lanes).repeat_interleave(2),
                       torch.full((2 * lanes,), 0.1, dtype=torch.float64))


def _update(graph, *, initial=False, raw_seed=None):
    count = len(graph.timestamps)
    return GraphUpdate(
        graph, torch.full((count,), -1, dtype=torch.long) if initial else torch.arange(count),
        torch.full((count,), initial, dtype=torch.bool) if raw_seed is None else raw_seed,
    )


def _oracle_initial(model, graph):
    boundary = model.hierarchy_config["after_layer"]
    raw_n = len(graph.timestamps)
    coarse_n = raw_n // 2
    result = {name: [] for name in ("membranes", "previous_spikes", "spike_sums",
                                    "local_ticks", "last_pulses", "layer_outputs")}
    for index, layer in enumerate(model.encoder.layers):
        count = raw_n if index < boundary else coarse_n
        for name, tensors in result.items():
            value = torch.zeros(count, dtype=torch.long) if name == "local_ticks" else torch.zeros(count, 1)
            if name == "membranes":
                value[:] = layer.threshold * 0.5
            tensors.append(value)
    result["pooled_last"] = torch.zeros(coarse_n, 1)
    return result


@torch.no_grad()
def _oracle(model, graph, previous, *, steps, raw_seed, topology_seed, active_graphs=None):
    """Direct IF equations with topology clocks separate from input pulses."""
    state = _oracle_initial(model, graph) if previous is None else copy.deepcopy(previous)
    boundary = model.hierarchy_config["after_layer"]
    active_graphs = torch.ones(len(graph.timestamps) // 2, dtype=torch.bool) if active_graphs is None else active_graphs
    raw_active = active_graphs.repeat_interleave(2)
    for _tick in range(steps):
        old_pulses = state["last_pulses"]
        new_pulses = []
        for index, layer in enumerate(model.encoder.layers):
            if index == 0:
                x = graph.graph.node_features
                active = raw_seed & raw_active
            elif index == boundary:
                # Handwritten Eq18 for the known two-member cells.
                x = new_pulses[-1].reshape(-1, 2, 1).mean(dim=1)
                input_support = x.ne(0).any(1) | state["pooled_last"].ne(0).any(1)
                active = (topology_seed | input_support) & active_graphs
                pooled_now = x
            else:
                x = new_pulses[-1]
                pulse_support = x.ne(0).any(1) | old_pulses[index - 1].ne(0).any(1)
                base = raw_seed if index < boundary else topology_seed
                active = (base | pulse_support) & (raw_active if index < boundary else active_graphs)
            current = x @ layer.root + layer.bias
            integrated = state["membranes"][index][active] + current[active]
            if model.snn_dynamics == "literal_eq15":
                integrated += state["previous_spikes"][index][active]
            spike = torch.where(integrated >= layer.threshold, layer.threshold, 0.0)
            state["membranes"][index][active] = integrated - spike
            state["previous_spikes"][index][active] = spike
            state["spike_sums"][index][active] += spike
            state["local_ticks"][index][active] += 1
            state["layer_outputs"][index][active] = (
                state["spike_sums"][index][active] / state["local_ticks"][index][active, None])
            emitted = torch.zeros_like(state["last_pulses"][index])
            emitted[active] = spike
            lane_active = raw_active if index < boundary else active_graphs
            emitted[~lane_active] = old_pulses[index][~lane_active]
            new_pulses.append(emitted)
        state["last_pulses"] = new_pulses
        state["pooled_last"] = pooled_now
    return state


def _assert_equations(prefix, hierarchy, expected):
    for field in ("membranes", "previous_spikes", "spike_sums", "local_ticks",
                  "last_pulses", "layer_outputs"):
        actual = getattr(prefix, field) + getattr(hierarchy.suffix, field)
        for index, (got, wanted) in enumerate(zip(actual, expected[field], strict=True)):
            torch.testing.assert_close(got, wanted, atol=2e-6, rtol=2e-6,
                                       msg=f"layer {index} {field}: independent local-clock equation")


@pytest.mark.parametrize("dynamics", ["standard_if", "literal_eq15"])
@pytest.mark.parametrize("depth,boundary", [(2, 1), (6, 3)])
def test_arrival_interleaves_prefix_pool_suffix_and_repeated_equal_pulses(dynamics, depth, boundary):
    model = _model((0.6,) + (0.7,) * (depth - 1), boundary=boundary, dynamics=dynamics)
    graph = _graph()
    expected = _oracle(model, graph, None, steps=5, raw_seed=torch.ones(2, dtype=torch.bool),
                       topology_seed=torch.ones(1, dtype=torch.bool))
    prefix, hierarchy, _ = update_hierarchy(model, _update(graph, initial=True), None, None,
                                           (8, 8), mode="snn", simulation_steps=5)
    _assert_equations(prefix, hierarchy, expected)
    torch.testing.assert_close(hierarchy.pool.graph.graph.node_features, expected["pooled_last"])


@pytest.mark.parametrize("dynamics", ["standard_if", "literal_eq15"])
@pytest.mark.parametrize("depth", [2, 3])
def test_readout_pulse_off_is_one_input_change_not_persistent_or_broadcast_topology(dynamics, depth):
    model = _model((0.6,) + (0.0,) * (depth - 1),
                   biases=[0.0] + [0.21] * (depth - 1), dynamics=dynamics)
    graph = _graph()
    expected = _oracle(model, graph, None, steps=1, raw_seed=torch.ones(2, dtype=torch.bool),
                       topology_seed=torch.ones(1, dtype=torch.bool))
    prefix, hierarchy, _ = update_hierarchy(model, _update(graph, initial=True), None, None,
                                           (8, 8), mode="snn", simulation_steps=1)
    _assert_equations(prefix, hierarchy, expected)
    expected = _oracle(model, graph, expected, steps=4, raw_seed=torch.zeros(2, dtype=torch.bool),
                       topology_seed=torch.zeros(1, dtype=torch.bool))
    prefix, hierarchy, _ = update_hierarchy(model, _update(graph), prefix, hierarchy,
                                           (8, 8), mode="snn", simulation_steps=4)
    _assert_equations(prefix, hierarchy, expected)


@pytest.mark.parametrize("dynamics", ["standard_if", "literal_eq15"])
def test_idle_lane_keeps_membranes_ticks_and_pending_pulses_while_other_lane_advances(dynamics):
    model = _model((0.6, 0.7, 0.4), dynamics=dynamics)
    graph = _graph(lanes=2)
    expected = _oracle(model, graph, None, steps=1, raw_seed=torch.ones(4, dtype=torch.bool),
                       topology_seed=torch.ones(2, dtype=torch.bool))
    prefix, hierarchy, _ = update_hierarchy(model, _update(graph, initial=True), None, None,
                                           (8, 8), mode="snn", simulation_steps=1)
    _assert_equations(prefix, hierarchy, expected)
    incoming = copy.deepcopy(expected)
    expected = _oracle(model, graph, expected, steps=4, raw_seed=torch.zeros(4, dtype=torch.bool),
                       topology_seed=torch.zeros(2, dtype=torch.bool), active_graphs=torch.tensor([True, False]))
    prefix, hierarchy, _ = update_hierarchy(model, _update(graph), prefix, hierarchy, (8, 8),
                                           mode="snn", simulation_steps=4,
                                           active_graphs=torch.tensor([True, False]))
    _assert_equations(prefix, hierarchy, expected)
    for field in ("membranes", "local_ticks", "last_pulses"):
        for index, values in enumerate(expected[field]):
            idle = slice(2, 4) if index == 0 else slice(1, 2)
            torch.testing.assert_close(values[idle], incoming[field][index][idle], atol=0, rtol=0)


@torch.no_grad()
def test_six_layer_bn_folding_normalization_and_mean_pool_have_one_continuous_scale_chain():
    model = _model((0.6, 0.7, 0.8, 0.9, 1.1, 1.2), boundary=3)
    # Fresh unconverted parameters with nontrivial BN at all six layers.
    for index, layer in enumerate(model.encoder.layers):
        layer._bn_is_folded = False
        layer._snn_is_normalized = False
        layer.bn_bypassed.fill_(False)
        layer.snn_normalized.fill_(False)
        layer.norm.weight.fill_(1.0 + 0.1 * index)
        layer.norm.bias.fill_(0.1 + 0.01 * index)
        layer.norm.running_mean.fill_(0.02 * index)
        layer.norm.running_var.fill_(0.8 + 0.1 * index)
    graph = _graph()
    hidden = graph.graph.node_features
    expected_activations = []
    for index, layer in enumerate(model.encoder.layers):
        if index == 3:
            hidden = hidden.reshape(1, 2, 1).mean(1)
        affine = hidden @ layer.root + layer.bias
        hidden = torch.relu((affine - layer.norm.running_mean)
                            * layer.norm.weight / torch.sqrt(layer.norm.running_var + layer.norm.eps)
                            + layer.norm.bias)
        expected_activations.append(hidden.clone())
    expected = hidden.clone()
    actual, activations, _ = forward_snapshot(model, graph, (8, 8), calibration=True)
    torch.testing.assert_close(actual, expected)
    for got, wanted in zip(activations, expected_activations, strict=True):
        torch.testing.assert_close(got, wanted)
    model.encoder.fold_batch_norm()
    folded, _, _ = forward_snapshot(model, graph, (8, 8))
    torch.testing.assert_close(folded, expected)
    # Explicit non-unit channel scales isolate accidental reset/double-scaling
    # at the pooling boundary from calibration-data selection.
    for index, layer in enumerate(model.encoder.layers):
        layer.calibration_activation_max.fill_(index + 2.0)
    model.encoder.apply_parameter_normalization()
    normalized, _, _ = forward_snapshot(model, graph, (8, 8))
    torch.testing.assert_close(normalized * model.encoder.output_activation_scale(normalized), expected)
    prefix, suffix = stages(model)
    assert prefix.layers[-1] is model.encoder.layers[2]
    assert suffix.layers[0] is model.encoder.layers[3]
    assert suffix.input_is_spiking


@pytest.mark.parametrize("dynamics", ["standard_if", "literal_eq15"])
def test_equal_pooled_pulses_each_tick_are_not_suppressed_when_topology_is_unchanged(dynamics):
    model = _model((1.0, 0.25), dynamics=dynamics)
    graph = _graph()
    graph.graph.node_features[:, 0] = 1.0
    raw_seed = torch.ones(2, dtype=torch.bool)
    expected = _oracle(model, graph, None, steps=1, raw_seed=raw_seed,
                       topology_seed=torch.ones(1, dtype=torch.bool))
    prefix, hierarchy, _ = update_hierarchy(model, _update(graph, initial=True), None, None,
                                           (8, 8), mode="snn", simulation_steps=1)
    expected = _oracle(model, graph, expected, steps=4, raw_seed=raw_seed,
                       topology_seed=torch.zeros(1, dtype=torch.bool))
    prefix, hierarchy, _ = update_hierarchy(model, _update(graph, raw_seed=raw_seed), prefix,
                                           hierarchy, (8, 8), mode="snn", simulation_steps=4)
    assert not hierarchy.pool.work["topology_changed_nodes"].any()
    assert hierarchy.pool.graph.graph.node_features.item() == 1.0
    assert expected["local_ticks"][1].item() == 5
    _assert_equations(prefix, hierarchy, expected)


@pytest.mark.parametrize("dynamics", ["standard_if", "literal_eq15"])
def test_pack_split_retry_preserves_the_independent_pulse_clock_equations(dynamics):
    model = _model((0.6, 0.7, 0.4), dynamics=dynamics)
    graph = _graph(lanes=2)
    expected = _oracle(model, graph, None, steps=3, raw_seed=torch.ones(4, dtype=torch.bool),
                       topology_seed=torch.ones(2, dtype=torch.bool))
    prefix, hierarchy, _ = update_hierarchy(model, _update(graph, initial=True), None, None,
                                           (8, 8), mode="snn", simulation_steps=3)
    raw_lanes = _split_state(graph, prefix, 2)
    coarse_lanes = split_hierarchy(hierarchy, raw_lanes, 2)
    states = [SimpleNamespace(graph=raw, encoder=cache, hierarchy=coarse)
              for (raw, cache), coarse in zip(raw_lanes, coarse_lanes, strict=True)]
    packed_raw, packed_prefix = _pack_previous(states, torch.device("cpu"))
    packed_hierarchy = pack_hierarchy(model, states, packed_raw, (8, 8))
    _assert_equations(packed_prefix, packed_hierarchy, expected)
    assert packed_hierarchy.pool.raw_graph is packed_raw
    assert packed_hierarchy.suffix.graph is packed_hierarchy.pool.graph
    incoming = copy.deepcopy(expected)
    expected = _oracle(model, graph, expected, steps=4, raw_seed=torch.zeros(4, dtype=torch.bool),
                       topology_seed=torch.zeros(2, dtype=torch.bool))
    old_pool_features = packed_hierarchy.pool.graph.graph.node_features.clone()
    old_sums = packed_hierarchy.pool.feature_sums.clone()
    for _retry in range(2):
        actual_prefix, actual_hierarchy, _ = update_hierarchy(
            model, _update(packed_raw), packed_prefix, packed_hierarchy, (8, 8),
            mode="snn", simulation_steps=4)
        _assert_equations(actual_prefix, actual_hierarchy, expected)
        _assert_equations(packed_prefix, packed_hierarchy, incoming)
        torch.testing.assert_close(packed_hierarchy.pool.graph.graph.node_features,
                                   old_pool_features, atol=0, rtol=0)
        torch.testing.assert_close(packed_hierarchy.pool.feature_sums, old_sums, atol=0, rtol=0)


@torch.no_grad()
def test_ann_six_layer_feature_only_change_reaches_suffix_without_coarse_topology_change():
    model = _model((0.6, 0.7, 0.8, 0.9, 1.1, 1.2), boundary=3)
    graph = _graph()
    prefix, hierarchy, _ = update_hierarchy(model, _update(graph, initial=True), None, None,
                                           (8, 8), mode="ann", simulation_steps=1)
    changed = _graph()
    changed.graph.node_features[:, 0] *= 1.5
    prefix, hierarchy, _ = update_hierarchy(
        model, _update(changed, raw_seed=torch.ones(2, dtype=torch.bool)), prefix,
        hierarchy, (8, 8), mode="ann", simulation_steps=1)
    assert not hierarchy.pool.work["topology_changed_nodes"].any()
    hidden = changed.graph.node_features
    for index, layer in enumerate(model.encoder.layers):
        if index == 3:
            hidden = hidden.reshape(1, 2, 1).mean(1)
        hidden = torch.relu(hidden @ layer.root + layer.bias)
        actual = prefix.layer_outputs[index] if index < 3 else hierarchy.suffix.layer_outputs[index - 3]
        torch.testing.assert_close(actual, hidden)
