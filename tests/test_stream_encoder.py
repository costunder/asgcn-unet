"""Small synthetic CPU references, not real-data or GPU performance evidence."""

from dataclasses import replace

import pytest
import torch

from asgcn_unet.graph import ASGCNEncoder, EventGraph
from asgcn_unet.stream_encoder import StreamEncoderState, update_encoder
from asgcn_unet.stream_graph import GraphUpdate, StreamGraph


@pytest.fixture(autouse=True)
def synthetic_cpu_threads():
    prior = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(prior)


def _update(features, pairs, *, old=None, changed=None, batch=None):
    features = torch.as_tensor(features, dtype=torch.float32).reshape(-1, 4)
    n = features.shape[0]
    edge_index = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous()
    # Deliberately unsorted edges exercise CSR construction without a sorted
    # topology assumption. A constant scalar coordinate isolates encoder maths.
    graph = EventGraph(features, features.clone(), edge_index, torch.full((len(pairs), 1), 0.3))
    state = StreamGraph(
        graph, torch.zeros(n, dtype=torch.long) if batch is None else torch.tensor(batch),
        torch.arange(n, dtype=torch.float64),
    )
    return GraphUpdate(
        state, torch.full((n,), -1, dtype=torch.long) if old is None else torch.tensor(old),
        torch.ones(n, dtype=torch.bool) if changed is None else torch.tensor(changed),
    )


def _updates():
    torch.manual_seed(432)
    x = torch.rand(6, 4)
    x[:, 3] = torch.tensor([1., -1., 1., -1., 1., -1.])
    initial = _update(x[:5], [(1, 2), (4, 3), (0, 1), (2, 1), (3, 4), (1, 0)])
    append = _update(
        x, [(1, 2), (4, 3), (0, 1), (2, 1), (3, 4), (1, 0), (5, 2), (2, 5)],
        old=[0, 1, 2, 3, 4, -1], changed=[False, False, True, False, False, True],
    )
    expire = _update(
        x[[0, 2, 3, 4, 5]], [(3, 2), (2, 3), (4, 1), (1, 4)],
        old=[0, 2, 3, 4, 5], changed=[True, True, False, False, False],
    )
    advance = replace(expire, old_indices=torch.arange(5), changed_nodes=torch.zeros(5, dtype=torch.bool))
    return initial, append, expire, advance


def _encoder(calibrated=False, *, full=False, backend="torch"):
    torch.manual_seed(991)
    # Reduced channels/depth are confined to this synthetic reference fixture.
    model = ASGCNEncoder(64 if full else 5, 6 if full else 3, spline_backend=backend).eval()
    if calibrated:
        with torch.no_grad():
            model.reset_activation_maxima()
            _, activations = model.forward_ann(_updates()[0].state.graph, return_activations=True)
            model.update_activation_maxima(activations)
            model.fold_batch_norm()
            model.apply_parameter_normalization()
    return model


def _copy_fields(state):
    return {name: tuple(t.clone() for t in getattr(state, name)) for name in (
        "layer_outputs", "membranes", "previous_spikes", "spike_sums", "local_ticks", "last_pulses"
    )}


def _assert_fields(state, expected):
    for name, tensors in expected.items():
        assert len(getattr(state, name)) == len(tensors)
        for actual, reference in zip(getattr(state, name), tensors, strict=True):
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=2e-6)


@torch.no_grad()
def _dense_local_reference(encoder, update, previous, steps, dynamics):
    """Independent whole-graph affine reference with the SAME local clock rule.

    This slow full-graph implementation is intentionally tests-only. It does
    not call the production incidence index, partial affine or active frontier.
    """
    graph = update.state.graph
    n = len(update.old_indices)
    old = update.old_indices
    keep = old >= 0
    outputs, membranes, previous_spikes, sums, ticks, last = [], [], [], [], [], []
    for i, layer in enumerate(encoder.layers):
        values = {}
        for name in ("layer_outputs", "membranes", "previous_spikes", "spike_sums", "local_ticks", "last_pulses"):
            target = torch.zeros(n, dtype=torch.long) if name == "local_ticks" else torch.zeros(n, layer.out_channels)
            if name == "membranes":
                target[:] = layer.threshold * 0.5
            if previous is not None:
                target[keep] = getattr(previous, name)[i][old[keep]]
            values[name] = target
        outputs.append(values["layer_outputs"])
        membranes.append(values["membranes"])
        previous_spikes.append(values["previous_spikes"])
        sums.append(values["spike_sums"])
        ticks.append(values["local_ticks"])
        last.append(values["last_pulses"])
    basis = encoder._basis_cache(graph)
    source, destination = graph.edge_index
    for _ in range(steps):
        pulse = []
        for i, layer in enumerate(encoder.layers):
            active = update.changed_nodes.clone()
            if i == 0:
                x = graph.node_features
            else:
                x = pulse[i - 1]
                support = x.ne(0).any(1) | last[i - 1].ne(0).any(1)
                active |= support
                active[destination[support[source]]] = True
            # Entire graph/entire node set is deliberately computed here.
            current = layer.affine(x, graph.edge_index, graph.edge_attr, basis, graph.in_degree)
            integrated = membranes[i][active] + current[active]
            if dynamics == "literal_eq15":
                integrated += previous_spikes[i][active]
            spike = torch.where(integrated >= layer.threshold, layer.threshold, 0.)
            membranes[i][active] = integrated - spike
            previous_spikes[i][active] = spike
            sums[i][active] += spike
            ticks[i][active] += 1
            outputs[i][active] = sums[i][active] / ticks[i][active, None]
            emitted = torch.zeros(n, layer.out_channels)
            emitted[active] = spike
            pulse.append(emitted)
        last = pulse
    return StreamEncoderState(
        update.state, outputs[-1], tuple(outputs), tuple(membranes), tuple(previous_spikes),
        tuple(sums), tuple(ticks), tuple(last), "snn", dynamics,
    )


@pytest.mark.parametrize("folded", [False, True])
@pytest.mark.parametrize("backend", ["torch", "torch_fused"])
def test_ann_incremental_matches_full_snapshot_after_arrival_and_expiry(folded, backend):
    encoder = _encoder(backend=backend)
    if folded:
        encoder.fold_batch_norm()
    state = None
    for update in _updates():
        before = None if state is None else _copy_fields(state)
        result = update_encoder(encoder, update, state)
        expected, _ = encoder.forward_ann(update.state.graph)
        torch.testing.assert_close(result.outputs, expected, rtol=1e-5, atol=1e-6)
        if before is not None:
            _assert_fields(state, before)
        state = result
    assert state.work["updated_nodes_per_layer"] == [0, 0, 0]
    assert state.work["message_edges_per_layer"] == [0, 0, 0]
    assert state.work["topology_indexed_edges"] == 0


@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
@pytest.mark.parametrize("steps", [1, 4, 16])
def test_snn_matches_independent_dense_local_clock_reference(dynamics, steps):
    encoder = _encoder(calibrated=True)
    state = reference = None
    for update in _updates():
        before = None if state is None else _copy_fields(state)
        actual = update_encoder(encoder, update, state, mode="snn", simulation_steps=steps, dynamics=dynamics)
        reference = _dense_local_reference(encoder, update, reference, steps, dynamics)
        _assert_fields(actual, _copy_fields(reference))
        if before is not None:
            _assert_fields(state, before)
        state = actual
    assert state.work["global_clock_snapshot_equivalent"] is False


@pytest.mark.parametrize("mode", ["ann", "snn"])
def test_full_six_layer_64_channel_encoder_contract(mode):
    encoder = _encoder(calibrated=mode == "snn", full=True)
    assert len(encoder.layers) == 6 and all(layer.out_channels == 64 for layer in encoder.layers)
    state = update_encoder(encoder, _updates()[0], mode=mode, simulation_steps=4)
    assert state.outputs.shape == (5, 64)
    assert len(state.layer_outputs) == 6
    assert all(torch.isfinite(value).all() for value in state.layer_outputs)


def test_ann_only_affected_incidence_is_projected(monkeypatch):
    encoder = _encoder()
    initial, append, *_ = _updates()
    state = update_encoder(encoder, initial)
    # Neither global encoder nor global layer affine may be an incremental path.
    def forbidden(*args, **kwargs):
        raise AssertionError("full-graph affine is not an incremental implementation")
    monkeypatch.setattr(encoder, "forward_ann", forbidden)
    for layer in encoder.layers:
        monkeypatch.setattr(layer, "affine", forbidden)
    result = update_encoder(encoder, append, state)
    assert result.work["message_edges_per_layer"][0] == 3
    assert result.work["message_edges_per_layer"][0] < append.state.graph.edge_index.shape[1]
    assert result.work["projected_sources_per_layer"][0] == 3
    for before, after in zip(state.layer_outputs, result.layer_outputs, strict=True):
        torch.testing.assert_close(after[3:5], before[3:5], rtol=0, atol=0)


@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
def test_zero_source_edges_keep_full_degree_and_bias_root_and_pulse_end(dynamics):
    encoder = _encoder(calibrated=True)
    # Synthetic explicit weights exercise one persistent pulse and one zero source.
    with torch.no_grad():
        for layer in encoder.layers:
            layer.weight.fill_(0.1)
            layer.root.fill_(0.05)
            layer.bias.fill_(0.2)
            layer.threshold.fill_(1.)
    initial = _update([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], [(0, 2), (1, 2)])
    state = update_encoder(encoder, initial, mode="snn", simulation_steps=1, dynamics=dynamics)
    # Expire the zero-source edge without changing node count; degree changes 2->1.
    changed = _update(initial.state.graph.node_features, [(0, 2)], old=[0, 1, 2], changed=[False, True, True])
    expected = _dense_local_reference(encoder, changed, state, 4, dynamics)
    actual = update_encoder(encoder, changed, state, mode="snn", simulation_steps=4, dynamics=dynamics)
    _assert_fields(actual, _copy_fields(expected))
    advance = replace(changed, changed_nodes=torch.zeros(3, dtype=torch.bool))
    expected = _dense_local_reference(encoder, advance, actual, 1, dynamics)
    result = update_encoder(encoder, advance, actual, mode="snn", simulation_steps=1, dynamics=dynamics)
    _assert_fields(result, _copy_fields(expected))
    assert torch.equal(result.local_ticks[0], actual.local_ticks[0])
    assert torch.count_nonzero(result.last_pulses[0]) == 0


@pytest.mark.parametrize("mode", ["ann", "snn"])
def test_packed_disjoint_streams_match_separate_updates(mode):
    encoder = _encoder(calibrated=mode == "snn")
    features = torch.tensor([[0.3, 0.2, 0.1, 1.], [0.4, 0.1, 0.2, -1.]])
    single = _update(features, [(0, 1), (1, 0)])
    packed = _update(torch.cat((features, features)), [(0, 1), (3, 2), (1, 0), (2, 3)], batch=[0, 0, 1, 1])
    one = update_encoder(encoder, single, mode=mode, simulation_steps=4)
    two = update_encoder(encoder, packed, mode=mode, simulation_steps=4)
    for name, values in _copy_fields(one).items():
        for expected, actual in zip(values, getattr(two, name), strict=True):
            torch.testing.assert_close(actual, torch.cat((expected, expected)), rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("mode", ["ann", "snn"])
def test_all_nodes_expire_and_rearrival_resets_only_new_neurons(mode):
    encoder = _encoder(calibrated=mode == "snn")
    state = update_encoder(encoder, _updates()[0], mode=mode, simulation_steps=4)
    empty = _update([], [], old=[], changed=[])
    # Empty list's inferred dtype is not the index contract.
    empty = replace(empty, old_indices=torch.empty(0, dtype=torch.long), changed_nodes=torch.empty(0, dtype=torch.bool))
    removed = update_encoder(encoder, empty, state, mode=mode, simulation_steps=4)
    assert removed.outputs.shape == (0, 5)
    fresh = update_encoder(encoder, _updates()[0], removed, mode=mode, simulation_steps=4)
    reference = update_encoder(encoder, _updates()[0], mode=mode, simulation_steps=4)
    _assert_fields(fresh, _copy_fields(reference))


def test_retry_same_previous_is_deterministic_without_mutation():
    encoder = _encoder(calibrated=True)
    initial, append, *_ = _updates()
    state = update_encoder(encoder, initial, mode="snn", simulation_steps=4)
    saved = _copy_fields(state)
    a = update_encoder(encoder, append, state, mode="snn", simulation_steps=4)
    b = update_encoder(encoder, append, state, mode="snn", simulation_steps=4)
    _assert_fields(a, _copy_fields(b))
    _assert_fields(state, saved)


@pytest.mark.parametrize("mode", ["ann", "snn"])
def test_cpu_autocast_transition_is_finite_and_preserves_previous_state(mode):
    encoder = _encoder(calibrated=mode == "snn")
    initial, append, *_ = _updates()
    state = update_encoder(encoder, initial, mode=mode, simulation_steps=4)
    saved = _copy_fields(state)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        first = update_encoder(encoder, append, state, mode=mode, simulation_steps=4)
        retry = update_encoder(encoder, append, state, mode=mode, simulation_steps=4)
    assert torch.isfinite(first.outputs).all()
    _assert_fields(first, _copy_fields(retry))
    _assert_fields(state, saved)


def test_explicit_contract_failures():
    update = _updates()[0]
    with pytest.raises(ValueError, match="eval-only"):
        update_encoder(_encoder().train(), update)
    with pytest.raises(RuntimeError, match="calibrated"):
        update_encoder(_encoder(), update, mode="snn")
    with pytest.raises(ValueError, match="positive integer"):
        update_encoder(_encoder(), update, simulation_steps=True)
    with pytest.raises(ValueError, match="every arriving"):
        update_encoder(_encoder(), replace(update, changed_nodes=torch.zeros(5, dtype=torch.bool)))
    with pytest.raises(ValueError, match="require previous"):
        update_encoder(_encoder(), replace(update, old_indices=torch.arange(5)))
    encoder = _encoder(calibrated=True)
    state = update_encoder(encoder, update, mode="snn")
    with pytest.raises(ValueError, match="mode/dynamics"):
        update_encoder(encoder, _updates()[1], state, mode="snn", dynamics="standard_if")
