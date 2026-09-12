"""Full-parameter model parity on tiny CPU fixtures; not training/quality evidence."""

import copy
from dataclasses import fields

import pytest
import torch

from asgcn_unet.batching import pack_samples
from asgcn_unet.implicit_radius import ImplicitRadiusGraph
from asgcn_unet.training import TrainingState
from tests.test_stream_model import model, sample


@pytest.fixture(autouse=True)
def _bounded_cpu_and_explicit_storage(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)

    def no_materialized_attributes(self, name):
        if name in {"edge_index", "edge_attr"}:
            raise AssertionError(f"Implicit model path attempted materialized {name}")
        raise AttributeError(name)

    monkeypatch.setattr(ImplicitRadiusGraph, "__getattr__", no_materialized_attributes, raising=False)
    yield
    torch.set_num_threads(previous)


def _pair(*, calibrated=False, dynamics="literal_eq15"):
    reference = model().eval()
    reference.snn_dynamics = dynamics
    if calibrated:
        reference.fold_batch_norm()
        with torch.no_grad():
            _, details = reference.calibrate_stream_batch([sample(0, "a"), sample(0, "b")])
            reference.calibrate_stream_batch([sample(1, "a"), sample(1, "b")],
                                              [item["recurrent_state"] for item in details])
            reference.apply_parameter_normalization()
    implicit = copy.deepcopy(reference)
    implicit.graph_storage = "implicit_radius"
    assert len(implicit.encoder.layers) == 6 and implicit.encoder.hidden_dim == 64
    assert sum(parameter.numel() for parameter in implicit.parameters()) == 4_409_617
    return implicit, reference


def _assert_graph(actual, expected):
    graph, reference = actual.graph, expected.graph
    assert isinstance(graph, ImplicitRadiusGraph)
    for name in ("node_features", "positions", "in_degree"):
        torch.testing.assert_close(getattr(graph, name), getattr(reference, name), rtol=0, atol=0)
    torch.testing.assert_close(actual.timestamps, expected.timestamps, rtol=0, atol=0)
    torch.testing.assert_close(actual.node_batch, expected.node_batch, rtol=0, atol=0)
    assert graph.edge_count == reference.edge_index.shape[1]


def _assert_cache(actual, expected):
    assert (actual is None) == (expected is None)
    if actual is None:
        return
    for field in fields(actual):
        value, reference = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, reference, rtol=2e-5, atol=2e-6, msg=field.name)
        elif isinstance(value, tuple):
            for item, target in zip(value, reference, strict=True):
                tolerance = 0. if not item.is_floating_point() or field.name in {
                    "previous_spikes", "spike_sums", "last_pulses"} else 2e-6
                torch.testing.assert_close(item, target, rtol=0. if tolerance == 0 else 2e-5,
                                           atol=tolerance, msg=field.name)


def _assert_states(actual, expected):
    _assert_graph(actual.graph, expected.graph)
    _assert_cache(actual.encoder, expected.encoder)
    assert bool(actual.finite()) and bool(expected.finite())
    if actual.decoder is not None:
        torch.testing.assert_close(actual.decoder, expected.decoder, rtol=2e-5, atol=2e-6)
    for name in ("origin_seconds", "watermark_seconds", "sequence_index", "sequence_identity", "last_event_id"):
        assert getattr(actual, name) == getattr(expected, name)
    assert actual.contract != expected.contract  # Never reinterpret old storage provenance.


def test_full_depth_training_loss_every_parameter_gradient_and_optimizer_parity():
    implicit, reference = _pair()
    implicit.train()
    reference.train()
    nets = [implicit, reference]
    states = [TrainingState(independent_sequences=True) for _ in nets]
    optimizers = [torch.optim.SGD(net.parameters(), lr=.001) for net in nets]
    for index in range(2):
        frames = pack_samples([sample(index, "a"), sample(index, "b")])
        outputs, losses, details = [], [], []
        for net, state in zip(nets, states):
            contexts = state.prepare(frames)
            prediction, diagnostic = net.forward_training_batch(frames, [value[0] for value in contexts])
            loss = (prediction - frames.targets).square().mean()
            loss.backward()
            outputs.append(prediction)
            losses.append(loss)
            details.append(diagnostic)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(losses[0], losses[1], rtol=2e-5, atol=2e-6)
        for (name, value), (other_name, expected) in zip(implicit.named_parameters(), reference.named_parameters(), strict=True):
            assert name == other_name
            assert value.grad is not None and expected.grad is not None, name
            assert bool(torch.isfinite(value.grad).all()), name
            torch.testing.assert_close(value.grad, expected.grad, rtol=2e-4, atol=2e-5, msg=name)
        before = [layer.weight.detach().clone() for layer in implicit.encoder.layers]
        for optimizer in optimizers:
            optimizer.step()
        for layer, old in zip(implicit.encoder.layers, before):
            assert not torch.equal(layer.weight, old)
        for (name, value), (_, expected) in zip(implicit.named_parameters(), reference.named_parameters(), strict=True):
            torch.testing.assert_close(value, expected, rtol=2e-5, atol=2e-6, msg=name)
        for actual, expected in zip(details[0], details[1]):
            _assert_states(actual["recurrent_state"], expected["recurrent_state"])
            assert actual["recurrent_state"].encoder is None
        for state, prediction, diagnostic, optimizer in zip(states, outputs, details, optimizers):
            state.commit(frames, prediction, diagnostic, frames.targets)
            optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
@pytest.mark.parametrize("mode,dynamics", [("ann", "literal_eq15"), ("snn", "literal_eq15"), ("snn", "standard_if")])
def test_incremental_full_model_outputs_and_local_clocks_match_across_expiry_and_empty_frames(mode, dynamics):
    implicit, reference = _pair(calibrated=mode == "snn", dynamics=dynamics)
    states = [[None, None], [None, None]]
    for index in range(6):
        # Last step starts a fresh independent lane while its neighbor continues.
        frames = [sample(index, "a"), sample(index, "b", empty=index in {1, 4})]
        if index == 5:
            frames[0] = sample(0, "fresh")
            states[0][0] = states[1][0] = None
        outputs, diagnostics = [], []
        for net, incoming in zip((implicit, reference), states):
            prediction, details = net.forward_batch(frames, incoming, inference_mode=mode, simulation_steps=4)
            outputs.append(prediction)
            diagnostics.append(details)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-5, atol=2e-6)
        for actual, expected in zip(*diagnostics):
            _assert_states(actual["recurrent_state"], expected["recurrent_state"])
            assert actual["nodes"] == expected["nodes"] and actual["edges"] == expected["edges"]
            for key in ("spike_counts", "firing_rate_denominators"):
                for value, target in zip(actual[key], expected[key], strict=True):
                    torch.testing.assert_close(value, target, rtol=0, atol=0, msg=key)
        states = [[detail["recurrent_state"] for detail in result] for result in diagnostics]


@torch.no_grad()
@pytest.mark.parametrize("mode,dynamics", [("ann", "literal_eq15"), ("snn", "literal_eq15"), ("snn", "standard_if")])
def test_implicit_batch_independence_and_cloned_state_replay(mode, dynamics):
    net, _ = _pair(calibrated=mode == "snn", dynamics=dynamics)
    separate = copy.deepcopy(net)
    states, single_states = [None, None], [None, None]
    for index in range(3):
        frames = [sample(index, "a"), sample(index, "b", empty=index == 1)]
        frames[0]["events"] = frames[0]["events"][:2]
        frames[0]["event_ids"] = frames[0]["event_ids"][:2]
        frames[0]["metadata"]["stream_time"]["arrival_group_counts"] = (2,)
        snapshots = [None if state is None else state.clone() for state in states]
        output, details = net.forward_batch(frames, states, inference_mode=mode, simulation_steps=4)
        repeated, _repeated_details = net.forward_batch(frames, snapshots, inference_mode=mode, simulation_steps=4)
        torch.testing.assert_close(output, repeated, rtol=0, atol=0)
        expected = []
        for lane in range(2):
            prediction, detail = separate.forward_sample(frames[lane], recurrent_state=single_states[lane],
                                                          inference_mode=mode, simulation_steps=4)
            single_states[lane] = detail["recurrent_state"]
            _assert_cache(details[lane]["recurrent_state"].encoder, single_states[lane].encoder)
            if states[lane] is not None:
                _assert_cache(states[lane].encoder, snapshots[lane].encoder)
                torch.testing.assert_close(states[lane].decoder, snapshots[lane].decoder, rtol=0, atol=0)
            expected.append(prediction)
        torch.testing.assert_close(output, torch.cat(expected), rtol=2e-5, atol=2e-6)
        states = [detail["recurrent_state"] for detail in details]


@torch.no_grad()
def test_calibration_uses_the_same_full_six_layer_activations_and_conversion():
    implicit, reference = _pair()
    implicit.fold_batch_norm()
    reference.fold_batch_norm()
    states = [[None, None], [None, None]]
    for index in range(2):
        frames = [sample(index, "a"), sample(index, "b")]
        results = [net.calibrate_stream_batch(frames, incoming)[1]
                   for net, incoming in zip((implicit, reference), states)]
        for layer, expected in zip(implicit.encoder.layers, reference.encoder.layers):
            torch.testing.assert_close(layer.calibration_activation_max, expected.calibration_activation_max,
                                       rtol=2e-5, atol=2e-6)
        states = [[detail["recurrent_state"] for detail in result] for result in results]
    implicit.apply_parameter_normalization()
    reference.apply_parameter_normalization()
    for (name, value), (_, expected) in zip(implicit.named_parameters(), reference.named_parameters(), strict=True):
        torch.testing.assert_close(value, expected, rtol=2e-5, atol=2e-6, msg=name)


@torch.no_grad()
def test_model_rejects_reusing_state_from_other_storage_contract():
    implicit, reference = _pair()
    _, implicit_details = implicit.forward_sample(sample())
    _, reference_details = reference.forward_sample(sample())
    for net, wrong in ((implicit, reference_details), (reference, implicit_details)):
        with pytest.raises(ValueError, match="continuity"):
            net.forward_sample(sample(1), recurrent_state=wrong["recurrent_state"])


@torch.no_grad()
def test_static_global_t_encoder_rejects_implicit_local_clock_graph():
    implicit, _ = _pair()
    _, details = implicit.forward_sample(sample())
    with pytest.raises(TypeError, match="static global-T.*local clock"):
        implicit.encoder.forward_snn(details["recurrent_state"].graph.graph, simulation_steps=4)


@torch.no_grad()
@pytest.mark.parametrize("mode,dynamics", [("ann", "literal_eq15"), ("snn", "literal_eq15"), ("snn", "standard_if")])
def test_all_empty_cold_start_all_nodes_expire_and_later_arrivals_match(mode, dynamics):
    implicit, reference = _pair(calibrated=mode == "snn", dynamics=dynamics)
    states = [[None, None], [None, None]]
    for index in range(6):
        frames = [sample(index, sequence, empty=index not in {1, 5}) for sequence in ("a", "b")]
        outputs, results = [], []
        for net, incoming in zip((implicit, reference), states):
            prediction, details = net.forward_batch(frames, incoming, inference_mode=mode, simulation_steps=4)
            outputs.append(prediction)
            results.append(details)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-5, atol=2e-6)
        for actual, expected in zip(*results):
            _assert_states(actual["recurrent_state"], expected["recurrent_state"])
            if index in {0, 4}:
                assert actual["nodes"] == expected["nodes"] == 0
                assert actual["edges"] == expected["edges"] == 0
            elif index in {1, 5}:
                assert actual["nodes"] == expected["nodes"] == 4
        states = [[detail["recurrent_state"] for detail in result] for result in results]
