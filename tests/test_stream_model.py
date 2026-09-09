"""CPU synthetic smoke/contract tests, never real-data performance evidence."""

import copy

import pytest
import torch

from asgcn_unet.batching import pack_samples
from asgcn_unet.model import ASGCNUNet, rasterize_batch
from asgcn_unet.stream_state import StreamingReconstructionState
from asgcn_unet.training import TrainingState


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def model():
    torch.manual_seed(101)
    return ASGCNUNet(architecture_version=3, graph_execution="event_driven", stream_config={
        "window_seconds": 0.025, "time_scale_seconds": 0.1,
        "node_time_feature": "physical_frame_offset", "clock": "event_local_pending_off_v1",
        "arrival_policy": "simultaneous_equal_timestamp",
    }, spline_backend="torch", max_graph_edges=2000000)


def sample(index=0, sequence="first", *, empty=False):
    times = [0.001 + index * 0.01, 0.001 + index * 0.01, 0.003 + index * 0.01,
             0.005 + index * 0.01]
    events = torch.tensor([[5, 5, times[0], 1], [5, 6, times[1], -1],
                           [6, 6, times[2], 1], [7, 6, times[3], -1]], dtype=torch.float64)
    ids = torch.tensor([[index, row] for row in range(4)], dtype=torch.long)
    if empty:
        events, ids = events[:0], ids[:0]
    return {"events": events, "event_ids": ids, "target": torch.full((1, 32, 32), 0.3),
            "sensor_size": (32, 32), "sample_id": f"{sequence}/{index}", "metadata": {
                "scene": sequence, "sequence_id": sequence, "sequence_index": index,
                "dataset_sampling_ratio": 1.0, "stream_time": {
                    "schema": "physical_seconds_v1", "interval_start_seconds": index * 0.01,
                    "interval_end_seconds": (index + 1) * 0.01, "sequence_origin_seconds": 0.0,
                    "arrival_group_counts": () if empty else (2, 1, 1),
                },
            }}


def test_full_six_layer_64_channel_training_gradient_and_optimizer_smoke():
    net = model().train()
    assert len(net.encoder.layers) == 6 and net.encoder.hidden_dim == 64
    assert sum(value.numel() for value in net.parameters()) == 4409617
    state = TrainingState(independent_sequences=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for index in range(2):
        samples = pack_samples([sample(index, "a"), sample(index, "b")])
        context = state.prepare(samples)
        prediction, diagnostics = net.forward_training_batch(samples, [value[0] for value in context])
        loss = (prediction - samples.targets).square().mean()
        loss.backward()
        for layer in net.encoder.layers:
            assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
        assert any(value.grad is not None for value in net.decoder.parameters())
        before = net.encoder.layers[0].weight.detach().clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert not torch.equal(before, net.encoder.layers[0].weight)
        state.commit(samples, prediction, diagnostics, samples.targets)
        assert all(isinstance(value[2], StreamingReconstructionState) for value in state.values.values())
        assert all(value[2].encoder is None for value in state.values.values())
        assert all(len(value[2].graph.timestamps) == 4 * (index + 1) for value in state.values.values())


@torch.no_grad()
def test_incremental_ann_readout_matches_full_graph_and_retains_prior_nodes():
    net = model().eval()
    state = None
    for index in range(5):
        previous_decoder = None if state is None else state.decoder
        prediction, details = net.forward_sample(sample(index), recurrent_state=state)
        state = details["recurrent_state"]
        assert state.encoder is not None
        exact, _ = net.encoder.forward_ann(state.graph.graph)
        torch.testing.assert_close(state.encoder.outputs, exact, atol=1e-6, rtol=1e-5)
        raster = rasterize_batch(exact, state.graph.graph, state.graph.node_batch, 1, (32, 32), 4)
        expected, _ = net.decoder(raster, (32, 32), previous_decoder)
        torch.testing.assert_close(prediction, expected, atol=1e-6, rtol=1e-5)
        assert bool((state.graph.timestamps >= state.watermark_seconds - 0.025).all())
        if index:
            assert len(state.graph.timestamps) > 4


@torch.no_grad()
def test_packed_independent_streams_match_single_streams():
    batched, single = model().eval(), model().eval()
    states, singles = [None, None], [None, None]
    for index in range(3):
        frames = [sample(index, "a"), sample(index, "b", empty=index == 1)]
        prediction, details = batched.forward_batch(pack_samples(frames), states)
        states = [item["recurrent_state"] for item in details]
        expected = []
        for lane in range(2):
            output, detail = single.forward_sample(frames[lane], recurrent_state=singles[lane])
            singles[lane] = detail["recurrent_state"]
            expected.append(output)
        torch.testing.assert_close(prediction, torch.cat(expected), atol=2e-6, rtol=1e-5)


@torch.no_grad()
@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
def test_snn_conversion_and_stateful_batched_reconstruction_smoke(dynamics):
    net = model().eval()
    net.snn_dynamics = dynamics
    net.fold_batch_norm()
    _, data = net.calibrate_stream_batch([sample(0, "a"), sample(0, "b")])
    net.calibrate_stream_batch([sample(1, "a"), sample(1, "b")],
                               [item["recurrent_state"] for item in data])
    net.apply_parameter_normalization()
    prediction, first = net.forward_batch([sample(0, "a"), sample(0, "b")], inference_mode="snn",
                                          simulation_steps=4)
    incoming = [item["recurrent_state"] for item in first]
    cloned = [item.clone() for item in incoming]
    result, second = net.forward_batch([sample(1, "a"), sample(1, "b")], incoming,
                                       inference_mode="snn", simulation_steps=4)
    repeated, _ = net.forward_batch([sample(1, "a"), sample(1, "b")], incoming,
                                    inference_mode="snn", simulation_steps=4)
    assert prediction.shape == result.shape == (2, 1, 32, 32)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, repeated, atol=0, rtol=0)
    for old, saved, new in zip(incoming, cloned, second):
        for membrane, expected in zip(old.encoder.membranes, saved.encoder.membranes):
            torch.testing.assert_close(membrane, expected, atol=0, rtol=0)
        assert len(new["recurrent_state"].graph.timestamps) > len(old.graph.timestamps)


def test_future_window_normalized_input_and_static_state_rejected():
    net = model().eval()
    frame = sample()
    del frame["metadata"]["stream_time"]
    with pytest.raises(ValueError, match="physical_seconds"):
        net.forward_sample(frame)
    with pytest.raises(TypeError, match="Static decoder"):
        net.forward_sample(sample(), recurrent_state=torch.zeros(1, 192, 2, 2))
    with pytest.raises(ValueError, match="stream"):
        net.forward_batch([sample(), sample()])
    future = sample()
    future["events"][-1, 2] = 0.02
    with pytest.raises(ValueError, match="future leakage"):
        net.forward_sample(future)
    wrong_groups = sample()
    wrong_groups["metadata"]["stream_time"]["arrival_group_counts"] = (4,)
    with pytest.raises(ValueError, match="grouping"):
        net.forward_sample(wrong_groups)


def test_graph_requires_causal_context_even_without_decoder_memory():
    from asgcn_unet.engine import _requires_causal_context
    assert _requires_causal_context({"graph_execution": "event_driven", "recurrent": False})
    assert not _requires_causal_context({"recurrent": False})


def test_inference_state_is_not_accepted_after_contract_change():
    net = model().eval()
    with torch.no_grad():
        _, data = net.forward_sample(sample())
        net.stream_config = copy.deepcopy(net.stream_config)
        net.stream_config["window_seconds"] = 0.05
        with pytest.raises(ValueError, match="continuity"):
            net.forward_sample(sample(1), recurrent_state=data["recurrent_state"])


@torch.no_grad()
@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
def test_snn_idle_lane_clock_does_not_advance_on_other_stream_arrivals(dynamics):
    net = model().eval()
    net.snn_dynamics = dynamics
    net.fold_batch_norm()
    net.calibrate_stream_batch([sample(0, "a"), sample(0, "b")])
    net.apply_parameter_normalization()
    separate = copy.deepcopy(net)
    state_batch, state_single = [None, None], [None, None]
    for index in range(3):
        frames = [sample(index, "a"), sample(index, "b", empty=index == 1)]
        if index != 1:
            frames[0]["events"] = frames[0]["events"][:2]
            frames[0]["event_ids"] = frames[0]["event_ids"][:2]
            frames[0]["metadata"]["stream_time"]["arrival_group_counts"] = (2,)
        prediction, details = net.forward_batch(frames, state_batch, inference_mode="snn", simulation_steps=4)
        expected = []
        for lane in range(2):
            value, detail = separate.forward_sample(frames[lane], recurrent_state=state_single[lane],
                                                    inference_mode="snn", simulation_steps=4)
            state_single[lane] = detail["recurrent_state"]
            expected.append(value)
            for actual, target in zip(details[lane]["recurrent_state"].encoder.local_ticks,
                                      state_single[lane].encoder.local_ticks):
                torch.testing.assert_close(actual, target, atol=0, rtol=0)
            for actual, target in zip(details[lane]["spike_counts"], detail["spike_counts"]):
                torch.testing.assert_close(actual, target, atol=0, rtol=0)
        state_batch = [detail["recurrent_state"] for detail in details]
        torch.testing.assert_close(prediction, torch.cat(expected), atol=1e-6, rtol=1e-5)
