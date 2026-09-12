"""Full 6x64/base48 parameter CPU synthetic integration, not GPU/quality evidence."""

import copy

import pytest
import torch

from asgcn_unet.batching import pack_samples
from asgcn_unet.hierarchy import forward_snapshot
from asgcn_unet.model import ASGCNUNet, rasterize_batch
from asgcn_unet.training import TrainingState
from tests.test_implicit_stream_model import _assert_cache
from tests.test_stream_model import sample


def model(storage="implicit_radius", factor=1):
    torch.manual_seed(101)
    return ASGCNUNet(
        architecture_version=4, graph_execution="event_driven", graph_storage=storage,
        event_sampling_factor=factor, stream_config={
            "window_seconds": .025, "time_scale_seconds": .1,
            "node_time_feature": "physical_frame_offset", "clock": "event_local_pending_off_v1",
            "arrival_policy": "simultaneous_equal_timestamp"},
        hierarchy_config={"after_layer": 4, "spatial_cell_pixels": 2, "temporal_cell_seconds": .008,
                          "edge_pseudo": "mean_fine_distance_over_radius"})


@pytest.fixture(autouse=True)
def bounded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("storage", ["materialized", "implicit_radius"])
def test_training_full_depth_gradients_and_optimizer_through_pool(storage):
    net = model(storage).train()
    assert len(net.encoder.layers) == 6 and net.encoder.hidden_dim == 64
    assert sum(parameter.numel() for parameter in net.parameters()) == 4_409_617
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    state = TrainingState(independent_sequences=True)
    for index in range(2):
        batch = pack_samples([sample(index, "a"), sample(index, "b")])
        contexts = state.prepare(batch)
        prediction, details = net.forward_training_batch(batch, [item[0] for item in contexts])
        loss = (prediction - batch.targets).square().mean()
        loss.backward()
        for name, parameter in net.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        before = [layer.weight.detach().clone() for layer in net.encoder.layers]
        optimizer.step()
        for layer, previous in zip(net.encoder.layers, before):
            assert not torch.equal(layer.weight, previous)
        assert all(item["recurrent_state"].encoder is None and item["recurrent_state"].hierarchy is None
                   for item in details)
        assert all(item["sampling_offset"] == 4 * (index + 1) for item in details)
        state.commit(batch, prediction, details, batch.targets)
        optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
@pytest.mark.parametrize("storage", ["materialized", "implicit_radius"])
def test_incremental_ann_matches_fresh_hierarchy_at_each_readout(storage):
    net = model(storage).eval()
    state = None
    for index in range(7):
        decoder = None if state is None else state.decoder
        prediction, details = net.forward_sample(sample(index, empty=index in {2, 3, 4}), recurrent_state=state)
        state = details["recurrent_state"]
        expected, _, graph = forward_snapshot(net, state.graph, (32, 32))
        torch.testing.assert_close(state.hierarchy.suffix.outputs, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(state.hierarchy.pool.graph.graph.positions, graph.graph.positions)
        raster = rasterize_batch(expected, graph.graph, graph.node_batch, 1, (32, 32), 4)
        reference, _ = net.decoder(raster, (32, 32), decoder)
        torch.testing.assert_close(prediction, reference, atol=2e-6, rtol=2e-5)


def calibrated(storage, dynamics, factor=1):
    net = model(storage, factor).eval()
    net.snn_dynamics = dynamics
    net.fold_batch_norm()
    states = [None, None]
    for index in range(2):
        _, details = net.calibrate_stream_batch([sample(index, "a"), sample(index, "b")], states)
        states = [item["recurrent_state"] for item in details]
    net.apply_parameter_normalization()
    return net


@torch.no_grad()
@pytest.mark.parametrize("storage", ["materialized", "implicit_radius"])
@pytest.mark.parametrize("mode,dynamics", [("ann", "literal_eq15"), ("snn", "literal_eq15"), ("snn", "standard_if")])
def test_independent_batch_clocks_sampling_clone_and_expiry(storage, mode, dynamics):
    net = calibrated(storage, dynamics, factor=3) if mode == "snn" else model(storage, 3).eval()
    single = copy.deepcopy(net)
    states, singles = [None, None], [None, None]
    for index in range(6):
        frames = [sample(index, "a", empty=index == 3), sample(index, "b", empty=index in {1, 2, 3, 4})]
        snapshots = [None if item is None else item.clone() for item in states]
        result, details = net.forward_batch(frames, states, inference_mode=mode, simulation_steps=4)
        if mode == "snn":
            assert details[0]["stream_execution"]["pooling"]["reused_quotient_updates"] > 0
        repeated, _ = net.forward_batch(frames, snapshots, inference_mode=mode, simulation_steps=4)
        torch.testing.assert_close(result, repeated, atol=0, rtol=0)
        expected = []
        for lane in range(2):
            prediction, detail = single.forward_sample(frames[lane], recurrent_state=singles[lane],
                                                        inference_mode=mode, simulation_steps=4)
            singles[lane] = detail["recurrent_state"]
            actual = details[lane]["recurrent_state"]
            _assert_cache(actual.encoder, singles[lane].encoder)
            _assert_cache(actual.hierarchy.suffix, singles[lane].hierarchy.suffix)
            assert actual.sampling_offset == singles[lane].sampling_offset
            assert bool(actual.finite())
            expected.append(prediction)
        torch.testing.assert_close(result, torch.cat(expected), atol=3e-6, rtol=3e-5)
        states = [item["recurrent_state"] for item in details]


@torch.no_grad()
def test_calibration_observes_six_layers_on_both_sides_of_pool():
    net = model().eval()
    net.fold_batch_norm()
    net.encoder.reset_activation_maxima()
    _, details = net.calibrate_stream_batch([sample(0, "a"), sample(0, "b")])
    activations = details[0]["activations"]
    assert len(activations) == 6
    assert activations[0].shape[0] == 8
    assert activations[4].shape[0] < activations[3].shape[0]
    assert (net.encoder.calibration_samples_seen == 2).all()
    for layer, hidden in zip(net.encoder.layers, activations):
        torch.testing.assert_close(layer.calibration_activation_max, hidden.amax(dim=0))


@torch.no_grad()
@pytest.mark.parametrize("calibration", [False, True])
def test_inference_caches_cannot_leak_into_training_or_calibration(calibration):
    net = model().eval()
    _, details = net.forward_sample(sample())
    state = details["recurrent_state"]
    if calibration:
        with pytest.raises(ValueError, match="raw-only training state"):
            net.calibrate_stream_batch([sample(1)], [state])
    else:
        net.train()
        with pytest.raises(ValueError, match="raw-only training state"):
            net.forward_training_batch([sample(1)], [state])
