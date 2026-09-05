from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager

import h5py
import pytest
import torch
from torch import nn
from torch.utils.data import Subset

from asgcn_unet.batching import SequenceBatchSampler, concatenate_graphs, sequence_key
from asgcn_unet.data.eventhdr import EventHDRDataset
from asgcn_unet.graph import EventGraph, build_event_graph
from asgcn_unet.model import ASGCNUNet
from tests.fixtures import make_eventhdr


@pytest.fixture(autouse=True, scope="module")
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class IndexedDataset:
    """Metadata fixture: any attempt to decode a sample is a test failure."""

    def __init__(self, records, crop_size=None):
        self.samples = records
        self.crop_size = crop_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        raise AssertionError("The sampler must not decode events or image pixels")


def _records(lengths, shapes=None):
    records = []
    for stream, length in enumerate(lengths):
        for frame in range(length):
            records.append({
                "scene": f"scene-{stream}",
                "source_file": f"{stream}.h5",
                "sequence_index": frame,
                "sensor_size": (17, 21) if shapes is None else shapes[stream][frame],
            })
    return records


def _assert_schedule(sampler):
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert Counter(index for batch in batches for index in batch) == Counter(
        range(len(sampler.dataset))
    )
    seen = defaultdict(list)
    for batch in batches:
        assert 1 <= len(batch) <= sampler.batch_size
        assert len({sampler._shapes[index] for index in batch}) == 1
        keys = []
        for index in batch:
            base = sampler.dataset
            while isinstance(base, Subset):
                index = base.indices[index]
                base = base.dataset
            record = base.samples[index]
            key = sequence_key(record)
            keys.append(key)
            seen[key].append(record["sequence_index"])
        assert len(keys) == len(set(keys))
    assert all(indices == sorted(set(indices)) for indices in seen.values())


@pytest.mark.parametrize("lengths", [[], [0], [1], [5], [1, 5, 2, 4], [4, 4, 4, 4]])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
def test_sampler_complete_chronological_coverage(lengths, batch_size):
    sampler = SequenceBatchSampler(IndexedDataset(_records(lengths)), batch_size)
    _assert_schedule(sampler)
    first = list(sampler)
    sampler.set_epoch(3)
    assert list(sampler) == first
    if first:
        first[0].clear()
        assert next(iter(sampler))  # Mutating a yielded batch cannot corrupt future epochs.


def test_sampler_refills_lanes_immediately_and_keeps_final_partial_batch():
    sampler = SequenceBatchSampler(IndexedDataset(_records([1, 4, 2])), 2)
    assert list(sampler) == [[0, 1], [5, 2], [6, 3], [4]]


def test_sampler_mixed_geometry_and_within_stream_resize_do_not_duplicate_streams():
    shapes = [
        [(17, 21), (9, 13), (17, 21)],
        [(9, 13)] * 4,
        [(17, 21)] * 2,
    ]
    sampler = SequenceBatchSampler(IndexedDataset(_records([3, 4, 2], shapes)), 3)
    _assert_schedule(sampler)
    # The resized frame in stream zero advances only after its preceding frame.
    assert list(sampler)[:2] == [[0, 7], [1, 3]]


def test_sampler_shuffle_only_admission_order_is_seeded_and_epoch_deterministic():
    dataset = IndexedDataset(_records([4] * 7))
    left = SequenceBatchSampler(dataset, 3, shuffle_sequences=True, seed=29)
    right = SequenceBatchSampler(dataset, 3, shuffle_sequences=True, seed=29)
    before = list(left)
    assert before == list(right)
    left.set_epoch(2)
    right.set_epoch(2)
    assert list(left) == list(right)
    assert list(left) != before
    _assert_schedule(left)


def test_sampler_nested_subset_preserves_indices_and_allows_chronological_gaps():
    dataset = IndexedDataset(_records([5, 5]))
    selected = Subset(Subset(dataset, [0, 1, 2, 3, 5, 6, 7, 9]), [0, 2, 4, 7])
    sampler = SequenceBatchSampler(selected, 2)
    assert list(sampler) == [[0, 2], [1, 3]]
    assert sampler.final_sequence_indices == {("scene-0", "0.h5"): 2,
                                               ("scene-1", "1.h5"): 4}
    with pytest.raises(TypeError):
        sampler.final_sequence_indices[("scene-0", "0.h5")] = 99
    _assert_schedule(sampler)


@pytest.mark.parametrize("indices", [[0, 0], [2, 1]])
def test_sampler_rejects_duplicate_or_reverse_selected_frames(indices):
    with pytest.raises(ValueError, match="duplicate|chronological"):
        SequenceBatchSampler(Subset(IndexedDataset(_records([4])), indices), 2)


def test_sequence_keys_distinguish_same_scene_different_source():
    records = _records([2, 2])
    for record in records:
        record["scene"] = "shared-scene"
    sampler = SequenceBatchSampler(IndexedDataset(records), 2)
    assert list(sampler) == [[0, 2], [1, 3]]
    assert sequence_key({"metadata": records[0]}) == ("shared-scene", "0.h5")
    assert sequence_key({"scene": "old", "sequence_id": "new"}) == ("new", "")


@pytest.mark.parametrize("batch_size", [0, -1, True, 2.5, "2"])
def test_sampler_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        SequenceBatchSampler(IndexedDataset([]), batch_size)


@pytest.mark.parametrize("record", [
    {"sequence_index": 0, "sensor_size": (8, 8)},
    {"scene": "a", "sequence_index": -1, "sensor_size": (8, 8)},
    {"scene": "a", "sequence_index": True, "sensor_size": (8, 8)},
    {"scene": "a", "sequence_index": 0, "sensor_size": (0, 8)},
    {"scene": "a", "source_file": 4, "sequence_index": 0, "sensor_size": (8, 8)},
])
def test_sampler_rejects_malformed_metadata(record):
    with pytest.raises((TypeError, ValueError)):
        SequenceBatchSampler(IndexedDataset([record]), 2)


@pytest.mark.parametrize("crop", [None, (11, 17), (100, 100)])
def test_sampler_eventhdr_uses_shape_metadata_without_image_decode(tmp_path, monkeypatch, crop):
    make_eventhdr(tmp_path, frames=5)
    dataset = EventHDRDataset(
        tmp_path, crop_size=crop, target_normalization={"mode": "integer_dtype_max"}
    )
    expected = [dataset[index]["sensor_size"] for index in range(len(dataset))]
    original = h5py.Dataset.__getitem__

    def guarded_getitem(node, key):
        if node.name.startswith("/images/"):
            raise AssertionError("GT pixel access during sampler construction")
        return original(node, key)

    def no_array(*args, **kwargs):
        raise AssertionError("No dataset array conversion during sampler construction")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    monkeypatch.setattr(h5py.Dataset, "__array__", no_array)
    monkeypatch.setattr(h5py.Dataset, "read_direct", no_array)
    try:
        sampler = SequenceBatchSampler(dataset, 3)
        assert sampler._shapes == tuple(expected)
        assert sampler.sample_sensor_sizes == tuple(expected)
        assert sampler.sample_sensor_sizes is sampler.sample_sensor_sizes
        assert list(sampler) == [[index] for index in range(len(dataset))]
    finally:
        dataset.close()


def _sample(name, *, count=7, sensor_size=(17, 21), requires_grad=False):
    height, width = sensor_size
    events = torch.tensor([
        [float((index * 3 + 1) % width), float((index * 2 + 1) % height),
         index * 0.001, float(index % 2)]
        for index in range(count)
    ], dtype=torch.float32).reshape(-1, 4).requires_grad_(requires_grad)
    return {
        "events": events,
        "sensor_size": sensor_size,
        "sample_id": f"{name}/frame0",
        "metadata": {"scene": name, "source_file": f"{name}.h5", "sequence_index": 0},
    }


def _model(*, recurrent=True):
    torch.manual_seed(79)
    return ASGCNUNet(
        hidden_dim=8, graph_layers=2, graph_radius=2.0,
        graph_chunk_size=5, spline_chunk_size=13,
        raster_downsample=2, decoder_channels=4, recurrent=recurrent,
    )


def _make_graph(sample):
    return build_event_graph(
        sample["events"], sample["sensor_size"], graph_radius=2.0,
        event_sampling_factor=1, graph_position_dims=3, graph_chunk_size=5,
    )


@pytest.mark.parametrize("counts", [[3, 5], [0, 3, 0, 1], [0, 0]])
def test_disjoint_union_preserves_every_local_edge_attribute_degree_and_gradient(counts):
    graphs = [_make_graph(_sample(str(i), count=count)) for i, count in enumerate(counts)]
    for graph in graphs:
        graph.node_features = graph.node_features.detach().requires_grad_()
        graph.edge_attr = graph.edge_attr.detach().requires_grad_()
    union = concatenate_graphs(graphs)
    offset = edge_start = 0
    membership = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    for graph in graphs:
        nodes, edges = len(graph.node_features), graph.edge_index.shape[1]
        torch.testing.assert_close(
            union.edge_index[:, edge_start:edge_start + edges], graph.edge_index + offset
        )
        torch.testing.assert_close(union.edge_attr[edge_start:edge_start + edges], graph.edge_attr)
        torch.testing.assert_close(union.in_degree[offset:offset + nodes], graph.in_degree)
        offset += nodes
        edge_start += edges
    assert torch.equal(membership[union.edge_index[0]], membership[union.edge_index[1]])
    (union.node_features.sum() + union.edge_attr.sum()).backward()
    for graph in graphs:
        torch.testing.assert_close(graph.node_features.grad, torch.ones_like(graph.node_features))
        torch.testing.assert_close(graph.edge_attr.grad, torch.ones_like(graph.edge_attr))


def test_disjoint_union_rejects_missing_graphs_and_mixed_dtype():
    with pytest.raises(ValueError, match="empty graph"):
        concatenate_graphs([])
    graph = _make_graph(_sample("a"))
    different = EventGraph(
        graph.node_features.double(), graph.positions.double(),
        graph.edge_index, graph.edge_attr.double(), graph.in_degree,
    )
    with pytest.raises(ValueError, match="dtype"):
        concatenate_graphs([graph, different])


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("counts", [[0], [1], [0, 1, 7], [5, 8, 3], [0, 0]])
def test_batch_eval_matches_independent_samples_and_diagnostics(recurrent, counts):
    model = _model(recurrent=recurrent).eval()
    samples = [_sample(str(i), count=count) for i, count in enumerate(counts)]
    with torch.no_grad():
        individual = [model.forward_sample(sample) for sample in samples]
        prediction, diagnostics = model.forward_training_batch(samples)
    assert prediction.shape == (len(samples), 1, 17, 21)
    torch.testing.assert_close(prediction, torch.cat([result[0] for result in individual]),
                               rtol=2e-5, atol=2e-6)
    for actual, (_, expected) in zip(diagnostics, individual, strict=True):
        assert actual.keys() == expected.keys()
        for key in actual:
            if key == "edges":
                # Packed topology retains device counts; compare exact integer
                # values without requiring a per-frame device-to-host conversion.
                torch.testing.assert_close(
                    torch.as_tensor(actual[key]), torch.as_tensor(expected[key]), rtol=0, atol=0
                )
            elif isinstance(actual[key], torch.Tensor):
                torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-6)
            else:
                assert actual[key] == expected[key]


def test_training_batch_calls_encoder_and_decoder_once_and_pools_batchnorm(monkeypatch):
    model = _model().train()
    samples = [_sample("a", count=5, requires_grad=True),
               _sample("b", count=8, requires_grad=True)]
    calls = Counter()
    original_ann = model.encoder.forward_ann
    captured = {}

    def encoder(graph, *args, **kwargs):
        calls["encoder"] += 1
        captured["graph"] = graph
        return original_ann(graph, *args, **kwargs)

    def decoder_hook(_module, args):
        calls["decoder"] += 1
        assert args[0].shape[0] == 2

    def no_sample(*args, **kwargs):
        raise AssertionError("The batch path must not call forward_sample")

    monkeypatch.setattr(model.encoder, "forward_ann", encoder)
    monkeypatch.setattr(model, "forward_sample", no_sample)
    hook = model.decoder.register_forward_pre_hook(decoder_hook)
    try:
        prediction, diagnostics = model.forward_training_batch(samples)
        prediction.square().mean().backward()
    finally:
        hook.remove()
    assert calls == {"encoder": 1, "decoder": 1}
    assert len(captured["graph"].node_features) == 13
    for sample in samples:
        gradient = sample["events"].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    for module in (model.encoder, model.decoder):
        grads = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert sum(grad.abs().sum() for grad in grads) > 0
    assert all(norm.num_batches_tracked.item() == 1 for norm in model.encoder.modules()
               if isinstance(norm, nn.BatchNorm1d))
    assert [detail["nodes"] for detail in diagnostics] == [5, 8]


def test_batch_one_training_and_gradients_preserve_existing_single_sample_behavior():
    baseline, batched = _model().train(), _model().train()
    left, right = _sample("a", requires_grad=True), _sample("a", requires_grad=True)
    individual, _ = baseline.forward_sample(left)
    prediction, _ = batched.forward_training_batch([right])
    torch.testing.assert_close(prediction, individual, rtol=0, atol=0)
    individual.square().mean().backward()
    prediction.square().mean().backward()
    torch.testing.assert_close(right["events"].grad, left["events"].grad, rtol=0, atol=0)
    for expected, actual in zip(baseline.parameters(), batched.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


def test_recurrent_state_follows_sample_order_and_none_is_per_lane_zero():
    model = _model().eval()
    samples = [_sample(name, count=count) for name, count in [("a", 5), ("b", 8), ("c", 3)]]
    with torch.no_grad():
        previous = [model.forward_sample(sample)[1]["recurrent_state"] for sample in samples]
        # Reorder lanes, admit a fresh stream, and carry the other two states.
        current = [samples[2], _sample("new"), samples[0]]
        states = [previous[2], None, previous[0]]
        expected = [model.forward_sample(sample, recurrent_state=state)
                    for sample, state in zip(current, states, strict=True)]
        prediction, details = model.forward_training_batch(current, states)
        torch.testing.assert_close(prediction, torch.cat([item[0] for item in expected]),
                                   rtol=2e-5, atol=2e-6)
        for actual, (_, individual) in zip(details, expected, strict=True):
            torch.testing.assert_close(actual["recurrent_state"], individual["recurrent_state"],
                                       rtol=2e-5, atol=2e-6)
        # Both state and predictions remain routed correctly for the next batch.
        carried = [detail["recurrent_state"] for detail in details]
        predicted_next, _ = model.forward_training_batch(current, carried)
        expected_next = [model.forward_sample(sample, recurrent_state=state)[0]
                         for sample, state in zip(current, carried, strict=True)]
        torch.testing.assert_close(predicted_next, torch.cat(expected_next),
                                   rtol=2e-5, atol=2e-6)


def test_wrong_shape_state_resets_like_existing_decoder():
    model = _model().eval()
    samples = [_sample("a"), _sample("b")]
    with torch.no_grad():
        _, detail = model.forward_sample(samples[0])
        states = [detail["recurrent_state"], torch.ones(1, 16, 1, 1)]
        expected = [model.forward_sample(sample, recurrent_state=state)[0]
                    for sample, state in zip(samples, states, strict=True)]
        actual, _ = model.forward_training_batch(samples, states)
    torch.testing.assert_close(actual, torch.cat(expected), rtol=2e-5, atol=2e-6)


def test_recurrent_state_gradients_do_not_cross_batch_lanes():
    model = _model().eval()
    samples = [_sample("a"), _sample("b")]
    with torch.no_grad():
        states = [model.forward_sample(sample)[1]["recurrent_state"] for sample in samples]
    states = [state.detach().requires_grad_() for state in states]
    prediction, _ = model.forward_training_batch(samples, states)
    prediction[0].square().mean().backward()
    assert states[0].grad is not None and states[0].grad.abs().sum() > 0
    assert states[1].grad is not None and torch.count_nonzero(states[1].grad) == 0


def test_autocast_recurrent_mixed_none_state_remains_finite_and_matches_single_samples():
    model = _model().eval()
    samples = [_sample("a"), _sample("b")]
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        previous = model.forward_sample(samples[0])[1]["recurrent_state"]
        states = [previous, None]
        expected = [model.forward_sample(sample, recurrent_state=state)[0]
                    for sample, state in zip(samples, states, strict=True)]
        actual, details = model.forward_training_batch(samples, states)
    assert torch.isfinite(actual).all()
    assert all(torch.isfinite(detail["recurrent_state"]).all() for detail in details)
    torch.testing.assert_close(actual, torch.cat(expected), rtol=0.02, atol=0.005)


def test_all_empty_training_batch_still_updates_decoder_without_changing_bn_statistics():
    model = _model().train()
    prediction, diagnostics = model.forward_training_batch([
        _sample("a", count=0), _sample("b", count=0)
    ])
    prediction.square().mean().backward()
    assert torch.isfinite(prediction).all()
    assert all(detail["nodes"] == detail["edges"] == 0 for detail in diagnostics)
    assert model.decoder.head.weight.grad.abs().sum() > 0
    assert all(norm.num_batches_tracked.item() == 0 for norm in model.encoder.modules()
               if isinstance(norm, nn.BatchNorm1d))


def test_batch_input_validation():
    model = _model()
    with pytest.raises(ValueError, match="at least one"):
        model.forward_training_batch([])
    with pytest.raises(ValueError, match="sensor_size"):
        model.forward_training_batch([_sample("a"), _sample("b", sensor_size=(18, 21))])
    with pytest.raises(ValueError, match="one frame"):
        model.forward_training_batch([_sample("a"), _sample("a")])
    with pytest.raises(ValueError, match="one entry"):
        model.forward_training_batch([_sample("a")], [])
    with pytest.raises(TypeError, match="tensor or None"):
        model.forward_training_batch([_sample("a")], ["state"])


def test_batch_timing_scopes_are_nonoverlapping_operator_stages():
    visits = []

    class Timer:
        @contextmanager
        def scope(self, label, *, gpu):
            assert gpu is False
            visits.append((label, "start"))
            yield
            visits.append((label, "end"))

    _model().eval().forward_training_batch([_sample("a"), _sample("b")], timing=Timer())
    assert visits == [(stage, event) for stage in ("graph", "encoder", "decoder")
                      for event in ("start", "end")]


@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
@pytest.mark.parametrize("steps", [4, 8, 16, 32])
def test_public_snn_batch_matches_independent_predictions_and_statistics(dynamics, steps):
    model = _model().eval()
    model.snn_dynamics = dynamics
    samples = [_sample("a", count=5), _sample("empty", count=0), _sample("b", count=8)]
    model.reset_activation_maxima()
    model.calibrate_batch(samples)
    model.fold_batch_norm()
    model.apply_parameter_normalization()
    with torch.no_grad():
        expected = [model.forward_sample(sample, "snn", steps) for sample in samples]
        predictions, diagnostics = model(samples, "snn", steps)
    assert len(predictions) == len(samples)
    for actual, detail, (reference, reference_detail) in zip(
        predictions, diagnostics, expected, strict=True
    ):
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
        for key in ("nodes", "edges", "isolated_nodes", "max_degree"):
            torch.testing.assert_close(
                torch.as_tensor(detail[key]), torch.as_tensor(reference_detail[key]),
                rtol=0, atol=0,
            )
        for key in ("firing_rates", "spike_counts", "firing_rate_denominators"):
            torch.testing.assert_close(detail[key], reference_detail[key], rtol=0, atol=0)
        torch.testing.assert_close(
            detail["recurrent_state"], reference_detail["recurrent_state"], rtol=2e-5, atol=2e-6
        )


def test_calibration_batch_counts_empty_attempts_and_preserves_observed_maxima():
    single = _model().eval()
    batch = _model().eval()
    samples = [_sample("a", count=5), _sample("empty", count=0), _sample("b", count=8)]
    single.reset_activation_maxima()
    batch.reset_activation_maxima()
    for sample in samples:
        single.calibrate_sample(sample)
    batch.calibrate_batch(samples)
    assert single.calibration_summary() == batch.calibration_summary()
    assert batch.calibration_summary()["attempted_samples"] == 3
    assert batch.calibration_summary()["valid_samples_per_layer"] == [2, 2]
    for expected, actual in zip(single.encoder.layers, batch.encoder.layers, strict=True):
        torch.testing.assert_close(
            actual.calibration_activation_max, expected.calibration_activation_max,
            rtol=2e-5, atol=2e-6,
        )


def test_single_frame_batch_uses_all_three_timing_scopes():
    visits = []

    class Timer:
        @contextmanager
        def scope(self, label, *, gpu):
            visits.append((label, "start"))
            yield
            visits.append((label, "end"))

    _model().eval().forward_batch([_sample("a")], timing=Timer())
    assert visits == [(stage, event) for stage in ("graph", "encoder", "decoder")
                      for event in ("start", "end")]
