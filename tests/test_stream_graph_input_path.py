"""Small synthetic CPU regressions through the real loader/model graph path.

No original dataset, GPU, training run, quality result or paper-equivalence claim.
The model retains its six 64-channel layers; tiny HDF5 inputs are test fixtures.
"""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from asgcn_unet.batching import pack_samples
from asgcn_unet.data.eventhdr import EventHDRDataset
from asgcn_unet.model import ASGCNUNet
from asgcn_unet.stream_geometry import physical_node_positions, prepare_stream_nodes
from asgcn_unet.stream_model import _metadata, _prepared


@pytest.fixture(autouse=True)
def synthetic_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _model(storage, architecture=3):
    options = {} if architecture == 3 else {"hierarchy_config": {
        "after_layer": 4, "spatial_cell_pixels": 4, "temporal_cell_seconds": 0.004,
        "edge_pseudo": "mean_fine_distance_over_radius",
    }}
    return ASGCNUNet(
        architecture_version=architecture, graph_execution="event_driven", graph_storage=storage,
        event_sampling_factor=1, graph_radius=0.3, graph_position_dims=3,
        max_graph_edges=2_000_000, spline_backend="torch", stream_config={
            "window_seconds": 0.5, "time_scale_seconds": 1.0,
            "node_time_feature": "physical_frame_offset", "clock": "event_local_pending_off_v1",
            "arrival_policy": "simultaneous_equal_timestamp",
        }, **options,
    )


def _fixture_h5(path, stored_indices):
    raw = np.array([[1, 1, 9.0, -1], [1, 1, 10.0, 1], [2, 1, 10.5, -1],
                    [2, 1, 10.5, 1], [2, 2, 10.75, 1], [3, 2, 10.875, -1],
                    [3, 3, 11.0, 1], [6, 3, 11.125, -1]], dtype=np.float64)
    with h5py.File(path, "x") as handle:
        events = handle.create_group("events")
        for index, name in enumerate(("xs", "ys", "ts", "ps")):
            events.create_dataset(name, data=raw[:, index])
        for index, (readout, end) in enumerate(((11.0, 5), (11.125, 7), (12.0, 7))):
            image = handle.create_group("images") if index == 0 else handle["images"]
            image = image.create_dataset(f"image{index:09d}", data=np.zeros((32, 32), np.uint8))
            image.attrs["timestamp"] = readout * 2
            if stored_indices:
                image.attrs["event_idx"] = end
    return raw


def _edges(graph, storage):
    if storage == "materialized":
        return graph.edge_index, graph.edge_attr[:, 0]
    chunks = list(graph.index.iter_directed_neighbors())
    if not chunks:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float64)
    return (torch.stack((torch.cat([value[0] for value in chunks]),
                         torch.cat([value[1] for value in chunks]))),
            torch.cat([value[2].flatten() for value in chunks]))


@torch.no_grad()
@pytest.mark.parametrize("storage", ["materialized", "implicit_radius"])
@pytest.mark.parametrize("architecture", [3, 4])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("stored_indices", [False, True])
def test_original_rows_to_actual_model_graph_match_full_independent_oracle(
    tmp_path, storage, architecture, training, stored_indices,
):
    raw = _fixture_h5(tmp_path / "synthetic.h5", stored_indices)
    dataset = EventHDRDataset(tmp_path, max_events=None, crop_size=None, frame_stride=1,
                              event_time_contract="physical_seconds_v1",
                              timestamp_scale_to_seconds=1.0, interval_timestamp_scale_to_seconds=0.5)
    net = _model(storage, architecture).train(training)
    assert len(net.encoder.layers) == 6 and net.encoder.hidden_dim == 64
    state = None
    try:
        for frame_index in range(len(dataset)):
            full, topology = dataset[frame_index], dataset.get_topology_sample(frame_index)
            assert torch.equal(full["events"], topology["events"])
            assert torch.equal(full["event_ids"], topology["event_ids"])
            packed = pack_samples([topology])
            records, _ = _metadata(net, packed, [state])
            model_nodes = _prepared(net, packed, records)
            direct_nodes = prepare_stream_nodes(packed, records, time_scale_seconds=1.0)
            for actual, expected in zip(model_nodes, direct_nodes, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            _, details = net.forward_sample(full, recurrent_state=state)
            state = details["recurrent_state"]
            # Independently reconstruct from authoritative delivered source rows,
            # not the production coordinates, index, sampling or retained mask.
            end = dataset.samples[frame_index]["end_idx"]
            delivered = torch.from_numpy(raw[:end].copy())
            selected = delivered[delivered[:, 2] >= records[0][3] - 0.5]
            expected_positions = torch.stack((selected[:, 0] / 31, selected[:, 1] / 31,
                                               selected[:, 2] - 9.0, (selected[:, 3] + 1) / 2), dim=1)
            torch.testing.assert_close(state.graph.timestamps, selected[:, 2], atol=0, rtol=0)
            torch.testing.assert_close(state.graph.graph.positions, expected_positions, atol=0, rtol=0)
            distance = torch.linalg.vector_norm(
                (expected_positions[:, None, :3] - expected_positions[None, :, :3]) / 0.3, dim=-1)
            adjacency = distance < 1
            adjacency.fill_diagonal_(False)
            expected_edges = adjacency.nonzero().t()
            edges, attributes = _edges(state.graph.graph, storage)
            count = len(selected)
            order = torch.argsort(edges[0] * max(count, 1) + edges[1])
            assert torch.equal(edges[:, order], expected_edges)
            torch.testing.assert_close(attributes[order], distance[adjacency], atol=0, rtol=0)
            assert torch.equal(state.graph.graph.in_degree, adjacency.sum(0))
            assert details["nodes"] == count
    finally:
        dataset.close()


def test_overlapping_frame_interval_rejected_by_actual_model_metadata():
    net = _model("materialized")
    sample = {"events": torch.empty((0, 4), dtype=torch.float64),
              "event_ids": torch.empty((0, 2), dtype=torch.long), "sensor_size": (32, 32),
              "metadata": {"sequence_id": "synthetic", "sequence_index": 0, "stream_time": {
                  "schema": "physical_seconds_v1", "interval_start_seconds": 0.0,
                  "interval_end_seconds": 1.0, "sequence_origin_seconds": 0.0,
                  "arrival_group_counts": (),
              }}}
    with torch.no_grad():
        _, details = net.forward_sample(sample)
    sample["metadata"]["sequence_index"] = 1
    sample["metadata"]["stream_time"].update(interval_start_seconds=0.5, interval_end_seconds=2.0)
    with pytest.raises(ValueError, match="continuity"):
        _metadata(net, pack_samples([sample]), [details["recurrent_state"]])


def test_feature_overflow_refused_before_graph_construction():
    sample = {"events": torch.tensor([[0, 0, 1, 1]], dtype=torch.float64),
              "event_ids": torch.tensor([[0, 0]]), "sensor_size": (32, 32)}
    packed = pack_samples([sample])
    records = [(('synthetic', ''), 0, 0.0, 1.0, 0.0, (1,))]
    # Finite float64 positions but unrepresentable float32 features.
    with pytest.raises(ValueError, match="features overflowed"):
        _prepared(SimpleNamespace(stream_config={"time_scale_seconds": 1e-40}), packed, records)


def test_physical_builder_does_not_accept_already_reduced_clock_precision():
    with pytest.raises(ValueError, match="float64"):
        physical_node_positions(torch.tensor([[0, 0, 1, 1]], dtype=torch.float32), (32, 32),
                                origin_seconds=0.0, time_scale_seconds=1.0)


@torch.no_grad()
@pytest.mark.parametrize("storage", ["materialized", "implicit_radius"])
def test_each_actual_arrival_and_expiration_matches_raw_prefix_oracle(tmp_path, monkeypatch, storage):
    from asgcn_unet import stream_model

    raw = torch.from_numpy(_fixture_h5(tmp_path / "synthetic.h5", True))
    dataset = EventHDRDataset(tmp_path, max_events=None, crop_size=None, frame_stride=1,
                              event_time_contract="physical_seconds_v1",
                              timestamp_scale_to_seconds=1.0, interval_timestamp_scale_to_seconds=0.5)
    net, original_update = _model(storage, 4).eval(), stream_model._update
    cursor, watermark, readout, checked = 0, 9.0, None, 0

    def raw_positions(rows):
        return torch.stack((rows[:, 0] / 31, rows[:, 1] / 31, rows[:, 2] - 9.0,
                            (rows[:, 3] + 1) / 2), dim=1)

    def observed(model, previous, features, positions, timestamps, node_batch, cutoffs):
        nonlocal cursor, watermark, checked
        if len(timestamps):
            arriving = raw[cursor:cursor + len(timestamps)]
            assert torch.equal(timestamps, arriving[:, 2])
            torch.testing.assert_close(positions, raw_positions(arriving), rtol=0, atol=0)
            cursor += len(timestamps)
            watermark = max(watermark, float(arriving[-1, 2]))
        else:
            watermark = readout
        assert torch.equal(cutoffs, torch.tensor([watermark - 0.5], dtype=torch.float64))
        update = original_update(model, previous, features, positions, timestamps, node_batch, cutoffs)
        selected = raw[:cursor][raw[:cursor, 2] >= watermark - 0.5]
        expected = raw_positions(selected)
        torch.testing.assert_close(update.state.graph.positions, expected, rtol=0, atol=0)
        assert torch.equal(update.state.timestamps, selected[:, 2])
        distances = torch.linalg.vector_norm((expected[:, None, :3] - expected[None, :, :3]) / 0.3, dim=-1)
        adjacency = distances < 1
        adjacency.fill_diagonal_(False)
        edges, attributes = _edges(update.state.graph, storage)
        order = torch.argsort(edges[0] * max(len(selected), 1) + edges[1])
        assert torch.equal(edges[:, order], adjacency.nonzero().t())
        torch.testing.assert_close(attributes[order], distances[adjacency], rtol=0, atol=0)
        assert torch.equal(update.state.graph.in_degree, adjacency.sum(0))
        checked += 1
        return update

    monkeypatch.setattr(stream_model, "_update", observed)
    state = None
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            readout = sample["metadata"]["stream_time"]["interval_end_seconds"]
            _, details = net.forward_sample(sample, recurrent_state=state)
            state = details["recurrent_state"]
        assert cursor == 7  # Row7 is a legitimate not-yet-delivered predecessor.
        assert checked == 9  # Six equal-timestamp arrivals plus three readouts.
        assert len(state.graph.timestamps) == 0
    finally:
        dataset.close()
