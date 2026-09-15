"""Synthetic CPU timestamp precision regressions; no real data/model/GPU run."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from asgcn_unet.data import EventHDRDataset, eventhdr
from asgcn_unet.stream_input import (
    PHYSICAL_EVENT_TIME_CONTRACT,
    hdr_boundary_policy,
    to_physical_seconds,
)


def _physical(scale=1.0):
    return {
        "event_time_contract": PHYSICAL_EVENT_TIME_CONTRACT,
        "max_events": None,
        "timestamp_scale_to_seconds": scale,
        "interval_timestamp_scale_to_seconds": scale,
    }


def _hdr(root: Path, timestamps: np.ndarray, boundaries, *, image_shapes=None):
    """Write a small fixture preserving its explicit source timestamp dtype."""
    path = root / "synthetic-clock.h5"
    with h5py.File(path, "x") as handle:
        events = handle.create_group("events")
        events.create_dataset("ts", data=timestamps)
        events.create_dataset("xs", data=np.arange(len(timestamps), dtype=np.int16) % 8)
        events.create_dataset("ys", data=np.zeros(len(timestamps), dtype=np.int16))
        events.create_dataset("ps", data=np.zeros(len(timestamps), dtype=np.bool_))
        images = handle.create_group("images")
        for index, (timestamp, endpoint) in enumerate(boundaries):
            shape = image_shapes[index] if image_shapes is not None else (8, 8)
            image = images.create_dataset(
                f"image{index:09d}", data=np.zeros(shape, dtype=np.uint8),
            )
            image.attrs["timestamp"] = timestamp
            if endpoint is not None:
                image.attrs["event_idx"] = endpoint
    return path


@pytest.mark.parametrize("values", [
    np.array([2**53, 2**53 + 1], dtype=np.int64),
    np.array([-2**53 - 1, -2**53], dtype=np.int64),
    np.array([2**64 - 2, 2**64 - 1], dtype=np.uint64),
    np.array([float(2**53), 2**53 + 1], dtype=object),
])
def test_raw_values_are_compared_before_float64_cast(values):
    with pytest.raises(ValueError, match="loses timestamp resolution"):
        to_physical_seconds(values, 1.0, source="synthetic raw clock")


@pytest.mark.parametrize("container", [list, tuple])
def test_native_mixed_python_container_cannot_coerce_away_integer_distinction(container):
    values = container([float(2**53), 2**53 + 1])
    with pytest.raises(ValueError, match="loses timestamp resolution"):
        to_physical_seconds(values, 1.0, source="synthetic mixed Python clock")


@pytest.mark.parametrize("container", [list, tuple])
def test_valid_native_mixed_python_container_keeps_float64_output(container):
    values = container([0, 1.0, 2, 2.5])
    actual = to_physical_seconds(values, 1e-6, source="synthetic mixed Python clock")
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, np.asarray(values, dtype=np.float64) * 1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.int64, np.uint64])
def test_valid_source_values_keep_existing_float64_results(dtype):
    original = np.array([0, 1, 1, 10, 100], dtype=dtype)
    expected = np.asarray(original, dtype=np.float64) * 1e-6
    actual = to_physical_seconds(original, 1e-6, source="synthetic exact clock")
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)


def test_equal_large_integer_timestamps_remain_equal_not_deduplicated():
    original = np.array([2**53 + 1, 2**53 + 1, 2**53 + 4], dtype=np.int64)
    actual = to_physical_seconds(original, 1.0, source="synthetic duplicate clock")
    np.testing.assert_array_equal(actual, original.astype(np.float64))
    assert len(actual) == len(original)


def test_boundary_validation_cannot_hide_original_integer_collision():
    with pytest.raises(ValueError, match="loses timestamp resolution"):
        hdr_boundary_policy(
            np.array([2**53, 2**53 + 1, 2**53 + 4], dtype=np.int64),
            1, 2**53 + 4, source="synthetic boundary",
        )


@pytest.mark.parametrize("topology_only", [False, True])
def test_hdr_payload_rejects_interior_collision_before_model_or_crop(tmp_path, topology_only):
    # Stored boundary checks inspect only endpoint neighbors. The corrupt pair
    # is inside the delivered block, so actual payload conversion must catch it.
    raw = np.array([0, 2**53, 2**53 + 1, 2**53 + 4, 2**53 + 8], dtype=np.int64)
    path = _hdr(tmp_path, raw, [(2**53 + 10, len(raw))])
    original_bytes = path.read_bytes()
    dataset = EventHDRDataset(tmp_path, **_physical(), crop_size=(2, 2))
    try:
        with pytest.raises(ValueError, match="loses timestamp resolution"):
            if topology_only:
                dataset.get_topology_sample(0)
            else:
                dataset[0]
    finally:
        dataset.close()
    assert path.read_bytes() == original_bytes


def test_recovered_index_checks_integer_collision_across_chunk_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(eventhdr, "_TIMESTAMP_CHUNK_SIZE", 2)
    path = _hdr(
        tmp_path, np.array([0, 2**53, 2**53 + 1, 2**53 + 4], dtype=np.int64),
        [(2**53 + 4, None)],
    )
    original_bytes = path.read_bytes()
    with pytest.raises(ValueError, match="loses timestamp resolution"):
        EventHDRDataset(tmp_path, **_physical())
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("stored", [False, True])
@pytest.mark.parametrize("first_boundary", [2**53, float(2**53)])
def test_hdr_frame_attributes_are_checked_before_integer_cast_loss(
    tmp_path, stored, first_boundary,
):
    endpoint = 0 if stored else None
    _hdr(tmp_path, np.array([], dtype=np.int64),
         [(first_boundary, endpoint), (2**53 + 1, endpoint)])
    with pytest.raises(ValueError, match="loses timestamp resolution"):
        EventHDRDataset(tmp_path, **_physical())


@pytest.mark.parametrize("dtype", [np.int16, np.int64, np.float64])
def test_hdr_valid_source_dtype_keeps_float64_samples_ids_and_boundary_policy(tmp_path, dtype):
    raw = np.array([0, 1, 1, 2, 3, 4], dtype=dtype)
    _hdr(tmp_path, raw, [(2, 3), (4, 6)])
    dataset = EventHDRDataset(tmp_path, **_physical(1e-6))
    try:
        for index, row_ids in enumerate(([0, 1, 2], [3, 4, 5])):
            sample = dataset[index]
            expected = torch.from_numpy(raw[row_ids].astype(np.float64) * 1e-6)
            assert sample["events"].dtype == torch.float64
            assert torch.equal(sample["events"][:, 2], expected)
            assert sample["event_ids"].tolist() == [[0, row] for row in row_ids]
        assert dataset.samples[0]["end_idx"] == 3
        assert dataset.samples[1]["end_idx"] == 6
    finally:
        dataset.close()


def _forbid_image_pixel_reads(monkeypatch):
    """Index validation may read image metadata but never image payloads."""
    def guard(original):
        def checked(dataset, *args, **kwargs):
            if dataset.name.startswith("/images/"):
                pytest.fail("Physical index validation must not decode image pixels")
            return original(dataset, *args, **kwargs)
        return checked

    for name in ("__getitem__", "__array__", "read_direct"):
        monkeypatch.setattr(h5py.Dataset, name, guard(getattr(h5py.Dataset, name)))


@pytest.mark.parametrize("second_shape", [(9, 8), (8, 9)])
@pytest.mark.parametrize("frame_stride", [1, 2])
def test_physical_sequence_sensor_change_is_rejected_from_metadata_before_graph(
    tmp_path, monkeypatch, second_shape, frame_stride,
):
    path = _hdr(tmp_path, np.array([0, 1, 2, 3], dtype=np.float64), [(2, 2), (3, 4)],
                image_shapes=[(8, 8), second_shape])
    original_bytes = path.read_bytes()
    _forbid_image_pixel_reads(monkeypatch)
    with pytest.raises(ValueError, match="requires fixed sensor_size"):
        EventHDRDataset(tmp_path, **_physical(), frame_stride=frame_stride)
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("second_shape", [(8, 8), (8, 8, 1), (8, 8, 3)])
@pytest.mark.parametrize("target_channels", [1, 3])
def test_same_physical_sensor_allows_existing_hw_hwc_layouts_without_pixel_read(
    tmp_path, monkeypatch, second_shape, target_channels,
):
    _hdr(tmp_path, np.array([0, 1, 2, 3], dtype=np.float64), [(2, 2), (3, 4)],
         image_shapes=[(8, 8), second_shape])
    _forbid_image_pixel_reads(monkeypatch)
    dataset = EventHDRDataset(tmp_path, **_physical(), target_channels=target_channels)
    try:
        assert len(dataset) == 2
        assert dataset.get_topology_sample(0)["sensor_size"] == (8, 8)
        assert dataset.get_topology_sample(1)["sensor_size"] == (8, 8)
    finally:
        dataset.close()


def test_legacy_frame_input_still_allows_variable_sensor_sizes(tmp_path):
    _hdr(tmp_path, np.array([0, 1, 2, 3], dtype=np.float64), [(2, 2), (3, 4)],
         image_shapes=[(8, 8), (9, 8)])
    dataset = EventHDRDataset(tmp_path, max_events=None)
    try:
        assert dataset[0]["sensor_size"] == (8, 8)
        assert dataset[1]["sensor_size"] == (9, 8)
    finally:
        dataset.close()
