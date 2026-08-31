from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

import asgcn_unet.data.eventhdr as eventhdr_module
from asgcn_unet.data.eventhdr import EventHDRDataset


def _make_hdr(root: Path, *, channels: int | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "1.h5"
    height, width = 24, 32
    total = 72
    indices = np.arange(total)
    with h5py.File(path, "w") as h5:
        events = h5.create_group("events")
        events.create_dataset("xs", data=(indices * 7 % width).astype(np.int16))
        events.create_dataset("ys", data=(indices * 5 % height).astype(np.int16))
        events.create_dataset("ts", data=np.linspace(0.0, 0.005, total))
        events.create_dataset("ps", data=np.resize(np.array([-1, 0, 1], np.int8), total))
        images = h5.create_group("images")
        shape = (height, width) if channels is None else (height, width, channels)
        array = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        for index, event_end in enumerate((0, 20, 20, 60, 72)):
            image = images.create_dataset(
                f"image{index:09d}", data=array, compression="gzip"
            )
            image.attrs["event_idx"] = event_end
            image.attrs["timestamp"] = 0.001 * (index + 1)
    return path


def _dataset(root: Path, **kwargs) -> EventHDRDataset:
    return EventHDRDataset(
        root,
        target_normalization={"mode": "integer_dtype_max"},
        file_to_scene={"1.h5": "shared-scene"},
        **kwargs,
    )


def _assert_same_sample(full: dict, topology: dict) -> None:
    assert set(topology) == {"events", "sample_id", "sensor_size", "metadata"}
    assert torch.equal(topology["events"], full["events"])
    assert topology["events"].dtype == torch.float32
    assert topology["events"].is_contiguous()
    assert topology["sample_id"] == full["sample_id"]
    assert topology["sensor_size"] == full["sensor_size"] == tuple(full["target"].shape[-2:])
    assert topology["metadata"] == full["metadata"]


@pytest.mark.parametrize("channels", [None, 1, 3, 4])
@pytest.mark.parametrize("target_channels", [1, 3])
@pytest.mark.parametrize("crop", [None, "center", "random"])
def test_topology_access_matches_full_sample_without_preprocessing_drift(
    tmp_path, channels: int | None, target_channels: int, crop: str | None
) -> None:
    _make_hdr(tmp_path, channels=channels)
    dataset = _dataset(
        tmp_path,
        target_channels=target_channels,
        max_events=7,
        crop_size=[11, 17] if crop else None,
        random_crop=crop == "random",
        seed=197,
    )
    try:
        for index in range(len(dataset)):
            full = dataset[index]
            topology = dataset.get_topology_sample(index)
            _assert_same_sample(full, topology)
            assert len(topology["events"]) <= 7
            assert topology["sample_id"].startswith("shared-scene/1.h5/image")
        # Both naturally empty intervals are retained, never dropped by profiling.
        assert len(dataset.get_topology_sample(0)["events"]) == 0
        assert len(dataset.get_topology_sample(2)["events"]) == 0
        assert dataset.get_topology_sample(2)["metadata"]["zero_event_interval"] is True
    finally:
        dataset.close()


@pytest.mark.parametrize("frame_stride", [1, 2, 3])
@pytest.mark.parametrize("recover_indices", [False, True])
def test_topology_preserves_frame_stride_event_indices_and_timestamps(
    tmp_path, frame_stride: int, recover_indices: bool
) -> None:
    path = _make_hdr(tmp_path)
    if recover_indices:
        with h5py.File(path, "r+") as h5:
            for image in h5["images"].values():
                del image.attrs["event_idx"]
    dataset = _dataset(tmp_path, frame_stride=frame_stride, max_events=None)
    try:
        for index in range(len(dataset)):
            _assert_same_sample(dataset[index], dataset.get_topology_sample(index))
        source = dataset.get_topology_sample(-1)["metadata"]["event_idx_source"]
        assert source == ("timestamp_predecessor_v1" if recover_indices else "stored")
    finally:
        dataset.close()


def test_topology_never_reads_image_arrays_or_runs_target_normalization(
    tmp_path, monkeypatch
) -> None:
    _make_hdr(tmp_path, channels=3)
    dataset = _dataset(tmp_path, max_events=11, crop_size=[12, 16], random_crop=True)
    reference = dataset[3]
    array_reads: list[str] = []
    original_getitem = h5py.Dataset.__getitem__
    original_array = h5py.Dataset.__array__
    original_read_direct = h5py.Dataset.read_direct

    def check_not_image(node) -> None:
        if node.name.startswith("/images/"):
            raise AssertionError(f"Topology access decoded GT pixels: {node.name}")
        array_reads.append(node.name)

    def guarded_getitem(node, *args, **kwargs):
        check_not_image(node)
        return original_getitem(node, *args, **kwargs)

    def guarded_array(node, *args, **kwargs):
        check_not_image(node)
        return original_array(node, *args, **kwargs)

    def guarded_read_direct(node, *args, **kwargs):
        check_not_image(node)
        return original_read_direct(node, *args, **kwargs)

    def no_target_conversion(*_args, **_kwargs):
        raise AssertionError("Topology access called target normalization/tone mapping")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    monkeypatch.setattr(h5py.Dataset, "__array__", guarded_array)
    monkeypatch.setattr(h5py.Dataset, "read_direct", guarded_read_direct)
    monkeypatch.setattr(eventhdr_module, "image_array_to_tensor", no_target_conversion)
    try:
        _assert_same_sample(reference, dataset.get_topology_sample(3))
        assert set(array_reads) == {"/events/xs", "/events/ys", "/events/ts", "/events/ps"}
        with pytest.raises(AssertionError, match="decoded GT pixels"):
            dataset[3]
    finally:
        dataset.close()


@pytest.mark.parametrize("field,value,message", [
    ("xs", 100, "coordinates must lie within"),
    ("ps", 5, "polarity values"),
    ("ts", np.nan, "timestamps must be finite"),
    ("ts", -1.0, "monotonically non-decreasing"),
])
def test_topology_still_validates_every_original_event_value(
    tmp_path, field: str, value: float, message: str
) -> None:
    path = _make_hdr(tmp_path)
    with h5py.File(path, "r+") as h5:
        h5[f"events/{field}"][1] = value
    dataset = _dataset(tmp_path)
    try:
        for getter in (dataset.__getitem__, dataset.get_topology_sample):
            with pytest.raises(ValueError, match=message):
                getter(1)
    finally:
        dataset.close()


@pytest.mark.parametrize("shape", [(10,), (2, 3, 4, 1)])
def test_topology_rejects_invalid_image_dimensions_using_metadata(tmp_path, shape) -> None:
    path = _make_hdr(tmp_path)
    with h5py.File(path, "r+") as h5:
        group = h5["images"]
        attrs = dict(group["image000000001"].attrs)
        del group["image000000001"]
        image = group.create_dataset("image000000001", shape=shape, dtype=np.uint16)
        image.attrs.update(attrs)
    dataset = _dataset(tmp_path)
    try:
        for getter in (dataset.__getitem__, dataset.get_topology_sample):
            with pytest.raises(ValueError, match="Expected HxW or HxWxC"):
                getter(1)
    finally:
        dataset.close()


def test_topology_rejects_insufficient_color_channels(tmp_path) -> None:
    _make_hdr(tmp_path, channels=2)
    dataset = _dataset(tmp_path, target_channels=1)
    try:
        with pytest.raises(IndexError):
            dataset[1]
        with pytest.raises(ValueError, match="at least three channels"):
            dataset.get_topology_sample(1)
    finally:
        dataset.close()


@pytest.mark.parametrize("dtype", [np.bool_, np.float32])
def test_topology_checks_target_dtype_metadata_without_reading_pixels(tmp_path, dtype) -> None:
    path = _make_hdr(tmp_path)
    with h5py.File(path, "r+") as h5:
        group = h5["images"]
        attrs = dict(group["image000000001"].attrs)
        del group["image000000001"]
        image = group.create_dataset("image000000001", shape=(24, 32), dtype=dtype)
        image.attrs.update(attrs)
    dataset = _dataset(tmp_path)
    try:
        error = TypeError if dtype == np.bool_ else ValueError
        for getter in (dataset.__getitem__, dataset.get_topology_sample):
            with pytest.raises(error, match="dtype"):
                getter(1)
    finally:
        dataset.close()


def test_topology_uses_each_image_shape_not_optional_sensor_attributes(tmp_path) -> None:
    path = _make_hdr(tmp_path)
    with h5py.File(path, "r+") as h5:
        h5.attrs["sensor_resolution"] = [999, 999]
        group = h5["images"]
        attrs = dict(group["image000000001"].attrs)
        del group["image000000001"]
        image = group.create_dataset("image000000001", shape=(48, 64), dtype=np.uint16)
        image.attrs.update(attrs)
    dataset = _dataset(tmp_path)
    try:
        _assert_same_sample(dataset[0], dataset.get_topology_sample(0))
        _assert_same_sample(dataset[1], dataset.get_topology_sample(1))
        assert dataset.get_topology_sample(0)["sensor_size"] == (24, 32)
        assert dataset.get_topology_sample(1)["sensor_size"] == (48, 64)
    finally:
        dataset.close()


def test_topology_preserves_existing_integer_crop_size_coercion(tmp_path) -> None:
    _make_hdr(tmp_path)
    dataset = _dataset(tmp_path, crop_size=[11.5, 17])
    try:
        _assert_same_sample(dataset[1], dataset.get_topology_sample(1))
        assert dataset.get_topology_sample(1)["sensor_size"] == (11, 17)
    finally:
        dataset.close()


def test_topology_is_not_misrepresented_as_target_pixel_validation(tmp_path) -> None:
    path = _make_hdr(tmp_path)
    with h5py.File(path, "r+") as h5:
        group = h5["images"]
        attrs = dict(group["image000000001"].attrs)
        del group["image000000001"]
        image = group.create_dataset(
            "image000000001", data=np.full((24, 32), np.nan, dtype=np.float32)
        )
        image.attrs.update(attrs)
    dataset = EventHDRDataset(tmp_path, target_normalization={"mode": "already_normalized"})
    try:
        topology = dataset.get_topology_sample(1)
        assert topology["sensor_size"] == (24, 32)
        assert len(topology["events"]) == 20
        with pytest.raises(ValueError, match="NaN or Inf"):
            dataset[1]
    finally:
        dataset.close()


@pytest.mark.parametrize("tone_map,tone_map_mu,message", [
    ("log", 0, "tone_map_mu must be finite and positive"),
    ("log", float("inf"), "tone_map_mu must be finite and positive"),
    ("invalid", 5000, "Unknown tone_map"),
])
def test_topology_rejects_invalid_target_metadata_configuration(
    tmp_path, tone_map: str, tone_map_mu: float, message: str
) -> None:
    _make_hdr(tmp_path)
    dataset = _dataset(tmp_path, tone_map=tone_map, tone_map_mu=tone_map_mu)
    try:
        for getter in (dataset.__getitem__, dataset.get_topology_sample):
            with pytest.raises(ValueError, match=message):
                getter(1)
    finally:
        dataset.close()
