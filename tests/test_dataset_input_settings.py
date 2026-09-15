"""Small synthetic CPU input-setting regressions; no model or real data runs."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from asgcn_unet.data import EventHDRDataset, build_dataset


def _synthetic_hdr(root):
    with h5py.File(root / "synthetic-settings.h5", "x") as handle:
        events = handle.create_group("events")
        events.create_dataset("xs", data=np.arange(8, dtype=np.int16))
        events.create_dataset("ys", data=np.arange(8, dtype=np.int16))
        events.create_dataset("ts", data=np.arange(8, dtype=np.float64))
        events.create_dataset("ps", data=np.zeros(8, dtype=np.bool_))
        images = handle.create_group("images")
        for index, (timestamp, endpoint) in enumerate(((3, 4), (7, 8))):
            image = images.create_dataset(
                f"image{index:09d}", data=np.zeros((8, 8), dtype=np.uint8),
            )
            image.attrs["timestamp"] = timestamp
            image.attrs["event_idx"] = endpoint


def _config(root, **overrides):
    return {
        "type": "eventhdr", "root": str(root), "max_events": None,
        "event_time_contract": "physical_seconds_v1",
        "timestamp_scale_to_seconds": 1.0,
        "interval_timestamp_scale_to_seconds": 1.0,
        "crop_size": [4, 4], "frame_stride": 1, **overrides,
    }


@pytest.mark.parametrize("split", ["train", "val", "calibration"])
def test_explicit_false_is_preserved_and_actual_crop_is_centered(tmp_path, split):
    _synthetic_hdr(tmp_path)
    config = _config(tmp_path, random_crop=False)
    dataset = build_dataset(config, split=split)
    try:
        assert dataset.random_crop is False
        assert config["random_crop"] is False
        first = dataset.get_topology_sample(0)
        second = dataset.get_topology_sample(1)
        expected_crop = {"left": 2, "top": 2, "width": 4, "height": 4}
        assert first["metadata"]["crop"] == expected_crop
        assert second["metadata"]["crop"] == expected_crop
        assert first["event_ids"].tolist() == [[0, 2], [0, 3]]
        assert second["event_ids"].tolist() == [[0, 4], [0, 5]]
    finally:
        dataset.close()


@pytest.mark.parametrize("split", ["train", "val", "calibration"])
@pytest.mark.parametrize("crop_size", [None, [4, 4]])
def test_unspecified_random_crop_preserves_existing_split_default(tmp_path, split, crop_size):
    _synthetic_hdr(tmp_path)
    config = _config(tmp_path, crop_size=crop_size)
    dataset = build_dataset(config, split=split)
    try:
        assert dataset.random_crop is (split == "train" and crop_size is not None)
        assert "random_crop" not in config
    finally:
        dataset.close()


@pytest.mark.parametrize("split", ["train", "val", "calibration"])
def test_explicit_true_is_not_replaced_by_a_split_default(tmp_path, split):
    _synthetic_hdr(tmp_path)
    dataset = build_dataset(_config(tmp_path, random_crop=True), split=split)
    try:
        assert dataset.random_crop is True
    finally:
        dataset.close()


@pytest.mark.parametrize("stride", [0, -1, 1.5, 2.0, True, False, "2", None, np.int64(2)])
def test_invalid_frame_stride_is_refused_before_dataset_discovery(tmp_path, stride):
    # The directory deliberately has no dataset. A settings error must not be
    # hidden by clamping/coercion or turn into a later missing-data exception.
    with pytest.raises(ValueError, match="frame_stride.*positive integer"):
        EventHDRDataset(tmp_path, frame_stride=stride)


@pytest.mark.parametrize("stride,ends", [(1, [4, 8]), (2, [4])])
def test_valid_frame_stride_keeps_existing_readout_indices(tmp_path, stride, ends):
    _synthetic_hdr(tmp_path)
    dataset = build_dataset(_config(tmp_path, frame_stride=stride, crop_size=None))
    try:
        assert dataset.frame_stride == stride
        assert [sample["end_idx"] for sample in dataset.samples] == ends
        assert dataset.get_topology_sample(0)["sensor_size"] == (8, 8)
    finally:
        dataset.close()
