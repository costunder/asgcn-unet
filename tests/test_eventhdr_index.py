from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from asgcn_unet.cli import _inspect_one_split
from asgcn_unet.data import EventHDRDataset, eventhdr
from tests.fixtures import make_eventhdr

_DERIVED = "timestamp_predecessor_v1"
_POLICY = "stored_or_timestamp_predecessor_v1"


def _make_index_file(
    path: Path,
    event_timestamps: list[float],
    frame_timestamps: list[float],
    stored_indices: list[int | float | None] | None = None,
) -> Path:
    """Write only the arrays and frame timestamps, without optional root metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if stored_indices is None:
        stored_indices = [None] * len(frame_timestamps)
    assert len(stored_indices) == len(frame_timestamps)
    with h5py.File(path, "w") as h5:
        events = h5.create_group("events")
        count = len(event_timestamps)
        events.create_dataset("ts", data=np.asarray(event_timestamps, dtype=np.float64))
        events.create_dataset("xs", data=np.arange(count, dtype=np.int16) % 8)
        events.create_dataset("ys", data=np.zeros(count, dtype=np.int16))
        events.create_dataset("ps", data=np.ones(count, dtype=np.uint8))
        images = h5.create_group("images")
        for index, (timestamp, event_idx) in enumerate(zip(frame_timestamps, stored_indices)):
            node = images.create_dataset(
                f"image{index:09d}", data=np.full((8, 8), index + 1, dtype=np.uint16)
            )
            node.attrs["timestamp"] = timestamp
            if event_idx is not None:
                node.attrs["event_idx"] = event_idx
    return path


def _assert_sample_boundaries(
    dataset: EventHDRDataset, expected: list[int], sources: list[str]
) -> None:
    assert len(dataset) == len(expected) == len(sources)
    for index, (end, source) in enumerate(zip(expected, sources)):
        start = expected[index - 1] if index else 0
        assert dataset.samples[index]["start_idx"] == start
        assert dataset.samples[index]["end_idx"] == end
        sample = dataset[index]
        metadata = sample["metadata"]
        assert metadata["event_start_idx"] == start
        assert metadata["event_end_idx"] == end
        assert metadata["event_idx_source"] == source
        assert metadata["raw_event_count"] == end - start
        assert metadata["zero_event_interval"] is (start == end)
        assert len(sample["events"]) == end - start


def test_missing_indices_use_predecessor_with_duplicates_across_chunk_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(eventhdr, "_TIMESTAMP_CHUNK_SIZE", 3)
    path = _make_index_file(
        tmp_path / "hdr" / "test.h5",
        [0, 1, 1, 1, 2, 2, 2, 3, 4, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
    )
    with h5py.File(path, "r") as h5:
        assert not dict(h5.attrs)
        assert all("event_idx" not in node.attrs for node in h5["images"].values())
    dataset = EventHDRDataset(path.parent, max_events=None)
    try:
        # The first event at each frame timestamp is excluded, as is its predecessor.
        # Equal event timestamps must use their leftmost position, including at a chunk edge.
        _assert_sample_boundaries(dataset, [0, 3, 6, 7, 9, 10], [_DERIVED] * 6)
        assert dataset.zero_event_intervals == 1
        assert dataset.event_indexing == {
            "test.h5": {"policy": _POLICY, "stored_images": 0, "derived_images": 6}
        }
    finally:
        dataset.close()


def test_stored_indices_are_preserved_without_running_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_eventhdr(tmp_path / "hdr")

    def unexpected_recovery(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Files with all event_idx attributes must not use timestamp recovery")

    monkeypatch.setattr(eventhdr, "_recover_event_indices", unexpected_recovery)
    dataset = EventHDRDataset(path.parent, max_events=None)
    try:
        # These stored values intentionally differ from the timestamp-predecessor policy.
        _assert_sample_boundaries(dataset, [96, 192, 288, 384], ["stored"] * 4)
        assert dataset.event_indexing == {
            "test.h5": {"policy": _POLICY, "stored_images": 4, "derived_images": 0}
        }
    finally:
        dataset.close()


def test_partial_missing_indices_preserve_stored_values_and_report_provenance(
    tmp_path: Path,
) -> None:
    path = _make_index_file(
        tmp_path / "hdr" / "test.h5",
        [0, 1, 2, 3, 4, 5, 6],
        [1.5, 2.5, 3.5, 4.5],
        [None, 3, None, 7],
    )
    dataset = EventHDRDataset(path.parent, max_events=None)
    try:
        _assert_sample_boundaries(dataset, [1, 3, 3, 7], [_DERIVED, "stored", _DERIVED, "stored"])
        assert dataset.event_indexing == {
            "test.h5": {"policy": _POLICY, "stored_images": 2, "derived_images": 2}
        }
    finally:
        dataset.close()


def test_inspection_reports_relative_file_provenance_for_all_images_with_stride(
    tmp_path: Path,
) -> None:
    _make_index_file(
        tmp_path / "one" / "test.h5",
        [0, 1, 2, 3, 4, 5, 6],
        [1.5, 2.5, 3.5, 4.5],
        [None, 3, None, 7],
    )
    _make_index_file(
        tmp_path / "two" / "test.h5", [0, 1, 2, 3, 4, 5, 6], [2.5, 4.5], [2, 4]
    )
    dataset = EventHDRDataset(tmp_path, max_events=None, frame_stride=2)
    try:
        report = _inspect_one_split(dataset, samples=0, validate_all=True)
        assert len(dataset) == 3
        assert report["event_indexing"] == {
            "one/test.h5": {"policy": _POLICY, "stored_images": 2, "derived_images": 2},
            "two/test.h5": {"policy": _POLICY, "stored_images": 2, "derived_images": 0},
        }
        assert report["event_indexing"] == dataset.event_indexing
    finally:
        dataset.close()


def test_recovery_opens_read_only_and_leaves_source_bytes_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _make_index_file(tmp_path / "hdr" / "test.h5", [0, 1, 2, 3], [1, 2, 3])
    before = hashlib.sha256(path.read_bytes()).digest()
    original_open = h5py.File
    modes: list[str] = []

    def read_only_open(name: Any, mode: str = "r", *args: Any, **kwargs: Any) -> h5py.File:
        assert mode == "r", "Index recovery must never open input HDF5 files for writing"
        modes.append(mode)
        return original_open(name, mode, *args, **kwargs)

    monkeypatch.setattr(eventhdr.h5py, "File", read_only_open)
    dataset = EventHDRDataset(path.parent, max_events=None)
    try:
        _assert_sample_boundaries(dataset, [0, 1, 2], [_DERIVED] * 3)
    finally:
        dataset.close()
    assert modes
    assert hashlib.sha256(path.read_bytes()).digest() == before
    with original_open(path, "r") as h5:
        assert not dict(h5.attrs)
        assert all("event_idx" not in node.attrs for node in h5["images"].values())


@pytest.mark.parametrize("frame_stride", [1, 2])
def test_empty_event_arrays_recover_zero_indices_without_dropping_frames(
    tmp_path: Path, frame_stride: int
) -> None:
    path = _make_index_file(tmp_path / "hdr" / "test.h5", [], [-10, 0, 1, 10])
    dataset = EventHDRDataset(path.parent, max_events=None, frame_stride=frame_stride)
    try:
        selected = 4 if frame_stride == 1 else 2
        _assert_sample_boundaries(dataset, [0] * selected, [_DERIVED] * selected)
        assert dataset.zero_event_intervals == selected
        assert [dataset[index]["metadata"]["timestamp"] for index in range(selected)] == (
            [-10, 0, 1, 10] if frame_stride == 1 else [-10, 1]
        )
        assert dataset.event_indexing["test.h5"]["derived_images"] == 4
    finally:
        dataset.close()


def test_outlying_frame_timestamps_are_clamped_when_ranges_overlap(tmp_path: Path) -> None:
    path = _make_index_file(
        tmp_path / "hdr" / "test.h5", [10, 20, 30, 40], [0, 10, 20, 30, 40, 50]
    )
    dataset = EventHDRDataset(path.parent, max_events=None)
    try:
        _assert_sample_boundaries(dataset, [0, 0, 0, 1, 2, 3], [_DERIVED] * 6)
        assert dataset.zero_event_intervals == 3
    finally:
        dataset.close()


@pytest.mark.parametrize("frames", [[0, 1], [50, 60]])
def test_recovery_rejects_completely_disjoint_timestamp_ranges(
    tmp_path: Path, frames: list[float]
) -> None:
    path = _make_index_file(tmp_path / "hdr" / "test.h5", [10, 20, 30, 40], frames)
    with pytest.raises(ValueError, match="disjoint|do not overlap|non-overlapping"):
        EventHDRDataset(path.parent)


def test_frame_stride_aggregates_recovered_intervals_and_keeps_zero_event_frames(
    tmp_path: Path,
) -> None:
    path = _make_index_file(
        tmp_path / "hdr" / "test.h5",
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 1, 3, 4, 6, 7],
    )
    dataset = EventHDRDataset(path.parent, max_events=None, frame_stride=2)
    try:
        _assert_sample_boundaries(dataset, [0, 0, 3, 6], [_DERIVED] * 4)
        assert dataset.zero_event_intervals == 2
        assert [sample["sequence_index"] for sample in dataset.samples] == [0, 1, 2, 3]
        assert [sample["timestamp"] for sample in dataset.samples] == [0, 1, 4, 7]
        assert [sample["t0"] for sample in dataset.samples] == [None, 0, 1, 4]
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("position", "value", "message"),
    [
        (1, np.nan, "timestamps must be finite"),
        (3, np.inf, "timestamps must be finite"),
        (10, np.nan, "timestamps must be finite"),
        (1, -1, "timestamps must be monotonically"),
        (3, 1.5, "timestamps must be monotonically"),
        (10, 8.5, "timestamps must be monotonically"),
    ],
)
def test_recovery_validates_event_timestamps_within_chunks_across_chunks_and_in_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    position: int,
    value: float,
    message: str,
) -> None:
    monkeypatch.setattr(eventhdr, "_TIMESTAMP_CHUNK_SIZE", 3)
    timestamps = [float(index) for index in range(11)]
    timestamps[position] = value
    path = _make_index_file(tmp_path / "hdr" / "test.h5", timestamps, [1, 2])
    with pytest.raises(ValueError, match=message):
        EventHDRDataset(path.parent)


def test_recovery_reads_timestamp_array_in_bounded_chunks_through_the_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(eventhdr, "_TIMESTAMP_CHUNK_SIZE", 3)
    path = _make_index_file(
        tmp_path / "hdr" / "test.h5", [float(index) for index in range(11)], [1, 2]
    )
    original_getitem = h5py.Dataset.__getitem__
    covered: set[int] = set()

    def bounded_getitem(node: h5py.Dataset, key: Any, *args: Any, **kwargs: Any) -> Any:
        if node.name == "/events/ts":
            if isinstance(key, tuple):
                assert len(key) == 1
                key = key[0]
            if isinstance(key, slice):
                positions = range(*key.indices(len(node)))
                assert len(positions) <= 3, "Recovery must not materialize the full event array"
                covered.update(positions)
            else:
                assert isinstance(key, (int, np.integer))
                covered.add(int(key) % len(node))
        return original_getitem(node, key, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", bounded_getitem)
    dataset = EventHDRDataset(path.parent, max_events=None)
    try:
        assert [sample["end_idx"] for sample in dataset.samples] == [0, 1]
        assert covered == set(range(11))
    finally:
        dataset.close()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (-1, "outside"),
        (99, "outside"),
        (1.5, "must be an integer"),
        (np.nan, "must be finite"),
    ],
)
def test_invalid_stored_index_is_rejected_instead_of_replaced(
    tmp_path: Path, value: float, message: str
) -> None:
    path = _make_index_file(
        tmp_path / "hdr" / "test.h5", [0, 1, 2, 3, 4], [1, 2, 3], [None, value, None]
    )
    with pytest.raises(ValueError, match=message):
        EventHDRDataset(path.parent)


def test_combined_stored_and_recovered_indices_must_remain_monotonic(tmp_path: Path) -> None:
    path = _make_index_file(
        tmp_path / "hdr" / "test.h5", [0, 1, 2, 3, 4], [1, 2, 3], [3, None, None]
    )
    with pytest.raises(ValueError, match="event_idx values must be monotonically"):
        EventHDRDataset(path.parent)


@pytest.mark.parametrize("frame_timestamp", [np.nan, np.inf, -1.0])
def test_recovery_does_not_bypass_frame_timestamp_validation(
    tmp_path: Path, frame_timestamp: float
) -> None:
    path = _make_index_file(tmp_path / "hdr" / "test.h5", [0, 1, 2, 3], [1, frame_timestamp])
    message = "must be finite" if not np.isfinite(frame_timestamp) else "monotonically"
    with pytest.raises(ValueError, match=message):
        EventHDRDataset(path.parent)


def test_missing_frame_timestamp_cannot_be_recovered_from_root_metadata(tmp_path: Path) -> None:
    path = _make_index_file(tmp_path / "hdr" / "test.h5", [0, 1, 2, 3], [1, 2, 3])
    with h5py.File(path, "a") as h5:
        del h5["images/image000000001"].attrs["timestamp"]
        h5.attrs["t0"] = 0.0
        h5.attrs["tn"] = 3.0
        h5.attrs["num_imgs"] = 3
    with pytest.raises(ValueError, match="missing 'timestamp'"):
        EventHDRDataset(path.parent)
