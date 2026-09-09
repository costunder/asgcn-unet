"""CPU-only synthetic input-contract tests, not training or dataset validation."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from asgcn_unet.data import EventAidRZipDataset, EventHDRDataset, build_dataset
from asgcn_unet.stream_input import (
    LEGACY_EVENT_TIME_CONTRACT,
    PHYSICAL_EVENT_TIME_CONTRACT,
    arrival_group_counts,
    hdr_boundary_policy,
    stream_time_metadata,
    to_physical_seconds,
)

PHYSICAL = {"event_time_contract": PHYSICAL_EVENT_TIME_CONTRACT, "max_events": None}


def _physical(scale):
    return {**PHYSICAL, "timestamp_scale_to_seconds": scale,
            "interval_timestamp_scale_to_seconds": scale}


def _hdr(
    root: Path, timestamps: list[float] | np.ndarray,
    boundaries: list[tuple[float, int | None]],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "synthetic.h5"
    with h5py.File(path, "w") as h5:
        events = h5.create_group("events")
        size = len(timestamps)
        events.create_dataset("ts", data=np.asarray(timestamps, dtype=np.float64))
        events.create_dataset("xs", data=np.arange(size, dtype=np.int16) % 8)
        events.create_dataset("ys", data=np.full(size, 3, dtype=np.int16))
        events.create_dataset("ps", data=np.arange(size, dtype=np.int16) % 2)
        images = h5.create_group("images")
        for index, (timestamp, endpoint) in enumerate(boundaries):
            image = images.create_dataset(
                f"image{index:09d}", data=np.full((8, 8), index, dtype=np.uint16),
            )
            image.attrs["timestamp"] = timestamp
            if endpoint is not None:
                image.attrs["event_idx"] = endpoint
    return path


def _aid(root: Path, rows: list[str], frame_times: list[int]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "R-synthetic.zip"
    output = io.BytesIO()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(output, format="PNG")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("shape.txt", "8 8\n")
        archive.writestr("timestamps.txt", "\n".join(map(str, frame_times)))
        for index in range(1, len(frame_times) + 1):
            archive.writestr(f"gt/{index:06d}_img.png", output.getvalue())
            archive.writestr(
                f"event/{index:06d}.txt", rows[index - 1] if index <= len(rows) else "",
            )
    return path


@pytest.mark.parametrize("dataset", [EventHDRDataset, EventAidRZipDataset])
@pytest.mark.parametrize("scale", [None, True, 0, -1, float("nan"), float("inf"), "1e-6"])
def test_physical_contract_requires_explicit_finite_positive_scale(tmp_path, dataset, scale):
    with pytest.raises((ValueError, TypeError), match="timestamp_scale_to_seconds"):
        dataset(tmp_path, **PHYSICAL, timestamp_scale_to_seconds=scale,
                interval_timestamp_scale_to_seconds=1)


@pytest.mark.parametrize("dataset", [EventHDRDataset, EventAidRZipDataset])
@pytest.mark.parametrize("cap", [0, -1, 8192])
def test_physical_contract_rejects_every_non_null_cap(tmp_path, dataset, cap):
    with pytest.raises(ValueError, match="max_events=null"):
        dataset(tmp_path, event_time_contract=PHYSICAL_EVENT_TIME_CONTRACT,
                timestamp_scale_to_seconds=1, max_events=cap)


def test_factory_requires_explicit_uncapped_contract_before_reading_data(tmp_path):
    with pytest.raises(ValueError, match="max_events=null"):
        build_dataset({"type": "eventhdr", "root": str(tmp_path),
                       "event_time_contract": PHYSICAL_EVENT_TIME_CONTRACT,
                       "timestamp_scale_to_seconds": 1.0})
    with pytest.raises(ValueError, match="requires physical_seconds"):
        EventHDRDataset(tmp_path, timestamp_scale_to_seconds=1.0)
    with pytest.raises(ValueError, match="Unsupported event_time_contract"):
        EventHDRDataset(tmp_path, event_time_contract="guess_units")


def test_hdr_physical_precision_ids_crop_and_fixed_origin(tmp_path):
    times = 1_000_000.0 + np.arange(16, dtype=np.float64) * 1e-5
    _hdr(tmp_path, times, [(times[8], 8), (times[-1], 16)])
    dataset = EventHDRDataset(tmp_path, **_physical(1.0),
                              crop_size=(4, 4))
    try:
        first, second = dataset[0], dataset[1]
        assert first["events"].dtype == torch.float64
        assert first["event_ids"].dtype == torch.int64
        assert first["event_ids"].tolist() == [[0, value] for value in range(2, 6)]
        assert second["event_ids"].tolist() == [[0, value] for value in range(10, 14)]
        assert torch.equal(first["events"][:, 2], torch.from_numpy(times[2:6]))
        assert first["metadata"]["stream_time"]["sequence_origin_seconds"] == times[0]
        assert second["metadata"]["stream_time"]["sequence_origin_seconds"] == times[0]
        assert second["metadata"]["stream_time"]["interval_start_seconds"] == times[8]
        assert dataset.get_topology_sample(1)["event_ids"].tolist() == second["event_ids"].tolist()
        assert torch.equal(dataset[1]["events"], second["events"])
    finally:
        dataset.close()


def test_hdr_preserves_all_events_beyond_old_cap_and_explicit_units(tmp_path):
    timestamps = np.arange(8201, dtype=np.float64)
    _hdr(tmp_path, timestamps, [(8201.0, len(timestamps))])
    dataset = build_dataset({"type": "eventhdr", "root": str(tmp_path), **_physical(1e-6)})
    try:
        sample = dataset[0]
        assert len(sample["events"]) == 8201
        assert sample["metadata"]["dataset_sampling_ratio"] == 1.0
        assert sample["event_ids"][-1].tolist() == [0, 8200]
        assert torch.equal(sample["events"][:, 2], torch.from_numpy(timestamps * 1e-6))
        assert sample["metadata"]["stream_time"]["interval_end_seconds"] == 8201e-6
    finally:
        dataset.close()


def test_hdr_documented_predecessor_late_delivery_is_not_clock_repair(tmp_path):
    _hdr(tmp_path, list(range(7)), [(2.5, None), (4.5, None)])
    dataset = EventHDRDataset(tmp_path, **_physical(1))
    try:
        first, second = dataset[0], dataset[1]
        assert first["event_ids"].tolist() == [[0, 0], [0, 1]]
        assert second["event_ids"].tolist() == [[0, 2], [0, 3]]
        assert second["events"][:, 2].tolist() == [2, 3]
        contract = second["metadata"]["stream_time"]
        assert contract["interval_start_seconds"] == 2.5
        assert contract["late_predecessor_event_count"] == 1
        assert contract["late_predecessor_source_row"] == 2
        assert contract["late_predecessor_seconds"] == 0.5
        assert contract["clock_correction_applied"] is False
    finally:
        dataset.close()


def test_hdr_rejects_stored_clock_mismatch_without_repair(tmp_path):
    _hdr(tmp_path, list(range(10)), [(2.5, 8)])
    with pytest.raises(ValueError, match="clock/index mismatch"):
        EventHDRDataset(tmp_path, **_physical(1))
    legacy = EventHDRDataset(tmp_path)
    try:
        assert legacy.samples[0]["end_idx"] == 8
    finally:
        legacy.close()


def test_hdr_empty_stream_has_fixed_origin_and_does_not_drop_readouts(tmp_path):
    _hdr(tmp_path, [], [(10, 0), (20, 0)])
    dataset = EventHDRDataset(tmp_path, **_physical(1e-3))
    try:
        assert len(dataset) == 2
        for sample in [dataset[1], dataset[0]]:
            assert sample["events"].shape == (0, 4)
            assert sample["events"].dtype == torch.float64
            assert sample["event_ids"].shape == (0, 2)
            assert sample["metadata"]["stream_time"]["sequence_origin_seconds"] == 0.01
    finally:
        dataset.close()


def test_aid_physical_seconds_ids_and_fixed_origin(tmp_path):
    _aid(tmp_path, ["1000000 0 3 1\n1000001 3 3 0\n1000999 7 3 1",
                    "1001000 2 3 1\n1001001 5 3 0"], [1000000, 1001000, 1002000])
    dataset = EventAidRZipDataset(tmp_path, **_physical(1e-6),
                                  crop_size=(4, 4))
    try:
        first, second = dataset[0], dataset[1]
        assert first["events"].dtype == torch.float64
        assert first["event_ids"].tolist() == [[1, 1]]
        assert second["event_ids"].tolist() == [[2, 0], [2, 1]]
        assert first["events"][0, 2].item() == 1000001 * 1e-6
        for sample in [first, second]:
            assert sample["metadata"]["stream_time"]["sequence_origin_seconds"] == 1.0
            assert sample["metadata"]["event_timestamp_diagnostics"]["strict_interval_validation"]
        assert second["metadata"]["stream_time"]["interval_start_seconds"] == 1001000 * 1e-6
    finally:
        dataset.close()


def test_aid_uncapped_physical_input_does_not_call_legacy_subsample(tmp_path, monkeypatch):
    from asgcn_unet.data import eventaid_r
    rows = "\n".join(f"{index} 0 0 1" for index in range(8201))
    _aid(tmp_path, [rows], [0, 9000])
    def forbidden(*args, **kwargs):
        pytest.fail("Physical inputs must not pass through the legacy float32 subsampler")
    monkeypatch.setattr(eventaid_r, "stratified_subsample", forbidden)
    dataset = EventAidRZipDataset(tmp_path, **_physical(1e-6))
    try:
        sample = dataset[0]
        assert len(sample["events"]) == 8201
        assert sample["event_ids"][-1].tolist() == [1, 8200]
    finally:
        dataset.close()


@pytest.mark.parametrize("rows", ["99 0 0 1", "201 0 0 1", "0.0001 0 0 1"])
def test_aid_physical_clock_mismatch_fails_before_crop(tmp_path, rows):
    _aid(tmp_path, [rows], [100, 200])
    dataset = EventAidRZipDataset(tmp_path, **_physical(1e-6),
                                  crop_size=(2, 2))
    try:
        with pytest.raises(ValueError, match="clock mismatch"):
            dataset[0]
    finally:
        dataset.close()


def test_legacy_hdr_events_remain_bitwise_original_normalized_float32(tmp_path):
    raw_times = np.asarray([10, 11, 13, 15, 19, 20], dtype=np.float64)
    _hdr(tmp_path, raw_times, [(20, 6)])
    dataset = EventHDRDataset(tmp_path, max_events=4)
    try:
        sample = dataset[0]
        expected = np.column_stack((np.arange(6) % 8, np.full(6, 3),
                                    (raw_times - 10) / 10, np.where(np.arange(6) % 2, 1, -1)))
        expected = expected.astype(np.float32)[np.linspace(0, 5, 4, dtype=np.int64)]
        assert torch.equal(sample["events"], torch.from_numpy(expected))
        assert "event_ids" not in sample
        assert "stream_time" not in sample["metadata"]
    finally:
        dataset.close()


def test_legacy_aid_static_parser_is_unchanged_and_only_physical_is_strict():
    raw = b"100 0 0 1\n102 1 1 0\n105 2 2 1\n"
    legacy, diagnostics = EventAidRZipDataset._read_events(raw, interval_t0=0, interval_t1=1)
    explicit, explicit_diagnostics = EventAidRZipDataset._read_events(
        raw, interval_t0=0, interval_t1=1, event_time_contract=LEGACY_EVENT_TIME_CONTRACT,
    )
    expected = np.asarray([[0, 0, 0, 1], [1, 1, 0.4, -1], [2, 2, 1, 1]], dtype=np.float32)
    assert np.array_equal(legacy, expected)
    assert np.array_equal(legacy, explicit)
    assert diagnostics == explicit_diagnostics
    assert diagnostics["strict_interval_validation"] is False


@pytest.mark.parametrize("timestamps,boundary,index,expected", [
    ([0, 1, 1, 2], 1, 1, "timestamp_left_boundary"),
    ([0, 1, 1, 2], 1, 3, "timestamp_right_boundary"),
    ([0, 1, 1, 2], 1, 0, "timestamp_predecessor_v1"),
    ([0, 1, 1, 2], 3, 3, "timestamp_predecessor_v1"),
])
def test_hdr_boundary_proof_uses_exact_original_rows(timestamps, boundary, index, expected):
    assert hdr_boundary_policy(np.asarray(timestamps), index, boundary, source="synthetic") == expected


def test_seconds_overflow_and_unproven_multiple_late_rows_fail():
    with pytest.raises(ValueError, match="finite"):
        to_physical_seconds([1e308], 10, source="synthetic")
    with pytest.raises(ValueError, match="clock mismatch"):
        stream_time_metadata(np.asarray([0.1, 0.2]), interval_start_seconds=1,
                             interval_end_seconds=2, sequence_origin_seconds=0,
                             timestamp_scale_to_seconds=1, allow_predecessor_row=True,
                             interval_timestamp_scale_to_seconds=1,
                             source="synthetic")


@pytest.mark.parametrize("values,expected", [([], ()), ([1], (1,)),
                                                        ([1, 1, 2, 3, 3], (2, 1, 2))])
def test_equal_raw_timestamp_arrival_groups(values, expected):
    assert arrival_group_counts(np.asarray(values, dtype=np.float64)) == expected


@pytest.mark.parametrize("kind", ["hdr", "aid"])
def test_arrival_groups_are_recomputed_after_roi_filtering(tmp_path, kind):
    if kind == "hdr":
        _hdr(tmp_path, [100, 100, 100, 200, 200, 300, 400, 400], [(400, 8)])
        dataset = EventHDRDataset(tmp_path, **_physical(1e-6),
                                  crop_size=(4, 4))
    else:
        _aid(tmp_path, [("100 0 3 1\n100 2 3 1\n100 3 3 0\n200 7 3 1\n"
                        "200 4 3 1\n300 5 3 0\n400 7 3 1")], [100, 400])
        dataset = EventAidRZipDataset(tmp_path, **_physical(1e-6),
                                      crop_size=(4, 4))
    try:
        sample = dataset[0]
        expected = (1, 2, 1) if kind == "hdr" else (2, 1, 1)
        assert sample["metadata"]["stream_time"]["arrival_group_counts"] == expected
        assert sum(expected) == len(sample["events"])
    finally:
        dataset.close()


def test_clock_conversion_rejects_timestamp_merging_and_underflow():
    with pytest.raises(ValueError, match="loses timestamp resolution"):
        to_physical_seconds([1e-300, 2e-300], 1e-100, source="synthetic")


def test_aid_empty_events_preserve_readout_and_arrival_shape(tmp_path):
    _aid(tmp_path, [""], [100, 200])
    dataset = EventAidRZipDataset(tmp_path, **_physical(1e-3))
    try:
        sample = dataset[0]
        assert sample["events"].dtype == torch.float64
        assert sample["events"].shape == (0, 4)
        assert sample["event_ids"].shape == (0, 2)
        assert sample["metadata"]["stream_time"]["arrival_group_counts"] == ()
        assert sample["metadata"]["stream_time"]["source_interval_start"] == 100
        assert sample["metadata"]["t0_us"] == 100000
        assert sample["metadata"]["dt_us"] == 100000
    finally:
        dataset.close()


@pytest.mark.parametrize("offset", [-1, 1])
def test_aid_parts_origin_is_fixed_from_first_published_row(tmp_path, offset):
    from tests.test_eventaid_parts import _IDS, _PARTS, _upload_members, _write_upload
    _write_upload(tmp_path, _upload_members())
    dataset = EventAidRZipDataset(tmp_path, **_physical(1e-6),
                                  target_offset=offset)
    try:
        for index in reversed(range(len(dataset))):
            sample = dataset[index]
            part = sample["metadata"]["part_index"]
            first_frame = _PARTS[part][0]
            origin = (1000 + _IDS.index(first_frame) * 1000) * 1e-6
            assert sample["metadata"]["stream_time"]["sequence_origin_seconds"] == origin
            assert sample["metadata"]["sequence_id"] == f"R-parts/part-{part:03d}"
    finally:
        dataset.close()


def test_hdr_physical_index_and_sample_keep_the_same_sequence_group(tmp_path):
    from asgcn_unet.batching import SequenceBatchSampler
    _hdr(tmp_path, [0, 1, 2, 3], [(2, 2), (3, 4)])
    dataset = EventHDRDataset(tmp_path, **_physical(1),
                              file_to_scene={"synthetic.h5": "official-train-group"})
    try:
        sampler = SequenceBatchSampler(dataset, batch_size=2)
        assert dataset.samples[0]["sequence_id"] == "synthetic.h5"
        assert dataset[0]["metadata"]["sequence_id"] == "synthetic.h5"
        assert sampler.final_sequence_indices == {("synthetic.h5", "synthetic.h5"): 1}
    finally:
        dataset.close()


@pytest.mark.parametrize("dataset", [EventHDRDataset, EventAidRZipDataset])
@pytest.mark.parametrize("scale", [None, True, 0, -1, float("nan"), float("inf"), "1e-6"])
def test_physical_contract_requires_separate_explicit_interval_scale(tmp_path, dataset, scale):
    with pytest.raises((ValueError, TypeError), match="interval_timestamp_scale_to_seconds"):
        dataset(tmp_path, **PHYSICAL, timestamp_scale_to_seconds=1,
                interval_timestamp_scale_to_seconds=scale)


@pytest.mark.parametrize("recovered", [False, True])
def test_hdr_event_seconds_and_frame_microseconds_share_clock_without_reindexing(tmp_path, recovered):
    endpoints = [None, None] if recovered else [2, 4]
    _hdr(tmp_path, [1.0, 1.1, 1.2, 1.3],
         [(1_200_000, endpoints[0]), (1_400_000, endpoints[1])])
    dataset = EventHDRDataset(tmp_path, **PHYSICAL, timestamp_scale_to_seconds=1,
                              interval_timestamp_scale_to_seconds=1e-6)
    try:
        first, second = dataset[0], dataset[1]
        expected = [[0, 0]] if recovered else [[0, 0], [0, 1]]
        assert first["event_ids"].tolist() == expected
        assert second["metadata"]["stream_time"]["interval_start_seconds"] == 1.2
        assert second["metadata"]["stream_time"]["sequence_origin_seconds"] == 1.0
        assert first["metadata"]["stream_time"]["source_interval_end"] == 1_200_000
        assert first["metadata"]["timestamp"] == 1.2
        assert second["metadata"]["dt_us"] == 200_000
        if recovered:
            assert second["event_ids"].tolist() == [[0, 1], [0, 2]]
            assert second["events"][0, 2].item() == 1.1
            assert second["metadata"]["stream_time"]["late_predecessor_source_row"] == 1
    finally:
        dataset.close()


def test_aid_txt_seconds_and_frame_microseconds_are_independently_scaled(tmp_path):
    _aid(tmp_path, ["1.0 0 0 1\n1.001 1 1 0\n1.009 2 2 1"], [1_000_000, 1_010_000])
    dataset = EventAidRZipDataset(tmp_path, **PHYSICAL, timestamp_scale_to_seconds=1,
                                  interval_timestamp_scale_to_seconds=1e-6)
    try:
        sample = dataset[0]
        assert sample["events"][:, 2].tolist() == [1.0, 1.001, 1.009]
        timing = sample["metadata"]["stream_time"]
        assert timing["interval_start_seconds"] == 1.0
        assert timing["timestamp_scale_to_seconds"] == 1
        assert timing["interval_timestamp_scale_to_seconds"] == 1e-6
        assert timing["source_interval_start"] == 1_000_000
        diagnostics = sample["metadata"]["event_timestamp_diagnostics"]
        assert diagnostics["timestamp_unit"] == "seconds"
        assert diagnostics["outside_interval_count"] == 0
        assert diagnostics["event_timestamp_min"] == diagnostics["interval_t0"] == 1.0
        assert diagnostics["source_event_timestamp_min"] == 1.0
        assert diagnostics["source_interval_t0"] == 1_000_000
    finally:
        dataset.close()


@pytest.mark.parametrize("event_rows,frame_scale", [
    ("1.0 0 0 1", 1.0), ("101.0 0 0 1", 1e-6),
])
def test_wrong_declared_unit_or_clock_offset_is_never_inferred(tmp_path, event_rows, frame_scale):
    _aid(tmp_path, [event_rows], [1_000_000, 1_010_000])
    dataset = EventAidRZipDataset(tmp_path, **PHYSICAL, timestamp_scale_to_seconds=1,
                                  interval_timestamp_scale_to_seconds=frame_scale)
    try:
        with pytest.raises(ValueError, match="clock mismatch"):
            dataset[0]
    finally:
        dataset.close()
