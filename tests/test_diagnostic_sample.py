"""CPU-only synthetic fixtures; never substitute these for evaluation results."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import h5py
import pytest
import torch

from asgcn_unet.data import build_dataset
from asgcn_unet.data.eventaid_r import EventAidRZipDataset
from asgcn_unet.data.eventhdr import EventHDRDataset
from asgcn_unet.diagnostic_sample import DiagnosticSampleError, read_diagnostic_sample
from asgcn_unet.engine import _dataset_sample_identity
from tests.fixtures import make_eventaid, make_eventhdr
from tests.test_eventaid_parts import _upload_members, _write_upload
from tests.test_eventhdr_index import _make_index_file

_BUDGET = 64 * 1024**2


def _config(root: Path, kind: str, **kwargs) -> dict:
    return {"type": kind, "root": str(root), "target_channels": 1,
            "tone_map": "log", "tone_map_mu": 5000.0,
            "target_normalization": {"mode": "integer_dtype_max"}, **kwargs}


def _baseline(config: dict, index: int) -> tuple[dict, dict]:
    reader = build_dataset(config, split="eval")
    try:
        return _dataset_sample_identity(reader, index), reader[index]
    finally:
        reader.close()


def _assert_same(actual: dict, expected: dict) -> None:
    assert torch.equal(actual["events"], expected["events"])
    assert torch.equal(actual["target"], expected["target"])
    assert actual["sample_id"] == expected["sample_id"]
    assert actual["sensor_size"] == expected["sensor_size"]
    metadata = dict(actual["metadata"])
    diagnostic = metadata.pop("diagnostic_source_read")
    assert diagnostic["full_source_hash_verified"] is False
    assert diagnostic["estimated_working_memory_bytes"] <= diagnostic["memory_budget_bytes"]
    assert metadata == expected["metadata"]


def _forbid_constructor(*args, **kwargs):
    raise AssertionError("Dataset-wide constructor/index is forbidden")


@pytest.mark.parametrize("index", [0, 1, 3])
@pytest.mark.parametrize("crop,max_events,channels", [(None, None, 1), ([20, 24], 17, 3)])
def test_hdr_exact_preprocessing_without_constructor(
    tmp_path, monkeypatch, index, crop, max_events, channels,
):
    path = make_eventhdr(tmp_path / "hdr")
    cfg = _config(path.parent, "eventhdr", crop_size=crop,
                  max_events=max_events, target_channels=channels)
    identity, expected = _baseline(cfg, index)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(EventHDRDataset, "__init__", _forbid_constructor)
    monkeypatch.setattr(EventHDRDataset, "_build_index", _forbid_constructor)
    monkeypatch.setattr(torch.cuda, "init", _forbid_constructor)
    actual = read_diagnostic_sample({"dataset": cfg}, identity, memory_budget_bytes=_BUDGET)
    _assert_same(actual, expected)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(p.name for p in path.parent.iterdir()) == ["test.h5"]


@pytest.mark.parametrize("stride,index", [(2, 0), (2, 1), (3, 1)])
@pytest.mark.parametrize("stored", [True, False])
def test_hdr_frame_stride_and_legacy_index_parity(tmp_path, stride, index, stored):
    path = make_eventhdr(tmp_path / "hdr", frames=7)
    if not stored:
        with h5py.File(path, "r+") as handle:
            for image in handle["images"].values():
                del image.attrs["event_idx"]
    cfg = _config(path.parent, "eventhdr", frame_stride=stride, max_events=31)
    identity, expected = _baseline(cfg, index)
    _assert_same(read_diagnostic_sample(cfg, identity, memory_budget_bytes=_BUDGET), expected)


def test_hdr_legacy_duplicate_timestamp_and_empty_window(tmp_path):
    path = _make_index_file(tmp_path / "hdr" / "test.h5", [0, 1, 1, 1, 2, 2, 3], [1, 1, 2, 3])
    cfg = _config(path.parent, "eventhdr", max_events=None)
    for index in range(4):
        identity, expected = _baseline(cfg, index)
        _assert_same(read_diagnostic_sample(cfg, identity, memory_budget_bytes=_BUDGET), expected)


@pytest.mark.parametrize("offset,index", [(-1, 0), (0, 1), (1, 0), (1, 2)])
@pytest.mark.parametrize("upload", [False, True])
def test_aid_exact_preprocessing_offsets_and_parts(tmp_path, monkeypatch, offset, index, upload):
    path = (_write_upload(tmp_path / "aid", _upload_members()) if upload
            else make_eventaid(tmp_path / "aid", frames=6))
    if not upload and offset <= 0:
        # The ordinary fixture has timestamps for offset=1 only. Extend this
        # tiny synthetic archive's timing rows to cover its final event block.
        with zipfile.ZipFile(path, "r") as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        members["timestamps.txt"] += b"1060000\n"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in members.items():
                archive.writestr(name, value)
    cfg = _config(path.parent, "eventaid_r_zip", target_offset=offset,
                  max_events=7, crop_size=[6, 8])
    identity, expected = _baseline(cfg, index)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(EventAidRZipDataset, "__init__", _forbid_constructor)
    monkeypatch.setattr(EventAidRZipDataset, "_build_index", _forbid_constructor)
    monkeypatch.setattr(torch.cuda, "init", _forbid_constructor)
    _assert_same(read_diagnostic_sample(cfg, identity, memory_budget_bytes=_BUDGET), expected)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_aid_nested_case_and_backslash_member_names(tmp_path):
    members = {"nested\\R-parts\\" + name.upper().replace("/", "\\"): value
               for name, value in _upload_members().items()}
    path = _write_upload(tmp_path / "aid", members)
    cfg = _config(path.parent, "eventaid_r_zip", max_events=None)
    identity, expected = _baseline(cfg, 4)
    _assert_same(read_diagnostic_sample(cfg, identity, memory_budget_bytes=_BUDGET), expected)


@pytest.mark.parametrize("kind", ["eventhdr", "eventaid_r_zip"])
def test_memory_refusal_before_expensive_decode(tmp_path, monkeypatch, kind):
    path = (make_eventhdr(tmp_path / "hdr") if kind == "eventhdr"
            else make_eventaid(tmp_path / "aid"))
    cfg = _config(path.parent, kind)
    identity, _ = _baseline(cfg, 1)
    cls = EventHDRDataset if kind == "eventhdr" else EventAidRZipDataset
    monkeypatch.setattr(cls, "__getitem__", _forbid_constructor)
    with pytest.raises(DiagnosticSampleError, match="budget"):
        read_diagnostic_sample(cfg, identity, memory_budget_bytes=1)


def test_memory_reserve_callback_runs_before_source_payload_decode(tmp_path, monkeypatch):
    path = make_eventaid(tmp_path / "aid")
    cfg = _config(path.parent, "eventaid_r_zip")
    identity, _ = _baseline(cfg, 0)
    stages = []

    def reserve(amount, stage):
        stages.append((amount, stage))
        if stage == "Selected ZIP event and image decode":
            raise RuntimeError("Current cgroup memory unavailable")

    monkeypatch.setattr(EventAidRZipDataset, "__getitem__", _forbid_constructor)
    with pytest.raises(RuntimeError, match="cgroup"):
        read_diagnostic_sample(cfg, identity, memory_budget_bytes=_BUDGET, reserve=reserve)
    assert len(stages) >= 4
    assert all(amount <= _BUDGET for amount, _ in stages)


@pytest.mark.parametrize("key,value", [
    ("source_file", "../test.h5"), ("image_key", "../image0"),
    ("start_idx", 1), ("end_idx", 0), ("timestamp", 17),
    ("sequence_index", 2), ("group", "another.h5"),
    ("start_idx", True), ("dataset_index", -1),
])
def test_hdr_rejects_changed_or_unsafe_saved_identity(tmp_path, key, value):
    path = make_eventhdr(tmp_path / "hdr")
    cfg = _config(path.parent, "eventhdr")
    identity, _ = _baseline(cfg, 1)
    identity[key] = value
    with pytest.raises((DiagnosticSampleError, FileNotFoundError)):
        read_diagnostic_sample(cfg, identity, memory_budget_bytes=_BUDGET)


@pytest.mark.parametrize("key,value", [
    ("group", "R-../../bad"), ("event_name", "event_upload/000003.txt"),
    ("target_name", "gt_upload/000003_img.jpg"), ("frame_id", 3),
    ("sequence_index", 3), ("part_index", 1), ("sequence_id", "wrong/part-000"),
    ("t0_us", 0), ("t1_us", True),
])
def test_aid_rejects_changed_or_cross_part_identity(tmp_path, key, value):
    path = _write_upload(tmp_path / "aid", _upload_members())
    cfg = _config(path.parent, "eventaid_r_zip")
    identity, _ = _baseline(cfg, 0)
    identity[key] = value
    with pytest.raises(DiagnosticSampleError):
        read_diagnostic_sample(cfg, identity, memory_budget_bytes=_BUDGET)


def test_hdf5_external_source_link_refused(tmp_path):
    path = make_eventhdr(tmp_path / "hdr")
    cfg = _config(path.parent, "eventhdr")
    identity, _ = _baseline(cfg, 0)
    with h5py.File(path, "r+") as handle:
        del handle["events/xs"]
        handle["events/xs"] = h5py.ExternalLink("outside.h5", "events/xs")
    with pytest.raises(DiagnosticSampleError, match="links"):
        read_diagnostic_sample(cfg, identity, memory_budget_bytes=_BUDGET)


def test_zip_directory_budget_checked_before_zipfile_open(tmp_path, monkeypatch):
    path = make_eventaid(tmp_path / "aid")
    cfg = _config(path.parent, "eventaid_r_zip")
    identity, _ = _baseline(cfg, 0)
    monkeypatch.setattr(zipfile.ZipFile, "__init__", _forbid_constructor)
    with pytest.raises(DiagnosticSampleError, match="central directory"):
        read_diagnostic_sample(cfg, identity, memory_budget_bytes=1024**2)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_memory_budget_rejected_before_source_access(budget):
    with pytest.raises(DiagnosticSampleError, match="memory_budget_bytes"):
        read_diagnostic_sample({"type": "eventhdr", "root": "missing"},
                               {"dataset_index": 0}, memory_budget_bytes=budget)
