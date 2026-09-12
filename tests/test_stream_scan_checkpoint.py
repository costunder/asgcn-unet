"""Bounded CPU synthetic scan recovery tests; never research measurements."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from asgcn_unet import stream_scan_checkpoint as checkpoint
from asgcn_unet.artifact_lock import ArtifactWriterBusyError, exclusive_artifact_writer


@pytest.fixture(autouse=True)
def _bounded_cpu(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr("asgcn_unet.diagnostic_resources._snapshot", lambda: {"headroom_bytes": 64 * 1024**2})
    yield
    torch.set_num_threads(previous)


def _fixture(completed=1):
    batches = [[0, 3], [1, 4], [2, 5]]
    identity = {name: "a" * 64 for name in checkpoint.IDENTITY_FIELDS}
    identity["schedule_sha256"] = checkpoint._digest(batches)
    finals = {("lane-a", "source-a"): 2, ("lane-b", "source-b"): 2}
    records = [None] * 6
    states = {}
    for frame in range(completed):
        for lane, key in enumerate(finals):
            index = frame + lane * 3
            records[index] = {"dataset_index": index, "sample_id": f"{key[0]}/{frame}",
                              "sequence_identity": list(key), "sequence_index": frame,
                              "interval_start_seconds": float(frame), "interval_end_seconds": float(frame + 1),
                              "readout_nodes": 2, "readout_directed_edges": 2,
                              "prefix_union_nodes_upper_bound": 3, "prefix_union_directed_edges_upper_bound": 6}
            if frame == finals[key]:
                states.pop(key, None)
            else:
                times = torch.tensor([frame + 0.25, frame + 0.75], dtype=torch.float64)
                positions = torch.tensor([[0.1, 0.2, frame + 0.25, 0.0],
                                          [0.2, 0.2, frame + 0.75, 1.0]], dtype=torch.float64)
                states[key] = SimpleNamespace(positions=positions, timestamps=times,
                                              origin_seconds=0.0, watermark_seconds=float(frame + 1),
                                              sequence_index=frame, sequence_identity=key, contract="b" * 64,
                                              directed_edges=2)
    topology = {"samples": records, "dataset_samples": 6, "scanned_samples": 2 * completed,
                "scan_complete": completed == 3, "current_batch_indices": [2, 5]}
    return {"identity": identity, "batches": batches, "completed_batches": completed,
            "states": states, "topology": topology, "sequence_final_indices": finals,
            "memory_budget_mib": 8, "reserve_memory_mib": 1}


def _load(path, values, **overrides):
    options = {"expected_identity": values["identity"], "batches": values["batches"],
               "sequence_final_indices": values["sequence_final_indices"],
               "memory_budget_mib": values["memory_budget_mib"], "reserve_memory_mib": values["reserve_memory_mib"]}
    options.update(overrides)
    return checkpoint.load_scan_checkpoint(path, **options)


def test_roundtrip_preserves_exact_state_counts_cursor_and_owns_cpu_storage(tmp_path, monkeypatch):
    values = _fixture()
    path = tmp_path / "checkpoint"
    manifest = checkpoint.save_scan_checkpoint(path, **values)
    original_load = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    restored = _load(path, values)
    assert restored["completed_batches"] == 1 and restored["manifest"] == manifest
    assert restored["topology"] == values["topology"]
    assert calls == [{"map_location": "cpu", "weights_only": True, "mmap": True}]
    for key, original in values["states"].items():
        actual = restored["states"][key]
        assert torch.equal(actual.positions, original.positions)
        assert torch.equal(actual.timestamps, original.timestamps)
        assert actual.positions.data_ptr() != original.positions.data_ptr()
        assert actual.directed_edges == original.directed_edges and actual.sequence_identity == key
        assert actual.contract == original.contract and actual.positions.device.type == "cpu"
    # mmap storage is not retained by returned clones, so rotating the generation works on Windows too.
    checkpoint.save_scan_checkpoint(path, **_fixture(2))
    assert not (path / manifest["tensors"]["filename"]).exists()
    assert _load(path, values)["completed_batches"] == 2


def test_finished_lanes_are_absent_and_complete_empty_state_is_valid(tmp_path):
    values = _fixture(3)
    checkpoint.save_scan_checkpoint(tmp_path / "complete", **values)
    restored = _load(tmp_path / "complete", values)
    assert restored["states"] == {} and restored["completed_batches"] == 3


@pytest.mark.parametrize("changed", sorted(checkpoint.IDENTITY_FIELDS))
def test_changed_source_config_data_schedule_or_report_refuses_before_torch_load(tmp_path, monkeypatch, changed):
    values = _fixture()
    path = tmp_path / "checkpoint"
    checkpoint.save_scan_checkpoint(path, **values)
    identity = dict(values["identity"], **{changed: "c" * 64})
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("torch.load must not run"))
    with pytest.raises(ValueError, match="identity mismatch"):
        _load(path, values, expected_identity=identity)


@pytest.mark.parametrize("field", ["metadata", "tensors"])
def test_payload_hash_tampering_refused_before_torch_load(tmp_path, monkeypatch, field):
    values = _fixture()
    path = tmp_path / "checkpoint"
    manifest = checkpoint.save_scan_checkpoint(path, **values)
    file = path / manifest[field]["filename"]
    data = bytearray(file.read_bytes())
    data[len(data) // 2] ^= 1
    file.write_bytes(data)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("unverified tensor load"))
    with pytest.raises(ValueError, match="hash/size mismatch"):
        _load(path, values)


@pytest.mark.parametrize("mutation", ["missing_lane", "wrong_edges", "wrong_clock", "wrong_shape", "past_cursor", "nan"])
def test_invalid_snapshot_is_not_published(tmp_path, mutation):
    values = _fixture()
    key = next(iter(values["states"]))
    state = values["states"][key]
    if mutation == "missing_lane":
        values["states"].pop(key)
    elif mutation == "wrong_edges":
        state.directed_edges = 0
    elif mutation == "wrong_clock":
        state.watermark_seconds = 2.0
    elif mutation == "wrong_shape":
        state.positions = state.positions[:1]
    elif mutation == "past_cursor":
        values["topology"]["samples"][1] = dict(values["topology"]["samples"][0], dataset_index=1)
    else:
        state.timestamps[0] = float("nan")
    with pytest.raises((ValueError, TypeError)):
        checkpoint.save_scan_checkpoint(tmp_path / "invalid", **values)
    assert not (tmp_path / "invalid" / "latest.json").exists()


def test_failed_atomic_publication_preserves_previous_commit_and_unrelated_files(tmp_path, monkeypatch):
    values = _fixture()
    path = tmp_path / "checkpoint"
    original = checkpoint.save_scan_checkpoint(path, **values)
    sentinel = path / "unrelated-experiment.txt"
    sentinel.write_text("preserve me", encoding="utf-8")

    def interrupted(*args):
        raise KeyboardInterrupt("synthetic interruption before atomic publication")

    monkeypatch.setattr(checkpoint.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        checkpoint.save_scan_checkpoint(path, **_fixture(2))
    assert checkpoint.inspect_scan_checkpoint(path, memory_budget_mib=8) == original
    assert _load(path, values)["completed_batches"] == 1
    assert sentinel.read_text(encoding="utf-8") == "preserve me"


def test_owned_generation_rotation_preserves_other_files_and_refuses_cursor_rewind(tmp_path):
    path = tmp_path / "checkpoint"
    old = checkpoint.save_scan_checkpoint(path, **_fixture(1))
    sentinel = path / "other.pt"
    sentinel.write_bytes(b"not our generation")
    checkpoint.save_scan_checkpoint(path, **_fixture(2))
    assert not (path / old["metadata"]["filename"]).exists()
    assert not (path / old["tensors"]["filename"]).exists()
    assert sentinel.read_bytes() == b"not our generation"
    with pytest.raises(ValueError, match="backwards"):
        checkpoint.save_scan_checkpoint(path, **_fixture(1))


def test_foreign_directory_and_old_partial_json_are_not_resumed_or_overwritten(tmp_path):
    path = tmp_path / "foreign"
    path.mkdir()
    partial = path / "partial.json"
    partial.write_text('{"scanned_samples": 700}', encoding="utf-8")
    with pytest.raises(ValueError, match="partial JSON"):
        checkpoint.save_scan_checkpoint(path, **_fixture())
    with pytest.raises(ValueError, match="old partial JSON"):
        _load(path, _fixture())
    assert partial.read_text(encoding="utf-8") == '{"scanned_samples": 700}'


def test_low_ram_and_declared_budget_refuse_before_copy_or_load(tmp_path, monkeypatch):
    values = _fixture()
    path = tmp_path / "checkpoint"
    checkpoint.save_scan_checkpoint(path, **values)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("over-budget tensor load"))
    with pytest.raises((MemoryError, ValueError), match="budget"):
        _load(path, values, memory_budget_mib=0.001)
    monkeypatch.setattr("asgcn_unet.diagnostic_resources._snapshot", lambda: {"headroom_bytes": 100})
    with pytest.raises(MemoryError, match="headroom"):
        _load(path, values)
    with pytest.raises(MemoryError, match="headroom"):
        checkpoint.save_scan_checkpoint(tmp_path / "new", **values)


def test_final_sequence_contract_and_schedule_are_required_for_exact_resume(tmp_path):
    values = _fixture()
    path = tmp_path / "checkpoint"
    checkpoint.save_scan_checkpoint(path, **values)
    finals = dict(values["sequence_final_indices"])
    finals[next(iter(finals))] += 1
    with pytest.raises(ValueError, match="metadata"):
        _load(path, values, sequence_final_indices=finals)
    with pytest.raises(ValueError, match="schedule"):
        _load(path, values, batches=list(reversed(values["batches"])))


def test_active_writer_refused_without_deleting_its_lock(tmp_path):
    values = _fixture()
    path = tmp_path / "checkpoint"
    checkpoint.save_scan_checkpoint(path, **values)
    with exclusive_artifact_writer(path) as lock:
        with pytest.raises(ArtifactWriterBusyError):
            checkpoint.save_scan_checkpoint(path, **values)
        assert lock.exists()


def test_import_and_integrity_inspection_do_not_import_torch(tmp_path):
    values = _fixture()
    path = tmp_path / "checkpoint"
    checkpoint.save_scan_checkpoint(path, **values)
    code = ("import sys; from asgcn_unet.stream_scan_checkpoint import inspect_scan_checkpoint; "
            f"inspect_scan_checkpoint({str(path)!r}, memory_budget_mib=8); assert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_manifest_cursor_or_filename_tampering_rejected(tmp_path):
    values = _fixture()
    path = tmp_path / "checkpoint"
    manifest = checkpoint.save_scan_checkpoint(path, **values)
    tampered = copy.deepcopy(manifest)
    tampered["tensors"]["filename"] = "../other.pt"
    tampered["commitment_sha256"] = checkpoint._digest({key: value for key, value in tampered.items()
                                                       if key != "commitment_sha256"})
    (path / "latest.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="file record"):
        _load(path, values)
