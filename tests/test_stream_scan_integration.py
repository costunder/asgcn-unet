"""Small synthetic CPU checks; no server, research data, or GPU execution."""

import copy
import json

import pytest
import torch

from asgcn_unet import stream_preflight
from asgcn_unet.batching import SequenceBatchSampler
from tests.test_stream_preflight import (
    SyntheticStreams,
    _config,
    _fixture_cpu_ram,
    _fixture_provenance,
)


@pytest.fixture(autouse=True)
def bounded_threads(monkeypatch):
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    _fixture_cpu_ram(monkeypatch)
    yield
    torch.set_num_threads(prior)


def test_observed_memory_impossibility_stops_after_first_complete_batch():
    dataset = SyntheticStreams(frames=5)
    config = _config()
    progress, completed = {}, []
    with pytest.raises(RuntimeError, match="first proven-impossible batch"):
        stream_preflight._scan_stream_topology(
            dataset, config, torch.device("cpu"), list(SequenceBatchSampler(dataset, 2)),
            top_density_count=3, progress=progress, training_storage_budget_mib=1e-9,
            on_batch_complete=lambda states, topology, cursor, finals: completed.append(cursor),
        )
    assert completed == [1]
    assert progress["scanned_samples"] == 2
    assert progress["scan_complete"] is False
    assert progress["observed_training_storage_floor"]["measured_peak"] is False
    assert progress["phase_timing_ms"]["total_input_ms"] >= 0
    # Both two-node lanes are certified complete cells. The full counts still
    # prove the storage floor, without individual pair-distance evaluations.
    assert progress["candidate_pairs_visited"] == 0
    assert progress["bulk_query_blocks"] > 0
    assert progress["bulk_pairwise_evaluations_avoided"] == 2


def test_implicit_storage_does_not_use_materialized_edge_memory_floor():
    dataset, config = SyntheticStreams(), _config()
    config["model"]["graph_storage"] = "implicit_radius"
    progress = stream_preflight._scan_stream_topology(
        dataset, config, torch.device("cpu"), list(SequenceBatchSampler(dataset, 2)),
        top_density_count=3, training_storage_budget_mib=1e-9,
    )
    assert progress["scan_complete"]
    assert "observed_training_storage_floor" not in progress
    materialized = stream_preflight._graph_storage_floor(progress, list(SequenceBatchSampler(dataset, 2)))
    implicit = stream_preflight._graph_storage_floor(progress, list(SequenceBatchSampler(dataset, 2)), "implicit_radius")
    assert implicit["bytes"] < materialized["bytes"]
    assert implicit["total_training_peak_estimate"] is False


def test_interrupted_preflight_resumes_raw_checkpoint_without_recounting_prefix(monkeypatch, tmp_path):
    dataset, config = SyntheticStreams(frames=4), _config()
    _fixture_provenance(monkeypatch, dataset)
    original_scan = stream_preflight._scan_stream_topology
    first_checkpoint = tmp_path / "first-checkpoint"
    seen_first, seen_resumed = [], []

    def interrupt_after_checkpoint(*args, **kwargs):
        complete = kwargs["on_batch_complete"]
        def completed(states, topology, cursor, finals):
            complete(states, topology, cursor, finals)
            seen_first.append(cursor)
            raise KeyboardInterrupt()
        kwargs["on_batch_complete"] = completed
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(stream_preflight, "_scan_stream_topology", interrupt_after_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        stream_preflight.streaming_training_preflight(
            config, tmp_path / "interrupted.json", require_cuda=False,
            profile_samples=1, top_density_count=2, scan_checkpoint_dir=first_checkpoint,
        )
    saved = json.loads((tmp_path / "interrupted.json").read_text(encoding="utf-8"))
    assert saved["status"] == "interrupted" and not saved["report_eligible"]
    assert seen_first == [1]
    assert (first_checkpoint / "latest.json").is_file()
    preserved = (first_checkpoint / "latest.json").read_bytes()

    def observe_resumed(*args, **kwargs):
        assert kwargs["resume_state"]["completed_batches"] == 1
        complete = kwargs["on_batch_complete"]
        def completed(states, topology, cursor, finals):
            complete(states, topology, cursor, finals)
            seen_resumed.append(cursor)
        kwargs["on_batch_complete"] = completed
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(stream_preflight, "_scan_stream_topology", observe_resumed)
    report = stream_preflight.streaming_training_preflight(
        config, tmp_path / "resumed.json", require_cuda=False, profile_samples=1, top_density_count=2,
        scan_checkpoint_dir=tmp_path / "new-checkpoint", resume_checkpoint=first_checkpoint,
    )
    assert report["passed"], report["failure"]
    assert not report["report_eligible"]
    assert seen_resumed == [2, 3, 4]
    assert report["topology"]["resumed_completed_batches"] == 1
    assert (first_checkpoint / "latest.json").read_bytes() == preserved


def test_partial_json_alone_is_not_accepted_as_raw_resume(monkeypatch, tmp_path):
    dataset, config = SyntheticStreams(), _config()
    _fixture_provenance(monkeypatch, dataset)
    partial = tmp_path / "no-state"
    partial.mkdir()
    (partial / "stream-profile.json").write_text('{"topology":{"scanned_samples":2}}', encoding="utf-8")
    report = stream_preflight.streaming_training_preflight(
        config, tmp_path / "refused.json", require_cuda=False, profile_samples=1, top_density_count=2,
        resume_checkpoint=partial,
    )
    assert not report["passed"] and report["stage"] == "restore_scan_checkpoint"
    assert report["topology"] is None and report["batch_training_probe"] is None


def test_resume_impossible_saved_floor_refuses_before_any_gpu_transfer(monkeypatch):
    dataset, config = SyntheticStreams(frames=3), _config()
    batches = list(SequenceBatchSampler(dataset, 2))
    captured = {}
    def checkpoint(states, topology, cursor, finals):
        captured.update(states=states.copy(), topology=copy.deepcopy(topology), completed_batches=cursor)
    with pytest.raises(RuntimeError, match="first proven-impossible batch"):
        stream_preflight._scan_stream_topology(
            dataset, config, torch.device("cpu"), batches, top_density_count=2,
            on_batch_complete=checkpoint, training_storage_budget_mib=1e-9,
        )
    def forbidden(*args, **kwargs):
        raise AssertionError("The failed resume must not load a next frame")
    monkeypatch.setattr("asgcn_unet.stream_scan_loader.iter_scan_batches", forbidden)
    # CPU-only test: reaching any transfer to this fake CUDA device would fail.
    with pytest.raises(RuntimeError, match="no saved nodes were transferred"):
        stream_preflight._scan_stream_topology(
            dataset, config, torch.device("cuda:0"), batches, top_density_count=2,
            resume_state=captured, training_storage_budget_mib=1e-9,
        )


def test_complete_raw_scan_is_copied_without_scanning_again(monkeypatch, tmp_path):
    dataset, config = SyntheticStreams(), _config()
    _fixture_provenance(monkeypatch, dataset)
    first = tmp_path / "first-checkpoint"
    report = stream_preflight.streaming_training_preflight(
        config, tmp_path / "first.json", require_cuda=False, profile_samples=1,
        top_density_count=2, scan_checkpoint_dir=first,
    )
    assert report["passed"], report["failure"]
    preserved = (first / "latest.json").read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed topology must not be recounted")
    monkeypatch.setattr("asgcn_unet.stream_topology.count_stream_topology_update", forbidden)
    report = stream_preflight.streaming_training_preflight(
        config, tmp_path / "second.json", require_cuda=False, profile_samples=1,
        top_density_count=2, scan_checkpoint_dir=tmp_path / "second-checkpoint", resume_checkpoint=first,
    )
    assert report["passed"], report["failure"]
    assert report["topology"]["resumed_completed_batches"] == 3
    assert report["scan_checkpoint"]["completed_batches"] == 3
    assert (tmp_path / "second-checkpoint" / "latest.json").is_file()
    assert (first / "latest.json").read_bytes() == preserved
