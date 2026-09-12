"""CPU-only synthetic v4 preparation, exact scan resume and real-model probes."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from asgcn_unet import stream_preflight as preflight
from asgcn_unet import stream_scan_checkpoint as checkpoint
from asgcn_unet.batching import SequenceBatchSampler, sequence_key
from tests.test_stream_preflight import (
    PROJECT,
    SyntheticStreams,
    _config,
    _fixture_cpu_ram,
    _fixture_provenance,
    _prepare,
)
from tests.test_stream_preflight import (
    project as project,  # noqa: PLC0414 - pytest fixture re-export
)


@pytest.fixture(autouse=True)
def bounded_cpu(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    _fixture_cpu_ram(monkeypatch)
    yield
    torch.set_num_threads(previous)


def _hierarchy(*, factor=3):
    config = _config()
    config["model"].update({"architecture_version": 4, "event_sampling_factor": factor,
                            "graph_storage": "implicit_radius", "hierarchy_config": {
                                "after_layer": 1, "spatial_cell_pixels": 4,
                                "temporal_cell_seconds": 0.5,
                                "edge_pseudo": "mean_fine_distance_over_radius"}})
    return config


def _save_scan(tmp_path, dataset, config, *, stop=1):
    batches = list(SequenceBatchSampler(dataset, config["train"]["batch_size"]))
    identity = {field: "a" * 64 for field in checkpoint.IDENTITY_FIELDS}
    identity.update(config_sha256=checkpoint._digest(config), schedule_sha256=checkpoint._digest(batches))
    finals = {sequence_key(item): item["sequence_index"] for item in dataset.samples}
    path = tmp_path / "raw-scan"
    def save(states, topology, completed, final_indices):
        checkpoint.save_scan_checkpoint(path, identity=identity, batches=batches,
                                        completed_batches=completed, states=states, topology=topology,
                                        sequence_final_indices=final_indices, memory_budget_mib=32)
    result = preflight._scan_stream_topology(dataset, config, torch.device("cpu"), batches,
                                             top_density_count=2, on_batch_complete=save,
                                             pause_after_completed_batches=stop)
    restored = checkpoint.load_scan_checkpoint(path, expected_identity=identity, batches=batches,
                                               sequence_final_indices=finals, memory_budget_mib=32)
    return result, restored, batches, identity, path, finals


def test_preparation_v4_is_explicit_preserves_full_model_and_records_design(project):
    hierarchy = {"after_layer": 4, "spatial_cell_pixels": 4, "temporal_cell_seconds": 0.008,
                 "edge_pseudo": "mean_fine_distance_over_radius"}
    report = _prepare(project, hierarchy_config=hierarchy, event_sampling_factor=10, graph_storage="implicit_radius")
    assert report["architecture_design"]["paper_exact_hyperparameters"] is False
    assert report["architecture_design"]["new_training_required"] is True
    for kind, path in report["configs"].items():
        config = json.loads((project / path).read_text(encoding="utf-8"))
        assert config["model"]["architecture_version"] == 4
        assert config["model"]["event_sampling_factor"] == 10
        assert config["model"]["hierarchy_config"] == hierarchy
        assert config["dataset"]["max_events"] is None
        assert (config["model"]["graph_layers"], config["model"]["hidden_dim"], config["model"]["decoder_channels"]) == (6, 64, 48)
        if kind == "train":
            assert (config["train"]["batch_size"], config["train"]["epochs"]) == (16, 40)
    with pytest.raises(ValueError, match="v3 remains R=1"):
        _prepare(project, "runs/invalid-no-hierarchy", event_sampling_factor=2)


def test_v4_preparation_and_validator_stay_torch_free(project):
    code = (
        "import sys; from asgcn_unet.stream_preflight import prepare_streaming_experiment; "
        f"prepare_streaming_experiment({str(project)!r}, 'runs/torch-free', window_seconds=1, time_scale_seconds=1, "
        "hdr_timestamp_scale_to_seconds=1, aid_timestamp_scale_to_seconds=1, "
        "hdr_interval_timestamp_scale_to_seconds=1, aid_interval_timestamp_scale_to_seconds=1, "
        "hierarchy_config={'after_layer':4,'spatial_cell_pixels':4,'temporal_cell_seconds':0.1,"
        "'edge_pseudo':'mean_fine_distance_over_radius'},event_sampling_factor=2); "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True)


def test_cli_explicit_hierarchy_uses_recorded_raster_and_radius_design(monkeypatch, project):
    spec = importlib.util.spec_from_file_location("prepare_hierarchy_test", PROJECT / "scripts" / "prepare_streaming_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROJECT", project)
    arguments = ["--output-root", "runs/cli-hierarchy", "--window-seconds", "1", "--time-scale-seconds", "2",
                 "--hdr-timestamp-scale-to-seconds", "1", "--aid-timestamp-scale-to-seconds", "1",
                 "--hdr-interval-timestamp-scale-to-seconds", "1", "--aid-interval-timestamp-scale-to-seconds", "1",
                 "--hierarchical", "--event-sampling-factor", "7"]
    assert module.main(arguments) == 0
    baseline = json.loads((project / "configs/ablations/graph_unet-train.json").read_text())["model"]
    config = json.loads((project / "runs/cli-hierarchy/configs/train.json").read_text())
    assert config["model"]["hierarchy_config"] == {
        "after_layer": 4, "spatial_cell_pixels": baseline["raster_downsample"],
        "temporal_cell_seconds": baseline["graph_radius"] * 2, "edge_pseudo": "mean_fine_distance_over_radius"}


def test_sequence_sampling_counts_and_exact_checkpoint_resume_match_uninterrupted(tmp_path):
    dataset, config = SyntheticStreams(), _hierarchy()
    partial, restored, batches, identity, path, finals = _save_scan(tmp_path, dataset, config)
    assert not partial["scan_complete"] and partial["scanned_samples"] == 2
    assert all(state.sampling_offset == 2 and state.last_event_id == (0, 1) for state in restored["states"].values())
    resumed = preflight._scan_stream_topology(dataset, config, torch.device("cpu"), batches,
                                             top_density_count=2, resume_state=restored)
    complete = preflight._scan_stream_topology(dataset, config, torch.device("cpu"), batches, top_density_count=2)
    assert resumed["samples"] == complete["samples"] and resumed["scan_complete"]
    rows = complete["samples"][:3]
    assert [row["incoming_events"] for row in rows] == [2, 2, 2]
    assert [row["sampled_incoming_events"] for row in rows] == [1, 1, 0]
    assert [row["sampling_offset_after"] for row in rows] == [2, 4, 6]
    assert [row["readout_nodes"] for row in rows] == [1, 1, 1]
    changed = dict(identity, config_sha256="c" * 64)
    with pytest.raises(ValueError, match="identity mismatch"):
        checkpoint.load_scan_checkpoint(path, expected_identity=changed, batches=batches,
                                        sequence_final_indices=finals, memory_budget_mib=32)


def test_sampler_does_not_hide_invalid_unselected_raw_event():
    class InvalidRaw(SyntheticStreams):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["events"][1, 3] = 0  # R=3 would discard this row, but raw validation runs first.
            return sample
    dataset = InvalidRaw()
    with pytest.raises(ValueError, match="Invalid physical sensor event"):
        preflight._scan_stream_topology(dataset, _hierarchy(), torch.device("cpu"),
                                        list(SequenceBatchSampler(dataset, 2)), top_density_count=2)


def test_raw_cross_frame_identity_is_checked_before_sampling():
    class RepeatedRaw(SyntheticStreams):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            if sample["metadata"]["sequence_index"] == 1:
                sample["event_ids"][0, 1] = 0  # Unselected ordinal 2 repeats a raw ID from frame 0.
            return sample
    dataset = RepeatedRaw()
    with pytest.raises(ValueError, match="Raw event identity"):
        preflight._scan_stream_topology(dataset, _hierarchy(), torch.device("cpu"),
                                        list(SequenceBatchSampler(dataset, 2)), top_density_count=2)


@pytest.mark.parametrize("mutation", ["offset", "missing_offset", "last_id", "record_count"])
def test_checkpoint_rejects_sampling_state_tampering(tmp_path, mutation):
    dataset, config = SyntheticStreams(), _hierarchy()
    _, restored, batches, identity, _, finals = _save_scan(tmp_path, dataset, config)
    state = next(iter(restored["states"].values()))
    if mutation == "offset":
        state.sampling_offset += 1
    elif mutation == "missing_offset":
        del state.sampling_offset
    elif mutation == "last_id":
        state.last_event_id = (0, 0)
    else:
        restored["topology"]["samples"][0]["sampled_incoming_events"] = 0
    with pytest.raises(ValueError, match="sampling_offset|last_event_id|sampled count"):
        checkpoint.save_scan_checkpoint(tmp_path / "invalid", identity=identity, batches=batches,
                                        completed_batches=1, states=restored["states"], topology=restored["topology"],
                                        sequence_final_indices=finals, memory_budget_mib=32)


def test_real_cpu_v4_early_gate_then_full_scan_and_representative_probe(monkeypatch, tmp_path):
    dataset, config = SyntheticStreams(), _hierarchy()
    _fixture_provenance(monkeypatch, dataset)
    report = preflight.streaming_training_preflight(config, tmp_path / "v4-smoke.json", require_cuda=False,
                                                    profile_samples=1, top_density_count=2)
    assert report["passed"], report.get("failure")
    assert report["report_eligible"] is False
    assert report["topology"]["scan_complete"] and report["topology"]["scanned_samples"] == len(dataset)
    assert report["early_training_probe"]["probe"]["passed"]
    assert report["early_training_probe"]["physical_batch_size"] == 2
    assert report["early_training_probe"]["probe"]["steps"][0]["hierarchy_readout"]["topology_scan_complete"] is False
    assert report["topology"]["resumed_completed_batches"] == 1
    assert report["batch_training_probe"]["passed"]
    batches = list(SequenceBatchSampler(dataset, 2))
    preflight._validate_hierarchy_evidence(report, config, batches)
    tampered = copy.deepcopy(report)
    tampered["topology"]["samples"][0]["sampling_offset_after"] += 1
    with pytest.raises(ValueError, match="sampling counter"):
        preflight._validate_hierarchy_evidence(tampered, config, batches)
    tampered = copy.deepcopy(report)
    tampered["batch_training_probe"]["steps"][0].pop("hierarchy_readout")
    with pytest.raises(ValueError, match="hierarchy graph/memory evidence"):
        preflight._validate_hierarchy_evidence(tampered, config, batches)
    for step in report["batch_training_probe"]["steps"]:
        assert step["hierarchy_readout"]["peak_memory_includes_hierarchy"]
        assert step["hierarchy_readout"]["topology_scan_complete"] is True
        assert len(step["hierarchy_readout"]["counts"]) == step["batch_size"]
        assert step["peak_allocated_mib"] is None  # CPU smoke is not a VRAM measurement.


def test_early_gate_failure_keeps_committed_raw_state_and_does_not_scan_next_batch(monkeypatch, tmp_path):
    dataset, config = SyntheticStreams(), _hierarchy()
    _fixture_provenance(monkeypatch, dataset)
    def fail_probe(*args, **kwargs):
        kwargs["progress"].update(passed=False, phase="synthetic_before_forward_failure")
        raise RuntimeError("synthetic early allocation failure")
    monkeypatch.setattr(preflight, "_probe_stream_training", fail_probe)
    report = preflight.streaming_training_preflight(config, tmp_path / "failed-smoke.json", require_cuda=False,
                                                    profile_samples=1, top_density_count=2)
    assert not report["passed"] and report["failure"]["stage"] == "early_physical_training_probe"
    assert report["failure"]["probe_phase"] == "synthetic_before_forward_failure"
    assert report["scan_checkpoint"]["completed_batches"] == 1
    assert report["topology"]["scanned_samples"] == 2 and not report["topology"]["scan_complete"]
    checkpoint_path = Path(report["scan_checkpoint"]["path"])
    # Artifact labels may be redacted; the owned on-disk directory is deterministic.
    if not checkpoint_path.exists():
        checkpoint_path = tmp_path / "failed-smoke.scan-checkpoint"
    assert checkpoint_path.exists()
    assert dataset.closed
