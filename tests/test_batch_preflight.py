from __future__ import annotations

import copy
import json

import h5py
import pytest
import torch

import asgcn_unet.preflight as profile
from asgcn_unet.data import build_dataset
from asgcn_unet.model import ASGCNUNet
from asgcn_unet.utils import save_json
from tests.fixtures import make_eventhdr
from tests.test_gpu_preflight import _config


def _batch_config(tmp_path, *, batch_size=2, streams=3, frames=4, empty=False):
    root = tmp_path / "hdr"
    for index in range(streams):
        path = make_eventhdr(root / f"scene-{index}", frames=frames)
        if empty:
            with h5py.File(path, "r+") as handle:
                handle["images/image000000000"].attrs["event_idx"] = 0
    config = _config(root)
    config["train"]["batch_size"] = batch_size
    if batch_size > 1:
        config["train"]["batching"] = "independent_sequences"
    return config


def _records(dataset, config):
    return [
        profile._sample_topology(dataset.get_topology_sample(index), config["model"], index)
        for index in range(len(dataset))
    ]


def _run(config, output, **kwargs):
    return profile.training_preflight(
        config, output, profile_samples=1, top_density_count=2, require_cuda=False, **kwargs
    )


def _finite_measurement():
    return {
        "loss": {"total": 0.3},
        "gradient_norm": 0.2,
        "amp_enabled": False,
        "amp": {"scale_before": 1.0, "scale_after": 1.0, "retries": 0},
        "step_time_ms": 2.0,
        "peak_allocated_mib": 100.0,
        "peak_reserved_mib": 120.0,
    }


@pytest.fixture
def fake_steps(monkeypatch):
    calls = []

    def frame(model, criterion, optimizer, scaler, raw, selected, config, device, step, **kwargs):
        return {
            "dataset_index": selected["dataset_index"],
            "sample_id": selected["sample_id"],
            "nodes": selected["model_sampled_events"],
            "actual_directed_edges": selected["actual_directed_edges"],
            "temporal_loss_applied": False,
            "temporal_context_sample_id": None,
            **_finite_measurement(),
        }

    def batch(
        model, criterion, optimizer, scaler, dataset, records, selected, config, device, *, fresh
    ):
        calls.append((list(selected["dataset_indices"]), fresh))
        history = (
            config["model"].get("recurrent", True)
            or (config["train"].get("loss_weights") or {}).get("temporal", 0.0) > 0
        )
        return {
            **selected,
            **_finite_measurement(),
            "execution": "disjoint_graph_batch_and_vectorized_decoder",
            "initialization": "fresh_training_seed" if fresh else "shared_dense_probe_model",
            "context_policy": "none" if fresh else "one_batched_predecessor_replay_training_mode",
            "context_indices": selected["predecessor_indices"]
            if not fresh and history
            else [None] * selected["batch_size"],
        }

    monkeypatch.setattr(profile, "_gpu_step", frame)
    monkeypatch.setattr(profile, "_gpu_batch_step", batch)
    return calls


def test_batch_plan_uses_metadata_schedule_and_retains_partial_tail(tmp_path, monkeypatch):
    config = _batch_config(tmp_path)
    dataset = build_dataset(config["dataset"], split="train")
    try:
        records = _records(dataset, config)

        def no_pixels(self, index):
            pytest.fail("Planning must not decode an image/event sample")

        monkeypatch.setattr(type(dataset), "__getitem__", no_pixels)
        plan = profile._batch_plan(dataset, records, config, 3)
        assert plan["requested_batch_size"] == plan["largest_actual_batch_size"] == 2
        assert plan["sequence_count"] == 3
        assert plan["scheduled_frames"] == plan["dataset_samples"] == 12
        assert plan["partial_batches"] == 4
        assert len(plan["selected_dense"]) == 3
        assert plan["selected_numerical"][0]["batch_index"] == 0
        assert "largest_actual_batch" in plan["selected_numerical"][0]["reasons"]
        totals = [entry["sum_actual_directed_edges"] for entry in plan["selected_dense"]]
        assert totals == sorted(totals, reverse=True)
    finally:
        dataset.close()


@pytest.mark.parametrize("streams", [1, 2])
def test_batch_gate_rejects_requested_capacity_without_enough_independent_streams(
    tmp_path, streams
):
    config = _batch_config(tmp_path, batch_size=3, streams=streams)
    dataset = build_dataset(config["dataset"], split="train")
    try:
        with pytest.raises(ValueError, match="Cannot form.*full batch_size=3"):
            profile._make_batch_sampler(dataset, config)
    finally:
        dataset.close()


def test_batch_plan_requires_actual_full_size_even_if_sequences_have_different_geometry(tmp_path):
    config = _batch_config(tmp_path, batch_size=2, streams=2)
    dataset = build_dataset(config["dataset"], split="train")
    try:
        for item in dataset.samples:
            item["sensor_size"] = (32, 48) if "scene-0" in item["source_file"] else (64, 48)
        with pytest.raises(ValueError, match="largest_geometry_compatible_batch=1"):
            profile._make_batch_sampler(dataset, config)
    finally:
        dataset.close()


def test_actual_cpu_batch_probe_calls_vectorized_model_and_backward(tmp_path, monkeypatch):
    config = _batch_config(tmp_path, empty=True)
    calls = []
    original = ASGCNUNet.forward_training_batch

    def observed(self, samples, states, **kwargs):
        calls.append(len(samples))
        return original(self, samples, states, **kwargs)

    monkeypatch.setattr(ASGCNUNet, "forward_training_batch", observed)
    report = _run(config, tmp_path / "profile.json")
    assert report["status"] == "diagnostic_passed", report["training_probe"]["failure"]
    assert report["report_eligible"] is False
    batched = report["batch_training_probe"]
    assert batched["passed"] is True
    assert 2 in calls
    assert len(batched["dense_steps"]) == 1
    assert batched["numerical_steps"][0]["batch_size"] == 2
    assert "contains_empty_frame" in batched["numerical_steps"][0]["reasons"]
    for measured in batched["dense_steps"] + batched["numerical_steps"]:
        assert measured["execution"] == "disjoint_graph_batch_and_vectorized_decoder"
        assert measured["gradient_norm"] >= 0
        assert measured["step_time_ms"] > 0
        assert measured["peak_allocated_mib"] is None


def test_batch_failure_cannot_pass_using_successful_framewise_probes(
    tmp_path, monkeypatch, fake_steps
):
    config = _batch_config(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("actual batched decoder failed")

    monkeypatch.setattr(profile, "_gpu_batch_step", fail)
    report = _run(config, tmp_path / "failed.json")
    assert len(report["training_probe"]["steps"]) == 1
    assert report["batch_training_probe"]["passed"] is False
    assert report["checks"]["forward_backward"] is False
    assert report["passed"] is False
    assert "batched decoder" in report["training_probe"]["failure"]["message"]


def _as_passed(report):
    report["status"] = "passed"
    report["passed"] = report["report_eligible"] = True
    report["request"]["require_cuda"] = True
    report["checks"]["cuda_available"] = report["checks"]["cuda_oom_free"] = True
    return report


def test_final_verifier_rebuilds_actual_batch_schedule_and_seals_full_batch(
    tmp_path, monkeypatch, fake_steps
):
    config = _batch_config(tmp_path)
    path = tmp_path / "profile.json"
    report = _as_passed(_run(config, path))
    save_json(path, report)
    monkeypatch.setattr(profile.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profile, "resolve_device", lambda value: torch.device("cuda"))
    monkeypatch.setattr(profile, "_current_source_contract", lambda: report["source_provenance"])
    monkeypatch.setattr(profile, "_runtime_provenance", lambda device: report["runtime_provenance"])
    monkeypatch.setattr(
        profile, "_topology_implementation_contract", lambda device: report["topology_contract"]
    )
    monkeypatch.setattr(
        profile,
        "_training_protocol",
        lambda config, device: report["training_probe"]["training_protocol"],
    )
    verified = profile.verify_training_preflight(config, path)
    assert verified["batch_size"] == 2
    assert verified["batch_preflight"]["largest_measured_batch_size"] == 2
    assert verified["batch_preflight"]["measured_batches"] >= 2
    report["batch_training_probe"]["plan"]["schedule_sha256"] = "0" * 64
    save_json(path, report)
    with pytest.raises(ValueError, match="actual dataset sequence schedule"):
        profile.verify_training_preflight(config, path)


@pytest.mark.parametrize(
    "change", ["missing", "size", "execution", "context", "loss", "amp", "incomplete", "selection"]
)
def test_batch_gate_rejects_forged_or_incomplete_measurement(tmp_path, fake_steps, change):
    config = _batch_config(tmp_path)
    report = _as_passed(_run(config, tmp_path / "profile.json"))
    batch = report["batch_training_probe"]
    if change == "missing":
        report["batch_training_probe"] = None
    elif change == "size":
        batch["numerical_steps"][0]["batch_size"] = 1
    elif change == "execution":
        batch["dense_steps"][0]["execution"] = "sequential_frames"
    elif change == "context":
        batch["numerical_steps"][0]["context_indices"] = [0, 1]
    elif change == "loss":
        batch["dense_steps"][0]["loss"]["total"] = float("nan")
    elif change == "amp":
        batch["numerical_steps"][0]["amp"]["retries"] = 1
    elif change == "incomplete":
        batch["dense_steps"].clear()
    else:
        batch["plan"]["selected_dense"][0]["dataset_indices"] = [0, 0]
    with pytest.raises((ValueError, TypeError), match="[Bb]atch|AMP"):
        profile._validate_batch_probe(report)


def test_topology_reuse_allows_batch_optimizer_and_decoder_changes_but_reruns_gpu_gate(
    tmp_path, fake_steps
):
    original_config = _batch_config(tmp_path, batch_size=1)
    old_path = tmp_path / "old.json"
    old = _as_passed(_run(original_config, old_path))
    old_source, old_commit = min(profile.LEGACY_V2_TOPOLOGY_SOURCES)
    old["source_provenance"] = {
        "source_tree_sha256": old_source,
        "git_commit": old_commit,
        "git_source_dirty": False,
    }
    old["topology_contract"]["implementation"] = {"historical": "audited old import dependencies"}
    save_json(old_path, old)
    before = old_path.read_bytes()
    config = copy.deepcopy(original_config)
    config["train"].update(batch_size=2, batching="independent_sequences", learning_rate=0.002)
    config["model"]["hidden_dim"] = 8
    fake_steps.clear()
    new = _run(config, tmp_path / "batch.json", reuse_report=old_path)
    assert new["passed"] is True, new["training_probe"]["failure"]
    assert new["scan_provenance"]["new_samples"] == 0
    assert new["topology"] == old["topology"]
    assert new["batch_training_probe"]["passed"] is True
    assert any(len(indices) == 2 for indices, fresh in fake_steps)
    assert old_path.read_bytes() == before


@pytest.mark.parametrize("change", ["source", "dirty", "torch", "graph", "seed", "original_hash"])
def test_old_v2_reuse_does_not_accept_unknown_sources_or_different_topology_inputs(
    tmp_path, fake_steps, change
):
    config = _batch_config(tmp_path, batch_size=1)
    path = tmp_path / "old.json"
    old = _as_passed(_run(config, path))
    source, commit = min(profile.LEGACY_V2_TOPOLOGY_SOURCES)
    old["source_provenance"] = {
        "source_tree_sha256": source,
        "git_commit": commit,
        "git_source_dirty": False,
    }
    old["topology_contract"]["implementation"] = {"historical": "audited"}
    requested = copy.deepcopy(config)
    requested["train"].update(batch_size=2, batching="independent_sequences")
    if change == "source":
        old["source_provenance"]["source_tree_sha256"] = "0" * 64
    elif change == "dirty":
        old["source_provenance"]["git_source_dirty"] = True
    elif change == "torch":
        old["topology_contract"]["torch"] = "unverified-runtime"
    elif change == "graph":
        requested["model"]["graph_radius"] += 0.01
    elif change == "seed":
        requested["seed"] += 1
    else:
        old["config_provenance"]["sha256"] = "0" * 64
    save_json(path, old)
    fake_steps.clear()
    report = _run(requested, tmp_path / "rejected.json", reuse_report=path)
    assert report["passed"] is False
    assert fake_steps == []
    assert json.loads(path.read_text())["source_provenance"] == old["source_provenance"]
