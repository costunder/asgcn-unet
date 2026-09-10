"""Synthetic CPU-only checks of report evidence; no CUDA telemetry is measured."""

from __future__ import annotations

import copy

import pytest

from asgcn_unet import stream_preflight as streaming


def _report(*, explicit=True, cpu=False, original_guard=4):
    old = {"model": {"max_graph_edges": original_guard, "graph_layers": 6},
           "train": {"batch_size": 16}}
    current = copy.deepcopy(old)
    if explicit:
        current["model"]["max_graph_edges"] = max(original_guard or 0, 8)
    row = {"dataset_index": 0, "sequence_identity": ["synthetic", "part"], "sequence_index": 0,
           "readout_nodes": 3, "readout_directed_edges": 6,
           "prefix_union_nodes_upper_bound": 4, "prefix_union_directed_edges_upper_bound": 8}
    topology = {"scan_complete": True, "samples": [row], **{
        f"max_{name}": row[name] for name in (
            "readout_nodes", "readout_directed_edges", "prefix_union_nodes_upper_bound",
            "prefix_union_directed_edges_upper_bound")}}
    reserve = 1024.0 if explicit else 0.0
    memory = ({"measured": False, "scope": "cpu_smoke_only", "reserve_vram_mib": reserve}
              if cpu else {
                  "measured": True, "scope": "current_device_snapshot_not_a_peak_guarantee",
                  "reserve_vram_mib": reserve, "total_mib": 16384.0, "free_mib": 12000.0,
                  "allocated_mib": 100.0, "reserved_mib": 200.0, "allocator_reusable_mib": 100.0,
                  "available_after_reserve_mib": 12100.0 - reserve,
              })
    snapshot = ({name: None for name in ("device_free_mib", "device_total_mib",
                                        "peak_allocated_mib", "peak_reserved_mib")}
                if cpu else {"device_free_mib": 10000.0, "device_total_mib": 16384.0,
                             "peak_allocated_mib": 3000.0, "peak_reserved_mib": 4000.0})
    probe = {
        "passed": True, "phase": "completed", "current_batch_index": None,
        "failed_batch_index": None, "completed_batches": 1, "plan": {"replay_stop_batch": 0},
        "reserve_vram_mib": reserve, "reserve_is_hard_isolation": False,
        "memory_scope": "model_setup_and_every_chronological_batch_input_through_state_commit_and_release",
        "reserve_scope": "live_device_free_pre_and_post_batch_and_allocator_peak_against_device_total",
        "peak_allocated_mib": snapshot["peak_allocated_mib"],
        "peak_reserved_mib": snapshot["peak_reserved_mib"],
        "minimum_observed_device_free_mib": None if cpu else 9500.0,
        "steps": [copy.deepcopy(snapshot)],
    }
    if not cpu:
        probe["last_memory_snapshot"] = copy.deepcopy(snapshot)
    measurement = {
        "configured_max_graph_edges": original_guard, "measured_union_required_guard": 8,
        "proposed_max_graph_edges": max(original_guard or 0, 8),
        "effective_max_graph_edges": current["model"]["max_graph_edges"],
        "explicit_measured_guard_requested": explicit,
        "selection": "max(original_guard_or_zero, full_train_prefix_union_edges, 1)",
        "topology_sha256": streaming._digest(topology), "evaluation_memory_certified": False,
        "raw_graph_storage_floor": streaming._graph_storage_floor(topology, [[0]]),
        "device_memory": memory,
    }
    if explicit:
        measurement["derived_config"] = "synthetic-new-config.json"
    return {
        "synthetic_cpu_test_only": True, "report_eligible": False,
        "request": {"use_measured_edge_guard": explicit, "require_cuda": not cpu,
                    "reserve_vram_mib": reserve},
        "checks": {"cuda_available": not cpu},
        "input_config_provenance": {"config": old, "sha256": streaming._digest(old)},
        "config_provenance": {"config": current, "sha256": streaming._digest(current)},
        "topology": topology, "guard_measurement": measurement, "batch_training_probe": probe,
    }


def _check(report):
    # A fresh outer commitment must not make internally inconsistent evidence valid.
    report["commitment_sha256"] = streaming._digest(report)
    streaming._validate_guard_measurement(report, report["config_provenance"]["config"], [[0]])


@pytest.mark.parametrize("explicit,cpu,guard", [
    (True, False, 4), (True, False, None), (True, True, 4),
    (False, False, 10), (False, False, None), (False, True, 10),
])
def test_consistent_synthetic_guard_and_memory_evidence(explicit, cpu, guard):
    report = _report(explicit=explicit, cpu=cpu, original_guard=guard)
    _check(report)
    assert report["report_eligible"] is False


@pytest.mark.parametrize("key", ["guard_measurement", "input_config_provenance"])
def test_new_guard_evidence_cannot_be_omitted(key):
    report = _report()
    del report[key]
    with pytest.raises(ValueError):
        _check(report)


@pytest.mark.parametrize("field,value", [
    ("configured_max_graph_edges", 5), ("measured_union_required_guard", 7),
    ("proposed_max_graph_edges", 7475202), ("effective_max_graph_edges", 9),
    ("explicit_measured_guard_requested", False), ("evaluation_memory_certified", True),
    ("selection", "static-profile"), ("topology_sha256", "bad"), ("derived_config", None),
])
def test_guard_selection_and_provenance_tampering_is_rejected(field, value):
    report = _report()
    report["guard_measurement"][field] = value
    with pytest.raises(ValueError):
        _check(report)


def test_other_model_changes_are_rejected_even_with_new_config_digest():
    report = _report()
    provenance = report["config_provenance"]
    provenance["config"]["model"]["graph_layers"] = 5
    provenance["sha256"] = streaming._digest(provenance["config"])
    with pytest.raises(ValueError, match="only"):
        _check(report)


@pytest.mark.parametrize("change", ["cached-max", "count-type", "floor"])
def test_cached_counts_and_storage_floor_are_recomputed(change):
    report = _report()
    if change == "cached-max":
        report["topology"]["max_readout_nodes"] = 4
        report["guard_measurement"]["topology_sha256"] = streaming._digest(report["topology"])
    elif change == "count-type":
        report["topology"]["samples"][0]["readout_nodes"] = True
    else:
        report["guard_measurement"]["raw_graph_storage_floor"]["bytes"] += 1
    with pytest.raises(ValueError):
        _check(report)


@pytest.mark.parametrize("field,value", [
    ("reserve_vram_mib", 1023.0), ("allocator_reusable_mib", 99.0),
    ("available_after_reserve_mib", 12000.0), ("allocated_mib", 201.0),
    ("free_mib", float("nan")), ("measured", False), ("scope", "hard-guarantee"),
])
def test_device_snapshot_arithmetic_and_reserve_are_checked(field, value):
    report = _report()
    report["guard_measurement"]["device_memory"][field] = value
    with pytest.raises(ValueError):
        _check(report)


@pytest.mark.parametrize("field,value", [
    ("reserve_vram_mib", 1023), ("minimum_observed_device_free_mib", 1023),
    ("minimum_observed_device_free_mib", 10001), ("peak_reserved_mib", 16000),
    ("peak_allocated_mib", 2000), ("completed_batches", 0),
    ("reserve_is_hard_isolation", True), ("memory_scope", "forward-only"),
    ("reserve_scope", "allocator-only"), ("phase", "failed"), ("last_memory_snapshot", None),
])
def test_probe_coverage_peak_and_free_memory_are_checked(field, value):
    report = _report()
    report["batch_training_probe"][field] = value
    with pytest.raises((ValueError, TypeError)):
        _check(report)


@pytest.mark.parametrize("field,value", [
    ("device_free_mib", 1023), ("device_total_mib", 32000),
    ("peak_allocated_mib", 4001), ("peak_reserved_mib", 4001),
])
def test_each_measured_step_is_checked_against_probe_summaries(field, value):
    report = _report()
    report["batch_training_probe"]["steps"][0][field] = value
    with pytest.raises(ValueError):
        _check(report)


def test_cpu_evidence_cannot_be_promoted_to_cuda():
    report = _report(cpu=True)
    report["report_eligible"] = True
    with pytest.raises(ValueError, match="CPU smoke"):
        _check(report)


def test_cpu_evidence_cannot_contain_invented_gpu_measurements():
    report = _report(cpu=True)
    report["batch_training_probe"]["steps"][0]["device_free_mib"] = 2000
    with pytest.raises(ValueError, match="invented"):
        _check(report)


def test_float_round_trip_tolerance_is_small_and_explicit():
    report = _report()
    report["guard_measurement"]["device_memory"]["available_after_reserve_mib"] += 1e-8
    _check(report)


@pytest.mark.parametrize("explicit", [False, True])
def test_actual_cpu_smoke_report_matches_guard_verifier(monkeypatch, tmp_path, explicit):
    import torch

    from asgcn_unet.preflight import _make_batch_sampler
    from tests.test_stream_preflight import SyntheticStreams, _config, _fixture_provenance

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dataset = SyntheticStreams()
        _fixture_provenance(monkeypatch, dataset)
        config = _config()
        config["model"]["max_graph_edges"] = 8 if explicit else 100
        options = ({"measured_guard_config_output": tmp_path / "derived.json",
                    "reserve_vram_mib": 1024} if explicit else {})
        report = streaming.streaming_training_preflight(
            config, tmp_path / "report.json", require_cuda=False,
            profile_samples=1, top_density_count=2, **options,
        )
        assert report["passed"], report["failure"]
        assert report["report_eligible"] is False
        streaming._validate_guard_measurement(
            report, report["config_provenance"]["config"], list(_make_batch_sampler(dataset, config)),
        )
    finally:
        torch.set_num_threads(previous_threads)
