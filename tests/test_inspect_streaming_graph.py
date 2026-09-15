"""Synthetic config/workflow tests; no original data, training or GPU execution."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_streaming_graph.py"
SPEC = importlib.util.spec_from_file_location("inspect_streaming_graph", SCRIPT)
inspect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspect)


@pytest.fixture
def study(tmp_path):
    (tmp_path / "pyproject.toml").write_text("# synthetic checkout", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    (data / "26.h5").write_bytes(b"synthetic placeholder never opened as HDF5")
    config = {
        "dataset": {"type": "eventhdr", "root": "data", "max_events": None,
                    "event_time_contract": "physical_seconds_v1", "timestamp_scale_to_seconds": 0.5,
                    "interval_timestamp_scale_to_seconds": 0.25, "crop_size": None, "frame_stride": 1},
        "model": {"architecture_version": 3, "graph_execution": "event_driven", "encoder_kind": "graph",
                  "event_sampling_factor": 1, "graph_position_dims": 3, "graph_radius": 0.125,
                  "max_graph_edges": 17, "stream_config": {
                      "window_seconds": 0.03125, "time_scale_seconds": 0.0625,
                      "node_time_feature": "physical_frame_offset", "clock": "event_local_pending_off_v1",
                      "arrival_policy": "simultaneous_equal_timestamp"}},
        "train": {"batching": "independent_sequences", "validation_context_frames": None},
    }
    path = tmp_path / "train.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return tmp_path, path, config


def test_import_does_not_load_gpu_or_model_libraries():
    code = f"import runpy,sys; runpy.run_path({str(SCRIPT)!r},run_name='synthetic'); " \
           "assert not {'torch','numpy','h5py'} & set(sys.modules)"
    completed = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                               timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr


def test_plan_inherits_actual_geometry_without_defaults_or_source_change(study):
    root, path, _ = study
    before = path.read_bytes()
    plan = inspect.load_plan(path, "26.h5", 37, workspace=root)
    assert plan["arguments"] == {
        "source_file": root / "data" / "26.h5", "frame_index": 37,
        "window_seconds": 0.03125, "time_scale_seconds": 0.0625, "radius": 0.125,
        "timestamp_scale_to_seconds": 0.5, "interval_timestamp_scale_to_seconds": 0.25,
        "target_options": {},
    }
    assert plan["contract"]["settings_changed"] is False
    assert plan["contract"]["existing_max_graph_edges"] == 17
    assert plan["contract"]["training_executed"] is False
    assert path.read_bytes() == before


def test_plan_preserves_explicit_target_metadata_contract(study):
    root, path, config = study
    options = {"target_channels": 3, "target_normalization": {"mode": "known_scale", "scale": 2.0},
               "tone_map": "linear", "tone_map_mu": 3000.0}
    config["dataset"].update(options)
    path.write_text(json.dumps(config), encoding="utf-8")
    plan = inspect.load_plan(path, "26.h5", 0, workspace=root)
    assert plan["arguments"]["target_options"] == options


@pytest.mark.parametrize("section,key,value", [
    ("dataset", "max_events", 8192), ("dataset", "frame_stride", 2),
    ("dataset", "crop_size", [16, 16]), ("model", "graph_position_dims", 4),
    ("model", "event_sampling_factor", 2), ("model", "graph_radius", float("nan")),
])
def test_unsupported_contract_is_rejected_not_replaced(study, section, key, value):
    root, path, config = study
    config[section][key] = value
    path.write_text(json.dumps(config), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises((ValueError, TypeError)):
        inspect.load_plan(path, "26.h5", 0, workspace=root)
    assert path.read_bytes() == before


def test_selected_file_must_belong_to_declared_training_split(study):
    root, path, config = study
    config["dataset"]["split_manifest"] = "split.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    manifest = {"status": "final", "split_schema": "official_separate_roots_v1", "train_files": ["1.h5"]}
    (root / "split.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="training split"):
        inspect.load_plan(path, "26.h5", 0, workspace=root)


def test_allowed_files_is_honored_without_split_manifest(study):
    root, path, config = study
    config["dataset"]["allowed_files"] = ["1.h5"]
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="allowed_files"):
        inspect.load_plan(path, "26.h5", 0, workspace=root)


def test_file_key_normalization_matches_production():
    assert inspect._file_keys(["part\\26.h5", "./27.h5"]) == ["part/26.h5", "27.h5"]
    with pytest.raises(ValueError, match="Duplicate"):
        inspect._file_keys(["part\\26.h5", "part/26.h5"])


def test_manifest_change_refuses_success_report(monkeypatch, study):
    root, path, config = study
    config["dataset"]["split_manifest"] = "split.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    manifest = {"status": "final", "split_schema": "official_separate_roots_v1", "train_files": ["26.h5"]}
    manifest_path = root / "split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(inspect, "preflight", lambda **kwargs: {"synthetic": True})
    def mutate_manifest(**kwargs):
        manifest["train_files"] = ["1.h5"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return {"synthetic": True}
    monkeypatch.setattr(inspect, "audit_raw_event_graph", mutate_manifest)
    with pytest.raises(RuntimeError, match="manifest changed"):
        inspect.inspect_graph(config_path=path, source_file="26.h5", frame_index=0, cpu_threads=1,
                              memory_budget_mib=512, reserve_memory_mib=128, workspace=root)
    assert not (root / "runs").exists()


def test_resource_refusal_precedes_config_and_dataset_reads(monkeypatch, study):
    root, path, _ = study
    def refuse(**kwargs):
        raise RuntimeError("synthetic resource refusal")
    def no_plan(*args, **kwargs):
        pytest.fail("Plan read after a resource refusal")
    monkeypatch.setattr(inspect, "preflight", refuse)
    monkeypatch.setattr(inspect, "load_plan", no_plan)
    with pytest.raises(RuntimeError, match="resource refusal"):
        inspect.inspect_graph(config_path=path, source_file="26.h5", frame_index=0, cpu_threads=1,
                              memory_budget_mib=512, reserve_memory_mib=128, workspace=root)
    assert not (root / "runs").exists()


def test_workflow_uses_new_output_and_no_training(monkeypatch, study):
    root, path, _ = study
    calls = []
    monkeypatch.setattr(inspect, "preflight", lambda **kwargs: {"synthetic": True})
    def synthetic_audit(**kwargs):
        calls.append(kwargs)
        return {"synthetic": True, "report_eligible": False, "total_directed_edges": None,
                "window": {"nodes": 0}, "queries": [], "timings": {"audit_elapsed_s": 0.0}}
    monkeypatch.setattr(inspect, "audit_raw_event_graph", synthetic_audit)
    visual_calls = []
    def synthetic_html(report, output, **kwargs):
        visual_calls.append((report, output, kwargs))
        return output
    monkeypatch.setattr(inspect, "save_html", synthetic_html)
    original = path.read_bytes()
    first = inspect.inspect_graph(config_path=path, source_file="26.h5", frame_index=0, cpu_threads=1,
                                  memory_budget_mib=512, reserve_memory_mib=128, workspace=root)
    second = inspect.inspect_graph(config_path=path, source_file="26.h5", frame_index=0, cpu_threads=1,
                                   memory_budget_mib=512, reserve_memory_mib=128, workspace=root,
                                   count_all_nodes=True)
    assert first != second and first.is_file() and second.is_file()
    assert calls[0]["count_all_nodes"] is False
    assert all(call["include_point_cloud"] is True for call in calls)
    assert len(visual_calls) == 2
    assert visual_calls[0][1] == first.with_suffix(".html")
    assert calls[1]["count_all_nodes"] is True
    assert calls[0]["radius"] == 0.125
    assert json.loads(first.read_text())["training_config_inspection"]["training_executed"] is False
    assert path.read_bytes() == original


def test_terminal_summary_retains_unmeasured_total_and_omits_large_neighbor_trace():
    report = {"window": {"nodes": 8}, "total_directed_edges": None,
              "timings": {"audit_elapsed_s": 0.25}, "queries": [{
                  "raw_row_id": 10, "in_degree": 2, "all_window_sources_checked": 8,
                  "oracle_match": True, "neighbors": [{"synthetic_trace": "not for terminal"}],
              }]}
    summary = inspect.compact_report(report)
    assert summary["nodes"] == 8
    assert summary["total_directed_edges"] is None
    assert summary["queries"] == [{"raw_row_id": 10, "in_degree": 2,
                                   "all_window_sources_checked": 8, "oracle_match": True}]
    assert summary["full_graph_oracle_verified"] is False
