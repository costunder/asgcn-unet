"""Bounded CPU synthetic preflight tests, never measured research results."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from asgcn_unet import preflight, stream_preflight
from asgcn_unet.batching import SequenceBatchSampler

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _bounded_cpu_threads(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    _fixture_cpu_ram(monkeypatch)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "synthetic-project"
    directory = root / "configs" / "ablations"
    directory.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='synthetic-test-only'\n", encoding="utf-8")
    for kind in ("train", "hdr", "aid"):
        shutil.copyfile(PROJECT / "configs" / "ablations" / f"graph_unet-{kind}.json",
                        directory / f"graph_unet-{kind}.json")
    return root


def _prepare(project, destination="runs/synthetic-stream-test", **kwargs):
    options = {
        "window_seconds": 0.02, "time_scale_seconds": 0.1,
        "hdr_timestamp_scale_to_seconds": 0.5, "aid_timestamp_scale_to_seconds": 0.25,
        "hdr_interval_timestamp_scale_to_seconds": 0.125,
        "aid_interval_timestamp_scale_to_seconds": 1e-6,
    }
    options.update(kwargs)
    return stream_preflight.prepare_streaming_experiment(project, destination, **options)


def _config():
    with (PROJECT / "configs" / "ablations" / "graph_unet-train.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    # Explicit synthetic CPU smoke configuration, separate from checked-in final configs.
    config["device"] = "cpu"
    config["model"].update({
        "architecture_version": 3, "graph_execution": "event_driven", "spline_backend": "torch",
        "hidden_dim": 4, "graph_layers": 2, "decoder_channels": 4, "graph_radius": 0.8,
        "stream_config": {"window_seconds": 1.5, "time_scale_seconds": 10.0,
                          "node_time_feature": "physical_frame_offset",
                          "clock": stream_preflight.STREAM_CLOCK,
                          "arrival_policy": stream_preflight.STREAM_ARRIVAL_POLICY},
    })
    config["dataset"].update({
        "max_events": None, "event_time_contract": "physical_seconds_v1",
        "timestamp_scale_to_seconds": 1.0, "interval_timestamp_scale_to_seconds": 1.0,
    })
    config["train"].update({"batch_size": 2, "epochs": 1, "amp": False, "num_workers": 0,
                             "validation_context_frames": None})
    return config


class SyntheticStreams:
    """Two independent three-frame streams, with no external files or GPU."""

    def __init__(self, *, frames=3, dense_final=False):
        self.frames = frames
        self.dense_final = dense_final
        self.event_time_contract = "physical_seconds_v1"
        self.target_channels = 1
        self.crop_size = None
        self.samples = [
            {"sequence_id": f"synthetic-{lane}", "sequence_index": frame, "sensor_size": (16, 16),
             "start_idx": frame * 4, "end_idx": frame * 4 + (4 if dense_final and frame == frames - 1 else 2)}
            for lane in range(2) for frame in range(frames)
        ]
        self.closed = False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        frame = item["sequence_index"]
        offsets = [0.15, 0.35, 0.55, 0.85] if self.dense_final and frame == self.frames - 1 else [0.2, 0.8]
        metadata = dict(item)
        metadata["stream_time"] = {
            "schema": "physical_seconds_v1", "interval_start_seconds": float(frame),
            "interval_end_seconds": float(frame + 1), "sequence_origin_seconds": 0.0,
            "arrival_group_counts": [1] * len(offsets),
        }
        return {
            "sample_id": f"{item['sequence_id']}/{frame}", "metadata": metadata,
            "sensor_size": (16, 16), "target": torch.full((1, 16, 16), 0.25),
            "events": torch.tensor([[0, 0, frame + offset, 1 if n % 2 else -1] for n, offset in enumerate(offsets)], dtype=torch.float64),
            "event_ids": torch.tensor([[0, frame * 4 + n] for n in range(len(offsets))], dtype=torch.long),
        }

    def close(self):
        self.closed = True


def _fixture_cpu_ram(monkeypatch):
    # Synthetic CPU fixtures are not measurements of this host or its Job Object.
    # Production resource probes remain fail-closed and are never bypassed.
    headroom = 128 * 1024**2
    monkeypatch.setattr("asgcn_unet.diagnostic_resources._snapshot", lambda: {"headroom_bytes": headroom})
    def synthetic_resources(**kwargs):
        return {
            "memory": {"effective_available_bytes": headroom, "process_rss_bytes": 1024**2},
            "cpu": {"effective_cpu_limit": 4}, "allocation_limits_verified": True,
            "synthetic_cpu_test_only": True,
        }
    monkeypatch.setattr("asgcn_unet.resources.collect_runtime_resources", synthetic_resources)
    monkeypatch.setattr("asgcn_unet.stream_scan_loader.collect_runtime_resources", synthetic_resources)


def _fixture_provenance(monkeypatch, dataset):
    _fixture_cpu_ram(monkeypatch)
    data = {"dataset_type": "synthetic_cpu_test_only", "content": {"sha256": "a" * 64},
            "source_files": {"synthetic": True}, "transform": {}, "split_manifest": {}}
    source = {"source_tree_sha256": "b" * 64, "synthetic_cpu_test_only": True}
    monkeypatch.setattr("asgcn_unet.data.build_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr("asgcn_unet.engine._enforce_training_split_status", lambda config: None)
    monkeypatch.setattr(preflight, "_data_provenance", lambda *args: data)
    monkeypatch.setattr("asgcn_unet.engine._current_source_contract", lambda: source)
    monkeypatch.setattr("asgcn_unet.engine._dataset_source_fingerprint", lambda data: {"synthetic": True})


def test_prepare_preserves_full_baseline_outside_explicit_contract_changes(project):
    report = _prepare(project)
    assert report["execution_performed"] is False and report["report_eligible"] is False
    allowed = {"model.architecture_version", "model.graph_execution", "model.stream_config",
               "dataset.max_events", "dataset.event_time_contract", "dataset.timestamp_scale_to_seconds",
               "dataset.interval_timestamp_scale_to_seconds", "output.run_dir", "eval.output_dir",
               "train.validation_context_frames", "eval.recurrent_context_frames"}
    for kind, relative in report["configs"].items():
        result = json.loads((project / relative).read_text(encoding="utf-8"))
        stream_preflight.validate_streaming_contract(result, training=kind == "train")
        assert {change["field"] for change in report["changes"][kind]} <= allowed
        assert result["model"]["hidden_dim"] == 64 and result["model"]["graph_layers"] == 6
        assert result["model"]["decoder_channels"] == 48
        assert result["dataset"]["max_events"] is None
        if kind == "train":
            assert result["train"]["batch_size"] == 16 and result["train"]["epochs"] == 40
            assert result["train"]["max_train_samples"] is None and result["train"]["max_val_samples"] is None
        elif kind == "aid":
            assert result["dataset"]["timestamp_scale_to_seconds"] == 0.25
            assert result["dataset"]["interval_timestamp_scale_to_seconds"] == 1e-6
            assert result["eval"]["max_graph_edges_override"] == 7475202


def test_implicit_preparation_requires_explicit_storage_choice_and_keeps_scale(project):
    report = _prepare(project, graph_storage="implicit_radius")
    for kind, relative in report["configs"].items():
        result = json.loads((project / relative).read_text(encoding="utf-8"))
        assert result["model"]["graph_storage"] == "implicit_radius"
        assert result["model"]["graph_radius"] == 0.08
        assert result["model"]["event_sampling_factor"] == 1
        assert result["model"]["graph_layers"] == 6
        assert result["model"]["hidden_dim"] == 64
        assert result["model"]["decoder_channels"] == 48
        assert result["dataset"]["max_events"] is None
        assert any(row["field"] == "model.graph_storage" for row in report["changes"][kind])
        if kind == "train":
            assert result["train"]["batch_size"] == 16
            assert result["train"]["epochs"] == 40
    assert report["execution_performed"] is False


def test_unknown_graph_storage_is_rejected_before_preparation(project):
    with pytest.raises(ValueError, match="graph_storage"):
        _prepare(project, graph_storage="approximate")
    assert not (project / "runs" / "synthetic-stream-test").exists()


@pytest.mark.parametrize("name", ["window_seconds", "time_scale_seconds", "hdr_timestamp_scale_to_seconds",
                                  "aid_timestamp_scale_to_seconds", "hdr_interval_timestamp_scale_to_seconds",
                                  "aid_interval_timestamp_scale_to_seconds"])
def test_no_timestamp_or_window_default_is_invented(project, name):
    with pytest.raises(ValueError, match=name):
        _prepare(project, **{name: float("nan")})
    assert not (project / "runs" / "synthetic-stream-test").exists()


def test_prepare_refuses_existing_output_without_overwriting(project):
    report = _prepare(project)
    original = (project / report["configs"]["train"]).read_bytes()
    with pytest.raises(FileExistsError):
        _prepare(project)
    assert (project / report["configs"]["train"]).read_bytes() == original


@pytest.mark.parametrize("destination", ["runs/ablations/graph_unet/new", "configs/new-study", "data/new-study"])
def test_prepare_refuses_nested_or_overlapping_experiments(project, destination):
    with pytest.raises(ValueError, match="overlap"):
        _prepare(project, destination)
    assert not (project / destination).exists()


def test_preparation_import_is_torch_free_and_has_no_execution_side_effects():
    code = "import sys; from asgcn_unet.stream_preflight import prepare_streaming_experiment; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_prepare_cli_requires_all_six_explicit_time_arguments():
    spec = importlib.util.spec_from_file_location("prepare_streaming_test", PROJECT / "scripts" / "prepare_streaming_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as error:
        module.main(["--output-root", "runs/not-created-test"])
    assert error.value.code == 2


def test_new_preflight_dispatch_does_not_call_legacy_topology(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(stream_preflight, "streaming_training_preflight", lambda *args, **kwargs: calls.append(kwargs) or {"stream": True})
    monkeypatch.setattr(preflight, "_run_training_preflight", lambda *args, **kwargs: pytest.fail("legacy path"))
    assert preflight.training_preflight(_config(), tmp_path / "report.json", require_cuda=False) == {"stream": True}
    assert calls[0]["require_cuda"] is False


def test_legacy_preflight_dispatch_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "_run_training_preflight", lambda *args, **kwargs: {"legacy": True})
    assert preflight.training_preflight({"model": {"architecture_version": 2}}, tmp_path / "report.json") == {"legacy": True}


def test_legacy_report_rejected_before_any_device_probe(monkeypatch, tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"schema": "asgcn_training_preflight_v2", "passed": True}), encoding="utf-8")
    monkeypatch.setattr("asgcn_unet.utils.resolve_device", lambda value: pytest.fail("must reject before GPU probe"))
    with pytest.raises(ValueError, match="legacy/static"):
        preflight.verify_training_preflight(_config(), path)


def test_streaming_does_not_allow_legacy_report_reuse_or_partial_resume(tmp_path):
    for options in ({"reuse_report": "static.json"}, {"resume_scan": True}):
        with pytest.raises(ValueError, match="cannot reuse"):
            preflight.training_preflight(_config(), tmp_path / "not-created.json", **options)
    assert not (tmp_path / "not-created.json").exists()


def test_streaming_cli_refuses_unverified_training_before_train(monkeypatch, tmp_path):
    from asgcn_unet import cli
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    args = cli.build_parser().parse_args(["train", "--config", str(path), "--allow-unverified-preflight"])
    monkeypatch.setattr(cli, "train", lambda *args, **kwargs: pytest.fail("training must not start"))
    with pytest.raises(ValueError, match="legacy bypass"):
        cli._execute_command(args)


def test_actual_window_and_prefix_bound_are_distinct_and_retain_previous_frames():
    dataset = SyntheticStreams()
    config = _config()
    batches = list(SequenceBatchSampler(dataset, 2))
    topology = stream_preflight._scan_stream_topology(dataset, config, torch.device("cpu"), batches, top_density_count=3)
    assert topology["scan_complete"] is True and topology["scanned_samples"] == 6
    assert topology["arrival_prefix_peak_measured"] is False
    assert topology["max_readout_nodes"] == 3
    assert topology["max_prefix_union_nodes_upper_bound"] == 5
    assert any(row["readout_nodes"] > row["incoming_events"] for row in topology["samples"])
    assert topology["bulk_query_blocks"] > 0
    assert topology["bulk_pairwise_evaluations_avoided"] > 0
    for row in topology["samples"]:
        nodes, edges = row["readout_nodes"], row["readout_directed_edges"]
        assert row["readout_mean_in_degree"] == edges / nodes
        assert row["readout_edge_density"] == edges / (nodes * (nodes - 1))
        geometry = row["radius_geometry"]
        assert geometry["sensor_size_hw"] == [16, 16]
        assert geometry["spatial_semiaxes_pixels_xy"] == [12., 12.]
        assert geometry["temporal_semiaxis_seconds"] == 8.
        assert geometry["window_seconds"] == 1.5
    plan = stream_preflight._stream_probe_plan(batches, topology["samples"], 2, 1)
    assert plan["scheduled_frames"] == 6
    assert 0 in plan["selected_batch_indices"]
    assert plan["largest_actual_batch_size"] == 2


def test_cpu_full_synthetic_scan_and_real_training_probe_never_qualifies_as_gpu_result(monkeypatch, tmp_path):
    dataset = SyntheticStreams(frames=5, dense_final=True)
    _fixture_provenance(monkeypatch, dataset)
    config = _config()
    path = tmp_path / "cpu-smoke.json"
    report = preflight.training_preflight(config, path, profile_samples=1, top_density_count=2, require_cuda=False)
    assert report["passed"] is True, report["failure"]
    assert report["schema"] == stream_preflight.REPORT_SCHEMA
    assert report["report_eligible"] is False and report["status"] == "cpu_smoke_passed_non_reporting"
    assert dataset.closed
    probe = report["batch_training_probe"]
    assert probe["passed"] and probe["trainable_parameter_count"] > 0
    assert probe["replayed_predecessor_frames"] > 0
    assert all(step["gradient_norm"] >= 0 and step["step_time_ms"] > 0 for step in probe["steps"])
    assert any(step["incoming_contexts"] == 2 for step in probe["steps"])
    assert all(step["peak_allocated_mib"] is None for step in probe["steps"])
    with pytest.raises(ValueError, match="non-reporting"):
        preflight.verify_training_preflight(config, path)
    with pytest.raises(FileExistsError):
        preflight.training_preflight(config, path, require_cuda=False)


def test_conservative_bound_guard_failure_is_not_reported_as_actual_prefix_peak(monkeypatch, tmp_path):
    dataset = SyntheticStreams()
    _fixture_provenance(monkeypatch, dataset)
    config = _config()
    config["model"]["max_graph_edges"] = 8
    monkeypatch.setattr(stream_preflight, "_probe_stream_training", lambda *args: pytest.fail("guard must block training probe"))
    report = preflight.training_preflight(config, tmp_path / "bound-failure.json", profile_samples=1,
                                          top_density_count=2, require_cuda=False)
    assert not report["passed"] and not report["report_eligible"]
    assert "conservative" in report["failure"]["scope_note"]
    assert dataset.closed


@pytest.mark.parametrize("mutation_phase", [None, "scan", "probe"])
def test_preflight_rehashes_same_size_source_content_after_cpu_probes(monkeypatch, tmp_path, mutation_phase):
    """Real byte hashes around synthetic CPU work; no research data or GPU."""
    from asgcn_unet import engine

    _fixture_cpu_ram(monkeypatch)
    source_path = tmp_path / "synthetic-source.bin"
    initial_bytes, changed_bytes = b"synthetic-original", b"synthetic-modified"
    assert len(initial_bytes) == len(changed_bytes)
    source_path.write_bytes(initial_bytes)
    initial_stat = source_path.stat()
    dataset = SyntheticStreams()
    dataset.root, dataset.files = tmp_path, [source_path]
    config = _config()
    config["dataset"].update(root=str(tmp_path), split_manifest=None)
    initial_provenance = preflight._data_provenance(dataset, config)
    monkeypatch.setattr("asgcn_unet.data.build_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(engine, "_enforce_training_split_status", lambda config: None)
    monkeypatch.setattr(engine, "_current_source_contract", lambda: {"source_tree_sha256": "b" * 64})
    original_scan = stream_preflight._scan_stream_topology
    original_probe = stream_preflight._probe_stream_training

    def change_same_size_source():
        source_path.write_bytes(changed_bytes)
        os.utime(source_path, ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns))
        assert source_path.stat().st_size == initial_stat.st_size
        assert engine._dataset_source_fingerprint(dataset) == initial_provenance["source_files"]

    def scan_then_change(*args, **kwargs):
        result = original_scan(*args, **kwargs)
        if mutation_phase == "scan":
            change_same_size_source()
        return result

    def probe_then_change(*args, **kwargs):
        result = original_probe(*args, **kwargs)
        if mutation_phase == "probe":
            change_same_size_source()
        return result

    monkeypatch.setattr(stream_preflight, "_scan_stream_topology", scan_then_change)
    monkeypatch.setattr(stream_preflight, "_probe_stream_training", probe_then_change)
    output = tmp_path / "content-integrity-cpu-smoke.json"
    report = preflight.training_preflight(config, output, profile_samples=1,
                                          top_density_count=2, require_cuda=False)
    assert dataset.closed
    assert report["data_provenance"] == initial_provenance
    assert report["batch_training_probe"]["passed"] is True
    assert report["report_eligible"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == report
    if mutation_phase is None:
        assert report["passed"] is True and report["status"] == "cpu_smoke_passed_non_reporting"
        assert report["failure"] is None
    else:
        assert report["passed"] is False and report["status"] == "failed"
        assert report["failure"]["type"] == "ValueError"
        assert "Dataset content or provenance changed" in report["failure"]["message"]
        assert engine._dataset_content_fingerprint(dataset) != initial_provenance["content"]


def test_missing_or_changed_contract_fields_are_rejected():
    for modify in (
        lambda value: value["dataset"].update(max_events=8192),
        lambda value: value["model"]["stream_config"].update(clock="static_T"),
        lambda value: value["train"].update(validation_context_frames=32),
    ):
        config = _config()
        modify(config)
        with pytest.raises(ValueError):
            stream_preflight.validate_streaming_contract(config, training=True)
    config = _config()
    del config["dataset"]["interval_timestamp_scale_to_seconds"]
    with pytest.raises(TypeError, match="interval_timestamp_scale"):
        stream_preflight.validate_streaming_contract(config, training=True)


def test_tampered_stream_report_is_rejected_without_gpu(tmp_path):
    report = {"schema": stream_preflight.REPORT_SCHEMA, "passed": True, "arbitrary": "original"}
    report["commitment_sha256"] = stream_preflight._digest(report)
    report["arbitrary"] = "modified"
    with pytest.raises(ValueError, match="commitment mismatch"):
        stream_preflight._validated_report(report, tmp_path / "report.json")


def test_nonfull_requested_cuda_batch_refused_before_device_selection(monkeypatch, tmp_path):
    monkeypatch.setattr("asgcn_unet.utils.resolve_device", lambda value: pytest.fail("GPU should not be selected"))
    with pytest.raises(ValueError, match="physical batch_size=16"):
        preflight.training_preflight(_config(), tmp_path / "report.json", require_cuda=True)
