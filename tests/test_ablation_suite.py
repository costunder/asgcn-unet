"""Synthetic, read-only/mocked orchestration tests; never train or use CUDA."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from asgcn_unet import ablation_suite as suite

PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("family", suite.FAMILIES)
@pytest.mark.parametrize("split", ["train", "hdr", "aid"])
def test_full_configs_only_change_requested_architecture_and_paths(family, split):
    actual = json.loads(suite.config_path(PROJECT, family, split).read_text())
    assert actual == suite.expected_config(PROJECT, family, split)
    assert actual["dataset"]["max_events"] == 8192
    assert actual["dataset"]["crop_size"] is None
    assert actual["model"]["event_sampling_factor"] == 1
    if split == "train":
        assert actual["train"]["epochs"] == 40
        assert actual["train"]["batch_size"] == 16
        assert actual["train"]["max_train_samples"] is None
        assert actual["train"]["max_val_samples"] is None
    else:
        assert actual["eval"]["max_samples"] is None


def test_full_plan_contains_all_families_modes_datasets_and_guards():
    commands = suite.plan_commands(PROJECT)
    assert sum(c.stage == "train" for c in commands) == 4
    assert sum(c.stage == "profile" for c in commands) == 4
    assert sum(c.stage == "calibrate" for c in commands) == 2
    assert sum(c.stage == "evaluate" for c in commands) == 40
    assert sum(c.stage == "benchmark" for c in commands) == 40
    assert commands[0].stage == "environment"
    assert max(i for i, command in enumerate(commands) if command.stage == "profile") < min(
        i for i, command in enumerate(commands) if command.stage == "train"
    )
    for command in commands:
        assert "--allow-unverified-preflight" not in command.argv
        assert "--allow-unsealed-checkpoint-for-non-reporting" not in command.argv
        assert "--overwrite" not in command.argv
        if command.stage == "calibrate":
            assert suite._option(command.argv, "--samples") == "all"
        if command.family in {"unet", "transformer"}:
            assert "snn" not in command.argv


def test_plan_has_no_execution_or_environment_mutation(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "assigned-MIG")
    before = dict(suite.os.environ)

    def forbidden(*args, **kwargs):
        raise AssertionError("Plan launched a process")

    suite.execute_commands(
        suite.plan_commands(PROJECT), PROJECT, runner=forbidden, emit=lambda _: None
    )
    assert dict(suite.os.environ) == before


def test_explicit_execution_preserves_gpu_allocation_and_stops_at_failure(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "assigned-MIG")
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "assigned-MIG"
        assert kwargs["check"] is True
        assert "shell" not in kwargs
        raise subprocess.CalledProcessError(75, argv)

    with pytest.raises(subprocess.CalledProcessError):
        suite.execute_commands(
            suite.plan_commands(PROJECT), PROJECT, execute=True, runner=runner, emit=lambda _: None
        )
    assert len(calls) == 1


def test_summary_missing_is_not_zero_or_old_fast_results():
    rows = suite.collect_summary(PROJECT)
    assert len(rows) == 40
    assert {r["group"] for r in rows} == {"A", "B", "B-ANN-control", "C", "D", "E"}
    assert all(r["frames"] is None and r["micro_psnr"] is None for r in rows)
    assert all("runs" in r["run"] and "ablations" in r["run"] for r in rows)
    assert "N/A" in suite.render_summary(rows)


@pytest.mark.parametrize("families", [(), ("unet", "unet"), ("unknown",)])
def test_invalid_families_refused(families):
    with pytest.raises(ValueError):
        suite.plan_commands(PROJECT, families=families)


def test_nested_config_paths_resolve_project_root():
    from asgcn_unet.utils import resolve_experiment_paths

    path = suite.config_path(PROJECT, "graph_unet", "train")
    resolved = resolve_experiment_paths(suite._load(path), path)
    assert Path(resolved["dataset"]["root"]) == PROJECT / "data/EventHDR/train"
    assert Path(resolved["output"]["run_dir"]) == PROJECT / "runs/ablations/graph_unet"


def test_modified_scale_refused_before_execution(monkeypatch):
    original = suite._load

    def load(path):
        value = copy.deepcopy(original(path))
        if path.name == "unet-train.json":
            value["train"]["batch_size"] = 1
        return value

    monkeypatch.setattr(suite, "_load", load)
    with pytest.raises(ValueError, match="contract differs"):
        suite.plan_commands(PROJECT)


def test_resume_completed_profile_is_verified_not_overwritten(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text('{"passed":true}')
    command = suite.Command(
        "unet",
        "profile",
        (
            sys.executable,
            "-B",
            "-m",
            "asgcn_unet.cli",
            "profile",
            "--config",
            "config.json",
            "--output",
            str(profile),
        ),
    )
    calls = []
    suite.execute_commands(
        [command],
        tmp_path,
        execute=True,
        resume=True,
        runner=lambda argv, **kw: calls.append(argv),
        emit=lambda _: None,
    )
    assert calls[0][4] == "verify-profile"
    assert "--report" in calls[0]
    assert profile.read_text() == '{"passed":true}'


def test_resume_incomplete_quality_refuses_without_archive(tmp_path):
    mode = tmp_path / "eval/ann"
    mode.mkdir(parents=True)
    marker = mode / "frames.csv"
    marker.write_text("original incomplete artifact")
    argv = (
        sys.executable,
        "-B",
        "-m",
        "asgcn_unet.cli",
        "evaluate",
        "--config",
        "config.json",
        "--checkpoint",
        "best.pt",
        "--output-dir",
        str(mode.parent),
        "--inference-mode",
        "ann",
    )
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="0 0\n")

    with pytest.raises(ValueError, match="Incomplete quality"):
        suite.execute_commands(
            [suite.Command("unet", "evaluate", argv)],
            tmp_path,
            execute=True,
            resume=True,
            runner=runner,
            emit=lambda _: None,
        )
    assert len(calls) == 1
    assert "--preserve-incomplete" not in calls[0]
    assert marker.read_text() == "original incomplete artifact"


def _synthetic_project(tmp_path):
    """Config/report fixtures only, not real training or metrics evidence."""
    for name in ("fast.json", "hdr-fast.json", "aid-fast.json"):
        destination = tmp_path / "configs" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text((PROJECT / "configs" / name).read_text())
    for family in suite.FAMILIES:
        for split in ("train", "hdr", "aid"):
            destination = suite.config_path(tmp_path, family, split)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(suite.config_path(PROJECT, family, split).read_text())
    return tmp_path


def _identity(value):
    import hashlib

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return {"contract": value, "sha256": hashlib.sha256(encoded).hexdigest()}


def _synthetic_report(project, family, *, runtime="runtime-1", training=None):
    """Explicit dummy metric values used only to test report parsing."""
    directory = project / "runs/ablations" / family / "eval/hdr/ann"
    directory.mkdir(parents=True)
    quality = {
        "quality": {
            "frames": 2,
            "micro": {"psnr": 12.0, "ssim": 0.5},
            "macro": {"psnr": 13.0, "ssim": 0.6},
        },
        "inference_mode": "ann",
        "simulation_steps": None,
        "snn_dynamics": None,
        "checkpoint_model_sha256": "synthetic-model",
        "report_eligible": False,
        "execution": {"model": {"total_parameters": 42, "trainable_parameters": 42}},
        "evaluation_protocol": {
            "model_config": _identity(
                suite._load(suite.config_path(project, family, "hdr"))["model"]
            ),
            "checkpoint": {
                "training_config": _identity(
                    training or suite._load(suite.config_path(project, family, "train"))
                )
            },
            "evaluation_dataset": {"sha256": "same-synthetic-data"},
            "runtime": _identity({"gpu_name": runtime, "device_type": "cpu", "test_only": True}),
            "source": _identity({"source_tree_sha256": "same-synthetic-source"}),
            "precision": _identity({"effective": "fp32", "test_only": True}),
        },
    }
    (directory / "metrics.json").write_text(json.dumps(quality))
    (directory / "benchmark.json").write_text(
        json.dumps(
            {
                "mean_ms": 20.0,
                "fps": 50.0,
                "peak_gpu_memory_mb": 10.0,
                "inference_mode": "ann",
                "checkpoint_model_sha256": "synthetic-model",
                "io_excluded": True,
                "report_eligible": False,
                "benchmark_protocol": {
                    "runtime": quality["evaluation_protocol"]["runtime"],
                    "source": quality["evaluation_protocol"]["source"],
                    "precision": quality["evaluation_protocol"]["precision"],
                    "evaluation_dataset": _identity(
                        {"selected": ["synthetic-frame"], "test_only": True}
                    ),
                    "model_config": quality["evaluation_protocol"]["model_config"],
                },
            }
        )
    )
    return directory


def test_summary_checks_stored_settings_and_reports_actual_parameter_counts(tmp_path):
    project = _synthetic_project(tmp_path)
    _synthetic_report(project, "unet")
    _synthetic_report(project, "graph_unet")
    rows = [r for r in suite.collect_summary(project) if r["frames"] is not None]
    assert len(rows) == 2
    assert all(r["parameters"] == 42 for r in rows)
    assert all(r["training_settings_match"] is True for r in rows)
    assert all(
        r["quality_dataset_claimed_matched"]
        and r["quality_runtime_matched"]
        and r["quality_source_matched"]
        for r in rows
    )
    assert all(
        r["benchmark_contracts_verified"]
        and r["benchmark_runtime_matched"]
        and r["benchmark_evaluation_dataset_matched"]
        for r in rows
    )
    assert all(r["quality_dataset_hash_verified"] is False for r in rows)
    assert all(r["quality_eligible"] is False for r in rows)


def test_summary_does_not_hide_different_training_or_hardware(tmp_path):
    project = _synthetic_project(tmp_path)
    _synthetic_report(project, "unet")
    training = suite._load(suite.config_path(project, "graph_unet", "train"))
    training["train"]["epochs"] = 1
    _synthetic_report(project, "graph_unet", runtime="different-synthetic-gpu", training=training)
    rows = [r for r in suite.collect_summary(project) if r["frames"] is not None]
    assert all(r["quality_runtime_matched"] is False for r in rows)
    graph = next(r for r in rows if r["family"] == "graph_unet")
    assert graph["training_settings_match"] is False
    assert "training settings mismatch" in graph["status"]


def test_summary_rejects_tampered_model_contract_even_if_folder_label_matches(tmp_path):
    project = _synthetic_project(tmp_path)
    directory = _synthetic_report(project, "unet")
    report = json.loads((directory / "metrics.json").read_text())
    report["evaluation_protocol"]["model_config"]["sha256"] = "tampered"
    (directory / "metrics.json").write_text(json.dumps(report))
    row = next(r for r in suite.collect_summary(project) if r["frames"] is not None)
    assert "model contract mismatch" in row["status"]


def test_contract_hash_preserves_unicode():
    assert suite._checked_contract(_identity({"name": "한글"})) == {"name": "한글"}


def test_empty_resume_inspector_output_is_explicit_error(tmp_path):
    (tmp_path / "eval/ann").mkdir(parents=True)
    command = suite.Command(
        "unet",
        "evaluate",
        (
            sys.executable,
            "-B",
            "-m",
            "asgcn_unet.cli",
            "evaluate",
            "--config",
            "config.json",
            "--checkpoint",
            "best.pt",
            "--output-dir",
            str(tmp_path / "eval"),
            "--inference-mode",
            "ann",
        ),
    )
    with pytest.raises(ValueError, match="Invalid evaluation resume inspector"):
        suite.execute_commands(
            [command],
            tmp_path,
            execute=True,
            resume=True,
            runner=lambda *a, **kw: SimpleNamespace(stdout=""),
            emit=lambda _: None,
        )


def test_all_family_profiles_fail_before_any_training(tmp_path):
    commands = suite.plan_commands(PROJECT)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if "profile" in argv and "transformer-train.json" in " ".join(argv):
            raise subprocess.CalledProcessError(1, argv)

    with pytest.raises(subprocess.CalledProcessError):
        suite.execute_commands(commands, PROJECT, execute=True, runner=runner, emit=lambda _: None)
    assert not any("train" in argv for argv in calls)
    assert sum("profile" in argv for argv in calls) == 4


def test_benchmark_different_gpu_does_not_borrow_quality_runtime_match(tmp_path):
    project = _synthetic_project(tmp_path)
    _synthetic_report(project, "unet")
    directory = _synthetic_report(project, "graph_unet")
    path = directory / "benchmark.json"
    benchmark = json.loads(path.read_text())
    benchmark["benchmark_protocol"]["runtime"] = _identity(
        {"gpu_name": "different-GPU", "test_only": True}
    )
    path.write_text(json.dumps(benchmark))
    rows = [row for row in suite.collect_summary(project) if row["frames"] is not None]
    assert all(row["quality_runtime_matched"] is True for row in rows)
    assert all(row["benchmark_runtime_matched"] is False for row in rows)
    graph = next(row for row in rows if row["family"] == "graph_unet")
    assert "benchmark runtime mismatch" in graph["status"]


@pytest.mark.parametrize(
    "field", ["runtime", "source", "precision", "evaluation_dataset", "model_config"]
)
def test_tampered_benchmark_contracts_are_not_marked_verified(tmp_path, field):
    project = _synthetic_project(tmp_path)
    _synthetic_report(project, "unet")
    directory = _synthetic_report(project, "graph_unet")
    path = directory / "benchmark.json"
    benchmark = json.loads(path.read_text())
    benchmark["benchmark_protocol"][field]["contract"]["tampered"] = True
    path.write_text(json.dumps(benchmark))
    rows = [row for row in suite.collect_summary(project) if row["frames"] is not None]
    graph = next(row for row in rows if row["family"] == "graph_unet")
    assert graph["benchmark_contracts_verified"] is False
    assert "benchmark" in graph["status"]
    if field != "model_config":
        assert all(row[f"benchmark_{field}_matched"] is False for row in rows)


def test_transformer_is_identity_encoder_ann_only_with_separate_run():
    assert suite.FAMILIES["transformer"] == ("identity", "transformer", "E")
    assert suite.modes("transformer") == suite.modes("unet") == (("ann", None, None),)
    for split in ("train", "hdr", "aid"):
        config = suite._load(suite.config_path(PROJECT, "transformer", split))
        assert config["model"]["encoder_kind"] == "identity"
        assert config["model"]["decoder_kind"] == "transformer"
        assert config["model"]["spline_backend"] == "torch"
        assert config["model"]["transformer_config"] == suite.TRANSFORMER_CONFIG
        assert (
            config["dataset"] == suite._load(suite.config_path(PROJECT, "unet", split))["dataset"]
        )
    commands = suite.plan_commands(PROJECT, families=("transformer",))
    assert not any(command.stage == "calibrate" for command in commands)
    assert not any("snn" in command.argv for command in commands)
    assert all("graph_transformer" not in " ".join(command.argv) for command in commands)


def test_graph_transformer_configs_preserved_but_excluded_from_active_suite(tmp_path):
    assert "graph_transformer" not in suite.FAMILIES
    for split in ("train", "hdr", "aid"):
        legacy = suite._load(suite.config_path(PROJECT, "graph_transformer", split))
        assert legacy["model"]["encoder_kind"] == "graph"
        assert legacy["model"]["decoder_kind"] == "transformer"
    with pytest.raises(ValueError, match="preserved legacy"):
        suite.plan_commands(PROJECT, families=("graph_transformer",))
    project = _synthetic_project(tmp_path)
    old = project / "runs/ablations/graph_transformer/eval/hdr/ann/metrics.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"legacy_original": true}')
    rows = suite.collect_summary(project)
    assert all(row["family"] != "graph_transformer" for row in rows)
    assert old.read_text() == '{"legacy_original": true}'


def test_transformer_ann_is_primary_E_and_explicit_comparison_validation_fields(tmp_path):
    project = _synthetic_project(tmp_path)
    _synthetic_report(project, "unet")
    _synthetic_report(project, "transformer")
    rows = [row for row in suite.collect_summary(project) if row["frames"] is not None]
    assert {row["group"] for row in rows} == {"A", "E"}
    for row in rows:
        assert row["quality_model_contract_valid"] is True
        assert row["quality_mode_valid"] is True
        assert row["benchmark_mode_valid"] is True
        assert row["benchmark_checkpoint_match"] is True
        assert row["benchmark_model_contract_valid"] is True


def test_comparison_validation_fields_expose_invalid_mode_checkpoint_and_model(tmp_path):
    project = _synthetic_project(tmp_path)
    directory = _synthetic_report(project, "transformer")
    quality_path = directory / "metrics.json"
    quality = json.loads(quality_path.read_text())
    quality["simulation_steps"] = 4
    quality["evaluation_protocol"]["model_config"]["sha256"] = "tampered"
    quality_path.write_text(json.dumps(quality))
    benchmark_path = directory / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text())
    benchmark["inference_mode"] = "snn"
    benchmark["checkpoint_model_sha256"] = "other-model"
    benchmark["benchmark_protocol"]["model_config"]["sha256"] = "tampered"
    benchmark_path.write_text(json.dumps(benchmark))
    row = next(row for row in suite.collect_summary(project) if row["frames"] is not None)
    assert row["quality_model_contract_valid"] is False
    assert row["quality_mode_valid"] is False
    assert row["benchmark_mode_valid"] is False
    assert row["benchmark_checkpoint_match"] is False
    assert row["benchmark_model_contract_valid"] is False
