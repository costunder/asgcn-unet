from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.eval_resume as resume
from asgcn_unet.engine import (
    _canonical_sha256,
    _current_source_contract,
    _hashed_contract,
    _public_config,
)
from asgcn_unet.utils import load_json, resolve_experiment_paths

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/aid-fast.json"
DATASET_COMMON = {
    "content": {"algorithm": "sha256-full-files-v1", "files": 14, "bytes": 1, "sha256": "a" * 64},
    "transform": {"type": "eventaid_r_zip", "fixture": True},
    "manifest": {"fixture": True},
    "coverage": {"samples": 1, "fixture": True},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _args(output: Path, checkpoint: Path):
    return resume._parser().parse_args(
        [
            "--config", str(CONFIG_PATH),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(output),
            "--inference-mode", "snn",
            "--simulation-steps", "4",
            "--snn-dynamics", "literal_eq15",
            "--max-graph-edges", "7475202",
            "--benchmark-warmup", "10",
            "--benchmark-steps", "100",
        ]
    )


def _protocol(output: Path, checkpoint: Path, kind: str) -> dict:
    config = resolve_experiment_paths(load_json(CONFIG_PATH), CONFIG_PATH)
    config["eval"]["output_dir"] = str(output.resolve())
    execution = {
        "inference_mode": "snn",
        "simulation_steps": 4,
        "snn_dynamics": "literal_eq15",
        "graph_edge_guard": {
            "configured_max_graph_edges": 2_000_000,
            "requested_max_graph_edges_override": 7_475_202,
            "effective_max_graph_edges": 7_475_202,
        },
    }
    if kind == "compute_benchmark":
        execution.update(warmup_steps=10, measured_steps=100)
    protocol = {
        "schema": "asgcn_reporting_protocol_v1",
        "kind": kind,
        "report_eligible": True,
        "report_ineligible_reasons": [],
        "public_config": _hashed_contract(_public_config(config)),
        "model_config": _hashed_contract(config["model"]),
        "checkpoint": {"checkpoint_file_sha256": _sha256(checkpoint)},
        "evaluation_dataset": _hashed_contract({**DATASET_COMMON, "sampling": {"fixture": True}}),
        "execution": _hashed_contract(execution),
        "source": _hashed_contract(_current_source_contract()),
        "runtime": _hashed_contract({"fixture": True}),
        "precision": _hashed_contract({"fixture": True}),
    }
    protocol["protocol_sha256"] = _canonical_sha256(protocol)
    return protocol


def _write_complete_mode(output: Path, checkpoint: Path) -> Path:
    run_dir = output / "snn_literal_eq15_T4"
    run_dir.mkdir(parents=True)
    metrics = {
        "report_eligible": True,
        "report_ineligible_reasons": [],
        "evaluation_protocol": _protocol(output, checkpoint, "quality_evaluation"),
        "quality": {"frames": 1},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "frames.csv").write_text("sample_id\nfixture\n", encoding="utf-8")
    benchmark = {
        "report_eligible": True,
        "report_ineligible_reasons": [],
        "benchmark_protocol": _protocol(output, checkpoint, "compute_benchmark"),
        "frames": 100,
    }
    (run_dir / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")
    return run_dir


def _stub_request_validation(monkeypatch: pytest.MonkeyPatch, checkpoint: Path) -> None:
    monkeypatch.setattr(
        resume,
        "load_model_checkpoint",
        lambda *args, **kwargs: (SimpleNamespace(max_graph_edges=2_000_000), {}),
    )
    monkeypatch.setattr(
        resume,
        "_reporting_checkpoint_contract",
        lambda *args, **kwargs: ({"checkpoint_file_sha256": _sha256(checkpoint)}, True, []),
    )
    monkeypatch.setattr(resume, "_current_dataset_common", lambda config: DATASET_COMMON)
    monkeypatch.setattr(
        resume,
        "_current_runtime_contract",
        lambda config, model: {"fixture": True},
    )


def test_resume_validator_accepts_matching_complete_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "best_snn.pt"
    checkpoint.write_bytes(b"sealed-checkpoint-fixture")
    output = tmp_path / "aid"
    _write_complete_mode(output, checkpoint)
    _stub_request_validation(monkeypatch, checkpoint)

    assert resume.inspect_mode(_args(output, checkpoint)) == (1, 1)


def test_resume_validator_reports_each_completed_half_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "best_snn.pt"
    checkpoint.write_bytes(b"sealed-checkpoint-fixture")
    _stub_request_validation(monkeypatch, checkpoint)

    quality_output = tmp_path / "quality/aid"
    quality_dir = _write_complete_mode(quality_output, checkpoint)
    (quality_dir / "benchmark.json").unlink()
    assert resume.inspect_mode(_args(quality_output, checkpoint)) == (1, 0)

    benchmark_output = tmp_path / "benchmark/aid"
    benchmark_dir = _write_complete_mode(benchmark_output, checkpoint)
    (benchmark_dir / "metrics.json").unlink()
    (benchmark_dir / "frames.csv").unlink()
    assert resume.inspect_mode(_args(benchmark_output, checkpoint)) == (0, 1)


def test_resume_validator_reports_partial_mode_without_reusing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "best_snn.pt"
    checkpoint.write_bytes(b"sealed-checkpoint-fixture")
    output = tmp_path / "aid"
    (output / "snn_literal_eq15_T4/predictions").mkdir(parents=True)
    _stub_request_validation(monkeypatch, checkpoint)

    assert resume.inspect_mode(_args(output, checkpoint)) == (0, 0)


def test_resume_validator_treats_symlinked_mode_directory_as_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "best_snn.pt"
    checkpoint.write_bytes(b"sealed-checkpoint-fixture")
    output = tmp_path / "aid"
    target = tmp_path / "external-mode"
    target.mkdir()
    output.mkdir()
    try:
        (output / "snn_literal_eq15_T4").symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    _stub_request_validation(monkeypatch, checkpoint)

    assert resume.inspect_mode(_args(output, checkpoint)) == (0, 0)


def test_resume_validator_refuses_a_different_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "best_snn.pt"
    checkpoint.write_bytes(b"sealed-checkpoint-fixture")
    output = tmp_path / "aid"
    _write_complete_mode(output, checkpoint)
    _stub_request_validation(monkeypatch, checkpoint)
    monkeypatch.setattr(
        resume,
        "_current_runtime_contract",
        lambda config, model: {"fixture": False},
    )

    with pytest.raises(resume.ArtifactMismatch, match="runtime mismatch"):
        resume.inspect_mode(_args(output, checkpoint))


def test_resume_validator_refuses_mismatched_complete_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "best_snn.pt"
    checkpoint.write_bytes(b"sealed-checkpoint-fixture")
    output = tmp_path / "aid"
    run_dir = _write_complete_mode(output, checkpoint)
    _stub_request_validation(monkeypatch, checkpoint)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    protocol = metrics["evaluation_protocol"]
    protocol["execution"]["contract"]["simulation_steps"] = 8
    protocol["execution"]["sha256"] = _canonical_sha256(protocol["execution"]["contract"])
    unsigned = dict(protocol)
    unsigned.pop("protocol_sha256")
    protocol["protocol_sha256"] = _canonical_sha256(unsigned)
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(resume.ArtifactMismatch, match="simulation_steps mismatch"):
        resume.inspect_mode(_args(output, checkpoint))


def test_resume_validator_refuses_changed_data_or_nonreporting_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "best_snn.pt"
    checkpoint.write_bytes(b"sealed-checkpoint-fixture")
    output = tmp_path / "aid"
    run_dir = _write_complete_mode(output, checkpoint)
    _stub_request_validation(monkeypatch, checkpoint)
    monkeypatch.setattr(
        resume,
        "_current_dataset_common",
        lambda config: {**DATASET_COMMON, "content": {"sha256": "b" * 64}},
    )
    with pytest.raises(resume.ArtifactMismatch, match="evaluation_dataset.content"):
        resume.inspect_mode(_args(output, checkpoint))

    monkeypatch.setattr(resume, "_current_dataset_common", lambda config: DATASET_COMMON)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["report_eligible"] = False
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(resume.ArtifactMismatch, match="report_eligible mismatch"):
        resume.inspect_mode(_args(output, checkpoint))
