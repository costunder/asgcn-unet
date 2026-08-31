"""Rejected train requests must not republish metadata over an existing run."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from asgcn_unet import cli, engine
from asgcn_unet.utils import save_json
from tests.test_training_batch import _config


@pytest.fixture(autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _gate(label: str) -> dict:
    return {
        "schema": "asgcn_preflight_verification_v1",
        "status": "bypassed_non_reporting",
        "report_eligible": False,
        "report": "pytest-only.json",
        "warning": label,
    }


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*") if path.is_file()
    }


def _cli_args(config_path: Path, *extra):
    return cli.build_parser().parse_args([
        "train", "--config", str(config_path), *extra,
    ])


@pytest.mark.parametrize("occupied", ["config.json", "history.json", "last.pt", "best.pt"])
def test_cli_rejected_fresh_run_preserves_existing_gate_and_all_artifact_bytes(
    tmp_path, monkeypatch, occupied
):
    config = _config(tmp_path, batch_size=1)
    run_dir = Path(config["output"]["run_dir"])
    run_dir.mkdir()
    (run_dir / "preflight_gate.json").write_bytes(b'{"original":"gate"}\r\n')
    (run_dir / occupied).write_bytes(b"original protected artifact\n")
    config_path = tmp_path / "request.json"
    save_json(config_path, config)
    before = _snapshot(run_dir)
    monkeypatch.setattr(cli, "verify_training_preflight", lambda *_args: _gate("replacement"))
    with pytest.raises(ValueError, match="Fresh training run_dir is not empty"):
        cli._execute_command(_cli_args(config_path))
    assert _snapshot(run_dir) == before


def test_cli_does_not_publish_gate_when_engine_rejects_dataset_preparation(tmp_path, monkeypatch):
    config = _config(tmp_path, batch_size=1)
    run_dir = Path(config["output"]["run_dir"])
    run_dir.mkdir()
    (run_dir / "preflight_gate.json").write_bytes(b'{"original":"standalone gate"}\n')
    before = _snapshot(run_dir)
    config_path = tmp_path / "request.json"
    save_json(config_path, config)
    monkeypatch.setattr(cli, "verify_training_preflight", lambda *_args: _gate("replacement"))

    def invalid_data(*_args, **_kwargs):
        raise ValueError("dataset validation rejected")

    monkeypatch.setattr(engine, "build_dataset", invalid_data)
    with pytest.raises(ValueError, match="dataset validation rejected"):
        cli._execute_command(_cli_args(config_path))
    assert _snapshot(run_dir) == before


def test_validated_cli_fresh_training_publishes_gate_with_config_before_first_update(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, batch_size=1)
    config_path = tmp_path / "request.json"
    save_json(config_path, config)
    run_dir = Path(config["output"]["run_dir"])
    gate = _gate("pytest CPU publication only")
    monkeypatch.setattr(cli, "verify_training_preflight", lambda *_args: gate)
    original_step = engine._training_step
    calls = 0

    def observed_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert json.loads((run_dir / "preflight_gate.json").read_text()) == gate
        published = json.loads((run_dir / "config.json").read_text())
        assert published["preflight_gate"] == gate
        assert (run_dir / ".data_hash_cache.json").is_file()
        return original_step(*args, **kwargs)

    monkeypatch.setattr(engine, "_training_step", observed_step)
    cli._execute_command(_cli_args(config_path))
    assert calls > 0
    checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["preflight_gate"] == gate


@pytest.mark.parametrize("failure", [
    "gate", "optimizer", "scheduler", "scaler", "rng_state", "rng_schema",
])
def test_rejected_resume_preserves_gate_config_cache_and_checkpoints(
    tmp_path, monkeypatch, failure
):
    config = _config(tmp_path, batch_size=1)
    config["preflight_gate"] = _gate("original")
    engine.train(config)
    run_dir = Path(config["output"]["run_dir"])
    checkpoint_path = run_dir / "last.pt"
    before = _snapshot(run_dir)
    resume_config = copy.deepcopy(config)
    resume_config["train"]["epochs"] = 2
    if failure == "gate":
        resume_config["preflight_gate"] = _gate("different verified request")
        message = "preflight gate"
    else:
        original_load = engine.load_model_checkpoint

        def altered_load(path, *args, **kwargs):
            model, checkpoint = original_load(path, *args, **kwargs)
            if Path(path) == checkpoint_path:
                if failure == "rng_schema":
                    checkpoint["rng_state"] = {}
                else:
                    checkpoint.pop(failure)
            return model, checkpoint

        monkeypatch.setattr(engine, "load_model_checkpoint", altered_load)
        message = {
            "optimizer": "no optimizer state",
            "scheduler": "no scheduler state",
            "scaler": "no GradScaler state",
            "rng_state": "no RNG state",
            "rng_schema": "rng_state is missing",
        }[failure]
    with pytest.raises(ValueError, match=message):
        engine.train(resume_config, resume_from=checkpoint_path)
    assert _snapshot(run_dir) == before
