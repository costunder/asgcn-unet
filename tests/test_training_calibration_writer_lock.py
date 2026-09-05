"""CPU control-flow tests for output ownership, not training/calibration results."""

import pytest

from asgcn_unet import engine
from asgcn_unet.artifact_lock import ArtifactWriterBusyError, exclusive_artifact_writer


def test_direct_training_entry_rejects_existing_writer_before_training(tmp_path, monkeypatch):
    run = tmp_path / "run"
    config = {"output": {"run_dir": str(run)}}
    monkeypatch.setattr(engine, "validate_experiment_config", lambda config: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("A locked run must not enter training")

    monkeypatch.setattr(engine, "_train", forbidden)
    with exclusive_artifact_writer(run) as lock:
        original = lock.read_bytes()
        with pytest.raises(ArtifactWriterBusyError):
            engine.train(config)
        assert lock.read_bytes() == original
    assert not run.exists()


def test_training_failure_releases_only_its_own_writer(tmp_path, monkeypatch):
    run = tmp_path / "run"
    monkeypatch.setattr(engine, "validate_experiment_config", lambda config: None)

    def fail(*args, **kwargs):
        raise RuntimeError("explicit unit-test failure")

    monkeypatch.setattr(engine, "_train", fail)
    with pytest.raises(RuntimeError, match="explicit unit-test failure"):
        engine.train({"output": {"run_dir": str(run)}})
    with exclusive_artifact_writer(run):
        pass


@pytest.mark.parametrize("overwrite", [False, True])
def test_calibration_writer_is_exclusive_even_with_overwrite(tmp_path, monkeypatch, overwrite):
    output = tmp_path / "snn.pt"
    original = b"preserved unit-test artifact; not a checkpoint"
    output.write_bytes(original)

    def forbidden(*args, **kwargs):
        raise AssertionError("A locked calibration output must not enter calibration")

    monkeypatch.setattr(engine, "validate_experiment_config", forbidden)
    with exclusive_artifact_writer(output) as lock:
        lock_bytes = lock.read_bytes()
        with pytest.raises(ArtifactWriterBusyError):
            engine.calibrate({}, tmp_path / "ann.pt", output, overwrite=overwrite)
        assert lock.read_bytes() == lock_bytes
    assert output.read_bytes() == original


def test_invalid_calibration_releases_its_writer_without_creating_checkpoint(tmp_path):
    output = tmp_path / "same.pt"
    with pytest.raises(ValueError, match="must be different files"):
        engine.calibrate({}, output, output)
    assert not output.exists()
    with exclusive_artifact_writer(output):
        pass
