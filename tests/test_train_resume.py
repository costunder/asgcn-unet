"""Committed minibatch resume on temporary real-format HDF5 fixtures."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from asgcn_unet import engine
from asgcn_unet.batching import SequenceBatchSampler
from asgcn_unet.data import EventHDRDataset, build_dataset
from tests.test_training_batch import _checkpoint, _config


@pytest.fixture(autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _pause_after_updates(monkeypatch, count: int) -> dict[str, int]:
    """Request a stop only after a successful real optimizer update returns."""
    completed = {"updates": 0}
    original = engine._training_step

    def training_step(*args, **kwargs):
        result = original(*args, **kwargs)
        completed["updates"] += 1
        return result

    class StopAfterUpdates:
        def __init__(self, time_limit_seconds=None):
            self.time_limit_seconds = time_limit_seconds

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def poll(self):
            return completed["updates"] >= count

        @property
        def pause_requested(self):
            return self.poll()

        @property
        def reason(self):
            return "test_pause" if self.poll() else None

    monkeypatch.setattr(engine, "StopRequest", StopAfterUpdates)
    monkeypatch.setattr(engine, "_training_step", training_step)
    return completed


def _assert_training_equal(actual: dict, expected: dict) -> None:
    for key in ("model", "optimizer", "scheduler", "scaler", "val"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual["best_model_state_sha256"] == expected["best_model_state_sha256"]
    assert actual["rng_state"]["python"] == expected["rng_state"]["python"]
    actual_np = actual["rng_state"]["numpy"]
    expected_np = expected["rng_state"]["numpy"]
    assert actual_np[0] == expected_np[0]
    np.testing.assert_array_equal(actual_np[1], expected_np[1])
    assert actual_np[2:] == expected_np[2:]
    torch.testing.assert_close(
        actual["rng_state"]["torch"], expected["rng_state"]["torch"], rtol=0, atol=0
    )
    assert len(actual["history"]) == len(expected["history"])
    for actual_epoch, expected_epoch in zip(actual["history"], expected["history"], strict=True):
        for key in ("epoch", "train_loss", "val", "val_sampling", "learning_rate", "amp"):
            assert actual_epoch[key] == expected_epoch[key]
        for key in ("frames", "optimizer_steps", "batch_size_limit"):
            assert actual_epoch["performance"][key] == expected_epoch["performance"][key]


@pytest.mark.parametrize("batch_size,stop_after", [(1, 2), (2, 1), (2, 2), (2, 5)])
def test_mid_epoch_resume_matches_uninterrupted_training(
    tmp_path, monkeypatch, batch_size, stop_after
) -> None:
    config = _config(tmp_path, batch_size=batch_size)
    config["train"]["epochs"] = 2
    config["output"]["run_dir"] = str(tmp_path / "continuous")
    engine.train(config)
    expected = _checkpoint(tmp_path / "continuous")

    config["output"]["run_dir"] = str(tmp_path / "resumed")
    with monkeypatch.context() as paused_patch:
        completed = _pause_after_updates(paused_patch, stop_after)
        with pytest.raises(engine.TrainingPaused) as stopped:
            engine.train(config)
    assert completed["updates"] == stop_after
    assert Path(stopped.value.checkpoint_path) == tmp_path / "resumed" / "last.pt"
    assert stopped.value.reason == "test_pause"
    partial = _checkpoint(tmp_path / "resumed")
    progress = partial["epoch_progress"]
    assert progress["version"] == 1
    assert progress["epoch"] == partial["epoch"] + 1
    assert progress["seen"] > 0
    assert progress["next_batch"] > 0
    assert progress["training_state"]
    assert len(partial["history"]) == partial["epoch"]

    engine.train(config, resume_from=tmp_path / "resumed" / "last.pt")
    _assert_training_equal(_checkpoint(tmp_path / "resumed"), expected)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_resume_does_not_decode_or_train_committed_prefix(
    tmp_path, monkeypatch, batch_size
) -> None:
    config = _config(tmp_path, batch_size=batch_size)
    dataset = build_dataset(config["dataset"], split="train")
    try:
        batches = (
            list(SequenceBatchSampler(dataset, batch_size, seed=config["seed"]))
            if batch_size > 1
            else [[index] for index in range(len(dataset))]
        )
    finally:
        dataset.close()
    with monkeypatch.context() as paused_patch:
        _pause_after_updates(paused_patch, 1)
        with pytest.raises(engine.TrainingPaused):
            engine.train(config)
    checkpoint_path = Path(config["output"]["run_dir"]) / "last.pt"
    before = _checkpoint(checkpoint_path.parent)
    assert before["epoch_progress"]["next_batch"] == 1
    assert before["epoch_progress"]["seen"] == len(batches[0])

    decoded = []
    original = EventHDRDataset.__getitem__

    def observed_getitem(dataset, index):
        if dataset.root == Path(config["dataset"]["root"]):
            decoded.append(index)
        return original(dataset, index)

    monkeypatch.setattr(EventHDRDataset, "__getitem__", observed_getitem)
    engine.train(config, resume_from=checkpoint_path)
    assert decoded == [index for batch in batches[1:] for index in batch]
    result = _checkpoint(checkpoint_path.parent)
    assert result["history"][0]["performance"]["frames"] == 7
    assert result["history"][0]["performance"]["optimizer_steps"] == len(batches)


def test_partial_resume_with_persistent_prefetch_workers_matches_rng_and_weights(
    tmp_path, monkeypatch
) -> None:
    config = _config(tmp_path)
    config["train"].update({
        "epochs": 2, "num_workers": 2, "persistent_workers": True, "prefetch_factor": 2,
    })
    config["output"]["run_dir"] = str(tmp_path / "continuous")
    engine.train(config)
    expected = _checkpoint(tmp_path / "continuous")
    config["output"]["run_dir"] = str(tmp_path / "resumed")
    with monkeypatch.context() as paused_patch:
        _pause_after_updates(paused_patch, 2)
        with pytest.raises(engine.TrainingPaused):
            engine.train(config)
    engine.train(config, resume_from=tmp_path / "resumed" / "last.pt")
    _assert_training_equal(_checkpoint(tmp_path / "resumed"), expected)


def test_repeated_partial_pauses_preserve_cumulative_cursor_and_epoch_metrics(
    tmp_path, monkeypatch
) -> None:
    config = _config(tmp_path)
    config["output"]["run_dir"] = str(tmp_path / "continuous")
    engine.train(config)
    expected = _checkpoint(tmp_path / "continuous")
    config["output"]["run_dir"] = str(tmp_path / "resumed")
    resume_path = tmp_path / "resumed" / "last.pt"
    for expected_cursor in (1, 2, 3):
        with monkeypatch.context() as paused_patch:
            _pause_after_updates(paused_patch, 1)
            with pytest.raises(engine.TrainingPaused):
                engine.train(config, resume_from=resume_path if expected_cursor > 1 else None)
        partial = _checkpoint(resume_path.parent)
        assert partial["epoch_progress"]["next_batch"] == expected_cursor
        assert partial["epoch"] == 0
        assert not partial["history"]
    engine.train(config, resume_from=resume_path)
    _assert_training_equal(_checkpoint(resume_path.parent), expected)


def test_periodic_checkpoint_recovers_after_failure_before_next_update(
    tmp_path, monkeypatch
) -> None:
    config = _config(tmp_path)
    config["output"]["run_dir"] = str(tmp_path / "continuous")
    engine.train(config)
    expected = _checkpoint(tmp_path / "continuous")
    config["output"]["run_dir"] = str(tmp_path / "resumed")
    completed = 0
    original = engine._training_step

    def fail_after_saved_batch(*args, **kwargs):
        nonlocal completed
        if completed == 1:
            raise RuntimeError("injected failure before next update")
        result = original(*args, **kwargs)
        completed += 1
        return result

    with monkeypatch.context() as failing_patch:
        failing_patch.setattr(engine, "_training_step", fail_after_saved_batch)
        with pytest.raises(RuntimeError, match="injected failure"):
            engine.train(config, checkpoint_seconds=1e-9)
    partial = _checkpoint(tmp_path / "resumed")
    assert partial["epoch"] == 0
    assert partial["epoch_progress"]["next_batch"] == 1
    assert partial["epoch_progress"]["seen"] == 2
    engine.train(config, resume_from=tmp_path / "resumed" / "last.pt")
    _assert_training_equal(_checkpoint(tmp_path / "resumed"), expected)


@pytest.mark.parametrize("corruption", [
    "cursor", "bool_cursor", "bool_version", "seen", "epoch", "state",
    "empty_context", "wrong_context_index", "wrong_context_key",
])
def test_corrupt_partial_progress_is_rejected_without_rewriting_run(
    tmp_path, monkeypatch, corruption
) -> None:
    config = _config(tmp_path)
    with monkeypatch.context() as paused_patch:
        _pause_after_updates(paused_patch, 1)
        with pytest.raises(engine.TrainingPaused):
            engine.train(config)
    run_dir = Path(config["output"]["run_dir"])
    checkpoint = _checkpoint(run_dir)
    progress = copy.deepcopy(checkpoint["epoch_progress"])
    if corruption == "cursor":
        progress["next_batch"] = 99999
    elif corruption == "bool_cursor":
        progress["next_batch"] = True
    elif corruption == "bool_version":
        progress["version"] = True
    elif corruption == "seen":
        progress["seen"] += 1
    elif corruption == "epoch":
        progress["epoch"] += 1
    elif corruption == "state":
        progress.pop("training_state")
    elif corruption == "empty_context":
        progress["training_state"]["entries"] = []
    elif corruption == "wrong_context_index":
        progress["training_state"]["entries"][0]["sequence_index"] += 1
    else:
        entry = progress["training_state"]["entries"][0]
        entry["key"] = ("unseen-sequence", entry["key"][1])
    checkpoint["epoch_progress"] = progress
    torch.save(checkpoint, run_dir / "last.pt")
    before = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    with pytest.raises((TypeError, ValueError), match="(?i)progress|resume|state|cursor|batch|epoch|seen"):
        engine.train(config, resume_from=run_dir / "last.pt")
    after = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    assert after == before


def test_stop_after_last_training_batch_resumes_validation_without_retraining(
    tmp_path, monkeypatch
) -> None:
    config = _config(tmp_path)
    config["train"]["validate_every"] = None
    config["output"]["run_dir"] = str(tmp_path / "continuous")
    engine.train(config)
    expected = _checkpoint(tmp_path / "continuous")
    config["output"]["run_dir"] = str(tmp_path / "resumed")
    with monkeypatch.context() as paused_patch:
        _pause_after_updates(paused_patch, 4)
        with pytest.raises(engine.TrainingPaused):
            engine.train(config)
    before = _checkpoint(tmp_path / "resumed")
    assert before["epoch"] == 0
    assert before["epoch_progress"]["next_batch"] == 4
    assert before["epoch_progress"]["seen"] == 7
    assert not before["terminal_validation_state"]["completed"]
    assert not (tmp_path / "resumed" / "best.pt").exists()

    def already_trained(*args, **kwargs):
        raise AssertionError("Completed training frames must not be replayed for validation resume")

    monkeypatch.setattr(engine, "_training_step", already_trained)
    engine.train(config, resume_from=tmp_path / "resumed" / "last.pt")
    result = _checkpoint(tmp_path / "resumed")
    _assert_training_equal(result, expected)
    assert result["terminal_validation_state"]["completed"]
