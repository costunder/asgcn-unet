"""CPU-only synthetic tests of probe memory accounting, never CUDA measurements."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from asgcn_unet import engine, preflight, stream_preflight, training, utils
from asgcn_unet.batching import SequenceBatchSampler, pack_samples
from tests.test_stream_preflight import SyntheticStreams, _config


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _policy_fixture(monkeypatch, *, fail_reserve=None, fail_commit=None, allocator_peak=None):
    """Mock allocator telemetry only; all fixture tensors stay on CPU."""
    dataset, config, progress = SyntheticStreams(), _config(), {}
    batches = list(SequenceBatchSampler(dataset, 2))
    plan = {"selected_batch_indices": [0, 2], "replay_stop_batch": 2}
    topology = {"samples": [{"readout_nodes": 2, "readout_directed_edges": 0} for _ in dataset.samples]}
    events = []
    telemetry = {"peak": 10, "reserved": 10}

    def memory_info(device):
        events.append(("free", progress["current_batch_index"], progress["phase"]))
        failure = (progress["current_batch_index"], progress["phase"]) == fail_reserve
        return (50 if failure else 500) * 1024**2, 1000 * 1024**2

    def reset(device):
        events.append(("reset", progress["current_batch_index"]))
        telemetry.update(peak=10, reserved=10)

    def observe(value):
        telemetry["peak"] = max(telemetry["peak"], value)
        telemetry["reserved"] = max(telemetry["reserved"], value + 10)

    def output(samples):
        observe(70)
        return samples.targets.clone(), [
            {"nodes": 2, "edges": 0, "stream_execution": {"training_dense_snapshot": True}}
            for _ in samples
        ]

    model = SimpleNamespace(parameters=list, forward_training_batch=lambda samples, contexts: output(samples))
    model.to = lambda device: model
    model.train = lambda: model

    def load(dataset, indices, device):
        number = progress["current_batch_index"]
        events.append(("load", number))
        observe(120)
        return pack_samples([dataset[index] for index in indices]), {"synthetic_cpu_fixture": True}

    class State:
        def __init__(self, **kwargs):
            pass

        def prepare(self, samples):
            return [(None, None, None)] * len(samples)

        def commit(self, samples, prediction, diagnostics, target):
            number = progress["current_batch_index"]
            events.append(("commit", number))
            observe([180, 300, 170][number])
            if allocator_peak is not None and number == allocator_peak:
                telemetry["reserved"] = 950
            if number == fail_commit:
                raise RuntimeError("synthetic state commit allocation failure")

        def release_finished(self, samples, final):
            events.append(("release", progress["current_batch_index"]))

    def step(model, optimizer, scaler, forward_loss, **kwargs):
        samples = forward_loss.__defaults__[0]
        prediction, diagnostics = output(samples)
        return (prediction, diagnostics, samples.targets), 1.0, 0.25, {"synthetic_cpu_fixture": True}

    monkeypatch.setattr(engine, "build_model", lambda config: model)
    monkeypatch.setattr(engine, "_build_optimizer", lambda *args: object())
    monkeypatch.setattr(engine, "_make_grad_scaler", lambda *args: object())
    monkeypatch.setattr(engine, "_training_step", step)
    monkeypatch.setattr(training, "TrainingState", State)
    monkeypatch.setattr(preflight, "_load_packed_probe_batch", load)
    monkeypatch.setattr(utils, "set_seed", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(torch.cuda, "mem_get_info", memory_info)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: telemetry["peak"] * 1024**2)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: telemetry["reserved"] * 1024**2)
    args = dataset, config, torch.device("cuda"), batches, topology, plan
    return args, progress, events


def test_selected_peak_includes_input_and_commit_and_replay_peak_is_not_discarded(monkeypatch):
    args, progress, events = _policy_fixture(monkeypatch)
    report = stream_preflight._probe_stream_training(*args, reserve_vram_mib=100, progress=progress)
    assert report is progress and report["passed"]
    assert [step["batch_index"] for step in report["steps"]] == [0, 2]
    assert [step["peak_allocated_mib"] for step in report["steps"]] == [180, 170]
    assert report["peak_allocated_mib"] == 300 and report["peak_reserved_mib"] == 310
    assert report["replayed_predecessor_frames"] == 2
    for number in range(3):
        assert events.index(("reset", number)) < events.index(("load", number))
        assert events.index(("commit", number)) < events.index(("release", number))
        assert events.index(("release", number)) < events.index(("free", number, "after_state_commit_and_release"))
    assert all(step["scope"].endswith("state_commit_and_release") for step in report["steps"])
    assert report["reserve_is_hard_isolation"] is False


@pytest.mark.parametrize("phase", ["before_input_load", "after_state_commit_and_release"])
def test_live_free_reserve_failure_stops_before_next_batch_and_keeps_completed_steps(monkeypatch, phase):
    args, progress, events = _policy_fixture(monkeypatch, fail_reserve=(1, phase))
    with pytest.raises(RuntimeError, match="live device free"):
        stream_preflight._probe_stream_training(*args, reserve_vram_mib=100, progress=progress)
    assert not progress["passed"] and progress["failed_batch_index"] == 1
    assert progress["phase"] == phase and progress["failure_type"] == "RuntimeError"
    assert [step["batch_index"] for step in progress["steps"]] == [0]
    assert ("load", 2) not in events
    assert (("load", 1) in events) == (phase == "after_state_commit_and_release")
    assert progress["minimum_observed_device_free_mib"] == 50


def test_allocator_peak_budget_is_separate_from_live_free_memory(monkeypatch):
    args, progress, events = _policy_fixture(monkeypatch, allocator_peak=1)
    with pytest.raises(RuntimeError, match="does not account for other processes"):
        stream_preflight._probe_stream_training(*args, reserve_vram_mib=100, progress=progress)
    assert progress["last_memory_snapshot"]["device_free_mib"] == 500
    assert progress["last_memory_snapshot"]["peak_reserved_mib"] == 950
    assert progress["failed_batch_index"] == 1 and ("load", 2) not in events


def test_state_commit_failure_is_not_a_successful_probe_and_preserves_prior_steps(monkeypatch):
    args, progress, events = _policy_fixture(monkeypatch, fail_commit=2)
    with pytest.raises(RuntimeError, match="state commit allocation failure"):
        stream_preflight._probe_stream_training(*args, reserve_vram_mib=100, progress=progress)
    assert progress["failed_batch_index"] == 2 and progress["phase"] == "state_commit_and_release"
    assert progress["completed_batches"] == 2 and not progress["passed"]
    assert [step["batch_index"] for step in progress["steps"]] == [0]
    assert ("release", 2) not in events


@pytest.mark.parametrize("reserve", [-1, True, float("nan"), float("inf"), "100"])
def test_invalid_reserve_rejected_before_device_or_model(monkeypatch, reserve):
    monkeypatch.setattr(engine, "build_model", lambda config: pytest.fail("model must not be allocated"))
    with pytest.raises(ValueError, match="reserve_vram_mib"):
        stream_preflight._probe_stream_training(None, {}, torch.device("cpu"), [], {}, {}, reserve_vram_mib=reserve)


def test_real_cpu_training_probe_progress_and_stateful_smoke():
    dataset, config = SyntheticStreams(), _config()
    batches = list(SequenceBatchSampler(dataset, 2))
    topology = stream_preflight._scan_stream_topology(dataset, config, torch.device("cpu"), batches, top_density_count=2)
    plan = {"selected_batch_indices": [0, 2], "replay_stop_batch": 2}
    progress = {}
    report = stream_preflight._probe_stream_training(dataset, config, torch.device("cpu"), batches,
                                                     topology, plan, progress=progress)
    assert report is progress and report["passed"] and report["completed_batches"] == 3
    assert report["steps"][-1]["incoming_contexts"] == 2
    assert report["peak_allocated_mib"] is None and report["peak_reserved_mib"] is None
    assert all(row["loss"]["total"] >= 0 and row["step_time_ms"] > 0 for row in report["steps"])
