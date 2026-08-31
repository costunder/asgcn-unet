from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from asgcn_unet import timing


class FakeCuda:
    """Event/stream ordering oracle: these numbers are not GPU measurements."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.events: list[FakeEvent] = []
        self.current_index = 2
        self.synchronized = False
        self.streams = {index: SimpleNamespace(device=torch.device(f"cuda:{index}")) for index in (2, 3)}
        self.durations = [1.0, 3.0, 5.0, 7.0, 9.0]

    def current_stream(self, device):
        selected = torch.device(device)
        index = self.current_index if selected.index is None else selected.index
        self.calls.append(("stream", str(selected)))
        return self.streams[index]

    def Event(self, *, enable_timing):
        assert enable_timing is True
        event = FakeEvent(self, len(self.events))
        self.events.append(event)
        self.calls.append(("event", event.index))
        return event

    def synchronize(self, device):
        self.calls.append(("synchronize", str(device)))
        self.synchronized = True


class FakeEvent:
    def __init__(self, cuda: FakeCuda, index: int):
        self.cuda = cuda
        self.index = index
        self.stream = None

    def record(self, stream):
        self.stream = stream
        self.cuda.calls.append(("record", self.index, str(stream.device)))

    def elapsed_time(self, end):
        assert self.cuda.synchronized, "elapsed_time was read before the collection barrier"
        assert self.stream is end.stream
        assert end.index == self.index + 1
        self.cuda.calls.append(("elapsed", self.index, end.index))
        return self.cuda.durations[self.index // 2]


def _fake_cuda(monkeypatch):
    cuda = FakeCuda()
    monkeypatch.setattr(timing, "torch", SimpleNamespace(device=torch.device, cuda=cuda))
    return cuda


def _clock(monkeypatch, values):
    iterator = iter(values)
    monkeypatch.setattr(timing, "perf_counter", lambda: next(iterator))


@pytest.mark.parametrize("device", ["cpu", "cuda:3"])
def test_disabled_timer_never_queries_clocks_or_cuda(monkeypatch, device):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled timing must not inspect clocks or CUDA")

    monkeypatch.setattr(timing, "perf_counter", forbidden)
    monkeypatch.setattr(
        timing,
        "torch",
        SimpleNamespace(
            device=torch.device,
            cuda=SimpleNamespace(current_stream=forbidden, Event=forbidden, synchronize=forbidden),
        ),
    )
    recorder = timing.StageTimer(device)
    for _ in range(100):
        with recorder.scope("backward"):
            pass
        assert recorder.step() is False
    report = recorder.collect()
    assert report["enabled"] is False
    assert report["window_complete"] is False
    assert report["measured_steps"] == 0
    assert report["cuda_events_measured"] is False
    assert report["recorded_scopes"] == 0


def test_cpu_window_excludes_warmup_and_post_window_and_summarizes(monkeypatch):
    cuda = _fake_cuda(monkeypatch)
    _clock(monkeypatch, [0.0, 0.001, 1.0, 1.003, 2.0, 2.005])
    recorder = timing.StageTimer("cpu", enabled=True, warmup_steps=2, measurement_steps=3)
    completed = []
    for _ in range(8):
        with recorder.scope("backward"):
            pass
        completed.append(recorder.step())
    assert completed == [False, False, False, False, True, False, False, False]
    report = recorder.collect()
    summary = report["stages"]["backward"]["host_wall"]
    assert summary["count"] == 3
    assert summary["total_ms"] == pytest.approx(9.0)
    assert summary["mean_ms"] == pytest.approx(3.0)
    assert summary["p50_ms"] == pytest.approx(3.0)
    assert summary["p95_ms"] == pytest.approx(4.8)
    assert report["cpu_diagnostic_only"] is True
    assert report["cuda_events_measured"] is False
    assert report["cuda_event_device"] is None
    assert report["stages"]["backward"]["cuda_elapsed"]["count"] == 0
    assert report["window_complete"] is True
    assert report["measured_steps"] == 3
    assert cuda.calls == []


def test_cuda_events_record_selected_stream_without_per_stage_barriers(monkeypatch):
    cuda = _fake_cuda(monkeypatch)
    _clock(monkeypatch, [0.0, 0.010, 1.0, 1.020, 2.0, 2.030])
    recorder = timing.StageTimer("cuda:3", enabled=True, warmup_steps=0, measurement_steps=1)
    with recorder.scope("dataload", gpu=False):
        pass
    with recorder.scope("encoder"):
        pass
    with recorder.scope("decoder"):
        pass
    assert recorder.step() is True
    assert all(call[0] not in {"synchronize", "elapsed"} for call in cuda.calls)
    assert [event.stream.device for event in cuda.events] == [torch.device("cuda:3")] * 4
    report = recorder.collect()
    assert [call for call in cuda.calls if call[0] == "synchronize"] == [("synchronize", "cuda:3")]
    assert report["cuda_event_device"] == "cuda:3"
    assert report["cuda_events_measured"] is True
    assert report["cpu_diagnostic_only"] is False
    assert report["stages"]["dataload"]["host_wall"]["mean_ms"] == pytest.approx(10)
    assert report["stages"]["dataload"]["cuda_elapsed"]["count"] == 0
    assert report["stages"]["encoder"]["host_wall"]["mean_ms"] == pytest.approx(20)
    assert report["stages"]["encoder"]["cuda_elapsed"]["mean_ms"] == 1
    assert report["stages"]["decoder"]["cuda_elapsed"]["mean_ms"] == 3
    assert "not GPU utilization" in report["interpretation"]
    assert recorder.collect() == report
    assert len([call for call in cuda.calls if call[0] == "synchronize"]) == 1
    # Consumers cannot mutate the cached result.
    report["stages"]["encoder"]["cuda_elapsed"]["count"] = 999
    assert recorder.collect()["stages"]["encoder"]["cuda_elapsed"]["count"] == 1


def test_unindexed_cuda_device_is_pinned_at_first_recording(monkeypatch):
    cuda = _fake_cuda(monkeypatch)
    _clock(monkeypatch, [0.0, 0.001, 1.0, 1.001])
    recorder = timing.StageTimer("cuda", enabled=True, warmup_steps=0, measurement_steps=1)
    with recorder.scope("graph"):
        pass
    cuda.current_index = 3
    with recorder.scope("encoder"):
        pass
    recorder.step()
    report = recorder.collect()
    assert report["cuda_event_device"] == "cuda:2"
    assert [call for call in cuda.calls if call[0] == "stream"] == [("stream", "cuda"), ("stream", "cuda:2")]
    assert [call for call in cuda.calls if call[0] == "synchronize"] == [("synchronize", "cuda:2")]


def test_collect_without_gpu_spans_does_not_synchronize(monkeypatch):
    cuda = _fake_cuda(monkeypatch)
    _clock(monkeypatch, [0.0, 0.005])
    recorder = timing.StageTimer("cuda:2", enabled=True, warmup_steps=0, measurement_steps=1)
    with recorder.scope("dataload", gpu=False):
        pass
    recorder.step()
    report = recorder.collect()
    assert not report["cuda_events_measured"]
    assert report["cuda_event_device"] is None
    assert cuda.calls == []


def test_collection_finalizes_partial_window_and_no_more_work_is_recorded(monkeypatch):
    _clock(monkeypatch, [0.0, 0.005])
    recorder = timing.StageTimer("cpu", enabled=True, warmup_steps=0, measurement_steps=50)
    with recorder.scope("loss", gpu=False):
        pass
    recorder.step()
    report = recorder.collect()
    assert not report["window_complete"]
    assert report["measured_steps"] == 1
    assert not recorder.collecting
    with recorder.scope("loss"):
        pass
    assert recorder.step() is False
    assert recorder.collect() == report


def test_scope_record_cap_bounds_event_allocation_and_discloses_drops(monkeypatch):
    cuda = _fake_cuda(monkeypatch)
    _clock(monkeypatch, [0.0, 0.005, 1.0, 1.010])
    recorder = timing.StageTimer(
        "cuda:2", enabled=True, warmup_steps=0, measurement_steps=1, max_records=2
    )
    for _ in range(100):
        with recorder.scope("encoder"):
            pass
    recorder.step()
    report = recorder.collect()
    assert len(cuda.events) == 4
    assert report["recorded_scopes"] == 2
    assert report["dropped_scopes"] == 98


def test_inflight_collect_and_step_are_rejected_but_scope_can_finish(monkeypatch):
    _clock(monkeypatch, [0.0, 0.001])
    recorder = timing.StageTimer("cpu", enabled=True, warmup_steps=0, measurement_steps=1)
    with recorder.scope("model"):
        with pytest.raises(RuntimeError, match="scope is active"):
            recorder.collect()
        with pytest.raises(RuntimeError, match="Close all timing scopes"):
            recorder.step()
    assert recorder.step()
    assert recorder.collect()["recorded_scopes"] == 1


def test_nested_scopes_do_not_double_count_as_exclusive_time(monkeypatch):
    _clock(monkeypatch, [0.0, 0.001, 0.002, 0.005])
    recorder = timing.StageTimer("cpu", enabled=True, warmup_steps=0, measurement_steps=1)
    with recorder.scope("model"), recorder.scope("encoder"):
        pass
    recorder.step()
    report = recorder.collect()
    assert report["stages"]["model"]["host_wall"]["total_ms"] == 5
    assert report["stages"]["encoder"]["host_wall"]["total_ms"] == 1
    assert "may overlap" in report["interpretation"]


def test_failed_stage_preserves_exception_and_is_not_aggregated(monkeypatch):
    _clock(monkeypatch, [0.0])
    recorder = timing.StageTimer("cpu", enabled=True, warmup_steps=0, measurement_steps=1)
    with pytest.raises(ArithmeticError, match="training failure"), recorder.scope("backward"):
        raise ArithmeticError("training failure")
    report = recorder.collect()
    assert report["failed_scopes"] == 1
    assert report["recorded_scopes"] == 0
    assert not report["window_complete"]


def test_cuda_context_error_is_not_silently_reported_as_cpu(monkeypatch):
    cuda = _fake_cuda(monkeypatch)

    def unavailable(device):
        raise RuntimeError("CUDA context unavailable")

    cuda.current_stream = unavailable
    recorder = timing.StageTimer("cuda", enabled=True, warmup_steps=0, measurement_steps=1)
    with pytest.raises(RuntimeError, match="CUDA context unavailable"), recorder.scope("transfer"):
        pass
    assert recorder._active_scopes == 0
    assert not recorder.collect()["cpu_diagnostic_only"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": 1},
        {"warmup_steps": -1},
        {"warmup_steps": True},
        {"measurement_steps": 0},
        {"measurement_steps": 1.0},
        {"max_records": 0},
        {"max_records": False},
    ],
)
def test_invalid_recorder_configuration_is_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        timing.StageTimer("cpu", **kwargs)


def test_unknown_label_and_invalid_gpu_flag_are_rejected_in_active_window():
    recorder = timing.StageTimer("cpu", enabled=True, warmup_steps=0, measurement_steps=1)
    with pytest.raises(ValueError, match="Unknown training timing label"):
        recorder.scope("typo")
    with pytest.raises(TypeError, match="gpu must be a boolean"):
        recorder.scope("loss", gpu=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable on this test host")
def test_real_cuda_events_measure_an_actual_training_operation():
    device = torch.device("cuda", torch.cuda.current_device())
    recorder = timing.StageTimer(device, enabled=True, warmup_steps=0, measurement_steps=1)
    x = torch.randn((128, 128), device=device, requires_grad=True)
    with recorder.scope("model"):
        loss = (x @ x).square().mean()
    with recorder.scope("backward"):
        loss.backward()
    recorder.step()
    report = recorder.collect()
    assert report["cuda_events_measured"]
    assert not report["cpu_diagnostic_only"]
    for label in ("model", "backward"):
        stats = report["stages"][label]["cuda_elapsed"]
        assert stats["count"] == 1
        assert stats["mean_ms"] >= 0
