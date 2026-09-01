"""Boundary utilities; generated contexts exist only in test memory."""

from __future__ import annotations

import copy
import signal
import threading

import pytest
import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from asgcn_unet import checkpoint
from asgcn_unet.checkpoint import (
    CursorBatchSampler,
    StopRequest,
    capture_training_state,
    restore_training_state,
)
from asgcn_unet.training import TrainingState


class _DecodeTracked:
    def __init__(self):
        self.decoded = []

    def __len__(self):
        return 7

    def __getitem__(self, index):
        self.decoded.append(index)
        return index


def test_cursor_does_not_decode_completed_prefix_and_keeps_partial_tail():
    dataset = _DecodeTracked()
    schedule = BatchSampler(SequentialSampler(dataset), batch_size=3, drop_last=False)
    cursor = CursorBatchSampler(schedule, start=1)
    loader = DataLoader(dataset, batch_sampler=cursor, num_workers=0)
    assert len(cursor) == 2 and cursor.total_batches == 3
    assert [batch.tolist() for batch in loader] == [[3, 4, 5], [6]]
    assert dataset.decoded == [3, 4, 5, 6]
    cursor.set_start(3)
    assert len(cursor) == 0 and list(cursor) == []
    cursor.set_start(0)
    assert list(cursor) == [[0, 1, 2], [3, 4, 5], [6]]


@pytest.mark.parametrize("start", [-1, 4, True, 1.5, "1", None])
def test_cursor_rejects_invalid_start(start):
    with pytest.raises((TypeError, ValueError), match="cursor"):
        CursorBatchSampler([[1], [2], [3]], start=start)


def test_cursor_requires_sized_iterable_and_revalidates_schedule_changes():
    with pytest.raises(TypeError, match="sized"):
        CursorBatchSampler(iter([[0]]))
    schedule = [[0], [1]]
    cursor = CursorBatchSampler(schedule, start=2)
    schedule.pop()
    with pytest.raises(ValueError, match="outside"):
        len(cursor)
    with pytest.raises(ValueError, match="outside"):
        iter(cursor)


def test_cursor_set_epoch_delegates_and_resets():
    class Schedule:
        epoch = 0

        def __len__(self):
            return 2

        def __iter__(self):
            return iter([[self.epoch], [self.epoch + 1]])

        def set_epoch(self, epoch):
            self.epoch = epoch

    schedule = Schedule()
    cursor = CursorBatchSampler(schedule, start=1)
    cursor.set_epoch(4)
    assert cursor.start == 0 and list(cursor) == [[4], [5]]
    plain = CursorBatchSampler([[2]], start=1)
    plain.set_epoch(2)
    assert list(plain) == [[2]]
    for invalid in (-1, 1.5, True):
        with pytest.raises(ValueError, match="epoch"):
            cursor.set_epoch(invalid)


def _state(*, independent=True):
    state = TrainingState(independent_sequences=independent)
    key = ("sequence-a", "sample.h5" if independent else "")
    state.values[key] = (
        7, (8, 10), torch.randn(1, 4, 2, 3, requires_grad=True),
        torch.randn(1, 1, 8, 10, requires_grad=True), torch.randn(1, 1, 8, 10),
    )
    state.last_key = key
    return state


@pytest.mark.parametrize("independent", [False, True])
def test_context_round_trip_detaches_and_copies_all_tensor_storage(independent):
    original = _state(independent=independent)
    payload = capture_training_state(original)
    restored = restore_training_state(payload, independent_sequences=independent, device="cpu")
    assert restored.last_key == original.last_key
    assert restored.independent_sequences is independent
    key = original.last_key
    assert restored.values[key][:2] == original.values[key][:2]
    for offset, name in enumerate(("recurrent", "prediction", "target"), start=2):
        source = original.values[key][offset]
        saved = payload["entries"][0][name]
        loaded = restored.values[key][offset]
        torch.testing.assert_close(source, saved, rtol=0, atol=0)
        torch.testing.assert_close(source, loaded, rtol=0, atol=0)
        assert len({source.data_ptr(), saved.data_ptr(), loaded.data_ptr()}) == 3
        assert saved.device.type == loaded.device.type == "cpu"
        assert not saved.requires_grad and not loaded.requires_grad
    restored.values[key][3].fill_(42)
    assert not torch.equal(restored.values[key][3], payload["entries"][0]["prediction"])


def test_empty_context_and_evicted_last_key_round_trip():
    state = TrainingState(independent_sequences=True)
    assert restore_training_state(
        capture_training_state(state), independent_sequences=True, device="cpu"
    ).values == {}
    state.last_key = ("finished-sequence", "sample.h5")
    restored = restore_training_state(
        capture_training_state(state), independent_sequences=True, device="cpu"
    )
    assert restored.values == {} and restored.last_key == state.last_key


def test_none_recurrent_and_missing_sequence_index_are_preserved():
    state = _state()
    value = state.values[state.last_key]
    state.values[state.last_key] = (None, value[1], None, value[3], value[4])
    restored = restore_training_state(
        capture_training_state(state), independent_sequences=True, device="cpu"
    )
    assert restored.values[state.last_key][:3] == (None, (8, 10), None)


@pytest.mark.parametrize("field,value", [
    ("version", 2), ("version", True), ("entries", ()),
    ("last_key", "sequence"), ("last_key", ("", "sample.h5")),
    ("last_key", None), ("independent_sequences", False), ("independent_sequences", 1),
])
def test_context_rejects_corrupt_top_level_fields(field, value):
    payload = capture_training_state(_state())
    payload[field] = value
    with pytest.raises((TypeError, ValueError)):
        restore_training_state(payload, independent_sequences=True, device="cpu")


@pytest.mark.parametrize("field,value", [
    ("key", ("sequence", 3)), ("sequence_index", -1),
    ("sequence_index", True), ("sequence_index", 3.5),
    ("sensor_size", (0, 10)), ("sensor_size", (8, True)), ("sensor_size", (8, 11)),
    ("recurrent", torch.zeros(4, 2, 3)), ("prediction", None),
    ("prediction", torch.zeros(1, 1, 8, 10, dtype=torch.int64)),
    ("target", torch.zeros(2, 1, 8, 10)),
    ("recurrent", torch.full((1, 4, 2, 3), float("nan"))),
    ("prediction", torch.full((1, 1, 8, 10), float("inf"))),
    ("target", torch.full((1, 1, 8, 10), float("-inf"))),
])
def test_context_rejects_corrupt_entries_before_copy(field, value, monkeypatch):
    payload = capture_training_state(_state())
    payload["entries"][0][field] = value

    def forbidden_transfer(*args, **kwargs):
        raise AssertionError("Invalid checkpoints must be rejected before any device transfer")

    monkeypatch.setattr(torch.Tensor, "to", forbidden_transfer)
    with pytest.raises((TypeError, ValueError)):
        restore_training_state(payload, independent_sequences=True, device="cpu")


def test_context_rejects_unknown_missing_duplicate_and_incompatible_keys():
    original = capture_training_state(_state())
    candidates = []
    payload = copy.deepcopy(original)
    payload["unknown"] = 1
    candidates.append(payload)
    payload = copy.deepcopy(original)
    del payload["entries"][0]["target"]
    candidates.append(payload)
    payload = copy.deepcopy(original)
    payload["entries"].append(copy.deepcopy(payload["entries"][0]))
    candidates.append(payload)
    for payload in candidates:
        with pytest.raises(ValueError):
            restore_training_state(payload, independent_sequences=True, device="cpu")
    single = capture_training_state(_state(independent=False))
    single["last_key"] = ("different-sequence", "")
    with pytest.raises(ValueError, match="disagree"):
        restore_training_state(single, independent_sequences=False, device="cpu")


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf"), True, "3"])
def test_stop_request_rejects_invalid_time_limit(limit):
    with pytest.raises(ValueError, match="positive"):
        StopRequest(limit)


def _fake_handlers(monkeypatch):
    original = {signal.SIGINT: object(), signal.SIGTERM: object()}
    installed = original.copy()
    monkeypatch.setattr(signal, "getsignal", lambda signum: installed[signum])
    monkeypatch.setattr(signal, "signal", lambda signum, fn: installed.update({signum: fn}))
    return original, installed


def test_signal_only_requests_pause_and_handlers_restore_even_on_failure(monkeypatch):
    original, installed = _fake_handlers(monkeypatch)
    stop = StopRequest()
    with pytest.raises(RuntimeError, match="training error"), stop:
        assert not stop.pause_requested
        installed[signal.SIGTERM](signal.SIGTERM, None)
        assert stop.poll() is True and stop.reason == "SIGTERM"
        installed[signal.SIGINT](signal.SIGINT, None)
        assert stop.reason == "SIGTERM", "The first request must not be overwritten"
        raise RuntimeError("training error")
    assert installed == original


def test_time_limit_starts_on_entry_and_stays_latched(monkeypatch):
    _fake_handlers(monkeypatch)
    clock = [10.0]
    monkeypatch.setattr(checkpoint.time, "monotonic", lambda: clock[0])
    stop = StopRequest(5)
    clock[0] = 500.0
    assert not stop.poll()
    with stop:
        clock[0] = 504.99
        assert not stop.pause_requested
        clock[0] = 505.0
        assert stop.pause_requested and stop.reason == "time_limit"
        clock[0] = 504.0
        assert stop.poll(), "Once requested a pause stays latched"
    with stop:
        assert not stop.poll(), "A new context resets the timer and previous reasons"


def test_stop_request_does_not_install_process_signal_handlers_on_worker_thread(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Only the main thread can install process signal handlers")

    monkeypatch.setattr(signal, "signal", forbidden)
    monkeypatch.setattr(signal, "getsignal", forbidden)
    errors = []

    def worker():
        try:
            with StopRequest() as stop:
                assert not stop.poll()
        except BaseException as error:  # noqa: BLE001 - propagate worker-thread failures.
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert errors == []


def test_stop_request_rejects_nested_reentry_and_restores_partial_install(monkeypatch):
    original, installed = _fake_handlers(monkeypatch)
    with StopRequest() as stop, pytest.raises(RuntimeError, match="twice"):
        stop.__enter__()
    assert installed == original

    def partial_failure(signum, callback):
        if signum == signal.SIGTERM:
            raise RuntimeError("handler install rejected")
        installed[signum] = callback

    monkeypatch.setattr(signal, "signal", partial_failure)
    stop = StopRequest()
    with pytest.raises(RuntimeError, match="install rejected"), stop:
        raise AssertionError("The context must not start after handler installation failure")
    assert installed == original
    assert not stop._active
