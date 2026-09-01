"""Safe training-boundary interruption, index cursors and causal-state snapshots."""

from __future__ import annotations

import math
import signal
import threading
import time
from collections.abc import Iterable, Iterator, Sized
from itertools import islice
from numbers import Real
from types import TracebackType
from typing import Any, Self

import torch
from torch.utils.data import Sampler

from .training import TrainingState


class CursorBatchSampler(Sampler[list[int]]):
    """Resume a deterministic batch schedule without decoding skipped samples.

    The underlying sampler must reproduce its complete epoch schedule on every
    iteration. Only index batches are skipped; a DataLoader receives the remaining
    indices and therefore neither reads nor prefetches the completed prefix.
    """

    def __init__(self, sampler: Iterable[list[int]], start: int = 0) -> None:
        if not isinstance(sampler, Sized) or not isinstance(sampler, Iterable):
            raise TypeError("A cursor requires an iterable, sized batch sampler")
        self.sampler = sampler
        self.start = 0
        self.set_start(start)

    @property
    def total_batches(self) -> int:
        return len(self.sampler)

    def set_start(self, start: int) -> None:
        if isinstance(start, bool) or not isinstance(start, int):
            raise TypeError("Batch cursor must be a nonnegative integer")
        if not 0 <= start <= self.total_batches:
            raise ValueError("Batch cursor is outside the complete epoch schedule")
        self.start = start

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Sampler epoch must be a nonnegative integer")
        set_epoch = getattr(self.sampler, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(epoch)
        self.set_start(0)

    def __len__(self) -> int:
        self.set_start(self.start)
        return self.total_batches - self.start

    def __iter__(self) -> Iterator[list[int]]:
        self.set_start(self.start)
        return islice(iter(self.sampler), self.start, None)


def _context_key(value: Any, name: str) -> tuple[str, str]:
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or not all(isinstance(part, str) for part in value) or not value[0]):
        raise ValueError(f"{name} must contain a nonempty sequence identity and source string")
    return tuple(value)


def _context_index(value: Any) -> int | None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError("Training context sequence_index must be nonnegative or None")
    return value


def _context_size(value: Any) -> tuple[int, int]:
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(isinstance(part, bool) or not isinstance(part, int) or part < 1
                   for part in value)):
        raise ValueError("Training context sensor_size must contain two positive integers")
    return tuple(value)


def _context_tensor(value: Any, name: str, *, optional: bool = False) -> torch.Tensor | None:
    if optional and value is None:
        return None
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or not value.is_floating_point() or value.ndim != 4
            or value.shape[0] != 1 or any(size < 1 for size in value.shape)):
        raise ValueError(f"Training context {name} must be a floating 1xCxHxW tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"Training context {name} contains non-finite values")
    return value


def _validated_context(payload: Any, independent_sequences: bool) -> list[dict[str, Any]]:
    if not isinstance(independent_sequences, bool):
        raise TypeError("independent_sequences must be a boolean")
    expected = {"version", "independent_sequences", "last_key", "entries"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("Training context checkpoint has an invalid schema")
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("Unsupported training context checkpoint version")
    if (not isinstance(payload["independent_sequences"], bool)
            or payload["independent_sequences"] != independent_sequences):
        raise ValueError("Training context batching mode does not match the training configuration")
    last_key = payload["last_key"]
    if last_key is not None:
        last_key = _context_key(last_key, "Training context last_key")
    entries = payload["entries"]
    if not isinstance(entries, list):
        raise TypeError("Training context entries must be a list")
    if not independent_sequences and len(entries) > 1:
        raise ValueError("Single-frame training cannot retain multiple sequence contexts")
    seen = set()
    validated = []
    for entry in entries:
        fields = {"key", "sequence_index", "sensor_size", "recurrent", "prediction", "target"}
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError("Training context entry has an invalid schema")
        key = _context_key(entry["key"], "Training context key")
        if key in seen:
            raise ValueError("Training context contains duplicate sequence keys")
        seen.add(key)
        index = _context_index(entry["sequence_index"])
        size = _context_size(entry["sensor_size"])
        recurrent = _context_tensor(entry["recurrent"], "recurrent", optional=True)
        prediction = _context_tensor(entry["prediction"], "prediction")
        target = _context_tensor(entry["target"], "target")
        if prediction.shape != target.shape or tuple(prediction.shape[-2:]) != size:
            raise ValueError("Training context prediction/target shape differs from sensor_size")
        if not independent_sequences and (key[1] or last_key != key):
            raise ValueError("Single-frame training context key and last_key disagree")
        validated.append({
            "key": key, "sequence_index": index, "sensor_size": size,
            "recurrent": recurrent, "prediction": prediction, "target": target,
        })
    if entries and last_key is None:
        raise ValueError("Retained training contexts require a last_key")
    # release_finished deliberately leaves last_key intact, even if its value
    # was evicted. Requiring membership would reject valid boundary snapshots.
    return validated


def capture_training_state(state: TrainingState) -> dict[str, Any]:
    """Capture independent CPU-owned, detached tensors at a successful boundary."""
    if not isinstance(state, TrainingState):
        raise TypeError("A checkpoint requires TrainingState")
    entries = []
    for key, value in state.values.items():
        if not isinstance(value, tuple) or len(value) != 5:
            raise ValueError("Training context value must contain five fields")
        entries.append({
            "key": key, "sequence_index": value[0], "sensor_size": value[1],
            "recurrent": value[2], "prediction": value[3], "target": value[4],
        })
    payload = {
        "version": 1, "independent_sequences": state.independent_sequences,
        "last_key": state.last_key, "entries": entries,
    }
    validated = _validated_context(payload, state.independent_sequences)
    for entry in validated:
        for name in ("recurrent", "prediction", "target"):
            if entry[name] is not None:
                entry[name] = entry[name].detach().to(device="cpu", copy=True)
    payload["entries"] = validated
    return payload


def restore_training_state(
    payload: dict[str, Any], *, independent_sequences: bool, device: torch.device | str
) -> TrainingState:
    """Validate before transfer; never share storage with a loaded checkpoint."""
    entries = _validated_context(payload, independent_sequences)
    state = TrainingState(independent_sequences=independent_sequences)
    for entry in entries:
        tensors = tuple(
            None if entry[name] is None else entry[name].detach().to(device=device, copy=True)
            for name in ("recurrent", "prediction", "target")
        )
        state.values[entry["key"]] = (entry["sequence_index"], entry["sensor_size"], *tensors)
    state.last_key = (
        None if payload["last_key"] is None else tuple(payload["last_key"])
    )
    return state


class StopRequest:
    """Defer SIGINT/SIGTERM to a caller-controlled, consistent training boundary.

    A handler only records the first signal. Checkpoint writing, CUDA completion
    and exceptions belong to the training loop, never to this asynchronous hook.
    This cannot intercept SIGKILL, loss of power or a scheduler's hard deadline.
    """

    def __init__(self, time_limit_seconds: float | None = None) -> None:
        if time_limit_seconds is not None and (
            isinstance(time_limit_seconds, bool) or not isinstance(time_limit_seconds, Real)
            or not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0
        ):
            raise ValueError("Training time limit must be a finite positive number")
        self.time_limit_seconds = time_limit_seconds
        self._reason: str | None = None
        self._started: float | None = None
        self._active = False
        self._handlers: dict[int, Any] = {}

    def _handle_signal(self, signum: int, frame: Any) -> None:
        del frame
        if self._reason is None:
            self._reason = signal.Signals(signum).name

    def __enter__(self) -> Self:
        if self._active:
            raise RuntimeError("A StopRequest context cannot be entered twice")
        self._active = True
        self._reason = None
        self._started = time.monotonic()
        try:
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    previous = signal.getsignal(signum)
                    signal.signal(signum, self._handle_signal)
                    self._handlers[signum] = previous
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            for signum, previous in self._handlers.items():
                signal.signal(signum, previous)
        finally:
            self._handlers.clear()
            self._active = False

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def pause_requested(self) -> bool:
        return self.poll()

    def poll(self) -> bool:
        if (self._reason is None and self._active and self.time_limit_seconds is not None
                and time.monotonic() - self._started >= self.time_limit_seconds):
            self._reason = "time_limit"
        return self._reason is not None
