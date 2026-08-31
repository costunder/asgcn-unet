"""Opt-in, bounded timing of the actual training path.

Host wall spans measure Python/dispatch/wait time, not GPU execution. CUDA
events measure elapsed stream time, not GPU utilization or exclusive kernel
time. Nested spans may overlap and must not be summed as a step duration.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager, nullcontext
from time import perf_counter
from typing import Any

import torch

_LABELS = (
    "dataload",
    "transfer",
    "graph",
    "encoder",
    "decoder",
    "model",
    "loss",
    "backward",
    "gradient_check",
    "optimizer",
    "step",
)
_NO_SCOPE = nullcontext()


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _summary(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "total_ms": 0.0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    total = math.fsum(values)
    return {
        "count": len(values),
        "total_ms": total,
        "mean_ms": total / len(values),
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
    }


class StageTimer:
    """Time at most ``measurement_steps`` after a fixed warmup.

    Call ``step()`` once after each complete training iteration. It returns
    true only when the requested window has just completed. ``collect()``
    explicitly finalizes the window, including a partial window, and performs
    one selected-device synchronization if CUDA events were recorded. Repeated
    collection returns the cached report without another synchronization.

    Disabled, warmup and completed-window scopes return a shared no-op context;
    they do not read clocks, inspect CUDA or create events. The record cap is an
    additional bound on repeated/nested scopes within an individual iteration.
    """

    def __init__(
        self,
        device: torch.device | str,
        *,
        enabled: bool = False,
        warmup_steps: int = 10,
        measurement_steps: int = 50,
        max_records: int | None = None,
    ) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        self.enabled = enabled
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("StageTimer supports CPU diagnostics and CUDA training only")
        self.warmup_steps = _integer(warmup_steps, "warmup_steps", 0)
        self.measurement_steps = _integer(measurement_steps, "measurement_steps", 1)
        self.max_records = _integer(
            self.measurement_steps * len(_LABELS) * 4 if max_records is None else max_records,
            "max_records",
            1,
        )
        self._step_index = 0
        self._active_scopes = 0
        self._reserved_records = 0
        self._dropped_scopes = 0
        self._failed_scopes = 0
        self._cuda_device: torch.device | None = None
        self._records: list[tuple[str, float, Any, Any]] = []
        self._report: dict[str, Any] | None = None

    @property
    def collecting(self) -> bool:
        return (
            self.enabled
            and self._report is None
            and self.warmup_steps <= self._step_index < self.warmup_steps + self.measurement_steps
        )

    def scope(self, label: str, *, gpu: bool = True):
        if not self.collecting:
            return _NO_SCOPE
        if label not in _LABELS:
            raise ValueError(f"Unknown training timing label: {label!r}")
        if not isinstance(gpu, bool):
            raise TypeError("gpu must be a boolean")
        if self._reserved_records >= self.max_records:
            self._dropped_scopes += 1
            return _NO_SCOPE
        return self._scope(label, gpu)

    @contextmanager
    def _scope(self, label: str, gpu: bool):
        # Scope creation and entry may be separated by caller code.
        if not self.collecting:
            yield
            return
        if self._reserved_records >= self.max_records:
            self._dropped_scopes += 1
            yield
            return
        self._reserved_records += 1
        self._active_scopes += 1
        start_event = end_event = None
        stream = None
        try:
            if gpu and self.device.type == "cuda":
                # Explicitly use the selected device's current stream. Do not
                # silently fall back to CPU if that CUDA context is unavailable.
                stream = torch.cuda.current_stream(self._cuda_device or self.device)
                if self._cuda_device is None:
                    self._cuda_device = torch.device(stream.device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(stream)
            started = perf_counter()
            try:
                yield
            except BaseException:
                self._failed_scopes += 1
                raise
            else:
                host_ms = (perf_counter() - started) * 1000.0
                if end_event is not None:
                    end_event.record(stream)
                self._records.append((label, host_ms, start_event, end_event))
        finally:
            self._active_scopes -= 1

    def step(self) -> bool:
        if not self.enabled or self._report is not None:
            return False
        if self._active_scopes:
            raise RuntimeError("Close all timing scopes before completing a training step")
        end = self.warmup_steps + self.measurement_steps
        if self._step_index >= end:
            return False
        self._step_index += 1
        return self._step_index == end

    def collect(self) -> dict[str, Any]:
        if self._active_scopes:
            raise RuntimeError("Cannot collect timings while a scope is active")
        if self._report is not None:
            return copy.deepcopy(self._report)
        cuda_records = [record for record in self._records if record[2] is not None]
        if cuda_records:
            # The only barrier introduced by this recorder: once per window,
            # never at individual stage boundaries.
            torch.cuda.synchronize(self._cuda_device)
        host_spans: dict[str, list[float]] = {label: [] for label in _LABELS}
        cuda: dict[str, list[float]] = {label: [] for label in _LABELS}
        for label, host_ms, start_event, end_event in self._records:
            host_spans[label].append(host_ms)
            if start_event is not None:
                elapsed = float(start_event.elapsed_time(end_event))
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise RuntimeError("CUDA timing returned a non-finite or negative duration")
                cuda[label].append(elapsed)
        measured_steps = min(
            max(self._step_index - self.warmup_steps, 0), self.measurement_steps
        )
        self._report = {
            "format_version": 1,
            "enabled": self.enabled,
            "device": str(self.device),
            "cuda_event_device": str(self._cuda_device) if cuda_records else None,
            "cpu_diagnostic_only": self.enabled and self.device.type == "cpu",
            "cuda_events_measured": bool(cuda_records),
            "warmup_steps": self.warmup_steps,
            "requested_steps": self.measurement_steps,
            "measured_steps": measured_steps,
            "window_complete": self.enabled and measured_steps == self.measurement_steps,
            "max_records": self.max_records,
            "recorded_scopes": len(self._records),
            "dropped_scopes": self._dropped_scopes,
            "failed_scopes": self._failed_scopes,
            "stages": {
                label: {
                    "host_wall": _summary(host_spans[label]),
                    "cuda_elapsed": _summary(cuda[label]),
                }
                for label in _LABELS
            },
            "interpretation": (
                "Host wall time includes dispatch and waits, not GPU execution time. "
                "CUDA events measure elapsed time on the recorded stream, not GPU utilization "
                "or exclusive kernel time. Nested stage durations may overlap."
            ),
        }
        self._records.clear()
        return copy.deepcopy(self._report)
