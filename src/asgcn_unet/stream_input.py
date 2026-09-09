"""Explicit physical-clock input contracts; no timestamp inference or rescaling fallback.

The legacy window-normalized input is unchanged. Physical streams retain every
event in the configured ROI and use source row IDs before any model-side R sampling.
Event timestamps and published frame boundaries require separate explicit unit
scales. They must already share a clock origin; no offset is inferred or applied.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np

LEGACY_EVENT_TIME_CONTRACT = "window_normalized_v1"
PHYSICAL_EVENT_TIME_CONTRACT = "physical_seconds_v1"


def reject_streaming_frame_diagnostic(model_config, dataset_config=None, sample=None):
    """Legacy frame graph tools cannot certify a graph with persistent context."""
    dataset_config = dataset_config or {}
    metadata = (sample or {}).get("metadata", {})
    timing = metadata.get("stream_time", {}) if isinstance(metadata, dict) else {}
    if (model_config.get("architecture_version") == 3
            or model_config.get("graph_execution") == "event_driven"
            or dataset_config.get("event_time_contract") == PHYSICAL_EVENT_TIME_CONTRACT
            or (isinstance(timing, dict) and timing.get("schema") == PHYSICAL_EVENT_TIME_CONTRACT)):
        raise ValueError("This diagnostic supports only v2 static frame graphs. Event-driven v3 requires "
                         "causal predecessor stream state, not legacy frame normalization or a reset sample. "
                         "Use the streaming training preflight and chronological evaluation; saved PNGs "
                         "remain viewable, but legacy graph export is not a streaming graph verification.")


def validate_event_time_contract(
    contract: str,
    timestamp_scale_to_seconds: float | None,
    max_events: int | None,
    *,
    interval_timestamp_scale_to_seconds: float | None = None,
) -> float | None:
    if not isinstance(contract, str) or contract not in {
        LEGACY_EVENT_TIME_CONTRACT, PHYSICAL_EVENT_TIME_CONTRACT,
    }:
        raise ValueError(f"Unsupported event_time_contract: {contract!r}")
    if contract == LEGACY_EVENT_TIME_CONTRACT:
        if timestamp_scale_to_seconds is not None or interval_timestamp_scale_to_seconds is not None:
            raise ValueError("timestamp_scale_to_seconds requires physical_seconds_v1")
        return None
    if max_events is not None:
        raise ValueError("physical_seconds_v1 requires explicit max_events=null; no event cap")
    scale = timestamp_scale_to_seconds
    if scale is None:
        raise ValueError("physical_seconds_v1 requires explicit timestamp_scale_to_seconds")
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise TypeError("timestamp_scale_to_seconds must be a real number, not bool")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("timestamp_scale_to_seconds must be finite and positive")
    frame_scale = interval_timestamp_scale_to_seconds
    if frame_scale is None:
        raise ValueError("physical_seconds_v1 requires explicit interval_timestamp_scale_to_seconds")
    if isinstance(frame_scale, bool) or not isinstance(frame_scale, Real):
        raise TypeError("interval_timestamp_scale_to_seconds must be a real number, not bool")
    if not math.isfinite(float(frame_scale)) or frame_scale <= 0:
        raise ValueError("interval_timestamp_scale_to_seconds must be finite and positive")
    return scale


def to_physical_seconds(values: Any, scale: float, *, source: str) -> np.ndarray:
    """Apply the declared source unit scale without inferring a clock offset."""
    original = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = original * scale
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Invalid physical clock in {source}: seconds must be finite")
    if np.any((original != 0) & (result == 0)) or (
        original.ndim == 1 and np.any(
            (original[1:] != original[:-1]) & (result[1:] == result[:-1])
        )
    ):
        raise ValueError(f"Physical clock conversion loses timestamp resolution in {source}")
    return result


def arrival_group_counts(event_seconds: np.ndarray) -> tuple[int, ...]:
    """CPU collation metadata for consecutive equal timestamps after ROI filtering.

Declared scaling must not merge distinct source timestamps, as checked by
``to_physical_seconds``. Thus these groups equal the original raw timestamp
groups without carrying another per-event timestamp allocation to the GPU.
"""
    values = np.asarray(event_seconds)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("arrival timestamps must be a finite vector")
    if np.any(values[1:] < values[:-1]):
        raise ValueError("arrival timestamps must be monotonically non-decreasing")
    if len(values) == 0:
        return ()
    boundaries = np.concatenate(([0], np.flatnonzero(values[1:] != values[:-1]) + 1, [len(values)]))
    return tuple(np.diff(boundaries).tolist())


def hdr_boundary_policy(
    timestamps: Any, index: int, boundary: float, *, source: str,
    timestamp_scale_to_seconds: float = 1.0,
    interval_timestamp_scale_to_seconds: float = 1.0,
) -> str:
    """Validate a stored/recovered boundary against its adjacent original rows.

Both ordinary left/right timestamp boundaries and the published predecessor
index are accepted, without changing any index. The predecessor convention is
max(searchsorted(left)-1, 0), not a half-open time interval: its excluded row
can be delivered by the next frame after that row's physical timestamp.
"""
    count = len(timestamps)
    if not math.isfinite(boundary) or not 0 <= index <= count:
        raise ValueError(f"Invalid EventHDR physical boundary in {source}")
    boundary = float(to_physical_seconds(boundary, interval_timestamp_scale_to_seconds, source=source))
    if count == 0:
        return "empty_event_stream"
    lo, hi = max(0, index - 1), min(count, index + 2)
    neighbors = np.asarray(timestamps[lo:hi], dtype=np.float64)
    neighbors = to_physical_seconds(neighbors, timestamp_scale_to_seconds, source=source)
    if not np.all(np.isfinite(neighbors)) or np.any(neighbors[1:] < neighbors[:-1]):
        raise ValueError(f"Invalid EventHDR boundary timestamp rows in {source}")
    before = float(neighbors[index - 1 - lo]) if index else None
    after = float(neighbors[index - lo]) if index < count else None
    if (before is None or before < boundary) and (after is None or after >= boundary):
        return "timestamp_left_boundary"
    if (before is None or before <= boundary) and (after is None or after > boundary):
        return "timestamp_right_boundary"
    next_after = float(neighbors[index + 1 - lo]) if index + 1 < count else None
    if after is not None and after < boundary and (
        next_after is None or next_after >= boundary
    ):
        return "timestamp_predecessor_v1"
    raise ValueError(
        f"EventHDR physical clock/index mismatch in {source}: event_idx={index} "
        "is neither a timestamp boundary nor the documented predecessor row"
    )


def stream_time_metadata(
    event_seconds: np.ndarray,
    *,
    interval_start_seconds: float,
    interval_end_seconds: float,
    sequence_origin_seconds: float,
    timestamp_scale_to_seconds: float,
    interval_timestamp_scale_to_seconds: float,
    source: str,
    boundary_policy: str = "closed_timestamp_interval",
    allow_predecessor_row: bool = False,
) -> dict[str, Any]:
    """Check every source event before cropping and record any proven late row.

No epsilon, inferred unit, clock offset or dropped event repairs a mismatch.
The initial origin is fixed per sequence from its published first boundary and
first event, never from a later sample's min/max or its retained event count.
"""
    bounds = (interval_start_seconds, interval_end_seconds, sequence_origin_seconds)
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError(f"Invalid physical interval in {source}: bounds must be finite")
    if interval_end_seconds < interval_start_seconds or sequence_origin_seconds > interval_start_seconds:
        raise ValueError(f"Invalid physical interval ordering in {source}")
    event_seconds = np.asarray(event_seconds, dtype=np.float64)
    if event_seconds.ndim != 1 or not np.all(np.isfinite(event_seconds)):
        raise ValueError(f"Invalid physical event timestamps in {source}")
    if np.any(event_seconds[1:] < event_seconds[:-1]):
        raise ValueError(f"Physical event timestamps must be ordered in {source}")
    early = event_seconds < interval_start_seconds
    early_count = int(np.count_nonzero(early))
    allowed_early = allow_predecessor_row and early_count == 1 and bool(early[0])
    if (early_count and not allowed_early) or np.any(event_seconds > interval_end_seconds):
        raise ValueError(
            f"Physical event/frame clock mismatch in {source}: events must lie within "
            "the declared frame interval (only a proven EventHDR predecessor row is allowed)"
        )
    return {
        "schema": PHYSICAL_EVENT_TIME_CONTRACT,
        "interval_start_seconds": float(interval_start_seconds),
        "interval_end_seconds": float(interval_end_seconds),
        "sequence_origin_seconds": float(sequence_origin_seconds),
        "timestamp_scale_to_seconds": float(timestamp_scale_to_seconds),
        "interval_timestamp_scale_to_seconds": float(interval_timestamp_scale_to_seconds),
        "boundary_policy": boundary_policy,
        "strict_interval_validation": True,
        "late_predecessor_event_count": early_count,
        "late_predecessor_seconds": (
            float(interval_start_seconds - event_seconds[0]) if early_count else 0.0
        ),
        "clock_correction_applied": False,
    }
