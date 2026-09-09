"""Strict batch-scope aggregation of measured streaming execution counters."""

from __future__ import annotations

import copy
from typing import Any

_SCALARS = frozenset({"arrival_updates", "readout_updates", "incoming_events", "topology_indexed_edges"})
_LAYERS = frozenset({"updated_nodes_per_layer", "message_edges_per_layer", "projected_sources_per_layer"})
_REQUIRED = frozenset({"arrival_updates", "readout_updates", "incoming_events", "training_dense_snapshot"})


def _validate(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("stream_execution must be a batch-scope dictionary")
    if not _REQUIRED.issubset(value) or set(value) - (_SCALARS | _LAYERS | {"training_dense_snapshot"}):
        raise ValueError("stream_execution has missing or unsupported execution fields")
    if type(value["training_dense_snapshot"]) is not bool:
        raise TypeError("stream_execution.training_dense_snapshot must be boolean")
    lengths = set()
    for key, count in value.items():
        if key in _SCALARS and (type(count) is not int or count < 0):
            raise ValueError(f"stream_execution.{key} must be a nonnegative integer count")
        if key in _LAYERS:
            if not isinstance(count, list) or not count or any(type(item) is not int or item < 0 for item in count):
                raise ValueError(f"stream_execution.{key} must contain nonnegative per-layer integer counts")
            lengths.add(len(count))
    if len(lengths) > 1:
        raise ValueError("stream_execution per-layer counter lengths differ")
    return value


def aggregate_stream_execution(
    current: dict[str, Any] | None, diagnostics: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate repeated lane metadata and count one actual forward exactly once.

    Scalars and corresponding layer counters are summed; execution-mode booleans
    must remain identical. Missing counters are never fabricated as zero. Static
    diagnostics leave a previously empty accumulator as None. Mixing static and
    streaming diagnostics, mismatched lanes, or incompatible executions is an
    explicit error. Neither source diagnostics nor the old aggregate are mutated.
    """
    if not diagnostics:
        raise ValueError("A stream execution aggregation requires a nonempty physical batch")
    present = ["stream_execution" in detail for detail in diagnostics]
    if not any(present):
        if current is not None:
            raise ValueError("Cannot mix static diagnostics into a streaming execution aggregate")
        return None
    if not all(present):
        raise ValueError("A physical batch mixes missing and present stream_execution diagnostics")
    values = [_validate(detail["stream_execution"]) for detail in diagnostics]
    measured = values[0]
    if any(value != measured for value in values[1:]):
        raise ValueError("Per-lane stream_execution differs despite its shared batch scope")
    if current is None:
        return {"scope": "batch_once", "physical_batches": 1, "frames": len(diagnostics),
                **copy.deepcopy(measured)}
    previous = {key: value for key, value in current.items() if key not in {"scope", "physical_batches", "frames"}}
    _validate(previous)
    if (current.get("scope") != "batch_once" or type(current.get("physical_batches")) is not int
            or current["physical_batches"] < 1 or type(current.get("frames")) is not int or current["frames"] < 1):
        raise ValueError("Invalid previous stream execution aggregation scope/counts")
    if previous.keys() != measured.keys():
        raise ValueError("Streaming execution counter coverage changed between batches")
    if previous["training_dense_snapshot"] != measured["training_dense_snapshot"]:
        raise ValueError("Streaming execution training_dense_snapshot changed between batches")
    result = copy.deepcopy(current)
    result["physical_batches"] += 1
    result["frames"] += len(diagnostics)
    for key in _SCALARS & measured.keys():
        result[key] += measured[key]
    for key in _LAYERS & measured.keys():
        if len(previous[key]) != len(measured[key]):
            raise ValueError("Streaming per-layer counter lengths changed between batches")
        result[key] = [left + right for left, right in zip(previous[key], measured[key], strict=True)]
    return result
