"""Sequence-global ordinal sampling after full raw-stream validation.

This module does not validate raw event chronology or cross-frame identities:
the streaming caller must validate *all raw events* before calling it. Sampling
must never hide a malformed event or replace the raw last-event ID in state.
Only the explicit factor changes event selection; clocks, targets, geometry and
the physical batch remain unchanged. Counters count every raw event, not just
selected events, and belong to the sequence rather than the frame/batch lane.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .batching import PackedSampleBatch


@dataclass(frozen=True)
class StreamSamplingResult:
    packed: PackedSampleBatch
    keep_mask: torch.Tensor
    next_offsets: tuple[int, ...]
    arrival_group_counts: tuple[tuple[int, ...], ...]


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _groups(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list/tuple of positive integer counts")
    return tuple(_integer(count, name, minimum=1) for count in value)


def _retained_count(count: int, prior: int, factor: int) -> int:
    first = (-prior) % factor
    return 0 if first >= count else 1 + (count - 1 - first) // factor


def sample_stream_batch(
    packed: PackedSampleBatch, prior_offsets: Sequence[int], *, factor: int,
) -> StreamSamplingResult:
    """Keep raw sequence ordinals ``(prior + local_ordinal) % factor == 0``.

    ``prior_offsets`` are Python integer counts of previously validated *raw*
    events for each independent sequence in packed order. They need not fit in
    int64: Python modulo is applied before constructing tensor indices, avoiding
    counter overflow on long sequences. ``factor`` must fit positive int64 for
    the tensor remainder operation. Neither argument has an inferred default.

    Arrival groups are recomputed arithmetically from already-validated raw
    equal-timestamp groups, without host reads of GPU tensors or per-lane tensor
    forwards. Empty selections retain their sample/target and updated raw count.
    R=1 returns the original packed object, preserving the existing v3 data path.
    """
    _integer(factor, "factor", minimum=1)
    if factor > torch.iinfo(torch.long).max:
        raise ValueError("factor exceeds the int64 tensor-index range")
    if not isinstance(packed, PackedSampleBatch) or not len(packed):
        raise TypeError("Sampling requires a nonempty PackedSampleBatch")
    if (not isinstance(prior_offsets, Sequence) or isinstance(prior_offsets, (str, bytes))
            or len(prior_offsets) != len(packed)):
        raise ValueError("prior_offsets must contain one raw count per packed sequence")
    prior = tuple(_integer(value, "prior_offsets") for value in prior_offsets)
    counts = tuple(_integer(value, "event_counts") for value in packed.event_counts)
    if len(counts) != len(packed) or sum(counts) != len(packed.events):
        raise ValueError("Packed event_counts do not cover the raw packed events")
    if packed.events.ndim != 2 or packed.events.shape[1] != 4 or packed.events.dtype != torch.float64:
        raise ValueError("Sampling requires validated float64 physical events [N,4]")
    if (packed.event_ids is None or packed.event_ids.dtype != torch.long
            or packed.event_ids.shape != (len(packed.events), 2)
            or packed.event_ids.device != packed.events.device):
        raise ValueError("Sampling requires raw same-device int64 event_ids [N,2]")

    updated_samples, selected_groups = [], []
    for sample, count, offset in zip(packed, counts, prior, strict=True):
        metadata = sample.get("metadata")
        timing = metadata.get("stream_time") if isinstance(metadata, dict) else None
        if not isinstance(timing, dict) or timing.get("schema") != "physical_seconds_v1":
            raise ValueError("Sampling requires validated physical_seconds_v1 metadata")
        groups = _groups(timing.get("arrival_group_counts"), "arrival_group_counts")
        if sum(groups) != count:
            raise ValueError("Raw arrival_group_counts must cover every raw event")
        current, selected = offset, []
        for size in groups:
            retained = _retained_count(size, current, factor)
            if retained:
                selected.append(retained)
            current += size
        selected_groups.append(tuple(selected))
        updated_samples.append({
            **sample, "metadata": {
                **metadata, "stream_time": {**timing, "arrival_group_counts": tuple(selected)},
            },
        })

    next_offsets = tuple(offset + count for offset, count in zip(prior, counts, strict=True))
    arrival_groups = tuple(selected_groups)
    if factor == 1:
        return StreamSamplingResult(
            packed, torch.ones(len(packed.events), device=packed.events.device, dtype=torch.bool),
            next_offsets, arrival_groups,
        )

    # All independent lanes share one tensor selection. Metadata loops above
    # contain no tensor work, device synchronization or graph/model execution.
    device = packed.events.device
    sizes = torch.tensor(counts, device=device, dtype=torch.long)
    lane = torch.repeat_interleave(torch.arange(len(packed), device=device), sizes)
    starts = sizes.cumsum(0) - sizes
    local_ordinal = torch.arange(len(packed.events), device=device) - starts[lane]
    phase = torch.tensor([(-offset) % factor for offset in prior], device=device, dtype=torch.long)
    keep = local_ordinal.remainder(factor) == phase[lane]
    selected_counts = tuple(_retained_count(count, offset, factor)
                            for offset, count in zip(prior, counts, strict=True))
    sampled = PackedSampleBatch(
        updated_samples, packed.events[keep], selected_counts, packed.targets, packed.event_ids[keep],
    )
    return StreamSamplingResult(sampled, keep, next_offsets, arrival_groups)


def replace_record_groups(records: Sequence[tuple], groups: Sequence[tuple[int, ...]]) -> list[tuple]:
    """Copy six-field ``stream_model._metadata`` records with sampled groups.

    Sequence identity/index and all three physical clocks are retained exactly.
    This helper does not mutate raw records or infer/revalidate their chronology.
    """
    if len(records) != len(groups):
        raise ValueError("One sampled arrival-group sequence is required per metadata record")
    result = []
    for record, counts in zip(records, groups, strict=True):
        if not isinstance(record, tuple) or len(record) != 6:
            raise ValueError("Expected six-field streaming metadata records")
        result.append((*record[:5], _groups(counts, "sampled arrival_group_counts")))
    return result
