"""Physical-event geometry shared by the model and graph inspection.

The sensor-normalized metric is this project's explicit adaptation, not an
assertion about the ASGCN authors' undocumented coordinate units. This module
does not select events, infer clock units, choose a radius, or reset an origin.
"""

from __future__ import annotations

import math

import torch


def physical_node_positions(events, sensor_size, *, origin_seconds, time_scale_seconds):
    """Map every supplied event to fixed float64 graph coordinates, without a cap.

    ``origin_seconds`` may be one fixed scalar or one origin per packed node.
    Polarity is retained as coordinate four; the configured radius builder alone
    chooses whether that coordinate participates in the distance.
    """
    if (isinstance(time_scale_seconds, bool) or not isinstance(time_scale_seconds, (int, float))
            or not math.isfinite(time_scale_seconds) or time_scale_seconds <= 0):
        raise ValueError("time_scale_seconds must be explicitly finite and positive")
    if (len(sensor_size) != 2
            or any(type(size) is not int or size < 1 for size in sensor_size)):
        raise ValueError("sensor_size must contain positive integer height and width")
    if (not isinstance(events, torch.Tensor) or events.layout != torch.strided
            or events.ndim != 2 or events.shape[1] != 4 or events.dtype != torch.float64):
        raise ValueError("Physical graph events must be float64 [N,4]; no implicit time conversion")
    height, width = sensor_size
    origin = torch.as_tensor(origin_seconds, dtype=torch.float64, device=events.device)
    if origin.ndim != 0 and origin.shape != (len(events),):
        raise ValueError("Use one fixed origin or one origin per packed node")
    if not bool(torch.stack((
        torch.isfinite(events).all(), torch.isfinite(origin).all(),
        ((events[:, 0] >= 0) & (events[:, 0] < width)).all(),
        ((events[:, 1] >= 0) & (events[:, 1] < height)).all(),
        ((events[:, 3] == -1) | (events[:, 3] == 1)).all(),
        (events[:, 2] >= origin).all(),
    )).all()):
        raise ValueError("Invalid physical sensor event values or sequence origin")
    positions = torch.stack((
        events[:, 0] / max(width - 1, 1),
        events[:, 1] / max(height - 1, 1),
        (events[:, 2] - origin) / time_scale_seconds,
        (events[:, 3] + 1) / 2,
    ), dim=1)
    if not bool(torch.isfinite(positions).all()):
        raise ValueError("Normalized graph coordinates overflowed")
    return positions


def prepare_stream_nodes(packed, records, *, time_scale_seconds):
    """Validate and prepare all packed raw nodes used by the actual model.

    Metadata records contain identity, sequence index, start, end, fixed origin,
    and equal-timestamp group counts. Preparation precedes any explicit ordinal
    sampling, so sampling cannot conceal malformed raw input.
    """
    device, events = packed.events.device, packed.events
    if len(records) != len(packed):
        raise ValueError("One physical clock record is required per packed stream")
    counts = torch.tensor(packed.event_counts, device=device, dtype=torch.long)
    node_batch = torch.repeat_interleave(torch.arange(len(packed), device=device), counts,
                                         output_size=len(events))
    groups = []
    for record, count in zip(records, packed.event_counts, strict=True):
        if (len(record) != 6 or not isinstance(record[5], (tuple, list))
                or any(type(size) is not int or size < 1 for size in record[5])
                or sum(record[5]) != count):
            raise ValueError("Physical arrival grouping must cover every input event")
        if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in record[2:5])
                or not record[4] <= record[2] <= record[3]):
            raise ValueError("Invalid physical interval or fixed sequence origin")
        groups.extend(record[5])
    ids = packed.event_ids
    if (ids is None or ids.shape != (len(events), 2) or ids.dtype != torch.long
            or ids.device != device):
        raise ValueError("Physical nodes require same-device int64 event_ids [N,2]")
    origin = torch.tensor([record[4] for record in records], device=device, dtype=torch.float64)[node_batch]
    positions = physical_node_positions(events, packed.sensor_size, origin_seconds=origin,
                                        time_scale_seconds=time_scale_seconds)
    # Expand group labels once on the device, not as a Python integer per event.
    group_ids = torch.repeat_interleave(
        torch.arange(len(groups), device=device), torch.tensor(groups, device=device, dtype=torch.long),
        output_size=len(events),
    )
    if len(events) > 1:
        same_stream = node_batch[1:] == node_batch[:-1]
        delta = events[1:, 2] - events[:-1, 2]
        same_group = group_ids[1:] == group_ids[:-1]
        increasing_id = (ids[1:, 0] > ids[:-1, 0]) | (
            (ids[1:, 0] == ids[:-1, 0]) & (ids[1:, 1] > ids[:-1, 1]))
        if not bool(((~same_stream) | ((delta >= 0) & (same_group == (delta == 0)) & increasing_id)).all()):
            raise ValueError("Physical event order, identity or equal-timestamp grouping is invalid")
    interval_start = events.new_tensor([record[2] for record in records])[node_batch]
    interval_end = events.new_tensor([record[3] for record in records])[node_batch]
    if not bool((events[:, 2] <= interval_end).all()):
        raise ValueError("Physical event timestamp is after its readout (future leakage)")
    features = torch.stack((positions[:, 0], positions[:, 1],
                            (events[:, 2] - interval_start) / time_scale_seconds, events[:, 3]), dim=1).float()
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Physical node features overflowed float32; no graph was constructed")
    return features, positions, events[:, 2], node_batch
