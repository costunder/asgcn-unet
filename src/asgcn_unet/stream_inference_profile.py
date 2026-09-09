"""Full-prefix state bootstrap for streaming inference resource measurements.

Every callback invocation owns fresh graph/IF/decoder state. The parent profiler
times the whole callback, including CPU prefix loading and packed transfers. Its
rates therefore describe bootstrap-inclusive diagnostic work, NOT steady-state
inference throughput. No dataset, checkpoint, or GPU mask is modified.
"""

from __future__ import annotations

import copy
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import Subset

from .batching import PackedSampleBatch, SequenceBatchSampler, pack_samples, sequence_key
from .training import TrainingState


def _base_mapping(dataset):
    base = dataset
    selected = list(range(len(dataset)))
    while isinstance(base, Subset):
        indices = base.indices
        if any(type(index) is not int or index < 0 or index >= len(base.dataset) for index in indices):
            raise ValueError("Streaming profile Subset indices must be valid nonnegative integers")
        selected = [indices[index] for index in selected]
        base = base.dataset
    if len(selected) != len(set(selected)):
        raise ValueError("Streaming profile dataset contains duplicate source indices")
    records = getattr(base, "samples", None)
    if not isinstance(records, list) or len(records) != len(base):
        raise TypeError("Streaming profile requires the original indexed dataset.samples list")
    streams: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    identities = {}
    for index, record in enumerate(records):
        identity = sequence_key(record)
        frame = record.get("sequence_index")
        if type(frame) is not int or frame < 0:
            raise ValueError("Streaming profile requires chronological sequence_index metadata")
        key = (identity, frame)
        if key in identities:
            raise ValueError("Streaming source contains duplicate sequence/frame identities")
        current = streams.setdefault(identity, [])
        if current and frame != records[current[-1]]["sequence_index"] + 1:
            raise ValueError("Streaming source has an incomplete or unordered sequence prefix")
        current.append(index)
        identities[key] = index
    source_to_local = {source: local for local, source in enumerate(selected)}
    selected_sources = set(selected)
    for stream in streams.values():
        included_positions = [position for position, index in enumerate(stream) if index in selected_sources]
        if included_positions and not set(stream[:max(included_positions) + 1]).issubset(selected_sources):
            raise ValueError("Streaming profile Subset is missing required full-sequence prefix frames")
    shapes = SequenceBatchSampler(dataset, 1).sample_sensor_sizes
    lookup, prefixes = {}, {}
    for identity, stream in streams.items():
        for position, index in enumerate(stream):
            if index not in source_to_local:
                continue
            local = source_to_local[index]
            key = (identity, records[index]["sequence_index"])
            lookup[key] = local
            # Store one immutable stream list and a position, not O(F^2) prefixes.
            prefixes[local] = (stream, position)
    return base, source_to_local, lookup, prefixes, shapes


class _StreamProfileCallback:
    def __init__(self, dataset, device, run_forward: Callable):
        self._dataset = dataset
        self._device = torch.device(device)
        self._run_forward = run_forward
        self._base, self._source_to_local, self._lookup, self._prefixes, self._shapes = _base_mapping(dataset)
        self._history: list[dict[str, Any]] = []

    @property
    def last_report(self):
        return None if not self._history else copy.deepcopy(self._history[-1])

    @property
    def report(self):
        return {
            "schema": "asgcn_streaming_profile_bootstrap_v1", "report_eligible": False,
            "bootstrap_included_in_timing": True, "steady_state_throughput": False,
            "context_policy": "full_sequence_prefix_from_original_start_each_invocation",
            "state_reused_between_calls": False,
            "calls": len(self._history),
            "full_prefix_frames": sum(row["full_prefix_frames"] for row in self._history),
            "peak_live_states": max((row["peak_live_states"] for row in self._history), default=0),
            "target_frames": sum(row["target_frames"] for row in self._history),
            "state_residency": "conservative_includes_target_inputs_and_all_bootstrapped_target_stream_states",
            "limitation": "Full-prefix replay can be expensive; rates include bootstrap and are not steady-state FPS.",
            "invocations": copy.deepcopy(self._history),
        }

    def _target_index(self, sample):
        metadata = sample.get("metadata", {})
        if type(metadata.get("sequence_index")) is not int:
            raise ValueError("Streaming profile target requires an integer sequence_index")
        key = (sequence_key(sample), metadata.get("sequence_index"))
        if key not in self._lookup:
            raise ValueError("Streaming profile target is not in the validated source dataset")
        index = self._lookup[key]
        if tuple(sample["sensor_size"]) != self._shapes[index]:
            raise ValueError("Streaming profile target geometry differs from the indexed dataset")
        return index

    def _forward_commit(self, packed, state):
        contexts = state.prepare(packed)
        prediction, diagnostics = self._run_forward(packed, contexts, timing=None)
        if (not isinstance(prediction, torch.Tensor) or packed.targets is None
                or prediction.shape != packed.targets.shape or len(diagnostics) != len(packed)):
            raise RuntimeError("Streaming profile forward must produce every target reconstruction and diagnostic")
        if any(detail.get("recurrent_state") is None for detail in diagnostics):
            raise RuntimeError("Streaming profiling requires graph/IF state even when decoder recurrence is disabled")
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("Streaming profile reconstruction is nonfinite")
        state.commit(packed, prediction, diagnostics, packed.targets)
        return prediction, diagnostics

    def __call__(self, targets: PackedSampleBatch):
        targets = pack_samples(targets)
        # The configured logical device can be 'cuda' while tensor.device is
        # 'cuda:<current>'. Do not select or alter that allocated device.
        if (targets.events.device != self._device
                and not (self._device.type == "cuda" and self._device.index is None
                         and targets.events.device.type == "cuda")):
            raise ValueError("Streaming profile targets must already be on the measured device")
        if targets.targets is None:
            raise ValueError("Streaming reconstruction profiling requires target tensors")
        target_indices = [self._target_index(sample) for sample in targets]
        identities = [sequence_key(sample) for sample in targets]
        if len(identities) != len(set(identities)):
            raise ValueError("Streaming profile target batch contains duplicate/dependent sequence identities")
        queues = []
        for index in target_indices:
            stream, position = self._prefixes[index]
            queues.append(deque(self._source_to_local[source] for source in stream[:position]))
        # State is strictly invocation-local. Only scalar/count metadata survives
        # below, never an inference tensor or state from an earlier trial.
        state = TrainingState(independent_sequences=True)
        bootstrap_frames, bootstrap_batches, peak_live_states, shape_resets = 0, 0, 0, 0
        with torch.inference_mode():
            while any(queues):
                first = next(queue[0] for queue in queues if queue)
                shape = self._shapes[first]
                indices = [queue.popleft() for queue in queues if queue and self._shapes[queue[0]] == shape]
                raw = [self._dataset[index] for index in indices]
                if any(sample["events"].device.type != "cpu" for sample in raw):
                    raise ValueError("Streaming prefix data loading must remain on CPU before packed transfer")
                if [self._target_index(sample) for sample in raw] != indices:
                    raise ValueError("Streaming prefix data identity changed after index validation")
                for sample in raw:
                    previous = state.values.get(sequence_key(sample))
                    if previous is not None and tuple(sample["sensor_size"]) != previous[1]:
                        shape_resets += 1
                packed = pack_samples(raw).to(targets.events.device)
                prefix_prediction, prefix_diagnostics = self._forward_commit(packed, state)
                bootstrap_frames += len(packed)
                bootstrap_batches += 1
                peak_live_states = max(peak_live_states, len(state.values))
                del packed, raw, prefix_prediction, prefix_diagnostics
            for sample in targets:
                previous = state.values.get(sequence_key(sample))
                if previous is not None and tuple(sample["sensor_size"]) != previous[1]:
                    shape_resets += 1
            prediction, diagnostics = self._forward_commit(targets, state)
            peak_live_states = max(peak_live_states, len(state.values))
        record = {
            "bootstrap_included_in_timing": True, "steady_state_throughput": False,
            "full_prefix_frames": bootstrap_frames, "prefix_batches": bootstrap_batches,
            "target_frames": len(targets), "target_indices": list(target_indices),
            "peak_live_states": peak_live_states, "shape_change_state_resets": shape_resets,
            "state_reused_between_calls": False,
        }
        self._history.append(record)
        # Release bootstrap's cloned state before returning caller-owned results.
        state.values.clear()
        return prediction, diagnostics


def make_stream_profile_callback(dataset, device, run_forward) -> Callable:
    """Return a run_batch callback that replays full, shape-compatible prefixes.

    A Subset is accepted only when every selected stream includes its original
    prefix. Nonconsecutive profile targets themselves are supported: omitted
    predecessor targets are loaded and replayed from that validated dataset.
    ``callback.report`` contains metadata for the parent scheduling report.
    """
    if not callable(run_forward):
        raise TypeError("run_forward must be callable")
    return _StreamProfileCallback(dataset, device, run_forward)
