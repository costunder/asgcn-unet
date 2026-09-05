"""Shared causal state and loss semantics for training and its CUDA gate."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.nn import functional as F

from .batching import PackedSampleBatch, sequence_key


def batching_contract(batch_size: int) -> dict[str, Any] | None:
    if batch_size == 1:
        return None
    return {
        "version": "independent_sequences_v1",
        "graph": "disjoint_union_no_cross_sample_edges",
        "encoder_batch_norm": "pooled_nodes_in_minibatch",
        "decoder": "one_vectorized_call_per_minibatch",
        "state": "per_sequence_detached_after_success",
        "loss": "mean_per_frame_temporal_missing_context_zero",
        "partial_batch": "keep_every_frame_no_padding",
        "order": "deterministic_shape_compatible_sequence_lanes",
    }


class TrainingState:
    """Route state by sequence identity, not a batch lane that can be reassigned."""

    def __init__(self, *, independent_sequences: bool) -> None:
        self.independent_sequences = independent_sequences
        self.values: dict[tuple[str, str], tuple[Any, ...]] = {}
        self.last_key: tuple[str, str] | None = None

    def _key(self, sample: dict[str, Any]) -> tuple[str, str]:
        if self.independent_sequences:
            return sequence_key(sample)
        metadata = sample.get("metadata", {})
        return str(metadata.get("sequence_id") or metadata.get("scene", "unknown")), ""

    def prepare(self, samples: list[dict[str, Any]]) -> list[tuple[Any, Any, Any]]:
        contexts = []
        keys = [self._key(sample) for sample in samples]
        if len(keys) != len(set(keys)):
            raise ValueError("A training batch must not contain two frames of one sequence")
        for key, sample in zip(keys, samples, strict=True):
            metadata = sample.get("metadata", {})
            index = metadata.get("sequence_index")
            size = tuple(sample["sensor_size"])
            previous = self.values.get(key)
            if not self.independent_sequences and key != self.last_key:
                previous = None
            continues = (
                previous is not None
                and index is not None
                and previous[0] is not None
                and index == previous[0] + 1
                and size == previous[1]
            )
            contexts.append(previous[2:] if continues else (None, None, None))
        return contexts

    def commit(
        self,
        samples: list[dict[str, Any]],
        prediction: torch.Tensor,
        diagnostics: list[dict[str, Any]],
        target: torch.Tensor,
    ) -> None:
        if not self.independent_sequences:
            self.values.clear()
        def detached_context(tensor):
            if tensor is None:
                return None
            # A retained lane must not keep the entire source batch allocation alive.
            return tensor.detach().clone() if self.independent_sequences else tensor.detach()

        for index, (sample, detail) in enumerate(zip(samples, diagnostics, strict=True)):
            key = self._key(sample)
            state = detail["recurrent_state"]
            self.values[key] = (
                sample.get("metadata", {}).get("sequence_index"),
                tuple(sample["sensor_size"]),
                detached_context(state),
                detached_context(prediction[index : index + 1]),
                detached_context(target[index : index + 1]),
            )
            self.last_key = key

    def release_finished(self, samples, final_sequence_indices) -> None:
        """Finished lanes must not retain GPU frames throughout a whole epoch."""
        for sample in samples:
            key = self._key(sample)
            index = sample.get("metadata", {}).get("sequence_index")
            if key in final_sequence_indices and index == final_sequence_indices[key]:
                self.values.pop(key, None)


def forward_training_loss(
    model,
    criterion,
    samples: list[dict[str, Any]],
    contexts: list[tuple[Any, Any, Any]],
    *,
    batch_mode: bool,
    amp_enabled: bool,
    temporal_weight: float,
    timing=None,
):
    """One actual model call; failed AMP attempts never update the context store."""
    if not samples or len(contexts) != len(samples):
        raise ValueError("Training needs one incoming context for every frame")
    device = samples[0]["events"].device
    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        if batch_mode:
            prediction, diagnostics = model.forward_training_batch(
                samples, [context[0] for context in contexts], timing=timing
            )
        else:
            if len(samples) != 1:
                raise ValueError("The baseline path requires exactly one frame")
            with timing.scope("model") if timing is not None else nullcontext():
                prediction, detail = model.forward_sample(
                    samples[0], recurrent_state=contexts[0][0]
                )
            diagnostics = [detail]
        with timing.scope("loss") if timing is not None else nullcontext():
            target = (
                samples.targets if isinstance(samples, PackedSampleBatch)
                else torch.stack([sample["target"] for sample in samples])
            )
            loss, parts = criterion(prediction, target)
            valid = [i for i, context in enumerate(contexts) if context[1] is not None]
            if temporal_weight > 0 and valid:
                previous_prediction = torch.cat([contexts[i][1] for i in valid])
                previous_target = torch.cat([contexts[i][2] for i in valid])
                # The steady-state batch usually has context in every lane.
                # Reusing the whole tensors avoids two advanced-index gathers
                # (and CUDA index construction) without changing the reduction.
                current_prediction = prediction if len(valid) == len(samples) else prediction[valid]
                current_target = target if len(valid) == len(samples) else target[valid]
                temporal = F.l1_loss(
                    current_prediction - previous_prediction,
                    current_target - previous_target,
                ) * (len(valid) / len(samples))
                loss = loss + temporal_weight * temporal
                parts["temporal"] = temporal.detach()
    return loss, parts, (prediction, diagnostics, target)
