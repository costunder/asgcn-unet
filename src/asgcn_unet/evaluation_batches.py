"""Full-coverage, causal evaluation with independent sequence lanes.

Only artifact/metadata assembly is per frame. Transfers, graph construction,
encoder, decoder and image metrics operate on physical tensor batches.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from .batching import move_batch
from .metrics import batch_frame_metrics
from .timing import StageTimer
from .training import TrainingState


@torch.no_grad()
def evaluation_frames(
    loader,
    batch_plan,
    *,
    device,
    run_forward,
    independent_sequences: bool,
    final_sequence_indices=None,
    lpips_model=None,
    statistics: dict[str, Any],
    timing_steps: int = 50,
    timing_warmup: int = 10,
):
    """Yield original-indexed rows without dropping tails or resetting live lanes.

    Latency is batch completion latency, never divided by batch size. Measured
    throughput is reported separately from this per-frame waiting time.
    """
    state = TrainingState(independent_sequences=independent_sequences)
    timer = StageTimer(
        device, enabled=True, warmup_steps=timing_warmup,
        measurement_steps=timing_steps,
    )
    statistics.update({"batches": 0, "frames": 0, "model_seconds": 0.0,
                       "physical_batch_histogram": {}, "peak_live_sequences": 0,
                       "input_shapes": {}, "graph_statistics": {}})
    iterator = iter(loader)
    started = time.perf_counter()
    diagnostic_names = ("nodes", "edges", "isolated_nodes", "isolate_ratio", "max_degree")
    for indices in batch_plan:
        with timer.scope("dataload", gpu=False):
            try:
                cpu_batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError("Evaluation loader ended before its declared batch plan") from error
        if len(cpu_batch) != len(indices):
            raise RuntimeError("Evaluation collate changed the number of declared frames")
        with timer.scope("transfer"):
            samples = move_batch(cpu_batch, device)
        targets = samples.targets
        if targets is None:
            raise ValueError("Quality evaluation requires real target images")
        image_shape = "x".join(str(value) for value in targets.shape[1:])
        shape_counts = statistics["input_shapes"]
        shape_counts[image_shape] = shape_counts.get(image_shape, 0) + len(samples)
        contexts = state.prepare(samples)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_start = time.perf_counter()
        prediction, diagnostics = run_forward(samples, contexts, timer)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - model_start
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise FloatingPointError("Invalid evaluation batch latency")
        if prediction.shape != targets.shape or len(diagnostics) != len(samples):
            raise RuntimeError("Evaluation output does not match its physical batch")
        metric_prediction, metric_target = prediction.float(), targets.float()
        states = [detail["recurrent_state"] for detail in diagnostics
                  if detail["recurrent_state"] is not None]
        finite = [torch.isfinite(prediction).all(), torch.isfinite(targets).all()]
        if states:
            finite.append(torch.isfinite(torch.cat(states)).all())
        if not bool(torch.stack(finite).all()):
            raise FloatingPointError(f"Nonfinite evaluation tensors in dataset indices {indices}")
        with timer.scope("loss"):
            valid = [i for i, context in enumerate(contexts)
                     if context[1] is not None and context[2] is not None]
            temporal = prediction.new_zeros(len(samples), dtype=torch.float32)
            if valid:
                previous_prediction = torch.cat([contexts[i][1] for i in valid])
                previous_target = torch.cat([contexts[i][2] for i in valid])
                changes = ((metric_prediction[valid] - previous_prediction)
                           - (metric_target[valid] - previous_target))
                temporal[valid] = changes.abs().flatten(1).mean(1)
            metric_rows = batch_frame_metrics(
                metric_prediction, metric_target, lpips_model,
                extra_metrics={"temporal_l1": temporal},
            )
            # Graph statistics remain on device until this single packed transfer.
            diagnostic_values = torch.stack([
                torch.stack([torch.as_tensor(detail[name], device=device, dtype=torch.float64)
                             .reshape(()) for name in diagnostic_names])
                for detail in diagnostics
            ]).cpu().tolist()
        state.commit(samples, metric_prediction, diagnostics, metric_target)
        statistics["peak_live_sequences"] = max(statistics["peak_live_sequences"], len(state.values))
        if final_sequence_indices is not None:
            state.release_finished(samples, final_sequence_indices)
        statistics["batches"] += 1
        statistics["frames"] += len(samples)
        statistics["model_seconds"] += elapsed
        histogram = statistics["physical_batch_histogram"]
        histogram[str(len(samples))] = histogram.get(str(len(samples)), 0) + 1
        valid_set = set(valid)
        for lane, index in enumerate(indices):
            metrics = metric_rows[lane]
            if lane not in valid_set:
                metrics.pop("temporal_l1")
            detail = dict(diagnostics[lane])
            detail.update(zip(diagnostic_names, diagnostic_values[lane], strict=True))
            for name in ("nodes", "edges", "isolated_nodes", "max_degree"):
                detail[name] = int(detail[name])
            for name in ("nodes", "edges"):
                value = detail[name]
                aggregate = statistics["graph_statistics"].setdefault(
                    name, {"count": 0, "total": 0, "min": value, "max": value},
                )
                aggregate["count"] += 1
                aggregate["total"] += value
                aggregate["min"] = min(aggregate["min"], value)
                aggregate["max"] = max(aggregate["max"], value)
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"Nonfinite evaluation metric at dataset index {index}")
            yield (index, samples[lane], prediction[lane:lane + 1],
                   metric_target[lane:lane + 1], metrics, detail, elapsed * 1000.0)
        timer.step()
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("Evaluation loader produced undeclared extra frames")
    statistics["end_to_end_seconds"] = time.perf_counter() - started
    for aggregate in statistics["graph_statistics"].values():
        aggregate["mean"] = aggregate["total"] / aggregate["count"]
    statistics["throughput_frames_per_second"] = (
        statistics["frames"] / statistics["model_seconds"] if statistics["model_seconds"] else None
    )
    statistics["end_to_end_frames_per_second"] = (
        statistics["frames"] / statistics["end_to_end_seconds"]
        if statistics["end_to_end_seconds"] else None
    )
    statistics["latency_scope"] = "physical_batch_completion_not_amortized"
    statistics["throughput_scope"] = "model_graph_encoder_decoder_excludes_io_metrics_artifacts"
    statistics["end_to_end_scope"] = "loader_transfer_model_metrics_prediction_artifacts"
    statistics["worker_startup_included"] = False
    statistics["profiling_included"] = False
    statistics["model_loading_included"] = False
    statistics["summary_serialization_included"] = False
    statistics["timing"] = timer.collect()
