"""Measured inference scheduling diagnostics, never quality evaluation results."""

from __future__ import annotations

import gc
import math
import os
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import Any

import torch

from .allocation import require_gpu_allocation
from .batching import PackedSampleBatch, SequenceBatchSampler
from .resources import collect_runtime_resources, summarize_resource_interval


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'non-negative' if allow_zero else 'positive'} integer")
    return value


def _candidates(value: Any, name: str, *, allow_zero: bool = False) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must contain explicit measurement candidates")
    candidates = [_positive_int(item, name, allow_zero=allow_zero) for item in value]
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"{name} must not contain duplicates")
    return sorted(candidates)


def _probe_plan(
    topology: SequenceBatchSampler,
    probe_indices: list[int],
    batch_size: int,
    *,
    calibration: bool,
) -> list[list[int]]:
    """Schedule diagnostic samples without concurrent copies of a recurrent stream."""
    selected = set(probe_indices)
    streams = [
        deque(index for index in indices if index in selected)
        for indices in topology.sequence_indices
    ]
    shapes = topology.sample_sensor_sizes
    if calibration:
        buckets: OrderedDict[tuple[int, int], list[int]] = OrderedDict()
        for index in probe_indices:
            buckets.setdefault(shapes[index], []).append(index)
        return [
            indices[start : start + batch_size]
            for indices in buckets.values()
            for start in range(0, len(indices), batch_size)
        ]
    batches = []
    while any(streams):
        first = next(stream[0] for stream in streams if stream)
        shape = shapes[first]
        batch = []
        for stream in streams:
            if stream and shapes[stream[0]] == shape:
                batch.append(stream.popleft())
                if len(batch) == batch_size:
                    break
        batches.append(batch)
    return batches


def _resources(device: torch.device) -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    result: dict[str, Any] = {
        "device": str(device), "host_cpu_count": os.cpu_count(),
        "cpu_affinity": affinity, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cpu_threads": torch.get_num_threads(),
        "allocation": collect_runtime_resources(include_cuda=False),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        result.update(
            gpu_name=props.name, visible_gpu_count=torch.cuda.device_count(),
            gpu_total_memory_bytes=int(total), gpu_free_memory_bytes=int(free),
            mig="MIG" in props.name,
        )
    return result


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_trial(
    dataset: Any,
    device: torch.device,
    *,
    batches: list[list[int]],
    requested_batch_size: int,
    num_workers: int,
    warmup: int,
    steps: int,
    memory_fraction: float,
    run_batch: Callable[[PackedSampleBatch], Any],
    loader_factory: Callable[[Any, Any, int], Any],
) -> dict[str, Any]:
    # Cover every representative and mandatory dense frame, even if there are
    # more geometry/sequence batches than the requested minimum timing steps.
    measured_plan = [batches[index % len(batches)] for index in range(max(steps, len(batches)))]
    warmup_plan = [batches[index % len(batches)] for index in range(warmup)]
    trial: dict[str, Any] = {
        "requested_batch_size": requested_batch_size, "num_workers": num_workers,
        "warmup_batches": warmup, "requested_steps": steps,
        "measured_batches": len(measured_plan),
        "actual_batch_sizes": [len(batch) for batch in measured_plan],
        "probe_indices": sorted({index for batch in measured_plan for index in batch}),
        "samples": sum(map(len, measured_plan)),
        "timing_includes_io": True, "timing_includes_host_to_device": True,
        "worker_startup_included": False,
    }
    loader = iterator = cpu_batch = device_batch = result = None
    timings: list[dict[str, Any]] = []
    host_before = collect_runtime_resources(include_cuda=False)
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            _synchronize(device)
            free, total = torch.cuda.mem_get_info(device)
            baseline = torch.cuda.memory_reserved(device)
            trial["memory_budget_bytes"] = int(min(
                total * memory_fraction, baseline + free * memory_fraction
            ))
            torch.cuda.reset_peak_memory_stats(device)
        loader = loader_factory(dataset, warmup_plan + measured_plan, num_workers)
        iterator = iter(loader)
        with torch.inference_mode():
            for indices in warmup_plan:
                cpu_batch = next(iterator)
                if len(cpu_batch) != len(indices):
                    raise RuntimeError("Diagnostic DataLoader changed the requested batch size")
                device_batch = cpu_batch.to(device)
                result = run_batch(device_batch)
                _synchronize(device)
                result = device_batch = cpu_batch = None
            for indices in measured_plan:
                _synchronize(device)
                start = time.perf_counter()
                cpu_start = time.process_time()
                cpu_batch = next(iterator)
                if len(cpu_batch) != len(indices):
                    raise RuntimeError("Diagnostic DataLoader changed the requested batch size")
                loaded = time.perf_counter()
                device_batch = cpu_batch.to(device)
                _synchronize(device)
                transferred = time.perf_counter()
                result = run_batch(device_batch)
                _synchronize(device)
                finished = time.perf_counter()
                timings.append({
                    "samples": len(indices), "seconds": finished - start,
                    "data_wait_seconds": loaded - start,
                    "host_to_device_seconds": transferred - loaded,
                    "forward_seconds": finished - transferred,
                    "main_process_cpu_seconds": time.process_time() - cpu_start,
                    "events": sum(cpu_batch.event_counts),
                    "event_tensor_shape": list(cpu_batch.events.shape),
                    "target_tensor_shape": (
                        None if cpu_batch.targets is None else list(cpu_batch.targets.shape)
                    ),
                    "sensor_size": list(cpu_batch.sensor_size),
                })
                result = device_batch = cpu_batch = None
        seconds = sum(float(row["seconds"]) for row in timings)
        if seconds <= 0 or not math.isfinite(seconds):
            raise RuntimeError("Inference profiling produced an invalid elapsed time")
        trial.update(
            status="ok", seconds=seconds,
            samples_per_second=trial["samples"] / seconds, batch_timings=timings,
            events_per_second=sum(row["events"] for row in timings) / seconds,
        )
        if device.type == "cuda":
            trial.update(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                steady_allocated_bytes=torch.cuda.memory_allocated(device),
                steady_reserved_bytes=torch.cuda.memory_reserved(device),
            )
            if trial["peak_reserved_bytes"] > trial["memory_budget_bytes"]:
                trial["status"] = "memory_margin_exceeded"
        else:
            trial.update(peak_allocated_bytes=None, peak_reserved_bytes=None)
    except torch.cuda.OutOfMemoryError as error:
        if device.type != "cuda":
            raise
        trial.update(status="cuda_out_of_memory", error=str(error), batch_timings=timings)
    finally:
        # Release only this diagnostic's objects. Deleting our loader/iterator
        # lets DataLoader close its own workers normally; no process is signalled.
        del result, device_batch, cpu_batch, iterator, loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    host_after = collect_runtime_resources(include_cuda=False)
    trial["host_resources_before"] = host_before
    trial["host_resources_after"] = host_after
    trial["host_resource_interval"] = summarize_resource_interval(host_before, host_after)
    return trial


def profile_inference_batches(
    dataset: Any,
    model: Any,
    device: torch.device,
    *,
    section: dict[str, Any],
    run_batch: Callable[[PackedSampleBatch], Any],
    loader_factory: Callable[[Any, Any, int], Any],
    calibration: bool = False,
) -> dict[str, Any]:
    """Select physical batch/workers from actual, non-reporting diagnostic runs.

    ``run_batch`` receives a device-resident packed batch and must use the same
    inference mode, T and precision as the planned full run. Recurrent state is
    reset by the callback for these nonconsecutive diagnostic samples. Calibration
    callers must reset observed maxima after profiling and before full calibration.
    No model/data/sampling/T setting is changed here. CPU profiling is accepted
    only with an explicit ``profile_debug_cpu=True`` and is labelled unvalidated
    for CUDA. Actual representative topology coverage is not a worst-case proof.
    """
    device = torch.device(device)
    # Direct callers can bypass resolve_device: allocation must be checked
    # before even querying CUDA properties or constructing diagnostic tensors.
    gpu_allocation = require_gpu_allocation() if device.type == "cuda" else None
    debug_cpu = section.get("profile_debug_cpu", False)
    if not isinstance(debug_cpu, bool):
        raise TypeError("profile_debug_cpu must be a boolean")
    if device.type != "cuda" and not debug_cpu:
        raise RuntimeError(
            "Production automatic inference profiling requires the allocated CUDA device; "
            "CPU unit/smoke diagnostics require profile_debug_cpu=True"
        )
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Inference profiling supports allocated CUDA or explicit debug CPU")
    batches = _candidates(section.get("batch_candidates", [1, 2, 4, 8, 16]), "batch_candidates")
    workers = _candidates(
        section.get("worker_candidates", [0, 2, 4]), "worker_candidates", allow_zero=True
    )
    warmup = _positive_int(section.get("profile_warmup", 1), "profile_warmup", allow_zero=True)
    steps = _positive_int(section.get("profile_steps", 3), "profile_steps")
    fraction = section.get("profile_memory_fraction", 0.8)
    if isinstance(fraction, bool) or not isinstance(fraction, (float, int)) or not 0 < fraction <= 1:
        raise ValueError("profile_memory_fraction must lie in (0,1]")
    if not len(dataset):
        raise ValueError("Cannot profile an empty inference dataset")
    topology = SequenceBatchSampler(dataset, max(batches))
    representatives = {
        index
        for stream in topology.sequence_indices
        for index in (stream[0], stream[len(stream) // 2], stream[-1])
    }
    mandatory = section.get("batch_probe_indices", [])
    if not isinstance(mandatory, (list, tuple)):
        raise TypeError("batch_probe_indices must be a list of dataset indices")
    for index in mandatory:
        _positive_int(index, "batch_probe_indices", allow_zero=True)
        if index >= len(dataset):
            raise ValueError(f"Mandatory batch probe index {index} is outside this dataset")
    representatives.update(mandatory)
    selected_indices = sorted(representatives)
    resources = _resources(device)
    cores = resources["allocation"]["cpu"]["effective_cpu_limit"]
    admissible_workers = [value for value in workers if cores is None or value <= cores]
    if not admissible_workers:
        raise RuntimeError("No worker candidate fits the process CPU allocation")
    baseline_workers = max(admissible_workers)
    report: dict[str, Any] = {
        "schema": "asgcn_inference_batch_profile_v1", "report_eligible": False,
        "report_ineligible_reasons": ["resource scheduling diagnostic; not a quality evaluation"],
        "debug_cpu": device.type == "cpu", "cuda_measured": device.type == "cuda",
        "calibration": calibration, "dataset_size": len(dataset),
        "gpu_allocation": gpu_allocation,
        "sequence_count": topology.sequence_count, "resources": resources,
        "representative_policy": "each_sequence_first_middle_last_plus_explicit_dense_indices",
        "probe_indices": selected_indices, "mandatory_probe_indices": list(mandatory),
        "recurrent_state": "reset_for_nonconsecutive_diagnostic_samples",
        "state_residency": "callback retains per-sequence allocations as required by the full run",
        "batch_candidates": batches, "worker_candidates": workers,
        "skipped_workers_outside_cpu_allocation": sorted(set(workers) - set(admissible_workers)),
        "memory_fraction": fraction, "trials": [],
        "search_policy": "batch_sweep_at_largest_admissible_worker_then_worker_sweep",
        "limitations": [
            "Representative frames and explicit dense probes do not prove full-run peak memory.",
            "Two-phase search is measured locally; it does not exhaust every batch/worker pair.",
            "Main-process CPU timings exclude DataLoader worker CPU consumption.",
        ],
    }
    previous_training = model.training
    model.eval()
    try:
        plans = {
            size: _probe_plan(topology, selected_indices, size, calibration=calibration)
            for size in batches
        }
        for size in batches:
            trial = _run_trial(
                dataset, device, batches=plans[size], requested_batch_size=size,
                num_workers=baseline_workers, warmup=warmup, steps=steps,
                memory_fraction=float(fraction), run_batch=run_batch, loader_factory=loader_factory,
            )
            trial["phase"] = "batch_sweep"
            report["trials"].append(trial)
        successful = [trial for trial in report["trials"] if trial["status"] == "ok"]
        if not successful:
            details = [(trial["requested_batch_size"], trial["status"]) for trial in report["trials"]]
            raise RuntimeError(
                f"No measured inference batch candidate has the required memory margin: {details}"
            )
        best = max(successful, key=lambda trial: trial["samples_per_second"])
        selected_candidate = best["requested_batch_size"]
        for num_workers in admissible_workers:
            if num_workers == baseline_workers:
                continue
            trial = _run_trial(
                dataset, device, batches=plans[selected_candidate],
                requested_batch_size=selected_candidate, num_workers=num_workers,
                warmup=warmup, steps=steps, memory_fraction=float(fraction),
                run_batch=run_batch, loader_factory=loader_factory,
            )
            trial["phase"] = "worker_sweep"
            report["trials"].append(trial)
            if trial["status"] == "ok" and trial["samples_per_second"] > best["samples_per_second"]:
                best = trial
        # A nominal B16 with only seven independent same-shape streams is a
        # measured B7, not evidence for B16. Persist the actual selected size.
        selected_size = max(best["actual_batch_sizes"])
        report["selected"] = {
            "batch_size": selected_size, "requested_batch_candidate": selected_candidate,
            "num_workers": best["num_workers"], "samples_per_second": best["samples_per_second"],
            "concurrency_limited": selected_size < selected_candidate,
            "actual_batch_sizes": best["actual_batch_sizes"],
        }
        return {"batch_size": selected_size, "num_workers": best["num_workers"], "report": report}
    finally:
        model.train(previous_training)
