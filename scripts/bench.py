"""Isolated real-EventHDR training throughput comparisons, never training resume.

Every trial runs in a fresh process. Timed frame identities and per-stream warmup
lengths are shared across batch sizes; epoch throughput/quality is not inferred.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# A detached benchmark checkout must not import another checkout's editable install.
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import torch
from torch.utils.data import Subset

from asgcn_unet.batching import SequenceBatchSampler, sequence_key
from asgcn_unet.data import build_dataset
from asgcn_unet.engine import (
    _build_optimizer,
    _current_source_contract,
    _dataset_content_fingerprint,
    _dataset_sample_identity,
    _make_grad_scaler,
    _optimizer_mode,
    _public_config,
    _training_protocol,
    _training_step,
)
from asgcn_unet.losses import ReconstructionLoss
from asgcn_unet.model import ASGCNUNet
from asgcn_unet.preflight import _runtime_provenance
from asgcn_unet.timing import StageTimer
from asgcn_unet.training import TrainingState, forward_training_loss
from asgcn_unet.utils import (
    load_json,
    move_sample,
    resolve_experiment_paths,
    save_json,
    set_seed,
    validate_experiment_config,
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def redact(text: str, config: dict, output: Path) -> str:
    roots = [PROJECT, output, Path.home()]
    roots.extend(
        Path(value).expanduser().resolve()
        for section in ("dataset", "output", "train")
        for key, value in config.get(section, {}).items()
        if key in {"root", "val_root", "split_manifest", "file_manifest", "run_dir", "resume"}
        and isinstance(value, str)
    )
    for root in sorted(roots, key=lambda p: len(str(p)), reverse=True):
        for spelling in {str(root), root.as_posix()}:
            text = text.replace(spelling, "$PATH")
    text = re.sub(r"(?i)(?<![A-Za-z0-9_])/(?:home|Users)/[^/\s\"']+", "$HOME", text)
    text = re.sub(r"(?i)\b[A-Z]:[\\/]Users[\\/][^\\/\s\"']+", "$HOME", text)
    for machine_name in {socket.gethostname(), os.getenv("HOSTNAME", ""), os.getenv("COMPUTERNAME", "")}:
        if machine_name:
            text = re.sub(re.escape(machine_name), "$HOST", text, flags=re.IGNORECASE)
    return text


def _positive(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def select_windows(dataset, *, streams: int, warmup: int, frames: int, seed: int) -> dict:
    """Choose metadata-only, contiguous equal-length windows of one sensor shape."""
    for name, value in (("streams", streams), ("warmup", warmup), ("frames", frames)):
        _positive(value, name)
    if frames < 2:
        raise ValueError("At least two measured frames per stream are required")
    geometry = SequenceBatchSampler(dataset, 1, seed=seed).sample_sensor_sizes
    grouped = defaultdict(list)
    for index, record in enumerate(dataset.samples):
        grouped[sequence_key(record)].append(index)
    required = warmup + frames
    candidates = defaultdict(dict)
    rng = random.Random(seed)
    for key, indices in sorted(grouped.items()):
        segments: list[list[int]] = []
        for index in indices:
            if not segments or (
                dataset.samples[index]["sequence_index"]
                != dataset.samples[segments[-1][-1]]["sequence_index"] + 1
                or geometry[index] != geometry[segments[-1][-1]]
            ):
                segments.append([])
            segments[-1].append(index)
        for shape in sorted({geometry[segment[0]] for segment in segments}):
            eligible = [s for s in segments if geometry[s[0]] == shape and len(s) >= required]
            if eligible:
                segment = rng.choice(eligible)
                start = rng.randrange(len(segment) - required + 1)
                candidates[shape][key] = segment[start : start + required]
    shapes = [shape for shape, groups in candidates.items() if len(groups) >= streams]
    if not shapes:
        raise ValueError(
            f"Need {streams} independent same-shape streams with {required} contiguous frames each; "
            "the selected real dataset cannot form this comparison"
        )
    shape = min(shapes, key=lambda s: (-len(candidates[s]), s))
    keys = sorted(candidates[shape])
    rng.shuffle(keys)
    windows = [candidates[shape][key] for key in sorted(keys[:streams])]
    entries = []
    for stream, window in enumerate(windows):
        for offset, index in enumerate(window):
            entries.append(
                {
                    **_dataset_sample_identity(dataset, index),
                    "stream": stream,
                    "phase": "warmup" if offset < warmup else "measure",
                    "window_offset": offset,
                }
            )
    selection = {
        "schema": "asgcn_real_training_windows_v1",
        "selection_policy": "seeded_contiguous_equal_length_independent_same_geometry",
        "seed": seed,
        "streams": streams,
        "sensor_size": list(shape),
        "warmup_frames_per_stream": warmup,
        "measured_frames_per_stream": frames,
        "warmup_frames": streams * warmup,
        "measured_frames": streams * frames,
        "entries": entries,
    }
    selection["sha256"] = digest(selection)
    return selection


def validate_selection(dataset, selection: dict) -> None:
    original = {key: value for key, value in selection.items() if key != "sha256"}
    if selection.get("sha256") != digest(original):
        raise ValueError("Benchmark selection hash is invalid")
    entries = selection["entries"]
    seen = set()
    for entry in entries:
        index = entry["dataset_index"]
        if index in seen or not 0 <= index < len(dataset):
            raise ValueError("Benchmark selection has a duplicate/out-of-range frame")
        seen.add(index)
        identity = _dataset_sample_identity(dataset, index)
        if any(entry.get(key) != value for key, value in identity.items()):
            raise ValueError("Benchmark frame identity differs from the selected real dataset")
    regenerated = select_windows(
        dataset,
        streams=selection["streams"],
        warmup=selection["warmup_frames_per_stream"],
        frames=selection["measured_frames_per_stream"],
        seed=selection["seed"],
    )
    if regenerated != selection:
        raise ValueError("Benchmark selection differs from its deterministic metadata plan")


def batch_schedule(dataset, selection: dict, size: int):
    _positive(size, "batch size")
    if selection["streams"] % size:
        raise ValueError("Every batch size must divide the common stream count")
    subset = Subset(dataset, [entry["dataset_index"] for entry in selection["entries"]])
    sampler = SequenceBatchSampler(subset, size, seed=selection["seed"])
    schedule = list(sampler)
    if not schedule or any(len(batch) != size for batch in schedule):
        raise ValueError("The selected comparison must form genuine full-size batches")
    phases = []
    for batch in schedule:
        labels = {selection["entries"][index]["phase"] for index in batch}
        if len(labels) != 1:
            raise ValueError("Warmup and measurement frames must never mix inside a batch")
        phases.append(labels.pop())
    if sorted(i for batch in schedule for i in batch) != list(range(len(subset))):
        raise ValueError("The batch schedule dropped or repeated selected frames")
    return subset, sampler, schedule, phases


def variant_config(config: dict, size: int, backend: str, chunk: int | None) -> dict:
    variant = copy.deepcopy(config)
    variant["device"] = "cuda"
    variant["train"].update(
        batch_size=size, batching="single_frame" if size == 1 else "independent_sequences"
    )
    variant["model"]["spline_backend"] = backend
    if chunk is not None:
        variant["model"]["spline_chunk_size"] = chunk
    validate_experiment_config(variant)
    return variant


def _source_contract() -> dict:
    return {
        **_current_source_contract(),
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _file_signatures(dataset, selection: dict) -> list[dict]:
    paths = sorted(
        {Path(dataset.samples[entry["dataset_index"]]["path"]) for entry in selection["entries"]}
    )
    result = []
    for path in paths:
        stat = path.stat()
        result.append(
            {
                "file": path.relative_to(dataset.root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        )
    return result


def _fresh(config: dict, device: torch.device):
    set_seed(int(config.get("seed", 2026)))
    model = ASGCNUNet(**config["model"]).to(device).train()
    optimizer = _build_optimizer(model, config["train"])
    amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    return (
        model,
        optimizer,
        _make_grad_scaler(amp),
        ReconstructionLoss(config["train"].get("loss_weights")),
        amp,
    )


def _step(model, optimizer, scaler, criterion, samples, contexts, config, amp, *, step, timer=None):
    def forward_loss():
        return forward_training_loss(
            model,
            criterion,
            samples,
            contexts,
            batch_mode=int(config["train"]["batch_size"]) > 1,
            amp_enabled=amp,
            temporal_weight=float((config["train"].get("loss_weights") or {}).get("temporal", 0)),
            timing=timer,
        )

    return _training_step(
        model,
        optimizer,
        scaler,
        forward_loss,
        optimizer_mode=_optimizer_mode(config["train"]),
        max_norm=float(config["train"].get("grad_clip", 1)),
        epoch=0,
        step=step,
        sample_id=[sample["sample_id"] for sample in samples],
        timing=timer,
    )


def _numeric_snapshot(config: dict, dataset, indices: list[int], device: torch.device) -> dict:
    model, optimizer, scaler, criterion, amp = _fresh(config, device)
    samples = [move_sample(dataset[index], device) for index in indices]
    payload, loss, norm, scale = _step(
        model,
        optimizer,
        scaler,
        criterion,
        samples,
        [(None, None, None)] * len(samples),
        config,
        amp,
        step=0,
    )
    prediction, diagnostics, _target = payload
    tensors = {"prediction": prediction.detach().float().cpu()}
    tensors.update(
        (
            "gradient/" + name,
            None if parameter.grad is None else parameter.grad.detach().float().cpu(),
        )
        for name, parameter in model.named_parameters()
    )
    tensors.update(
        ("buffer/" + name, value.detach().float().cpu())
        for name, value in model.named_buffers()
        if "running_" in name
    )
    return {
        "tensors": tensors,
        "loss": loss,
        "gradient_norm": norm,
        "amp": scale,
        "topology": [(int(item["nodes"]), int(item["edges"])) for item in diagnostics],
    }


def compare_snapshots(reference: dict, candidate: dict, *, amp: bool) -> dict:
    atol, rtol = (2e-3, 2e-2) if amp else (2e-5, 2e-4)
    l2_atol, l2_rtol = (5e-6, 0.05) if amp else (1e-7, 2e-4)
    if reference["topology"] != candidate["topology"]:
        raise ValueError("Backend parity failed: graph nodes/edges differ")
    if set(reference["tensors"]) != set(candidate["tensors"]):
        raise ValueError("Backend parity failed: tensor inventories differ")
    if not math.isclose(
        reference["gradient_norm"], candidate["gradient_norm"], abs_tol=atol, rel_tol=rtol
    ):
        raise ValueError("Backend parity failed: pre-clipping gradient norm differs")
    maximum = 0.0
    compared = 0
    gradient_error_sq = gradient_reference_sq = 0.0
    maximum_gradient_relative_l2 = 0.0
    for name, expected in reference["tensors"].items():
        actual = candidate["tensors"][name]
        if expected is None or actual is None:
            if expected is not actual:
                raise ValueError(f"Backend parity failed: missing gradient {name}")
            continue
        if expected.shape != actual.shape or not torch.allclose(
            expected, actual, atol=atol, rtol=rtol
        ):
            raise ValueError(f"Backend parity failed: {name} exceeds atol={atol}, rtol={rtol}")
        if expected.numel():
            maximum = max(maximum, float((expected - actual).abs().max()))
        if name.startswith("gradient/"):
            error_l2 = float(torch.linalg.vector_norm((expected - actual).double()))
            reference_l2 = float(torch.linalg.vector_norm(expected.double()))
            if error_l2 > l2_atol + l2_rtol * reference_l2:
                raise ValueError(f"Backend parity failed: {name} gradient L2 direction differs")
            gradient_error_sq += error_l2**2
            gradient_reference_sq += reference_l2**2
            maximum_gradient_relative_l2 = max(
                maximum_gradient_relative_l2, error_l2 / max(reference_l2, l2_atol)
            )
        compared += 1
    gradient_error = math.sqrt(gradient_error_sq)
    gradient_reference = math.sqrt(gradient_reference_sq)
    if gradient_error > l2_atol + l2_rtol * gradient_reference:
        raise ValueError("Backend parity failed: whole-gradient L2 direction differs")
    for name in reference["loss"]:
        if not math.isclose(
            reference["loss"][name],
            candidate["loss"].get(name, float("nan")),
            abs_tol=atol,
            rel_tol=rtol,
        ):
            raise ValueError(f"Backend parity failed: loss {name}")
    return {
        "passed": True,
        "atol": atol,
        "rtol": rtol,
        "compared_tensors": compared,
        "max_absolute_difference": maximum,
        "gradient_comparison": "clipped_direction_per_parameter_and_global_l2_plus_preclip_norm",
        "gradient_l2_atol": l2_atol,
        "gradient_l2_rtol": l2_rtol,
        "gradient_l2_error": gradient_error,
        "reference_gradient_l2": gradient_reference,
        "max_parameter_gradient_relative_l2": maximum_gradient_relative_l2,
    }


def numerical_gate(config: dict, reference_config: dict, subset, schedule, device) -> dict:
    counts = [
        sum(
            int(subset.dataset.samples[subset.indices[i]]["end_idx"])
            - int(subset.dataset.samples[subset.indices[i]]["start_idx"])
            for i in batch
        )
        for batch in schedule
    ]
    choices: dict[int, list[str]] = {0: ["first"]}
    for index, reason in (
        (min(range(len(counts)), key=counts.__getitem__), "sparse_raw_events"),
        (max(range(len(counts)), key=counts.__getitem__), "dense_raw_events"),
    ):
        choices.setdefault(index, []).append(reason)
    measurements = []
    for index, reasons in choices.items():
        reference = _numeric_snapshot(reference_config, subset, schedule[index], device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        candidate = _numeric_snapshot(config, subset, schedule[index], device)
        comparison = compare_snapshots(
            reference,
            candidate,
            amp=bool(config["train"].get("amp", True)) and device.type == "cuda",
        )
        measurements.append(
            {
                "batch_index": index,
                "dataset_indices": [subset.indices[i] for i in schedule[index]],
                "reasons": reasons,
                "reference_amp": reference["amp"],
                "candidate_amp": candidate["amp"],
                "topology": [
                    {"nodes": nodes, "edges": edges} for nodes, edges in candidate["topology"]
                ],
                **comparison,
            }
        )
    if not any(item["edges"] > 0 for row in measurements for item in row["topology"]):
        raise ValueError(
            "No graph edges in the selected numerical cases; the spline backend was not exercised"
        )
    return {
        "passed": True,
        "nonempty_spline_exercised": True,
        "reference_backend": "torch",
        "state": "fresh_seed_no_context_same_batch",
        "density_basis": "raw_event_interval_not_full_graph_scan",
        "measurements": measurements,
    }


def export_trace(profiler, path: Path, config: dict) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("Refusing to overwrite an existing operator trace")
    with tempfile.TemporaryDirectory(prefix="asgcn-bench-trace-") as temporary:
        raw_path = Path(temporary) / "trace.json"
        raw_path.touch(mode=0o600, exist_ok=False)
        profiler.export_chrome_trace(str(raw_path))
        trace = json.loads(raw_path.read_text(encoding="utf-8"))

        def sanitized(value):
            if isinstance(value, str):
                return redact(value, config, path.parent)
            if isinstance(value, list):
                return [sanitized(item) for item in value]
            if isinstance(value, dict):
                return {sanitized(key): sanitized(item) for key, item in value.items()}
            return value

        save_json(path, sanitized(trace))


def exercise(config, subset, sampler, schedule, phases, device, *, trace_path=None, trace_steps=0):
    active_profilers = []
    try:
        return _exercise_impl(
            config,
            subset,
            sampler,
            schedule,
            phases,
            device,
            trace_path=trace_path,
            trace_steps=trace_steps,
            active_profilers=active_profilers,
        )
    finally:
        pending_error = sys.exc_info()[0] is not None
        for profiler in active_profilers:
            try:
                profiler.__exit__(None, None, None)
            except Exception:
                if not pending_error:
                    raise


def _exercise_impl(
    config: dict,
    subset,
    sampler,
    schedule,
    phases,
    device,
    *,
    trace_path=None,
    trace_steps=0,
    active_profilers=None,
) -> dict:
    """Run every selected frame; synchronize only at timed segment boundaries."""
    model, optimizer, scaler, criterion, amp = _fresh(config, device)
    size = int(config["train"]["batch_size"])
    state = TrainingState(independent_sequences=size > 1)
    cuda = device.type == "cuda"
    measured_frames = warmup_frames = 0
    losses, norms, retries = [], [], []
    decode_ms = elapsed_ms = cuda_ms = 0.0
    allocated = reserved = 0.0
    segments = 0
    active = False
    start_wall = 0.0
    start_event = end_event = None
    timer = StageTimer(
        device,
        enabled=trace_path is not None,
        warmup_steps=0,
        measurement_steps=max(1, trace_steps),
    )
    profiler = None
    trace_completed = 0
    if trace_path is not None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities, record_shapes=True, profile_memory=True, with_stack=False
        )

    def finish_segment():
        nonlocal active, elapsed_ms, cuda_ms, allocated, reserved
        if not active:
            return
        if cuda:
            end_event.record()
            torch.cuda.synchronize(device)
            cuda_ms += float(start_event.elapsed_time(end_event))
            allocated = max(allocated, torch.cuda.max_memory_allocated(device) / 1024**2)
            reserved = max(reserved, torch.cuda.max_memory_reserved(device) / 1024**2)
        elapsed_ms += (time.perf_counter() - start_wall) * 1000
        active = False

    for step, (batch, phase) in enumerate(zip(schedule, phases, strict=True)):
        measured = phase == "measure"
        if active and not measured:
            finish_segment()
        if measured and not active:
            if cuda:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                start_event, end_event = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start_event.record()
            start_wall = time.perf_counter()
            active = True
            segments += 1
        tracing = measured and profiler is not None and trace_completed < trace_steps
        if tracing and trace_completed == 0:
            profiler.__enter__()
            active_profilers.append(profiler)
        recorder = timer if tracing else None
        decode_start = time.perf_counter()
        with recorder.scope("dataload", gpu=False) if recorder is not None else nullcontext():
            raw = [subset[index] for index in batch]
        if measured:
            decode_ms += (time.perf_counter() - decode_start) * 1000
        with recorder.scope("transfer") if recorder is not None else nullcontext():
            samples = [move_sample(sample, device) for sample in raw]
        contexts = state.prepare(samples)
        payload, loss, norm, scale = _step(
            model,
            optimizer,
            scaler,
            criterion,
            samples,
            contexts,
            config,
            amp,
            step=step,
            timer=recorder,
        )
        prediction, diagnostics, target = payload
        state.commit(samples, prediction, diagnostics, target)
        if size > 1:
            state.release_finished(samples, sampler.final_sequence_indices)
            if len(state.values) > size:
                raise RuntimeError("Benchmark state storage exceeded active sequence lanes")
        if measured:
            measured_frames += len(samples)
            losses.extend([loss["total"]] * len(samples))
            norms.append(norm)
            retries.append(int(scale["retries"]))
        else:
            warmup_frames += len(samples)
        if tracing:
            profiler.step()
            trace_completed += 1
            timer.step()
            if trace_completed == trace_steps:
                active_profilers.pop()
                profiler.__exit__(None, None, None)
        # Previous batch autograd outputs must not inflate the next batch's peak.
        del payload, prediction, diagnostics, target, samples, contexts, raw
    finish_segment()
    if profiler is not None:
        if trace_completed < trace_steps:
            if trace_completed:
                active_profilers.pop()
                profiler.__exit__(None, None, None)
            raise ValueError("Not enough measured batches for the requested operator trace")
        export_trace(profiler, trace_path, config)
    if measured_frames == 0 or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise RuntimeError("Benchmark did not complete a valid measured window")
    return {
        "status": "passed",
        "cuda_measured": cuda,
        "frames_per_second": measured_frames * 1000 / elapsed_ms,
        "measured_frames": measured_frames,
        "warmup_frames": warmup_frames,
        "optimizer_steps_measured": len(norms),
        "timed_segments": segments,
        "wall_ms_including_decode_transfer_step": elapsed_ms,
        "host_decode_ms": decode_ms,
        "cuda_stream_elapsed_ms": cuda_ms if cuda else None,
        "peak_allocated_mib": allocated if cuda else None,
        "peak_reserved_mib": reserved if cuda else None,
        "loss_mean_per_frame": statistics.fmean(losses),
        "gradient_norm_min": min(norms),
        "gradient_norm_max": max(norms),
        "amp_retries": sum(retries),
        "amp_retried_steps": sum(n > 0 for n in retries),
        "amp_scale_final": float(scaler.get_scale()),
        "operator_trace_only": profiler is not None,
        "stages": timer.collect() if profiler is not None else None,
    }


def worker(job: dict) -> int:
    output = Path(job["output"]).expanduser().resolve()
    config = job["config"]
    protect_output(output.parent, config)
    if output.exists() or output.is_symlink():
        raise FileExistsError("The benchmark worker refuses an existing trial output")
    dataset = None
    report = {
        "schema": "asgcn_training_performance_trial_v1",
        "status": "failed",
        "variant": job["variant"],
        "selection_sha256": job["selection"]["sha256"],
        "source": _source_contract(),
        "config_sha256": digest(_public_config(config)),
    }
    try:
        torch.set_num_threads(job["threads"])
        device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "This real-data training benchmark requires CUDA; CPU fallback is disabled"
            )
        torch.cuda.init()
        report["runtime"] = _runtime_provenance(device)
        if report["source"] != job["source"]:
            raise ValueError("Source changed after the comparison plan was created")
        dataset = build_dataset(config["dataset"], split="train")
        validate_selection(dataset, job["selection"])
        if _file_signatures(dataset, job["selection"]) != job["file_signatures"]:
            raise ValueError("Selected source files changed before this benchmark trial")
        subset, sampler, schedule, phases = batch_schedule(
            dataset, job["selection"], config["train"]["batch_size"]
        )
        report["schedule_sha256"] = digest(schedule)
        report["training_protocol"] = _training_protocol(config, device)
        report["numerical_gate"] = numerical_gate(
            config, job["reference_config"], subset, schedule, device
        )
        torch.cuda.empty_cache()
        report["measurement"] = exercise(config, subset, sampler, schedule, phases, device)
        if (
            report["measurement"]["measured_frames"] != job["selection"]["measured_frames"]
            or report["measurement"]["warmup_frames"] != job["selection"]["warmup_frames"]
        ):
            raise ValueError("Trial did not consume exactly the common warmup/measured frame set")
        if job["trace_steps"]:
            torch.cuda.empty_cache()
            trace_path = output.with_suffix(".trace.json")
            report["operator_trace"] = exercise(
                config,
                subset,
                sampler,
                schedule,
                phases,
                device,
                trace_path=trace_path,
                trace_steps=job["trace_steps"],
            )
        if _file_signatures(dataset, job["selection"]) != job["file_signatures"]:
            raise ValueError("Selected source files changed during this benchmark trial")
        if _source_contract() != job["source"]:
            raise ValueError("Executable source changed during this benchmark trial")
        report["status"] = "passed"
    except (
        OSError,
        ImportError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        ArithmeticError,
        AssertionError,
    ) as error:
        report["failure"] = {
            "type": type(error).__name__,
            "message": redact(str(error), config, output.parent)[:3000],
        }
        report["failure_category"] = (
            "cuda_out_of_memory"
            if isinstance(error, torch.cuda.OutOfMemoryError)
            or "out of memory" in str(error).lower()
            else "failed"
        )
    finally:
        if dataset is not None and hasattr(dataset, "close"):
            dataset.close()
    save_json(output, report)
    return 0 if report["status"] == "passed" else 1


def _parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--output", default="runs/bench")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument(
        "--backends", nargs="+", choices=["torch", "torch_fused", "triton"], default=["torch"]
    )
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        help="Compare explicit spline chunks; default: configured chunk",
    )
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--warmup-per-stream", type=int, default=8)
    parser.add_argument("--frames-per-stream", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--trace-steps",
        type=int,
        default=0,
        help="Separate instrumented pass; excluded from throughput",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def protect_output(output: Path, config: dict) -> None:
    """Do not create benchmark children inside data, source or any known run."""
    protected = [
        PROJECT / name
        for name in ("src", "scripts", "tests", "configs", "constraints", "docs", ".git")
    ]
    protected.append(Path(config["output"]["run_dir"]).resolve())
    protected.extend(
        Path(config["dataset"][key]).resolve()
        for key in ("root", "val_root")
        if config["dataset"].get(key)
    )
    for root in protected:
        if output == root or root in output.parents or output in root.parents:
            raise ValueError(
                "Benchmark output must be separate from dataset, source, and training directories"
            )
    runs = PROJECT / "runs"
    if runs in output.parents and not output.relative_to(runs).parts[0].startswith("bench"):
        raise ValueError(
            "Inside runs/, use a new dedicated bench* directory, not another experiment run"
        )
    for parent in output.parents:
        if any(
            (parent / name).exists()
            for name in ("last.pt", "best.pt", "history.json", "preflight_gate.json")
        ):
            raise ValueError("Benchmark output cannot be nested inside an existing training run")


def aggregate(trials: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for trial in trials:
        variant = trial["variant"]
        grouped[(variant["batch_size"], variant["backend"], variant["spline_chunk_size"])].append(
            trial
        )
    summaries = []
    for (size, backend, chunk), rows in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        passed = [row["measurement"] for row in rows if row["status"] == "passed"]
        fps = [row["frames_per_second"] for row in passed]
        summaries.append(
            {
                "batch_size": size,
                "backend": backend,
                "spline_chunk_size": chunk,
                "passed_repeats": len(passed),
                "requested_repeats": len(rows),
                "all_repeats_passed": len(passed) == len(rows),
                "frames_per_second_median": statistics.median(fps) if fps else None,
                "frames_per_second_min": min(fps) if fps else None,
                "frames_per_second_max": max(fps) if fps else None,
                "peak_allocated_mib_max": max(row["peak_allocated_mib"] for row in passed)
                if passed
                else None,
                "peak_reserved_mib_max": max(row["peak_reserved_mib"] for row in passed)
                if passed
                else None,
            }
        )
    return summaries


def run(args) -> int:
    if args.worker:
        return worker(json.load(sys.stdin))
    for name in ("streams", "warmup_per_stream", "frames_per_stream", "repeats", "threads"):
        _positive(getattr(args, name), name)
    if args.trace_steps < 0:
        raise ValueError("trace-steps must be nonnegative")
    if len(set(args.batches)) != len(args.batches) or any(
        size not in {1, 4, 8, 16} for size in args.batches
    ):
        raise ValueError("Choose distinct real batch sizes from 1, 4, 8, 16")
    if any(args.streams % size for size in args.batches):
        raise ValueError("Common stream count must be divisible by every requested batch size")
    if len(set(args.backends)) != len(args.backends):
        raise ValueError("Backends must not repeat")
    chunks = args.chunks or [None]
    if len(set(chunks)) != len(chunks):
        raise ValueError("Spline chunks must not repeat")
    for chunk in chunks:
        if chunk is not None:
            _positive(chunk, "spline chunk")
    config = resolve_experiment_paths(load_json(args.config), args.config)
    if config.get("dataset", {}).get("type") != "eventhdr":
        raise ValueError("This benchmark uses real EventHDR training windows only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this command never substitutes CPU or generated data")
    output = Path(args.output).expanduser().resolve()
    protect_output(output, config)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    dataset = build_dataset(config["dataset"], split="train")
    try:
        selection = select_windows(
            dataset,
            streams=args.streams,
            warmup=args.warmup_per_stream,
            frames=args.frames_per_stream,
            seed=int(config.get("seed", 2026)),
        )
        files = sorted(
            {dataset.samples[entry["dataset_index"]]["path"] for entry in selection["entries"]}
        )
        content = _dataset_content_fingerprint(SimpleNamespace(root=dataset.root, files=files))
        signatures = _file_signatures(dataset, selection)
    finally:
        dataset.close()
    source = _source_contract()
    report = {
        "schema": "asgcn_real_training_comparison_v1",
        "status": "running",
        "config": _public_config(config),
        "config_sha256": digest(_public_config(config)),
        "source": source,
        "selected_source_content": content,
        "selection": selection,
        # The parent deliberately owns no CUDA context while a trial is running.
        "runtime": None,
        "cpu_threads": args.threads,
        "repeats": args.repeats,
        "trials": [],
        "scope": {
            "full_dataset_training": False,
            "checkpoint_created_or_modified": False,
            "same_frame_set": True,
            "same_per_stream_warmup": True,
            "same_optimizer_step_count": False,
            "batch_sizes_numerically_equivalent": False,
            "loader": "synchronous_real_decode_no_prefetch",
            "cache_policy": "OS_file_cache_not_flushed",
            "gpu_exclusive_allocation_required": True,
            "exclusive_allocation_automatically_verified": False,
            "baseline_scope": "current_checkout_torch_backend_not_previous_commit",
            "remaining_jit_compile_time_included": True,
        },
        "interpretation": "Measured window throughput only. Different batch sizes change BatchNorm pooling and optimizer update frequency; loss equality across batch sizes is not claimed. Do not run beside training on the same GPU/MIG allocation.",
    }
    save_json(output / "report.json", report)
    print(
        "Use an idle, dedicated GPU/MIG allocation. Existing jobs are not stopped; GPU exclusivity is not automatically verified.",
        flush=True,
    )
    variants = [
        (size, backend, chunk, repeat)
        for repeat in range(args.repeats)
        for size in args.batches
        for backend in args.backends
        for chunk in chunks
    ]
    random.Random(int(config.get("seed", 2026))).shuffle(variants)
    failed = False
    for index, (size, backend, chunk, repeat) in enumerate(variants):
        variant = {
            "batch_size": size,
            "backend": backend,
            "spline_chunk_size": chunk
            if chunk is not None
            else config["model"].get("spline_chunk_size"),
            "repeat": repeat,
        }
        label = f"trial-{index:03d}-b{size}-{backend}"
        trial_path = output / f"{label}.json"
        trial_config = variant_config(config, size, backend, chunk)
        job = {
            "output": str(trial_path),
            "config": trial_config,
            "reference_config": variant_config(config, size, "torch", None),
            "variant": variant,
            "selection": selection,
            "source": source,
            "file_signatures": signatures,
            "threads": args.threads,
            "trace_steps": args.trace_steps if repeat == 0 else 0,
        }
        print(
            f"[{index + 1}/{len(variants)}] B{size} {backend} chunk={variant['spline_chunk_size']} repeat={repeat + 1}",
            flush=True,
        )
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(job),
            text=True,
            capture_output=True,
            check=False,
        )
        if trial_path.is_file():
            trial = load_json(trial_path)
        else:
            trial = {
                "status": "failed",
                "variant": variant,
                "failure": {
                    "type": "WorkerFailure",
                    "message": redact(completed.stderr[-3000:], config, output),
                },
            }
            save_json(trial_path, trial)
        valid = completed.returncode == 0 and trial.get("status") == "passed"
        if trial.get("runtime") is not None:
            if report["runtime"] is None:
                report["runtime"] = trial["runtime"]
            elif report["runtime"] != trial["runtime"]:
                valid = False
                trial["failure"] = {
                    "type": "RuntimeChanged",
                    "message": "CUDA/software runtime changed between benchmark trials",
                }
        failed = failed or not valid
        report["trials"].append(
            {
                "file": trial_path.name,
                "variant": variant,
                "status": "passed" if valid else "failed",
                "measurement": trial.get("measurement"),
                "failure": trial.get("failure"),
            }
        )
        save_json(output / "report.json", report)
        if valid:
            measured = trial["measurement"]
            print(
                f"  {measured['frames_per_second']:.2f} frames/s; peak allocated {measured['peak_allocated_mib']:.1f} MiB, reserved {measured['peak_reserved_mib']:.1f} MiB; AMP retries {measured['amp_retries']}",
                flush=True,
            )
        else:
            print(
                "  FAILED: " + trial.get("failure", {}).get("message", "worker did not complete"),
                flush=True,
            )
    final_content = _dataset_content_fingerprint(
        SimpleNamespace(root=Path(config["dataset"]["root"]), files=files)
    )
    if final_content != content or _source_contract() != source:
        failed = True
        report["failure"] = (
            "Selected source data or executable source changed during the comparison"
        )
    report["data_content_unchanged"] = final_content == content
    report["summary"] = aggregate(report["trials"])
    report["status"] = "failed" if failed else "passed"
    save_json(output / "report.json", report)
    return int(failed)


def main() -> int:
    args = _parse()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Benchmark interrupted; completed trial reports are retained.", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(
            "Benchmark refused: " + redact(str(error), {}, Path(args.output).resolve()),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
