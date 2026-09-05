"""Read-only allocation and execution reports; unavailable measurements stay null.

Importing this module never imports torch or initializes CUDA. CPU and memory
limits describe this process's allocation, not permission to use other devices.
"""

from __future__ import annotations

import ctypes
import math
import os
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _positive_number(value: str | None) -> float | None:
    try:
        number = float(value) if value is not None else None
    except ValueError:
        return None
    return number if number is not None and math.isfinite(number) and number > 0 else None


def _nonnegative_integer(value: str | None) -> int | None:
    return int(value) if value is not None and value.isdigit() else None


def _cpu_set_count(value: str | None) -> int | None:
    if not value:
        return None
    result: set[int] = set()
    try:
        for item in value.split(","):
            if "-" in item:
                start, end = (int(number) for number in item.split("-", 1))
                if start < 0 or end < start:
                    return None
                result.update(range(start, end + 1))
            else:
                cpu = int(item)
                if cpu < 0:
                    return None
                result.add(cpu)
    except ValueError:
        return None
    return len(result) or None


def _minimum(values: list[int | float | None]) -> int | float | None:
    known = [value for value in values if value is not None]
    return min(known) if known else None


def _cgroup_directories(root: Path, membership: str | None) -> dict[str, list[Path]]:
    """Include ancestors: a leaf may be unlimited while its parent is limited."""
    candidates: dict[str, list[Path]] = {"unified": [], "cpu": [], "memory": [], "cpuset": []}
    for line in (membership or "").splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, relative = fields
        # A cgroup namespace can report /../.. membership. Do not escape root.
        parts = Path(relative.lstrip("/")).parts
        if ".." in parts:
            continue
        if not controllers:
            entries = [("unified", root)]
        else:
            names = controllers.split(",")
            entries = [
                (name, base)
                for name in ("cpu", "memory", "cpuset")
                if name in names
                for base in (root / controllers, root / name)
            ]
        for kind, base in entries:
            current = base.joinpath(*parts)
            while True:
                if current not in candidates[kind]:
                    candidates[kind].append(current)
                if current == base:
                    break
                current = current.parent
    # Namespaced containers commonly expose only their delegated root.
    if root not in candidates["unified"]:
        candidates["unified"].append(root)
    return candidates


def _cgroup_resources(root: Path, membership: str | None) -> dict[str, Any]:
    directories = _cgroup_directories(root, membership)
    quotas: list[float] = []
    cpusets: list[int] = []
    memory_limits: list[int] = []
    headrooms: list[int] = []
    measurements: list[dict[str, Any]] = []
    for kind, paths in directories.items():
        for directory in paths:
            quota = None
            limit = None
            usage = None
            if kind == "unified":
                raw = (_read_text(directory / "cpu.max") or "").split()
                if len(raw) == 2:
                    numerator, denominator = map(_positive_number, raw)
                    if numerator is not None and denominator is not None:
                        quota = numerator / denominator
                limit = _nonnegative_integer(_read_text(directory / "memory.max"))
                usage_raw = _read_text(directory / "memory.current")
                cpuset = _cpu_set_count(_read_text(directory / "cpuset.cpus.effective"))
            elif kind == "cpu":
                numerator = _positive_number(_read_text(directory / "cpu.cfs_quota_us"))
                denominator = _positive_number(_read_text(directory / "cpu.cfs_period_us"))
                if numerator is not None and denominator is not None:
                    quota = numerator / denominator
                usage_raw, cpuset = None, None
            elif kind == "memory":
                limit = _nonnegative_integer(_read_text(directory / "memory.limit_in_bytes"))
                # v1 uses a page-aligned LONG_MAX sentinel for unlimited memory.
                if limit is not None and limit >= 2**60:
                    limit = None
                usage_raw = _read_text(directory / "memory.usage_in_bytes")
                cpuset = None
            else:
                usage_raw = None
                cpuset = _cpu_set_count(_read_text(directory / "cpuset.cpus"))
            if usage_raw is not None and usage_raw.isdigit():
                usage = int(usage_raw)
            if quota is not None:
                quotas.append(quota)
            if cpuset is not None:
                cpusets.append(cpuset)
            if limit is not None:
                memory_limits.append(int(limit))
                if usage is not None:
                    headrooms.append(max(0, int(limit) - usage))
            if quota is not None or limit is not None or cpuset is not None or usage is not None:
                measurements.append(
                    {
                        "controller": kind,
                        "cpu_quota_cores": quota,
                        "cpuset_cpu_count": cpuset,
                        "memory_limit_bytes": int(limit) if limit is not None else None,
                        "memory_current_bytes": usage,
                    }
                )
    return {
        "cpu_quota_cores": _minimum(quotas),
        "cpuset_cpu_count": _minimum(cpusets),
        "memory_limit_bytes": _minimum(memory_limits),
        "memory_headroom_bytes": _minimum(headrooms),
        "measurements": measurements,
        "available": bool(measurements),
    }


def _windows_memory() -> tuple[int | None, int | None, int | None]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong)
            for name in (
                "total",
                "available",
                "page_total",
                "page_available",
                "virtual_total",
                "virtual_available",
                "extended",
            )
        ]

    class ProcessMemory(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("page_faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak_working_set",
                "working_set",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    system_ok = kernel.GlobalMemoryStatusEx(ctypes.byref(status))
    memory = ProcessMemory()
    memory.cb = ctypes.sizeof(memory)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    process_ok = psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(memory), memory.cb
    )
    return (
        int(status.total) if system_ok else None,
        int(status.available) if system_ok else None,
        int(memory.working_set) if process_ok else None,
    )


def _windows_affinity_count() -> int | None:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    success = kernel.GetProcessAffinityMask(
        kernel.GetCurrentProcess(),
        ctypes.byref(process_mask),
        ctypes.byref(system_mask),
    )
    return process_mask.value.bit_count() if success and process_mask.value else None


def _linux_memory(proc_root: Path) -> tuple[int | None, int | None, int | None]:
    def fields(path: Path) -> dict[str, int]:
        result = {}
        for line in (_read_text(path) or "").splitlines():
            key, separator, value = line.partition(":")
            parts = value.split()
            if separator and len(parts) == 2 and parts[1] == "kB" and parts[0].isdigit():
                result[key] = int(parts[0]) * 1024
        return result

    memory_fields = fields(proc_root / "meminfo")
    process = fields(proc_root / "self" / "status")
    return memory_fields.get("MemTotal"), memory_fields.get("MemAvailable"), process.get("VmRSS")


def collect_runtime_resources(*, device: Any = None, include_cuda: bool = False) -> dict[str, Any]:
    """Collect host/process allocation limits without modifying any settings.

    CUDA is queried only on explicit opt-in. Missing OS counters are represented
    by None and named in unavailable_fields, never by invented zero measurements.
    """
    logical = os.cpu_count()
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
        except OSError:
            affinity = None
    elif os.name == "nt":
        affinity = _windows_affinity_count()
    scheduler_keys = (
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "PBS_NP",
        "NSLOTS",
        "NCPUS",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
    )
    scheduler = {key: os.environ[key] for key in scheduler_keys if key in os.environ}
    # CPUs per task describe this process; whole-node scheduler allocations do not.
    scheduler_cpu = _minimum(
        [
            _positive_number(os.environ.get(key))
            for key in ("SLURM_CPUS_PER_TASK", "NSLOTS", "NCPUS")
        ]
    )
    if platform.system() == "Linux":
        total, available, rss = _linux_memory(Path("/proc"))
        cgroup = _cgroup_resources(Path("/sys/fs/cgroup"), _read_text(Path("/proc/self/cgroup")))
    else:
        total, available, rss = _windows_memory() if os.name == "nt" else (None, None, None)
        cgroup = _cgroup_resources(Path("/__asgcn_unavailable_cgroup__"), None)
    cpu_limit = _minimum(
        [logical, affinity, scheduler_cpu, cgroup["cpu_quota_cores"], cgroup["cpuset_cpu_count"]]
    )
    gpu: dict[str, Any] = {
        "queried": include_cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selected_device": str(device) if device is not None else None,
        "available": None,
        "visible_device_count": None,
        "devices": None,
    }
    if include_cuda:
        from .allocation import require_gpu_allocation

        gpu["allocation_evidence"] = require_gpu_allocation()
        import torch

        gpu["available"] = bool(torch.cuda.is_available())
        if gpu["available"]:
            # Under MIG, pre-initialization NVML counts can be physical-device
            # counts. Initialize first, just like scripts/check_env.py.
            torch.cuda.init()
        gpu["visible_device_count"] = torch.cuda.device_count() if gpu["available"] else 0
        if gpu["available"] and gpu["visible_device_count"] < 1:
            raise RuntimeError("CUDA initialized but reported no visible devices")
        gpu["devices"] = []
        for index in range(gpu["visible_device_count"]):
            properties = torch.cuda.get_device_properties(index)
            name = properties.name
            uuid = str(getattr(properties, "uuid", ""))
            gpu["devices"].append(
                {
                    "visible_index": index,
                    "name": name,
                    "total_memory_bytes": int(properties.total_memory),
                    "mig": True if "MIG" in name or uuid.startswith("MIG-") else None,
                    "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
                }
            )
    result = {
        "schema": "asgcn_resource_report_v1",
        "monotonic_seconds": time.monotonic(),
        "cpu": {
            "logical_cpu_count": logical,
            "affinity_cpu_count": affinity,
            "scheduler_cpu_limit": scheduler_cpu,
            "effective_cpu_limit": cpu_limit,
            "process_cpu_seconds": time.process_time(),
            "utilization_percent": None,
            "utilization_note": "An interval measurement is required; a resource snapshot is not utilization.",
        },
        "memory": {
            "host_total_bytes": total,
            "host_available_bytes": available,
            "process_rss_bytes": rss,
            "effective_limit_bytes": _minimum([total, cgroup["memory_limit_bytes"]]),
            "effective_available_bytes": _minimum([available, cgroup["memory_headroom_bytes"]]),
        },
        "cgroup": cgroup,
        "scheduler": scheduler,
        "gpu": gpu,
    }
    result["unavailable_fields"] = [
        f"{section}.{key}"
        for section in ("cpu", "memory", "gpu")
        for key, value in result[section].items()
        if value is None
    ]
    return result


def summarize_resource_interval(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Measure process CPU usage over two snapshots; 100% means one CPU core.

    DataLoader child processes and other jobs are not part of process_cpu_percent.
    Equal wall-clock readings can occur within the monotonic clock's resolution;
    retain the counters but do not fabricate an elapsed time or CPU usage rate.
    """
    elapsed = after["monotonic_seconds"] - before["monotonic_seconds"]
    cpu_seconds = after["cpu"]["process_cpu_seconds"] - before["cpu"]["process_cpu_seconds"]
    if (
        not math.isfinite(elapsed)
        or elapsed < 0
        or not math.isfinite(cpu_seconds)
        or cpu_seconds < 0
    ):
        raise ValueError(
            "Resource snapshots must be in increasing time order with valid CPU counters"
        )
    percent = 100.0 * cpu_seconds / elapsed if elapsed > 0 else None
    limit = after["cpu"].get("effective_cpu_limit")
    return {
        "wall_seconds": elapsed,
        "process_cpu_seconds": cpu_seconds,
        "process_cpu_percent": percent,
        "process_cpu_allocation_percent": percent / limit
        if percent is not None and limit is not None and limit > 0
        else None,
        "process_cpu_utilization_status": "measured"
        if elapsed > 0
        else "unavailable_zero_wall_interval",
        "process_cpu_utilization_note": None
        if elapsed > 0
        else "Monotonic clock readings are equal; the interval is below clock resolution.",
        "includes_child_processes": False,
        "rss_before_bytes": before["memory"].get("process_rss_bytes"),
        "rss_after_bytes": after["memory"].get("process_rss_bytes"),
        "gpu_utilization_percent": None,
        "gpu_utilization_note": "Memory counters are not GPU utilization measurements.",
    }


def build_execution_report(
    model: Any,
    config: Mapping[str, Any],
    *,
    phase: str,
    device: Any = None,
    physical_batch_size: int | None = None,
    gradient_accumulation_steps: int = 1,
    data_parallel_workers: int = 1,
    dataset_size: int | None = None,
    used_samples: int | None = None,
    optimization_steps: int | None = None,
    input_shapes: Any = None,
    graph_statistics: Any = None,
    loader: Any = None,
    include_cuda: bool | None = None,
) -> dict[str, Any]:
    """Describe actual caller-supplied execution, retaining unmeasured fields.

    The batch size is the configured physical maximum, not a claim that a final
    incomplete batch or every sequence lane contains that many samples.
    """
    if phase == "calibration":
        section = config.get("calibration", config.get("train", config.get("eval", {})))
    else:
        section = config.get("train" if phase == "train" else "eval", {})
    if not isinstance(section, Mapping):
        raise TypeError("Execution config section must be a mapping")
    batch = physical_batch_size if physical_batch_size is not None else section.get("batch_size")
    for label, value in (
        ("physical_batch_size", batch),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
        ("data_parallel_workers", data_parallel_workers),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{label} must be a positive integer")
    for label, value in (
        ("dataset_size", dataset_size),
        ("used_samples", used_samples),
        ("optimization_steps", optimization_steps),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{label} must be a non-negative integer")
    if dataset_size is not None and used_samples is not None and used_samples > dataset_size:
        raise ValueError("used_samples cannot exceed dataset_size")
    parameters = list(model.parameters()) if model is not None else None
    total = sum(parameter.numel() for parameter in parameters) if parameters is not None else None
    trainable = (
        sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        if parameters is not None
        else None
    )
    data = config.get("dataset", {})
    actual_device = str(device) if device is not None else None
    cuda_requested = actual_device is not None and actual_device.startswith("cuda")
    return {
        "schema": "asgcn_execution_report_v1",
        "phase": phase,
        "model": {
            "name": type(model).__name__ if model is not None else None,
            "config": dict(config.get("model", {})),
            "total_parameters": total,
            "trainable_parameters": trainable,
        },
        "device": {"requested": config.get("device"), "effective": actual_device},
        "batching": {
            "physical_batch_size": batch,
            "physical_batch_size_kind": "configured_maximum",
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "data_parallel_workers": data_parallel_workers,
            "effective_batch_size": batch * gradient_accumulation_steps * data_parallel_workers
            if batch is not None
            else None,
        },
        "loader": {
            key: getattr(loader, key, section.get(key)) if loader is not None else section.get(key)
            for key in (
                "batching",
                "num_workers",
                "persistent_workers",
                "prefetch_factor",
                "pin_memory",
                "cache",
            )
        },
        "data": {
            "config": dict(data),
            "dataset_size": dataset_size,
            "used_samples": used_samples,
            "used_ratio": used_samples / dataset_size
            if dataset_size and used_samples is not None
            else None,
            "input_shapes": input_shapes,
            "graph_statistics": graph_statistics,
            "sampling_ratio": None,
            "sampling_ratio_note": "Measure retained/raw events per input; max_events is not a fixed sampling ratio.",
        },
        "epochs": section.get("epochs") if phase == "train" else None,
        "optimization_steps": optimization_steps,
        "precision": {
            key: section.get(key, config.get(key)) for key in ("precision", "amp", "tf32")
        },
        "resources": collect_runtime_resources(
            device=device, include_cuda=cuda_requested if include_cuda is None else include_cuda
        ),
    }
