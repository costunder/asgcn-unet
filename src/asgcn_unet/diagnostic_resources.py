"""Fail-closed, read-only CPU/RAM preflight for diagnostic exports.

No torch import, CUDA initialization, subprocess, signal, allocation change, or
OS resource-limit change occurs here. A snapshot is not memory isolation: other
jobs can consume RAM after it and a caller can exceed its declared budget.
"""

from __future__ import annotations

import ctypes
import math
import os
import platform
import re
import time
from pathlib import Path, PurePosixPath
from typing import Any

from .resources import _linux_memory, _windows_affinity_count, _windows_memory


class DiagnosticResourceError(RuntimeError):
    """The requested diagnostic cannot pass its resource preflight."""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise DiagnosticResourceError(
            f"Cannot measure required resource file {path}: {exc}"
        ) from exc


def _integer(value: str, label: str, *, positive: bool = False) -> int:
    if not re.fullmatch(r"[0-9]+", value):
        raise DiagnosticResourceError(f"Invalid {label}: {value!r}")
    number = int(value)
    if positive and number == 0:
        raise DiagnosticResourceError(f"{label} must be positive")
    return number


def _posix_path(value: str, label: str) -> PurePosixPath:
    result = PurePosixPath(value)
    if not result.is_absolute() or ".." in result.parts or "\\" in value:
        raise DiagnosticResourceError(f"Unverifiable {label}: {value!r}")
    return result


def _unescape(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _cgroup_locations(proc_root: Path) -> dict[str, dict[str, Any]]:
    """Resolve membership against actual mountinfo, never guessed /sys paths."""
    memberships: dict[str, PurePosixPath] = {}
    for line in _read(proc_root / "self/cgroup").splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3 or not fields[0].isdigit():
            raise DiagnosticResourceError("Malformed /proc/self/cgroup membership")
        kinds = ["unified"] if not fields[1] else fields[1].split(",")
        for kind in kinds:
            if kind in {"unified", "memory", "cpu"}:
                if kind in memberships:
                    raise DiagnosticResourceError(f"Ambiguous {kind} cgroup membership")
                memberships[kind] = _posix_path(fields[2], "cgroup membership")

    mounts: list[dict[str, Any]] = []
    for line in _read(proc_root / "self/mountinfo").splitlines():
        before, separator, after = line.partition(" - ")
        left, right = before.split(), after.split()
        if not separator or len(left) < 6 or len(right) < 3:
            raise DiagnosticResourceError("Malformed /proc/self/mountinfo")
        if right[0] not in {"cgroup", "cgroup2"}:
            continue
        root = _posix_path(_unescape(left[3]), "cgroup mount root")
        mount = Path(_unescape(left[4]))
        if not mount.is_absolute() or ".." in mount.parts:
            raise DiagnosticResourceError("Cgroup mount location is not an absolute safe path")
        mount = mount.resolve(strict=True)
        controllers = set(left[5].split(",")) | set(right[2].split(","))
        mounts.append({"root": root, "mount": mount, "fs": right[0], "controllers": controllers})

    # Hybrid hosts can expose an empty v2 hierarchy while both relevant
    # controllers live in v1. Only resolve hierarchies that control this work.
    required = {"memory"} if "memory" in memberships else {"unified"}
    if "cpu" in memberships:
        required.add("cpu")
    elif "unified" in memberships:
        required.add("unified")
    locations: dict[str, dict[str, Any]] = {}
    for kind, group in memberships.items():
        if kind not in required:
            continue
        candidates = []
        for entry in mounts:
            matches = (
                entry["fs"] == "cgroup2"
                if kind == "unified"
                else (entry["fs"] == "cgroup" and kind in entry["controllers"])
            )
            if not matches or not group.is_relative_to(entry["root"]):
                continue
            relative = group.relative_to(entry["root"])
            leaf = entry["mount"].joinpath(*relative.parts).resolve(strict=True)
            if not leaf.is_relative_to(entry["mount"]):
                raise DiagnosticResourceError(
                    "Cgroup membership resolves outside its verified mount"
                )
            candidates.append(dict(entry, leaf=leaf, membership=str(group)))
        if not candidates:
            raise DiagnosticResourceError(f"Cannot resolve actual {kind} cgroup mount/membership")
        # Prefer the widest visible hierarchy so ancestor limits are included.
        chosen = min(candidates, key=lambda entry: len(entry["root"].parts))
        if str(chosen["root"]) != "/":
            raise DiagnosticResourceError(
                f"{kind} cgroup ancestors are hidden above mount root {chosen['root']}; "
                "their remaining memory/CPU allocation cannot be verified"
            )
        locations[kind] = chosen
    if not locations or not ({"unified", "memory"} & locations.keys()):
        raise DiagnosticResourceError("Cannot verify a Linux memory cgroup hierarchy")
    return locations


def _ancestors(location: dict[str, Any]):
    current, mount = location["leaf"], location["mount"]
    while True:
        yield current
        if current == mount:
            break
        current = current.parent


def _linux_cgroups(proc_root: Path) -> dict[str, Any]:
    locations = _cgroup_locations(proc_root)
    memory: list[dict[str, Any]] = []
    quotas: list[dict[str, Any]] = []
    for kind, location in locations.items():
        for directory in _ancestors(location):
            is_v2 = kind == "unified"
            is_root = directory == location["mount"]
            if kind == "memory" or (is_v2 and "memory" not in locations):
                limit_file = directory / ("memory.max" if is_v2 else "memory.limit_in_bytes")
                usage_file = directory / ("memory.current" if is_v2 else "memory.usage_in_bytes")
                # The true v2 hierarchy root has no memory.max limit interface.
                if is_v2 and is_root and not limit_file.exists():
                    memory.append(
                        {
                            "path": str(directory),
                            "limit_bytes": None,
                            "current_bytes": None,
                            "headroom_bytes": None,
                            "status": "v2_hierarchy_root_host_memory_applies",
                        }
                    )
                else:
                    raw = _read(limit_file)
                    unlimited = raw == "max" if is_v2 else False
                    limit = None if unlimited else _integer(raw, str(limit_file))
                    if not is_v2 and limit is not None and limit >= 2**60:
                        limit = None  # v1 page-aligned LONG_MAX unlimited sentinel.
                    usage = _integer(_read(usage_file), str(usage_file))
                    memory.append(
                        {
                            "path": str(directory),
                            "limit_bytes": limit,
                            "current_bytes": usage,
                            "headroom_bytes": max(0, limit - usage) if limit is not None else None,
                            "status": "measured",
                        }
                    )
            if kind == "cpu" or (is_v2 and "cpu" not in locations):
                quota_file = directory / ("cpu.max" if is_v2 else "cpu.cfs_quota_us")
                if is_v2 and is_root and not quota_file.exists():
                    quota = None
                elif is_v2:
                    fields = _read(quota_file).split()
                    if len(fields) != 2:
                        raise DiagnosticResourceError(f"Invalid CPU quota in {quota_file}")
                    period = _integer(fields[1], "CPU quota period", positive=True)
                    quota = (
                        None
                        if fields[0] == "max"
                        else (_integer(fields[0], "CPU quota", positive=True) / period)
                    )
                else:
                    raw = _read(quota_file)
                    period = _integer(
                        _read(directory / "cpu.cfs_period_us"), "CPU quota period", positive=True
                    )
                    quota = (
                        None
                        if raw == "-1"
                        else (_integer(raw, "CPU quota", positive=True) / period)
                    )
                quotas.append({"path": str(directory), "quota_cores": quota})
    headrooms = [item["headroom_bytes"] for item in memory if item["headroom_bytes"] is not None]
    limits = [item["quota_cores"] for item in quotas if item["quota_cores"] is not None]
    return {
        "source": "/proc/self/cgroup + /proc/self/mountinfo + controller files",
        "memory_headroom_bytes": min(headrooms) if headrooms else None,
        "cpu_quota_cores": min(limits) if limits else None,
        "memory_measurements": memory,
        "cpu_measurements": quotas,
        "locations": {
            key: {
                "mount": str(value["mount"]),
                "leaf": str(value["leaf"]),
                "membership": value["membership"],
            }
            for key, value in locations.items()
        },
        "visibility": "verified mounted hierarchies; host-hidden namespace ancestors are not exposed",
    }


def _windows_in_job() -> bool:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    result = ctypes.c_int()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(result)):
        raise DiagnosticResourceError("Cannot inspect Windows Job Object membership")
    return bool(result.value)


def _snapshot() -> dict[str, Any]:
    system = platform.system()
    if system == "Linux":
        total, available, rss = _linux_memory(Path("/proc"))
        cgroup = _linux_cgroups(Path("/proc"))
        try:
            affinity = len(os.sched_getaffinity(0))
        except (OSError, AttributeError) as exc:
            raise DiagnosticResourceError("Cannot measure this process's CPU affinity") from exc
        source = "/proc/meminfo MemAvailable and /proc/self/status VmRSS"
    elif system == "Windows":
        if _windows_in_job():
            raise DiagnosticResourceError(
                "Windows Job Object membership detected; nested memory/CPU restrictions cannot "
                "be verified by this exporter. No resource limits were changed."
            )
        total, available, rss = _windows_memory()
        affinity = _windows_affinity_count()
        cgroup = {
            "source": "not applicable: Windows; Job Object membership checked absent",
            "memory_headroom_bytes": None,
            "cpu_quota_cores": None,
        }
        source = "GlobalMemoryStatusEx and GetProcessMemoryInfo"
    else:
        raise DiagnosticResourceError(f"Resource preflight is not implemented for {system}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (total, available, rss, affinity)
    ):
        raise DiagnosticResourceError(
            "Cannot measure current available RAM, process RSS, or CPU affinity"
        )
    if available > total:
        raise DiagnosticResourceError(
            "Available RAM exceeds total RAM; resource snapshot is invalid"
        )
    limits = [float(affinity)]
    if cgroup["cpu_quota_cores"] is not None:
        limits.append(cgroup["cpu_quota_cores"])
    scheduler = {}
    for key in ("SLURM_CPUS_PER_TASK", "NSLOTS", "NCPUS"):
        if key in os.environ:
            scheduler[key] = _integer(os.environ[key], key, positive=True)
            limits.append(float(scheduler[key]))
    headroom = available
    if cgroup["memory_headroom_bytes"] is not None:
        headroom = min(headroom, cgroup["memory_headroom_bytes"])
    return {
        "headroom_bytes": headroom,
        "system": {
            "platform": system,
            "total_bytes": total,
            "available_bytes": available,
            "process_rss_bytes": rss,
            "source": source,
        },
        "cpu": {
            "affinity_count": affinity,
            "quota_cores": cgroup["cpu_quota_cores"],
            "effective_cores": min(limits),
            "scheduler_limits": scheduler,
            "source": "process affinity, mounted CPU quotas, explicit scheduler task limits",
        },
        "cgroup": cgroup,
    }


def preflight(*, budget_bytes: int, reserve_bytes: int, cpu_threads: int) -> dict[str, Any]:
    """Check explicit incremental RAM budget + reserve against measured headroom.

    Call before heavy imports/dataset reads and again before major allocations.
    No settings are changed. The caller must enforce its algorithmic budget and
    explicitly set its own thread pools separately. Fractional CPU quotas allow
    one OS thread; the kernel enforces the fractional scheduling quota.
    """
    for label, value in (
        ("budget_bytes", budget_bytes),
        ("reserve_bytes", reserve_bytes),
        ("cpu_threads", cpu_threads),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise DiagnosticResourceError(f"{label} must be an explicit positive integer")
    try:
        result = _snapshot()
    except (OSError, UnicodeError) as exc:
        raise DiagnosticResourceError(f"Cannot verify diagnostic allocation: {exc}") from exc
    max_threads = max(1, math.floor(result["cpu"]["effective_cores"]))
    if cpu_threads > max_threads:
        raise DiagnosticResourceError(
            f"Requested cpu_threads={cpu_threads} exceeds measured allocation "
            f"({result['cpu']['effective_cores']:g} cores; at most {max_threads} threads). "
            "No thread or affinity setting was changed."
        )
    required = budget_bytes + reserve_bytes
    if required > result["headroom_bytes"]:
        raise DiagnosticResourceError(
            f"Insufficient measured RAM headroom: {result['headroom_bytes']:,} bytes available; "
            f"budget {budget_bytes:,} + reserve {reserve_bytes:,} = {required:,} bytes required. "
            "No graph was reduced and no GPU/CPU fallback was attempted."
        )
    result.update(
        {
            "schema": "asgcn_diagnostic_resource_preflight_v1",
            "passed": True,
            "monotonic_seconds": time.monotonic(),
            "budget_bytes": budget_bytes,
            "reserve_bytes": reserve_bytes,
            "required_headroom_bytes": required,
            "cpu_threads": cpu_threads,
            "max_cpu_threads": max_threads,
            "gpu_queried": False,
            "resource_settings_modified": False,
            "hard_memory_isolation": False,
            "limitations": [
                "Point-in-time free-memory snapshot, not peak RSS enforcement or memory isolation.",
                "Other jobs may consume RAM after this check; repeat it before major allocations.",
                "The caller must account for dataset decoding, libraries, and temporary buffers.",
                "Host-hidden cgroup namespace ancestor limits are not exposed by proc/mountinfo.",
                "CPU availability is allocation capacity, not exclusive ownership or current idle CPU.",
            ],
        }
    )
    return result
