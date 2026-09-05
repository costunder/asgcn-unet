"""Read-only, pre-CUDA allocation checks; never select or change a GPU mask.

An explicit CUDA selector is a declaration of the caller's current allocation,
not proof of ownership. Without one we require a kernel device whitelist, not
device count, a scheduler job id, or a GPU/MIG product name.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_UUID_TEMPLATE = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
_MIG = re.compile(rf"(?:MIG-{_UUID}|MIG-GPU-{_UUID}/[0-9]+/[0-9]+)\Z")


def _gpu_uuid_prefix(token: str) -> bool:
    if not token.startswith("GPU-"):
        return False
    value = token[4:]
    # CUDA accepts unique prefixes, but a prefix must retain UUID punctuation.
    return (
        8 <= len(value) <= len(_UUID_TEMPLATE)
        and not value.endswith("-")
        and all(
            character == "-" if expected == "-" else character in "0123456789abcdefABCDEF"
            for character, expected in zip(value, _UUID_TEMPLATE, strict=False)
        )
    )


def _cuda_selectors(value: str) -> list[str] | None:
    tokens = value.split(",")
    canonical: list[str] = []
    for token in tokens:
        # Do not normalize a mask into a different CUDA runtime interpretation.
        if re.fullmatch(r"[0-9]+", token):
            if len(token) > 10 or int(token) > 2**31 - 1:
                return None
            key = str(int(token))
        elif _gpu_uuid_prefix(token) or _MIG.fullmatch(token):
            key = token.lower()
        else:
            return None
        if key in canonical:
            return None
        canonical.append(key)
    return tokens


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _unescape_mount(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _device_cgroup(proc_root: Path) -> Path | None:
    membership = _read_text(proc_root / "self/cgroup")
    mounts = _read_text(proc_root / "self/mountinfo")
    if membership is None or mounts is None:
        return None
    group: PurePosixPath | None = None
    for line in membership.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and "devices" in fields[1].split(","):
            group = PurePosixPath(fields[2])
            break
    if group is None or not group.is_absolute() or ".." in group.parts:
        return None
    for line in mounts.splitlines():
        before, separator, after = line.partition(" - ")
        left, right = before.split(), after.split()
        if not separator or len(left) < 6 or len(right) < 3 or right[0] != "cgroup":
            continue
        if "devices" not in set(left[5].split(",")) | set(right[2].split(",")):
            continue
        root = PurePosixPath(_unescape_mount(left[3]))
        if not root.is_absolute() or ".." in root.parts:
            continue
        try:
            relative = group.relative_to(root)
        except ValueError:
            continue
        mount = Path(_unescape_mount(left[4])).resolve()
        candidate = mount.joinpath(*relative.parts).resolve()
        if candidate.is_relative_to(mount):
            return candidate
    return None


def _nvidia_nodes(proc_root: Path, dev_root: Path) -> dict[int, set[int]] | None:
    devices = _read_text(proc_root / "devices")
    if devices is None:
        return None
    majors: set[int] = set()
    character = False
    for line in devices.splitlines():
        if line == "Character devices:":
            character = True
        elif line == "Block devices:":
            character = False
        elif character:
            fields = line.split()
            if (
                len(fields) == 2
                and fields[0].isdigit()
                and fields[1] in {"nvidia", "nvidia-frontend"}
            ):
                majors.add(int(fields[0]))
    result: dict[int, set[int]] = {}
    try:
        for node in dev_root.glob("nvidia[0-9]*"):
            if not re.fullmatch(r"nvidia[0-9]+", node.name):
                continue
            info = node.stat()
            if not stat.S_ISCHR(info.st_mode):
                continue
            major, minor = os.major(info.st_rdev), os.minor(info.st_rdev)
            if major not in majors or minor != int(node.name[6:]):
                return None
            result.setdefault(major, set()).add(minor)
    except OSError:
        return None
    return result or None


def _restricted_gpu_whitelist(value: str, nodes: dict[int, set[int]]) -> bool:
    allowed: dict[int, dict[int, set[str]]] = {major: {} for major in nodes}
    for line in value.splitlines():
        fields = line.split()
        if len(fields) != 3 or fields[0] not in {"a", "b", "c"}:
            return False
        match = re.fullmatch(r"(\*|[0-9]+):(\*|[0-9]+)", fields[1])
        if match is None or not fields[2] or set(fields[2]) - set("rwm"):
            return False
        if fields[0] == "b" or not set(fields[2]) & set("rw"):
            continue
        major_text, minor_text = match.groups()
        for major in nodes:
            if major_text != "*" and int(major_text) != major:
                continue
            if fields[0] == "a" or major_text == "*" or minor_text == "*":
                return False
            minor = int(minor_text)
            # The NVIDIA control node does not select a physical GPU.
            if minor == 255:
                continue
            allowed[major].setdefault(minor, set()).update(fields[2])
    return all(
        set(allowed[major]) == visible
        and all({"r", "w"} <= permissions for permissions in allowed[major].values())
        for major, visible in nodes.items()
    )


def _container_device_evidence(
    *, proc_root: Path = Path("/proc"), dev_root: Path = Path("/dev")
) -> dict[str, Any] | None:
    # cgroup v2 device-BPF policy is not readable through devices.list. Fail
    # closed there; an explicit CUDA selector remains supported on every OS.
    cgroup = _device_cgroup(proc_root)
    if cgroup is None:
        return None
    whitelist = _read_text(cgroup / "devices.list")
    nodes = _nvidia_nodes(proc_root, dev_root)
    if whitelist is None or nodes is None or not _restricted_gpu_whitelist(whitelist, nodes):
        return None
    return {
        "source": "kernel_cgroup_v1_device_whitelist",
        "visible_gpu_device_nodes": sum(len(minors) for minors in nodes.values()),
        "kernel_device_restriction_verified": True,
    }


def inspect_gpu_allocation(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Inspect allocation evidence without importing torch or querying a GPU."""
    environment = os.environ if environ is None else environ
    evidence: dict[str, Any] = {
        "schema": "asgcn_gpu_allocation_v1",
        "verified": False,
        "source": None,
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "mask_modified": False,
        "ownership_verified": False,
    }
    if "CUDA_VISIBLE_DEVICES" in environment:
        selectors = _cuda_selectors(environment["CUDA_VISIBLE_DEVICES"])
        if selectors is None:
            evidence["reason"] = "CUDA_VISIBLE_DEVICES is empty, disabled, duplicated, or malformed"
            return evidence
        evidence.update(verified=True, source="explicit_cuda_visible_devices", selectors=selectors)
        evidence["reason"] = (
            "Caller supplied explicit CUDA selectors; the caller must own this allocation"
        )
        return evidence
    container = _container_device_evidence()
    if container is not None:
        evidence.update(container, verified=True)
        evidence["reason"] = "Visible GPU nodes match a restrictive kernel device whitelist"
        return evidence
    evidence["reason"] = (
        "No explicit CUDA_VISIBLE_DEVICES or verifiable kernel GPU device restriction. "
        "Scheduler job IDs, GPU counts, MIG names, and NVIDIA_VISIBLE_DEVICES alone are insufficient"
    )
    return evidence


def require_gpu_allocation() -> dict[str, Any]:
    """Reject unallocated/ambiguous GPU access before any CUDA initialization."""
    evidence = inspect_gpu_allocation()
    if not evidence["verified"]:
        raise RuntimeError(
            "CUDA allocation safety check failed: " + evidence["reason"] + ". "
            "Preserve your scheduler/container allocation, or explicitly set CUDA_VISIBLE_DEVICES "
            "to the identifier(s) assigned to your current job. No GPU was selected or mask changed. "
            "Use device=cpu explicitly only for CPU diagnostics/tests."
        )
    return evidence
