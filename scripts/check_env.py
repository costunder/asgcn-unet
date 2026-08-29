from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import sys
from pathlib import Path, PurePosixPath

import torch

from asgcn_recon.data import load_eventhdr_split_manifest


def _eventhdr_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([*root.rglob("*.h5"), *root.rglob("*.hdf5")])


def _count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.glob(pattern)) if root.exists() else 0


def _locked_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", maxsplit=1)
        versions[name.strip().lower().replace("_", "-")] = version.strip()
    return versions


def _check_lock(path: Path) -> dict[str, dict[str, str | None]]:
    mismatches: dict[str, dict[str, str | None]] = {}
    for name, expected in _locked_versions(path).items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        actual_public = actual.split("+", maxsplit=1)[0] if actual else None
        expected_public = expected.split("+", maxsplit=1)[0]
        if actual_public != expected_public:
            mismatches[name] = {"expected": expected, "actual": actual}
    return mismatches


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ASGCN server readiness")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-eventhdr-smoke", action="store_true")
    parser.add_argument("--require-eventhdr-train", action="store_true")
    parser.add_argument("--require-eventhdr-eval", action="store_true")
    parser.add_argument("--require-eventaid-all", action="store_true")
    parser.add_argument("--require-full-data", action="store_true")
    parser.add_argument("--lock", type=Path, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_root = (args.data_root or project_root / "data").resolve()
    runs_root = (args.runs_root or project_root / "runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    gpu_memory_gib = [
        round(torch.cuda.get_device_properties(index).total_memory / (1024**3), 2)
        for index in range(torch.cuda.device_count())
    ]
    lock_path = args.lock.resolve() if args.lock else None
    lock_mismatches = _check_lock(lock_path) if lock_path and lock_path.is_file() else None
    lock_python_match = None
    if lock_path:
        match = re.fullmatch(r"py(\d)(\d+)", lock_path.stem)
        if match:
            expected_python = f"{match.group(1)}.{match.group(2)}"
            lock_python_match = platform.python_version().startswith(f"{expected_python}.")
    train_files = _eventhdr_files(data_root / "EventHDR" / "train")
    eval_files = _eventhdr_files(data_root / "EventHDR" / "eval")
    train_root = data_root / "EventHDR" / "train"
    train_present = {path.relative_to(train_root).as_posix() for path in train_files}
    smoke_manifest_path = project_root / "manifests" / "eventhdr_smoke.json"
    smoke_required: set[str] = set()
    if smoke_manifest_path.is_file():
        smoke_manifest = load_eventhdr_split_manifest(smoke_manifest_path)
        smoke_required = {
            PurePosixPath(str(value).replace("\\", "/")).as_posix()
            for value in (
                list(smoke_manifest.get("train_files", []))
                + list(smoke_manifest.get("val_files", []))
            )
        }
    data_disk = shutil.disk_usage(data_root if data_root.exists() else project_root)
    runs_disk = shutil.disk_usage(runs_root)
    libc_name, libc_version = platform.libc_ver()
    report = {
        "project_root": str(project_root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "libc": {"name": libc_name or None, "version": libc_version or None},
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_devices": devices,
        "gpu_memory_gib": gpu_memory_gib,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "data_root": str(data_root),
        "eventhdr_train_h5": len(train_files),
        "eventhdr_smoke_h5": len(smoke_required & train_present),
        "eventhdr_eval_h5": len(eval_files),
        "eventaid_r_zip": _count_files(data_root / "EventAid-R", "R-*.zip"),
        "runs_root": str(runs_root),
        "runs_writable": os.access(runs_root, os.W_OK),
        "data_disk_free_gib": round(data_disk.free / (1024**3), 2),
        "runs_disk_free_gib": round(runs_disk.free / (1024**3), 2),
        "lock_file": str(lock_path) if lock_path else None,
        "constraint_versions_match": (not lock_mismatches if lock_mismatches is not None else None),
        "constraint_python_match": lock_python_match,
        "lock_mismatches": lock_mismatches,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    problems: list[str] = []
    if not report["runs_writable"]:
        problems.append(f"Run directory is not writable: {runs_root}")
    if args.require_cuda and not cuda_available:
        problems.append("CUDA was required but torch.cuda.is_available() is false")
    if lock_path and not lock_path.is_file():
        problems.append(f"Dependency lock does not exist: {lock_path}")
    elif lock_mismatches:
        problems.append(f"Installed packages differ from dependency lock: {lock_path}")
    if lock_python_match is False:
        problems.append(f"Python version does not match dependency profile: {lock_path}")
    locked_torch = (
        _locked_versions(lock_path).get("torch") if lock_path and lock_path.is_file() else None
    )
    if (
        platform.system() == "Linux"
        and libc_name.lower() == "glibc"
        and locked_torch == "2.13.0"
        and _version_tuple(libc_version) < (2, 28)
    ):
        problems.append(
            f"torch 2.13.0 wheel profile requires glibc>=2.28; found {libc_version}. "
            "Use a newer cluster container/module instead of building from source blindly"
        )
    require_eventhdr_train = args.require_full_data or args.require_eventhdr_train
    require_eventhdr_eval = args.require_full_data or args.require_eventhdr_eval
    require_eventaid_all = args.require_full_data or args.require_eventaid_all
    expected_counts = {
        "eventhdr_train_h5": (51, require_eventhdr_train),
        "eventhdr_eval_h5": (19, require_eventhdr_eval),
        "eventaid_r_zip": (14, require_eventaid_all),
    }
    for key, (expected, required) in expected_counts.items():
        actual = int(report[key])
        if required and actual < expected:
            problems.append(f"{key} has {actual} files; at least {expected} are required")
    if args.require_eventhdr_smoke:
        if not smoke_required:
            problems.append(f"Smoke manifest has no required H5 files: {smoke_manifest_path}")
        smoke_missing = sorted(smoke_required - train_present)
        if smoke_missing:
            problems.append(
                "EventHDR train directory is missing smoke manifest files: "
                + ", ".join(smoke_missing)
            )
    if require_eventhdr_train:
        manifest_path = project_root / "manifests" / "eventhdr_split.json"
        if manifest_path.is_file():
            manifest = load_eventhdr_split_manifest(manifest_path)
            required = {
                PurePosixPath(str(value).replace("\\", "/")).as_posix()
                for value in (
                    list(manifest.get("train_files", [])) + list(manifest.get("val_files", []))
                )
            }
            missing = sorted(required - train_present)
            if missing:
                problems.append(
                    "EventHDR train directory is missing manifest files: "
                    + ", ".join(missing[:8])
                    + (" ..." if len(missing) > 8 else "")
                )
    if require_eventaid_all:
        aid_manifest_path = project_root / "manifests" / "eventaid_r.json"
        if aid_manifest_path.is_file():
            aid_manifest = json.loads(aid_manifest_path.read_text(encoding="utf-8"))
            aid_root = data_root / "EventAid-R"
            aid_present = {path.name for path in aid_root.glob("R-*.zip")}
            aid_required = {
                f"{item['scene']}.zip"
                for item in aid_manifest.get("files", [])
                if isinstance(item, dict) and item.get("scene")
            }
            aid_missing = sorted(aid_required - aid_present)
            if aid_missing:
                problems.append(
                    "EventAid-R directory is missing manifest files: "
                    + ", ".join(aid_missing[:8])
                    + (" ..." if len(aid_missing) > 8 else "")
                )
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
