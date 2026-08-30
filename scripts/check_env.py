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
from pathlib import Path

import torch
from torch.cuda import DeferredCudaCallError

from asgcn_unet.data import load_eventhdr_split_manifest

_OFFICIAL_EVENTHDR_TRAIN = {f"{index}.h5" for index in range(1, 52)}
_OFFICIAL_EVENTHDR_EVAL = {f"{index}.h5" for index in range(1, 20)}


def _logical_path(path: Path | None, project_root: Path, fallback: str) -> str | None:
    """Return a stable public label instead of a host-specific absolute path."""

    if path is None:
        return None
    try:
        relative = path.relative_to(project_root)
    except ValueError:
        return fallback
    if relative == Path("."):
        return "$PROJECT_ROOT"
    return f"$PROJECT_ROOT/{relative.as_posix()}"


def _redact_host_paths(message: str, replacements: list[tuple[Path, str]]) -> str:
    """Replace configured host paths in a routine error with portable labels."""

    result = message
    ordered = sorted(replacements, key=lambda item: len(str(item[0])), reverse=True)
    for path, label in ordered:
        variants = {str(path), path.as_posix()}
        for variant in sorted(variants, key=len, reverse=True):
            result = result.replace(f"{variant}\\", f"{label}/")
            result = result.replace(f"{variant}/", f"{label}/")
            result = result.replace(variant, label)
    return result


def _host_error(
    error: BaseException,
    replacements: list[tuple[Path, str]],
    include_private_host_provenance: bool,
) -> str:
    message = str(error)
    if include_private_host_provenance:
        return message
    return _redact_host_paths(message, replacements)


def _eventhdr_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([*root.rglob("*.h5"), *root.rglob("*.hdf5")])


def _count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.glob(pattern)) if root.exists() else 0


def _exact_coverage_problem(
    label: str,
    present: set[str],
    expected: set[str],
) -> str | None:
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if not missing and not extra:
        return None
    details = []
    if missing:
        details.append("missing=" + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
    if extra:
        details.append("extra=" + ", ".join(extra[:8]) + (" ..." if len(extra) > 8 else ""))
    return (
        f"{label} must contain exactly {len(expected)} official files "
        f"({'; '.join(details)})"
    )


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


def _cuda_inventory() -> tuple[bool, list[str], list[float]]:
    if not torch.cuda.is_available():
        return False, [], []

    # Before initialization, NVML may report more GPUs than the CUDA runtime
    # can enumerate under MIG. Do not capture that count in a range first.
    torch.cuda.init()
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("CUDA initialized but reported no visible devices")
    properties = [torch.cuda.get_device_properties(index) for index in range(count)]
    return (
        True,
        [device.name for device in properties],
        [round(device.total_memory / (1024**3), 2) for device in properties],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ASGCN-U-Net server readiness")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-eventhdr-train", action="store_true")
    parser.add_argument("--require-eventhdr-eval", action="store_true")
    parser.add_argument("--require-eventaid-all", action="store_true")
    parser.add_argument("--require-full-data", action="store_true")
    parser.add_argument("--lock", type=Path, default=None)
    parser.add_argument(
        "--include-private-host-provenance",
        action="store_true",
        help=(
            "PRIVATE: include hostname and exact host paths for local diagnostics; "
            "do not publish or attach this output"
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_root = (args.data_root or project_root / "data").resolve()
    runs_root = (args.runs_root or project_root / "runs").resolve()
    lock_path = args.lock.resolve() if args.lock else None
    path_replacements = [
        (data_root, "$DATA_ROOT"),
        (runs_root, "$RUNS_ROOT"),
        (project_root, "$PROJECT_ROOT"),
    ]
    if lock_path is not None:
        path_replacements.append((lock_path, "$LOCK_FILE"))
    try:
        runs_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        message = _host_error(
            error,
            path_replacements,
            args.include_private_host_provenance,
        )
        raise SystemExit(f"Cannot create $RUNS_ROOT: {message}") from None

    try:
        cuda_available, devices, gpu_memory_gib = _cuda_inventory()
    except (AssertionError, RuntimeError, OSError, DeferredCudaCallError) as error:
        # Deferred CUDA failures can embed an original traceback containing
        # private paths. Routine reports expose the exception type, not its text.
        detail = type(error).__name__
        if args.include_private_host_provenance:
            detail = f"{detail}: {error}"
        raise SystemExit(
            f"CUDA device probe failed ({detail}). "
            "Check the GPU allocation, driver/PyTorch CUDA compatibility and "
            "scheduler-provided device visibility. Restart Python after changes; "
            "do not bypass the CUDA requirement."
        ) from None
    try:
        lock_mismatches = _check_lock(lock_path) if lock_path and lock_path.is_file() else None
    except (OSError, TypeError, ValueError) as error:
        message = _host_error(
            error,
            path_replacements,
            args.include_private_host_provenance,
        )
        raise SystemExit(f"Dependency lock check failed: {message}") from None
    lock_python_match = None
    if lock_path:
        match = re.fullmatch(r"py(\d)(\d+)", lock_path.stem)
        if match:
            expected_python = f"{match.group(1)}.{match.group(2)}"
            lock_python_match = platform.python_version().startswith(f"{expected_python}.")
    try:
        train_files = _eventhdr_files(data_root / "EventHDR" / "train")
        eval_files = _eventhdr_files(data_root / "EventHDR" / "eval")
        data_disk = shutil.disk_usage(data_root if data_root.exists() else project_root)
        runs_disk = shutil.disk_usage(runs_root)
    except OSError as error:
        message = _host_error(
            error,
            path_replacements,
            args.include_private_host_provenance,
        )
        raise SystemExit(f"Environment filesystem check failed: {message}") from None
    train_root = data_root / "EventHDR" / "train"
    train_present = {path.relative_to(train_root).as_posix() for path in train_files}
    eval_root = data_root / "EventHDR" / "eval"
    eval_present = {path.relative_to(eval_root).as_posix() for path in eval_files}
    libc_name, libc_version = platform.libc_ver()
    report = {
        "project_root": "$PROJECT_ROOT",
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
        "data_root": "$DATA_ROOT",
        "eventhdr_train_h5": len(train_files),
        "eventhdr_eval_h5": len(eval_files),
        "eventaid_r_zip": _count_files(data_root / "EventAid-R", "R-*.zip"),
        "runs_root": "$RUNS_ROOT",
        "runs_writable": os.access(runs_root, os.W_OK),
        "data_disk_free_gib": round(data_disk.free / (1024**3), 2),
        "runs_disk_free_gib": round(runs_disk.free / (1024**3), 2),
        "lock_file": _logical_path(lock_path, project_root, "$LOCK_FILE"),
        "constraint_versions_match": (not lock_mismatches if lock_mismatches is not None else None),
        "constraint_python_match": lock_python_match,
        "lock_mismatches": lock_mismatches,
    }
    if args.include_private_host_provenance:
        report["private_host_provenance"] = {
            "hostname": socket.gethostname(),
            "project_root": str(project_root),
            "data_root": str(data_root),
            "runs_root": str(runs_root),
            "lock_file": str(lock_path) if lock_path else None,
            "publication_warning": "private local diagnostics; do not publish",
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    problems: list[str] = []
    if not report["runs_writable"]:
        problems.append("$RUNS_ROOT is not writable")
    if args.require_cuda and not cuda_available:
        problems.append("CUDA was required but torch.cuda.is_available() is false")
    if lock_path and not lock_path.is_file():
        problems.append(f"Dependency lock does not exist: {report['lock_file']}")
    elif lock_mismatches:
        problems.append(f"Installed packages differ from dependency lock: {report['lock_file']}")
    if lock_python_match is False:
        problems.append(f"Python version does not match dependency profile: {report['lock_file']}")
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
    if require_eventhdr_train:
        problem = _exact_coverage_problem(
            "eventhdr_train_h5", train_present, _OFFICIAL_EVENTHDR_TRAIN
        )
        if problem:
            problems.append(problem)
    if require_eventhdr_eval:
        problem = _exact_coverage_problem(
            "eventhdr_eval_h5", eval_present, _OFFICIAL_EVENTHDR_EVAL
        )
        if problem:
            problems.append(problem)
    if require_eventhdr_train:
        manifest_path = project_root / "manifests" / "eventhdr_split.json"
        if not manifest_path.is_file():
            problems.append(
                "Official EventHDR split manifest is missing: "
                "$PROJECT_ROOT/manifests/eventhdr_split.json"
            )
        else:
            try:
                manifest = load_eventhdr_split_manifest(manifest_path)
            except (OSError, TypeError, ValueError) as error:
                message = _host_error(
                    error,
                    path_replacements,
                    args.include_private_host_provenance,
                )
                problems.append(
                    f"EventHDR split manifest validation failed: {message}"
                )
            else:
                if manifest.get("status") != "final" or manifest.get("split_schema") != (
                    "official_separate_roots_v1"
                ):
                    problems.append(
                        "EventHDR training requires a final "
                        "official_separate_roots_v1 manifest"
                    )
                manifest_train = set(manifest.get("train_files", []))
                manifest_eval = set(manifest.get("val_files", []))
                if manifest_train != _OFFICIAL_EVENTHDR_TRAIN:
                    problems.append(
                        "EventHDR split manifest train root must declare exactly "
                        "1.h5 through 51.h5"
                    )
                if manifest_eval != _OFFICIAL_EVENTHDR_EVAL:
                    problems.append(
                        "EventHDR split manifest eval root must declare exactly "
                        "1.h5 through 19.h5"
                    )
    if require_eventaid_all:
        aid_manifest_path = project_root / "manifests" / "eventaid_r.json"
        if aid_manifest_path.is_file():
            try:
                aid_manifest = json.loads(aid_manifest_path.read_text(encoding="utf-8"))
                aid_root = data_root / "EventAid-R"
                aid_present = {path.name for path in aid_root.glob("R-*.zip")}
            except (OSError, TypeError, ValueError) as error:
                message = _host_error(
                    error,
                    path_replacements,
                    args.include_private_host_provenance,
                )
                problems.append(f"EventAid-R manifest validation failed: {message}")
            else:
                aid_required = {
                    f"{item['scene']}.zip"
                    for item in aid_manifest.get("files", [])
                    if isinstance(item, dict) and item.get("scene")
                }
                problem = _exact_coverage_problem("eventaid_r_zip", aid_present, aid_required)
                if problem:
                    problems.append(problem)
        else:
            problems.append(
                "EventAid-R manifest is missing: $PROJECT_ROOT/manifests/eventaid_r.json"
            )
    if problems:
        message = "; ".join(problems)
        if args.include_private_host_provenance:
            message = message.replace("$PROJECT_ROOT", str(project_root))
            message = message.replace("$DATA_ROOT", str(data_root))
            message = message.replace("$RUNS_ROOT", str(runs_root))
            if lock_path is not None:
                message = message.replace("$LOCK_FILE", str(lock_path))
        else:
            message = _redact_host_paths(message, path_replacements)
        raise SystemExit(message)


if __name__ == "__main__":
    main()
