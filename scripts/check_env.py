from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import sys
from pathlib import Path

import torch


def _count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.glob(pattern)) if root.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ASGCN server readiness")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_root = (args.data_root or project_root / "data").resolve()
    runs_root = (args.runs_root or project_root / "runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(data_root if data_root.exists() else project_root)

    cuda_available = torch.cuda.is_available()
    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    report = {
        "project_root": str(project_root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_devices": devices,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "data_root": str(data_root),
        "eventhdr_train_h5": _count_files(data_root / "EventHDR" / "train", "*.h5"),
        "eventhdr_eval_h5": _count_files(data_root / "EventHDR" / "eval", "*.h5"),
        "eventaid_r_zip": _count_files(data_root / "EventAid-R", "R-*.zip"),
        "runs_root": str(runs_root),
        "runs_writable": os.access(runs_root, os.W_OK),
        "disk_free_gib": round(disk.free / (1024**3), 2),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    problems: list[str] = []
    if not report["runs_writable"]:
        problems.append(f"Run directory is not writable: {runs_root}")
    if args.require_cuda and not cuda_available:
        problems.append("CUDA was required but torch.cuda.is_available() is false")
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
