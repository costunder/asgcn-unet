"""Plan or explicitly execute the full architecture-comparison experiment suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from asgcn_unet.ablation_suite import (
    FAMILIES,
    STAGES,
    collect_summary,
    execute_commands,
    plan_commands,
    render_summary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="plan")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly run expensive stages; otherwise print commands only",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify completed artifacts and resume last.pt; never archive/overwrite incomplete results",
    )
    parser.add_argument("--families", nargs="+", choices=tuple(FAMILIES), default=list(FAMILIES))
    parser.add_argument(
        "--cpu-threads", type=int, default=4, help="CPU helpers for existing full CUDA preflight"
    )
    args = parser.parse_args(argv)
    try:
        if args.stage == "summary":
            print(render_summary(collect_summary(PROJECT, tuple(args.families))))
            return 0
        if args.stage == "plan" and args.execute:
            raise ValueError("plan never executes; choose --stage all (or one stage) explicitly")
        commands = plan_commands(
            PROJECT, stage=args.stage, families=tuple(args.families), cpu_threads=args.cpu_threads
        )
        print("Architecture ablation suite: A / B(+ANN control) / C+D / E(+ANN control).")
        print(
            "Full existing data, resolution, 40 epochs and physical batch 16 retained; all T and both dynamics retained."
        )
        print(
            "No GPU is selected or mask changed. Concurrent studies on the same allocation are not launched."
        )
        if not args.execute:
            print("PLAN ONLY: no data loading, CUDA, training, evaluation, or output writes.")
        execute_commands(
            commands,
            PROJECT,
            execute=args.execute,
            resume=args.resume,
            cpu_threads=args.cpu_threads,
        )
        if args.execute and args.stage in {"all", "eval"}:
            print(render_summary(collect_summary(PROJECT, tuple(args.families))))
        return 0
    except (ValueError, TypeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Ablation stage stopped: {error}", file=sys.stderr)
        print(
            "Existing runs and sessions were not deleted or terminated. Inspect the failed stage before resuming.",
            file=sys.stderr,
        )
        return error.returncode if isinstance(error, subprocess.CalledProcessError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
