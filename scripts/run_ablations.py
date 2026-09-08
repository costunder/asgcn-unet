"""Plan or explicitly execute the full architecture-comparison experiment suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from asgcn_unet.ablation_comparison import render_a_e, render_bcd_t4
from asgcn_unet.ablation_suite import (
    EXECUTION_PROFILES,
    FAMILIES,
    STAGES,
    collect_summary,
    execute_commands,
    get_execution_profile,
    plan_commands,
    render_summary,
)


def _print_results(args: argparse.Namespace, families: tuple[str, ...]) -> None:
    options = {} if args.profile == "full" else {"profile": args.profile}
    rows = collect_summary(PROJECT, families, **options)
    if args.profile == "full":
        print(render_a_e(rows))
    else:
        print(render_bcd_t4(rows))
    if args.details:
        print("\n## Selected experiments and stored provenance\n")
        print(render_summary(rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="plan")
    parser.add_argument(
        "--profile", choices=tuple(EXECUTION_PROFILES), default="full",
        help="full keeps the complete sweep; bcd-throughput-t4 runs B/C/D ANN and both T=4 dynamics only",
    )
    parser.add_argument(
        "--details", action="store_true",
        help="Append all selected rows, macro metrics and stored provenance after the compact table",
    )
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
    parser.add_argument("--families", nargs="+", choices=tuple(FAMILIES), default=None)
    parser.add_argument(
        "--cpu-threads", type=int, default=4, help="CPU helpers for existing full CUDA preflight"
    )
    args = parser.parse_args(argv)
    try:
        profile = get_execution_profile(args.profile)
        families = tuple(args.families) if args.families is not None else profile.families
        if args.stage == "summary":
            _print_results(args, families)
            return 0
        if args.stage == "plan" and args.execute:
            raise ValueError("plan never executes; choose --stage all (or one stage) explicitly")
        commands = plan_commands(
            PROJECT, stage=args.stage, families=families, cpu_threads=args.cpu_threads,
            profile=args.profile,
        )
        print(f"Execution profile: {profile.name}; selected families: {', '.join(families)}")
        if profile.name == "full":
            print("Primary: A (U-Net only) vs E (Transformer only); no GNN/SNN in either.")
            print("Supplementary: B (+ANN control), C/D. Old graph_transformer is not E.")
        else:
            print("B/C/D throughput-first: ANN controls and T=4 with both SNN dynamics.")
            print("A/E are not scheduled. T=8/16/32 are deferred, not completed or deleted.")
            print("T is an inference setting: B and C/D each share one ANN training across their modes.")
        print(
            "Full existing data, resolution, 40 epochs and physical batch 16 retained."
        )
        if any(FAMILIES[family][0] != "identity" for family in families):
            print(f"Selected SNN steps: {profile.simulation_steps}; dynamics: {profile.dynamics}")
        else:
            print("A/E are ANN-only: no SNN calibration or T sweep.")
        counts = Counter(command.stage for command in commands)
        print("Planned stages: " + ", ".join(f"{stage}={count}" for stage, count in counts.items()))
        print("Existing train AMP and measured batching remain enabled; evaluation remains FP32 with TF32 off.")
        print("Higher FPS is not guaranteed; benchmark values are compute-only single-frame latency references.")
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
            _print_results(args, families)
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
