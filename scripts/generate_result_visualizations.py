"""Generate actual result PNGs and spatiotemporal graphs from completed evaluations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT / "src"))

from asgcn_unet.offline_viewer import ExportLimits
from asgcn_unet.result_visualization import (
    generate_result_visualizations,
    unique_output_directory,
)


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root", required=True, help="completed root containing aid/ and/or hdr/"
    )
    parser.add_argument("--output-dir", help="new output folder; default is a fresh unique sibling")
    parser.add_argument("--aid-config", default=str(PROJECT / "configs/aid-fast.json"))
    parser.add_argument("--hdr-config", default=str(PROJECT / "configs/hdr-fast.json"))
    parser.add_argument(
        "--cpu-threads",
        type=_positive,
        required=True,
        help="explicit allocated CPU thread count; checked before use",
    )
    parser.add_argument(
        "--memory-budget-mib",
        type=_positive,
        required=True,
        help="explicit incremental diagnostic planning budget, not an OS RAM limit",
    )
    parser.add_argument(
        "--reserve-memory-mib",
        type=_positive,
        required=True,
        help="free RAM that must remain beyond the planning budget",
    )
    parser.add_argument(
        "--display-edges",
        type=_positive,
        default=5000,
        help="drawn edges only; ALL graph nodes and exact topology statistics retained",
    )
    parser.add_argument(
        "--max-output-mib",
        type=_positive,
        default=256,
        help="HTML byte guard; exceeding it fails without reducing any input",
    )
    args = parser.parse_args(argv)
    try:
        root = Path(args.eval_root).expanduser()
        if not root.is_absolute():
            root = PROJECT / root
        output = (
            Path(args.output_dir).expanduser() if args.output_dir else unique_output_directory(root)
        )
        if not output.is_absolute():
            output = PROJECT / output
        print("Creating actual saved GT/prediction PNGs + actual event graphs.", flush=True)
        print(
            "No SSH/web server/GPU/model inference. Original evaluation files remain unchanged.",
            flush=True,
        )
        result = generate_result_visualizations(
            root,
            output,
            configs={"aid": args.aid_config, "hdr": args.hdr_config},
            memory_budget_bytes=args.memory_budget_mib * 1024 * 1024,
            reserve_memory_bytes=args.reserve_memory_mib * 1024 * 1024,
            cpu_threads=args.cpu_threads,
            display_edges=args.display_edges,
            limits=ExportLimits(max_output_bytes=args.max_output_mib * 1024 * 1024),
        )
        print(f"Completed: {result['generated_frames']} actual saved frames, PNGs and graphs.")
        print(f"Offline result: {output / 'index.html'}")
        print(f"PNG / graph files: {output}")
        print("Open the generated index.html locally; no running server or SSH tunnel is needed.")
        return 0
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        MemoryError,
    ) as error:
        print(f"Visualization generation failed: {error}", file=sys.stderr)
        print(
            "Existing experiments and source files were not modified. No graph-free result "
            "was substituted. If created, the new output folder contains a failure note.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
