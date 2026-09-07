"""Export saved reconstruction PNGs and metrics into one offline HTML file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT / "src"))

from asgcn_unet.offline_viewer import (
    ExportLimits,
    empty_payload,
    export_results_html,
    render_payload_html,
)


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--eval-root", help="completed root containing aid/ and/or hdr/")
    source.add_argument("--empty", action="store_true", help="explicit data-absent template")
    parser.add_argument("--output", required=True, help="new HTML path; never overwrite")
    parser.add_argument("--title", default="ASGCN-U-Net offline results")
    parser.add_argument("--graph-json", help="existing asgcn_offline_graphs_v1 JSON; never build")
    parser.add_argument("--max-input-mib", type=_positive, default=2048)
    parser.add_argument("--max-output-mib", type=_positive, default=256)
    parser.add_argument("--max-metadata-mib", type=_positive, default=8)
    parser.add_argument("--max-graph-mib", type=_positive, default=16)
    parser.add_argument("--max-png-mib", type=_positive, default=32)
    parser.add_argument("--max-decoded-png-mib", type=_positive, default=128)
    args = parser.parse_args(argv)
    if args.empty and args.graph_json:
        parser.error("--graph-json requires --eval-root with matching saved frame identities")
    limits = ExportLimits(**{
        f"max_{name}_bytes": getattr(args, f"max_{name}_mib") * 1024 * 1024
        for name in ("input", "output", "metadata", "graph", "png", "decoded_png")
    })
    try:
        print("Offline export: no SSH, server, CUDA, model loading, inference, or graph construction.",
              file=sys.stderr, flush=True)
        print("Byte guards reject the whole export; saved images are never resized/subsetted.",
              file=sys.stderr, flush=True)
        if args.empty:
            payload = empty_payload(args.title)
            payload["notes"].insert(0, "EXPLICIT EMPTY TEMPLATE: no real images or graph data included.")
            result = render_payload_html(payload, args.output, limits=limits)
        else:
            result = export_results_html(args.eval_root, args.output, title=args.title,
                                         graph_json=args.graph_json, limits=limits)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as error:
        print(f"offline export failed: {error}. Existing inputs/results were not changed.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
