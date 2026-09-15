"""Export saved raw graph coordinates/queries as one offline HTML, without inference."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from asgcn_unet.graph_inspection_view import export_saved_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--cpu-threads", required=True, type=int)
    parser.add_argument("--memory-budget-mib", required=True, type=int)
    parser.add_argument("--reserve-memory-mib", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        runs = PROJECT / "runs"
        if runs.resolve() != runs:
            raise ValueError("Output runs directory must remain inside this checkout")
        runs.mkdir(exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix="graph-view-", dir=runs)) / "graph.html"
        output = export_saved_report(args.report, destination, workspace=PROJECT,
                                     memory_budget_bytes=args.memory_budget_mib * 1024**2,
                                     reserve_memory_bytes=args.reserve_memory_mib * 1024**2,
                                     cpu_threads=args.cpu_threads)
        print(f"Offline graph saved: {output}")
        print("Open this single HTML locally. No web server, SSH tunnel, GPU or model is needed.")
        print("Old reports show only their saved query neighborhoods; missing nodes are not invented.")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, MemoryError) as error:
        print(f"Graph export failed: {error}", file=sys.stderr)
        print("Source reports and existing experiments unchanged. No job was stopped.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
