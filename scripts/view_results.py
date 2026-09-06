"""View saved GT/prediction PNGs and CPU-reconstructed event graphs without evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT / "src"))

from asgcn_unet.result_viewer import ResultViewer, ViewerHTTPServer


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, help="completed root containing aid/ and hdr/")
    parser.add_argument("--aid-config", default=str(PROJECT / "configs/aid-fast.json"))
    parser.add_argument("--hdr-config", default=str(PROJECT / "configs/hdr-fast.json"))
    parser.add_argument("--port", type=_positive, default=8765)
    parser.add_argument("--cpu-threads", type=_positive, default=4,
                        help="CPU diagnostic threads; never selects or initializes CUDA")
    parser.add_argument("--display-edges", type=_positive, default=5000,
                        help="maximum drawn edges only; full model graph is preserved")
    parser.add_argument("--check", action="store_true", help="print saved-image catalog without serving")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535; no privileged bind")
    viewer = None
    try:
        import torch

        torch.set_num_threads(args.cpu_threads)
        root = Path(args.eval_root).expanduser()
        if not root.is_absolute():
            root = PROJECT / root
        viewer = ResultViewer(root, configs={"aid": args.aid_config, "hdr": args.hdr_config},
                              display_edges=args.display_edges)
        catalog = viewer.catalog()
        if args.check:
            print(json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        with ViewerHTTPServer(viewer, args.port) as server:
            print("Read-only viewer. No training, inference, CUDA selection or evaluation/data writes.")
            print(f"CPU graph diagnostics: {args.cpu_threads} threads; one graph cached in RAM.")
            for data in catalog["datasets"]:
                print(f"{data['label']}: {len(data['frames'])} saved image frames / "
                      f"{data['total_frames']} evaluated frames")
            for warning in catalog["warnings"]:
                print("WARNING: " + warning, file=sys.stderr)
            print("Use an SSH local forward when viewing a remote server; bind is loopback only.")
            print("Open this complete private URL in your browser:\n" + server.url, flush=True)
            print("Ctrl+C stops only this viewer process; the SSH session remains open.", flush=True)
            try:
                server.serve_forever(poll_interval=0.25)
            except KeyboardInterrupt:
                print("Viewer stopped; evaluation files were not changed.")
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"viewer failed: {error}", file=sys.stderr)
        return 1
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
