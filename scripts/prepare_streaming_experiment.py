"""Create an independent physical-time streaming study without executing it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from asgcn_unet.stream_preflight import prepare_streaming_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="new independent directory inside this checkout")
    parser.add_argument("--window-seconds", type=float, required=True)
    parser.add_argument("--time-scale-seconds", type=float, required=True)
    parser.add_argument("--hdr-timestamp-scale-to-seconds", type=float, required=True)
    parser.add_argument("--aid-timestamp-scale-to-seconds", type=float, required=True)
    parser.add_argument("--hdr-interval-timestamp-scale-to-seconds", type=float, required=True)
    parser.add_argument("--aid-interval-timestamp-scale-to-seconds", type=float, required=True,
                        help="separate frame-boundary scale; EventAid t0_us/t1_us fields are microseconds (1e-6)")
    args = parser.parse_args(argv)
    try:
        report = prepare_streaming_experiment(
            PROJECT, args.output_root, window_seconds=args.window_seconds,
            time_scale_seconds=args.time_scale_seconds,
            hdr_timestamp_scale_to_seconds=args.hdr_timestamp_scale_to_seconds,
            aid_timestamp_scale_to_seconds=args.aid_timestamp_scale_to_seconds,
            hdr_interval_timestamp_scale_to_seconds=args.hdr_interval_timestamp_scale_to_seconds,
            aid_interval_timestamp_scale_to_seconds=args.aid_interval_timestamp_scale_to_seconds,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Streaming experiment preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print("Only new configuration files were created. No GPU, training, evaluation, SSH, or server was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
