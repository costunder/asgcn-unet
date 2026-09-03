"""Probe one evaluation sample without creating normal evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from asgcn_unet.sample_probe import probe_evaluation_sample, save_probe_result
from asgcn_unet.utils import load_json, resolve_experiment_paths, resolve_path


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def _nonnegative_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="evaluation dataset configuration")
    parser.add_argument("--checkpoint", required=True, help="ANN or calibrated SNN checkpoint")
    parser.add_argument("--sample-index", required=True, type=_nonnegative_integer)
    parser.add_argument(
        "--max-graph-edges",
        required=True,
        type=_positive_integer,
        help="runtime-only edge guard; cannot be below config.model.max_graph_edges",
    )
    parser.add_argument("--inference-mode", choices=("ann", "snn"), default="ann")
    parser.add_argument("--simulation-steps", type=_positive_integer, default=16)
    parser.add_argument(
        "--snn-dynamics",
        choices=("literal_eq15", "standard_if"),
        default=None,
    )
    parser.add_argument("--output", required=True, help="new diagnostic JSON path")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = resolve_experiment_paths(load_json(config_path), config_path)
    checkpoint = resolve_path(args.checkpoint, PROJECT)
    output = resolve_path(args.output, PROJECT)
    if output.exists():
        raise SystemExit(f"probe failed: Probe output already exists: {output}")
    try:
        result = probe_evaluation_sample(
            config,
            checkpoint,
            sample_index=args.sample_index,
            max_graph_edges=args.max_graph_edges,
            inference_mode=args.inference_mode,
            simulation_steps=args.simulation_steps,
            snn_dynamics=args.snn_dynamics,
        )
        save_probe_result(output, result)
    except (FileNotFoundError, IndexError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"probe failed: {error}") from None
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
