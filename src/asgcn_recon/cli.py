from __future__ import annotations

import argparse
import json
from typing import Any

from tqdm import tqdm

from .data import build_dataset
from .engine import benchmark, calibrate, evaluate, train
from .utils import experiment_base_dir, load_json, resolve_experiment_paths, resolve_path


def _inspect_one_split(dataset: Any, samples: int, validate_all: bool = False) -> dict[str, Any]:
    details = []
    preview_count = min(samples, len(dataset))
    count = len(dataset) if validate_all else preview_count
    indices = tqdm(range(count), desc="validate-data", disable=not validate_all)
    for index in indices:
        item = dataset[index]
        if index < preview_count:
            details.append(
                {
                    "sample_id": item["sample_id"],
                    "events": int(item["events"].shape[0]),
                    "target_shape": list(item["target"].shape),
                    "sensor_size": list(item["sensor_size"]),
                    "metadata": item["metadata"],
                }
            )
    result: dict[str, Any] = {
        "samples": len(dataset),
        "validated_samples": count,
        "validation_complete": count == len(dataset),
        "preview": details,
    }
    if hasattr(dataset, "scene_info"):
        result["scenes"] = dataset.scene_info
    if hasattr(dataset, "files"):
        result["files"] = len(dataset.files)
    return result


def inspect_dataset(
    config: dict[str, Any], samples: int = 3, validate_all: bool = False
) -> dict[str, Any]:
    if samples < 0:
        raise ValueError("inspect samples must be non-negative")
    data_config = config["dataset"]
    result: dict[str, Any] = {
        "dataset_type": data_config["type"],
        "root": data_config["root"],
    }
    if data_config["type"] == "eventhdr" and data_config.get("split_manifest"):
        split_details: dict[str, Any] = {}
        for split in ("train", "val"):
            dataset = build_dataset(data_config, split=split)
            try:
                split_details[split] = _inspect_one_split(dataset, samples, validate_all)
            finally:
                if hasattr(dataset, "close"):
                    dataset.close()
        result["splits"] = split_details
        result["samples"] = sum(detail["samples"] for detail in split_details.values())
        return result

    dataset = build_dataset(data_config, split="eval")
    try:
        result.update(_inspect_one_split(dataset, samples, validate_all))
    finally:
        if hasattr(dataset, "close"):
            dataset.close()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASGCN Event-to-Frame experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = subparsers.add_parser("inspect", help="validate dataset structure")
    inspect_cmd.add_argument("--config", required=True)
    inspect_cmd.add_argument("--samples", type=int, default=3)
    inspect_cmd.add_argument(
        "--validate-all",
        action="store_true",
        help="decode and validate every selected sample while keeping only the preview",
    )

    train_cmd = subparsers.add_parser("train", help="train on EventHDR")
    train_cmd.add_argument("--config", required=True)
    train_cmd.add_argument(
        "--resume",
        help="resume from a checkpoint (relative paths are resolved from the repository root)",
    )

    eval_cmd = subparsers.add_parser("evaluate", help="evaluate quality and latency")
    eval_cmd.add_argument("--config", required=True)
    eval_cmd.add_argument("--checkpoint", required=True)
    eval_cmd.add_argument("--inference-mode", choices=["ann", "snn"], default="ann")
    eval_cmd.add_argument("--simulation-steps", type=int, default=16)
    eval_cmd.add_argument(
        "--snn-dynamics",
        choices=["literal_eq15", "standard_if"],
        default=None,
        help="inference-only override; the checkpoint architecture remains unchanged",
    )

    bench_cmd = subparsers.add_parser("benchmark", help="benchmark compute-only latency")
    bench_cmd.add_argument("--config", required=True)
    bench_cmd.add_argument("--checkpoint", required=True)
    bench_cmd.add_argument("--warmup", type=int, default=10)
    bench_cmd.add_argument("--steps", type=int, default=100)
    bench_cmd.add_argument("--inference-mode", choices=["ann", "snn"], default="ann")
    bench_cmd.add_argument("--simulation-steps", type=int, default=16)
    bench_cmd.add_argument(
        "--snn-dynamics",
        choices=["literal_eq15", "standard_if"],
        default=None,
    )

    calibrate_cmd = subparsers.add_parser("calibrate", help="calibrate ANN-to-SNN thresholds")
    calibrate_cmd.add_argument("--config", required=True)
    calibrate_cmd.add_argument("--checkpoint", required=True)
    calibrate_cmd.add_argument("--output", required=True)
    calibrate_cmd.add_argument("--samples", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = resolve_path(args.config, ".")
    config = resolve_experiment_paths(load_json(config_path), config_path)
    base_dir = experiment_base_dir(config_path)
    if args.command == "inspect":
        result = inspect_dataset(config, args.samples, args.validate_all)
    elif args.command == "train":
        resume = resolve_path(args.resume, base_dir) if args.resume else None
        result = {"best_checkpoint": str(train(config, resume_from=resume))}
    elif args.command == "evaluate":
        result = evaluate(
            config,
            resolve_path(args.checkpoint, base_dir),
            inference_mode=args.inference_mode,
            simulation_steps=args.simulation_steps,
            snn_dynamics=args.snn_dynamics,
        )
    elif args.command == "benchmark":
        result = benchmark(
            config,
            resolve_path(args.checkpoint, base_dir),
            warmup=args.warmup,
            steps=args.steps,
            inference_mode=args.inference_mode,
            simulation_steps=args.simulation_steps,
            snn_dynamics=args.snn_dynamics,
        )
    elif args.command == "calibrate":
        result = {
            "calibrated_checkpoint": str(
                calibrate(
                    config,
                    resolve_path(args.checkpoint, base_dir),
                    resolve_path(args.output, base_dir),
                    samples=args.samples,
                )
            )
        }
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
