"""Validate whether both halves of one evaluation mode are safe to reuse."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from asgcn_unet.artifact_lock import (
    ArtifactWriterBusyError,
    ArtifactWriterOwnershipError,
    exclusive_artifact_writer,
)
from asgcn_unet.data.factory import build_dataset
from asgcn_unet.engine import (
    _canonical_sha256,
    _current_source_contract,
    _dataset_content_fingerprint,
    _dataset_coverage_summary,
    _evaluation_dataset_transform_contract,
    _evaluation_manifest_contract,
    _evaluation_runtime_contract,
    _hashed_contract,
    _inference_precision,
    _load_data_hash_cache,
    _public_config,
    _reporting_checkpoint_contract,
    load_model_checkpoint,
)
from asgcn_unet.utils import (
    experiment_base_dir,
    load_json,
    resolve_device,
    resolve_experiment_paths,
    resolve_path,
)


class ArtifactMismatch(ValueError):
    """A completed-looking artifact does not match the requested recovery contract."""


def _regular_nonempty(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and path.stat().st_size > 0


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactMismatch(f"{path.name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ArtifactMismatch(f"{path.name} must contain a JSON object")
    return value


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ArtifactMismatch(
            f"{name} mismatch: existing={actual!r}, requested={expected!r}"
        )


def _validate_hashed_contract(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("contract"), dict):
        raise ArtifactMismatch(f"{name} is not a hashed contract")
    try:
        expected_sha256 = _canonical_sha256(value["contract"])
    except (TypeError, ValueError, OverflowError) as error:
        raise ArtifactMismatch(f"{name} contract is invalid") from error
    _require_equal(f"{name}.sha256", value.get("sha256"), expected_sha256)
    return value["contract"]


def _validate_protocol_commitments(name: str, protocol: dict[str, Any]) -> None:
    committed = dict(protocol)
    recorded_sha256 = committed.pop("protocol_sha256", None)
    try:
        expected_sha256 = _canonical_sha256(committed)
    except (TypeError, ValueError, OverflowError) as error:
        raise ArtifactMismatch(f"{name} is not canonical JSON") from error
    _require_equal(f"{name}.protocol_sha256", recorded_sha256, expected_sha256)
    for field in ("public_config", "model_config", "execution", "source", "runtime", "precision"):
        _validate_hashed_contract(f"{name}.{field}", protocol.get(field))
    _validate_hashed_contract(
        f"{name}.evaluation_dataset", protocol.get("evaluation_dataset")
    )


def _current_dataset_common(config: dict[str, Any]) -> dict[str, Any]:
    dataset = build_dataset(config["dataset"], split="eval")
    try:
        eval_config = config.get("eval", {})
        output_base = Path(eval_config.get("output_dir", "runs/evaluation"))
        digest_cache = _load_data_hash_cache(
            output_base / ".data_hash_cache.json",
            bool(eval_config.get("rehash_data", False)),
        )
        return {
            "content": _dataset_content_fingerprint(dataset, digest_cache),
            "transform": _evaluation_dataset_transform_contract(config),
            "manifest": _evaluation_manifest_contract(config),
            "coverage": _dataset_coverage_summary(dataset, config["dataset"]),
        }
    finally:
        if hasattr(dataset, "close"):
            dataset.close()


def _current_runtime_contract(config: dict[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    device = resolve_device(config.get("device", "auto"))
    precision, _ = _inference_precision(config.get("eval", {}), device, model)
    return _evaluation_runtime_contract(device, precision)


def _validate_common(
    value: dict[str, Any],
    *,
    protocol_key: str,
    protocol_kind: str,
    public_config: dict[str, Any],
    checkpoint_sha256: str,
    source_tree_sha256: str,
    current_runtime: dict[str, Any],
    mode: str,
    simulation_steps: int,
    dynamics: str | None,
    configured_guard: int | None,
    requested_guard: int | None,
    dataset_common: dict[str, Any],
) -> dict[str, Any]:
    _require_equal("report_eligible", value.get("report_eligible"), True)
    _require_equal("report_ineligible_reasons", value.get("report_ineligible_reasons"), [])
    protocol = value.get(protocol_key)
    if not isinstance(protocol, dict):
        raise ArtifactMismatch(f"missing {protocol_key}")
    _validate_protocol_commitments(protocol_key, protocol)
    _require_equal(f"{protocol_key}.kind", protocol.get("kind"), protocol_kind)
    _require_equal(f"{protocol_key}.report_eligible", protocol.get("report_eligible"), True)
    _require_equal(
        f"{protocol_key}.report_ineligible_reasons",
        protocol.get("report_ineligible_reasons"),
        [],
    )
    _require_equal(
        f"{protocol_key}.public_config",
        protocol.get("public_config"),
        public_config,
    )
    checkpoint = protocol.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ArtifactMismatch(f"missing {protocol_key}.checkpoint")
    _require_equal(
        f"{protocol_key}.checkpoint.checkpoint_file_sha256",
        checkpoint.get("checkpoint_file_sha256"),
        checkpoint_sha256,
    )
    source = protocol.get("source", {}).get("contract", {})
    _require_equal(
        f"{protocol_key}.source.source_tree_sha256",
        source.get("source_tree_sha256"),
        source_tree_sha256,
    )
    _require_equal(
        f"{protocol_key}.runtime",
        protocol["runtime"]["contract"],
        current_runtime,
    )
    execution = protocol.get("execution", {}).get("contract")
    if not isinstance(execution, dict):
        raise ArtifactMismatch(f"missing {protocol_key}.execution contract")
    expected_steps = simulation_steps if mode == "snn" else None
    expected_dynamics = dynamics if mode == "snn" else None
    _require_equal(f"{protocol_key}.inference_mode", execution.get("inference_mode"), mode)
    _require_equal(
        f"{protocol_key}.simulation_steps",
        execution.get("simulation_steps"),
        expected_steps,
    )
    _require_equal(
        f"{protocol_key}.snn_dynamics",
        execution.get("snn_dynamics"),
        expected_dynamics,
    )
    expected_guard = {
        "configured_max_graph_edges": configured_guard,
        "requested_max_graph_edges_override": requested_guard,
        "effective_max_graph_edges": (
            requested_guard if requested_guard is not None else configured_guard
        ),
    }
    _require_equal(
        f"{protocol_key}.graph_edge_guard",
        execution.get("graph_edge_guard"),
        expected_guard,
    )
    evaluation_dataset = protocol["evaluation_dataset"]["contract"]
    for field, expected_value in dataset_common.items():
        _require_equal(
            f"{protocol_key}.evaluation_dataset.{field}",
            evaluation_dataset.get(field),
            expected_value,
        )
    return execution


def _validate_quality(
    metrics_path: Path,
    frames_path: Path,
    **expected: Any,
) -> None:
    metrics = _load_object(metrics_path)
    _validate_common(
        metrics,
        protocol_key="evaluation_protocol",
        protocol_kind="quality_evaluation",
        **expected,
    )
    quality = metrics.get("quality")
    if not isinstance(quality, dict):
        raise ArtifactMismatch("metrics.json is missing quality")
    frames = quality.get("frames")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ArtifactMismatch("metrics.json has an invalid quality frame count")
    try:
        with frames_path.open("r", encoding="utf-8", newline="") as handle:
            csv_frames = sum(1 for _ in csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise ArtifactMismatch("frames.csv is not a valid readable CSV") from error
    _require_equal("frames.csv row count", csv_frames, frames)


def _validate_benchmark(
    benchmark_path: Path,
    *,
    benchmark_warmup: int,
    benchmark_steps: int,
    **expected: Any,
) -> None:
    benchmark = _load_object(benchmark_path)
    execution = _validate_common(
        benchmark,
        protocol_key="benchmark_protocol",
        protocol_kind="compute_benchmark",
        **expected,
    )
    _require_equal("benchmark warmup", execution.get("warmup_steps"), benchmark_warmup)
    _require_equal("benchmark measured steps", execution.get("measured_steps"), benchmark_steps)
    _require_equal("benchmark frame count", benchmark.get("frames"), benchmark_steps)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inference-mode", choices=("ann", "snn"), required=True)
    parser.add_argument("--simulation-steps", type=int, required=True)
    parser.add_argument("--snn-dynamics", choices=("literal_eq15", "standard_if"))
    parser.add_argument("--max-graph-edges", type=int)
    parser.add_argument("--benchmark-warmup", type=int, required=True)
    parser.add_argument("--benchmark-steps", type=int, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--preserve-incomplete", action="store_true")
    return parser


def _requested_graph_guard(config: dict[str, Any], explicit: int | None) -> int | None:
    """Resolve the same explicit-over-config inference-only guard as evaluation."""
    requested = explicit
    if requested is None:
        requested = config.get("eval", {}).get("max_graph_edges_override")
    if requested is not None and (type(requested) is not int or requested < 1):
        raise ArtifactMismatch("max graph edges override must be a positive integer")
    return requested


def inspect_mode(args: argparse.Namespace) -> tuple[int, int]:
    if args.simulation_steps < 1:
        raise ArtifactMismatch("simulation steps must be positive")
    if args.max_graph_edges is not None and args.max_graph_edges < 1:
        raise ArtifactMismatch("max graph edges must be positive")
    if args.benchmark_warmup < 0 or args.benchmark_steps < 1:
        raise ArtifactMismatch("benchmark warmup/steps are invalid")
    config_path = resolve_path(args.config, Path.cwd())
    base_dir = experiment_base_dir(config_path)
    config = resolve_experiment_paths(load_json(config_path), config_path)
    requested_guard = _requested_graph_guard(config, args.max_graph_edges)
    output_dir = resolve_path(args.output_dir, base_dir)
    config["eval"]["output_dir"] = str(output_dir)
    checkpoint_path = resolve_path(args.checkpoint, base_dir)
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise ArtifactMismatch("checkpoint is not a regular file")

    if args.inference_mode == "snn" and args.snn_dynamics is None:
        raise ArtifactMismatch("SNN recovery requires --snn-dynamics")
    dynamics = args.snn_dynamics if args.inference_mode == "snn" else None
    run_label = (
        "ann"
        if args.inference_mode == "ann"
        else f"snn_{dynamics}_T{args.simulation_steps}"
    )
    run_dir = output_dir / run_label
    mode_dir_is_symlink = run_dir.is_symlink()
    metrics_path = run_dir / "metrics.json"
    frames_path = run_dir / "frames.csv"
    benchmark_path = run_dir / "benchmark.json"

    quality_complete = (
        not mode_dir_is_symlink
        and _regular_nonempty(metrics_path)
        and _regular_nonempty(frames_path)
    )
    benchmark_complete = not mode_dir_is_symlink and _regular_nonempty(benchmark_path)
    try:
        model, checkpoint = load_model_checkpoint(
            checkpoint_path,
            torch.device("cpu"),
            config["model"],
        )
        checkpoint_contract, report_eligible, report_ineligible_reasons = (
            _reporting_checkpoint_contract(
                checkpoint,
                checkpoint_path,
                args.inference_mode,
                allow_unsealed_checkpoint_for_non_reporting=False,
            )
        )
    except Exception as error:
        raise ArtifactMismatch("checkpoint validation failed") from error
    configured_guard = model.max_graph_edges
    if requested_guard is not None and (
        configured_guard is None or requested_guard < configured_guard
    ):
        raise ArtifactMismatch(
            "max graph edges must not lower the configured finite guard"
        )
    _require_equal("checkpoint report eligibility", report_eligible, True)
    _require_equal("checkpoint report ineligibility reasons", report_ineligible_reasons, [])
    current_runtime = _current_runtime_contract(config, model)
    if args.require_cuda and current_runtime.get("device_type") != "cuda":
        raise ArtifactMismatch("recovery requires a CUDA runtime")
    dataset_common = (
        _current_dataset_common(config)
        if quality_complete or benchmark_complete
        else {}
    )
    expected = {
        "public_config": _hashed_contract(_public_config(config)),
        "checkpoint_sha256": checkpoint_contract["checkpoint_file_sha256"],
        "source_tree_sha256": _current_source_contract()["source_tree_sha256"],
        "current_runtime": current_runtime,
        "mode": args.inference_mode,
        "simulation_steps": args.simulation_steps,
        "dynamics": dynamics,
        "configured_guard": configured_guard,
        "requested_guard": requested_guard,
        "dataset_common": dataset_common,
    }
    if quality_complete:
        _validate_quality(metrics_path, frames_path, **expected)
    if benchmark_complete:
        _validate_benchmark(
            benchmark_path,
            benchmark_warmup=args.benchmark_warmup,
            benchmark_steps=args.benchmark_steps,
            **expected,
        )
    return int(quality_complete), int(benchmark_complete)


def _preserve_incomplete_artifact(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.parent / f"{path.name}.incomplete-{stamp}-{os.getpid()}-{uuid4().hex[:8]}"
    try:
        # Reserve a new container exclusively: rename can never replace an old backup.
        backup.mkdir(mode=0o700)
        destination = backup / path.name
        if destination.exists() or destination.is_symlink():
            raise ArtifactMismatch("recovery backup destination unexpectedly exists")
        path.rename(destination)
    except OSError as error:
        raise ArtifactMismatch(f"could not preserve incomplete artifact: {path.name}") from error
    print(f"[eval-resume] preserved incomplete artifact: {backup.name}/{path.name}", file=sys.stderr)


def prepare_mode(args: argparse.Namespace) -> tuple[int, int]:
    """Inspect and preserve incomplete output while holding the actual writer lock."""
    config_path = resolve_path(args.config, Path.cwd())
    output_dir = resolve_path(args.output_dir, experiment_base_dir(config_path))
    run_label = (
        "ann" if args.inference_mode == "ann"
        else f"snn_{args.snn_dynamics}_T{args.simulation_steps}"
    )
    run_dir = output_dir / run_label
    with exclusive_artifact_writer(run_dir):
        quality_complete, benchmark_complete = inspect_mode(args)
        if quality_complete and benchmark_complete:
            return quality_complete, benchmark_complete
        if quality_complete:
            _preserve_incomplete_artifact(run_dir / "benchmark.json")
        elif benchmark_complete:
            for name in ("metrics.json", "frames.csv", "predictions"):
                _preserve_incomplete_artifact(run_dir / name)
        else:
            _preserve_incomplete_artifact(run_dir)
        return quality_complete, benchmark_complete


def main() -> int:
    args = _parser().parse_args()
    quality_complete, benchmark_complete = (
        prepare_mode(args) if args.preserve_incomplete else inspect_mode(args)
    )
    print(f"{quality_complete} {benchmark_complete}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArtifactMismatch, ArtifactWriterBusyError, ArtifactWriterOwnershipError) as error:
        raise SystemExit(f"evaluation resume refused: {error}") from None
