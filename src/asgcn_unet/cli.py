from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .data import build_dataset
from .engine import TrainingPaused, _artifact_path_label, benchmark, calibrate, evaluate, train
from .preflight import training_preflight, verify_training_preflight
from .recovery import archive_uncheckpointed_run
from .utils import experiment_base_dir, load_json, resolve_experiment_paths, resolve_path


def _inspect_path_labels(config: dict[str, Any]) -> list[tuple[Path, str]]:
    """Map resolved inspection inputs to stable labels suitable for public logs."""

    dataset = config.get("dataset", {})
    labels = {
        "root": "$DATA_ROOT",
        "val_root": "$VAL_DATA_ROOT",
        "split_manifest": "$SPLIT_MANIFEST",
        "file_manifest": "$FILE_MANIFEST",
    }
    replacements: list[tuple[Path, str]] = []
    for key, label in labels.items():
        value = dataset.get(key)
        if isinstance(value, str) and value:
            replacements.append((Path(value).expanduser().resolve(), label))
    return replacements


def _redact_inspect_text(value: str, replacements: list[tuple[Path, str]]) -> str:
    """Redact configured host paths while retaining relative file/member context."""

    result = value
    ordered = sorted(replacements, key=lambda item: len(str(item[0])), reverse=True)
    for path, label in ordered:
        variants = {str(path), path.as_posix()}
        for variant in sorted(variants, key=len, reverse=True):
            result = result.replace(f"{variant}\\", f"{label}/")
            result = result.replace(f"{variant}/", f"{label}/")
            result = result.replace(variant, label)
    return result


def _redact_inspect_value(value: Any, replacements: list[tuple[Path, str]]) -> Any:
    if isinstance(value, dict):
        return {key: _redact_inspect_value(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_inspect_value(item, replacements) for item in value]
    if isinstance(value, tuple):
        return [_redact_inspect_value(item, replacements) for item in value]
    if isinstance(value, Path):
        return _redact_inspect_text(str(value.resolve()), replacements)
    if isinstance(value, str):
        return _redact_inspect_text(value, replacements)
    return value


def _inspect_one_split(dataset: Any, samples: int, validate_all: bool = False) -> dict[str, Any]:
    details = []
    timestamp_records: list[dict[str, Any]] = []
    preview_count = min(samples, len(dataset))
    count = len(dataset) if validate_all else preview_count
    indices = tqdm(range(count), desc="validate-data", disable=not validate_all)
    for index in indices:
        item = dataset[index]
        timestamp_diagnostics = item.get("metadata", {}).get("event_timestamp_diagnostics")
        if isinstance(timestamp_diagnostics, dict):
            timestamp_records.append(timestamp_diagnostics)
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
    if hasattr(dataset, "zero_event_intervals"):
        result["zero_event_intervals"] = int(dataset.zero_event_intervals)
    if hasattr(dataset, "event_indexing"):
        result["event_indexing"] = dataset.event_indexing
    if timestamp_records:
        def finite_values(key: str) -> list[float]:
            return [
                float(record[key])
                for record in timestamp_records
                if record.get(key) is not None
            ]

        event_minima = finite_values("event_timestamp_min")
        event_maxima = finite_values("event_timestamp_max")
        interval_starts = finite_values("interval_t0")
        interval_ends = finite_values("interval_t1")
        span_ratios = finite_values("event_to_interval_span_ratio")
        offsets = finite_values("event_min_offset_from_t0")
        total_events = sum(int(record.get("event_count", 0)) for record in timestamp_records)
        outside_events = sum(
            int(record.get("outside_interval_count", 0)) for record in timestamp_records
        )
        result["event_timestamp_diagnostics"] = {
            "validated_blocks": len(timestamp_records),
            "event_count": total_events,
            "event_timestamp_min": min(event_minima) if event_minima else None,
            "event_timestamp_max": max(event_maxima) if event_maxima else None,
            "interval_t0_min": min(interval_starts) if interval_starts else None,
            "interval_t1_max": max(interval_ends) if interval_ends else None,
            "event_to_interval_span_ratio_min": min(span_ratios) if span_ratios else None,
            "event_to_interval_span_ratio_max": max(span_ratios) if span_ratios else None,
            "event_min_offset_from_t0_min": min(offsets) if offsets else None,
            "event_min_offset_from_t0_max": max(offsets) if offsets else None,
            "outside_interval_count": outside_events,
            "outside_interval_fraction": (
                outside_events / total_events if total_events else None
            ),
            "strict_interval_validation": False,
            "interpretation": (
                "diagnostic only until all official archives establish a common timestamp basis"
            ),
        }
    return result


def _calibration_sample_limit(value: str) -> int | None:
    if value.strip().lower() == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("samples must be a positive integer or 'all'") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("samples must be a positive integer or 'all'")
    return parsed


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def inspect_dataset(
    config: dict[str, Any],
    samples: int = 3,
    validate_all: bool = False,
    include_private_host_provenance: bool = False,
) -> dict[str, Any]:
    if samples < 0:
        raise ValueError("inspect samples must be non-negative")
    data_config = config["dataset"]
    path_labels = _inspect_path_labels(config)
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
    else:
        dataset = build_dataset(data_config, split="eval")
        try:
            result.update(_inspect_one_split(dataset, samples, validate_all))
        finally:
            if hasattr(dataset, "close"):
                dataset.close()

    if include_private_host_provenance:
        result["private_host_provenance"] = {
            "data_root": data_config["root"],
            "validation_data_root": data_config.get("val_root"),
            "split_manifest": data_config.get("split_manifest"),
            "file_manifest": data_config.get("file_manifest"),
            "publication_warning": "private local diagnostics; do not publish",
        }
        return result
    result = _redact_inspect_value(result, path_labels)
    result["root"] = "$DATA_ROOT"
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASGCN-U-Net event-to-frame experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = subparsers.add_parser("inspect", help="validate dataset structure")
    inspect_cmd.add_argument("--config", required=True)
    inspect_cmd.add_argument("--samples", type=int, default=3)
    inspect_cmd.add_argument(
        "--validate-all",
        action="store_true",
        help="decode and validate every selected sample while keeping only the preview",
    )
    inspect_cmd.add_argument(
        "--include-private-host-provenance",
        action="store_true",
        help=(
            "PRIVATE: include exact dataset paths for local diagnostics; "
            "do not publish or attach this output"
        ),
    )

    train_cmd = subparsers.add_parser("train", help="train on EventHDR")
    train_cmd.add_argument("--config", required=True)
    train_cmd.add_argument(
        "--max-hours", type=float,
        help="save and pause at a safe batch boundary after this invocation's time budget",
    )
    train_cmd.add_argument(
        "--checkpoint-seconds", type=float, default=300.0,
        help="periodic checkpoint interval at successful batch boundaries (default: 300)",
    )
    train_cmd.add_argument(
        "--resume",
        help="resume from a checkpoint (relative paths are resolved from the repository root)",
    )
    train_cmd.add_argument(
        "--preflight-report",
        default="runs/profile.json",
        help="passed CUDA preflight report required before training",
    )
    train_cmd.add_argument(
        "--allow-unverified-preflight",
        action="store_true",
        help="NON-REPORTING ONLY: explicitly bypass the CUDA preflight gate",
    )
    train_cmd.add_argument(
        "--restart-uncheckpointed",
        action="store_true",
        help="archive metadata-only failed training output before a fresh run; stop old jobs first",
    )

    profile_cmd = subparsers.add_parser(
        "profile",
        help="scan every EventHDR train graph and run CUDA forward/backward preflight",
    )
    profile_cmd.add_argument("--config", required=True)
    profile_cmd.add_argument("--output", required=True)
    profile_cmd.add_argument(
        "--resume-scan", action="store_true",
        help="resume a matching interrupted scan; never overwrite a passed report",
    )
    profile_cmd.add_argument(
        "--reuse-report",
        help="reuse verified topology records from an older report; rerun all GPU probes",
    )
    profile_cmd.add_argument(
        "--cpu-threads", type=_positive_integer, default=4,
        help="CPU helper threads during the CUDA topology scan (default: 4)",
    )
    profile_cmd.add_argument(
        "--samples",
        type=_positive_integer,
        default=3,
        help="number of densest samples used for forward/backward (default: 3)",
    )
    profile_cmd.add_argument(
        "--top-density",
        type=_positive_integer,
        default=10,
        help="number of highest-edge-count samples recorded (default: 10)",
    )

    verify_cmd = subparsers.add_parser(
        "verify-profile",
        help="re-bind a passed preflight report to current config/data/source/CUDA",
    )
    verify_cmd.add_argument("--config", required=True)
    verify_cmd.add_argument("--report", required=True)

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
    eval_cmd.add_argument(
        "--allow-unsealed-checkpoint-for-non-reporting",
        action="store_true",
        help=(
            "SYNTHETIC TESTS ONLY: accept missing/mismatched checkpoint protocols and "
            "permanently mark the produced metrics report_eligible=false"
        ),
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
    bench_cmd.add_argument(
        "--allow-unsealed-checkpoint-for-non-reporting",
        action="store_true",
        help=(
            "SYNTHETIC TESTS ONLY: accept missing/mismatched checkpoint protocols and "
            "permanently mark the benchmark report_eligible=false"
        ),
    )

    calibrate_cmd = subparsers.add_parser("calibrate", help="calibrate ANN-to-SNN thresholds")
    calibrate_cmd.add_argument("--config", required=True)
    calibrate_cmd.add_argument("--checkpoint", required=True)
    calibrate_cmd.add_argument("--output", required=True)
    calibrate_cmd.add_argument(
        "--samples",
        type=_calibration_sample_limit,
        default=None,
        metavar="N|all",
        help=(
            "balanced calibration sample count; default 'all' uses every training frame; "
            "a partial limit requires the explicit non-reporting override"
        ),
    )
    calibrate_cmd.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing calibrated output checkpoint",
    )
    calibrate_cmd.add_argument(
        "--allow-unsealed-calibration",
        action="store_true",
        help=(
            "permit a protocol mismatch only for non-reporting tests; the output is "
            "permanently marked sealed=false"
        ),
    )
    eval_cmd.add_argument("--output-dir", help="separate evaluation artifacts from other experiments")
    bench_cmd.add_argument("--output-dir", help="separate timing artifacts from other experiments")
    return parser


def _execute_command(args: argparse.Namespace) -> None:
    config_path = resolve_path(args.config, ".")
    try:
        config = resolve_experiment_paths(load_json(config_path), config_path)
    except Exception as error:
        if args.command != "inspect" or args.include_private_host_provenance:
            raise
        message = _redact_inspect_text(str(error), [(config_path, "$CONFIG")])
        raise SystemExit(f"Dataset inspection failed: {message}") from None
    base_dir = experiment_base_dir(config_path)
    if args.command in {"evaluate", "benchmark"} and args.output_dir:
        config["eval"]["output_dir"] = str(resolve_path(args.output_dir, base_dir))
    if args.command == "inspect":
        try:
            result = inspect_dataset(
                config,
                args.samples,
                args.validate_all,
                include_private_host_provenance=args.include_private_host_provenance,
            )
        except Exception as error:
            if args.include_private_host_provenance:
                raise
            replacements = [
                *_inspect_path_labels(config),
                (config_path, "$CONFIG"),
            ]
            message = _redact_inspect_text(str(error), replacements)
            raise SystemExit(f"Dataset inspection failed: {message}") from None
    elif args.command == "profile":
        torch.set_num_threads(args.cpu_threads)
        result = training_preflight(
            config,
            resolve_path(args.output, base_dir),
            profile_samples=args.samples,
            top_density_count=args.top_density,
            require_cuda=True,
            resume_scan=args.resume_scan,
            reuse_report=resolve_path(args.reuse_report, base_dir) if args.reuse_report else None,
        )
    elif args.command == "verify-profile":
        result = verify_training_preflight(
            config,
            resolve_path(args.report, base_dir),
        )
    elif args.command == "train":
        resume = resolve_path(args.resume, base_dir) if args.resume else None
        if args.restart_uncheckpointed and args.allow_unverified_preflight:
            raise ValueError("Restarting uncheckpointed output requires a verified CUDA profile")
        if args.restart_uncheckpointed and (resume or config["train"].get("resume")):
            raise ValueError("Restarting uncheckpointed output cannot be combined with resume")
        report_path = resolve_path(args.preflight_report, base_dir)
        if args.allow_unverified_preflight:
            print(
                "WARNING: CUDA preflight was explicitly bypassed; this run is "
                "non-reporting and must not support A100/A6000 memory claims.",
                file=sys.stderr,
            )
            preflight_gate = {
                "schema": "asgcn_preflight_verification_v1",
                "status": "bypassed_non_reporting",
                "report_eligible": False,
                "report": report_path.name,
                "warning": "unverified CUDA memory/topology preflight",
            }
        else:
            preflight_gate = verify_training_preflight(config, report_path)
        if args.restart_uncheckpointed:
            archived = archive_uncheckpointed_run(config["output"]["run_dir"], base_dir)
            if archived is not None:
                print(f"Archived uncheckpointed run metadata: {_artifact_path_label(archived)}")
        config["preflight_gate"] = preflight_gate
        run_options = {}
        if args.max_hours is not None:
            run_options["max_seconds"] = args.max_hours * 3600
        if args.checkpoint_seconds != 300.0:
            run_options["checkpoint_seconds"] = args.checkpoint_seconds
        result = {
            "best_checkpoint": _artifact_path_label(train(config, resume_from=resume, **run_options))
        }
    elif args.command == "evaluate":
        if args.allow_unsealed_checkpoint_for_non_reporting:
            print(
                "WARNING: unsealed checkpoint override enabled; all outputs are "
                "permanently non-reporting.",
                file=sys.stderr,
            )
        result = evaluate(
            config,
            resolve_path(args.checkpoint, base_dir),
            inference_mode=args.inference_mode,
            simulation_steps=args.simulation_steps,
            snn_dynamics=args.snn_dynamics,
            allow_unsealed_checkpoint_for_non_reporting=(
                args.allow_unsealed_checkpoint_for_non_reporting
            ),
        )
    elif args.command == "benchmark":
        if args.allow_unsealed_checkpoint_for_non_reporting:
            print(
                "WARNING: unsealed checkpoint override enabled; all outputs are "
                "permanently non-reporting.",
                file=sys.stderr,
            )
        result = benchmark(
            config,
            resolve_path(args.checkpoint, base_dir),
            warmup=args.warmup,
            steps=args.steps,
            inference_mode=args.inference_mode,
            simulation_steps=args.simulation_steps,
            snn_dynamics=args.snn_dynamics,
            allow_unsealed_checkpoint_for_non_reporting=(
                args.allow_unsealed_checkpoint_for_non_reporting
            ),
        )
    elif args.command == "calibrate":
        result = {
            "calibrated_checkpoint": _artifact_path_label(
                calibrate(
                    config,
                    resolve_path(args.checkpoint, base_dir),
                    resolve_path(args.output, base_dir),
                    samples=args.samples,
                    overwrite=args.overwrite,
                    allow_unsealed_calibration=args.allow_unsealed_calibration,
                )
            )
        }
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if args.command == "profile" and not result["passed"]:
        raise SystemExit(1)


def _public_error_path_labels(args: argparse.Namespace) -> list[tuple[Path, str]]:
    labels: list[tuple[Path, str]] = [
        (Path.cwd().resolve(), "$PROJECT_ROOT"),
        (Path.home().resolve(), "$HOME"),
    ]
    try:
        config_path = resolve_path(args.config, ".")
        labels.append((config_path, "$CONFIG"))
        config = resolve_experiment_paths(load_json(config_path), config_path)
        labels.extend(_inspect_path_labels(config))
        for section, key, label in (
            ("output", "run_dir", "$RUNS_ROOT"),
            ("eval", "output_dir", "$EVAL_OUTPUT"),
        ):
            value = config.get(section, {}).get(key)
            if isinstance(value, str) and value:
                labels.append((Path(value).expanduser().resolve(), label))
        base_dir = experiment_base_dir(config_path)
    except Exception:  # noqa: BLE001
        # This helper runs while another exception is already being handled.
        # Path discovery is best-effort and must never replace the original
        # command failure with a secondary redaction failure.
        base_dir = Path.cwd().resolve()

    for attribute, label in (
        ("checkpoint", "$CHECKPOINT"),
        ("resume", "$RESUME_CHECKPOINT"),
        ("output", "$OUTPUT"),
        ("output_dir", "$EVAL_OUTPUT"),
        ("report", "$PREFLIGHT_REPORT"),
        ("preflight_report", "$PREFLIGHT_REPORT"),
    ):
        value = getattr(args, attribute, None)
        if isinstance(value, str) and value:
            labels.append((resolve_path(value, base_dir), label))
    return labels


def _redact_public_error(error: Exception, args: argparse.Namespace) -> str:
    message = _redact_inspect_text(str(error), _public_error_path_labels(args))
    message = re.sub(r"(?i)(?<![A-Za-z0-9_])/(?:home|Users)/[^/\s]+", "$HOME", message)
    message = re.sub(r"(?i)\b[A-Z]:[\\/]Users[\\/][^\\/\s]+", "$HOME", message)
    hostnames = {
        socket.gethostname(),
        socket.getfqdn(),
        os.environ.get("HOSTNAME", ""),
        os.environ.get("COMPUTERNAME", ""),
    }
    for hostname in sorted((value for value in hostnames if value), key=len, reverse=True):
        message = re.sub(re.escape(hostname), "$HOST", message, flags=re.IGNORECASE)
    return message


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    private_provenance = (
        getattr(args, "include_private_host_provenance", False)
        or os.environ.get("INCLUDE_PRIVATE_HOST_PROVENANCE", "0") == "1"
    )
    try:
        _execute_command(args)
    except TrainingPaused as paused:
        print(json.dumps({
            "status": "paused", "reason": paused.reason,
            "resume_checkpoint": _artifact_path_label(paused.checkpoint_path),
            "message": "Saved successfully; resume with the same config and --resume. Training is not complete.",
        }, ensure_ascii=False, indent=2))
        raise SystemExit(75) from None
    except Exception as error:
        if private_provenance:
            raise
        try:
            message = _redact_public_error(error, args)
        except Exception:  # noqa: BLE001
            # Public error handling is a privacy boundary. If best-effort path
            # discovery or even an exception's __str__ fails, emit no details
            # rather than allowing a secondary traceback to expose host paths.
            message = "details suppressed because public-safe error rendering failed"
        raise SystemExit(f"{args.command} failed: {message}") from None


if __name__ == "__main__":
    main()
