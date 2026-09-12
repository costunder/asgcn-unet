"""Measure and probe a streaming edge guard without starting training or calibration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shlex
import sys
import tempfile
import uuid
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def _positive_integer(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True,
                        help="prepared v3 directory containing configs/train,hdr,aid.json")
    parser.add_argument("--use-measured-edge-guard", action="store_true", required=True,
                        help="authorize a measured guard in new configuration files")
    parser.add_argument("--reserve-vram-mib", required=True, type=_positive_float,
                        help="positive free-VRAM reserve required by the CUDA preflight")
    parser.add_argument("--cpu-threads", type=_positive_integer, default=4,
                        help="CPU helper threads, bounded by measured allocation (default: 4)")
    parser.add_argument("--resume-from",
                        help="previous recovery directory; committed raw scanner state is required to resume")
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _persist_recovery(path: Path, value: dict, previous_sha256: str | None = None) -> str:
    """Create metadata exclusively, then replace only this invocation's unchanged file."""
    from asgcn_unet.stream_preflight import _digest

    core = dict(value)
    core.pop("commitment_sha256", None)
    value["commitment_sha256"] = _digest(core)
    if previous_sha256 is None:
        _write_new_json(path, value)
    else:
        if path.resolve() != path or _sha256(path) != previous_sha256:
            raise ValueError("Recovery metadata changed outside this invocation; it was preserved")
        pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        _write_new_json(pending, value)
        if path.resolve() != path or _sha256(path) != previous_sha256:
            raise ValueError(f"Recovery metadata changed; it was preserved. Pending metadata: {pending}")
        os.replace(pending, path)
    return _sha256(path)


def _overlaps(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def _load_experiment_configs(experiment_root: Path) -> tuple[dict, dict]:
    from asgcn_unet.stream_preflight import validate_streaming_contract
    from asgcn_unet.utils import experiment_base_dir, load_json, resolve_experiment_paths

    project = PROJECT.resolve()
    if (not experiment_root.is_dir() or experiment_root == project
            or not experiment_root.is_relative_to(project)):
        raise ValueError("experiment-root must be an existing prepared directory inside this checkout")
    for name in ("src", "scripts", "configs", "data"):
        if _overlaps(experiment_root, (project / name).resolve()):
            raise ValueError("experiment-root overlaps a source, configuration, or data directory")
    directory = experiment_root / "configs"
    if directory.resolve() != directory:
        raise ValueError("Prepared configurations must remain inside this experiment")
    configs, sources = {}, {}
    for kind in ("train", "hdr", "aid"):
        path = directory / f"{kind}.json"
        if path.resolve() != path or not path.is_file():
            raise ValueError(f"Prepared configuration is missing or outside this experiment: {path}")
        if experiment_base_dir(path) != project:
            raise ValueError("Prepared configurations belong to a different checkout")
        configs[kind] = resolve_experiment_paths(load_json(path), path)
        validate_streaming_contract(configs[kind], training=kind == "train")
        sources[kind] = {"path": str(path), "sha256": _sha256(path)}
        section, key = ("output", "run_dir") if kind == "train" else ("eval", "output_dir")
        destination = Path(configs[kind][section][key]).resolve()
        if (destination == experiment_root or not destination.is_relative_to(experiment_root)
                or _overlaps(destination, directory)):
            raise ValueError(f"{kind} output must belong to this experiment, outside its configs")
        for key in ("root", "val_root"):
            value = configs[kind]["dataset"].get(key)
            if value and _overlaps(experiment_root, Path(value).resolve()):
                raise ValueError("Recovery output must not overlap an input dataset directory")
    model = configs["train"]["model"]
    if any(configs[kind]["model"] != model for kind in ("hdr", "aid")):
        raise ValueError("Prepared train, HDR, and Aid configurations must share the same model")
    return configs, sources


def _command(*arguments: str | Path) -> str:
    return shlex.join([sys.executable, "-B", "-m", "asgcn_unet.cli", *map(str, arguments)])


def _inspect_resume_checkpoint(checkpoint_dir, resources):
    from asgcn_unet.stream_scan_checkpoint import inspect_scan_checkpoint

    available = resources.get("memory", {}).get("effective_available_bytes")
    if (isinstance(available, bool) or not isinstance(available, (int, float))
            or not math.isfinite(available) or available <= 0):
        raise ValueError("Available RAM must be measured before inspecting a scanner checkpoint")
    return inspect_scan_checkpoint(checkpoint_dir, memory_budget_mib=available / 1024**2)


def _failure_summary(
    report: dict | None, stage: str, error: BaseException | None = None,
    report_path: Path | None = None,
) -> dict:
    read_error = None
    if report is None and report_path is not None and report_path.is_file():
        try:
            with report_path.open(encoding="utf-8") as handle:
                report = json.load(handle)
            if not isinstance(report, dict):
                raise TypeError("Saved preflight report is not a JSON object")
        except (OSError, ValueError, TypeError) as failure:
            read_error = str(failure)
            report = None
    failure = (report or {}).get("failure") or {}
    topology = (report or {}).get("topology") or {}
    return {
        "stage": failure.get("stage", stage),
        "probe_phase": failure.get("probe_phase"),
        "dataset_indices": failure.get("dataset_indices"),
        "type": type(error).__name__ if error is not None else failure.get("type"),
        "message": (str(error) or type(error).__name__) if error is not None else failure.get(
            "message", "Preflight did not pass its CUDA gate"),
        "saved_report_read_error": read_error,
        "max_readout_directed_edges": topology.get("max_readout_directed_edges"),
        "max_prefix_union_directed_edges_upper_bound": topology.get(
            "max_prefix_union_directed_edges_upper_bound"),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    from asgcn_unet.resources import collect_runtime_resources
    from asgcn_unet.stream_preflight import _differences, streaming_training_preflight
    from asgcn_unet.utils import load_json, resolve_path
    output_root, report, recovery, profile_path = None, None, None, None
    metadata_sha256 = None
    stage = "validate_prepared_experiment"
    result_code = 1
    try:
        experiment_root = resolve_path(args.experiment_root, PROJECT)
        configs, sources = _load_experiment_configs(experiment_root)
        resources = collect_runtime_resources(include_cuda=False)
        cpu_limit = resources["cpu"].get("effective_cpu_limit")
        cpu_threads = args.cpu_threads
        if isinstance(cpu_limit, (int, float)) and math.isfinite(cpu_limit) and cpu_limit > 0:
            cpu_threads = min(cpu_threads, max(1, math.floor(cpu_limit)))
        torch.set_num_threads(cpu_threads)
        output_root = Path(tempfile.mkdtemp(prefix="preflight-recovery-", dir=experiment_root))
        paths = {kind: output_root / "configs" / f"{kind}.json" for kind in configs}
        profile_path = output_root / "stream-profile.json"
        recovery = {
            "schema": "asgcn_streaming_preflight_recovery_v1", "status": "running",
            "report_eligible": False, "experiment_root": str(experiment_root),
            "output_root": str(output_root), "source_configs": sources,
            "configs": {kind: str(path) for kind, path in paths.items()},
            "preflight_report": str(profile_path), "changes": {}, "warnings": [],
            "request": {"use_measured_edge_guard": True, "reserve_vram_mib": args.reserve_vram_mib,
                        "requested_cpu_threads": args.cpu_threads, "cpu_threads": cpu_threads,
                        "effective_cpu_limit": cpu_limit},
            "training_executed": False, "calibration_executed": False,
            "next_commands": [], "failure": None,
        }
        recovery["request"]["resume_from"] = args.resume_from
        recovery["scan_checkpoint_dir"] = str(output_root / "scan-checkpoint")
        stage = "persist_initial_recovery_metadata"
        metadata_sha256 = _persist_recovery(output_root / "recovery.json", recovery)
        print(f"Recovery directory: {output_root}", flush=True)
        print(f"CPU helper threads: {cpu_threads}; requested VRAM reserve: {args.reserve_vram_mib:g} MiB", flush=True)
        resume_checkpoint = None
        if args.resume_from:
            from asgcn_unet.engine import (
                _artifact_path_label,
                _current_source_contract,
                _public_config,
            )
            from asgcn_unet.recovery_evidence import load_recovery_evidence

            stage = "inspect_saved_recovery_evidence"
            previous_dir = resolve_path(args.resume_from, PROJECT)
            evidence, previous_profile = load_recovery_evidence(
                previous_dir, experiment_root=experiment_root, source_configs=sources,
                public_train_config=_public_config(configs["train"]),
                resolved_train_config=configs["train"],
                current_source_contract=_current_source_contract(), reserve_vram_mib=args.reserve_vram_mib,
                profile_output_label=_artifact_path_label(previous_dir / "stream-profile.json"),
            )
            recovery["resume_evidence"] = evidence
            # Preserve prior count diagnostics in a failure summary, without
            # presenting the old report as this invocation's preflight result.
            report = {"topology": previous_profile["topology"]}
            floor = evidence["single_readout_storage_floor"]["graph_and_basis_mib"]
            storage = evidence["single_readout_storage_floor"]["graph_storage"]
            floor_label = ("one actual readout graph plus basis" if storage == "materialized"
                           else "one implicit readout raw node set (excluding index/scratch/training)")
            print(f"Saved scan evidence: {evidence['scanned_samples']}/{evidence['dataset_samples']} frames; "
                  f"{floor_label} requires at least {floor:,.2f} MiB.", flush=True)
            print("Historical evidence only: current source/data/device have not been certified by this report.", flush=True)
            if evidence["infeasible_on_saved_device"]:
                budget = evidence["saved_device_budget_after_requested_reserve_mib"]
                print(f"The recorded allocation allows at most {budget:,.2f} MiB after the requested reserve; "
                      f"the recorded {storage} graph cannot fit that allocation.", flush=True)
            if not evidence["checkpoint_manifest_present"]:
                raise ValueError(
                    "The previous recovery has no committed raw scanner checkpoint. A partial JSON report "
                    "cannot restore live streams/counters; exact resume is unavailable. No GPU probe or "
                    "full scan was started, and no frame-zero replay was substituted."
                )
            if not evidence["source_matches_current"]:
                raise ValueError(
                    "The executable source differs from the saved run. Historical counts were inspected, "
                    "but scanner-state migration is not authorized by a matching config alone; resume was refused."
                )
            stage = "inspect_saved_scanner_checkpoint"
            resume_checkpoint = Path(evidence["checkpoint_directory"])
            _inspect_resume_checkpoint(resume_checkpoint, resources)
            evidence["checkpoint_integrity_verified"] = True
            print("Committed scanner files verified. Current source/data/schedule identities will be checked "
                  "before continuing the saved scan; no completed frames may be silently replayed.", flush=True)
        stage = "streaming_preflight"
        if args.resume_from:
            metadata_sha256 = _persist_recovery(output_root / "recovery.json", recovery, metadata_sha256)
        report = streaming_training_preflight(
            copy.deepcopy(configs["train"]), profile_path, require_cuda=True,
            measured_guard_config_output=paths["train"], reserve_vram_mib=args.reserve_vram_mib,
            scan_checkpoint_dir=output_root / "scan-checkpoint", resume_checkpoint=resume_checkpoint,
        )
        if report.get("passed") is not True or report.get("report_eligible") is not True:
            recovery["status"] = "failed"
            recovery["failure"] = _failure_summary(report, stage)
        else:
            stage = "derive_matching_evaluation_configs"
            for source in sources.values():
                if _sha256(Path(source["path"])) != source["sha256"]:
                    raise ValueError("A source configuration changed during recovery")
            train_config = load_json(paths["train"])
            expected = copy.deepcopy(configs["train"])
            expected["model"]["max_graph_edges"] = train_config["model"]["max_graph_edges"]
            if train_config != expected:
                raise ValueError("Measured training configuration changed fields other than the edge guard")
            guard = train_config["model"]["max_graph_edges"]
            original_guard = configs["train"]["model"]["max_graph_edges"]
            if type(guard) is not int or guard < 1 or (original_guard is not None and guard < original_guard):
                raise ValueError("Measured guard must be a positive integer and cannot lower the original guard")
            recovery["changes"]["train"] = _differences(configs["train"], train_config)
            for kind in ("hdr", "aid"):
                current = copy.deepcopy(configs[kind])
                current["model"]["max_graph_edges"] = guard
                previous_override = current["eval"].get("max_graph_edges_override")
                current["eval"]["max_graph_edges_override"] = None
                current["eval"]["output_dir"] = str(output_root / "eval" / kind)
                if previous_override is not None:
                    recovery["warnings"].append(
                        f"{kind}: removed the inherited static evaluation edge guard {previous_override}; "
                        "it is not streaming topology evidence."
                    )
                _write_new_json(paths[kind], current)
                recovery["changes"][kind] = _differences(configs[kind], current)
            recovery["warnings"].append(
                "The training preflight does not certify HDR/Aid evaluation or SNN memory; "
                "their streaming evaluation remains unverified."
            )
            run_dir = Path(train_config["output"]["run_dir"])
            if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
                recovery["warnings"].append(
                    "The original training run directory is nonempty. It was preserved; "
                    "no fresh training or calibration command is provided."
                )
            elif train_config["train"].get("resume") is not None:
                recovery["warnings"].append(
                    "The training configuration requests resume. Existing checkpoints were preserved; "
                    "a changed edge guard requires separate resume compatibility review."
                )
            else:
                recovery["next_commands"] = [
                    _command("train", "--config", paths["train"], "--preflight-report", profile_path,
                             "--checkpoint-seconds", "300"),
                    _command("calibrate", "--config", paths["train"], "--checkpoint", run_dir / "best.pt",
                             "--output", output_root / "calibrated.pt", "--samples", "all"),
                ]
            recovery["status"], recovery["report_eligible"] = "passed", True
            result_code = 0
    except KeyboardInterrupt as error:
        if recovery is not None:
            recovery["status"] = "interrupted"
            recovery["failure"] = _failure_summary(report, stage, error, profile_path)
        else:
            print(f"Recovery interrupted at {stage}", file=sys.stderr)
        result_code = 130
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, FloatingPointError) as error:
        if recovery is not None:
            recovery["status"] = "failed"
            recovery["failure"] = _failure_summary(report, stage, error, profile_path)
        else:
            print(f"Recovery failed at {stage}: {error}", file=sys.stderr)
    if recovery is not None:
        try:
            _persist_recovery(output_root / "recovery.json", recovery, metadata_sha256)
        except (OSError, ValueError, TypeError) as error:
            print(f"Could not save new recovery metadata: {error}", file=sys.stderr)
            return 1
        print(f"Recovery metadata: {output_root / 'recovery.json'}")
        if profile_path.is_file():
            print(f"Preflight report: {recovery['preflight_report']}")
        else:
            print("No new preflight report was created; the failure was detected before a new scan.")
        if recovery["status"] == "passed":
            print("CUDA preflight passed. Matching train/HDR/Aid configurations:")
            for kind, path in recovery["configs"].items():
                print(f"  {kind}: {path}")
            for warning in recovery["warnings"]:
                print(f"Warning: {warning}")
            if recovery["next_commands"]:
                print("Run training next; run calibration only after full training succeeds:")
                for command in recovery["next_commands"]:
                    print(command)
        else:
            failure = recovery["failure"]
            print(f"Recovery {recovery['status']} at {failure['stage']}: {failure['message']}", file=sys.stderr)
            if failure.get("probe_phase") or failure.get("dataset_indices"):
                print(f"Probe phase: {failure.get('probe_phase')}; "
                      f"dataset indices: {failure.get('dataset_indices')}", file=sys.stderr)
            print(f"Count maxima: readout={failure['max_readout_directed_edges']}, "
                  f"prefix-union upper bound={failure['max_prefix_union_directed_edges_upper_bound']}",
                  file=sys.stderr)
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
