"""Inspect one complete raw graph window using an existing training configuration.

This is a CPU diagnostic, not training, preflight certification, or model inference.
It does not stop jobs, choose a GPU, modify a configuration, or resume a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from audit_raw_event_graph import audit_raw_event_graph, save_report

from asgcn_unet.diagnostic_resources import preflight
from asgcn_unet.graph_inspection_view import save_html
from asgcn_unet.stream_preflight import validate_streaming_contract


def _read_object(path):
    # Only configuration metadata is read here, never HDF5 payloads.
    if path.stat().st_size > 1024**2:
        raise ValueError("Configuration/manifest exceeds the 1-MiB diagnostic metadata budget")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("Configuration/manifest must contain a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _resolve(value, project):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("An explicit nonempty path is required")
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else project / path).resolve(strict=True)


def _file_keys(values):
    if not isinstance(values, list) or not values:
        raise ValueError("Declared training files must be a nonempty list")
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Training file keys must be nonempty strings")
        key = PurePosixPath(value.replace("\\", "/"))
        if (key.is_absolute() or ".." in key.parts or not key.name
                or ":" in key.parts[0] or key.suffix.lower() not in {".h5", ".hdf5"}):
            raise ValueError("Training file keys must be relative HDF5 paths")
        normalized.append(key.as_posix())
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate normalized training file keys")
    return normalized


def load_plan(config_path, source_file, frame_index, *, workspace=PROJECT):
    """Read immutable settings; refuse unsupported semantics instead of substituting."""
    project = Path(workspace).resolve(strict=True)
    config_path = _resolve(config_path, project)
    if not config_path.is_relative_to(project):
        raise ValueError("Use the existing training config inside this checkout")
    # Match experiment_base_dir's nearest pyproject rule without heavy utils imports.
    owner = next((parent for parent in config_path.parents
                  if (parent / "pyproject.toml").is_file()), config_path.parent)
    if owner != project:
        raise ValueError("Configuration belongs to a different checkout")
    config, config_hash = _read_object(config_path)
    if not isinstance(config.get("train"), dict):
        raise TypeError("This entry point requires a training config, not an evaluation config")
    validate_streaming_contract(config, training=True)
    data, model = config["dataset"], config["model"]
    if (model["event_sampling_factor"] != 1 or model.get("graph_position_dims") != 3
            or data.get("crop_size") is not None or data.get("random_crop", False)
            or data.get("frame_stride") != 1):
        raise ValueError("This diagnostic supports the current R=1, 3-axis, full-sensor, stride-1 "
                         "contract only; unsupported settings were not replaced")
    radius = model.get("graph_radius")
    if (isinstance(radius, bool) or not isinstance(radius, (int, float))
            or not 0 < radius < float("inf")):
        raise ValueError("model.graph_radius must be explicitly finite and positive")
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("frame_index must be an explicit nonnegative file-local frame")
    root = _resolve(data.get("root"), project)
    if not root.is_dir():
        raise ValueError("Training dataset root must be an existing directory")
    key = Path(_file_keys([str(source_file)])[0])
    if key.is_absolute() or ".." in key.parts or key.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("source_file must be a relative HDF5 key within dataset.root")
    source = (root / key).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError("Selected HDF5 source must remain inside dataset.root")
    manifest_record = None
    identity_inputs = [(config_path, config_hash)]
    if data.get("split_manifest"):
        manifest_path = _resolve(data["split_manifest"], project)
        manifest, manifest_hash = _read_object(manifest_path)
        files = _file_keys(manifest.get("train_files"))
        if (manifest.get("status") != "final"
                or manifest.get("split_schema") != "official_separate_roots_v1"
                or key.as_posix() not in files):
            raise ValueError("Selected source is not a member of the declared final training split")
        manifest_record = {"sha256": manifest_hash, "declared_training_files": len(files),
                           "coverage": "selected file membership only; full split not validated"}
        identity_inputs.append((manifest_path, manifest_hash))
    elif data.get("allowed_files") is not None:
        files = _file_keys(data["allowed_files"])
        if key.as_posix() not in files:
            raise ValueError("Selected source is excluded by dataset.allowed_files")
        manifest_record = {"selection": "dataset.allowed_files", "declared_training_files": len(files),
                           "coverage": "selected file membership only; full split not validated"}
    stream = model["stream_config"]
    arguments = {
        "source_file": source, "frame_index": frame_index,
        "window_seconds": stream["window_seconds"], "time_scale_seconds": stream["time_scale_seconds"],
        "radius": radius, "timestamp_scale_to_seconds": data["timestamp_scale_to_seconds"],
        "interval_timestamp_scale_to_seconds": data["interval_timestamp_scale_to_seconds"],
        "target_options": {key: data[key] for key in
                           ("target_channels", "target_normalization", "tone_map", "tone_map_mu") if key in data},
    }
    return {
        "arguments": arguments,
        "identity_inputs": identity_inputs,
        "contract": {
            "config": config_path.relative_to(project).as_posix(), "config_sha256": config_hash,
            "source_key": key.as_posix(), "split": "train", "manifest": manifest_record,
            "settings_changed": False, "architecture_version": model["architecture_version"],
            "sampling_factor": model["event_sampling_factor"], "position_dims": model["graph_position_dims"],
            "graph_storage_in_training": model.get("graph_storage", "materialized"),
            "graph_scope": "fine input radius graph before learned convolution or pooling",
            "existing_max_graph_edges": model.get("max_graph_edges"),
            "existing_guard_policy": "recorded only; diagnostic counts do not truncate or certify training",
            "training_executed": False, "full_preflight_executed": False,
        },
    }


def compact_report(report):
    """Pasteable measured results without expanding the potentially large edge trace."""
    return {
        "nodes": report["window"]["nodes"],
        "total_directed_edges": report["total_directed_edges"],
        "queries": [{key: query[key] for key in
                     ("raw_row_id", "in_degree", "all_window_sources_checked", "oracle_match")}
                    for query in report["queries"]],
        "audit_elapsed_s": report["timings"]["audit_elapsed_s"],
        "full_graph_oracle_verified": False,
        "note": "null edge total means not measured; selected queries are not whole-graph certification",
    }


def inspect_graph(*, config_path, source_file, frame_index, cpu_threads, memory_budget_mib,
                  reserve_memory_mib, count_all_nodes=False, workspace=PROJECT):
    for name, value in (("cpu_threads", cpu_threads), ("memory_budget_mib", memory_budget_mib),
                        ("reserve_memory_mib", reserve_memory_mib)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be an explicit positive integer")
    project = Path(workspace).resolve(strict=True)
    preflight(budget_bytes=memory_budget_mib * 1024**2, reserve_bytes=reserve_memory_mib * 1024**2,
              cpu_threads=cpu_threads)
    plan = load_plan(config_path, source_file, frame_index, workspace=project)
    print("CPU graph inspection only: one selected file/window, not the full training scan.", flush=True)
    print(json.dumps({**plan["contract"], "geometry": {
        key: value for key, value in plan["arguments"].items() if key != "source_file"
    }, "count_all_nodes": count_all_nodes}, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if count_all_nodes:
        print("Exact all-node degree counting requested: work can scale with all edges in this window. "
              "No full edge array will be retained; no time or GPU-fit guarantee.", flush=True)
    report = audit_raw_event_graph(
        **plan["arguments"], cpu_threads=cpu_threads, memory_budget_bytes=memory_budget_mib * 1024**2,
        reserve_memory_bytes=reserve_memory_mib * 1024**2, count_all_nodes=count_all_nodes,
        include_point_cloud=True,
    )
    # Refuse attaching a now-stale config identity to a successful graph audit.
    for input_path, recorded_hash in plan["identity_inputs"]:
        _, current_hash = _read_object(input_path)
        if current_hash != recorded_hash:
            raise RuntimeError("Training config/manifest changed during inspection; no report was saved")
    report["training_config_inspection"] = plan["contract"]
    runs = project / "runs"
    if runs.resolve() != runs:
        raise ValueError("Diagnostic output runs directory must remain inside this checkout")
    runs.mkdir(exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="graph-inspection-", dir=runs))
    saved = save_report(report, destination / "graph.json", workspace=project)
    print(f"Graph inspection saved: {saved}", flush=True)
    visual = save_html(report, destination / "graph.html", workspace=project,
                       memory_budget_bytes=memory_budget_mib * 1024**2,
                       reserve_memory_bytes=reserve_memory_mib * 1024**2, cpu_threads=cpu_threads)
    print(f"Offline graph saved: {visual}", flush=True)
    print("Open graph.html locally: all window nodes + every recorded neighbor of each audited query. "
          "No full-edge scan, web server or SSH tunnel is needed.", flush=True)
    print("GRAPH_INSPECTION_SUMMARY", flush=True)
    print(json.dumps(compact_report(report), indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    print("Existing training/checkpoints/results unchanged. This report does not authorize training.", flush=True)
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-file", required=True,
                        help="HDF5 key within the configured training root, e.g. 26.h5")
    parser.add_argument("--frame-index", required=True, type=int,
                        help="Explicit zero-based frame in that file (diagnostic selection, not training subset)")
    parser.add_argument("--cpu-threads", required=True, type=int)
    parser.add_argument("--memory-budget-mib", required=True, type=int)
    parser.add_argument("--reserve-memory-mib", required=True, type=int)
    parser.add_argument("--count-all-nodes", action="store_true",
                        help="Explicitly count every node degree in this one window; may be expensive")
    args = parser.parse_args(argv)
    try:
        inspect_graph(config_path=args.config, source_file=args.source_file, frame_index=args.frame_index,
                      cpu_threads=args.cpu_threads, memory_budget_mib=args.memory_budget_mib,
                      reserve_memory_mib=args.reserve_memory_mib, count_all_nodes=args.count_all_nodes)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, MemoryError) as error:
        print(f"Graph inspection failed: {error}", file=sys.stderr)
        print("No job was stopped, no training started, and no existing experiment was overwritten.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
