"""Read committed prior diagnostics without loading datasets or initializing CUDA.

A partial report is not a scanner checkpoint or a passed preflight. Historical
source/data/device identities are recorded, never assumed to match a new run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _object(value, name):
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _load(path):
    with path.open(encoding="utf-8") as handle:
        return _object(json.load(handle), path.name)


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _source_contract(value):
    value = _object(value, "Executable source contract")
    commit, dirty = value.get("git_commit"), value.get("git_source_dirty")
    if (not _sha256(value.get("source_tree_sha256"))
            or (commit is not None and (not isinstance(commit, str)
                                       or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None))
            or (dirty is not None and type(dirty) is not bool)):
        raise ValueError("Executable source contract is malformed")


def load_recovery_evidence(previous_dir, *, experiment_root, source_configs, public_train_config,
                           resolved_train_config, current_source_contract, reserve_vram_mib,
                           profile_output_label):
    """Return validated historical diagnostics and the original partial report."""
    previous, experiment = Path(previous_dir).resolve(), Path(experiment_root).resolve()
    if (not previous.is_dir() or previous.parent != experiment
            or not previous.name.startswith("preflight-recovery-")):
        raise ValueError("resume-from must name a previous recovery directory in this experiment")
    if (isinstance(reserve_vram_mib, bool) or not isinstance(reserve_vram_mib, (int, float))
            or not math.isfinite(reserve_vram_mib) or reserve_vram_mib <= 0):
        raise ValueError("Recovery evidence requires an explicit finite positive VRAM reserve")
    recovery_path, profile_path = previous / "recovery.json", previous / "stream-profile.json"
    if recovery_path.resolve().parent != previous or profile_path.resolve().parent != previous:
        raise ValueError("Saved recovery/profile files must remain in their recovery directory")
    recovery, profile = _load(recovery_path), _load(profile_path)
    if recovery.get("schema") != "asgcn_streaming_preflight_recovery_v1":
        raise ValueError("Unsupported saved recovery schema")
    if (recovery.get("experiment_root") != str(experiment)
            or recovery.get("output_root") != str(previous)
            or recovery.get("preflight_report") != str(profile_path)):
        raise ValueError("Saved recovery directory and profile identities disagree")
    if recovery.get("training_executed") is not False or recovery.get("calibration_executed") is not False:
        raise ValueError("Saved recovery must explicitly record that it did not start training/calibration")
    if recovery.get("source_configs") != source_configs:
        raise ValueError("Current raw configuration paths/hashes differ from the previous recovery")
    declared = recovery.get("configs")
    if declared != {kind: str(previous / "configs" / f"{kind}.json") for kind in ("train", "hdr", "aid")}:
        raise ValueError("Saved derived config paths do not belong to the previous recovery")
    for kind, source in source_configs.items():
        expected = experiment / "configs" / f"{kind}.json"
        if (source.get("path") != str(expected) or not _sha256(source.get("sha256"))
                or hashlib.sha256(expected.read_bytes()).hexdigest() != source["sha256"]):
            raise ValueError("Current raw configuration content changed during evidence inspection")
    recovery_sealed = "commitment_sha256" in recovery
    if recovery_sealed:
        core = dict(recovery)
        if core.pop("commitment_sha256") != _digest(core):
            raise ValueError("Saved recovery commitment mismatch")
    if profile.get("schema") != "asgcn_streaming_training_preflight_v1":
        raise ValueError("Previous profile must be a physical streaming report, not a static profile")
    core = dict(profile)
    if core.pop("commitment_sha256", None) != _digest(core):
        raise ValueError("Saved partial profile has no valid commitment; uncommitted progress cannot authorize resume")
    incomplete_statuses = {"running", "interrupted", "failed"}
    if (profile.get("status") not in incomplete_statuses
            or profile.get("passed") is not False or profile.get("report_eligible") is not False
            or recovery.get("status") not in incomplete_statuses
            or recovery.get("report_eligible") is not False):
        raise ValueError("resume-from requires an incomplete, non-reporting recovery")
    if recovery.get("status") == "running" and not recovery_sealed:
        raise ValueError("Running recovery metadata requires a valid commitment before checkpoint resume")
    if profile.get("output") != profile_output_label:
        raise ValueError("Saved profile output identity differs from the referenced file")
    if profile.get("input_config_provenance") != {
        "config": public_train_config, "sha256": _digest(public_train_config),
    }:
        raise ValueError("Saved input config provenance differs from this prepared experiment")
    effective = _object(profile.get("config_provenance"), "Effective config provenance")
    old_config = _object(effective.get("config"), "Effective configuration")
    if effective.get("sha256") != _digest(old_config):
        raise ValueError("Saved effective configuration commitment is invalid")
    old_model = _object(old_config.get("model"), "Saved model")
    guard = old_model.get("max_graph_edges", "missing")
    if guard is not None and (type(guard) is not int or guard <= 0):
        raise ValueError("Saved effective edge guard is invalid")
    normalized = {**old_config, "model": {**_object(old_config.get("model"), "Saved model"),
                  "max_graph_edges": public_train_config["model"]["max_graph_edges"]}}
    if normalized != public_train_config:
        raise ValueError("Saved effective config changed fields other than its measured edge guard")
    saved_train_path = previous / "configs" / "train.json"
    if saved_train_path.exists():
        if saved_train_path.resolve() != saved_train_path:
            raise ValueError("Saved training configuration must remain in the previous recovery")
        expected_train = {**resolved_train_config, "model": {**resolved_train_config["model"],
                          "max_graph_edges": guard}}
        if _load(saved_train_path) != expected_train:
            raise ValueError("Saved training config file differs from its committed effective configuration")
    model = public_train_config["model"]
    storage = model.get("graph_storage", "materialized")
    if (model.get("architecture_version") not in {3, 4} or model.get("graph_execution") != "event_driven"
            or model.get("encoder_kind") != "graph" or storage not in {"materialized", "implicit_radius"}):
        raise ValueError("Saved memory evidence requires a recognized v3/v4 streaming graph backend")
    if model["architecture_version"] == 4:
        from .hierarchy_contract import validate_hierarchy_config
        validate_hierarchy_config(model.get("hierarchy_config"), model["graph_layers"])
        factor = model.get("event_sampling_factor")
        if type(factor) is not int or not 1 <= factor < 2**63:
            raise ValueError("Saved v4 evidence requires the explicit sampling factor")
    recorded_source = profile.get("source_provenance")
    _source_contract(recorded_source)
    _source_contract(current_source_contract)
    data = _object(profile.get("data_provenance"), "Raw data provenance")
    content = _object(data.get("content"), "Raw data content provenance")
    if not _sha256(content.get("sha256")):
        raise ValueError("Saved raw data content provenance is missing or malformed")
    topology = _object(profile.get("topology"), "Topology")
    if model["architecture_version"] == 4 and (
            topology.get("hierarchy_config") != model["hierarchy_config"]
            or topology.get("sampling_contract") != {"factor": model["event_sampling_factor"], "ordinal_origin": 0,
                                                     "counter": "all_raw_events", "reset": "sequence_start_only"}):
        raise ValueError("Saved v4 topology lacks the exact sampling/hierarchy contract")
    if (not isinstance(topology.get("samples"), list)
            or topology.get("arrival_prefix_peak_measured") is not False
            or topology.get("prefix_bound_kind") != "previous_live_window_union_all_current_frame_arrivals"):
        raise ValueError("Saved actual readout and conservative prefix-union scopes are invalid")
    rows = []
    fields = ("readout_nodes", "readout_directed_edges", "prefix_union_nodes_upper_bound",
              "prefix_union_directed_edges_upper_bound")
    for index, row in enumerate(topology["samples"]):
        if row is None:
            continue
        if (not isinstance(row, dict) or type(row.get("dataset_index")) is not int
                or row["dataset_index"] != index
                or any(type(row.get(name)) is not int or row[name] < 0 for name in fields)):
            raise ValueError("Saved per-frame topology contains invalid identities/counts")
        if (row["readout_nodes"] > row["prefix_union_nodes_upper_bound"]
                or row["readout_directed_edges"] > row["prefix_union_directed_edges_upper_bound"]):
            raise ValueError("Saved readout topology exceeds its conservative bound")
        rows.append(row)
    if not rows:
        raise ValueError("Saved profile has no completed topology records to inspect")
    for name in fields:
        cached = topology.get(f"max_{name}")
        if type(cached) is not int or cached != max(row[name] for row in rows):
            raise ValueError("Saved topology maxima disagree with the committed per-frame records")
    scanned, total = topology.get("scanned_samples"), topology.get("dataset_samples")
    if (type(scanned) is not int or type(total) is not int or total != len(topology["samples"])
            or not 0 <= scanned <= len(rows) <= total or type(topology.get("scan_complete")) is not bool
            or (topology["scan_complete"] and scanned != total)):
        raise ValueError("Saved scan coverage is inconsistent with its records")
    edge_bytes, edge_basis_bytes = (24, 56) if storage == "materialized" else (0, 0)
    densest = max(rows, key=lambda row: row["readout_directed_edges"] * edge_basis_bytes + row["readout_nodes"] * 72)
    graph_bytes = densest["readout_directed_edges"] * edge_bytes + densest["readout_nodes"] * 72
    graph_basis_bytes = densest["readout_directed_edges"] * edge_basis_bytes + densest["readout_nodes"] * 72
    gpu = _object(profile.get("runtime_provenance"), "Runtime provenance").get("gpu") or {}
    capacity = _object(gpu, "Saved GPU").get("total_memory_mib")
    if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, (int, float))
                                 or not math.isfinite(capacity) or capacity <= 0):
        raise ValueError("Saved GPU capacity is invalid")
    budget = None if capacity is None else capacity - reserve_vram_mib
    checkpoint = previous / "scan-checkpoint"
    if checkpoint.resolve() != checkpoint:
        raise ValueError("Saved scanner checkpoint must remain in its recovery directory")
    return {
        "schema": "asgcn_saved_recovery_evidence_v1", "report_eligible": False,
        "scope": "historical_partial_scan_diagnosis_only", "exact_resume_performed": False,
        "architecture_version": model["architecture_version"],
        "hierarchy_memory_certified": False,
        "current_data_revalidated": False, "current_device_revalidated": False,
        "source_matches_current": recorded_source["source_tree_sha256"] == current_source_contract["source_tree_sha256"],
        "source_metadata_matches_current": recorded_source == current_source_contract,
        "recorded_source_provenance": recorded_source, "current_source_provenance": current_source_contract,
        "recorded_data_sha256": content["sha256"],
        "previous_recovery": str(previous), "previous_profile": str(profile_path),
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "recovery_sha256": hashlib.sha256(recovery_path.read_bytes()).hexdigest(),
        "recovery_commitment_validated": recovery_sealed, "profile_commitment_validated": True,
        "scanned_samples": scanned, "dataset_samples": total, "scan_complete": topology["scan_complete"],
        "max_readout_directed_edges": topology["max_readout_directed_edges"],
        "max_prefix_union_directed_edges_upper_bound": topology["max_prefix_union_directed_edges_upper_bound"],
        "single_readout_storage_floor": {
            "graph_storage": storage, "edge_storage_materialized": storage == "materialized",
            "dataset_index": densest["dataset_index"], "graph_bytes": graph_bytes,
            "graph_and_basis_bytes": graph_basis_bytes, "graph_and_basis_mib": graph_basis_bytes / 1024**2,
            "formula": ("56 * actual_readout_edges + 72 * readout_nodes" if storage == "materialized"
                        else "72 * readout_nodes"),
            "scope": ("one_materialized_readout_graph_and_basis_excludes_other_graphs_and_training_costs"
                      if storage == "materialized" else
                      "one_implicit_readout_raw_nodes_only_excludes_index_projection_scratch_and_training_costs"),
            "measured_peak_vram": False,
        },
        "saved_device_total_mib": capacity, "requested_reserve_vram_mib": reserve_vram_mib,
        "saved_device_budget_after_requested_reserve_mib": budget,
        "infeasible_on_saved_device": None if budget is None else graph_basis_bytes / 1024**2 > budget,
        "checkpoint_directory": str(checkpoint), "checkpoint_manifest_present": (checkpoint / "latest.json").is_file(),
        "checkpoint_integrity_verified": False,
    }, profile
