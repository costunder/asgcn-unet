"""Separate input contracts, preparation and CUDA gates for persistent streams.

Legacy independent-frame topology reports are deliberately not accepted here.
Config preparation is standard-library-only and never probes or selects a GPU.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPORT_SCHEMA = "asgcn_streaming_training_preflight_v1"
VERIFICATION_SCHEMA = "asgcn_streaming_preflight_verification_v1"
STREAM_CLOCK = "event_local_pending_off_v1"
STREAM_ARRIVAL_POLICY = "simultaneous_equal_timestamp"


def is_streaming_config(config: dict[str, Any]) -> bool:
    model = config.get("model", {})
    return model.get("graph_execution") == "event_driven" or model.get("architecture_version") == 3


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be explicitly supplied as a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be explicitly supplied as a finite positive number")
    return result


def validate_streaming_contract(config: dict[str, Any], *, training: bool = False) -> None:
    model, dataset = config.get("model", {}), config.get("dataset", {})
    if model.get("architecture_version") != 3 or model.get("graph_execution") != "event_driven":
        raise ValueError("Streaming requires architecture_version=3 and graph_execution=event_driven")
    if model.get("encoder_kind") != "graph" or model.get("event_sampling_factor") != 1:
        raise ValueError("Streaming requires the graph encoder and event_sampling_factor=1")
    stream = model.get("stream_config")
    if not isinstance(stream, dict):
        raise TypeError("model.stream_config must explicitly define its physical time contract")
    if set(stream) != {
        "window_seconds", "time_scale_seconds", "node_time_feature", "clock", "arrival_policy",
    }:
        raise ValueError("stream_config has missing or unsupported contract fields")
    _positive_number(stream["window_seconds"], "window_seconds")
    _positive_number(stream["time_scale_seconds"], "time_scale_seconds")
    if (
        stream["node_time_feature"] != "physical_frame_offset"
        or stream["clock"] != STREAM_CLOCK or stream["arrival_policy"] != STREAM_ARRIVAL_POLICY
    ):
        raise ValueError("Unsupported streaming time-feature/clock/arrival policy")
    if dataset.get("event_time_contract") != "physical_seconds_v1":
        raise ValueError("Streaming datasets must declare event_time_contract=physical_seconds_v1")
    _positive_number(dataset.get("timestamp_scale_to_seconds"), "timestamp_scale_to_seconds")
    _positive_number(dataset.get("interval_timestamp_scale_to_seconds"), "interval_timestamp_scale_to_seconds")
    if "max_events" not in dataset or dataset["max_events"] is not None:
        raise ValueError("Streaming requires explicit dataset.max_events=null; frame linspace caps are invalid")
    if training and (
        dataset.get("type") != "eventhdr" or config.get("train", {}).get("batching") != "independent_sequences"
    ):
        raise ValueError("Streaming training preflight requires independent EventHDR sequence batches")
    if "train" in config and config["train"].get("validation_context_frames", "missing") is not None:
        raise ValueError("Streaming validation_context_frames must be null for exact causal context")
    if "eval" in config and config["eval"].get("recurrent_context_frames", "missing") is not None:
        raise ValueError("Streaming recurrent_context_frames must be null for exact causal context")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _differences(old: Any, new: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(old.keys() | new.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in old:
                result.append({"field": path, "before_present": False, "after": new[key]})
            elif key not in new:
                result.append({"field": path, "before": old[key], "after_present": False})
            else:
                result.extend(_differences(old[key], new[key], path))
        return result
    return [] if old == new else [{"field": prefix, "before": old, "after": new}]


def prepare_streaming_experiment(
    project_root: str | Path, output_root: str | Path, *, window_seconds: float,
    time_scale_seconds: float, hdr_timestamp_scale_to_seconds: float,
    aid_timestamp_scale_to_seconds: float,
    hdr_interval_timestamp_scale_to_seconds: float,
    aid_interval_timestamp_scale_to_seconds: float,
) -> dict[str, Any]:
    """Create exclusive new configs; preserve every nonapproved baseline field."""
    project = Path(project_root).resolve()
    destination = Path(output_root)
    destination = (destination if destination.is_absolute() else project / destination).resolve()
    if destination == project or not destination.is_relative_to(project):
        raise ValueError("output_root must be a new directory inside the owning project")
    if destination.exists():
        raise FileExistsError(f"Streaming output root already exists: {destination}")
    window = _positive_number(window_seconds, "window_seconds")
    scale = _positive_number(time_scale_seconds, "time_scale_seconds")
    hdr_scale = _positive_number(hdr_timestamp_scale_to_seconds, "hdr_timestamp_scale_to_seconds")
    aid_scale = _positive_number(aid_timestamp_scale_to_seconds, "aid_timestamp_scale_to_seconds")
    hdr_interval_scale = _positive_number(hdr_interval_timestamp_scale_to_seconds, "hdr_interval_timestamp_scale_to_seconds")
    aid_interval_scale = _positive_number(aid_interval_timestamp_scale_to_seconds, "aid_interval_timestamp_scale_to_seconds")
    # Never nest a new study in a configured model's checkpoints/evaluation output,
    # nor let it contain an existing source/data/config tree.
    protected = [project / "src", project / "scripts", project / "configs", project / "data"]
    for path in (project / "configs").rglob("*.json"):
        with path.open(encoding="utf-8") as handle:
            item = json.load(handle)
        if not isinstance(item, dict):
            continue
        for section, key in (("output", "run_dir"), ("eval", "output_dir")):
            value = item.get(section, {}).get(key)
            if value:
                root = Path(value)
                protected.append((root if root.is_absolute() else project / root).resolve())
    if any(destination.is_relative_to(path) or path.is_relative_to(destination) for path in protected):
        raise ValueError("Streaming output root overlaps an existing experiment, source, config, or data path")
    configs, source, differences = {}, {}, {}
    relative = destination.relative_to(project).as_posix()
    for kind in ("train", "hdr", "aid"):
        base_path = project / "configs" / "ablations" / f"graph_unet-{kind}.json"
        with base_path.open(encoding="utf-8") as handle:
            base = json.load(handle)
        current = copy.deepcopy(base)
        current["model"].update({
            "architecture_version": 3, "graph_execution": "event_driven",
            "stream_config": {
                "window_seconds": window, "time_scale_seconds": scale,
                "node_time_feature": "physical_frame_offset", "clock": STREAM_CLOCK,
                "arrival_policy": STREAM_ARRIVAL_POLICY,
            },
        })
        current["dataset"].update({
            "max_events": None, "event_time_contract": "physical_seconds_v1",
            "timestamp_scale_to_seconds": aid_scale if kind == "aid" else hdr_scale,
            "interval_timestamp_scale_to_seconds": aid_interval_scale if kind == "aid" else hdr_interval_scale,
        })
        if kind == "train":
            current["output"]["run_dir"] = f"{relative}/train"
            current["train"]["validation_context_frames"] = None
        else:
            current["eval"]["output_dir"] = f"{relative}/eval/{kind}"
            current["eval"]["recurrent_context_frames"] = None
        validate_streaming_contract(current, training=kind == "train")
        configs[kind] = current
        source[kind] = {"path": base_path.relative_to(project).as_posix(), "config_sha256": _digest(base)}
        differences[kind] = _differences(base, current)
    train = configs["train"]
    if (train["model"]["graph_layers"], train["model"]["hidden_dim"],
        train["model"]["decoder_channels"], train["train"]["batch_size"], train["train"]["epochs"]
    ) != (6, 64, 48, 16, 40):
        raise ValueError("Checked-in baseline differs from the approved full 6-layer/64/48/B16/40-epoch design")
    report = {
        "schema": "asgcn_streaming_experiment_preparation_v1", "report_eligible": False,
        "output_root": relative, "source_configs": source, "changes": differences,
        "training_required": "new independent training; old checkpoints are config-incompatible",
        "execution_performed": False,
        "warnings": [
            "Physical timestamp units must be verified from each original dataset, not guessed.",
            "The finite sliding window is a new input/state contract, not the previous frame graph.",
            "No frame linspace cap remains; explicit edge guards may fail and are never raised automatically.",
            "Existing static topology profiles and dense-frame probe indices do not certify this stream.",
            "Event timestamp and frame-interval timestamp scales are independent and both explicitly recorded.",
            "Validation and evaluation replay full causal context; bounded predecessor truncation is disabled.",
        ],
        "configs": {kind: f"{relative}/configs/{kind}.json" for kind in configs},
        "preflight_output": f"{relative}/stream-profile.json",
    }
    destination.mkdir(parents=True, exist_ok=False)
    config_directory = destination / "configs"
    config_directory.mkdir()
    for kind, config in configs.items():
        with (config_directory / f"{kind}.json").open("x", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    with (destination / "preparation.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return report


def _topology_model(config):
    """Only structural metadata; never allocate a full network for a graph scan."""
    model = config["model"]
    return SimpleNamespace(
        stream_config=model["stream_config"], graph_radius=model["graph_radius"],
        graph_position_dims=model["graph_position_dims"], graph_chunk_size=model["graph_chunk_size"],
        max_graph_edges=model["max_graph_edges"], event_sampling_factor=model["event_sampling_factor"],
        snn_dynamics=model["snn_dynamics"], decoder_kind=model["decoder_kind"],
        raster_downsample=model["raster_downsample"],
        encoder=SimpleNamespace(hidden_dim=model["hidden_dim"], layers=(None,) * model["graph_layers"]),
    )


def _scan_stream_topology(dataset, config, device, batches, *, top_density_count,
                          progress=None, on_progress=None):
    import torch
    from tqdm import tqdm

    from .batching import sequence_key
    from .preflight import _load_packed_probe_batch
    from .stream_model import _metadata, _prepared
    from .stream_topology import count_stream_topology_update

    model = _topology_model(config)
    states, records = {}, [None] * len(dataset)
    result = {} if progress is None else progress
    result.update({
        "scope": "complete_eventhdr_training_stream", "dataset_samples": len(dataset),
        "scanned_samples": 0, "scan_complete": False,
        "counting_method": "packed_float64_radius_incremental_count_only_v1",
        "unchanged_previous_edge_counts_reused": True,
        "full_edge_tensors_materialized": False,
        "arrival_prefix_peak_measured": False,
        "prefix_bound_kind": "previous_live_window_union_all_current_frame_arrivals",
        "statement": "Readout counts are actual; prefix counts are conservative upper bounds, not measured maxima.",
        "max_readout_nodes": 0, "max_readout_directed_edges": 0,
        "max_prefix_union_nodes_upper_bound": 0,
        "max_prefix_union_directed_edges_upper_bound": 0,
        "peak_candidate_pairs": 0, "candidate_pair_budget": 1_048_576,
        "top_density_samples": [], "samples": records, "current_batch_indices": [],
    })
    seen = set()
    final = {sequence_key(item): item["sequence_index"] for item in dataset.samples}
    with torch.no_grad():
        for indices in tqdm(batches, desc="stream-preflight-window-topology"):
            result["current_batch_indices"] = list(indices)
            if any(index in seen for index in indices):
                raise ValueError("Streaming topology schedule repeats frames")
            samples, _ = _load_packed_probe_batch(dataset, indices, device)
            previous = [states.get(sequence_key(sample)) for sample in samples]
            # Input validation is shared; this scanner retains raw nodes/clocks,
            # never an incomplete edge-free graph passed to a model/state API.
            metadata, contract = _metadata(model, samples, [None] * len(samples))
            features, positions, timestamps, node_batch = _prepared(model, samples, metadata)
            del features
            # Inference starts from the previous watermark, not necessarily the
            # next frame start: a proven predecessor row may precede that start
            # after a gap. Keep the complete previous live window in the bound.
            initial_watermarks = timestamps.new_tensor([
                old.watermark_seconds if old is not None else record[2]
                for old, record in zip(previous, metadata, strict=True)
            ])
            ends = timestamps.new_tensor([record[3] for record in metadata])
            window = model.stream_config["window_seconds"]
            for old, record in zip(previous, metadata, strict=True):
                if old is not None and (
                    old.contract != contract or old.sequence_identity != record[0]
                    or old.sequence_index + 1 != record[1] or old.origin_seconds != record[4]
                    or old.watermark_seconds > record[2]
                ):
                    raise ValueError("Streaming topology clock/sequence continuity mismatch or overlapping frame intervals")
            # Every actual arrival-prefix graph is a subgraph of this union. It
            # deliberately keeps events which will expire later inside the frame.
            # This count is a conservative bound, NOT an observed prefix maximum.
            incoming_count = len(timestamps)
            previous_edge_counts = torch.tensor(
                [old.directed_edges if old is not None else 0 for old in previous],
                device=device, dtype=torch.long,
            )
            positions = torch.cat([old.positions for old in previous if old is not None] + [positions])
            timestamps = torch.cat([old.timestamps for old in previous if old is not None] + [timestamps])
            node_batch = torch.cat([
                node_batch.new_full((len(old.timestamps),), lane)
                for lane, old in enumerate(previous) if old is not None
            ] + [node_batch])
            union_keep = timestamps >= (initial_watermarks - window)[node_batch]
            readout_keep = timestamps >= (ends - window)[node_batch]
            is_arrival = torch.arange(len(timestamps), device=device) >= len(timestamps) - incoming_count
            measured = count_stream_topology_update(
                positions, node_batch, union_keep, readout_keep, is_arrival, previous_edge_counts,
                batch_size=len(samples),
                radius=model.graph_radius, position_dims=model.graph_position_dims,
                chunk_size=model.graph_chunk_size,
            )
            counts = torch.stack((measured.readout_nodes, measured.readout_directed_edges,
                                  measured.union_nodes, measured.union_directed_edges)).cpu().tolist()
            result["peak_candidate_pairs"] = max(result["peak_candidate_pairs"], measured.peak_candidate_pairs)
            # Packed permutation followed by per-lane ownership only, not per-sample graph/model computation.
            order = torch.argsort(node_batch[readout_keep], stable=True)
            readout_positions = positions[readout_keep][order]
            readout_times = timestamps[readout_keep][order]
            offset = 0
            for lane, (index, sample, record) in enumerate(zip(indices, samples, metadata, strict=True)):
                actual_nodes, actual_edges, bound_nodes, bound_edges = [values[lane] for values in counts]
                if actual_nodes > bound_nodes or actual_edges > bound_edges:
                    raise RuntimeError("A readout graph exceeds its conservative arrival-prefix bound")
                records[index] = {
                    "dataset_index": index, "sample_id": sample.get("sample_id", str(index)),
                    "sequence_identity": list(record[0]), "sequence_index": record[1],
                    "interval_start_seconds": record[2], "interval_end_seconds": record[3],
                    "incoming_events": samples.event_counts[lane],
                    "arrival_groups": len(record[5]),
                    "readout_nodes": actual_nodes, "readout_directed_edges": actual_edges,
                    "prefix_union_nodes_upper_bound": bound_nodes,
                    "prefix_union_directed_edges_upper_bound": bound_edges,
                }
                state = SimpleNamespace(
                    positions=readout_positions[offset:offset + actual_nodes].clone(),
                    timestamps=readout_times[offset:offset + actual_nodes].clone(),
                    origin_seconds=record[4], watermark_seconds=record[3],
                    sequence_index=record[1], sequence_identity=record[0], contract=contract,
                    directed_edges=actual_edges,
                )
                offset += actual_nodes
                if record[1] == final[record[0]]:
                    states.pop(record[0], None)
                else:
                    states[record[0]] = state
                for field in ("readout_nodes", "readout_directed_edges", "prefix_union_nodes_upper_bound",
                              "prefix_union_directed_edges_upper_bound"):
                    result[f"max_{field}"] = max(result[f"max_{field}"], records[index][field])
            seen.update(indices)
            result["scanned_samples"] = len(seen)
            result["top_density_samples"] = sorted(
                result["top_density_samples"] + [records[index] for index in indices],
                key=lambda row: (-row["prefix_union_directed_edges_upper_bound"],
                                 -row["readout_directed_edges"], row["dataset_index"]),
            )[:top_density_count]
            if on_progress is not None:
                on_progress()
            del samples, previous, positions, timestamps, node_batch, readout_positions, readout_times
    if seen != set(range(len(dataset))) or any(record is None for record in records):
        raise ValueError("Streaming topology scan did not cover every training frame exactly once")
    result.update(scan_complete=True, current_batch_indices=[])
    return result


def _stream_probe_plan(batches, records, batch_size, profile_samples):
    if not records or not batches or max(map(len, batches)) != batch_size:
        raise ValueError("Streaming preflight cannot form the full configured physical batch")
    if len(batches) < profile_samples:
        raise ValueError("Fewer scheduled batches than requested streaming probes")
    flattened = [index for batch in batches for index in batch]
    if sorted(flattened) != list(range(len(records))):
        raise ValueError("Streaming probe schedule must cover each frame exactly once")
    entries = [{
        "batch_index": number, "dataset_indices": list(indices), "batch_size": len(indices),
        "readout_nodes": sum(records[index]["readout_nodes"] for index in indices),
        "readout_directed_edges": sum(records[index]["readout_directed_edges"] for index in indices),
        "prefix_union_directed_edges_upper_bound": sum(
            records[index]["prefix_union_directed_edges_upper_bound"] for index in indices
        ),
    } for number, indices in enumerate(batches)]
    ranked = sorted(entries, key=lambda entry: (
        -entry["prefix_union_directed_edges_upper_bound"], -entry["readout_directed_edges"], entry["batch_index"],
    ))
    selected = {entry["batch_index"] for entry in ranked[:profile_samples]}
    selected.add(0)
    selected.add(next(entry["batch_index"] for entry in entries if entry["batch_size"] == batch_size))
    selected.add(max(entries, key=lambda entry: entry["readout_nodes"])["batch_index"])
    selected.add(max(entries, key=lambda entry: entry["readout_directed_edges"])["batch_index"])
    resident_peak_batch = _graph_storage_floor({"samples": records}, batches)["batch_index"]
    if resident_peak_batch is not None:
        selected.add(resident_peak_batch)
    sparse = min((row for row in records if row["readout_nodes"] > 0),
                 key=lambda row: row["readout_nodes"], default=None)
    for entry in entries:
        if any(records[index]["readout_nodes"] == 0 for index in entry["dataset_indices"]):
            selected.add(entry["batch_index"])
            break
    if sparse is not None:
        selected.add(next(entry["batch_index"] for entry in entries
                          if sparse["dataset_index"] in entry["dataset_indices"]))
    from .training import batching_contract
    return {
        "schema": "asgcn_streaming_batch_probe_plan_v1", "schedule_sha256": _digest(batches),
        "batching_contract": batching_contract(batch_size), "requested_batch_size": batch_size,
        "scheduled_frames": len(records), "scheduled_batches": len(batches),
        "largest_actual_batch_size": batch_size,
        "selected_batch_indices": sorted(selected),
        "selected_batches": [entries[index] for index in sorted(selected)],
        "context_policy": "chronological_training_mode_replay_from_sequence_start_no_window_truncation",
        "replay_stop_batch": max(selected),
    }


def _probe_stream_training(dataset, config, device, batches, topology, plan, *, reserve_vram_mib=0, progress=None):
    import torch
    from tqdm import tqdm

    from .batching import sequence_key
    from .engine import (
        _build_optimizer,
        _make_grad_scaler,
        _optimizer_mode,
        _training_step,
        build_model,
    )
    from .losses import ReconstructionLoss
    from .preflight import _load_packed_probe_batch
    from .training import TrainingState, forward_training_loss
    from .utils import set_seed

    if (isinstance(reserve_vram_mib, bool) or not isinstance(reserve_vram_mib, (int, float))
            or not math.isfinite(reserve_vram_mib) or reserve_vram_mib < 0):
        raise ValueError("reserve_vram_mib must be a finite nonnegative number")
    if progress is not None and not isinstance(progress, dict):
        raise TypeError("Streaming probe progress must be a dictionary")
    progress = {} if progress is None else progress
    measured = []
    progress.update({
        "passed": False, "plan": plan, "steps": measured, "current_batch_index": None,
        "failed_batch_index": None, "phase": "initializing", "completed_batches": 0,
        "replayed_predecessor_frames": 0, "reserve_vram_mib": float(reserve_vram_mib),
        "peak_allocated_mib": None, "peak_reserved_mib": None,
        "minimum_observed_device_free_mib": None,
        "memory_scope": "model_setup_and_every_chronological_batch_input_through_state_commit_and_release",
        "reserve_scope": "live_device_free_pre_and_post_batch_and_allocator_peak_against_device_total",
        "reserve_is_hard_isolation": False,
    })
    started = time.perf_counter()

    def memory_snapshot(*, check_peak=False):
        if device.type != "cuda":
            return {"peak_allocated_mib": None, "peak_reserved_mib": None,
                    "device_free_mib": None, "device_total_mib": None}
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        free_mib, total_mib = free / 1024**2, total / 1024**2
        progress["peak_allocated_mib"] = max(progress["peak_allocated_mib"] or 0, allocated)
        progress["peak_reserved_mib"] = max(progress["peak_reserved_mib"] or 0, reserved)
        minimum = progress["minimum_observed_device_free_mib"]
        progress["minimum_observed_device_free_mib"] = free_mib if minimum is None else min(minimum, free_mib)
        snapshot = {"peak_allocated_mib": allocated, "peak_reserved_mib": reserved,
                    "device_free_mib": free_mib, "device_total_mib": total_mib}
        progress["last_memory_snapshot"] = snapshot
        if free_mib < reserve_vram_mib:
            raise RuntimeError(
                f"Streaming probe VRAM reserve failed during {progress['phase']}: "
                f"live device free={free_mib:.2f} MiB < reserve={reserve_vram_mib:.2f} MiB. "
                "No cache eviction, fallback, batch reduction, or continuation was attempted."
            )
        if check_peak and reserved > total_mib - reserve_vram_mib:
            raise RuntimeError(
                f"Streaming probe allocator peak exceeded device-total-minus-reserve during {progress['phase']}: "
                f"peak reserved={reserved:.2f} MiB; budget={total_mib - reserve_vram_mib:.2f} MiB. "
                "This allocator comparison does not account for other processes; live free memory is checked separately."
            )
        return snapshot

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    memory_snapshot()
    set_seed(int(config["seed"]))
    model = build_model(config["model"]).to(device).train()
    criterion = ReconstructionLoss(config["train"].get("loss_weights"))
    optimizer = _build_optimizer(model, config["train"])
    amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    scaler = _make_grad_scaler(amp)
    temporal_weight = float((config["train"].get("loss_weights") or {}).get("temporal", 0.0))
    state = TrainingState(independent_sequences=True)
    final = {sequence_key(item): item["sequence_index"] for item in dataset.samples}
    selected = set(plan["selected_batch_indices"])
    replayed_frames = 0
    progress["phase"] = "model_setup"
    memory_snapshot(check_peak=True)
    for number, indices in enumerate(tqdm(batches[:plan["replay_stop_batch"] + 1], desc="stream-preflight-stateful-train")):
        progress.update({"current_batch_index": number, "current_dataset_indices": list(indices),
                         "phase": "before_input_load"})
        try:
            memory_snapshot(check_peak=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            step_started = time.perf_counter()
            samples, input_pipeline = _load_packed_probe_batch(dataset, indices, device)
            contexts = state.prepare(samples)
            if number not in selected:
                progress["phase"] = "predecessor_replay"
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
                    prediction, diagnostics = model.forward_training_batch(samples, [entry[0] for entry in contexts])
                target = samples.targets
                if not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError("Non-finite reconstruction during causal predecessor replay")
            else:
                progress["phase"] = "forward_loss_backward_optimizer"
                def forward_loss(current_samples=samples, incoming_contexts=contexts):
                    return forward_training_loss(
                        model, criterion, current_samples, incoming_contexts, batch_mode=True, amp_enabled=amp,
                        temporal_weight=temporal_weight,
                    )
                payload, loss, gradient_norm, amp_info = _training_step(
                    model, optimizer, scaler, forward_loss, optimizer_mode=_optimizer_mode(config["train"]),
                    max_norm=float(config["train"]["grad_clip"]), epoch=0, step=number,
                    sample_id="stream-preflight:" + ",".join(map(str, indices)),
                )
                prediction, diagnostics, target = payload
                if prediction.shape != target.shape or prediction.shape[0] != len(indices):
                    raise RuntimeError("Streaming probe reconstruction does not match the actual batch target")
                for index, detail in zip(indices, diagnostics, strict=True):
                    expected = topology["samples"][index]
                    if detail["nodes"] != expected["readout_nodes"] or detail["edges"] != expected["readout_directed_edges"]:
                        raise RuntimeError("Streaming model readout topology differs from the full causal scan")
                    if not detail.get("stream_execution", {}).get("training_dense_snapshot"):
                        raise RuntimeError("Streaming preflight did not execute the declared causal-window training path")
            progress["phase"] = "state_commit_and_release"
            state.commit(samples, prediction, diagnostics, target)
            state.release_finished(samples, final)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - step_started) * 1000
            progress["phase"] = "after_state_commit_and_release"
            snapshot = memory_snapshot(check_peak=True)
            if number in selected:
                measured.append({
                    "batch_index": number, "dataset_indices": list(indices), "batch_size": len(indices),
                    "incoming_contexts": sum(context[0] is not None for context in contexts),
                    "loss": loss, "gradient_norm": gradient_norm, "amp": amp_info,
                    "step_time_ms": elapsed, "frames_per_second": len(indices) * 1000 / elapsed,
                    "input_pipeline": input_pipeline, "prediction_shape": list(prediction.shape),
                    **snapshot,
                    "scope": "input_load_stateful_forward_loss_backward_optimizer_state_commit_and_release",
                })
            else:
                replayed_frames += len(samples)
            progress.update({"completed_batches": number + 1, "phase": "batch_completed",
                             "replayed_predecessor_frames": replayed_frames,
                             "elapsed_including_context_replay_seconds": time.perf_counter() - started})
        except (KeyboardInterrupt, OSError, ValueError, TypeError, KeyError, RuntimeError, FloatingPointError) as error:
            progress.update({"failed_batch_index": number, "failure_type": type(error).__name__,
                             "elapsed_including_context_replay_seconds": time.perf_counter() - started})
            raise
        if number in selected:
            del payload, forward_loss
        del prediction, diagnostics, target, samples, contexts
    if [row["batch_index"] for row in measured] != plan["selected_batch_indices"]:
        raise RuntimeError("Not every selected streaming training batch completed")
    progress.update({
        "passed": True, "phase": "completed", "current_batch_index": None, "current_dataset_indices": [],
        "replayed_predecessor_frames": replayed_frames,
        "elapsed_including_context_replay_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "training_protocol_scope": "ANN causal-window learning; event-driven inference is a separate evaluation",
    })
    return progress


def _scope():
    return {
        "name": "streaming_stateful_physical_batch_training_steps",
        "topology_scope": "complete_eventhdr_training_stream", "absolute_vram_guarantee": False,
        "statement": (
            "Complete chronological readout topology plus conservative arrival-prefix union bounds; "
            "selected actual stateful physical-batch forward/loss/backward/optimizer probes. "
            "This is not an exact arrival-prefix maximum or an inference-memory/performance certification."
        ),
    }


def _graph_storage_floor(topology, batches):
    """Necessary graph/basis storage only; NOT an estimate of total training peak.

    A directed edge uses int64[2] + float64[1] (24 bytes). A node uses
    float32[4] + float64[4] + degree/time/batch int64-sized vectors (72 bytes).
    Old persistent sequence graphs coexist with the current readout graph and
    its long[E,2]/float64[E,2] linear spline basis (32 additional bytes per edge).
    Activations, autograd, model/optimizer, decoder states and copies are extra.
    """
    records = topology["samples"]
    final = {tuple(row["sequence_identity"]): row["sequence_index"] for row in records}
    resident = {}
    largest = {"bytes": 0, "batch_index": None, "dataset_indices": []}
    for number, indices in enumerate(batches):
        current = [(tuple(records[index]["sequence_identity"]), records[index]) for index in indices]
        required = sum(resident.values()) + sum(
            row["readout_directed_edges"] * 56 + row["readout_nodes"] * 72 for _, row in current
        )
        if required > largest["bytes"]:
            largest = {"bytes": required, "batch_index": number, "dataset_indices": list(indices)}
        for identity, row in current:
            if row["sequence_index"] == final[identity]:
                resident.pop(identity, None)
            else:
                resident[identity] = row["readout_directed_edges"] * 24 + row["readout_nodes"] * 72
    return {**largest, "mib": largest["bytes"] / 1024**2,
            "scope": "necessary_previous_raw_graphs_plus_current_readout_graph_and_linear_spline_basis",
            "total_training_peak_estimate": False}


def _cuda_memory_budget(device, reserve_vram_mib):
    import torch

    if device.type != "cuda":
        return {"measured": False, "scope": "cpu_smoke_only", "reserve_vram_mib": reserve_vram_mib}
    free, total = torch.cuda.mem_get_info(device)
    allocated, reserved = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
    reusable = max(0, reserved - allocated)
    return {
        "measured": True, "free_mib": free / 1024**2, "total_mib": total / 1024**2,
        "allocated_mib": allocated / 1024**2, "reserved_mib": reserved / 1024**2,
        "allocator_reusable_mib": reusable / 1024**2, "reserve_vram_mib": reserve_vram_mib,
        "available_after_reserve_mib": (free + reusable) / 1024**2 - reserve_vram_mib,
        "scope": "current_device_snapshot_not_a_peak_guarantee",
    }


def streaming_training_preflight(
    config, output_path, *, profile_samples=3, top_density_count=10, require_cuda=True,
    resume_scan=False, reuse_report=None, measured_guard_config_output=None, reserve_vram_mib=0,
):
    import torch

    from .data import build_dataset
    from .engine import (
        _artifact_path_label,
        _current_source_contract,
        _enforce_training_split_status,
        _public_config,
        _training_protocol,
    )
    from .preflight import _data_provenance, _make_batch_sampler, _runtime_provenance, _safe_failure
    from .utils import resolve_device, save_json, validate_experiment_config

    validate_experiment_config(config)
    validate_streaming_contract(config, training=True)
    config = copy.deepcopy(config)
    original_edge_guard = config["model"].get("max_graph_edges")
    if original_edge_guard is not None and (type(original_edge_guard) is not int or original_edge_guard < 1):
        raise ValueError("max_graph_edges must be a positive integer or None")
    derived_path = None if measured_guard_config_output is None else Path(measured_guard_config_output)
    if derived_path is not None:
        _positive_number(reserve_vram_mib, "reserve_vram_mib")
        if derived_path.exists() or derived_path.resolve() == Path(output_path).resolve():
            raise FileExistsError("Measured-guard config must be a new file separate from the report")
    elif reserve_vram_mib != 0:
        raise ValueError("reserve_vram_mib requires an explicit measured-guard config output")
    if resume_scan or reuse_report is not None:
        raise ValueError("Streaming preflight cannot reuse a static report or resume without serialized causal graph state; choose a new output")
    if (type(profile_samples) is not int or type(top_density_count) is not int
            or profile_samples < 1 or top_density_count < profile_samples):
        raise ValueError("Streaming profile_samples/top_density_count must be positive with top_density_count >= profile_samples")
    if require_cuda and config["train"].get("batch_size") != 16:
        raise ValueError("The approved full streaming CUDA gate requires physical batch_size=16")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"Streaming preflight output already exists: {destination}")
    device = resolve_device(config.get("device", "auto"))
    cuda_ready = device.type == "cuda" and torch.cuda.is_available()
    if require_cuda and not cuda_ready:
        raise RuntimeError("Streaming training preflight requires the explicitly allocated CUDA device")
    public_config = _public_config(config)
    report = {
        "schema": REPORT_SCHEMA, "status": "running", "passed": False, "report_eligible": False,
        "output": _artifact_path_label(destination), "measurement_scope": _scope(),
        "request": {"require_cuda": require_cuda, "profile_samples": profile_samples,
                    "top_density_count": top_density_count, "use_measured_edge_guard": derived_path is not None,
                    "reserve_vram_mib": reserve_vram_mib},
        "config_provenance": {"config": public_config, "sha256": _digest(public_config)},
        "source_provenance": _current_source_contract(), "runtime_provenance": _runtime_provenance(device),
        "input_config_provenance": {"config": public_config, "sha256": _digest(public_config)},
        "data_provenance": None, "topology": None, "batch_training_probe": None,
        "guard_measurement": None, "stage": "initialization",
        "checks": {"complete_topology_scan": False, "conservative_prefix_edge_guard": False,
                   "stateful_forward_backward": False, "cuda_available": cuda_ready},
        "training_protocol": _training_protocol(config, device),
        "failure": None,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    dataset = None
    try:
        report["stage"] = "dataset_identity"
        _enforce_training_split_status(config)
        dataset = build_dataset(config["dataset"], split="train")
        report["data_provenance"] = _data_provenance(dataset, config)
        batches = list(_make_batch_sampler(dataset, config))
        report["stage"] = "count_only_topology"
        report["topology"] = {}
        last_saved = time.monotonic()
        def persist_progress():
            nonlocal last_saved
            if time.monotonic() - last_saved >= 30:
                save_json(destination, report)
                last_saved = time.monotonic()
        _scan_stream_topology(
            dataset, config, device, batches, top_density_count=top_density_count,
            progress=report["topology"], on_progress=persist_progress,
        )
        report["checks"]["complete_topology_scan"] = True
        report["stage"] = "measured_edge_guard"
        topology = report["topology"]
        original_guard = config["model"]["max_graph_edges"]
        required = topology["max_prefix_union_directed_edges_upper_bound"]
        measured_guard = max(original_guard or 0, required, 1)
        storage_floor = _graph_storage_floor(topology, batches)
        memory = _cuda_memory_budget(device, reserve_vram_mib)
        report["guard_measurement"] = {
            "configured_max_graph_edges": original_guard, "measured_union_required_guard": required,
            "proposed_max_graph_edges": measured_guard, "effective_max_graph_edges": original_guard,
            "explicit_measured_guard_requested": derived_path is not None,
            "selection": "max(original_guard_or_zero, full_train_prefix_union_edges, 1)",
            "topology_sha256": _digest(topology), "raw_graph_storage_floor": storage_floor,
            "device_memory": memory, "evaluation_memory_certified": False,
        }
        print(f"Full stream count: {topology['scanned_samples']}/{topology['dataset_samples']} frames; "
              f"max readout edges={topology['max_readout_directed_edges']:,}; "
              f"conservative prefix-union edges={required:,}.", flush=True)
        save_json(destination, report)
        if derived_path is None and original_guard is not None and required > original_guard:
            raise RuntimeError(
                f"Full count-only scan completed: conservative prefix-union requires {required:,} directed edges "
                f"but configured max_graph_edges={original_guard:,}. No model was allocated or guard changed. "
                "Use recover_streaming_preflight.py with explicit --use-measured-edge-guard and a VRAM reserve "
                "to create a new config and measure the unchanged physical batch."
            )
        report["stage"] = "raw_graph_memory_floor"
        if memory["measured"] and storage_floor["mib"] > memory["available_after_reserve_mib"]:
            raise RuntimeError(
                f"Necessary graph and spline-basis storage alone is {storage_floor['mib']:.1f} MiB, exceeding "
                f"{memory['available_after_reserve_mib']:.1f} MiB currently available after the explicit reserve. "
                "This excludes activations/optimizer/copies; no model was allocated and no scale was reduced."
            )
        if derived_path is not None:
            config["model"]["max_graph_edges"] = measured_guard
            report["guard_measurement"]["effective_max_graph_edges"] = measured_guard
            report["guard_measurement"]["derived_config"] = _artifact_path_label(derived_path)
            public_config = _public_config(config)
            report["config_provenance"] = {"config": public_config, "sha256": _digest(public_config)}
            report["training_protocol"] = _training_protocol(config, device)
            derived_path.parent.mkdir(parents=True, exist_ok=True)
            with derived_path.open("x", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
            print(f"Explicit measured guard: {original_guard} -> {measured_guard}; "
                  f"physical batch remains {config['train']['batch_size']}. Starting stateful training probes.", flush=True)
        report["checks"]["conservative_prefix_edge_guard"] = True
        report["stage"] = "stateful_training_probe"
        plan = _stream_probe_plan(batches, report["topology"]["samples"], config["train"]["batch_size"], profile_samples)
        report["batch_training_probe"] = {}
        _probe_stream_training(dataset, config, device, batches, report["topology"], plan,
                               reserve_vram_mib=reserve_vram_mib, progress=report["batch_training_probe"])
        report["checks"]["stateful_forward_backward"] = True
        report["stage"] = "final_identity_verification"
        if _current_source_contract() != report["source_provenance"]:
            raise ValueError("Executable source changed during the streaming preflight")
        # Rehash every source without the training hash cache: path/size alone
        # cannot detect an in-place, same-size edit during the scan or probes.
        if _data_provenance(dataset, config) != report["data_provenance"]:
            raise ValueError("Dataset content or provenance changed during the streaming preflight")
        report["passed"] = True
        report["report_eligible"] = bool(cuda_ready and require_cuda)
        report["status"] = "passed" if report["report_eligible"] else "cpu_smoke_passed_non_reporting"
        report["stage"] = "complete"
    except KeyboardInterrupt as error:
        report["status"] = "interrupted"
        report["failure"] = _safe_failure(error, config, destination)
        report["failure"]["stage"] = report["stage"]
        report["failure"]["probe_phase"] = (report.get("batch_training_probe") or {}).get("phase")
        report["failure"]["dataset_indices"] = (report.get("batch_training_probe") or {}).get(
            "current_dataset_indices", (report.get("topology") or {}).get("current_batch_indices", []),
        )
        report["commitment_sha256"] = _digest(report)
        save_json(destination, report)
        raise
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, FloatingPointError) as error:
        report["status"] = "failed"
        report["failure"] = _safe_failure(error, config, destination)
        report["failure"]["stage"] = report["stage"]
        report["failure"]["probe_phase"] = (report.get("batch_training_probe") or {}).get("phase")
        report["failure"]["dataset_indices"] = (report.get("batch_training_probe") or {}).get(
            "current_dataset_indices", (report.get("topology") or {}).get("current_batch_indices", []),
        )
        report["failure"]["scope_note"] = (
            "A union-bound guard failure is conservative and does not prove an actual arrival prefix exceeded it. "
            "No cap, fallback, static-profile reuse, or experiment overwrite was applied."
        )
    finally:
        if dataset is not None and hasattr(dataset, "close"):
            dataset.close()
    report["commitment_sha256"] = _digest(report)
    save_json(destination, report)
    return report


def _validated_report(report, path):
    from .engine import _artifact_path_label

    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise ValueError("Streaming training requires its own stateful preflight schema; legacy/static reports are invalid")
    commitment = dict(report)
    claimed = commitment.pop("commitment_sha256", None)
    if claimed != _digest(commitment):
        raise ValueError("Streaming preflight report commitment mismatch")
    if (report.get("passed") is not True or report.get("report_eligible") is not True
            or report.get("status") != "passed" or report.get("output") != _artifact_path_label(path)
            or report.get("measurement_scope") != _scope()):
        raise ValueError("Streaming preflight is incomplete, non-reporting, or belongs to a different output")
    checks = report.get("checks", {})
    if not all(checks.get(name) is True for name in (
        "complete_topology_scan", "conservative_prefix_edge_guard", "stateful_forward_backward", "cuda_available",
    )):
        raise ValueError("Streaming preflight required checks did not all pass")
    topology = report.get("topology", {})
    if (topology.get("scan_complete") is not True or topology.get("arrival_prefix_peak_measured") is not False
            or topology.get("prefix_bound_kind") != "previous_live_window_union_all_current_frame_arrivals"):
        raise ValueError("Streaming topology scope or conservative prefix bound is invalid")
    probe = report.get("batch_training_probe", {})
    if probe.get("passed") is not True or not probe.get("steps"):
        raise ValueError("Streaming physical-batch training probes are missing")
    return report


def _validate_guard_measurement(report, public_config, batches):
    """Cross-check guard selection and measured reserve evidence without CUDA calls."""
    def number(value, label, *, positive=False):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or (positive and value == 0)):
            raise ValueError(f"Invalid measured guard {label}")
        return float(value)

    def close(left, right):
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-6)

    def at_least(left, right):
        return left >= right or close(left, right)

    request = report.get("request")
    measurement = report.get("guard_measurement")
    original = report.get("input_config_provenance")
    effective = report.get("config_provenance")
    if any(not isinstance(item, dict) for item in (request, measurement, original, effective)):
        raise ValueError("Streaming guard measurement and input/effective config provenance are required")
    explicit = request.get("use_measured_edge_guard")
    if type(explicit) is not bool or type(request.get("require_cuda")) is not bool:
        raise ValueError("Streaming measured guard request flags must be explicit booleans")
    reserve = number(request.get("reserve_vram_mib"), "requested reserve", positive=explicit)
    if not explicit and reserve != 0:
        raise ValueError("A VRAM reserve requires explicit measured-guard configuration")
    original_config = original.get("config")
    if (not isinstance(original_config, dict)
            or original != {"config": original_config, "sha256": _digest(original_config)}
            or effective != {"config": public_config, "sha256": _digest(public_config)}):
        raise ValueError("Streaming input/effective config provenance is inconsistent")
    old_guard = original_config.get("model", {}).get("max_graph_edges", "missing")
    new_guard = public_config.get("model", {}).get("max_graph_edges", "missing")
    for guard in (old_guard, new_guard):
        if guard is not None and (type(guard) is not int or guard < 1):
            raise ValueError("Streaming edge guard must be an explicit positive integer or null")
    topology = report.get("topology")
    if not isinstance(topology, dict) or topology.get("scan_complete") is not True:
        raise ValueError("Measured guard requires a complete topology scan")
    records = topology.get("samples")
    if not isinstance(records, list) or not records:
        raise ValueError("Measured guard topology records are missing")
    count_fields = ("readout_nodes", "readout_directed_edges", "prefix_union_nodes_upper_bound",
                    "prefix_union_directed_edges_upper_bound")
    for row in records:
        if not isinstance(row, dict) or any(type(row.get(name)) is not int or row[name] < 0
                                            for name in count_fields):
            raise ValueError("Measured guard topology counts must be nonnegative integers")
        if (row["readout_nodes"] > row["prefix_union_nodes_upper_bound"]
                or row["readout_directed_edges"] > row["prefix_union_directed_edges_upper_bound"]):
            raise ValueError("Measured readout graph exceeds its conservative prefix bound")
    for name in count_fields:
        value = topology.get(f"max_{name}")
        if type(value) is not int or value != max(row[name] for row in records):
            raise ValueError("Cached streaming topology maxima differ from their complete records")
    required = topology["max_prefix_union_directed_edges_upper_bound"]
    proposed = max(old_guard or 0, required, 1)
    expected_config = copy.deepcopy(original_config)
    if explicit:
        expected_config["model"]["max_graph_edges"] = proposed
    if expected_config != public_config:
        raise ValueError("Measured configuration must change only the explicitly authorized edge guard")
    expected_fields = {
        "configured_max_graph_edges": old_guard, "measured_union_required_guard": required,
        "proposed_max_graph_edges": proposed, "effective_max_graph_edges": new_guard,
        "explicit_measured_guard_requested": explicit,
        "selection": "max(original_guard_or_zero, full_train_prefix_union_edges, 1)",
        "topology_sha256": _digest(topology), "evaluation_memory_certified": False,
    }
    for name, expected in expected_fields.items():
        actual = measurement.get(name, "missing")
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Streaming guard measurement differs in {name}")
    if new_guard is not None and required > new_guard:
        raise ValueError("Complete streaming prefix bound exceeds the effective guard")
    derived = measurement.get("derived_config")
    if (explicit and (not isinstance(derived, str) or not derived)) or (not explicit and derived is not None):
        raise ValueError("Derived guard config provenance does not match explicit authorization")
    floor = _graph_storage_floor(topology, batches)
    if measurement.get("raw_graph_storage_floor") != floor:
        raise ValueError("Streaming raw-graph storage floor differs from the full sequence schedule")
    memory = measurement.get("device_memory")
    probe = report.get("batch_training_probe")
    if not isinstance(memory, dict) or type(memory.get("measured")) is not bool or not isinstance(probe, dict):
        raise ValueError("Measured guard device-memory and probe evidence are required")
    if not close(number(memory.get("reserve_vram_mib"), "device reserve"), reserve):
        raise ValueError("Device-memory reserve differs from its request")
    if not close(number(probe.get("reserve_vram_mib"), "probe reserve"), reserve):
        raise ValueError("Training-probe reserve differs from its request")
    if (probe.get("memory_scope") != "model_setup_and_every_chronological_batch_input_through_state_commit_and_release"
            or probe.get("reserve_scope") != "live_device_free_pre_and_post_batch_and_allocator_peak_against_device_total"
            or probe.get("reserve_is_hard_isolation") is not False
            or probe.get("passed") is not True or probe.get("phase") != "completed"
            or probe.get("current_batch_index") is not None or probe.get("failed_batch_index") is not None):
        raise ValueError("Streaming probe memory/reserve scope or completion evidence is invalid")
    plan = probe.get("plan")
    if (not isinstance(plan, dict) or type(probe.get("completed_batches")) is not int
            or probe["completed_batches"] != plan.get("replay_stop_batch", -2) + 1):
        raise ValueError("Streaming reserve observations do not cover the planned causal replay")
    steps = probe.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Streaming measured training steps are missing")
    if not memory["measured"]:
        if (request["require_cuda"] or report.get("report_eligible") is True
                or report.get("checks", {}).get("cuda_available") is True
                or memory.get("scope") != "cpu_smoke_only"):
            raise ValueError("CPU smoke memory evidence cannot certify a CUDA reserve")
        fields = ("peak_allocated_mib", "peak_reserved_mib", "minimum_observed_device_free_mib")
        if any(probe.get(name) is not None for name in fields) or any(
            step.get(name) is not None for step in steps
            for name in ("peak_allocated_mib", "peak_reserved_mib", "device_free_mib", "device_total_mib")
        ):
            raise ValueError("CPU smoke cannot contain invented CUDA memory measurements")
        return
    if memory.get("scope") != "current_device_snapshot_not_a_peak_guarantee":
        raise ValueError("Streaming device-memory snapshot scope is invalid")
    total = number(memory.get("total_mib"), "device total", positive=True)
    free = number(memory.get("free_mib"), "device free")
    allocated = number(memory.get("allocated_mib"), "current allocated")
    reserved = number(memory.get("reserved_mib"), "current reserved")
    reusable = number(memory.get("allocator_reusable_mib"), "allocator reusable")
    available = memory.get("available_after_reserve_mib")
    if (isinstance(available, bool) or not isinstance(available, (int, float))
            or not math.isfinite(available) or not close(available, free + reusable - reserve)
            or not close(reusable, max(0, reserved - allocated))
            or not at_least(total, free) or not at_least(total, reserved)
            or not at_least(reserved, allocated) or not at_least(available, floor["mib"])):
        raise ValueError("Streaming device-memory arithmetic or raw-graph floor is inconsistent")
    peak_allocated = number(probe.get("peak_allocated_mib"), "probe peak allocated")
    peak_reserved = number(probe.get("peak_reserved_mib"), "probe peak reserved")
    minimum_free = number(probe.get("minimum_observed_device_free_mib"), "minimum observed free")
    if (not at_least(peak_reserved, peak_allocated) or not at_least(total - reserve, peak_reserved)
            or not at_least(minimum_free, reserve) or not at_least(total, minimum_free)):
        raise ValueError("Streaming probe peak/free memory violates the explicit reserve")
    snapshots = [*steps, probe.get("last_memory_snapshot")]
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise TypeError("Streaming probe memory snapshot is missing")
        step_total = number(snapshot.get("device_total_mib"), "snapshot total", positive=True)
        step_free = number(snapshot.get("device_free_mib"), "snapshot free")
        step_allocated = number(snapshot.get("peak_allocated_mib"), "snapshot allocated")
        step_reserved = number(snapshot.get("peak_reserved_mib"), "snapshot reserved")
        if (not close(step_total, total) or not at_least(total, step_free)
                or not at_least(step_free, reserve) or not at_least(step_free, minimum_free)
                or not at_least(step_reserved, step_allocated)
                or not at_least(total - reserve, step_reserved)
                or not at_least(peak_allocated, step_allocated)
                or not at_least(peak_reserved, step_reserved)):
            raise ValueError("Streaming probe snapshot contradicts the reserve or peak summaries")


def verify_streaming_training_preflight(config, report_path):
    import torch

    from .data import build_dataset
    from .engine import (
        _artifact_path_label,
        _current_source_contract,
        _enforce_training_split_status,
        _file_sha256,
        _public_config,
        _training_protocol,
    )
    from .preflight import _data_provenance, _make_batch_sampler, _runtime_provenance
    from .utils import resolve_device, validate_experiment_config

    validate_experiment_config(config)
    validate_streaming_contract(config, training=True)
    path = Path(report_path)
    with path.open(encoding="utf-8") as handle:
        report = _validated_report(json.load(handle), path)
    device = resolve_device(config.get("device", "auto"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Streaming training preflight verification requires allocated CUDA")
    if config["train"]["batch_size"] != 16:
        raise ValueError("The approved streaming CUDA gate must measure physical batch_size=16")
    public = _public_config(config)
    if report["config_provenance"] != {"config": public, "sha256": _digest(public)}:
        raise ValueError("Streaming preflight config differs from the current experiment")
    if report["source_provenance"] != _current_source_contract():
        raise ValueError("Streaming preflight executable source differs")
    if report["runtime_provenance"] != _runtime_provenance(device):
        raise ValueError("Streaming preflight runtime/GPU differs")
    if report["training_protocol"] != _training_protocol(config, device):
        raise ValueError("Streaming training protocol differs from the measured preflight")
    _enforce_training_split_status(config)
    dataset = build_dataset(config["dataset"], split="train")
    try:
        if report["data_provenance"] != _data_provenance(dataset, config):
            raise ValueError("Streaming training data/transform/source identities differ")
        topology = report["topology"]
        records = topology["samples"]
        if topology["dataset_samples"] != len(dataset) or topology["scanned_samples"] != len(dataset):
            raise ValueError("Streaming topology scan does not cover the current full dataset")
        if len(records) != len(dataset) or [row["dataset_index"] for row in records] != list(range(len(dataset))):
            raise ValueError("Streaming topology record identities are incomplete or duplicated")
        for row in records:
            for field in ("incoming_events", "arrival_groups", "readout_nodes", "readout_directed_edges",
                          "prefix_union_nodes_upper_bound", "prefix_union_directed_edges_upper_bound"):
                if type(row.get(field)) is not int or row[field] < 0:
                    raise ValueError("Streaming topology contains invalid counts")
            if (row["readout_nodes"] > row["prefix_union_nodes_upper_bound"]
                    or row["readout_directed_edges"] > row["prefix_union_directed_edges_upper_bound"]):
                raise ValueError("Streaming readout counts exceed their recorded prefix bound")
            guard = config["model"]["max_graph_edges"]
            if guard is not None and row["prefix_union_directed_edges_upper_bound"] > guard:
                raise ValueError("Streaming prefix bound exceeds the configured edge guard")
        batches = list(_make_batch_sampler(dataset, config))
        _validate_guard_measurement(report, public, batches)
        plan = _stream_probe_plan(batches, records, 16, report["request"]["profile_samples"])
        probe = report["batch_training_probe"]
        if probe["plan"] != plan or [row["batch_index"] for row in probe["steps"]] != plan["selected_batch_indices"]:
            raise ValueError("Streaming measured batches differ from the current sequence schedule")
        if max(row["batch_size"] for row in probe["steps"]) != 16:
            raise ValueError("Streaming preflight never measured the full physical batch 16")
        for step, selection in zip(probe["steps"], plan["selected_batches"], strict=True):
            if step["batch_size"] != selection["batch_size"] or step["dataset_indices"] != selection["dataset_indices"]:
                raise ValueError("Streaming probe batch identity differs from its selected plan")
            for field in ("step_time_ms", "frames_per_second", "peak_allocated_mib", "peak_reserved_mib"):
                _positive_number(step[field], f"streaming probe {field}")
            if (isinstance(step["gradient_norm"], bool) or not math.isfinite(step["gradient_norm"])
                    or step["gradient_norm"] < 0):
                raise ValueError("Streaming probe gradient norm is invalid")
            loss = step.get("loss")
            if not isinstance(loss, dict) or not loss or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in loss.values()
            ):
                raise ValueError("Streaming probe reconstruction loss is missing or nonfinite")
    finally:
        if hasattr(dataset, "close"):
            dataset.close()
    return {
        "schema": VERIFICATION_SCHEMA, "status": "verified", "report_eligible": True,
        "report": _artifact_path_label(path), "report_sha256": _file_sha256(path),
        "measurement_scope": report["measurement_scope"],
        "config_sha256": report["config_provenance"]["sha256"],
        "data_sha256": report["data_provenance"]["content"]["sha256"],
        "source_tree_sha256": report["source_provenance"]["source_tree_sha256"],
        "gpu": report["runtime_provenance"]["gpu"], "batch_size": 16,
        "measured_steps": len(probe["steps"]),
        "batch_preflight": {
            "contract": plan["batching_contract"], "schedule_sha256": plan["schedule_sha256"],
            "measured_batches": len(probe["steps"]), "largest_measured_batch_size": 16,
        },
    }
