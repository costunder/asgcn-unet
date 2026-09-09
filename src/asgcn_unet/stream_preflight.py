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


def _scan_stream_topology(dataset, config, device, batches, *, top_density_count):
    import torch
    from tqdm import tqdm

    from .batching import sequence_key
    from .preflight import _load_packed_probe_batch
    from .stream_model import _metadata, _pack_previous, _prepared, _split_state, _update
    from .stream_state import StreamingReconstructionState

    model = _topology_model(config)
    states, records = {}, [None] * len(dataset)
    seen = set()
    final = {sequence_key(item): item["sequence_index"] for item in dataset.samples}
    with torch.no_grad():
        for indices in tqdm(batches, desc="stream-preflight-window-topology"):
            if any(index in seen for index in indices):
                raise ValueError("Streaming topology schedule repeats frames")
            samples, _ = _load_packed_probe_batch(dataset, indices, device)
            previous = [states.get(sequence_key(sample)) for sample in samples]
            metadata, contract = _metadata(model, samples, previous)
            graph, _ = _pack_previous(previous, device)
            features, positions, timestamps, node_batch = _prepared(model, samples, metadata)
            starts = timestamps.new_tensor([record[2] for record in metadata])
            ends = timestamps.new_tensor([record[3] for record in metadata])
            window = model.stream_config["window_seconds"]
            for old, record in zip(previous, metadata, strict=True):
                if old is not None and old.watermark_seconds > record[2]:
                    raise ValueError("Streaming topology scan requires nonoverlapping frame intervals")
            # Every actual arrival-prefix graph is a subgraph of this union. It
            # deliberately keeps events which will expire later inside the frame.
            # A guard refusal here is conservative, NOT an observed prefix maximum.
            union = _update(model, graph, features, positions, timestamps, node_batch, starts - window).state
            union_nodes = torch.bincount(union.node_batch, minlength=len(samples))
            union_edges = torch.bincount(
                union.node_batch[union.graph.edge_index[0]], minlength=len(samples),
            )
            readout = _update(model, union, features[:0], positions[:0], timestamps[:0],
                              node_batch[:0], ends - window).state
            readout_nodes = torch.bincount(readout.node_batch, minlength=len(samples))
            readout_edges = torch.bincount(
                readout.node_batch[readout.graph.edge_index[0]], minlength=len(samples),
            )
            counts = torch.stack((readout_nodes, readout_edges, union_nodes, union_edges)).cpu().tolist()
            lanes = _split_state(readout, None, len(samples))
            for lane, (index, sample, record, (lane_graph, _)) in enumerate(
                zip(indices, samples, metadata, lanes, strict=True)
            ):
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
                state = StreamingReconstructionState(
                    lane_graph, None, None, record[4], record[3], record[1], record[0], None, contract,
                )
                if record[1] == final[record[0]]:
                    states.pop(record[0], None)
                else:
                    states[record[0]] = state.detach().clone()
            seen.update(indices)
            del graph, union, readout, samples, previous, features, positions, timestamps, node_batch, lanes
    if seen != set(range(len(dataset))) or any(record is None for record in records):
        raise ValueError("Streaming topology scan did not cover every training frame exactly once")
    ranked = sorted(records, key=lambda row: (
        -row["prefix_union_directed_edges_upper_bound"], -row["readout_directed_edges"], row["dataset_index"],
    ))
    return {
        "scope": "complete_eventhdr_training_stream", "dataset_samples": len(dataset),
        "scanned_samples": len(records), "scan_complete": True,
        "arrival_prefix_peak_measured": False,
        "prefix_bound_kind": "previous_live_window_union_all_current_frame_arrivals",
        "statement": "Readout counts are actual; prefix counts are conservative upper bounds, not measured maxima.",
        "max_readout_nodes": max(row["readout_nodes"] for row in records),
        "max_readout_directed_edges": max(row["readout_directed_edges"] for row in records),
        "max_prefix_union_nodes_upper_bound": max(row["prefix_union_nodes_upper_bound"] for row in records),
        "max_prefix_union_directed_edges_upper_bound": max(
            row["prefix_union_directed_edges_upper_bound"] for row in records
        ),
        "top_density_samples": ranked[:top_density_count], "samples": records,
    }


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


def _probe_stream_training(dataset, config, device, batches, topology, plan):
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

    set_seed(int(config["seed"]))
    model = build_model(config["model"]).to(device).train()
    criterion = ReconstructionLoss(config["train"].get("loss_weights"))
    optimizer = _build_optimizer(model, config["train"])
    amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    scaler = _make_grad_scaler(amp)
    temporal_weight = float((config["train"].get("loss_weights") or {}).get("temporal", 0.0))
    state = TrainingState(independent_sequences=True)
    final = {sequence_key(item): item["sequence_index"] for item in dataset.samples}
    selected, measured = set(plan["selected_batch_indices"]), []
    replayed_frames = 0
    started = time.perf_counter()
    for number, indices in enumerate(tqdm(batches[:plan["replay_stop_batch"] + 1], desc="stream-preflight-stateful-train")):
        samples, input_pipeline = _load_packed_probe_batch(dataset, indices, device)
        contexts = state.prepare(samples)
        if number not in selected:
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
                prediction, diagnostics = model.forward_training_batch(samples, [entry[0] for entry in contexts])
            target = samples.targets
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError("Non-finite reconstruction during causal predecessor replay")
            replayed_frames += len(samples)
        else:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            step_started = time.perf_counter()
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
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - step_started) * 1000
            if prediction.shape != target.shape or prediction.shape[0] != len(indices):
                raise RuntimeError("Streaming probe reconstruction does not match the actual batch target")
            for index, detail in zip(indices, diagnostics, strict=True):
                expected = topology["samples"][index]
                if detail["nodes"] != expected["readout_nodes"] or detail["edges"] != expected["readout_directed_edges"]:
                    raise RuntimeError("Streaming model readout topology differs from the full causal scan")
                if not detail.get("stream_execution", {}).get("training_dense_snapshot"):
                    raise RuntimeError("Streaming preflight did not execute the declared causal-window training path")
            measured.append({
                "batch_index": number, "dataset_indices": list(indices), "batch_size": len(indices),
                "incoming_contexts": sum(context[0] is not None for context in contexts),
                "loss": loss, "gradient_norm": gradient_norm, "amp": amp_info,
                "step_time_ms": elapsed, "frames_per_second": len(indices) * 1000 / elapsed,
                "input_pipeline": input_pipeline, "prediction_shape": list(prediction.shape),
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None,
                "scope": "stateful_forward_loss_backward_optimizer_includes_live_context_residency",
            })
        state.commit(samples, prediction, diagnostics, target)
        state.release_finished(samples, final)
        if number in selected:
            del payload, forward_loss
        del prediction, diagnostics, target, samples, contexts
    if [row["batch_index"] for row in measured] != plan["selected_batch_indices"]:
        raise RuntimeError("Not every selected streaming training batch completed")
    return {
        "passed": True, "plan": plan, "steps": measured,
        "replayed_predecessor_frames": replayed_frames,
        "elapsed_including_context_replay_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "training_protocol_scope": "ANN causal-window learning; event-driven inference is a separate evaluation",
    }


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


def streaming_training_preflight(
    config, output_path, *, profile_samples=3, top_density_count=10, require_cuda=True,
    resume_scan=False, reuse_report=None,
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
        "request": {"require_cuda": require_cuda, "profile_samples": profile_samples, "top_density_count": top_density_count},
        "config_provenance": {"config": public_config, "sha256": _digest(public_config)},
        "source_provenance": _current_source_contract(), "runtime_provenance": _runtime_provenance(device),
        "data_provenance": None, "topology": None, "batch_training_probe": None,
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
        _enforce_training_split_status(config)
        dataset = build_dataset(config["dataset"], split="train")
        report["data_provenance"] = _data_provenance(dataset, config)
        batches = list(_make_batch_sampler(dataset, config))
        report["topology"] = _scan_stream_topology(
            dataset, config, device, batches, top_density_count=top_density_count,
        )
        report["checks"]["complete_topology_scan"] = True
        report["checks"]["conservative_prefix_edge_guard"] = True
        plan = _stream_probe_plan(batches, report["topology"]["samples"], config["train"]["batch_size"], profile_samples)
        report["batch_training_probe"] = _probe_stream_training(dataset, config, device, batches, report["topology"], plan)
        report["checks"]["stateful_forward_backward"] = True
        if _current_source_contract() != report["source_provenance"]:
            raise ValueError("Executable source changed during the streaming preflight")
        # Rehash every source without the training hash cache: path/size alone
        # cannot detect an in-place, same-size edit during the scan or probes.
        if _data_provenance(dataset, config) != report["data_provenance"]:
            raise ValueError("Dataset content or provenance changed during the streaming preflight")
        report["passed"] = True
        report["report_eligible"] = bool(cuda_ready and require_cuda)
        report["status"] = "passed" if report["report_eligible"] else "cpu_smoke_passed_non_reporting"
    except KeyboardInterrupt as error:
        report["status"] = "interrupted"
        report["failure"] = _safe_failure(error, config, destination)
        report["commitment_sha256"] = _digest(report)
        save_json(destination, report)
        raise
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, FloatingPointError) as error:
        report["status"] = "failed"
        report["failure"] = _safe_failure(error, config, destination)
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
