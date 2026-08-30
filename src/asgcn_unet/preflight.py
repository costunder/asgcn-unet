from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from tqdm import tqdm

from .data import build_dataset
from .engine import (
    _artifact_path_label,
    _build_optimizer,
    _centralize_gradients,
    _clip_and_validate_gradients,
    _current_source_contract,
    _dataset_content_fingerprint,
    _dataset_source_fingerprint,
    _dataset_transform_contract,
    _enforce_training_split_status,
    _ensure_finite_loss,
    _file_sha256,
    _make_grad_scaler,
    _optimizer_mode,
    _public_config,
    _split_manifest_contract,
    _training_protocol,
)
from .graph import prepare_event_nodes, radius_graph_topology, uniformly_sample_events
from .losses import ReconstructionLoss
from .model import ASGCNUNet
from .utils import move_sample, resolve_device, save_json, set_seed, validate_experiment_config


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _runtime_provenance(device: torch.device) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    selected_cuda = cuda_available and device.type == "cuda"
    if selected_cuda:
        # cuDNN version() itself enumerates device capabilities in PyTorch 2.13.
        # Initialize first so MIG uses the runtime count, not a pre-init NVML count.
        torch.cuda.init()
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "torch": str(torch.__version__),
        "requested_device": str(device),
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if selected_cuda else None,
        "gpu": None,
    }
    if selected_cuda:
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        runtime["gpu"] = {
            "index": index,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_mib": properties.total_memory / (1024**2),
            "multiprocessors": properties.multi_processor_count,
        }
    return runtime


def _base_report(
    config: dict[str, Any],
    output_path: Path,
    device: torch.device,
    *,
    require_cuda: bool,
    profile_samples: int,
    top_density_count: int,
) -> dict[str, Any]:
    public_config = _public_config(config)
    return {
        "schema": "asgcn_training_preflight_v1",
        "status": "running",
        "passed": False,
        "report_eligible": False,
        "output": _artifact_path_label(output_path),
        "request": {
            "require_cuda": require_cuda,
            "profile_samples": profile_samples,
            "top_density_count": top_density_count,
        },
        "measurement_scope": {
            "name": "selected_top_density_training_steps",
            "topology_scope": "complete_eventhdr_training_split",
            "absolute_vram_guarantee": False,
            "statement": (
                "Empirical gate for the selected highest-edge-count samples on the "
                "recorded GPU/runtime; it is not a proof of every future training step."
            ),
        },
        "checks": {
            "cuda_available": bool(torch.cuda.is_available() and device.type == "cuda"),
            "complete_topology_scan": False,
            "edge_guard": None,
            "forward_backward": False,
            "cuda_oom_free": None,
        },
        "config_provenance": {
            "sha256": _canonical_sha256(public_config),
            "resolved_paths_redacted": True,
            "config": public_config,
        },
        "data_provenance": None,
        "source_provenance": _current_source_contract(),
        "runtime_provenance": _runtime_provenance(device),
        "topology": None,
        "training_probe": {
            "selected_samples": [],
            "completed_samples": 0,
            "steps": [],
            "failure_category": None,
        },
    }


def _sample_topology(
    sample: dict[str, Any],
    model_config: dict[str, Any],
    dataset_index: int,
) -> dict[str, Any]:
    retained_events = int(sample["events"].shape[0])
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    raw_events = int(metadata.get("raw_event_count", retained_events))
    cropped_events = int(metadata.get("cropped_event_count", retained_events))
    if raw_events < cropped_events or cropped_events < retained_events:
        raise ValueError(
            f"Invalid event-count provenance for sample {sample.get('sample_id', dataset_index)}"
        )
    sampled = uniformly_sample_events(
        sample["events"], int(model_config.get("event_sampling_factor", 1))
    )
    _, positions = prepare_event_nodes(sampled, sample["sensor_size"])
    topology = radius_graph_topology(
        positions,
        float(model_config.get("graph_radius", 0.08)),
        position_dims=int(model_config.get("graph_position_dims", 3)),
        chunk_size=int(model_config.get("graph_chunk_size", 512)),
    )
    nodes = int(topology["nodes"])
    possible_edges = nodes * max(nodes - 1, 0)
    actual_edges = int(topology["actual_directed_edges"])
    max_edges = model_config.get("max_graph_edges", 2_000_000)
    edge_guard_passed = max_edges is None or actual_edges <= int(max_edges)
    return {
        "dataset_index": dataset_index,
        "sample_id": str(sample.get("sample_id", dataset_index)),
        "scene": str(metadata.get("scene", "unknown")),
        "sequence_index": (
            int(metadata["sequence_index"])
            if metadata.get("sequence_index") is not None
            else None
        ),
        "raw_events": raw_events,
        "cropped_events": cropped_events,
        "retained_events": retained_events,
        "model_sampled_events": nodes,
        "candidate_directed_edges": int(topology["candidate_directed_edges"]),
        "actual_directed_edges": actual_edges,
        "directed_edge_density": actual_edges / possible_edges if possible_edges else 0.0,
        "max_degree": int(topology["max_degree"]),
        "isolated_nodes": int(topology["isolated_nodes"]),
        "isolate_ratio": float(topology["isolate_ratio"]),
        "edge_guard_passed": edge_guard_passed,
    }


def _topology_summary(
    records: list[dict[str, Any]],
    *,
    dataset_size: int,
    data_max_events: int | None,
    model_config: dict[str, Any],
    top_density_count: int,
) -> dict[str, Any]:
    ordered = sorted(
        records,
        key=lambda item: (
            -int(item["actual_directed_edges"]),
            -int(item["candidate_directed_edges"]),
            -int(item["model_sampled_events"]),
            int(item["dataset_index"]),
        ),
    )
    total_nodes = sum(int(item["model_sampled_events"]) for item in records)
    total_isolates = sum(int(item["isolated_nodes"]) for item in records)
    max_edges = model_config.get("max_graph_edges", 2_000_000)
    return {
        "scan_scope": "complete_eventhdr_training_split",
        "scan_complete": len(records) == dataset_size,
        "samples_scanned": len(records),
        "dataset_samples": dataset_size,
        "dataset_max_events": data_max_events,
        "model_event_sampling_factor": int(model_config.get("event_sampling_factor", 1)),
        "edge_guard_limit": int(max_edges) if max_edges is not None else None,
        "edge_guard_exceeded_samples": sum(
            not bool(item["edge_guard_passed"]) for item in records
        ),
        "totals": {
            "raw_events": sum(int(item["raw_events"]) for item in records),
            "cropped_events": sum(int(item["cropped_events"]) for item in records),
            "retained_events": sum(int(item["retained_events"]) for item in records),
            "model_sampled_events": total_nodes,
            "candidate_directed_edges": sum(
                int(item["candidate_directed_edges"]) for item in records
            ),
            "actual_directed_edges": sum(
                int(item["actual_directed_edges"]) for item in records
            ),
            "isolated_nodes": total_isolates,
        },
        "max_degree": max((int(item["max_degree"]) for item in records), default=0),
        "isolate_ratio": total_isolates / total_nodes if total_nodes else 0.0,
        "density_rank_basis": [
            "actual_directed_edges_desc",
            "candidate_directed_edges_desc",
            "model_sampled_events_desc",
            "dataset_index_asc",
        ],
        "top_density_samples": ordered[:top_density_count],
        "samples": records,
    }


def _gpu_step(
    model: ASGCNUNet,
    criterion: ReconstructionLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    raw_sample: dict[str, Any],
    expected_topology: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    step: int,
    *,
    recurrent_state: torch.Tensor | None,
    previous_prediction: torch.Tensor | None,
    previous_target: torch.Tensor | None,
    context_sample_id: str | None,
) -> dict[str, Any]:
    train_config = config["train"]
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    sample = move_sample(raw_sample, device)
    sample_id = sample.get("sample_id", expected_topology["dataset_index"])
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.perf_counter()

    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        prediction, diagnostics = model.forward_sample(
            sample,
            recurrent_state=recurrent_state,
        )
        target = sample["target"].unsqueeze(0)
        loss, loss_parts = criterion(prediction, target)
        configured_loss_weights = train_config.get("loss_weights") or {}
        temporal_weight = float(configured_loss_weights.get("temporal", 0.0))
        temporal_applied = (
            temporal_weight > 0
            and previous_prediction is not None
            and previous_target is not None
        )
        if temporal_applied:
            temporal = F.l1_loss(
                prediction - previous_prediction,
                target - previous_target,
            )
            loss = loss + temporal_weight * temporal
            loss_parts["temporal"] = temporal.detach()
    loss_values = _ensure_finite_loss(
        loss,
        loss_parts,
        epoch=0,
        step=step,
        sample_id=sample_id,
    )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if _optimizer_mode(train_config) == "adam_gc":
        _centralize_gradients(model)
    gradient_norm = _clip_and_validate_gradients(
        model,
        float(train_config.get("grad_clip", 1.0)),
        epoch=0,
        step=step,
        sample_id=sample_id,
    )
    scaler.step(optimizer)
    scaler.update()

    if int(diagnostics["edges"]) != int(expected_topology["actual_directed_edges"]):
        raise RuntimeError("Topology probe disagrees with the training forward graph")
    if int(diagnostics["nodes"]) != int(expected_topology["model_sampled_events"]):
        raise RuntimeError("Topology probe disagrees with the training forward node count")

    if device.type == "cuda":
        end_event.record()
        end_event.synchronize()
        step_time_ms = float(start_event.elapsed_time(end_event))
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
    else:
        step_time_ms = (time.perf_counter() - start_time) * 1000.0
        peak_allocated = None
        peak_reserved = None
    if not math.isfinite(step_time_ms) or step_time_ms <= 0:
        raise FloatingPointError(f"Invalid preflight step time for sample {sample_id}")
    return {
        "dataset_index": int(expected_topology["dataset_index"]),
        "sample_id": str(sample_id),
        "nodes": int(diagnostics["nodes"]),
        "actual_directed_edges": int(diagnostics["edges"]),
        "loss": loss_values,
        "gradient_norm": gradient_norm,
        "step_time_ms": step_time_ms,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "amp_enabled": amp_enabled,
        "temporal_loss_applied": temporal_applied,
        "temporal_context_sample_id": context_sample_id,
    }


@torch.no_grad()
def _immediate_training_context(
    model: ASGCNUNet,
    dataset: Any,
    records: list[dict[str, Any]],
    selected: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, str | None]:
    """Replay one real predecessor so recurrent and temporal loss memory are represented."""
    temporal_weight = float(config["train"].get("loss_weights", {}).get("temporal", 0.0))
    recurrent = bool(config["model"].get("recurrent", True))
    if temporal_weight <= 0 and not recurrent:
        return None, None, None, None
    index = int(selected["dataset_index"])
    if index < 1:
        return None, None, None, None
    previous = records[index - 1]
    previous_sequence = previous.get("sequence_index")
    current_sequence = selected.get("sequence_index")
    if (
        previous["scene"] != selected["scene"]
        or previous_sequence is None
        or current_sequence is None
        or int(current_sequence) != int(previous_sequence) + 1
    ):
        return None, None, None, None

    sample = move_sample(dataset[index - 1], device)
    amp_enabled = bool(config["train"].get("amp", True)) and device.type == "cuda"
    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        prediction, diagnostics = model.forward_sample(sample, recurrent_state=None)
    state = diagnostics["recurrent_state"]
    return (
        state.detach() if state is not None else None,
        prediction.detach(),
        sample["target"].unsqueeze(0).detach(),
        str(sample.get("sample_id", index - 1)),
    )


def training_preflight(
    config: dict[str, Any],
    output_path: str | Path,
    *,
    profile_samples: int = 3,
    top_density_count: int = 10,
    require_cuda: bool = True,
) -> dict[str, Any]:
    """Gate full training with a complete topology scan and dense-sample train steps.

    The CLI always leaves ``require_cuda`` enabled. The CPU path exists only so the
    exact scan and optimizer contract can be regression-tested; its artifact is
    explicitly marked ineligible for accelerator reporting.
    """
    validate_experiment_config(config)
    if config.get("dataset", {}).get("type") != "eventhdr":
        raise ValueError("Training preflight requires dataset.type='eventhdr'")
    for value, name in (
        (profile_samples, "profile_samples"),
        (top_density_count, "top_density_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    if top_density_count < profile_samples:
        raise ValueError("top_density_count must be greater than or equal to profile_samples")

    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(
            f"Preflight output already exists: {destination}. Move it or choose a new output."
        )
    set_seed(int(config.get("seed", 2026)))
    device = resolve_device(config.get("device", "auto"))
    cuda_ready = bool(torch.cuda.is_available() and device.type == "cuda")
    report = _base_report(
        config,
        destination,
        device,
        require_cuda=require_cuda,
        profile_samples=profile_samples,
        top_density_count=top_density_count,
    )
    if require_cuda and not cuda_ready:
        report["status"] = "failed"
        report["training_probe"]["failure_category"] = "cuda_required_but_unavailable"
        save_json(destination, report)
        return report

    dataset = None
    try:
        _enforce_training_split_status(config)
        dataset = build_dataset(config["dataset"], split="train")
        content_fingerprint = _dataset_content_fingerprint(dataset)
        records: list[dict[str, Any]] = []
        for index in tqdm(range(len(dataset)), desc="preflight-topology"):
            records.append(_sample_topology(dataset[index], config["model"], index))
        topology = _topology_summary(
            records,
            dataset_size=len(dataset),
            data_max_events=config["dataset"].get("max_events"),
            model_config=config["model"],
            top_density_count=top_density_count,
        )
        report["topology"] = topology
        report["data_provenance"] = {
            "dataset_type": "eventhdr",
            "split": "train",
            "content": content_fingerprint,
            "source_files": _dataset_source_fingerprint(dataset),
            "transform": _dataset_transform_contract(config),
            "split_manifest": _split_manifest_contract(config),
        }
        scan_complete = bool(topology["scan_complete"])
        edge_guard_passed = int(topology["edge_guard_exceeded_samples"]) == 0
        report["checks"]["complete_topology_scan"] = scan_complete
        report["checks"]["edge_guard"] = edge_guard_passed

        if len(dataset) < profile_samples:
            raise ValueError("EventHDR training split has fewer samples than profile_samples")
        selected = topology["top_density_samples"][: min(profile_samples, len(dataset))]
        report["training_probe"]["selected_samples"] = [
            {
                "dataset_index": int(item["dataset_index"]),
                "sample_id": item["sample_id"],
                "actual_directed_edges": int(item["actual_directed_edges"]),
            }
            for item in selected
        ]
        if scan_complete and edge_guard_passed:
            model = ASGCNUNet(**config["model"]).to(device).train()
            criterion = ReconstructionLoss(config["train"].get("loss_weights"))
            optimizer = _build_optimizer(model, config["train"])
            amp_enabled = bool(config["train"].get("amp", True)) and device.type == "cuda"
            scaler = _make_grad_scaler(amp_enabled)
            report["training_probe"]["training_protocol"] = _training_protocol(config, device)
            for step, selected_topology in enumerate(selected, start=1):
                context = _immediate_training_context(
                    model,
                    dataset,
                    records,
                    selected_topology,
                    config,
                    device,
                )
                step_result = _gpu_step(
                    model,
                    criterion,
                    optimizer,
                    scaler,
                    dataset[int(selected_topology["dataset_index"])],
                    selected_topology,
                    config,
                    device,
                    step,
                    recurrent_state=context[0],
                    previous_prediction=context[1],
                    previous_target=context[2],
                    context_sample_id=context[3],
                )
                report["training_probe"]["steps"].append(step_result)
                report["training_probe"]["completed_samples"] = step
            report["checks"]["forward_backward"] = len(
                report["training_probe"]["steps"]
            ) == len(selected)
            report["checks"]["cuda_oom_free"] = True if cuda_ready else None
        else:
            report["training_probe"]["failure_category"] = "edge_guard_exceeded"
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, ArithmeticError) as error:
        message = str(error).lower()
        if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in message:
            category = "cuda_out_of_memory"
            report["checks"]["cuda_oom_free"] = False
        elif "max_graph_edges" in message:
            category = "edge_guard_exceeded_during_forward"
            report["checks"]["edge_guard"] = False
        else:
            category = "unexpected_" + type(error).__name__
        report["training_probe"]["failure_category"] = category
    finally:
        if dataset is not None and hasattr(dataset, "close"):
            dataset.close()

    checks = report["checks"]
    diagnostic_passed = bool(
        checks["complete_topology_scan"]
        and checks["edge_guard"]
        and checks["forward_backward"]
        and (checks["cuda_oom_free"] is not False)
    )
    report["report_eligible"] = bool(require_cuda and cuda_ready and diagnostic_passed)
    report["passed"] = bool(diagnostic_passed and (cuda_ready or not require_cuda))
    report["status"] = (
        "passed"
        if report["passed"] and report["report_eligible"]
        else "diagnostic_passed"
        if report["passed"]
        else "failed"
    )
    save_json(destination, report)
    return report


def _require_verified_report_contract(report: Any, report_path: Path) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise TypeError("Training preflight report must contain a JSON object")
    # Serializing with allow_nan=False rejects hand-written NaN/Infinity values.
    _canonical_sha256(report)
    if report.get("schema") != "asgcn_training_preflight_v1":
        raise ValueError("Training preflight report has an unsupported schema")
    if report.get("status") != "passed" or report.get("passed") is not True:
        raise ValueError("Training preflight report did not pass")
    if report.get("report_eligible") is not True:
        raise ValueError("Training preflight report is not eligible for a reporting run")
    request = report.get("request")
    if not isinstance(request, dict) or request.get("require_cuda") is not True:
        raise ValueError("Training preflight report was not produced in CUDA report mode")
    scope = report.get("measurement_scope")
    if not isinstance(scope, dict) or scope.get("name") != "selected_top_density_training_steps":
        raise ValueError("Training preflight report has no bounded measurement scope")
    if scope.get("absolute_vram_guarantee") is not False:
        raise ValueError("Training preflight report overstates its VRAM measurement scope")
    checks = report.get("checks")
    expected_checks = {
        "cuda_available": True,
        "complete_topology_scan": True,
        "edge_guard": True,
        "forward_backward": True,
        "cuda_oom_free": True,
    }
    if not isinstance(checks, dict) or any(
        checks.get(name) is not expected for name, expected in expected_checks.items()
    ):
        raise ValueError("Training preflight report does not satisfy every required gate")
    if report.get("output") != _artifact_path_label(report_path):
        raise ValueError("Training preflight report output identity does not match its file")

    probe = report.get("training_probe")
    if not isinstance(probe, dict) or probe.get("failure_category") is not None:
        raise ValueError("Training preflight report contains a failed training probe")
    selected = probe.get("selected_samples")
    steps = probe.get("steps")
    if not isinstance(selected, list) or not selected or not isinstance(steps, list):
        raise ValueError("Training preflight report has no measured dense-sample steps")
    if int(probe.get("completed_samples", -1)) != len(selected) or len(steps) != len(selected):
        raise ValueError("Training preflight report did not complete every selected sample")
    if int(request.get("profile_samples", -1)) != len(selected):
        raise ValueError("Training preflight report measured fewer samples than requested")
    for step, chosen in zip(steps, selected, strict=True):
        if not isinstance(step, dict):
            raise TypeError("Training preflight step must be an object")
        for field in ("dataset_index", "sample_id", "actual_directed_edges"):
            if step.get(field) != chosen.get(field):
                raise ValueError("Training preflight step identity differs from its selection")
        for field in ("step_time_ms", "peak_allocated_mib", "peak_reserved_mib"):
            value = step.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"Training preflight step has invalid {field}")
    topology = report.get("topology")
    if (
        not isinstance(topology, dict)
        or topology.get("scan_scope") != "complete_eventhdr_training_split"
        or topology.get("scan_complete") is not True
        or int(topology.get("samples_scanned", 0)) < 1
        or topology.get("samples_scanned") != topology.get("dataset_samples")
        or int(topology.get("edge_guard_exceeded_samples", -1)) != 0
    ):
        raise ValueError("Training preflight topology scan is incomplete or over the edge guard")
    ranked = topology.get("top_density_samples")
    if not isinstance(ranked, list) or len(ranked) < len(selected):
        raise ValueError("Training preflight report has an incomplete density ranking")
    for chosen, expected in zip(selected, ranked[: len(selected)], strict=True):
        if not isinstance(chosen, dict) or not isinstance(expected, dict):
            raise TypeError("Training preflight density entries must be objects")
        identity = ("dataset_index", "sample_id", "actual_directed_edges")
        if any(chosen.get(field) != expected.get(field) for field in identity):
            raise ValueError("Training preflight steps are not the top-density samples")
    return report


def verify_training_preflight(
    config: dict[str, Any],
    report_path: str | Path,
) -> dict[str, Any]:
    """Re-bind a passed report to the current config, data, source, and CUDA runtime."""
    validate_experiment_config(config)
    if config.get("dataset", {}).get("type") != "eventhdr":
        raise ValueError("Training preflight verification requires EventHDR")
    path = Path(report_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Training preflight report not found: {path}. Run the profile stage first."
        )
    with path.open("r", encoding="utf-8") as handle:
        report = _require_verified_report_contract(json.load(handle), path)

    device = resolve_device(config.get("device", "auto"))
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("A passed training preflight can only be verified on CUDA")
    public_config = _public_config(config)
    config_provenance = report.get("config_provenance")
    if (
        not isinstance(config_provenance, dict)
        or config_provenance.get("config") != public_config
        or config_provenance.get("sha256") != _canonical_sha256(public_config)
    ):
        raise ValueError("Training config differs from the preflight report")
    if report.get("source_provenance") != _current_source_contract():
        raise ValueError("Executable source differs from the preflight report")
    if report.get("runtime_provenance") != _runtime_provenance(device):
        raise ValueError("CUDA/software runtime differs from the preflight report")
    probe = report["training_probe"]
    if probe.get("training_protocol") != _training_protocol(config, device):
        raise ValueError("Training protocol differs from the preflight report")

    _enforce_training_split_status(config)
    dataset = build_dataset(config["dataset"], split="train")
    try:
        current_data = {
            "dataset_type": "eventhdr",
            "split": "train",
            "content": _dataset_content_fingerprint(dataset),
            "source_files": _dataset_source_fingerprint(dataset),
            "transform": _dataset_transform_contract(config),
            "split_manifest": _split_manifest_contract(config),
        }
        if report.get("data_provenance") != current_data:
            raise ValueError("EventHDR training data differs from the preflight report")
        if int(report["topology"]["dataset_samples"]) != len(dataset):
            raise ValueError("EventHDR sample count differs from the topology scan")
    finally:
        if hasattr(dataset, "close"):
            dataset.close()

    return {
        "schema": "asgcn_preflight_verification_v1",
        "status": "verified",
        "report_eligible": True,
        "report": _artifact_path_label(path),
        "report_sha256": _file_sha256(path),
        "measurement_scope": report["measurement_scope"],
        "config_sha256": config_provenance["sha256"],
        "data_sha256": report["data_provenance"]["content"]["sha256"],
        "source_tree_sha256": report["source_provenance"]["source_tree_sha256"],
        "gpu": report["runtime_provenance"]["gpu"],
        "measured_steps": int(probe["completed_samples"]),
    }


__all__ = ["training_preflight", "verify_training_preflight"]
