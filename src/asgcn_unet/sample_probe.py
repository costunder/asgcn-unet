from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from .data import build_dataset
from .engine import (
    _artifact_path_label,
    _canonical_sha256,
    _cuda_peak_memory,
    _dataset_sample_identity,
    _file_sha256,
    _inference_precision,
    _inference_precision_context,
    _public_config,
    _require_finite_structure,
    _require_finite_tensor,
    _reset_cuda_peak_memory,
    _set_inference_max_graph_edges,
    _set_inference_snn_dynamics,
    _validate_snn_request,
    load_model_checkpoint,
)
from .utils import move_sample, resolve_device, set_seed, validate_experiment_config


def _strict_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _scalar_list(values: list[torch.Tensor]) -> list[float]:
    return [float(value.detach().float().cpu()) for value in values]


def _device_summary(device: torch.device) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        summary.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_mib": properties.total_memory / (1024**2),
            }
        )
    else:
        summary.update({"gpu_name": None, "gpu_total_memory_mib": None})
    return summary


@torch.no_grad()
def probe_evaluation_sample(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    sample_index: int,
    max_graph_edges: int,
    inference_mode: str = "ann",
    simulation_steps: int = 16,
    snn_dynamics: str | None = None,
) -> dict[str, Any]:
    """Run exactly one eval sample as a non-reporting memory/topology diagnostic."""
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError("sample_index must be a nonnegative integer")
    max_graph_edges = _strict_positive_integer(max_graph_edges, "max_graph_edges")
    _validate_snn_request(inference_mode, simulation_steps)
    validate_experiment_config(config)
    set_seed(int(config.get("seed", 2026)))
    device = resolve_device(config.get("device", "auto"))
    checkpoint_path = Path(checkpoint_path)
    dataset = build_dataset(config["dataset"], split="eval")
    try:
        if sample_index >= len(dataset):
            raise IndexError(
                f"sample_index={sample_index} is outside evaluation dataset size {len(dataset)}"
            )
        model, checkpoint = load_model_checkpoint(checkpoint_path, device, config["model"])
        _validate_snn_request(
            inference_mode,
            simulation_steps,
            checkpoint,
            checkpoint_path,
        )
        graph_edge_guard = _set_inference_max_graph_edges(model, max_graph_edges)
        _set_inference_snn_dynamics(model, inference_mode, snn_dynamics)
        model.eval()
        precision, autocast_dtype = _inference_precision(config.get("eval", {}), device, model)
        sample = move_sample(dataset[sample_index], device)
        sample_id = sample.get("sample_id", sample_index)
        _require_finite_tensor(sample["target"], "target", sample_id)

        _reset_cuda_peak_memory(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with _inference_precision_context(device, precision, autocast_dtype):
            prediction, diagnostics = model.forward_sample(
                sample,
                inference_mode=inference_mode,
                simulation_steps=simulation_steps,
                recurrent_state=None,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not math.isfinite(latency_ms) or latency_ms <= 0:
            raise FloatingPointError(f"Invalid latency: sample={sample_id}")
        _require_finite_tensor(prediction, "prediction", sample_id)
        _require_finite_structure(diagnostics, "diagnostics", sample_id)

        metadata = sample.get("metadata", {})
        result = {
            "schema": "asgcn_eval_sample_probe_v1",
            "report_eligible": False,
            "report_ineligible_reasons": [
                "single-sample diagnostic with recurrent state reset; not a quality evaluation"
            ],
            "dataset": config["dataset"]["type"],
            "dataset_size": len(dataset),
            "sample": {
                **_public_config(_dataset_sample_identity(dataset, sample_index)),
                "sample_id": str(sample_id),
                "raw_events": int(metadata.get("raw_event_count", sample["events"].shape[0])),
                "cropped_events": int(
                    metadata.get("cropped_event_count", sample["events"].shape[0])
                ),
                "retained_events": int(sample["events"].shape[0]),
                "sensor_size": [int(value) for value in sample["sensor_size"]],
            },
            "checkpoint": {
                "path": _artifact_path_label(checkpoint_path),
                "file_sha256": _file_sha256(checkpoint_path),
                "model_state_sha256": checkpoint.get("model_state_sha256"),
                "checkpoint_type": checkpoint.get("checkpoint_type"),
                "epoch": checkpoint.get("epoch"),
            },
            "model_config_sha256": _canonical_sha256(_public_config(config["model"])),
            "inference": {
                "mode": inference_mode,
                "simulation_steps": simulation_steps if inference_mode == "snn" else None,
                "snn_dynamics": model.snn_dynamics if inference_mode == "snn" else None,
                "recurrent_state": "reset",
            },
            "graph_edge_guard": graph_edge_guard,
            "graph_topology": {
                "exact": True,
                "nodes": int(diagnostics["nodes"]),
                "actual_directed_edges": int(diagnostics["edges"]),
                "isolated_nodes": int(diagnostics["isolated_nodes"]),
                "isolate_ratio": float(diagnostics["isolate_ratio"]),
                "max_in_degree": int(diagnostics["max_degree"]),
            },
            "snn": {
                "firing_rates": _scalar_list(diagnostics["firing_rates"]),
                "spike_counts": _scalar_list(diagnostics["spike_counts"]),
            }
            if inference_mode == "snn"
            else None,
            "prediction_shape": list(prediction.shape),
            "latency_ms": latency_ms,
            "gpu_memory": _cuda_peak_memory(device),
            "precision": precision,
            "runtime": _device_summary(device),
        }
        _require_finite_structure(result, "sample_probe", sample_id)
        return result
    finally:
        if hasattr(dataset, "close"):
            dataset.close()


def save_probe_result(path: str | Path, result: dict[str, Any]) -> None:
    """Write one JSON result without replacing an existing path."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Probe output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"Probe output already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)

