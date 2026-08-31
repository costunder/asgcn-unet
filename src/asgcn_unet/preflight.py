from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import re
import socket
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from tqdm import tqdm

from .batching import SequenceBatchSampler, sequence_key
from .data import build_dataset
from .engine import (
    _AMP_MAX_RETRIES,
    _artifact_path_label,
    _build_optimizer,
    _current_source_contract,
    _dataset_content_fingerprint,
    _dataset_source_fingerprint,
    _dataset_transform_contract,
    _enforce_training_split_status,
    _file_sha256,
    _make_grad_scaler,
    _optimizer_mode,
    _public_config,
    _split_manifest_contract,
    _training_protocol,
    _training_step,
)
from .graph import prepare_event_nodes, radius_graph_topology, uniformly_sample_events
from .losses import ReconstructionLoss
from .model import ASGCNUNet
from .scan import ScanInUseError, ScanJournal
from .training import batching_contract, forward_training_loss
from .utils import move_sample, resolve_device, save_json, set_seed, validate_experiment_config

# These exact clean source trees were audited for the same event selection and
# strict-radius topology semantics. They authorize reuse of topology counts only,
# never reuse of previous GPU measurements or the old AMP training implementation.
LEGACY_TOPOLOGY_SOURCES = frozenset(
    {
        (
            "1f806946a8d7e2157e134f873088f5112b3c84a9d31e25816475b71beb36b4d6",
            "0eae40f0c665f979dc0f077b366a9ff93b7d28cf",
        ),
        (
            "043e3803ae817dd10355f4370a4ee8acddfb311ec7342c2f9b630fc2a8974bec",
            "940b3b8a999a49a05535fb6c24ac5fc93a507934",
        ),
        (
            "043e3803ae817dd10355f4370a4ee8acddfb311ec7342c2f9b630fc2a8974bec",
            "11fe7f75d64f693e4aec39990de5bf4019818deb",
        ),
    }
)

# Version-2 framewise reports with the already-audited CUDA topology scan. The
# new batch gate changes imports/optimizer execution but not these graph counts.
LEGACY_V2_TOPOLOGY_SOURCES = frozenset(
    {
        # B4 release before lookup batching: identical candidate/strict-radius
        # semantics; only topology counts may be reused, never GPU probe results.
        (
            "e47f63d738e034cf53fe22aa8323598f28e4a243128fea42ee90cab9eed22650",
            "ef843d8ed2fa98808e2befc3aef653de845b79a6",
        ),
        (
            "e47f63d738e034cf53fe22aa8323598f28e4a243128fea42ee90cab9eed22650",
            "8337757516592dab288c3a7df0a9fa2a2e2372bd",
        ),
        (
            "57ee2e525d652d9cf60d42f56519944f58bb6b9b98eeba1e66e6798b02831306",
            "c8b1da000ec394e210ecf148f96c61086dde74ed",
        ),
        (
            "57ee2e525d652d9cf60d42f56519944f58bb6b9b98eeba1e66e6798b02831306",
            "1e3c4652c0e451c5b2c86a7561de98004d3c2d21",
        ),
    }
)


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
        "schema": "asgcn_training_preflight_v2",
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
                "Empirical gate for first/sparse/empty numerical cases and selected "
                "highest-edge-count samples on the recorded GPU/runtime; it is not "
                "a proof of every future training step."
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
        "topology_contract": None,
        "scan_provenance": None,
        "batch_training_probe": None,
        "training_probe": {
            "selected_samples": [],
            "completed_samples": 0,
            "steps": [],
            "failure_category": None,
            "failure": None,
            "numerical_probes": [],
            "numerical_selection": [],
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
            int(metadata["sequence_index"]) if metadata.get("sequence_index") is not None else None
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
        "edge_guard_exceeded_samples": sum(not bool(item["edge_guard_passed"]) for item in records),
        "totals": {
            "raw_events": sum(int(item["raw_events"]) for item in records),
            "cropped_events": sum(int(item["cropped_events"]) for item in records),
            "retained_events": sum(int(item["retained_events"]) for item in records),
            "model_sampled_events": total_nodes,
            "candidate_directed_edges": sum(
                int(item["candidate_directed_edges"]) for item in records
            ),
            "actual_directed_edges": sum(int(item["actual_directed_edges"]) for item in records),
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


def _topology_implementation_contract(device: torch.device) -> dict[str, Any]:
    """Bind cache reuse to executable topology dependencies, not optimizer code.

    The implementation digest is deliberately conservative for dataset readers.
    Decoder/optimizer changes do not invalidate topology counts, while changes to
    event selection, normalization or graph predicates require a fresh scan.
    """
    package = Path(__file__).resolve().parent
    graph_functions = {
        "uniformly_sample_events",
        "prepare_event_nodes",
        "_spatial_hash_constants",
        "_radius_graph_candidate_chunks",
        "radius_graph_topology",
    }
    preflight_functions = {"_sample_topology", "_topology_summary"}
    modules: dict[str, str] = {}
    for relative, functions in (
        ("graph.py", graph_functions),
        ("preflight.py", preflight_functions),
    ):
        tree = ast.parse((package / relative).read_text(encoding="utf-8"))
        selected = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in functions
        ]
        found = {
            node.name
            for node in selected
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if found != functions:
            raise ValueError("A topology cache dependency could not be fingerprinted")
        modules[relative] = _canonical_sha256(
            [ast.dump(node, include_attributes=False) for node in selected]
        )
    for relative in ("data/eventhdr.py", "data/common.py", "data/factory.py", "scan.py"):
        modules[relative] = _file_sha256(package / relative)
    return {
        "schema": "asgcn_topology_implementation_v1",
        "semantics": "ordered_normalized_float32_strict_radius_v1",
        "implementation": modules,
        "torch": str(torch.__version__),
        "device_type": device.type,
    }


def _data_provenance(dataset: Any, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_type": "eventhdr",
        "split": "train",
        "content": _dataset_content_fingerprint(dataset),
        "source_files": _dataset_source_fingerprint(dataset),
        "transform": _dataset_transform_contract(config),
        "split_manifest": _split_manifest_contract(config),
    }


def _validate_topology_records(
    records: Any,
    dataset_size: int,
    model_config: dict[str, Any],
    *,
    dataset: Any = None,
    complete: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or len(records) > dataset_size:
        raise ValueError("Invalid topology record count")
    if complete and len(records) != dataset_size:
        raise ValueError("Reusable topology scan must contain every training sample")
    factor = int(model_config.get("event_sampling_factor", 1))
    guard = model_config.get("max_graph_edges", 2_000_000)
    identities: set[str] = set()
    items = getattr(dataset, "samples", None)
    integer_fields = (
        "raw_events",
        "cropped_events",
        "retained_events",
        "model_sampled_events",
        "candidate_directed_edges",
        "actual_directed_edges",
        "max_degree",
        "isolated_nodes",
    )
    for index, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or isinstance(record.get("dataset_index"), bool)
            or record.get("dataset_index") != index
        ):
            raise ValueError("Topology records must be a contiguous ordered prefix")
        if any(
            isinstance(record.get(field), bool)
            or not isinstance(record.get(field), int)
            or record[field] < 0
            for field in integer_fields
        ):
            raise ValueError("Topology counts must be nonnegative integers")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in identities:
            raise ValueError("Topology sample identities must be nonempty and unique")
        identities.add(sample_id)
        if not isinstance(record.get("scene"), str):
            raise TypeError("Topology scene must be a string")
        sequence = record.get("sequence_index")
        if sequence is not None and (
            isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0
        ):
            raise ValueError("Topology sequence index is invalid")
        raw, cropped, retained, nodes = (record[field] for field in integer_fields[:4])
        edges, candidates = record["actual_directed_edges"], record["candidate_directed_edges"]
        maximum, isolates = record["max_degree"], record["isolated_nodes"]
        possible = nodes * max(0, nodes - 1)
        expected_guard = guard is None or edges <= int(guard)
        if (
            not raw >= cropped >= retained >= nodes
            or nodes != (retained + factor - 1) // factor
            or not 0 <= edges <= candidates <= possible
            or not 0 <= maximum <= max(nodes - 1, 0)
            or not 0 <= isolates <= nodes
            or edges > maximum * (nodes - isolates)
            or (edges == 0) != (maximum == 0)
            or (edges == 0) != (isolates == nodes)
            or record.get("edge_guard_passed") is not expected_guard
            or record.get("directed_edge_density") != (edges / possible if possible else 0.0)
            or record.get("isolate_ratio") != (isolates / nodes if nodes else 0.0)
        ):
            raise ValueError("Topology record has inconsistent counts or graph statistics")
        if isinstance(items, list):
            item = items[index]
            expected_id = (
                f"{item['scene']}/{item['image_key']}"
                if item["scene"] == item["source_file"]
                else f"{item['scene']}/{item['source_file']}/{item['image_key']}"
            )
            if (
                sample_id != expected_id
                or record["scene"] != item["scene"]
                or sequence != item["sequence_index"]
                or raw != item["end_idx"] - item["start_idx"]
            ):
                raise ValueError("Cached topology identity differs from the EventHDR index")
    return records


def _scan_sample(dataset: Any, index: int, device: torch.device) -> dict[str, Any]:
    reader = getattr(dataset, "get_topology_sample", None)
    sample = reader(index) if callable(reader) else dataset[index]
    sample = dict(sample)
    # Transfer graph inputs only. No GT image is needed for the topology scan.
    sample["events"] = sample["events"].to(device=device, non_blocking=True)
    if sample["events"].device != device and not (
        device.type == "cuda" and device.index is None and sample["events"].is_cuda
    ):
        raise RuntimeError("Topology input did not reach the selected execution device")
    return sample


def _numerical_selection(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selections: dict[int, list[str]] = {0: ["first_chronological"]}
    empty = next((r for r in records if r["model_sampled_events"] == 0), None)
    sparse = min(
        (r for r in records if r["model_sampled_events"] > 0),
        key=lambda r: (r["model_sampled_events"], r["actual_directed_edges"], r["dataset_index"]),
        default=None,
    )
    for record, reason in ((empty, "first_empty"), (sparse, "sparsest_nonempty")):
        if record is not None:
            selections.setdefault(record["dataset_index"], []).append(reason)
    return [
        {
            "dataset_index": index,
            "sample_id": records[index]["sample_id"],
            "actual_directed_edges": records[index]["actual_directed_edges"],
            "reasons": reasons,
        }
        for index, reasons in selections.items()
    ]


def _safe_failure(error: BaseException, config: dict[str, Any], output: Path) -> dict[str, str]:
    message = str(error)
    roots = [Path(__file__).resolve().parents[2], output.parent]
    roots.extend(
        Path(value).expanduser().resolve()
        for key, value in config.get("dataset", {}).items()
        if key in {"root", "val_root", "split_manifest", "file_manifest"} and isinstance(value, str)
    )
    for root in sorted(roots, key=lambda value: len(str(value)), reverse=True):
        for variant in {str(root), root.as_posix()}:
            message = message.replace(variant, "$PATH")
    message = re.sub(r"(?i)(?<![A-Za-z0-9_])/(?:home|Users)/[^/\s]+", "$HOME", message)
    message = re.sub(r"(?i)\b[A-Z]:[\\/]Users[\\/][^\\/\s]+", "$HOME", message)
    hostnames = {
        socket.gethostname(),
        os.environ.get("HOSTNAME", ""),
        os.environ.get("COMPUTERNAME", ""),
    }
    for hostname in sorted((value for value in hostnames if value), key=len, reverse=True):
        message = re.sub(re.escape(hostname), "$HOST", message, flags=re.IGNORECASE)
    return {"type": type(error).__name__, "message": message[:2000]}


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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.perf_counter()

    def forward_loss():
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
        return loss, loss_parts, (diagnostics, temporal_applied)

    payload, loss_values, gradient_norm, amp_info = _training_step(
        model,
        optimizer,
        scaler,
        forward_loss,
        optimizer_mode=_optimizer_mode(train_config),
        max_norm=float(train_config.get("grad_clip", 1.0)),
        epoch=0,
        step=step,
        sample_id=sample_id,
    )
    diagnostics, temporal_applied = payload

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
        "amp": amp_info,
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
    temporal_weight = float((config["train"].get("loss_weights") or {}).get("temporal", 0.0))
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


def _make_batch_sampler(dataset: Any, config: dict[str, Any]) -> SequenceBatchSampler:
    batch_size = int(config["train"].get("batch_size", 1))
    if batch_size <= 1 or config["train"].get("batching") != "independent_sequences":
        raise ValueError("Batched preflight requires independent_sequences with batch_size > 1")
    sampler = SequenceBatchSampler(dataset, batch_size, seed=int(config.get("seed", 2026)))
    largest = max((len(batch) for batch in sampler), default=0)
    if sampler.sequence_count < batch_size or largest != batch_size:
        raise ValueError(
            f"Cannot form the requested full batch_size={batch_size}: "
            f"independent_sequences={sampler.sequence_count}, largest_geometry_compatible_batch={largest}"
        )
    return sampler


def _batch_plan(
    dataset: Any,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    count: int,
    *,
    sampler: SequenceBatchSampler | None = None,
) -> dict[str, Any]:
    sampler = _make_batch_sampler(dataset, config) if sampler is None else sampler
    batches = list(sampler)
    if len(batches) < count:
        raise ValueError(
            "The actual sequence schedule contains fewer batches than requested probes"
        )
    sizes = sampler.sample_sensor_sizes
    previous: dict[tuple[str, str], int] = {}
    predecessors: list[int | None] = []
    for index, item in enumerate(dataset.samples):
        key = sequence_key(item)
        predecessor = previous.get(key)
        if predecessor is not None and (
            item["sequence_index"] != dataset.samples[predecessor]["sequence_index"] + 1
            or sizes[index] != sizes[predecessor]
        ):
            predecessor = None
        predecessors.append(predecessor)
        previous[key] = index
    entries = []
    for number, indices in enumerate(batches):
        if len({sizes[index] for index in indices}) != 1:
            raise ValueError("Sequence batch schedule mixes incompatible sensor shapes")
        entries.append(
            {
                "batch_index": number,
                "dataset_indices": indices,
                "sample_ids": [records[index]["sample_id"] for index in indices],
                "batch_size": len(indices),
                "sensor_size": list(sizes[indices[0]]),
                "sum_nodes": sum(records[index]["model_sampled_events"] for index in indices),
                "sum_actual_directed_edges": sum(
                    records[index]["actual_directed_edges"] for index in indices
                ),
                "sum_candidate_directed_edges": sum(
                    records[index]["candidate_directed_edges"] for index in indices
                ),
                "predecessor_indices": [predecessors[index] for index in indices],
            }
        )
    ranked = sorted(
        entries,
        key=lambda entry: (
            -entry["sum_actual_directed_edges"],
            -entry["sum_candidate_directed_edges"],
            -entry["sum_nodes"],
            entry["batch_index"],
        ),
    )
    numerical: dict[int, list[str]] = {0: ["first_scheduled_batch"]}
    empty = next(
        (
            entry
            for entry in entries
            if any(records[i]["model_sampled_events"] == 0 for i in entry["dataset_indices"])
        ),
        None,
    )
    sparsest = min(
        (record for record in records if record["model_sampled_events"] > 0),
        key=lambda record: (
            record["model_sampled_events"],
            record["actual_directed_edges"],
            record["dataset_index"],
        ),
        default=None,
    )
    sparse = next(
        (
            entry
            for entry in entries
            if sparsest is not None and sparsest["dataset_index"] in entry["dataset_indices"]
        ),
        None,
    )
    largest = max(entry["batch_size"] for entry in entries)
    largest_entry = next(entry for entry in entries if entry["batch_size"] == largest)
    for entry, reason in (
        (empty, "contains_empty_frame"),
        (sparse, "contains_sparsest_nonempty_frame"),
        (largest_entry, "largest_actual_batch"),
    ):
        if entry is not None:
            numerical.setdefault(entry["batch_index"], []).append(reason)
    batch_size = int(config["train"]["batch_size"])
    return {
        "schema": "asgcn_sequence_batch_probe_plan_v1",
        "batching_contract": batching_contract(batch_size),
        "requested_batch_size": batch_size,
        "largest_actual_batch_size": largest,
        "sequence_count": sampler.sequence_count,
        "dataset_samples": len(records),
        "scheduled_batches": len(batches),
        "scheduled_frames": sum(len(batch) for batch in batches),
        "partial_batches": sum(len(batch) < batch_size for batch in batches),
        "schedule_sha256": _canonical_sha256(batches),
        "rank_basis": [
            "sum_actual_edges_desc",
            "sum_candidate_edges_desc",
            "sum_nodes_desc",
            "batch_index_asc",
        ],
        "selected_dense": ranked[:count],
        "selected_numerical": [
            dict(entries[index], reasons=reasons) for index, reasons in numerical.items()
        ],
    }


@torch.no_grad()
def _batch_predecessor_contexts(
    model: ASGCNUNet,
    dataset: Any,
    selected: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[tuple[Any, Any, Any]], list[int | None]]:
    contexts: list[tuple[Any, Any, Any]] = [(None, None, None)] * selected["batch_size"]
    used: list[int | None] = [None] * selected["batch_size"]
    temporal_weight = float((config["train"].get("loss_weights") or {}).get("temporal", 0.0))
    if not config["model"].get("recurrent", True) and temporal_weight <= 0:
        return contexts, used
    valid = [
        (slot, index)
        for slot, index in enumerate(selected["predecessor_indices"])
        if index is not None
    ]
    if not valid:
        return contexts, used
    samples = [move_sample(dataset[index], device) for _, index in valid]
    amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    # This is one actual batched predecessor replay, not B sequential forwards.
    # It is an empirical context-memory probe, not a full-history trajectory.
    with torch.autocast(device_type=device.type, enabled=amp):
        prediction, diagnostics = model.forward_training_batch(samples, [None] * len(samples))
    target = torch.stack([sample["target"] for sample in samples])
    for local, (slot, predecessor) in enumerate(valid):
        state = diagnostics[local]["recurrent_state"]
        contexts[slot] = (
            state.detach() if state is not None else None,
            prediction[local : local + 1].detach(),
            target[local : local + 1].detach(),
        )
        used[slot] = predecessor
    return contexts, used


def _gpu_batch_step(
    model: ASGCNUNet,
    criterion: ReconstructionLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    dataset: Any,
    records: list[dict[str, Any]],
    selected: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    *,
    fresh: bool,
) -> dict[str, Any]:
    samples = [move_sample(dataset[index], device) for index in selected["dataset_indices"]]
    if fresh:
        contexts = [(None, None, None)] * len(samples)
        predecessors: list[int | None] = [None] * len(samples)
    else:
        contexts, predecessors = _batch_predecessor_contexts(
            model, dataset, selected, config, device
        )
    amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    temporal_weight = float((config["train"].get("loss_weights") or {}).get("temporal", 0.0))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start_time = time.perf_counter()

    def forward_loss():
        return forward_training_loss(
            model,
            criterion,
            samples,
            contexts,
            batch_mode=True,
            amp_enabled=amp,
            temporal_weight=temporal_weight,
        )

    payload, loss, gradient_norm, amp_info = _training_step(
        model,
        optimizer,
        scaler,
        forward_loss,
        optimizer_mode=_optimizer_mode(config["train"]),
        max_norm=float(config["train"].get("grad_clip", 1.0)),
        epoch=0,
        step=selected["batch_index"],
        sample_id=" | ".join(selected["sample_ids"]),
    )
    prediction, diagnostics, target = payload
    if (
        len(diagnostics) != len(samples)
        or prediction.shape != target.shape
        or prediction.shape[0] != selected["batch_size"]
        or list(prediction.shape[-2:]) != selected["sensor_size"]
    ):
        raise RuntimeError("Batched training probe output does not match the actual selected batch")
    for index, detail in zip(selected["dataset_indices"], diagnostics, strict=True):
        if (
            int(detail["edges"]) != records[index]["actual_directed_edges"]
            or int(detail["nodes"]) != records[index]["model_sampled_events"]
        ):
            raise RuntimeError("Batched training graph differs from cached per-frame topology")
    if device.type == "cuda":
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end))
        allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    else:
        elapsed = (time.perf_counter() - start_time) * 1000
        allocated = reserved = None
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise FloatingPointError("Batched training probe produced an invalid elapsed time")
    return {
        **selected,
        "execution": "disjoint_graph_batch_and_vectorized_decoder",
        "initialization": "fresh_training_seed" if fresh else "shared_dense_probe_model",
        "context_policy": "none" if fresh else "one_batched_predecessor_replay_training_mode",
        "context_indices": predecessors,
        "loss": loss,
        "gradient_norm": gradient_norm,
        "amp_enabled": amp,
        "amp": amp_info,
        "step_time_ms": elapsed,
        "peak_allocated_mib": allocated,
        "peak_reserved_mib": reserved,
    }


def _run_batch_probe(
    report: dict[str, Any],
    dataset: Any,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    sampler: SequenceBatchSampler,
    count: int,
) -> None:
    plan = _batch_plan(dataset, records, config, count, sampler=sampler)
    probe: dict[str, Any] = {
        "plan": plan,
        "dense_steps": [],
        "numerical_steps": [],
        "passed": False,
    }
    report["batch_training_probe"] = probe
    criterion = ReconstructionLoss(config["train"].get("loss_weights"))
    amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    for selected in plan["selected_numerical"]:
        set_seed(int(config.get("seed", 2026)))
        model = ASGCNUNet(**config["model"]).to(device).train()
        optimizer = _build_optimizer(model, config["train"])
        scaler = _make_grad_scaler(amp)
        probe["numerical_steps"].append(
            _gpu_batch_step(
                model,
                criterion,
                optimizer,
                scaler,
                dataset,
                records,
                selected,
                config,
                device,
                fresh=True,
            )
        )
        del model, optimizer, scaler
    set_seed(int(config.get("seed", 2026)))
    model = ASGCNUNet(**config["model"]).to(device).train()
    optimizer = _build_optimizer(model, config["train"])
    scaler = _make_grad_scaler(amp)
    for selected in plan["selected_dense"]:
        probe["dense_steps"].append(
            _gpu_batch_step(
                model,
                criterion,
                optimizer,
                scaler,
                dataset,
                records,
                selected,
                config,
                device,
                fresh=False,
            )
        )
    probe["passed"] = True


def _topology_input_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    defaults = {
        "event_sampling_factor": 1,
        "graph_radius": 0.08,
        "graph_position_dims": 3,
        "graph_chunk_size": 512,
        "max_graph_edges": 2_000_000,
    }
    return {
        "seed": int(config.get("seed", 2026)),
        "dataset": _public_config(config.get("dataset", {})),
        "graph": {name: model.get(name, default) for name, default in defaults.items()},
    }


def _audited_source(source: Any, allowlist: frozenset[tuple[str, str]]) -> bool:
    return (
        isinstance(source, dict)
        and source.get("git_source_dirty") is False
        and (source.get("source_tree_sha256"), source.get("git_commit")) in allowlist
    )


def _reusable_topology(
    path: Path,
    config: dict[str, Any],
    dataset: Any,
    data_provenance: dict[str, Any],
    implementation: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise TypeError("Reusable profile must contain a JSON object")
    _canonical_sha256(report)
    schema = report.get("schema")
    if schema == "asgcn_training_preflight_v1":
        _require_verified_report_contract(report, path, allow_legacy=True)
        source = report.get("source_provenance")
        if not _audited_source(source, LEGACY_TOPOLOGY_SOURCES):
            raise ValueError("Legacy topology source is not in the audited compatibility allowlist")
        record_device = "cpu"
    elif schema == "asgcn_training_preflight_v2":
        source = report.get("source_provenance")
        if report.get("topology_contract") != implementation:
            if not _audited_source(source, LEGACY_V2_TOPOLOGY_SOURCES):
                raise ValueError("Reusable topology implementation differs from the current code")
            previous_implementation = report.get("topology_contract")
            if not isinstance(previous_implementation, dict) or any(
                previous_implementation.get(field) != implementation.get(field)
                for field in ("schema", "semantics", "torch", "device_type")
            ):
                raise ValueError(
                    "Audited topology reuse requires the same graph semantics and runtime"
                )
            _require_verified_report_contract(report, path)
        if report.get("output") != _artifact_path_label(path):
            raise ValueError("Reusable profile output identity does not match its file")
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("source_tree_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", source["source_tree_sha256"]) is None
        ):
            raise ValueError("Reusable profile has no executable source provenance")
        record_device = (
            report.get("scan_provenance", {})
            .get("origin", {})
            .get("record_device", implementation["device_type"])
        )
    else:
        raise ValueError("Unsupported reusable topology report schema")
    config_provenance = report.get("config_provenance")
    if (
        not isinstance(config_provenance, dict)
        or not isinstance(config_provenance.get("config"), dict)
        or config_provenance.get("sha256") != _canonical_sha256(config_provenance["config"])
    ):
        raise ValueError("Reusable topology original config hash is invalid")
    if _topology_input_config(config_provenance["config"]) != _topology_input_config(config):
        raise ValueError("Reusable topology inputs differ from the current experiment")
    if report.get("data_provenance") != data_provenance:
        raise ValueError("Reusable topology data differs from the current EventHDR files")
    topology = report.get("topology")
    if not isinstance(topology, dict):
        raise TypeError("Reusable profile contains no topology scan")
    records = _validate_topology_records(
        topology.get("samples"), len(dataset), config["model"], dataset=dataset, complete=True
    )
    previous_top_count = report.get("request", {}).get("top_density_count")
    if (
        isinstance(previous_top_count, bool)
        or not isinstance(previous_top_count, int)
        or previous_top_count < 1
    ):
        raise ValueError("Reusable topology density request is invalid")
    expected = _topology_summary(
        records,
        dataset_size=len(dataset),
        data_max_events=config["dataset"].get("max_events"),
        model_config=config["model"],
        top_density_count=previous_top_count,
    )
    if topology != expected:
        raise ValueError("Reusable topology summary does not match its complete sample records")
    return records, {
        "mode": "explicit_report_reuse",
        "source_report": _artifact_path_label(path),
        "source_report_sha256": _file_sha256(path),
        "source_provenance": report["source_provenance"],
        "record_device": record_device,
        "gpu_measurements_reused": False,
        "config_reuse_scope": "seed_dataset_graph_topology_inputs_only",
    }


def training_preflight(
    config: dict[str, Any],
    output_path: str | Path,
    *,
    profile_samples: int = 3,
    top_density_count: int = 10,
    require_cuda: bool = True,
    resume_scan: bool = False,
    reuse_report: str | Path | None = None,
) -> dict[str, Any]:
    journals: list[ScanJournal] = []
    try:
        return _run_training_preflight(
            config,
            output_path,
            profile_samples=profile_samples,
            top_density_count=top_density_count,
            require_cuda=require_cuda,
            resume_scan=resume_scan,
            reuse_report=reuse_report,
            journals=journals,
        )
    finally:
        # Keep the exclusive journal writer lock through final report publication.
        for journal in journals:
            journal.close()


def _run_training_preflight(
    config: dict[str, Any],
    output_path: str | Path,
    *,
    profile_samples: int,
    top_density_count: int,
    require_cuda: bool,
    resume_scan: bool,
    reuse_report: str | Path | None,
    journals: list[ScanJournal],
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

    if not isinstance(resume_scan, bool):
        raise TypeError("resume_scan must be a boolean")
    if resume_scan and reuse_report is not None:
        raise ValueError("--resume-scan and --reuse-report are mutually exclusive")
    destination = Path(output_path)
    journal_path = destination.with_suffix(".scan")
    if reuse_report is not None and Path(reuse_report).resolve() == destination.resolve():
        raise ValueError("Reused profile must be preserved; select a different output path")
    if destination.exists():
        if not resume_scan:
            raise FileExistsError(
                f"Preflight output already exists: {destination}. Move it or choose a new output."
            )
        with destination.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if (
            not isinstance(previous, dict)
            or previous.get("schema") != "asgcn_training_preflight_v2"
            or previous.get("status") not in {"failed", "interrupted"}
            or previous.get("passed") is not False
            or previous.get("output") != _artifact_path_label(destination)
        ):
            raise FileExistsError("Only a failed/interrupted profile can be explicitly resumed")
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
    journal = None
    try:
        _enforce_training_split_status(config)
        dataset = build_dataset(config["dataset"], split="train")
        if len(dataset) < profile_samples:
            raise ValueError("EventHDR training split has fewer samples than profile_samples")
        batch_sampler = (
            _make_batch_sampler(dataset, config)
            if int(config["train"].get("batch_size", 1)) > 1
            else None
        )
        data_provenance = _data_provenance(dataset, config)
        implementation = _topology_implementation_contract(device)
        report["data_provenance"] = data_provenance
        report["topology_contract"] = implementation
        contract = {
            "config": _public_config(config),
            "data": data_provenance,
            "topology_implementation": implementation,
            "dataset_samples": len(dataset),
        }
        reused: list[dict[str, Any]] = []
        origin = {"mode": "fresh", "record_device": device.type, "gpu_measurements_reused": False}
        if reuse_report is not None:
            reused, origin = _reusable_topology(
                Path(reuse_report), config, dataset, data_provenance, implementation
            )
        journal = ScanJournal(journal_path, contract, resume=resume_scan, origin=origin)
        journals.append(journal)
        _validate_topology_records(journal.records, len(dataset), config["model"], dataset=dataset)
        for record in reused:
            journal.append(record)
        if reused:
            journal.flush()
        reused_count = len(journal.records)
        report["scan_provenance"] = {
            "journal": _artifact_path_label(journal_path),
            "origin": journal.origin,
            "resumed": resume_scan,
            "reused_samples": reused_count,
            "new_sample_device": device.type,
            "new_samples": 0,
        }
        with torch.no_grad():
            for index in tqdm(
                range(reused_count, len(dataset)),
                initial=reused_count,
                total=len(dataset),
                desc="preflight-topology",
            ):
                sample = _scan_sample(dataset, index, device)
                journal.append(_sample_topology(sample, config["model"], index))
        journal.flush()
        records = journal.records
        report["scan_provenance"]["new_samples"] = len(records) - reused_count
        topology = _topology_summary(
            records,
            dataset_size=len(dataset),
            data_max_events=config["dataset"].get("max_events"),
            model_config=config["model"],
            top_density_count=top_density_count,
        )
        report["topology"] = topology
        scan_complete = bool(topology["scan_complete"])
        edge_guard_passed = int(topology["edge_guard_exceeded_samples"]) == 0
        report["checks"]["complete_topology_scan"] = scan_complete
        report["checks"]["edge_guard"] = edge_guard_passed

        selected = topology["top_density_samples"][: min(profile_samples, len(dataset))]
        report["training_probe"]["selected_samples"] = [
            {
                "dataset_index": int(item["dataset_index"]),
                "sample_id": item["sample_id"],
                "actual_directed_edges": int(item["actual_directed_edges"]),
            }
            for item in selected
        ]
        numerical_selection = _numerical_selection(records)
        report["training_probe"]["numerical_selection"] = numerical_selection
        if scan_complete and edge_guard_passed:
            criterion = ReconstructionLoss(config["train"].get("loss_weights"))
            amp_enabled = bool(config["train"].get("amp", True)) and device.type == "cuda"
            report["training_probe"]["training_protocol"] = _training_protocol(config, device)
            # Each numerical case starts from the same initialization as training.
            # Dense tests must not hide a bad first/empty/sparse backward by first
            # changing weights, BatchNorm buffers or the GradScaler scale.
            for selected_case in numerical_selection:
                set_seed(int(config.get("seed", 2026)))
                model = ASGCNUNet(**config["model"]).to(device).train()
                optimizer = _build_optimizer(model, config["train"])
                scaler = _make_grad_scaler(amp_enabled)
                index = selected_case["dataset_index"]
                measured = _gpu_step(
                    model,
                    criterion,
                    optimizer,
                    scaler,
                    dataset[index],
                    records[index],
                    config,
                    device,
                    0,
                    recurrent_state=None,
                    previous_prediction=None,
                    previous_target=None,
                    context_sample_id=None,
                )
                measured["reasons"] = selected_case["reasons"]
                measured["initialization"] = "fresh_training_seed_no_recurrent_context"
                report["training_probe"]["numerical_probes"].append(measured)
                del model, optimizer, scaler
            set_seed(int(config.get("seed", 2026)))
            model = ASGCNUNet(**config["model"]).to(device).train()
            optimizer = _build_optimizer(model, config["train"])
            scaler = _make_grad_scaler(amp_enabled)
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
            # Do not keep the framewise probe's model, optimizer or predecessor
            # tensors alive while measuring the actual batched training path.
            del model, optimizer, scaler, context
            if batch_sampler is not None:
                _run_batch_probe(
                    report, dataset, records, config, device, batch_sampler, profile_samples
                )
            report["checks"]["forward_backward"] = (
                len(report["training_probe"]["steps"]) == len(selected)
                and len(report["training_probe"]["numerical_probes"]) == len(numerical_selection)
                and (batch_sampler is None or report["batch_training_probe"]["passed"] is True)
            )
            report["checks"]["cuda_oom_free"] = True if cuda_ready else None
        else:
            report["training_probe"]["failure_category"] = "edge_guard_exceeded"
    except (ScanInUseError, FileExistsError):
        # An overlapping invocation may share this output. Never publish its
        # refusal over the active writer's eventual report.
        raise
    except KeyboardInterrupt as error:
        if journal is not None:
            journal.flush()
        report["status"] = "interrupted"
        report["training_probe"]["failure_category"] = "interrupted"
        report["training_probe"]["failure"] = _safe_failure(error, config, destination)
        report["scan_samples_committed"] = journal.committed if journal is not None else 0
        save_json(destination, report)
        raise
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        ArithmeticError,
        AssertionError,
    ) as error:
        if journal is not None:
            journal.flush()
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
        report["training_probe"]["failure"] = _safe_failure(error, config, destination)
        report["scan_samples_committed"] = journal.committed if journal is not None else 0
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


def _validate_probe_amp(measured: dict[str, Any], effective: bool, *, fresh: bool) -> None:
    if measured.get("amp_enabled") is not effective:
        raise ValueError("Training preflight AMP mode differs from its training protocol")
    amp = measured.get("amp")
    if not isinstance(amp, dict) or set(amp) != {"scale_before", "scale_after", "retries"}:
        raise ValueError("Training preflight AMP diagnostics are incomplete")
    for field in ("scale_before", "scale_after"):
        value = amp[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError("Training preflight AMP scale must be finite and positive")
    retries = amp["retries"]
    if (
        isinstance(retries, bool)
        or not isinstance(retries, int)
        or not 0 <= retries <= _AMP_MAX_RETRIES
    ):
        raise ValueError("Training preflight AMP retry count is invalid")
    if not effective and (retries != 0 or amp["scale_before"] != 1 or amp["scale_after"] != 1):
        raise ValueError("Training preflight reports AMP retries/scaling while AMP is disabled")
    if fresh and amp["scale_before"] != (65536.0 if effective else 1.0):
        raise ValueError("Training preflight numerical probe did not use the fresh GradScaler")


def _validate_batch_probe(
    report: dict[str, Any], expected_plan: dict[str, Any] | None = None
) -> None:
    config = report["config_provenance"]["config"]
    batch_size = int(config.get("train", {}).get("batch_size", 1))
    probe = report.get("batch_training_probe")
    if batch_size == 1:
        if probe is not None:
            raise ValueError("A framewise configuration must not claim a batched training gate")
        return
    if config["train"].get("batching") != "independent_sequences":
        raise ValueError("Batched training gate requires independent_sequences")
    if not isinstance(probe, dict) or probe.get("passed") is not True:
        raise ValueError("Batched training requires a passed actual-batch CUDA training probe")
    plan = probe.get("plan")
    if not isinstance(plan, dict) or plan.get("schema") != "asgcn_sequence_batch_probe_plan_v1":
        raise ValueError("Batched training probe has no valid sequence schedule plan")
    if (
        plan.get("batching_contract") != batching_contract(batch_size)
        or plan.get("requested_batch_size") != batch_size
        or plan.get("largest_actual_batch_size") != batch_size
        or plan.get("dataset_samples") != report["topology"]["dataset_samples"]
        or plan.get("scheduled_frames") != report["topology"]["dataset_samples"]
        or not isinstance(plan.get("schedule_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", plan["schedule_sha256"]) is None
    ):
        raise ValueError("Batched training probe does not represent the requested full batch")
    if expected_plan is not None and plan != expected_plan:
        raise ValueError("Batched training probe differs from the actual dataset sequence schedule")
    dense = plan.get("selected_dense")
    numerical = plan.get("selected_numerical")
    if (
        not isinstance(dense, list)
        or len(dense) != report["request"]["profile_samples"]
        or not isinstance(numerical, list)
        or not 1 <= len(numerical) <= 4
    ):
        raise ValueError("Batched training probe selection is incomplete")
    if not any(
        entry.get("batch_index") == 0 and "first_scheduled_batch" in entry.get("reasons", [])
        for entry in numerical
    ) or not any(
        entry.get("batch_size") == batch_size and "largest_actual_batch" in entry.get("reasons", [])
        for entry in numerical
    ):
        raise ValueError("Batched training numerical coverage lacks first/full-batch cases")
    effective = report["training_probe"]["training_protocol"]["mixed_precision"]["effective"]
    if not isinstance(effective, bool):
        raise TypeError("Batched training gate has no mixed-precision protocol")
    records = report["topology"]["samples"]
    history_enabled = (
        bool(config["model"].get("recurrent", True))
        or float((config["train"].get("loss_weights") or {}).get("temporal", 0.0)) > 0
    )
    measured_full = False
    for selection, field, fresh in (
        (numerical, "numerical_steps", True),
        (dense, "dense_steps", False),
    ):
        measurements = probe.get(field)
        if not isinstance(measurements, list) or len(measurements) != len(selection):
            raise ValueError("Batched training probe did not measure every selected batch")
        for chosen, measured in zip(selection, measurements, strict=True):
            if not isinstance(chosen, dict) or not isinstance(measured, dict):
                raise TypeError("Batched training selections and measurements must be objects")
            indices = chosen.get("dataset_indices")
            if (
                not isinstance(indices, list)
                or not indices
                or any(
                    isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(records)
                    for i in indices
                )
                or len(set(indices)) != len(indices)
                or chosen.get("batch_size") != len(indices)
                or not 1 <= len(indices) <= batch_size
                or chosen.get("sample_ids") != [records[i]["sample_id"] for i in indices]
                or chosen.get("sum_nodes")
                != sum(records[i]["model_sampled_events"] for i in indices)
                or chosen.get("sum_actual_directed_edges")
                != sum(records[i]["actual_directed_edges"] for i in indices)
                or chosen.get("sum_candidate_directed_edges")
                != sum(records[i]["candidate_directed_edges"] for i in indices)
            ):
                raise ValueError("Batched training selection differs from its per-frame topology")
            if any(measured.get(key) != value for key, value in chosen.items()):
                raise ValueError("Batched training measurement identity differs from its selection")
            if (
                measured.get("execution") != "disjoint_graph_batch_and_vectorized_decoder"
                or measured.get("initialization")
                != ("fresh_training_seed" if fresh else "shared_dense_probe_model")
                or measured.get("context_policy")
                != ("none" if fresh else "one_batched_predecessor_replay_training_mode")
                or measured.get("context_indices")
                != (
                    chosen["predecessor_indices"]
                    if not fresh and history_enabled
                    else [None] * len(indices)
                )
            ):
                raise ValueError("Batched training probe execution/context contract is invalid")
            _validate_probe_amp(measured, effective, fresh=fresh)
            for name in ("step_time_ms", "peak_allocated_mib", "peak_reserved_mib"):
                value = measured.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value <= 0
                ):
                    raise ValueError(f"Batched training probe has invalid {name}")
            loss = measured.get("loss")
            norm = measured.get("gradient_norm")
            if (
                not isinstance(loss, dict)
                or "total" not in loss
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in loss.values()
                )
                or isinstance(norm, bool)
                or not isinstance(norm, (int, float))
                or not math.isfinite(float(norm))
                or norm < 0
            ):
                raise ValueError("Batched training probe contains non-finite loss or gradients")
            measured_full = measured_full or len(indices) == batch_size
    if not measured_full:
        raise ValueError("No actual full-sized batch was measured by the CUDA training gate")


def _require_verified_report_contract(
    report: Any, report_path: Path, *, allow_legacy: bool = False
) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise TypeError("Training preflight report must contain a JSON object")
    # Serializing with allow_nan=False rejects hand-written NaN/Infinity values.
    _canonical_sha256(report)
    legacy = report.get("schema") == "asgcn_training_preflight_v1"
    if report.get("schema") != "asgcn_training_preflight_v2" and not (allow_legacy and legacy):
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
    if not legacy:
        public_config = report.get("config_provenance", {}).get("config")
        if not isinstance(public_config, dict) or not isinstance(public_config.get("model"), dict):
            raise ValueError("Training preflight has no topology configuration")
        records = _validate_topology_records(
            topology.get("samples"),
            topology["dataset_samples"],
            public_config["model"],
            complete=True,
        )
        expected_topology = _topology_summary(
            records,
            dataset_size=len(records),
            data_max_events=public_config.get("dataset", {}).get("max_events"),
            model_config=public_config["model"],
            top_density_count=int(request.get("top_density_count", 0)),
        )
        if topology != expected_topology:
            raise ValueError("Training preflight topology summary differs from its records")
        expected_numerical = _numerical_selection(records)
        if probe.get("numerical_selection") != expected_numerical:
            raise ValueError("Training preflight numerical coverage selection is incomplete")
        numerical = probe.get("numerical_probes")
        if not isinstance(numerical, list) or len(numerical) != len(expected_numerical):
            raise ValueError(
                "Training preflight did not complete first/empty/sparse numerical probes"
            )
        mixed = probe.get("training_protocol", {}).get("mixed_precision", {})
        effective_amp = mixed.get("effective")
        if not isinstance(effective_amp, bool):
            raise ValueError("Training preflight has no effective mixed-precision protocol")
        for measured, chosen in zip(numerical, expected_numerical, strict=True):
            if not isinstance(measured, dict) or any(
                measured.get(field) != chosen.get(field)
                for field in ("dataset_index", "sample_id", "actual_directed_edges", "reasons")
            ):
                raise ValueError("Training preflight numerical probe identity differs")
            if measured.get("initialization") != "fresh_training_seed_no_recurrent_context":
                raise ValueError(
                    "Training preflight numerical probes did not use fresh initialization"
                )
            _validate_probe_amp(measured, effective_amp, fresh=True)
            for field in ("step_time_ms", "peak_allocated_mib", "peak_reserved_mib"):
                value = measured.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                ):
                    raise ValueError(f"Training preflight numerical probe has invalid {field}")
        for measured in [*steps, *numerical]:
            _validate_probe_amp(measured, effective_amp, fresh=False)
            norm = measured.get("gradient_norm")
            loss = measured.get("loss")
            if (
                isinstance(norm, bool)
                or not isinstance(norm, (int, float))
                or not math.isfinite(float(norm))
                or norm < 0
                or not isinstance(loss, dict)
                or "total" not in loss
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in loss.values()
                )
            ):
                raise ValueError("Training preflight measured non-finite loss or gradients")
        _validate_batch_probe(report)
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
    if report.get("topology_contract") != _topology_implementation_contract(device):
        raise ValueError("Topology implementation differs from the preflight report")
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
        _validate_topology_records(
            report["topology"]["samples"],
            len(dataset),
            config["model"],
            dataset=dataset,
            complete=True,
        )
        if int(config["train"].get("batch_size", 1)) > 1:
            expected_plan = _batch_plan(
                dataset,
                report["topology"]["samples"],
                config,
                int(report["request"]["profile_samples"]),
            )
            _validate_batch_probe(report, expected_plan)
    finally:
        if hasattr(dataset, "close"):
            dataset.close()

    verification = {
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
    batch_size = int(config["train"].get("batch_size", 1))
    if batch_size > 1:
        batched = report["batch_training_probe"]
        plan = batched["plan"]
        verification["batch_size"] = batch_size
        verification["batch_preflight"] = {
            "contract": plan["batching_contract"],
            "schedule_sha256": plan["schedule_sha256"],
            "measured_batches": len(batched["dense_steps"]) + len(batched["numerical_steps"]),
            "largest_measured_batch_size": batch_size,
        }
    return verification


__all__ = ["training_preflight", "verify_training_preflight"]
