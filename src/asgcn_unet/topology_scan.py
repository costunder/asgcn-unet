from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .data import build_dataset
from .engine import (
    _artifact_path_label,
    _canonical_sha256,
    _dataset_content_fingerprint,
    _dataset_coverage_summary,
    _dataset_index_contract,
    _evaluation_dataset_transform_contract,
    _evaluation_manifest_contract,
    _file_sha256,
    _load_data_hash_cache,
)
from .graph import prepare_event_nodes, radius_graph_topology, uniformly_sample_events
from .scan import ScanJournal, canonical_hash
from .utils import resolve_device, save_json, set_seed, validate_experiment_config


def _topology_contract(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    model = config["model"]
    package = Path(__file__).resolve().parent
    dataset_module = {
        "eventhdr": "data/eventhdr.py",
        "eventaid_r_zip": "data/eventaid_r.py",
    }.get(config["dataset"]["type"])
    implementation_files = [
        "topology_scan.py",
        "graph.py",
        "data/common.py",
        "data/factory.py",
    ]
    if dataset_module is not None:
        implementation_files.append(dataset_module)
    return {
        "schema": "asgcn_eval_topology_v1",
        "semantics": "ordered_normalized_float32_strict_radius_v1",
        "event_sampling_factor": int(model.get("event_sampling_factor", 1)),
        "graph_radius": float(model.get("graph_radius", 0.08)),
        "graph_position_dims": int(model.get("graph_position_dims", 3)),
        "graph_chunk_size": int(model.get("graph_chunk_size", 512)),
        "implementation": {
            relative: _file_sha256(package / relative) for relative in implementation_files
        },
        "torch": str(torch.__version__),
        "device_type": device.type,
    }


def _sample_record(
    sample: dict[str, Any], model: dict[str, Any], dataset_index: int
) -> dict[str, Any]:
    retained = int(sample["events"].shape[0])
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    raw = int(metadata.get("raw_event_count", retained))
    cropped = int(metadata.get("cropped_event_count", retained))
    if raw < cropped or cropped < retained:
        raise ValueError(f"Invalid event-count provenance for sample {dataset_index}")
    sampled = uniformly_sample_events(
        sample["events"], int(model.get("event_sampling_factor", 1))
    )
    _, positions = prepare_event_nodes(sampled, sample["sensor_size"])
    topology = radius_graph_topology(
        positions,
        float(model.get("graph_radius", 0.08)),
        position_dims=int(model.get("graph_position_dims", 3)),
        chunk_size=int(model.get("graph_chunk_size", 512)),
    )
    nodes = int(topology["nodes"])
    possible = nodes * max(nodes - 1, 0)
    edges = int(topology["actual_directed_edges"])
    return {
        "dataset_index": dataset_index,
        "sample_id": str(sample.get("sample_id", dataset_index)),
        "scene": str(metadata.get("scene", "unknown")),
        "sequence_index": (
            int(metadata["sequence_index"])
            if metadata.get("sequence_index") is not None
            else None
        ),
        "raw_events": raw,
        "cropped_events": cropped,
        "retained_events": retained,
        "model_sampled_events": nodes,
        "candidate_directed_edges": int(topology["candidate_directed_edges"]),
        "actual_directed_edges": edges,
        "directed_edge_density": edges / possible if possible else 0.0,
        "max_degree": int(topology["max_degree"]),
        "isolated_nodes": int(topology["isolated_nodes"]),
        "isolate_ratio": float(topology["isolate_ratio"]),
    }


def _validate_records(
    records: list[dict[str, Any]], *, start_index: int, dataset_size: int
) -> None:
    if len(records) > dataset_size - start_index:
        raise ValueError("Topology journal contains too many records")
    identities: set[str] = set()
    for offset, record in enumerate(records):
        expected_index = start_index + offset
        if not isinstance(record, dict) or record.get("dataset_index") != expected_index:
            raise ValueError("Topology journal records are not a contiguous dataset range")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in identities:
            raise ValueError("Topology journal sample identities are invalid")
        identities.add(sample_id)
        for field in (
            "raw_events",
            "cropped_events",
            "retained_events",
            "model_sampled_events",
            "candidate_directed_edges",
            "actual_directed_edges",
            "max_degree",
            "isolated_nodes",
        ):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Topology journal contains an invalid graph count")


def scan_evaluation_topology(
    config: dict[str, Any],
    output_path: str | Path,
    *,
    start_index: int = 0,
    known_prefix_max_edges: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Count exact evaluation graph edges without building a model or edge list."""
    validate_experiment_config(config)
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index must be a non-negative integer")
    if known_prefix_max_edges is not None and (
        isinstance(known_prefix_max_edges, bool)
        or not isinstance(known_prefix_max_edges, int)
        or known_prefix_max_edges < 1
    ):
        raise ValueError("known_prefix_max_edges must be a positive integer")
    if start_index and known_prefix_max_edges is None:
        raise ValueError("A nonzero start_index requires known_prefix_max_edges")
    if not start_index and known_prefix_max_edges is not None:
        raise ValueError("known_prefix_max_edges is valid only with a nonzero start_index")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a boolean")

    destination = Path(output_path)
    journal_path = destination.with_suffix(".scan")
    cache_path = destination.with_suffix(".data_hash_cache.json")
    if destination.exists():
        raise FileExistsError(
            f"Topology scan output already exists: {destination}. Choose a new output."
        )
    if resume and not journal_path.exists():
        raise FileNotFoundError("No topology scan journal exists for --resume")
    if not resume and journal_path.exists():
        raise FileExistsError("Topology scan journal already exists; use --resume explicitly")

    set_seed(int(config.get("seed", 2026)))
    device = resolve_device(config.get("device", "auto"))
    dataset = build_dataset(config["dataset"], split="eval")
    journal: ScanJournal | None = None
    started = time.perf_counter()
    try:
        dataset_size = len(dataset)
        if start_index >= dataset_size:
            raise ValueError(
                f"start_index={start_index} must be smaller than dataset size {dataset_size}"
            )
        cache = _load_data_hash_cache(cache_path, rehash=False)
        content = _dataset_content_fingerprint(dataset, cache)
        save_json(cache_path, {"version": 1, "files": cache})
        dataset_contract = {
            "content": content,
            "index": _dataset_index_contract(dataset),
            "transform": _evaluation_dataset_transform_contract(config),
            "manifest": _evaluation_manifest_contract(config),
            "coverage": _dataset_coverage_summary(dataset, config["dataset"]),
        }
        contract = {
            "schema": "asgcn_eval_topology_scan_contract_v1",
            "seed": int(config.get("seed", 2026)),
            "dataset": dataset_contract,
            "topology": _topology_contract(config, device),
            "range": {
                "start_index": start_index,
                "stop_index_exclusive": dataset_size,
                "known_prefix_max_edges": known_prefix_max_edges,
            },
        }
        journal = ScanJournal(journal_path, contract, resume=resume)
        _validate_records(journal.records, start_index=start_index, dataset_size=dataset_size)
        completed = len(journal.records)
        try:
            with torch.no_grad():
                for index in tqdm(
                    range(start_index + completed, dataset_size),
                    initial=completed,
                    total=dataset_size - start_index,
                    desc="scan-eval-topology",
                ):
                    sample = dataset[index]
                    sample = dict(sample)
                    sample["events"] = sample["events"].to(device=device, non_blocking=True)
                    journal.append(_sample_record(sample, config["model"], index))
        finally:
            journal.flush()

        records = journal.records
        _validate_records(records, start_index=start_index, dataset_size=dataset_size)
        if len(records) != dataset_size - start_index:
            raise RuntimeError("Topology tail scan did not cover its complete requested range")
        tail_max = max(
            records,
            key=lambda item: (
                int(item["actual_directed_edges"]),
                int(item["candidate_directed_edges"]),
                -int(item["dataset_index"]),
            ),
        )
        tail_edges = int(tail_max["actual_directed_edges"])
        global_exact = start_index == 0 or tail_edges >= int(known_prefix_max_edges or 0)
        guard_upper_bound = max(tail_edges, int(known_prefix_max_edges or 0))
        global_sample = tail_max if global_exact else None
        result = {
            "schema": "asgcn_eval_topology_scan_v1",
            "status": "passed",
            "output": _artifact_path_label(destination),
            "journal": _artifact_path_label(journal_path),
            "contract": contract,
            "contract_sha256": canonical_hash(contract),
            "dataset_samples": dataset_size,
            "start_index": start_index,
            "stop_index_exclusive": dataset_size,
            "scanned_samples": len(records),
            "known_prefix_samples": start_index,
            "known_prefix_max_edges": known_prefix_max_edges,
            "tail_max_sample": tail_max,
            "global_max_sample": global_sample,
            "global_max_actual_directed_edges": tail_edges if global_exact else None,
            "global_max_is_exact": global_exact,
            "global_max_edges_lower_bound": tail_edges,
            "global_max_edges_upper_bound": guard_upper_bound,
            "global_edge_guard_upper_bound": guard_upper_bound,
            "elapsed_seconds": time.perf_counter() - started,
        }
        result["sha256"] = _canonical_sha256(
            {key: value for key, value in result.items() if key != "sha256"}
        )
        save_json(destination, result)
        return result
    finally:
        if journal is not None:
            journal.close()
        if hasattr(dataset, "close"):
            dataset.close()
