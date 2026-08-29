from __future__ import annotations

import copy
import hashlib
import math
import random
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import build_dataset, collate_samples, load_eventhdr_split_manifest
from .graph import PAPER_CORE_VERSION, PaperSplineConv
from .losses import ReconstructionLoss
from .metrics import (
    MetricAccumulator,
    frame_metrics,
    percentile,
    temporal_consistency_error,
)
from .model import ASGCNReconstructor
from .utils import (
    atomic_torch_save,
    load_json,
    move_sample,
    resolve_device,
    save_image,
    save_json,
    set_seed,
    write_frame_csv,
)


def build_model(config: dict[str, Any]) -> ASGCNReconstructor:
    return ASGCNReconstructor(**config)


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _model_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Return a deterministic digest that binds checkpoint metadata to tensor bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"Model state entry {name!r} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_model_state_digest(
    state: dict[str, torch.Tensor],
    checkpoint: dict[str, Any],
    checkpoint_path: str | Path,
) -> str:
    """Require every paper-core checkpoint to bind metadata to exact tensor bytes."""
    expected = checkpoint.get("model_state_sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing a valid model_state_sha256")
    computed = _model_state_sha256(state)
    if computed != expected:
        raise ValueError(
            f"Checkpoint {checkpoint_path} model_state_sha256 does not match tensor bytes"
        )
    return computed


def _validate_loaded_conversion_state(
    model: ASGCNReconstructor,
    metadata: dict[str, Any],
    checkpoint_path: str | Path,
) -> None:
    """Cross-check user-editable checkpoint metadata against persistent layer flags."""
    for name, tensor in model.state_dict().items():
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError(f"Checkpoint {checkpoint_path} contains non-finite state: {name}")
    bn_flags = [bool(layer.bn_bypassed.item()) for layer in model.encoder.layers]
    normalized_flags = [bool(layer.snn_normalized.item()) for layer in model.encoder.layers]
    if len(set(bn_flags)) > 1 or len(set(normalized_flags)) > 1:
        raise ValueError(f"Checkpoint {checkpoint_path} contains partially converted graph layers")
    state_bn_folded = all(bn_flags)
    state_normalized = all(normalized_flags)
    metadata_bn_folded = bool(metadata.get("batch_norm_folded"))
    metadata_normalized = bool(metadata.get("parameter_normalized"))
    if metadata_bn_folded != state_bn_folded:
        raise ValueError(
            f"Checkpoint {checkpoint_path} batch_norm_folded metadata disagrees "
            "with layer bn_bypassed state"
        )
    if metadata_normalized != state_normalized:
        raise ValueError(
            f"Checkpoint {checkpoint_path} parameter_normalized metadata disagrees "
            "with layer snn_normalized state"
        )
    checkpoint_type = metadata.get("checkpoint_type")
    if checkpoint_type == "snn_inference" and not (state_bn_folded and state_normalized):
        raise ValueError(
            f"Checkpoint {checkpoint_path} is labeled snn_inference but its graph "
            "layers are not fully BN-folded and Eq. (6)-normalized"
        )
    if checkpoint_type == "snn_inference":
        sample_counts = [int(value) for value in model.encoder.calibration_samples_seen.tolist()]
        if not sample_counts or min(sample_counts) < 1:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has graph layers without valid calibration"
            )
        for index, layer in enumerate(model.encoder.layers):
            if not bool((layer.activation_max > 0).all()):
                raise ValueError(
                    f"Checkpoint {checkpoint_path} layer {index} has non-positive lambda"
                )
            if not bool((layer.threshold > 0).all()):
                raise ValueError(
                    f"Checkpoint {checkpoint_path} layer {index} has non-positive threshold"
                )
            if not torch.equal(layer.threshold, torch.ones_like(layer.threshold)):
                raise ValueError(
                    f"Checkpoint {checkpoint_path} layer {index} threshold is not the "
                    "unit threshold produced by Eq. (6) conversion"
                )
        summary = metadata.get("snn_calibration_summary")
        if not isinstance(summary, dict):
            raise ValueError(f"Checkpoint {checkpoint_path} is missing calibration summary")
        if summary.get("valid_samples_per_layer") != sample_counts:
            raise ValueError(
                f"Checkpoint {checkpoint_path} calibration metadata disagrees with layer state"
            )
        minimum = min(sample_counts)
        if int(summary.get("minimum_valid_samples", 0) or 0) != minimum:
            raise ValueError(f"Checkpoint {checkpoint_path} calibration minimum is inconsistent")
        if int(metadata.get("snn_calibration_valid_samples", 0) or 0) != minimum:
            raise ValueError(
                f"Checkpoint {checkpoint_path} valid calibration count is inconsistent"
            )
        selected = int(metadata.get("snn_calibration_samples", 0) or 0)
        if selected < minimum:
            raise ValueError(
                f"Checkpoint {checkpoint_path} selected calibration count is inconsistent"
            )
    if checkpoint_type in {"ann_inference", "training"} and (state_bn_folded or state_normalized):
        raise ValueError(
            f"Checkpoint {checkpoint_path} is labeled {checkpoint_type} but contains "
            "converted SNN graph-layer state"
        )


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    fallback_model_config: dict[str, Any],
) -> tuple[ASGCNReconstructor, dict[str, Any]]:
    checkpoint = _load_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} must contain a dictionary")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise TypeError(
            f"Checkpoint {checkpoint_path} has no embedded model_config. Legacy/raw "
            "state dictionaries are incompatible with the paper-core architecture."
        )
    architecture_version = model_config.get("architecture_version")
    if architecture_version != PAPER_CORE_VERSION:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has architecture_version="
            f"{architecture_version!r}; paper-core version {PAPER_CORE_VERSION} is "
            "required. Legacy edge-MLP checkpoints cannot be loaded as ASGCN."
        )
    if model_config != fallback_model_config:
        raise ValueError(
            f"Checkpoint {checkpoint_path} model_config differs from config.model. "
            "Use the exact training architecture; inference-only SNN dynamics must "
            "be selected with the explicit --snn-dynamics override."
        )
    if "model" not in checkpoint or not isinstance(checkpoint["model"], dict):
        raise TypeError(
            f"Checkpoint {checkpoint_path} has no model state dictionary; raw state "
            "dictionaries are incompatible with the paper-core checkpoint protocol."
        )
    state = checkpoint.pop("model")
    _validate_model_state_digest(state, checkpoint, checkpoint_path)
    metadata = checkpoint
    model = build_model(model_config).to(device)
    model.load_state_dict(state, strict=True)
    _validate_loaded_conversion_state(model, metadata, checkpoint_path)
    del state
    return model, metadata


def _dataset_group_key(dataset, index: int) -> str:
    """Return a stable file/scene key without decoding the sample payload."""
    records = getattr(dataset, "samples", None)
    if records is not None:
        record = records[index]
        if isinstance(record, dict):
            if record.get("scene") is not None:
                return str(record["scene"])
            if record.get("path") is not None:
                path = Path(record["path"])
                root = getattr(dataset, "root", None)
                if root is not None:
                    try:
                        return path.relative_to(Path(root)).as_posix()
                    except ValueError:
                        pass
                return path.name
    # Supported datasets expose ``samples``. This fallback keeps custom datasets
    # usable in validation without making balanced sampling silently incorrect.
    sample = dataset[index]
    metadata = sample.get("metadata", {})
    return str(metadata.get("scene") or metadata.get("source") or "unknown")


def _balanced_sample_indices(dataset, limit: int | None, seed: int = 2026) -> list[int]:
    """Select near-equal, time-spread samples from every file/scene.

    Round-robin allocation prevents long files from consuming the complete
    validation/calibration budget. Within each group, linspace covers the whole
    sequence instead of only its prefix. Sorting the result keeps each scene's
    original temporal order for the recurrent decoder.
    """
    size = len(dataset)
    if limit is None or int(limit) >= size:
        return list(range(size))
    limit = int(limit)
    if limit < 1:
        raise ValueError("sample limit must be at least 1")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index in range(size):
        grouped[_dataset_group_key(dataset, index)].append(index)
    group_keys = sorted(grouped)
    random.Random(seed).shuffle(group_keys)

    quotas = {key: 0 for key in group_keys}
    allocated = 0
    depth = 0
    while allocated < limit:
        added = False
        for key in group_keys:
            indices = grouped[key]
            if depth < len(indices):
                quotas[key] += 1
                allocated += 1
                added = True
                if allocated == limit:
                    break
        if not added:
            break
        depth += 1

    selected: list[int] = []
    for key, quota in quotas.items():
        indices = grouped[key]
        if quota == 1:
            offsets = [len(indices) // 2]
        elif quota > 1:
            offsets = np.linspace(0, len(indices) - 1, num=quota, dtype=int).tolist()
        else:
            offsets = []
        selected.extend(indices[offset] for offset in offsets)
    return sorted(selected)


def _balanced_contiguous_indices(
    dataset,
    limit: int | None,
    seed: int = 2026,
    *,
    require_all_groups: bool = False,
) -> list[int]:
    """Select one deterministic contiguous window from every allocated group."""
    size = len(dataset)
    if limit is None or int(limit) >= size:
        return list(range(size))
    limit = int(limit)
    if limit < 1:
        raise ValueError("sample limit must be at least 1")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index in range(size):
        grouped[_dataset_group_key(dataset, index)].append(index)
    if require_all_groups and limit < len(grouped):
        raise ValueError(
            f"validation sample limit {limit} is smaller than the {len(grouped)} "
            "available groups; every validation group must be represented"
        )

    group_keys = sorted(grouped)
    allocation_order = list(group_keys)
    random.Random(seed).shuffle(allocation_order)
    quotas = {key: 0 for key in group_keys}
    allocated = 0
    depth = 0
    while allocated < limit:
        added = False
        for key in allocation_order:
            if depth < len(grouped[key]):
                quotas[key] += 1
                allocated += 1
                added = True
                if allocated == limit:
                    break
        if not added:
            break
        depth += 1

    window_rng = random.Random(f"contiguous:{seed}")
    selected: list[int] = []
    for key in group_keys:
        indices = grouped[key]
        quota = quotas[key]
        if quota:
            start = window_rng.randrange(len(indices) - quota + 1)
            selected.extend(indices[start : start + quota])
    return sorted(selected)


def _representative_schedule(
    dataset, count: int, seed: int, *, contiguous: bool = False
) -> list[int]:
    """Build an exact-length balanced schedule, cycling only if count exceeds data."""
    if count < 0:
        raise ValueError("sample count must be non-negative")
    if count == 0:
        return []
    sampler = _balanced_contiguous_indices if contiguous else _balanced_sample_indices
    base = sampler(dataset, min(count, len(dataset)), seed=seed)
    repeats = (count + len(base) - 1) // len(base)
    return (base * repeats)[:count]


def _prefix_context_schedule(
    dataset,
    scored_indices: list[int],
    max_context_frames: int | None = None,
) -> tuple[list[int], set[int]]:
    """Prepend contiguous predecessor frames as unscored recurrent context."""
    if not scored_indices:
        return [], set()
    if max_context_frames is not None and int(max_context_frames) < 0:
        raise ValueError("max_context_frames must be non-negative or null")
    scored = set(scored_indices)
    if len(scored) != len(scored_indices):
        raise ValueError("prefix context schedule requires unique scored indices")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        grouped[_dataset_group_key(dataset, index)].append(index)

    schedule: list[int] = []
    score_positions: set[int] = set()
    selected_groups = sorted(
        {_dataset_group_key(dataset, index) for index in scored_indices},
        key=lambda key: grouped[key][0],
    )
    for group in selected_groups:
        group_indices = grouped[group]
        selected = [index for index in group_indices if index in scored]
        first_position = group_indices.index(selected[0])
        last_position = group_indices.index(selected[-1])
        start_position = (
            0 if max_context_frames is None else max(0, first_position - int(max_context_frames))
        )
        for index in group_indices[start_position : last_position + 1]:
            position = len(schedule)
            schedule.append(index)
            if index in scored:
                score_positions.add(position)
    return schedule, score_positions


def _dataset_sample_identity(dataset, index: int) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "dataset_index": index,
        "group": _dataset_group_key(dataset, index),
    }
    records = getattr(dataset, "samples", None)
    if records is None or not isinstance(records[index], dict):
        return identity
    record = records[index]
    for key in (
        "source_file",
        "sequence_index",
        "frame_id",
        "image_key",
        "event_name",
        "target_name",
        "start_idx",
        "end_idx",
        "timestamp",
    ):
        value = record.get(key)
        if value is not None:
            identity[key] = value.item() if isinstance(value, np.generic) else value
    return identity


def _dataset_source_fingerprint(dataset) -> dict[str, Any]:
    root = Path(getattr(dataset, "root", ".")).resolve()
    sources = getattr(dataset, "files", None)
    if sources is None:
        sources = getattr(dataset, "zip_paths", [])
    files = []
    for raw_path in sources:
        path = Path(raw_path).resolve()
        stat = path.stat()
        try:
            key = path.relative_to(root).as_posix()
        except ValueError:
            key = path.as_posix()
        files.append(
            {
                "path": key,
                "size": stat.st_size,
            }
        )
    return {"files": files}


def _dataset_content_fingerprint(
    dataset, cache: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Hash every selected source file for path- and mtime-independent exact resume."""
    cache = {} if cache is None else cache
    root = Path(getattr(dataset, "root", ".")).resolve()
    sources = getattr(dataset, "files", None)
    if sources is None:
        sources = getattr(dataset, "zip_paths", [])
    combined = hashlib.sha256()
    total_bytes = 0
    entries: list[tuple[str, Path]] = []
    for value in sources:
        raw_path = Path(value).resolve()
        try:
            relative = raw_path.relative_to(root).as_posix()
        except ValueError:
            relative = raw_path.name
        entries.append((relative, raw_path))
    for relative, raw_path in sorted(entries):
        stat = raw_path.stat()
        cache_key = str(raw_path)
        signature = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
        cached = cache.get(cache_key)
        if not isinstance(cached, dict) or any(
            cached.get(key) != value for key, value in signature.items()
        ):
            file_hash = hashlib.sha256()
            size = 0
            with raw_path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    file_hash.update(chunk)
                    size += len(chunk)
            cached = {**signature, "sha256": file_hash.hexdigest()}
            cache[cache_key] = cached
        size = int(cached["size"])
        digest = str(cached["sha256"])
        total_bytes += size
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(size).encode("ascii"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return {
        "algorithm": "sha256-full-files-v1",
        "files": len(sources),
        "bytes": total_bytes,
        "sha256": combined.hexdigest(),
    }


def _load_data_hash_cache(path: Path, rehash: bool) -> dict[str, dict[str, Any]]:
    if rehash or not path.is_file():
        return {}
    try:
        payload = load_json(path)
    except (OSError, ValueError):
        return {}
    if payload.get("version") != 1 or not isinstance(payload.get("files"), dict):
        return {}
    return payload["files"]


def _sampling_summary(dataset, indices: list[int]) -> dict[str, Any]:
    counts = Counter(_dataset_group_key(dataset, index) for index in indices)
    available_counts = Counter(_dataset_group_key(dataset, index) for index in range(len(dataset)))
    return {
        "selected_samples": len(indices),
        "selected_groups": len(counts),
        "available_groups": len(available_counts),
        "per_group": dict(sorted(counts.items())),
        "available_per_group": dict(sorted(available_counts.items())),
        "selected": [_dataset_sample_identity(dataset, index) for index in indices],
        "source_fingerprint": _dataset_source_fingerprint(dataset),
    }


def _sampling_counts(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep routine epoch logs compact; exact identities live in the protocol."""
    keys = (
        "selected_samples",
        "selected_groups",
        "available_groups",
        "per_group",
        "available_per_group",
        "context_policy",
        "max_context_frames_per_group",
        "context_samples",
        "forward_samples",
    )
    return {key: summary[key] for key in keys if key in summary}


def _sample_sequence_info(
    sample: dict[str, Any],
) -> tuple[str, int | None, tuple[int, int]]:
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    scene = str(metadata.get("scene", "unknown"))
    raw_index = metadata.get("sequence_index")
    try:
        sequence_index = int(raw_index) if raw_index is not None else None
    except (TypeError, ValueError):
        sequence_index = None
    sensor_size = tuple(int(value) for value in sample["sensor_size"])
    return scene, sequence_index, sensor_size


def _continues_sequence(
    scene: str,
    sequence_index: int | None,
    sensor_size: tuple[int, int],
    previous_scene: str | None,
    previous_sequence_index: int | None,
    previous_sensor_size: tuple[int, int] | None,
) -> bool:
    if scene != previous_scene or sensor_size != previous_sensor_size:
        return False
    if sequence_index is None or previous_sequence_index is None:
        return True
    return sequence_index == previous_sequence_index + 1


def _macro_ssim(validation: dict[str, Any]) -> float:
    """Read the scene-balanced selection score, with legacy checkpoint support."""
    macro = validation.get("macro", {})
    if "ssim" in macro:
        return float(macro["ssim"])
    if "ssim" in validation:  # Checkpoints written before structured validation.
        return float(validation["ssim"])
    return float("-inf")


def _resume_best_macro_ssim(checkpoint: dict[str, Any]) -> float:
    """Reject legacy micro-SSIM best scores that cannot be compared to macro SSIM."""
    metric = checkpoint.get("best_metric")
    if metric == "macro_ssim":
        return float(checkpoint.get("best_ssim", _macro_ssim(checkpoint.get("val", {}))))
    validation = checkpoint.get("val", {})
    macro = validation.get("macro", {}) if isinstance(validation, dict) else {}
    if "ssim" in macro:
        return float(macro["ssim"])
    raise ValueError(
        "Resume checkpoint predates macro-SSIM model selection, so its best_ssim is "
        "not comparable. Start a new run or migrate the checkpoint with a verified "
        "macro validation score."
    )


def _validate_resume_best_pair(
    resume_checkpoint: dict[str, Any], best_checkpoint: dict[str, Any]
) -> None:
    """Ensure last.pt and best.pt belong to the same exact training run."""
    if best_checkpoint.get("validation_protocol") != resume_checkpoint.get("validation_protocol"):
        raise ValueError("Historical best.pt has a different validation protocol")
    if best_checkpoint.get("model_config") != resume_checkpoint.get("model_config"):
        raise ValueError("Historical best.pt has a different model configuration")
    if best_checkpoint.get("training_protocol") != resume_checkpoint.get("training_protocol"):
        raise ValueError("Historical best.pt has a different training protocol")
    for name, checkpoint in (
        ("resume checkpoint", resume_checkpoint),
        ("historical best.pt", best_checkpoint),
    ):
        if checkpoint.get("paper_core_version") != PAPER_CORE_VERSION:
            raise ValueError(f"{name} does not declare paper_core_version={PAPER_CORE_VERSION}")
    if best_checkpoint.get("best_metric") != "macro_ssim":
        raise ValueError("Historical best.pt does not use macro_ssim model selection")
    resume_best = _resume_best_macro_ssim(resume_checkpoint)
    historical_best = _resume_best_macro_ssim(best_checkpoint)
    if historical_best != resume_best:
        raise ValueError("Historical best.pt score does not match the resume checkpoint")
    if _macro_ssim(best_checkpoint.get("val", {})) != historical_best:
        raise ValueError("Historical best.pt validation score is internally inconsistent")
    if int(best_checkpoint.get("epoch", -1)) > int(resume_checkpoint.get("epoch", -1)):
        raise ValueError("Historical best.pt is newer than the resume checkpoint")
    best_digest = best_checkpoint.get("model_state_sha256")
    resume_digest = resume_checkpoint.get("best_model_state_sha256")
    if not isinstance(best_digest, str) or re.fullmatch(r"[0-9a-f]{64}", best_digest) is None:
        raise ValueError("Historical best.pt is missing a valid model state digest")
    if best_digest != resume_digest:
        raise ValueError("Historical best.pt model digest does not match the resume checkpoint")


def _validate_snn_request(
    inference_mode: str,
    simulation_steps: int,
    checkpoint: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
) -> None:
    if isinstance(simulation_steps, bool) or int(simulation_steps) != simulation_steps:
        raise ValueError("simulation_steps must be an integer")
    simulation_steps = int(simulation_steps)
    if inference_mode == "ann":
        if checkpoint is not None and (
            checkpoint.get("checkpoint_type") == "snn_inference"
            or bool(checkpoint.get("parameter_normalized"))
        ):
            location = f" {checkpoint_path}" if checkpoint_path is not None else ""
            raise ValueError(
                f"ANN inference requires an ANN checkpoint;{location} contains "
                "Eq. (6)-normalized SNN weights. Use best.pt for ANN inference or "
                "select inference_mode='snn'."
            )
        return
    if inference_mode != "snn":
        return
    if int(simulation_steps) < 1:
        raise ValueError("simulation_steps must be at least 1 for SNN inference")
    if checkpoint is None:
        return
    calibration_samples = int(checkpoint.get("snn_calibration_samples", 0) or 0)
    valid_calibration_samples = int(checkpoint.get("snn_calibration_valid_samples", 0) or 0)
    requirements_met = (
        checkpoint.get("checkpoint_type") == "snn_inference"
        and bool(checkpoint.get("batch_norm_folded"))
        and calibration_samples >= 1
        and valid_calibration_samples >= 1
        and checkpoint.get("paper_core_version") == PAPER_CORE_VERSION
        and bool(checkpoint.get("parameter_normalized"))
    )
    if not requirements_met:
        location = f" {checkpoint_path}" if checkpoint_path is not None else ""
        raise ValueError(
            f"SNN inference requires a calibrated checkpoint;{location} is missing "
            "checkpoint_type=snn_inference, batch_norm_folded=true, "
            "snn_calibration_samples>=1, "
            "snn_calibration_valid_samples>=1, "
            f"paper_core_version={PAPER_CORE_VERSION}, or parameter_normalized=true. "
            "Run calibrate first."
        )


def _set_inference_snn_dynamics(
    model: ASGCNReconstructor,
    inference_mode: str,
    override: str | None,
) -> None:
    if override is None:
        return
    if inference_mode != "snn":
        raise ValueError("snn_dynamics override is only valid for SNN inference")
    if override not in {"literal_eq15", "standard_if"}:
        raise ValueError("snn_dynamics must be 'literal_eq15' or 'standard_if'")
    model.snn_dynamics = override


def _inference_run_label(
    inference_mode: str,
    simulation_steps: int,
    snn_dynamics: str,
) -> str:
    return "ann" if inference_mode == "ann" else f"snn_{snn_dynamics}_T{int(simulation_steps)}"


def _prediction_artifact_stem(sample_id: Any, index: int) -> str:
    """Create a bounded, collision-resistant filename valid on Linux and Windows."""
    raw = str(sample_id)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")[:64] or "sample"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{int(index):08d}_{slug}_{digest}"


def _reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak_memory(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"peak_allocated_mib": None, "peak_reserved_mib": None}
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
    }


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _data_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool = False,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
):
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    loader_options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_samples,
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = (
            True if persistent_workers is None else bool(persistent_workers)
        )
        loader_options["worker_init_fn"] = _seed_worker
        if prefetch_factor is not None:
            if int(prefetch_factor) < 1:
                raise ValueError("prefetch_factor must be at least 1")
            loader_options["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**loader_options)


def _loader_kwargs(section: dict[str, Any]) -> dict[str, Any]:
    return {
        "persistent_workers": section.get("persistent_workers"),
        "prefetch_factor": section.get("prefetch_factor"),
    }


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # PyTorch before the unified torch.amp API.
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Any) -> None:
    if not isinstance(state, dict):
        raise TypeError("Exact resume requires a dictionary rng_state")
    missing = sorted({"python", "numpy", "torch"} - set(state))
    if missing:
        raise ValueError("Exact resume rng_state is missing: " + ", ".join(missing))
    if not torch.is_tensor(state["torch"]):
        raise ValueError("Exact resume rng_state['torch'] must be a tensor")
    if torch.cuda.is_available():
        cuda_state = state.get("cuda")
        if not isinstance(cuda_state, list) or len(cuda_state) != torch.cuda.device_count():
            raise ValueError(
                "Exact CUDA resume requires one rng_state['cuda'] tensor per visible device"
            )
        if any(not torch.is_tensor(value) for value in cuda_state):
            raise ValueError("Exact resume CUDA RNG entries must be tensors")
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Exact resume contains an invalid RNG state schema") from error


def _optimizer_mode(train_config: dict[str, Any]) -> str:
    mode = str(train_config.get("optimizer", "adamw")).strip().lower()
    if mode not in {"adamw", "adam_gc"}:
        raise ValueError("train.optimizer must be 'adamw' or 'adam_gc'")
    return mode


def _scheduler_spec(train_config: dict[str, Any]) -> dict[str, Any] | None:
    raw_milestones = train_config.get("lr_milestones")
    if raw_milestones is None or raw_milestones == []:
        return None
    if not isinstance(raw_milestones, (list, tuple)):
        raise TypeError("train.lr_milestones must be a list of positive epochs")
    milestones = sorted(int(value) for value in raw_milestones)
    if not milestones or any(value < 1 for value in milestones):
        raise ValueError("train.lr_milestones must contain positive epochs")
    gamma = float(train_config.get("lr_gamma", 0.1))
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("train.lr_gamma must be finite and greater than zero")
    return {
        "name": "MultiStepLR",
        "milestones": milestones,
        "gamma": gamma,
        "step_unit": "epoch",
        "step_timing": "after_epoch",
    }


def _build_optimizer(model: torch.nn.Module, train_config: dict[str, Any]) -> torch.optim.Optimizer:
    mode = _optimizer_mode(train_config)
    optimizer_class = torch.optim.AdamW if mode == "adamw" else torch.optim.Adam
    return optimizer_class(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 2e-4)),
        weight_decay=float(train_config.get("weight_decay", 1e-6)),
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer, train_config: dict[str, Any]
) -> torch.optim.lr_scheduler.MultiStepLR | None:
    spec = _scheduler_spec(train_config)
    if spec is None:
        return None
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=spec["milestones"],
        gamma=spec["gamma"],
    )


def _centralize_gradients(model: torch.nn.Module) -> None:
    """Apply paper-style gradient centralization to matrix/kernel gradients."""
    spline_parameters: set[int] = set()
    for module in model.modules():
        if not isinstance(module, PaperSplineConv):
            continue
        for parameter in (module.weight, module.root):
            if parameter is None:
                continue
            spline_parameters.add(id(parameter))
            gradient = parameter.grad
            if gradient is None or gradient.ndim <= 1:
                continue
            dimensions = tuple(range(gradient.ndim - 1))
            gradient.subtract_(gradient.mean(dim=dimensions, keepdim=True))

    for parameter in model.parameters():
        if id(parameter) in spline_parameters:
            continue
        gradient = parameter.grad
        if gradient is None or gradient.ndim <= 1:
            continue
        dimensions = tuple(range(1, gradient.ndim))
        gradient.subtract_(gradient.mean(dim=dimensions, keepdim=True))


def _source_tree_sha256(project_root: Path) -> str:
    """Hash executable project source so resume cannot cross silent code edits."""
    digest = hashlib.sha256()
    source_root = project_root / "src"
    files = sorted(path for path in source_root.rglob("*.py") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No Python source files found under {source_root}")
    for path in files:
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_provenance(project_root: Path) -> dict[str, Any]:
    """Return best-effort Git identity without making Git a runtime dependency."""

    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-c", f"safe.directory={project_root.as_posix()}", *arguments],
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=normal", "--", "src")
    return {
        "git_commit": commit,
        "git_source_dirty": None if status is None else bool(status),
    }


def _training_protocol(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Return every configured choice that can change the optimization trajectory.

    ``epochs``, logging cadence, the resume path, and output paths are deliberately
    absent: changing those does not alter an already completed optimizer step. The
    normalized values below make omitted defaults compare equal to explicit defaults.
    """
    train_config = config["train"]
    requested_amp = bool(train_config.get("amp", True))
    effective_amp = requested_amp and device.type == "cuda"
    configured_weights = train_config.get("loss_weights") or {}
    loss_weights = {
        "charbonnier": float(configured_weights.get("charbonnier", 1.0)),
        "ssim": float(configured_weights.get("ssim", 0.2)),
        "gradient": float(configured_weights.get("gradient", 0.1)),
        "temporal": float(configured_weights.get("temporal", 0.0)),
    }
    max_train_samples = train_config.get("max_train_samples")
    grad_clip = float(train_config.get("grad_clip", 1.0))
    if not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("train.grad_clip must be finite and greater than zero")
    num_workers = int(train_config.get("num_workers", 0))
    prefetch_factor = train_config.get("prefetch_factor")
    persistent_workers = train_config.get("persistent_workers")
    effective_persistent_workers = (
        None
        if num_workers == 0
        else True
        if persistent_workers is None
        else bool(persistent_workers)
    )
    effective_prefetch_factor = (
        None if num_workers == 0 else 2 if prefetch_factor is None else int(prefetch_factor)
    )
    optimizer_mode = _optimizer_mode(train_config)
    optimizer_name = "AdamW" if optimizer_mode == "adamw" else "Adam"
    raw_validate_every = train_config.get("validate_every", 1)
    validate_every = (
        None if raw_validate_every is None else max(1, int(raw_validate_every))
    )
    project_root = Path(__file__).resolve().parents[2]
    git_provenance = _git_provenance(project_root)
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        compute_capability = list(torch.cuda.get_device_capability(device_index))
    else:
        gpu_name = None
        compute_capability = None
    return {
        "version": 3,
        "seed": int(config.get("seed", 2026)),
        "optimizer": {
            "mode": optimizer_mode,
            "name": optimizer_name,
            "learning_rate": float(train_config.get("learning_rate", 2e-4)),
            "weight_decay": float(train_config.get("weight_decay", 1e-6)),
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "gradient_centralization": optimizer_mode == "adam_gc",
            "gradient_centralization_dimensions": "all_except_output",
        },
        "scheduler": _scheduler_spec(train_config),
        "loss_weights": loss_weights,
        "gradient_clipping": {
            "max_norm": grad_clip,
            "norm_type": 2.0,
        },
        "data_order": {
            "batch_size": int(train_config.get("batch_size", 1)),
            "max_train_samples": (None if max_train_samples is None else int(max_train_samples)),
            "shuffle": False,
            "num_workers": num_workers,
            "persistent_workers": effective_persistent_workers,
            "prefetch_factor": effective_prefetch_factor,
        },
        "mixed_precision": {
            "requested": requested_amp,
            "effective": effective_amp,
            "autocast_dtype": "float16" if effective_amp else None,
            "gradient_scaler": effective_amp,
        },
        "validate_every": validate_every,
        "checkpoint_selection": (
            "single_final_epoch" if validate_every is None else "best_validation_macro_ssim"
        ),
        "recurrent_state_detached_each_sample": True,
        "runtime": {
            "device_type": device.type,
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda if device.type == "cuda" else None,
            "cudnn": (torch.backends.cudnn.version() if device.type == "cuda" else None),
            "gpu_name": gpu_name,
            "compute_capability": compute_capability,
            "cuda_matmul_allow_tf32": (
                bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else None
            ),
            "cudnn_allow_tf32": (
                bool(torch.backends.cudnn.allow_tf32) if device.type == "cuda" else None
            ),
            "cudnn_benchmark": (
                bool(torch.backends.cudnn.benchmark) if device.type == "cuda" else None
            ),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        },
        "source": {
            "source_tree_sha256": _source_tree_sha256(project_root),
            **git_provenance,
        },
    }


def _validate_training_protocol(checkpoint: dict[str, Any], expected: dict[str, Any]) -> None:
    actual = checkpoint.get("training_protocol")
    if actual is None:
        raise ValueError(
            "Resume checkpoint is missing training_protocol and cannot provide an "
            "exact training resume. Start a new run with the current checkpoint schema."
        )
    if not isinstance(actual, dict):
        raise TypeError("Resume checkpoint training_protocol must be a dictionary")
    if actual != expected:
        keys = sorted(
            key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)
        )
        changed = ", ".join(keys) if keys else "unknown fields"
        raise ValueError("Resume training protocol differs from the checkpoint in: " + changed)


def _ensure_finite_loss(
    loss: torch.Tensor,
    loss_parts: dict[str, float],
    *,
    epoch: int,
    step: int,
    sample_id: Any,
) -> None:
    context = f"epoch={epoch}, step={step}, sample={sample_id}"
    invalid_parts = sorted(
        name for name, value in loss_parts.items() if not math.isfinite(float(value))
    )
    invalid_values = [] if bool(torch.isfinite(loss.detach()).all().item()) else ["total loss"]
    invalid_values.extend(f"{name} component" for name in invalid_parts)
    if invalid_values:
        raise FloatingPointError(f"Non-finite {', '.join(invalid_values)} at {context}")


def _clip_and_validate_gradients(
    model: torch.nn.Module,
    max_norm: float,
    *,
    epoch: int,
    step: int,
    sample_id: Any,
) -> float:
    """Clip gradients with one device synchronization for non-finite detection."""
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("train.grad_clip must be finite and greater than zero")
    try:
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm, norm_type=2.0, error_if_nonfinite=True
        )
    except RuntimeError as error:
        raise FloatingPointError(
            "Non-finite gradients after clipping validation at "
            f"epoch={epoch}, step={step}, sample={sample_id}"
        ) from error
    finite_norm = float(total_norm.detach().cpu())
    if not math.isfinite(finite_norm):
        raise FloatingPointError(
            "Non-finite gradient norm after clipping at "
            f"epoch={epoch}, step={step}, sample={sample_id}"
        )
    return finite_norm


def _validation_dataset(config: dict[str, Any]):
    return build_dataset(config["dataset"], split="val")


def _validation_protocol(
    config: dict[str, Any],
    val_sampling: dict[str, Any],
    train_dataset,
    val_dataset,
    digest_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    data = copy.deepcopy(config["dataset"])
    data.pop("root", None)
    data.pop("val_root", None)
    manifest_path = data.pop("split_manifest", None)
    manifest = load_eventhdr_split_manifest(manifest_path) if manifest_path else None
    if manifest is None:
        manifest_identity = None
    else:
        manifest_identity = {
            "status": str(manifest.get("status", "missing")).strip().lower(),
            "split_schema": manifest["split_schema"],
            "train_files": manifest["train_files"],
            "val_files": manifest["val_files"],
            "file_to_scene": manifest["file_to_scene"],
        }
        if manifest["split_schema"] == "physical_scenes_v1":
            manifest_identity.update(
                {
                    "scene_groups": manifest["scene_groups"],
                    "train_scenes": manifest["train_scenes"],
                    "val_scenes": manifest["val_scenes"],
                }
            )
    print("Verifying cached hashes or hashing train/validation files for exact resume...")
    return {
        "version": 5,
        "seed": int(config.get("seed", 2026)),
        "recurrent": bool(config["model"].get("recurrent", True)),
        "dataset_transform": data,
        "split_manifest": manifest_identity,
        "dataset_content": {
            "train": _dataset_content_fingerprint(train_dataset, digest_cache),
            "validation": _dataset_content_fingerprint(val_dataset, digest_cache),
        },
        "max_val_samples": config["train"].get("max_val_samples"),
        "sampling": val_sampling,
        "selection_metric": (
            "single_final_epoch_macro_ssim"
            if config["train"].get("validate_every", 1) is None
            else "macro_ssim"
        ),
        "ssim": "gaussian_valid_11_sigma1.5",
    }


def _enforce_training_split_status(config: dict[str, Any]) -> None:
    manifest_path = config.get("dataset", {}).get("split_manifest")
    if not manifest_path:
        return
    manifest = load_eventhdr_split_manifest(manifest_path)
    status = str(manifest.get("status", "missing")).strip().lower()
    if status != "final":
        raise ValueError(
            f"Training split manifest {manifest_path} has status='{status}', not 'final'. "
            "A provisional split cannot be used for training."
        )


@torch.no_grad()
def validate(
    model: ASGCNReconstructor,
    loader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
    score_positions: set[int] | None = None,
) -> dict[str, Any]:
    model.eval()
    accumulator = MetricAccumulator()
    current_scene = None
    previous_sequence_index = None
    previous_sensor_size = None
    recurrent_state = None
    for index, batch in enumerate(loader):
        if max_samples is not None and index >= max_samples:
            break
        if len(batch) != 1:
            raise ValueError("Stateful validation currently requires batch_size=1")
        sample = move_sample(batch[0], device)
        scene, sequence_index, sensor_size = _sample_sequence_info(sample)
        if not _continues_sequence(
            scene,
            sequence_index,
            sensor_size,
            current_scene,
            previous_sequence_index,
            previous_sensor_size,
        ):
            recurrent_state = None
        current_scene = scene
        previous_sequence_index = sequence_index
        previous_sensor_size = sensor_size
        prediction, diagnostics = model.forward_sample(sample, recurrent_state=recurrent_state)
        recurrent_state = diagnostics["recurrent_state"]
        if recurrent_state is not None:
            recurrent_state = recurrent_state.detach()
        if score_positions is None or index in score_positions:
            target = sample["target"].unsqueeze(0)
            accumulator.update(scene, sample["sample_id"], frame_metrics(prediction, target))
    return accumulator.summary()


def train(config: dict[str, Any], resume_from: str | Path | None = None) -> Path:
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    train_config = config["train"]
    _enforce_training_split_status(config)
    run_dir = Path(config["output"]["run_dir"])
    resume_path = resume_from or train_config.get("resume")
    if resume_path is not None:
        resume_path = Path(resume_path)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        if resume_path.resolve().parent != run_dir.resolve():
            raise ValueError(
                "Exact resume must use a checkpoint inside the configured run_dir so "
                "the historical best.pt remains available."
            )
    else:
        existing_artifacts = [
            path
            for name in ("last.pt", "best.pt", "history.json", "config.json")
            if (path := run_dir / name).exists()
        ]
        if existing_artifacts:
            raise ValueError(
                f"Fresh training run_dir is not empty: {run_dir}. Use --resume with "
                "last.pt or choose a new output.run_dir; existing results are not overwritten."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    data_config = copy.deepcopy(config["dataset"])
    train_dataset = build_dataset(data_config, split="train")
    val_dataset = _validation_dataset(config)
    batch_size = int(train_config.get("batch_size", 1))
    if batch_size != 1 and config["model"].get("recurrent", True):
        raise ValueError("The recurrent experiment uses chronological batch_size=1")
    train_loader = _data_loader(
        train_dataset,
        batch_size,
        int(train_config.get("num_workers", 0)),
        device,
        shuffle=False,
        **_loader_kwargs(train_config),
    )
    val_indices = _balanced_contiguous_indices(
        val_dataset,
        train_config.get("max_val_samples"),
        seed=seed,
        require_all_groups=True,
    )
    val_sampling = _sampling_summary(val_dataset, val_indices)
    recurrent_validation = bool(config["model"].get("recurrent", True))
    validation_context_frames = train_config.get("validation_context_frames", 64)
    if validation_context_frames is not None:
        validation_context_frames = int(validation_context_frames)
        if validation_context_frames < 0:
            raise ValueError("train.validation_context_frames must be non-negative or null")
    if recurrent_validation:
        val_schedule, val_score_positions = _prefix_context_schedule(
            val_dataset,
            val_indices,
            max_context_frames=validation_context_frames,
        )
        context_policy = (
            "full_group_prefix" if validation_context_frames is None else "bounded_predecessor"
        )
    else:
        val_schedule = val_indices
        val_score_positions = set(range(len(val_indices)))
        context_policy = "none_non_recurrent"
    val_sampling.update(
        {
            "context_policy": context_policy,
            "max_context_frames_per_group": validation_context_frames
            if recurrent_validation
            else 0,
            "context_samples": len(val_schedule) - len(val_indices),
            "forward_samples": len(val_schedule),
        }
    )
    val_sampling_counts = _sampling_counts(val_sampling)
    hash_cache_path = run_dir / ".data_hash_cache.json"
    digest_cache = _load_data_hash_cache(
        hash_cache_path, bool(train_config.get("rehash_data", False))
    )
    validation_protocol = _validation_protocol(
        config, val_sampling, train_dataset, val_dataset, digest_cache
    )
    save_json(hash_cache_path, {"version": 1, "files": digest_cache})
    val_loader = _data_loader(
        Subset(val_dataset, val_schedule),
        1,
        int(train_config.get("num_workers", 0)),
        device,
        **_loader_kwargs(train_config),
    )

    resume_checkpoint: dict[str, Any] | None = None
    if resume_path is not None:
        model, resume_checkpoint = load_model_checkpoint(resume_path, device, config["model"])
    else:
        model = build_model(config["model"]).to(device)
    optimizer_mode = _optimizer_mode(train_config)
    optimizer = _build_optimizer(model, train_config)
    scheduler = _build_scheduler(optimizer, train_config)
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    training_protocol = _training_protocol(config, device)
    scaler = _make_grad_scaler(amp_enabled)
    criterion = ReconstructionLoss(train_config.get("loss_weights"))
    temporal_weight = float(train_config.get("loss_weights", {}).get("temporal", 0.0))

    if resume_checkpoint is not None:
        if resume_checkpoint.get("model_config") != config["model"]:
            raise ValueError(
                "Exact resume requires config.model to match the checkpoint model_config"
            )
        if resume_checkpoint.get("validation_protocol") != validation_protocol:
            raise ValueError(
                "Resume validation protocol differs from the checkpoint. Keep the seed, "
                "dataset transforms, split manifest, validation sampling, and SSIM protocol fixed."
            )
        _validate_training_protocol(resume_checkpoint, training_protocol)
        historical_score = _resume_best_macro_ssim(resume_checkpoint)
        historical_path = run_dir / "best.pt"
        if math.isfinite(historical_score):
            if not historical_path.is_file():
                raise ValueError(
                    f"Exact resume requires the historical best checkpoint: {historical_path}"
                )
            historical_model, historical_best = load_model_checkpoint(
                historical_path,
                torch.device("cpu"),
                config["model"],
            )
            computed_best_digest = _model_state_sha256(historical_model.state_dict())
            if historical_best.get("model_state_sha256") != computed_best_digest:
                raise ValueError("Historical best.pt tensor bytes do not match its digest")
            _validate_resume_best_pair(resume_checkpoint, historical_best)
            del historical_model
            del historical_best
        elif historical_path.exists():
            raise ValueError(
                "Resume checkpoint has no validated best score, but run_dir contains a "
                "best.pt from another or inconsistent run"
            )
    save_json(run_dir / "config.json", config)

    best_ssim = float("-inf")
    best_model_state_sha256: str | None = None
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if resume_checkpoint is not None:
        if "optimizer" not in resume_checkpoint:
            raise ValueError(
                f"Checkpoint {resume_path} has model weights but no optimizer state; "
                "it cannot be used for exact training resume"
            )
        optimizer.load_state_dict(resume_checkpoint.pop("optimizer"))
        _optimizer_to(optimizer, device)
        if "scheduler" not in resume_checkpoint:
            raise ValueError(
                f"Checkpoint {resume_path} has no scheduler state/schema and cannot "
                "provide an exact training resume"
            )
        scheduler_state = resume_checkpoint.pop("scheduler")
        if scheduler is None:
            if scheduler_state is not None:
                raise ValueError("Resume checkpoint unexpectedly contains scheduler state")
        elif not isinstance(scheduler_state, dict):
            raise ValueError("Resume checkpoint is missing MultiStepLR scheduler state")
        else:
            scheduler.load_state_dict(scheduler_state)
        if "scaler" not in resume_checkpoint:
            raise ValueError(
                f"Checkpoint {resume_path} has no GradScaler state and cannot provide "
                "an exact training resume"
            )
        scaler_state = resume_checkpoint.pop("scaler")
        if not isinstance(scaler_state, dict):
            raise ValueError("Resume checkpoint GradScaler state must be a dictionary")
        scaler.load_state_dict(scaler_state)
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_ssim = _resume_best_macro_ssim(resume_checkpoint)
        best_model_state_sha256 = resume_checkpoint.get("best_model_state_sha256")
        history = list(resume_checkpoint.get("history", []))
        if "rng_state" not in resume_checkpoint:
            raise ValueError(
                f"Checkpoint {resume_path} has no RNG state and cannot provide an exact resume"
            )
        _restore_rng_state(resume_checkpoint.pop("rng_state"))

    epochs = int(train_config.get("epochs", 40))
    raw_validate_every = train_config.get("validate_every", 1)
    validate_every = (
        None if raw_validate_every is None else max(1, int(raw_validate_every))
    )
    max_train_samples = train_config.get("max_train_samples")
    for epoch in range(start_epoch, epochs + 1):
        epoch_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
        model.train()
        _reset_cuda_peak_memory(device)
        current_scene = None
        previous_sequence_index = None
        previous_sensor_size = None
        recurrent_state = None
        previous_prediction = None
        previous_target = None
        running_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"train {epoch:03d}/{epochs:03d}")
        for step, batch in enumerate(progress):
            if max_train_samples is not None and seen >= int(max_train_samples):
                break
            if len(batch) != 1:
                raise ValueError("Stateful training currently requires batch_size=1")
            sample = move_sample(batch[0], device)
            scene, sequence_index, sensor_size = _sample_sequence_info(sample)
            if not _continues_sequence(
                scene,
                sequence_index,
                sensor_size,
                current_scene,
                previous_sequence_index,
                previous_sensor_size,
            ):
                recurrent_state = None
                previous_prediction = None
                previous_target = None
            current_scene = scene
            previous_sequence_index = sequence_index
            previous_sensor_size = sensor_size
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction, diagnostics = model.forward_sample(
                    sample, recurrent_state=recurrent_state
                )
                target = sample["target"].unsqueeze(0)
                loss, loss_parts = criterion(prediction, target)
                if temporal_weight > 0 and previous_prediction is not None:
                    temporal = F.l1_loss(
                        prediction - previous_prediction,
                        target - previous_target,
                    )
                    loss = loss + temporal_weight * temporal
                    loss_parts["temporal"] = float(temporal.detach().cpu())
                    loss_parts["total"] = float(loss.detach().cpu())
            _ensure_finite_loss(
                loss,
                loss_parts,
                epoch=epoch,
                step=step,
                sample_id=sample.get("sample_id", "unknown"),
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if optimizer_mode == "adam_gc":
                _centralize_gradients(model)
            _clip_and_validate_gradients(
                model,
                float(train_config.get("grad_clip", 1.0)),
                epoch=epoch,
                step=step,
                sample_id=sample.get("sample_id", "unknown"),
            )
            scaler.step(optimizer)
            scaler.update()

            recurrent_state = diagnostics["recurrent_state"]
            if recurrent_state is not None:
                recurrent_state = recurrent_state.detach()
            previous_prediction = prediction.detach()
            previous_target = target.detach()
            running_loss += float(loss.detach().cpu())
            seen += 1
            if step % int(train_config.get("log_every", 20)) == 0:
                progress.set_postfix(loss=f"{running_loss / max(seen, 1):.4f}", **loss_parts)

        should_validate = epoch == epochs or (
            validate_every is not None and epoch % validate_every == 0
        )
        val_metrics = (
            validate(
                model,
                val_loader,
                device,
                max_samples=None,
                score_positions=val_score_positions,
            )
            if should_validate
            else {}
        )
        train_mean_loss = running_loss / max(seen, 1)
        if not math.isfinite(train_mean_loss):
            raise FloatingPointError(
                f"Non-finite mean training loss at epoch={epoch}: {train_mean_loss}"
            )
        validation_ssim = _macro_ssim(val_metrics)
        if should_validate and not math.isfinite(validation_ssim):
            raise FloatingPointError(
                f"Non-finite validation macro SSIM at epoch={epoch}: {validation_ssim}"
            )
        if scheduler is not None:
            scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_mean_loss,
            "val": val_metrics,
            "val_sampling": val_sampling_counts,
            "learning_rate": (
                epoch_learning_rates[0] if len(epoch_learning_rates) == 1 else epoch_learning_rates
            ),
            "gpu_memory": _cuda_peak_memory(device),
        }
        history.append(record)
        save_json(run_dir / "history.json", history)
        model_state = model.state_dict()
        model_state_sha256 = _model_state_sha256(model_state)
        checkpoint = {
            "checkpoint_type": "training",
            "epoch": epoch,
            "model": model_state,
            "model_state_sha256": model_state_sha256,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "model_config": (
                resume_checkpoint.get("model_config", config["model"])
                if resume_checkpoint is not None
                else config["model"]
            ),
            "config": config,
            "val": val_metrics,
            "val_sampling": val_sampling_counts,
            "best_ssim": best_ssim,
            "best_model_state_sha256": best_model_state_sha256,
            "best_metric": "macro_ssim",
            "checkpoint_selection": (
                "single_final_epoch"
                if validate_every is None
                else "best_validation_macro_ssim"
            ),
            "paper_core_version": PAPER_CORE_VERSION,
            "validation_protocol": validation_protocol,
            "training_protocol": training_protocol,
            "history": history,
            "rng_state": _capture_rng_state(),
        }
        if validation_ssim > best_ssim:
            best_ssim = validation_ssim
            best_model_state_sha256 = model_state_sha256
            checkpoint["best_ssim"] = best_ssim
            checkpoint["best_model_state_sha256"] = best_model_state_sha256
            best_checkpoint = {
                "checkpoint_type": "ann_inference",
                "epoch": checkpoint["epoch"],
                "model": checkpoint["model"],
                "model_config": checkpoint["model_config"],
                "val": checkpoint["val"],
                "val_sampling": checkpoint["val_sampling"],
                "best_ssim": checkpoint["best_ssim"],
                "best_metric": checkpoint["best_metric"],
                "checkpoint_selection": checkpoint["checkpoint_selection"],
                "model_state_sha256": best_model_state_sha256,
                "paper_core_version": checkpoint["paper_core_version"],
                "validation_protocol": checkpoint["validation_protocol"],
                "training_protocol": checkpoint["training_protocol"],
            }
            atomic_torch_save(best_checkpoint, run_dir / "best.pt")
        atomic_torch_save(checkpoint, run_dir / "last.pt")
        print(record)
    best_path = run_dir / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(
            "Training completed without a best.pt. Check that macro SSIM is finite and "
            "validation ran successfully."
        )
    return best_path


def _maybe_lpips(enabled: bool, device: torch.device):
    if not enabled:
        return None
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS requested. Install with: pip install -e '.[eval]'") from exc
    return lpips.LPIPS(net="alex").to(device).eval()


@torch.no_grad()
def evaluate(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    inference_mode: str = "ann",
    simulation_steps: int = 16,
    snn_dynamics: str | None = None,
) -> dict[str, Any]:
    _validate_snn_request(inference_mode, simulation_steps)
    set_seed(int(config.get("seed", 2026)))
    device = resolve_device(config.get("device", "auto"))
    dataset = build_dataset(config["dataset"], split="eval")
    eval_config = config.get("eval", {})
    eval_batch_size = int(eval_config.get("batch_size", 1))
    if eval_batch_size != 1:
        raise ValueError("Stateful evaluation requires eval.batch_size=1")
    loader = _data_loader(
        dataset,
        eval_batch_size,
        int(eval_config.get("num_workers", 0)),
        device,
        **_loader_kwargs(eval_config),
    )
    model, checkpoint = load_model_checkpoint(checkpoint_path, device, config["model"])
    _validate_snn_request(inference_mode, simulation_steps, checkpoint, checkpoint_path)
    _set_inference_snn_dynamics(model, inference_mode, snn_dynamics)
    model.eval()
    lpips_model = _maybe_lpips(bool(eval_config.get("lpips", False)), device)
    _reset_cuda_peak_memory(device)
    accumulator = MetricAccumulator()
    frame_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    realtime_factors: list[float] = []
    current_scene = None
    previous_sequence_index = None
    previous_sensor_size = None
    recurrent_state = None
    previous_prediction = None
    previous_target = None
    output_base = Path(eval_config.get("output_dir", "runs/evaluation"))
    run_label = _inference_run_label(
        inference_mode,
        simulation_steps,
        model.snn_dynamics,
    )
    output_dir = output_base / run_label
    protected_outputs = (
        output_dir / "metrics.json",
        output_dir / "frames.csv",
        output_dir / "predictions",
    )
    if any(path.exists() for path in protected_outputs):
        raise FileExistsError(
            f"Evaluation output already exists for {run_label}: {output_dir}. "
            "Move/remove that run or choose a new eval.output_dir; results are never "
            "silently overwritten."
        )
    save_limit = int(eval_config.get("save_predictions", 0))
    max_samples = eval_config.get("max_samples")
    saved = 0
    prediction_stems: set[str] = set()
    for index, batch in enumerate(tqdm(loader, desc=f"evaluate-{inference_mode}")):
        if max_samples is not None and index >= int(max_samples):
            break
        sample = move_sample(batch[0], device)
        scene, sequence_index, sensor_size = _sample_sequence_info(sample)
        if not _continues_sequence(
            scene,
            sequence_index,
            sensor_size,
            current_scene,
            previous_sequence_index,
            previous_sensor_size,
        ):
            recurrent_state = None
            previous_prediction = None
            previous_target = None
        current_scene = scene
        previous_sequence_index = sequence_index
        previous_sensor_size = sensor_size
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        prediction, diagnostics = model.forward_sample(
            sample,
            inference_mode=inference_mode,
            simulation_steps=simulation_steps,
            recurrent_state=recurrent_state,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - start) * 1000.0
        recurrent_state = diagnostics["recurrent_state"]
        if recurrent_state is not None:
            recurrent_state = recurrent_state.detach()
        target = sample["target"].unsqueeze(0)
        temporal_l1 = None
        metrics = frame_metrics(prediction, target, lpips_model)
        if previous_prediction is not None and previous_target is not None:
            temporal_l1 = float(
                temporal_consistency_error(
                    prediction,
                    previous_prediction,
                    target,
                    previous_target,
                ).cpu()
            )
            metrics["temporal_l1"] = temporal_l1
        previous_prediction = prediction.detach()
        previous_target = target.detach()
        accumulator.update(scene, sample["sample_id"], metrics)
        dt_us = sample["metadata"].get("dt_us")
        rtf = latency_ms / (float(dt_us) / 1000.0) if dt_us else None
        if rtf is not None:
            realtime_factors.append(rtf)
        row = {
            "scene": scene,
            "sample_id": sample["sample_id"],
            **metrics,
            "latency_ms": latency_ms,
            "rtf": rtf,
            "temporal_l1": temporal_l1,
            "raw_events": int(sample["metadata"].get("raw_event_count", sample["events"].shape[0])),
            "cropped_events": int(
                sample["metadata"].get("cropped_event_count", sample["events"].shape[0])
            ),
            "retained_events": int(sample["events"].shape[0]),
            "events": int(sample["events"].shape[0]),
            "dataset_sampling_ratio": diagnostics["dataset_sampling_ratio"],
            "model_sampling_factor": diagnostics["event_sampling_factor"],
            "effective_sampling_ratio": diagnostics["effective_sampling_ratio"],
            "nodes": diagnostics["nodes"],
            "edges": diagnostics["edges"],
            "isolated_nodes": int(diagnostics["isolated_nodes"]),
            "isolate_ratio": float(diagnostics["isolate_ratio"]),
            "max_degree": int(diagnostics["max_degree"]),
        }
        frame_rows.append(row)
        latencies.append(latency_ms)
        if saved < save_limit:
            safe_name = _prediction_artifact_stem(sample["sample_id"], index)
            if safe_name in prediction_stems:
                raise RuntimeError(f"Duplicate prediction artifact stem: {safe_name}")
            prediction_stems.add(safe_name)
            save_image(output_dir / "predictions" / f"{safe_name}_pred.png", prediction)
            save_image(output_dir / "predictions" / f"{safe_name}_gt.png", target)
            saved += 1

    quality = accumulator.summary()
    latency = _latency_summary(latencies)
    latency["deadline_miss_ratio"] = (
        sum(value > 1.0 for value in realtime_factors) / len(realtime_factors)
        if realtime_factors
        else None
    )
    latency["rtf_p95"] = percentile(realtime_factors, 0.95) if realtime_factors else None
    result = {
        "dataset": config["dataset"]["type"],
        "dataset_coverage": _dataset_coverage_summary(dataset, config["dataset"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "output_dir": str(output_dir),
        "inference_mode": inference_mode,
        "simulation_steps": simulation_steps if inference_mode == "snn" else None,
        "snn_dynamics": model.snn_dynamics if inference_mode == "snn" else None,
        "graph_topology": {
            "isolate_ratio": (
                sum(row["isolated_nodes"] for row in frame_rows)
                / sum(row["nodes"] for row in frame_rows)
                if sum(row["nodes"] for row in frame_rows) > 0
                else None
            ),
            "max_degree": max((row["max_degree"] for row in frame_rows), default=0),
        },
        "quality": quality,
        "latency": latency,
        "gpu_memory": _cuda_peak_memory(device),
    }
    save_json(output_dir / "metrics.json", result)
    write_frame_csv(output_dir / "frames.csv", frame_rows)
    return result


def _latency_summary(latencies: list[float]) -> dict[str, float | int | None]:
    if not latencies:
        return {"frames": 0}
    mean = statistics.fmean(latencies)
    return {
        "frames": len(latencies),
        "mean_ms": mean,
        "p50_ms": percentile(latencies, 0.50),
        "p90_ms": percentile(latencies, 0.90),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "max_ms": max(latencies),
        "fps": 1000.0 / mean,
    }


def _sample_event_counts(sample: dict[str, Any]) -> tuple[int, int]:
    """Return raw/source and retained counts, tolerating custom dataset metadata."""
    retained = int(sample["events"].shape[0])
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        return retained, retained
    value = metadata.get("raw_event_count")
    try:
        raw = int(value)
    except (TypeError, ValueError, OverflowError):
        raw = retained
    if isinstance(value, bool) or raw < retained:
        raw = retained
    return raw, retained


def _dataset_coverage_summary(dataset, data_config: dict[str, Any]) -> dict[str, Any]:
    dataset_type = data_config["type"]
    if dataset_type == "eventhdr":
        root = Path(dataset.root)
        files = sorted(path.relative_to(root).as_posix() for path in dataset.files)
        mapping = getattr(dataset, "file_to_scene", {})
        declared_semantics = getattr(dataset, "group_semantics", None)
        if declared_semantics == "h5_sequence_file_not_physical_scene":
            grouping = "source_h5_sequence_file"
        elif declared_semantics == "physical_scene":
            grouping = "physical_scene"
        else:
            grouping = (
                "physical_scene"
                if any(mapping.get(file_key) != file_key for file_key in files)
                else "source_h5_file"
            )
    elif dataset_type == "eventaid_r_zip":
        files = sorted(path.name for path in dataset.zip_paths)
        grouping = "eventaid_scene_zip"
    else:
        files = []
        grouping = "unknown"
    expected = data_config.get("expected_file_count")
    return {
        "file_count": len(files),
        "expected_file_count": int(expected) if expected is not None else None,
        "complete": expected is None or len(files) == int(expected),
        "files": files,
        "quality_grouping": grouping,
        "target_offset": (
            int(data_config.get("target_offset", 1)) if dataset_type == "eventaid_r_zip" else None
        ),
    }


@torch.no_grad()
def benchmark(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    warmup: int = 10,
    steps: int = 100,
    inference_mode: str = "ann",
    simulation_steps: int = 16,
    snn_dynamics: str | None = None,
) -> dict[str, Any]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    _validate_snn_request(inference_mode, simulation_steps)
    device = resolve_device(config.get("device", "auto"))
    dataset = build_dataset(config["dataset"], split="eval")
    model, checkpoint = load_model_checkpoint(checkpoint_path, device, config["model"])
    _validate_snn_request(inference_mode, simulation_steps, checkpoint, checkpoint_path)
    _set_inference_snn_dynamics(model, inference_mode, snn_dynamics)
    model.eval()
    cuda_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    cuda_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    latencies: list[float] = []
    raw_event_counts: list[int] = []
    retained_event_counts: list[int] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    isolated_node_counts: list[int] = []
    max_degrees: list[int] = []
    layer_spike_totals: list[float] = []
    layer_neuron_step_totals: list[int] = []
    realtime_factors: list[float] = []
    recurrent_state = None
    current_scene = None
    previous_sequence_index = None
    previous_sensor_size = None
    measured_state_resets = 0
    seed = int(config.get("seed", 2026))
    recurrent = model.decoder.recurrent is not None
    warmup_indices = _representative_schedule(dataset, warmup, seed, contiguous=False)
    measured_indices = _representative_schedule(dataset, steps, seed + 1, contiguous=recurrent)
    measured_schedule: list[tuple[bool, int]] = []
    context_frames = 0
    benchmark_context_frames = config.get("eval", {}).get("recurrent_context_frames", 32)
    if benchmark_context_frames is not None:
        benchmark_context_frames = int(benchmark_context_frames)
        if benchmark_context_frames < 0:
            raise ValueError("eval.recurrent_context_frames must be non-negative or null")
    if recurrent:
        # ``_representative_schedule`` cycles only when steps exceed the dataset.
        # Build each cycle separately so the prefix helper always receives unique indices.
        for offset in range(0, len(measured_indices), len(dataset)):
            chunk = measured_indices[offset : offset + len(dataset)]
            context_indices, score_positions = _prefix_context_schedule(
                dataset,
                chunk,
                max_context_frames=benchmark_context_frames,
            )
            context_frames += len(context_indices) - len(chunk)
            measured_schedule.extend(
                (position in score_positions, index)
                for position, index in enumerate(context_indices)
            )
    else:
        measured_schedule = [(True, index) for index in measured_indices]
    schedule = [(False, index) for index in warmup_indices] + measured_schedule
    for iteration, (measured, sample_index) in enumerate(schedule):
        if iteration == len(warmup_indices):
            recurrent_state = None
            current_scene = None
            previous_sequence_index = None
            previous_sensor_size = None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
        raw = dataset[sample_index]  # I/O intentionally outside the timer.
        sample = move_sample(raw, device)
        scene, sequence_index, sensor_size = _sample_sequence_info(sample)
        continuation = _continues_sequence(
            scene,
            sequence_index,
            sensor_size,
            current_scene,
            previous_sequence_index,
            previous_sensor_size,
        )
        if not continuation:
            recurrent_state = None
            if measured:
                measured_state_resets += 1
        current_scene = scene
        previous_sequence_index = sequence_index
        previous_sensor_size = sensor_size
        if measured:
            if cuda_start is not None:
                cuda_start.record()
            else:
                start = time.perf_counter()
        _, diagnostics = model.forward_sample(
            sample,
            inference_mode=inference_mode,
            simulation_steps=simulation_steps,
            recurrent_state=recurrent_state,
        )
        elapsed_ms = None
        if measured:
            if cuda_end is not None:
                cuda_end.record()
                cuda_end.synchronize()
                elapsed_ms = float(cuda_start.elapsed_time(cuda_end))
            else:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
        recurrent_state = diagnostics["recurrent_state"]
        if recurrent_state is not None:
            recurrent_state = recurrent_state.detach()
        if measured:
            assert elapsed_ms is not None
            latencies.append(elapsed_ms)
            raw_event_count, retained_event_count = _sample_event_counts(sample)
            raw_event_counts.append(raw_event_count)
            retained_event_counts.append(retained_event_count)
            node_counts.append(int(diagnostics["nodes"]))
            edge_counts.append(int(diagnostics["edges"]))
            isolated_node_counts.append(int(diagnostics["isolated_nodes"]))
            max_degrees.append(int(diagnostics["max_degree"]))
            spike_counts = diagnostics["spike_counts"]
            neuron_steps = diagnostics["firing_rate_denominators"]
            if spike_counts:
                if not layer_spike_totals:
                    layer_spike_totals = [0.0] * len(spike_counts)
                    layer_neuron_step_totals = [0] * len(neuron_steps)
                if len(spike_counts) != len(layer_spike_totals):
                    raise RuntimeError("SNN firing-stat layer count changed during benchmark")
                for layer_index, (spikes, denominator) in enumerate(
                    zip(spike_counts, neuron_steps, strict=True)
                ):
                    layer_spike_totals[layer_index] += (
                        float(spikes.detach().cpu()) if torch.is_tensor(spikes) else float(spikes)
                    )
                    layer_neuron_step_totals[layer_index] += int(denominator)
            metadata = sample.get("metadata", {})
            dt_us = metadata.get("dt_us") if isinstance(metadata, dict) else None
            if dt_us:
                realtime_factors.append(elapsed_ms / (float(dt_us) / 1000.0))
    elapsed_seconds = sum(latencies) / 1000.0
    raw_events_per_second = sum(raw_event_counts) / elapsed_seconds
    retained_events_per_second = sum(retained_event_counts) / elapsed_seconds
    graph_nodes_per_second = sum(node_counts) / elapsed_seconds
    total_raw_events = sum(raw_event_counts)
    total_neuron_steps = sum(layer_neuron_step_totals)
    layer_firing_rates = [
        spikes / neuron_steps if neuron_steps > 0 else None
        for spikes, neuron_steps in zip(
            layer_spike_totals,
            layer_neuron_step_totals,
            strict=True,
        )
    ]
    result: dict[str, Any] = {
        **_latency_summary(latencies),
        "raw_events_per_second": raw_events_per_second,
        "retained_events_per_second": retained_events_per_second,
        "graph_nodes_per_second": graph_nodes_per_second,
        # Deprecated compatibility alias; new consumers should use the retained rate.
        "events_per_second": retained_events_per_second,
        "mean_raw_events": statistics.fmean(raw_event_counts),
        "mean_retained_events": statistics.fmean(retained_event_counts),
        "retention_ratio": (
            sum(retained_event_counts) / total_raw_events if total_raw_events > 0 else None
        ),
        "mean_nodes": statistics.fmean(node_counts),
        "mean_edges": statistics.fmean(edge_counts),
        "mean_isolated_nodes": statistics.fmean(isolated_node_counts),
        "isolate_ratio": (
            sum(isolated_node_counts) / sum(node_counts) if sum(node_counts) > 0 else None
        ),
        "max_degree": max(max_degrees, default=0),
        "layer_firing_rates": layer_firing_rates or None,
        "mean_firing_rate": (
            sum(layer_spike_totals) / total_neuron_steps if total_neuron_steps > 0 else None
        ),
        "deadline_miss_ratio": (
            sum(value > 1.0 for value in realtime_factors) / len(realtime_factors)
            if realtime_factors
            else None
        ),
        "rtf_p95": percentile(realtime_factors, 0.95) if realtime_factors else None,
        "inference_mode": inference_mode,
        "simulation_steps": simulation_steps if inference_mode == "snn" else None,
        "snn_dynamics": model.snn_dynamics if inference_mode == "snn" else None,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
        ),
        "peak_gpu_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else None
        ),
        "gpu_memory": _cuda_peak_memory(device),
        "timer": "cuda_event" if device.type == "cuda" else "perf_counter",
        "io_excluded": True,
        "dataset_coverage": _dataset_coverage_summary(dataset, config["dataset"]),
        "sampling": _sampling_summary(dataset, measured_indices),
        "warmup_frames": len(warmup_indices),
        "recurrent_context_policy": (
            "full_group_prefix"
            if recurrent and benchmark_context_frames is None
            else "bounded_predecessor"
            if recurrent
            else None
        ),
        "max_recurrent_context_frames_per_group": benchmark_context_frames if recurrent else 0,
        "recurrent_context_frames": context_frames,
        "state_resets": measured_state_resets,
        "state_reset_ratio": measured_state_resets / len(measured_indices),
    }
    benchmark_base = Path(config.get("eval", {}).get("output_dir", "runs/evaluation"))
    benchmark_dir = benchmark_base / _inference_run_label(
        inference_mode,
        simulation_steps,
        model.snn_dynamics,
    )
    benchmark_path = benchmark_dir / "benchmark.json"
    if benchmark_path.exists():
        raise FileExistsError(
            f"Benchmark output already exists: {benchmark_path}. Move/remove the prior "
            "artifact or choose a new eval.output_dir."
        )
    result["output_path"] = str(benchmark_path)
    save_json(benchmark_path, result)
    return result


@torch.no_grad()
def calibrate(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path,
    samples: int | None = None,
    overwrite: bool = False,
) -> Path:
    if samples is not None and int(samples) < 1:
        raise ValueError("calibration samples must be at least 1")
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if checkpoint_path.resolve() == output_path.resolve():
        raise ValueError("ANN input and calibrated SNN output must be different files")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Calibrated checkpoint already exists: {output_path}. Move it or choose a new "
            "output path, or explicitly request overwrite."
        )
    _enforce_training_split_status(config)
    device = resolve_device(config.get("device", "auto"))
    data_config = copy.deepcopy(config["dataset"])
    # Calibration is restricted to EventHDR train, never EventAid-R.
    if data_config["type"] != "eventhdr":
        raise ValueError("SNN calibration must use EventHDR training data")
    dataset = build_dataset(data_config, split="calibration")
    model, checkpoint = load_model_checkpoint(checkpoint_path, device, config["model"])
    model.eval()
    model.fold_batch_norm()
    model.reset_activation_maxima()
    calibration_limit = len(dataset) if samples is None else min(int(samples), len(dataset))
    calibration_indices = _balanced_sample_indices(
        dataset, calibration_limit, seed=int(config.get("seed", 2026))
    )
    try:
        calibration_sampling = _sampling_summary(dataset, calibration_indices)
        for index in tqdm(calibration_indices, desc="calibrate-SNN"):
            sample = move_sample(dataset[index], device)
            model.calibrate_sample(sample, momentum=-1.0)
        calibration_summary = model.calibration_summary()
        model.apply_parameter_normalization()
        model_state = model.state_dict()
        inference_checkpoint = {
            "checkpoint_type": "snn_inference",
            "model": model_state,
            "model_state_sha256": _model_state_sha256(model_state),
            "model_config": checkpoint.get("model_config", config["model"]),
            "epoch": checkpoint.get("epoch"),
            "source_checkpoint": str(checkpoint_path),
            "batch_norm_folded": True,
            "snn_calibrated": True,
            "paper_core_version": PAPER_CORE_VERSION,
            "parameter_normalized": True,
            "snn_calibration_samples": len(calibration_indices),
            "snn_calibration_valid_samples": calibration_summary["minimum_valid_samples"],
            "snn_calibration_summary": calibration_summary,
            "snn_calibration_sampling": calibration_sampling,
        }
    finally:
        if hasattr(dataset, "close"):
            dataset.close()
    atomic_torch_save(inference_checkpoint, output_path)
    return output_path
