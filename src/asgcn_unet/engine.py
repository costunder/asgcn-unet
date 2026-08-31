from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .batching import SequenceBatchSampler
from .data import build_dataset, collate_samples, load_eventhdr_split_manifest
from .graph import PAPER_CORE_VERSION, PaperSplineConv
from .losses import ReconstructionLoss
from .metrics import (
    MetricAccumulator,
    frame_metrics,
    percentile,
    temporal_consistency_error,
)
from .model import ASGCNUNet
from .timing import StageTimer
from .training import TrainingState, batching_contract, forward_training_loss
from .utils import (
    atomic_torch_save,
    load_json,
    move_inference_sample,
    move_sample,
    resolve_device,
    save_image,
    save_json,
    set_seed,
    validate_experiment_config,
    write_frame_csv,
)

_AMP_MAX_RETRIES = 16


def _amp_retry_policy(enabled: bool) -> dict[str, Any] | None:
    if not enabled:
        return None
    return {
        "name": "same_sample_backoff_v1",
        "max_retries": _AMP_MAX_RETRIES,
        "scale_backoff": "grad_scaler_backoff_factor",
        "restore_model_buffers": True,
        "restore_rng": True,
        "advance_recurrent_state_on_success_only": True,
        "skip_samples": False,
        "nonfinite_forward_loss": "raise",
        "persistent_overflow": "raise",
    }


def build_model(config: dict[str, Any]) -> ASGCNUNet:
    return ASGCNUNet(**config)


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


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    """Hash a strict, order-independent JSON representation of a public contract."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hashed_contract(value: Any) -> dict[str, Any]:
    contract = copy.deepcopy(value)
    return {"contract": contract, "sha256": _canonical_sha256(contract)}


def _calibration_commitment_sha256(
    protocol: Any,
    selected_samples: Any,
    sampling: Any,
    summary_core: Any,
) -> str:
    return _canonical_sha256(
        {
            "version": 1,
            "calibration_protocol": protocol,
            "selected_samples": selected_samples,
            "sampling": sampling,
            "summary_core": summary_core,
        }
    )


def _calibration_summary_commitment_core(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise TypeError("calibration summary must be a dictionary")
    valid = summary.get("valid_samples_per_layer")
    dead = summary.get("dead_channels_per_layer")
    minimum = summary.get("minimum_valid_samples")
    if (
        not isinstance(valid, list)
        or not valid
        or not isinstance(dead, list)
        or len(dead) != len(valid)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in [*valid, *dead]
        )
        or not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or minimum != min(valid)
    ):
        raise ValueError("calibration summary core is invalid")
    return {
        "valid_samples_per_layer": list(valid),
        "minimum_valid_samples": minimum,
        "dead_channels_per_layer": list(dead),
    }


def _artifact_path_label(path: str | Path) -> str:
    """Return a public, portable label without leaking a host absolute path."""
    value = Path(path)
    project_root = Path(__file__).resolve().parents[2]
    try:
        return value.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return f"$EXTERNAL/{value.name}"


def _public_config(config: Any) -> Any:
    """Redact resolved host paths before a config becomes a shareable artifact."""

    private_identity_keys = {"host", "hostname", "host_name", "user", "username"}

    def redact(value: Any, key: str | None = None) -> Any:
        if key is not None and key.strip().lower() in private_identity_keys:
            return "$REDACTED"
        if isinstance(value, dict):
            return {item_key: redact(item, item_key) for item_key, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, tuple):
            return [redact(item) for item in value]
        if isinstance(value, str) and Path(value).is_absolute():
            return _artifact_path_label(value)
        return value

    return redact(copy.deepcopy(config))


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
    model: ASGCNUNet,
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
        attempted_samples = int(model.calibration_attempts.item())
        selected_samples = int(metadata.get("snn_calibration_samples", 0) or 0)
        if attempted_samples < 1 or selected_samples != attempted_samples:
            raise ValueError(
                f"Checkpoint {checkpoint_path} selected calibration count disagrees "
                "with persistent attempted calibration state"
            )
        if not bool(model.calibration_commitment_sealed.item()):
            raise ValueError(
                f"Checkpoint {checkpoint_path} has no persistent calibration commitment"
            )
        summary = metadata.get("snn_calibration_summary")
        if not isinstance(summary, dict):
            raise ValueError(f"Checkpoint {checkpoint_path} is missing calibration summary")
        try:
            summary_core = _calibration_summary_commitment_core(summary)
            state_summary_core = _calibration_summary_commitment_core(
                model.encoder.calibration_summary()
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has an invalid calibration summary"
            ) from error
        if summary_core != state_summary_core:
            raise ValueError(
                f"Checkpoint {checkpoint_path} calibration summary differs from layer state"
            )
        state_commitment = bytes(model.calibration_commitment_digest.tolist()).hex()
        try:
            metadata_commitment = _calibration_commitment_sha256(
                metadata.get("calibration_protocol"),
                selected_samples,
                metadata.get("snn_calibration_sampling"),
                summary_core,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has an invalid calibration commitment"
            ) from error
        if state_commitment != metadata_commitment:
            raise ValueError(
                f"Checkpoint {checkpoint_path} calibration metadata disagrees with its "
                "persistent tensor commitment"
            )
        sample_counts = [int(value) for value in model.encoder.calibration_samples_seen.tolist()]
        if not sample_counts or min(sample_counts) < 1:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has graph layers without valid calibration"
            )
        for index, layer in enumerate(model.encoder.layers):
            raw_max = layer.calibration_activation_max
            normalization_scale = layer.normalization_scale
            dead_mask = layer.dead_channel_mask
            if not bool(torch.isfinite(raw_max).all()) or bool(
                (raw_max < 0).any()
            ):
                raise ValueError(
                    f"Checkpoint {checkpoint_path} layer {index} has an invalid raw "
                    "calibration maximum"
                )
            expected_dead_mask = raw_max <= 0
            if not torch.equal(dead_mask, expected_dead_mask):
                raise ValueError(
                    f"Checkpoint {checkpoint_path} layer {index} dead-channel mask "
                    "differs from its raw calibration maximum"
                )
            expected_scale = torch.where(
                expected_dead_mask,
                torch.ones_like(raw_max),
                raw_max,
            ).clamp_min(1e-6)
            if (
                not bool(torch.isfinite(normalization_scale).all())
                or not bool((normalization_scale > 0).all())
                or not torch.equal(normalization_scale, expected_scale)
            ):
                raise ValueError(
                    f"Checkpoint {checkpoint_path} layer {index} effective normalization "
                    "scale differs from its raw calibration maximum"
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
        if selected_samples < minimum:
            raise ValueError(
                f"Checkpoint {checkpoint_path} selected calibration count is inconsistent"
            )
        if (
            summary.get("attempted_samples") != attempted_samples
            or summary.get("calibration_commitment_sha256") != state_commitment
            or summary.get("commitment_sealed") is not True
        ):
            raise ValueError(
                f"Checkpoint {checkpoint_path} calibration commitment summary is inconsistent"
            )
    if checkpoint_type in {"ann_inference", "training"} and (state_bn_folded or state_normalized):
        raise ValueError(
            f"Checkpoint {checkpoint_path} is labeled {checkpoint_type} but contains "
            "converted SNN graph-layer state"
        )
    if checkpoint_type in {"ann_inference", "training"} and (
        int(model.calibration_attempts.item()) != 0
        or bool(model.calibration_commitment_sealed.item())
        or bool(model.calibration_commitment_digest.any())
    ):
        raise ValueError(
            f"Checkpoint {checkpoint_path} is labeled {checkpoint_type} but contains "
            "persistent calibration commitment state"
        )


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    fallback_model_config: dict[str, Any],
) -> tuple[ASGCNUNet, dict[str, Any]]:
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
        "sequence_id",
        "part_index",
        "sequence_index",
        "frame_id",
        "image_key",
        "event_name",
        "target_name",
        "start_idx",
        "end_idx",
        "timestamp",
        "t0_us",
        "t1_us",
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
            key = path.name
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
        # Cache lookup remains host-local without publishing an account/mount path.
        cache_key = hashlib.sha256(str(raw_path).encode("utf-8")).hexdigest()
        signature = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
        cached = cache.get(cache_key)
        cache_valid = (
            isinstance(cached, dict)
            and all(cached.get(key) == value for key, value in signature.items())
            and isinstance(cached.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", cached["sha256"]) is not None
        )
        if not cache_valid:
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
    return {
        key: value
        for key, value in payload["files"].items()
        if (
            isinstance(key, str)
            and re.fullmatch(r"[0-9a-f]{64}", key)
            and isinstance(value, dict)
        )
    }


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


def _dataset_index_contract(dataset) -> dict[str, Any]:
    identities = [_dataset_sample_identity(dataset, index) for index in range(len(dataset))]
    counts = Counter(identity["group"] for identity in identities)
    return {
        "version": 1,
        "samples": len(identities),
        "groups": len(counts),
        "per_group": dict(sorted(counts.items())),
        "sample_identities_sha256": _canonical_sha256(identities),
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
    """Identify recurrent continuity, not the scene used for quality aggregation."""
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    sequence_id = str(metadata.get("sequence_id") or metadata.get("scene", "unknown"))
    raw_index = metadata.get("sequence_index")
    try:
        sequence_index = int(raw_index) if raw_index is not None else None
    except (TypeError, ValueError):
        sequence_index = None
    sensor_size = tuple(int(value) for value in sample["sensor_size"])
    return sequence_id, sequence_index, sensor_size


def _sample_metric_scene(sample: dict[str, Any]) -> str:
    """Keep official scene-level macro metrics even when a scene has several parts."""
    metadata = sample.get("metadata", {})
    return str(metadata.get("scene", "unknown")) if isinstance(metadata, dict) else "unknown"


def _continues_sequence(
    sequence_id: str,
    sequence_index: int | None,
    sensor_size: tuple[int, int],
    previous_sequence_id: str | None,
    previous_sequence_index: int | None,
    previous_sensor_size: tuple[int, int] | None,
) -> bool:
    if sequence_id != previous_sequence_id or sensor_size != previous_sensor_size:
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
    if best_checkpoint.get("preflight_gate") != resume_checkpoint.get("preflight_gate"):
        raise ValueError("Historical best.pt has a different preflight gate")
    if best_checkpoint.get("training_config") != resume_checkpoint.get("config"):
        raise ValueError("Historical best.pt has a different public training config")
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
    if inference_mode not in {"ann", "snn"}:
        raise ValueError("inference_mode must be 'ann' or 'snn'")
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


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _valid_source_contract(value: Any) -> bool:
    if not isinstance(value, dict) or not _is_sha256(value.get("source_tree_sha256")):
        return False
    commit = value.get("git_commit")
    if commit is not None and (
        not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None
    ):
        return False
    dirty = value.get("git_source_dirty")
    return dirty is None or isinstance(dirty, bool)


def _valid_training_protocol_contract(value: Any) -> bool:
    required_fields = {
        "version",
        "seed",
        "optimizer",
        "scheduler",
        "loss_weights",
        "gradient_clipping",
        "data_order",
        "mixed_precision",
        "validate_every",
        "checkpoint_selection",
        "terminal_validation",
        "recurrent_state_detached_each_sample",
        "runtime",
        "source",
    }
    if isinstance(value, dict) and value.get("version") == 6:
        required_fields.add("batching")
    if not isinstance(value, dict) or set(value) != required_fields:
        return False
    if (
        value.get("version") not in {5, 6}
        or not isinstance(value.get("seed"), int)
        or isinstance(value.get("seed"), bool)
        or value.get("recurrent_state_detached_each_sample") is not True
        or not _valid_source_contract(value.get("source"))
    ):
        return False
    if not isinstance(value.get("data_order"), dict):
        return False
    batch_size = value["data_order"].get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        return False
    if value["version"] != (5 if batch_size == 1 else 6):
        return False
    if batch_size > 1 and value.get("batching") != batching_contract(batch_size):
        return False
    if any(
        not isinstance(value.get(field), dict)
        for field in (
            "optimizer",
            "loss_weights",
            "gradient_clipping",
            "data_order",
            "mixed_precision",
            "runtime",
        )
    ):
        return False
    if value.get("scheduler") is not None and not isinstance(
        value.get("scheduler"), dict
    ):
        return False
    mixed_precision = value["mixed_precision"]
    if (
        set(mixed_precision)
        != {"requested", "effective", "autocast_dtype", "gradient_scaler", "overflow_policy"}
        or not isinstance(mixed_precision.get("effective"), bool)
        or mixed_precision.get("overflow_policy")
        != _amp_retry_policy(mixed_precision["effective"])
    ):
        return False
    validate_every = value.get("validate_every")
    if validate_every is not None and (
        not isinstance(validate_every, int)
        or isinstance(validate_every, bool)
        or validate_every < 1
    ):
        return False
    expected_selection = (
        "single_final_epoch"
        if validate_every is None
        else "best_validation_macro_ssim"
    )
    if value.get("checkpoint_selection") != expected_selection:
        return False
    terminal = value.get("terminal_validation")
    if validate_every is not None:
        return terminal is None
    return (
        isinstance(terminal, dict)
        and set(terminal) == {"mode", "planned_epoch"}
        and terminal.get("mode") == "single_final_epoch"
        and isinstance(terminal.get("planned_epoch"), int)
        and not isinstance(terminal.get("planned_epoch"), bool)
        and terminal["planned_epoch"] >= 1
    )


def _valid_dataset_content_contract(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("algorithm") == "sha256-full-files-v1"
        and isinstance(value.get("files"), int)
        and not isinstance(value.get("files"), bool)
        and value["files"] >= 1
        and isinstance(value.get("bytes"), int)
        and not isinstance(value.get("bytes"), bool)
        and value["bytes"] >= 0
        and _is_sha256(value.get("sha256"))
    )


def _valid_dataset_transform_contract(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("type"), str)
        and bool(value["type"].strip())
    )


def _valid_split_manifest_contract(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if (
        value.get("status") != "final"
        or value.get("split_schema") != "official_separate_roots_v1"
        or value.get("group_semantics") != "h5_sequence_file_not_physical_scene"
    ):
        return False
    train_files = value.get("train_files")
    val_files = value.get("val_files")
    file_to_group = value.get("file_to_group")
    if (
        not isinstance(train_files, list)
        or not train_files
        or not isinstance(val_files, list)
        or not val_files
        or not isinstance(file_to_group, dict)
        or set(file_to_group) != {"train", "val"}
    ):
        return False
    for split, files, prefix in (
        ("train", train_files, "official-train-h5::"),
        ("val", val_files, "official-eval-h5::"),
    ):
        if (
            any(not isinstance(name, str) or not name.strip() for name in files)
            or len(files) != len(set(files))
        ):
            return False
        mapping = file_to_group.get(split)
        if not isinstance(mapping, dict) or set(mapping) != set(files):
            return False
        if any(mapping.get(name) != f"{prefix}{name}" for name in files):
            return False
    return True


def _valid_validation_manifest_contract(transform: Any, manifest: Any) -> bool:
    if not _valid_dataset_transform_contract(transform):
        return False
    if transform.get("type") == "eventhdr":
        return _valid_split_manifest_contract(manifest)
    return manifest is None


def _valid_calibration_runtime_contract(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if any(
        not isinstance(value.get(field), str) or not value[field].strip()
        for field in ("device", "torch")
    ):
        return False
    for field in ("cuda_runtime", "gpu_name"):
        item = value.get(field)
        if item is not None and (not isinstance(item, str) or not item.strip()):
            return False
    capability = value.get("compute_capability")
    return capability is None or (
        isinstance(capability, list)
        and len(capability) == 2
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in capability
        )
    )


def _valid_full_calibration_sample_ids(value: Any, expected_samples: Any) -> bool:
    if (
        not isinstance(expected_samples, int)
        or isinstance(expected_samples, bool)
        or expected_samples < 1
        or not isinstance(value, list)
        or len(value) != expected_samples
    ):
        return False
    identities: list[str] = []
    for expected_index, identity in enumerate(value):
        if (
            not isinstance(identity, dict)
            or identity.get("dataset_index") != expected_index
            or not isinstance(identity.get("group"), str)
            or not identity["group"].strip()
        ):
            return False
        try:
            identities.append(_canonical_sha256(identity))
        except (TypeError, ValueError, OverflowError):
            return False
    return len(identities) == len(set(identities))


def _valid_full_calibration_sampling(
    value: Any,
    sample_ids: Any,
    expected_samples: Any,
    expected_files: Any,
) -> bool:
    if not isinstance(value, dict) or value.get("selected") != sample_ids:
        return False
    integer_fields = ("selected_samples", "selected_groups", "available_groups")
    if any(
        not isinstance(value.get(field), int)
        or isinstance(value[field], bool)
        or value[field] < 1
        for field in integer_fields
    ):
        return False
    if value["selected_samples"] != expected_samples:
        return False
    per_group = value.get("per_group")
    available_per_group = value.get("available_per_group")
    if (
        not isinstance(per_group, dict)
        or not per_group
        or per_group != available_per_group
        or value["selected_groups"] != len(per_group)
        or value["available_groups"] != len(per_group)
        or any(
            not isinstance(group, str)
            or not group.strip()
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
            for group, count in per_group.items()
        )
        or sum(per_group.values()) != expected_samples
    ):
        return False
    source = value.get("source_fingerprint")
    files = source.get("files") if isinstance(source, dict) else None
    if (
        not isinstance(expected_files, int)
        or isinstance(expected_files, bool)
        or not isinstance(files, list)
        or len(files) != expected_files
    ):
        return False
    paths: list[str] = []
    for item in files:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not item["path"].strip()
            or not isinstance(item.get("size"), int)
            or isinstance(item["size"], bool)
            or item["size"] < 0
        ):
            return False
        paths.append(item["path"])
    return len(paths) == len(set(paths))


def _valid_dataset_index_contract(value: Any, expected_files: Any) -> bool:
    if not isinstance(value, dict) or value.get("version") != 1:
        return False
    samples = value.get("samples")
    groups = value.get("groups")
    per_group = value.get("per_group")
    if (
        not isinstance(samples, int)
        or isinstance(samples, bool)
        or samples < 1
        or not isinstance(groups, int)
        or isinstance(groups, bool)
        or groups < 1
        or not isinstance(per_group, dict)
        or len(per_group) != groups
        or any(
            not isinstance(group, str)
            or not group.strip()
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
            for group, count in per_group.items()
        )
        or sum(per_group.values()) != samples
        or not _is_sha256(value.get("sample_identities_sha256"))
    ):
        return False
    source = value.get("source_fingerprint")
    files = source.get("files") if isinstance(source, dict) else None
    if (
        not isinstance(expected_files, int)
        or isinstance(expected_files, bool)
        or expected_files < 1
        or not isinstance(files, list)
        or len(files) != expected_files
    ):
        return False
    paths: list[str] = []
    for item in files:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not item["path"].strip()
            or not isinstance(item.get("size"), int)
            or isinstance(item["size"], bool)
            or item["size"] < 0
        ):
            return False
        paths.append(item["path"])
    return len(paths) == len(set(paths))


def _sampling_matches_dataset_index_contract(
    sampling: Any,
    index_contract: Any,
    expected_files: Any,
) -> bool:
    if not _valid_dataset_index_contract(index_contract, expected_files):
        return False
    selected = sampling.get("selected") if isinstance(sampling, dict) else None
    if not _valid_full_calibration_sample_ids(selected, index_contract["samples"]):
        return False
    if not _valid_full_calibration_sampling(
        sampling,
        selected,
        index_contract["samples"],
        expected_files,
    ):
        return False
    return (
        sampling.get("selected_groups") == index_contract["groups"]
        and sampling.get("available_groups") == index_contract["groups"]
        and sampling.get("per_group") == index_contract["per_group"]
        and sampling.get("available_per_group") == index_contract["per_group"]
        and sampling.get("source_fingerprint") == index_contract["source_fingerprint"]
        and _canonical_sha256(selected)
        == index_contract["sample_identities_sha256"]
    )


def _eventhdr_index_matches_manifest(
    index_contract: Any,
    manifest: Any,
    split: str,
) -> bool:
    if not _valid_split_manifest_contract(manifest) or split not in {"train", "val"}:
        return False
    files = manifest[f"{split}_files"]
    mapping = manifest["file_to_group"][split]
    if not isinstance(index_contract, dict):
        return False
    source = index_contract.get("source_fingerprint")
    fingerprint_files = source.get("files") if isinstance(source, dict) else None
    if not isinstance(fingerprint_files, list):
        return False
    paths = [item.get("path") for item in fingerprint_files if isinstance(item, dict)]
    return (
        len(paths) == len(fingerprint_files)
        and set(paths) == set(files)
        and index_contract.get("groups") == len(mapping)
        and set(index_contract.get("per_group", {})) == set(mapping.values())
    )


def _valid_eventhdr_sample_identities(value: Any, manifest: Any, split: str) -> bool:
    if not isinstance(value, list) or not _valid_split_manifest_contract(manifest):
        return False
    if split not in {"train", "val"} or not _valid_full_calibration_sample_ids(
        value, len(value)
    ):
        return False
    mapping = manifest["file_to_group"][split]
    sequence_indices: dict[str, list[int]] = defaultdict(list)
    for identity in value:
        source_file = identity.get("source_file")
        group = identity.get("group")
        sequence_index = identity.get("sequence_index")
        start_idx = identity.get("start_idx")
        end_idx = identity.get("end_idx")
        timestamp = identity.get("timestamp")
        if (
            not isinstance(source_file, str)
            or mapping.get(source_file) != group
            or not isinstance(identity.get("image_key"), str)
            or not identity["image_key"].strip()
            or not isinstance(sequence_index, int)
            or isinstance(sequence_index, bool)
            or sequence_index < 0
            or not isinstance(start_idx, int)
            or isinstance(start_idx, bool)
            or start_idx < 0
            or not isinstance(end_idx, int)
            or isinstance(end_idx, bool)
            or end_idx < start_idx
            or not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(float(timestamp))
        ):
            return False
        sequence_indices[group].append(sequence_index)
    return set(sequence_indices) == set(mapping.values()) and all(
        indices == list(range(len(indices))) for indices in sequence_indices.values()
    )


def _valid_preflight_gate(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if (
        value.get("schema") != "asgcn_preflight_verification_v1"
        or value.get("status") != "verified"
        or value.get("report_eligible") is not True
    ):
        return False
    if any(
        not _is_sha256(value.get(field))
        for field in (
            "report_sha256",
            "config_sha256",
            "data_sha256",
            "source_tree_sha256",
        )
    ):
        return False
    measured_steps = value.get("measured_steps")
    batch_size = value.get("batch_size", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        return False
    if batch_size > 1:
        batch = value.get("batch_preflight")
        if (
            not isinstance(batch, dict)
            or batch.get("contract") != batching_contract(batch_size)
            or not _is_sha256(batch.get("schedule_sha256"))
            or batch.get("largest_measured_batch_size") != batch_size
            or isinstance(batch.get("measured_batches"), bool)
            or not isinstance(batch.get("measured_batches"), int)
            or batch["measured_batches"] < 1
        ):
            return False
    scope = value.get("measurement_scope")
    return (
        isinstance(measured_steps, int)
        and not isinstance(measured_steps, bool)
        and measured_steps >= 1
        and isinstance(value.get("gpu"), dict)
        and isinstance(scope, dict)
        and scope.get("name") == "selected_top_density_training_steps"
        and scope.get("topology_scope") == "complete_eventhdr_training_split"
        and scope.get("absolute_vram_guarantee") is False
        and isinstance(scope.get("statement"), str)
        and bool(scope["statement"].strip())
    )


def _ann_reporting_reasons(checkpoint: dict[str, Any]) -> list[str]:
    """Return every reason an ANN checkpoint cannot support published results."""
    reasons: list[str] = []
    if checkpoint.get("checkpoint_type") != "ann_inference":
        reasons.append("checkpoint_type is not ann_inference")
    if checkpoint.get("paper_core_version") != PAPER_CORE_VERSION:
        reasons.append("paper_core_version is missing or unsupported")
    if (
        not isinstance(checkpoint.get("epoch"), int)
        or isinstance(checkpoint.get("epoch"), bool)
        or checkpoint["epoch"] < 1
    ):
        reasons.append("ANN checkpoint epoch is invalid")
    training_only = sorted(
        key
        for key in ("optimizer", "scheduler", "scaler", "rng_state", "history")
        if key in checkpoint
    )
    if training_only:
        reasons.append("ANN inference checkpoint contains training-only state")
    if bool(checkpoint.get("batch_norm_folded")) or bool(
        checkpoint.get("parameter_normalized")
    ):
        reasons.append("ANN inference checkpoint contains converted SNN state")
    preflight_gate = checkpoint.get("preflight_gate")
    if not _valid_preflight_gate(preflight_gate):
        reasons.append("verified CUDA training preflight gate is missing or invalid")

    training = checkpoint.get("training_protocol")
    if not _valid_training_protocol_contract(training):
        reasons.append("complete training_protocol v5/v6 contract is missing or invalid")
        training = None
    elif training["version"] == 6 and (
        not isinstance(preflight_gate, dict)
        or preflight_gate.get("batch_size") != training["data_order"]["batch_size"]
        or not _valid_preflight_gate(preflight_gate)
    ):
        reasons.append("sequence-batch training requires a matching full-batch CUDA gate")

    validation = checkpoint.get("validation_protocol")
    if not isinstance(validation, dict) or validation.get("version") != 7:
        reasons.append("validation_protocol v7 is missing")
        validation = None
    else:
        dataset_content = validation.get("dataset_content")
        content_valid = isinstance(dataset_content, dict) and all(
            _valid_dataset_content_contract(dataset_content.get(split))
            for split in ("train", "validation")
        )
        if not content_valid:
            reasons.append("validation_protocol dataset content identity is invalid")
        dataset_transform = validation.get("dataset_transform")
        transform_valid = _valid_dataset_transform_contract(dataset_transform)
        if not transform_valid:
            reasons.append("validation_protocol dataset transform is invalid")
        elif dataset_transform.get("type") != "eventhdr":
            reasons.append("ANN reporting source dataset is not EventHDR")
        manifest = validation.get("split_manifest")
        if transform_valid and not _valid_validation_manifest_contract(
            dataset_transform, manifest
        ):
            reasons.append("validation_protocol split manifest is invalid")
        dataset_index = validation.get("dataset_index")
        train_index = (
            dataset_index.get("train") if isinstance(dataset_index, dict) else None
        )
        val_index = (
            dataset_index.get("validation")
            if isinstance(dataset_index, dict)
            else None
        )
        train_files = (
            dataset_content["train"]["files"] if content_valid else None
        )
        val_files = (
            dataset_content["validation"]["files"] if content_valid else None
        )
        if not _valid_dataset_index_contract(train_index, train_files) or not (
            _valid_dataset_index_contract(val_index, val_files)
        ):
            reasons.append("validation_protocol dataset index identity is invalid")
        sampling = validation.get("sampling")
        if not _sampling_matches_dataset_index_contract(sampling, val_index, val_files):
            reasons.append("validation_protocol does not prove full validation sampling")
        if validation.get("max_val_samples") is not None:
            reasons.append("validation_protocol uses a partial validation sample limit")
        if validation.get("ssim") != "gaussian_valid_11_sigma1.5":
            reasons.append("validation_protocol SSIM definition is invalid")
        if transform_valid and dataset_transform.get("type") == "eventhdr":
            if not _eventhdr_index_matches_manifest(train_index, manifest, "train") or not (
                _eventhdr_index_matches_manifest(val_index, manifest, "val")
            ):
                reasons.append("validation_protocol dataset index is not bound to manifest")
            selected = sampling.get("selected") if isinstance(sampling, dict) else None
            if not _valid_eventhdr_sample_identities(selected, manifest, "val"):
                reasons.append("validation sample identities are not bound to manifest")
            if isinstance(sampling, dict):
                selected_samples = sampling.get("selected_samples")
                context_samples = sampling.get("context_samples")
                forward_samples = sampling.get("forward_samples")
                if (
                    context_samples != 0
                    or not isinstance(selected_samples, int)
                    or isinstance(selected_samples, bool)
                    or forward_samples != selected_samples
                ):
                    reasons.append("full validation context coverage is inconsistent")
        val_result = checkpoint.get("val")
        val_frames = val_result.get("frames") if isinstance(val_result, dict) else None
        selected_samples = (
            sampling.get("selected_samples") if isinstance(sampling, dict) else None
        )
        if val_frames != selected_samples:
            reasons.append("ANN validation frame count differs from its sealed sampling")

    training_config = checkpoint.get("training_config")
    if not isinstance(training_config, dict):
        reasons.append("public ANN training config identity is missing")
    else:
        train_config = training_config.get("train")
        public_model_config = training_config.get("model")
        if not isinstance(public_model_config, dict) or (
            checkpoint.get("model_config") != public_model_config
        ):
            reasons.append("ANN model config differs from its training config")
        if validation is not None and isinstance(public_model_config, dict) and (
            validation.get("seed") != training_config.get("seed", 2026)
            or validation.get("recurrent")
            != bool(public_model_config.get("recurrent", True))
        ):
            reasons.append("ANN validation seed/recurrent settings differ from config")
        if not isinstance(train_config, dict):
            reasons.append("public ANN training settings are missing")
        else:
            if train_config.get("max_train_samples") is not None:
                reasons.append("ANN training used a partial training sample limit")
            if train_config.get("max_val_samples") is not None:
                reasons.append("ANN training used a partial validation sample limit")
            if (
                validation is not None
                and validation.get("max_val_samples")
                != train_config.get("max_val_samples")
            ):
                reasons.append("ANN validation sample limit differs from training config")
            sampling = validation.get("sampling") if validation is not None else None
            transform = (
                validation.get("dataset_transform") if validation is not None else None
            )
            if (
                isinstance(sampling, dict)
                and isinstance(transform, dict)
                and transform.get("type") == "eventhdr"
            ):
                recurrent = bool(validation.get("recurrent"))
                expected_context = (
                    train_config.get("validation_context_frames", 64)
                    if recurrent
                    else 0
                )
                expected_policy = (
                    "full_group_prefix"
                    if recurrent and expected_context is None
                    else "bounded_predecessor"
                    if recurrent
                    else "none_non_recurrent"
                )
                if (
                    sampling.get("max_context_frames_per_group") != expected_context
                    or sampling.get("context_policy") != expected_policy
                ):
                    reasons.append(
                        "ANN validation context policy differs from training config"
                    )
            if training is not None:
                reasons.extend(
                    _training_protocol_config_reasons(training, training_config)
                )
        dataset_config = training_config.get("dataset")
        if not isinstance(dataset_config, dict) or dataset_config.get("type") != "eventhdr":
            reasons.append("ANN training config does not use EventHDR")
        elif not dataset_config.get("split_manifest"):
            reasons.append("ANN training config is missing its split manifest reference")

    if isinstance(preflight_gate, dict) and training is not None and validation is not None:
        dataset_content = validation.get("dataset_content")
        training_content = (
            dataset_content.get("train") if isinstance(dataset_content, dict) else None
        )
        training_source = training.get("source")
        if (
            not isinstance(training_content, dict)
            or preflight_gate.get("data_sha256") != training_content.get("sha256")
        ):
            reasons.append("preflight data identity is not bound to ANN training data")
        if (
            not isinstance(training_source, dict)
            or preflight_gate.get("source_tree_sha256")
            != training_source.get("source_tree_sha256")
        ):
            reasons.append("preflight source identity is not bound to ANN training source")
        if isinstance(training_config, dict):
            try:
                configured_transform = _dataset_transform_contract(training_config)
            except (KeyError, TypeError, AttributeError):
                configured_transform = None
            if validation.get("dataset_transform") != configured_transform:
                reasons.append(
                    "ANN validation transform is not bound to its training config"
                )
            config_without_gate = copy.deepcopy(training_config)
            config_without_gate.pop("preflight_gate", None)
            if preflight_gate.get("config_sha256") != _canonical_sha256(
                _public_config(config_without_gate)
            ):
                reasons.append("preflight config identity is not bound to ANN training config")

    try:
        validation_score = _macro_ssim(checkpoint.get("val", {}))
    except (AttributeError, TypeError, ValueError, OverflowError):
        validation_score = float("nan")
    try:
        best_score = float(checkpoint.get("best_ssim"))
    except (TypeError, ValueError, OverflowError):
        best_score = float("nan")
    if checkpoint.get("best_metric") != "macro_ssim":
        reasons.append("ANN checkpoint was not selected by macro_ssim")
    if not math.isfinite(validation_score) or not math.isfinite(best_score):
        reasons.append("ANN checkpoint has no finite validation selection score")
    elif validation_score != best_score:
        reasons.append("ANN validation score does not match its best score")

    terminal = training.get("terminal_validation") if training is not None else None
    eventhdr_reporting = (
        validation is not None
        and isinstance(validation.get("dataset_transform"), dict)
        and validation["dataset_transform"].get("type") == "eventhdr"
    )
    if eventhdr_reporting and terminal is None:
        reasons.append("EventHDR reporting requires sealed single-final-epoch validation")
    elif terminal is not None:
        state = checkpoint.get("terminal_validation_state")
        planned_epoch = terminal.get("planned_epoch") if isinstance(terminal, dict) else None
        if (
            not isinstance(terminal, dict)
            or terminal.get("mode") != "single_final_epoch"
            or not isinstance(planned_epoch, int)
            or isinstance(planned_epoch, bool)
            or not isinstance(state, dict)
            or state.get("completed") is not True
            or state.get("planned_epoch") != planned_epoch
            or state.get("completed_epoch") != planned_epoch
            or checkpoint.get("epoch") != planned_epoch
            or checkpoint.get("checkpoint_selection") != "single_final_epoch"
            or (
                validation is not None
                and validation.get("selection_metric")
                != "single_final_epoch_macro_ssim"
            )
        ):
            reasons.append("sealed final-only validation did not complete at its planned epoch")
    elif checkpoint.get("checkpoint_selection") != "best_validation_macro_ssim":
        reasons.append("ANN checkpoint selection protocol is invalid")
    return reasons


def _source_ann_reporting_contract(checkpoint: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "checkpoint_type",
        "epoch",
        "model_state_sha256",
        "model_config",
        "paper_core_version",
        "preflight_gate",
        "training_config",
        "val",
        "best_ssim",
        "best_metric",
        "checkpoint_selection",
        "validation_protocol",
        "training_protocol",
        "terminal_validation_state",
    )
    return _public_config({field: checkpoint.get(field) for field in fields})


def _snn_reporting_reasons(checkpoint: dict[str, Any]) -> list[str]:
    """Return every reason an SNN checkpoint cannot support published results."""
    reasons: list[str] = []
    protocol = checkpoint.get("calibration_protocol")
    if not isinstance(protocol, dict):
        return ["sealed calibration_protocol is missing"]
    if protocol.get("version") != 2:
        reasons.append("calibration_protocol v2 is missing")
    if protocol.get("sealed") is not True:
        reasons.append("calibration_protocol.sealed is not true")
    if protocol.get("unsealed_reasons") != []:
        reasons.append("calibration_protocol contains unsealed reasons")
    if protocol.get("source_checkpoint_type") != "ann_inference":
        reasons.append("calibration source is not a clean ANN inference checkpoint")
    for field in ("source_ann_model_sha256", "source_ann_checkpoint_sha256"):
        if not _is_sha256(protocol.get(field)):
            reasons.append(f"calibration_protocol {field} is invalid")
    source_ann_protocols: dict[str, dict[str, Any]] = {}
    for name, expected_versions in (("training", {5, 6}), ("validation", {7})):
        identity = protocol.get(f"source_ann_{name}_protocol")
        flat_digest = protocol.get(f"source_ann_{name}_protocol_sha256")
        contract = identity.get("contract") if isinstance(identity, dict) else None
        digest = identity.get("sha256") if isinstance(identity, dict) else None
        if (
            not isinstance(contract, dict)
            or contract.get("version") not in expected_versions
            or not _is_sha256(digest)
            or digest != _canonical_sha256(contract)
            or flat_digest != digest
        ):
            reasons.append(f"calibration source ANN {name} protocol identity is invalid")
        else:
            source_ann_protocols[name] = contract
    source_ann_reporting_identity = protocol.get("source_ann_reporting_contract")
    source_ann_reporting = (
        source_ann_reporting_identity.get("contract")
        if isinstance(source_ann_reporting_identity, dict)
        else None
    )
    source_ann_reporting_sha256 = (
        source_ann_reporting_identity.get("sha256")
        if isinstance(source_ann_reporting_identity, dict)
        else None
    )
    try:
        reporting_digest_matches = (
            isinstance(source_ann_reporting, dict)
            and _is_sha256(source_ann_reporting_sha256)
            and source_ann_reporting_sha256
            == _canonical_sha256(source_ann_reporting)
            and protocol.get("source_ann_reporting_contract_sha256")
            == source_ann_reporting_sha256
        )
    except (TypeError, ValueError, OverflowError):
        reporting_digest_matches = False
    if not reporting_digest_matches:
        reasons.append("calibration source ANN reporting contract identity is invalid")
        source_ann_reporting = None
    else:
        reasons.extend(
            f"calibration source ANN reporting contract: {reason}"
            for reason in _ann_reporting_reasons(source_ann_reporting)
        )
    training_config_identity = protocol.get("source_ann_training_config")
    source_ann_training_config = (
        training_config_identity.get("contract")
        if isinstance(training_config_identity, dict)
        else None
    )
    source_ann_training_config_sha256 = (
        training_config_identity.get("sha256")
        if isinstance(training_config_identity, dict)
        else None
    )
    if (
        not isinstance(source_ann_training_config, dict)
        or not _is_sha256(source_ann_training_config_sha256)
        or source_ann_training_config_sha256
        != _canonical_sha256(source_ann_training_config)
        or protocol.get("source_ann_training_config_sha256")
        != source_ann_training_config_sha256
    ):
        reasons.append("calibration source ANN training config identity is invalid")
    if isinstance(source_ann_reporting, dict):
        if source_ann_reporting.get("checkpoint_type") != protocol.get(
            "source_checkpoint_type"
        ):
            reasons.append("calibration source ANN checkpoint type lineage differs")
        if source_ann_reporting.get("model_state_sha256") != protocol.get(
            "source_ann_model_sha256"
        ):
            reasons.append("calibration source ANN model lineage differs")
        if source_ann_reporting.get("epoch") != protocol.get("source_epoch"):
            reasons.append("calibration source ANN epoch lineage differs")
        if source_ann_reporting.get("training_protocol") != source_ann_protocols.get(
            "training"
        ):
            reasons.append("calibration source ANN training protocol lineage differs")
        if source_ann_reporting.get("validation_protocol") != source_ann_protocols.get(
            "validation"
        ):
            reasons.append("calibration source ANN validation protocol lineage differs")
        if source_ann_reporting.get("training_config") != source_ann_training_config:
            reasons.append("calibration source ANN training config lineage differs")
        if checkpoint.get("epoch") != source_ann_reporting.get("epoch"):
            reasons.append("SNN checkpoint epoch differs from its source ANN")
        if checkpoint.get("model_config") != source_ann_reporting.get("model_config"):
            reasons.append("SNN checkpoint model config differs from its source ANN")
        if checkpoint.get("paper_core_version") != source_ann_reporting.get(
            "paper_core_version"
        ):
            reasons.append("SNN checkpoint paper-core version differs from its source ANN")
    content = protocol.get("dataset_content")
    if not _valid_dataset_content_contract(content):
        reasons.append("calibration dataset content identity is invalid")
    elif protocol.get("dataset_content_sha256") != content["sha256"]:
        reasons.append("calibration dataset content digest is inconsistent")
    transform = protocol.get("dataset_transform")
    if (
        not _valid_dataset_transform_contract(transform)
        or transform.get("type") != "eventhdr"
    ):
        reasons.append("calibration dataset transform is invalid")
    manifest = protocol.get("split_manifest")
    if not _valid_split_manifest_contract(manifest):
        reasons.append("calibration split manifest is invalid")
    if not _valid_calibration_runtime_contract(protocol.get("runtime")):
        reasons.append("calibration runtime identity is invalid")
    calibration_samples = checkpoint.get("snn_calibration_samples")
    sample_ids = protocol.get("selected_sample_ids")
    if not _valid_full_calibration_sample_ids(sample_ids, calibration_samples):
        reasons.append("calibration sample identities do not prove full coverage")
    expected_files = content.get("files") if isinstance(content, dict) else None
    if not _valid_full_calibration_sampling(
        checkpoint.get("snn_calibration_sampling"),
        sample_ids,
        calibration_samples,
        expected_files,
    ):
        reasons.append("calibration sampling contract does not prove full coverage")
    for field in ("training_source", "calibration_source"):
        if not _valid_source_contract(protocol.get(field)):
            reasons.append(f"calibration_protocol {field} identity is invalid")
    if (
        _valid_source_contract(protocol.get("training_source"))
        and _valid_source_contract(protocol.get("calibration_source"))
        and protocol["training_source"] != protocol["calibration_source"]
    ):
        reasons.append("calibration source differs from the ANN training source")

    training_protocol = source_ann_protocols.get("training")
    validation_protocol = source_ann_protocols.get("validation")
    source_training_source = (
        training_protocol.get("source") if isinstance(training_protocol, dict) else None
    )
    if not _valid_source_contract(source_training_source):
        reasons.append("calibration source ANN training source identity is invalid")
    elif protocol.get("training_source") != source_training_source:
        reasons.append("calibration training source differs from the source ANN protocol")
    source_transform = (
        validation_protocol.get("dataset_transform")
        if isinstance(validation_protocol, dict)
        else None
    )
    source_manifest = (
        validation_protocol.get("split_manifest")
        if isinstance(validation_protocol, dict)
        else None
    )
    if (
        not _valid_dataset_transform_contract(source_transform)
        or source_transform.get("type") != "eventhdr"
    ):
        reasons.append("calibration source ANN dataset transform is invalid")
    elif transform != source_transform:
        reasons.append("calibration transform differs from the source ANN protocol")
    if not _valid_split_manifest_contract(source_manifest):
        reasons.append("calibration source ANN split manifest is invalid")
    elif manifest != source_manifest:
        reasons.append("calibration split manifest differs from the source ANN protocol")
    source_dataset_content = (
        validation_protocol.get("dataset_content")
        if isinstance(validation_protocol, dict)
        else None
    )
    source_training_content = (
        source_dataset_content.get("train")
        if isinstance(source_dataset_content, dict)
        else None
    )
    if not _valid_dataset_content_contract(source_training_content):
        reasons.append("calibration source ANN training data identity is invalid")
    elif content != source_training_content:
        reasons.append("calibration data differs from the source ANN training data")
    source_dataset_index = (
        validation_protocol.get("dataset_index")
        if isinstance(validation_protocol, dict)
        else None
    )
    source_training_index = (
        source_dataset_index.get("train")
        if isinstance(source_dataset_index, dict)
        else None
    )
    source_training_files = (
        source_training_content.get("files")
        if isinstance(source_training_content, dict)
        else None
    )
    if not _valid_dataset_index_contract(
        source_training_index, source_training_files
    ):
        reasons.append("calibration source ANN training index identity is invalid")
    elif not _eventhdr_index_matches_manifest(
        source_training_index, source_manifest, "train"
    ):
        reasons.append("calibration source ANN training index is not bound to manifest")
    if not _valid_eventhdr_sample_identities(sample_ids, source_manifest, "train"):
        reasons.append("calibration sample identities are not bound to ANN training index")
    if not _sampling_matches_dataset_index_contract(
        checkpoint.get("snn_calibration_sampling"),
        source_training_index,
        source_training_files,
    ):
        reasons.append("calibration sampling differs from the ANN training index")
    if (
        isinstance(source_ann_training_config, dict)
        and _valid_dataset_transform_contract(source_transform)
    ):
        try:
            configured_transform = _dataset_transform_contract(source_ann_training_config)
        except (KeyError, TypeError, AttributeError):
            configured_transform = None
        if configured_transform != source_transform:
            reasons.append(
                "calibration source ANN transform is not bound to its training config"
            )
    source_preflight = protocol.get("source_preflight_gate")
    if (
        not isinstance(source_preflight, dict)
        or not _valid_preflight_gate(source_preflight.get("contract"))
        or source_preflight.get("sha256")
        != _canonical_sha256(source_preflight.get("contract"))
    ):
        reasons.append("calibration source preflight gate identity is invalid")
    elif checkpoint.get("preflight_gate") != source_preflight.get("contract"):
        reasons.append("SNN checkpoint preflight gate differs from its calibration source")
    else:
        preflight_contract = source_preflight["contract"]
        if (
            isinstance(source_ann_reporting, dict)
            and source_ann_reporting.get("preflight_gate") != preflight_contract
        ):
            reasons.append("calibration source ANN preflight lineage differs")
        training_source = (
            training_protocol.get("source") if isinstance(training_protocol, dict) else None
        )
        dataset_content = (
            validation_protocol.get("dataset_content")
            if isinstance(validation_protocol, dict)
            else None
        )
        training_content = (
            dataset_content.get("train") if isinstance(dataset_content, dict) else None
        )
        if (
            not isinstance(training_content, dict)
            or preflight_contract.get("data_sha256") != training_content.get("sha256")
        ):
            reasons.append("SNN source preflight data identity is not bound to ANN training data")
        if (
            not isinstance(training_source, dict)
            or preflight_contract.get("source_tree_sha256")
            != training_source.get("source_tree_sha256")
        ):
            reasons.append(
                "SNN source preflight source identity is not bound to ANN training source"
            )
        if isinstance(source_ann_training_config, dict):
            config_without_gate = copy.deepcopy(source_ann_training_config)
            config_without_gate.pop("preflight_gate", None)
            if preflight_contract.get("config_sha256") != _canonical_sha256(
                _public_config(config_without_gate)
            ):
                reasons.append(
                    "SNN source preflight config identity is not bound to ANN training config"
                )
    if checkpoint.get("report_eligible") is not True:
        reasons.append("SNN checkpoint is permanently marked non-reporting")
    if checkpoint.get("report_ineligible_reasons") != []:
        reasons.append("SNN checkpoint contains report-ineligible reasons")
    return reasons


def _reporting_checkpoint_contract(
    checkpoint: dict[str, Any],
    checkpoint_path: str | Path,
    inference_mode: str,
    *,
    allow_unsealed_checkpoint_for_non_reporting: bool,
) -> tuple[dict[str, Any], bool, list[str]]:
    """Validate publication lineage and return its path-free identity contract."""
    reasons = (
        _ann_reporting_reasons(checkpoint)
        if inference_mode == "ann"
        else _snn_reporting_reasons(checkpoint)
    )
    if reasons and not allow_unsealed_checkpoint_for_non_reporting:
        raise ValueError(
            "Checkpoint reporting protocol is not sealed: "
            + "; ".join(reasons)
            + ". Use the explicit non-reporting override only for synthetic tests."
        )
    if allow_unsealed_checkpoint_for_non_reporting:
        reasons = ["explicit non-reporting checkpoint override requested", *reasons]
    report_eligible = not reasons
    common: dict[str, Any] = {
        "checkpoint_type": checkpoint.get("checkpoint_type"),
        "checkpoint": _artifact_path_label(checkpoint_path),
        "checkpoint_file_sha256": _file_sha256(checkpoint_path),
        "model_state_sha256": checkpoint.get("model_state_sha256"),
        "epoch": checkpoint.get("epoch"),
        "paper_core_version": checkpoint.get("paper_core_version"),
    }
    if inference_mode == "ann":
        training = checkpoint.get("training_protocol")
        validation = checkpoint.get("validation_protocol")
        common.update(
            {
                "training_protocol": _hashed_contract(_public_config(training)),
                "validation_protocol": _hashed_contract(_public_config(validation)),
                "training_config": _hashed_contract(
                    _public_config(checkpoint.get("training_config"))
                ),
                "terminal_validation_state": _public_config(
                    checkpoint.get("terminal_validation_state")
                ),
                "preflight_gate": _hashed_contract(
                    _public_config(checkpoint.get("preflight_gate"))
                ),
                "selection": {
                    "metric": checkpoint.get("best_metric"),
                    "score": checkpoint.get("best_ssim"),
                    "checkpoint_selection": checkpoint.get("checkpoint_selection"),
                },
            }
        )
    else:
        calibration = checkpoint.get("calibration_protocol")
        common["calibration_protocol"] = _hashed_contract(_public_config(calibration))
        common["preflight_gate"] = _hashed_contract(
            _public_config(checkpoint.get("preflight_gate"))
        )
        common["source_ann"] = {
            "model_state_sha256": (
                calibration.get("source_ann_model_sha256")
                if isinstance(calibration, dict)
                else None
            ),
            "checkpoint_file_sha256": (
                calibration.get("source_ann_checkpoint_sha256")
                if isinstance(calibration, dict)
                else None
            ),
            "training_protocol_sha256": (
                calibration.get("source_ann_training_protocol_sha256")
                if isinstance(calibration, dict)
                else None
            ),
            "training_protocol": (
                calibration.get("source_ann_training_protocol")
                if isinstance(calibration, dict)
                else None
            ),
            "validation_protocol_sha256": (
                calibration.get("source_ann_validation_protocol_sha256")
                if isinstance(calibration, dict)
                else None
            ),
            "validation_protocol": (
                calibration.get("source_ann_validation_protocol")
                if isinstance(calibration, dict)
                else None
            ),
            "training_config_sha256": (
                calibration.get("source_ann_training_config_sha256")
                if isinstance(calibration, dict)
                else None
            ),
            "training_config": (
                calibration.get("source_ann_training_config")
                if isinstance(calibration, dict)
                else None
            ),
        }
    common["lineage_sha256"] = _canonical_sha256(common)
    return common, report_eligible, reasons


def _merge_reporting_reasons(
    report_eligible: bool,
    current_reasons: list[str],
    new_reasons: list[str],
    *,
    allow_non_reporting: bool,
    scope: str,
) -> tuple[bool, list[str]]:
    unique_new = [reason for reason in new_reasons if reason not in current_reasons]
    if unique_new and not allow_non_reporting:
        raise ValueError(
            f"{scope} reporting protocol is not sealed: "
            + "; ".join(unique_new)
            + ". Use the explicit non-reporting override only for synthetic tests."
        )
    if unique_new:
        current_reasons.extend(unique_new)
        report_eligible = False
    return report_eligible, current_reasons


def _set_inference_snn_dynamics(
    model: ASGCNUNet,
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


def _require_finite_tensor(value: torch.Tensor, label: str, sample_id: Any) -> None:
    if (value.is_floating_point() or value.is_complex()) and not bool(
        torch.isfinite(value).all()
    ):
        raise FloatingPointError(f"Non-finite {label}: sample={sample_id}")


def _require_finite_structure(value: Any, label: str, sample_id: Any) -> None:
    """Reject NaN/Inf anywhere in a public diagnostic or metric structure."""
    if torch.is_tensor(value):
        _require_finite_tensor(value, label, sample_id)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_structure(item, f"{label}.{key}", sample_id)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_structure(item, f"{label}[{index}]", sample_id)
        return
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"Non-finite {label}: sample={sample_id}")


def _positive_interval_us(value: Any, sample_id: Any) -> float | None:
    if value is None:
        return None
    try:
        interval = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid dt_us: sample={sample_id}") from error
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError(f"dt_us must be finite and positive: sample={sample_id}")
    return interval


def _model_parameter_dtype(model: torch.nn.Module) -> str:
    dtypes = {str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters()}
    if not dtypes:
        return "none"
    if len(dtypes) == 1:
        return next(iter(dtypes))
    return "mixed:" + ",".join(sorted(dtypes))


def _inference_precision(
    eval_config: dict[str, Any],
    device: torch.device,
    model: torch.nn.Module,
) -> tuple[dict[str, Any], torch.dtype | None]:
    requested = str(eval_config.get("precision", "fp32")).strip().lower()
    if requested not in {"fp32", "amp_fp16", "bf16"}:
        raise ValueError("eval.precision must be one of: fp32, amp_fp16, bf16")
    autocast_dtype: torch.dtype | None = None
    if requested == "amp_fp16":
        if device.type != "cuda":
            raise ValueError("eval.precision=amp_fp16 requires a CUDA device")
        autocast_dtype = torch.float16
    elif requested == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("eval.precision=bf16 requires CUDA BF16 support")
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("eval.precision=bf16 is supported only on CPU or CUDA")
        autocast_dtype = torch.bfloat16
    requested_tf32 = bool(eval_config.get("tf32", False))
    effective_tf32 = requested_tf32 and device.type == "cuda"
    return (
        {
            "requested": requested,
            "effective": (
                "float16_autocast"
                if autocast_dtype == torch.float16
                else "bfloat16_autocast"
                if autocast_dtype == torch.bfloat16
                else "fp32"
            ),
            "autocast_dtype": (
                str(autocast_dtype).removeprefix("torch.")
                if autocast_dtype is not None
                else None
            ),
            "model_parameter_dtype": _model_parameter_dtype(model),
            "device": str(device),
            "tf32": effective_tf32,
            "tf32_requested": requested_tf32,
        },
        autocast_dtype,
    )


@contextmanager
def _inference_precision_context(
    device: torch.device,
    precision: dict[str, Any],
    autocast_dtype: torch.dtype | None,
):
    old_matmul_tf32 = None
    old_cudnn_tf32 = None
    if device.type == "cuda":
        old_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        old_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
        effective_tf32 = bool(precision["tf32"])
        torch.backends.cuda.matmul.allow_tf32 = effective_tf32
        torch.backends.cudnn.allow_tf32 = effective_tf32
    try:
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            yield
    finally:
        if device.type == "cuda":
            assert old_matmul_tf32 is not None and old_cudnn_tf32 is not None
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


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
    batch_sampler=None,
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
    if batch_sampler is not None:
        loader_options.pop("batch_size")
        loader_options.pop("shuffle")
        loader_options["batch_sampler"] = batch_sampler
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
        # get_rng_state_all() enumerates devices before its first lazy state read.
        torch.cuda.init()
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
        # Resume validates the initialized MIG-visible count, not NVML's count.
        torch.cuda.init()
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


def _training_protocol_config_reasons(
    protocol: dict[str, Any], config: dict[str, Any]
) -> list[str]:
    """Cross-bind normalized optimization choices to the public training config."""
    reasons: list[str] = []
    train_config = config.get("train")
    if not isinstance(train_config, dict):
        return ["public ANN training settings are missing"]
    try:
        validate_experiment_config(config)
        optimizer_mode = _optimizer_mode(train_config)
        optimizer = {
            "mode": optimizer_mode,
            "name": "AdamW" if optimizer_mode == "adamw" else "Adam",
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
        }
        configured_weights = train_config.get("loss_weights") or {}
        loss_weights = {
            "charbonnier": float(configured_weights.get("charbonnier", 1.0)),
            "ssim": float(configured_weights.get("ssim", 0.2)),
            "gradient": float(configured_weights.get("gradient", 0.1)),
            "temporal": float(configured_weights.get("temporal", 0.0)),
        }
        num_workers = int(train_config.get("num_workers", 0))
        persistent_workers = train_config.get("persistent_workers")
        prefetch_factor = train_config.get("prefetch_factor")
        max_train_samples = train_config.get("max_train_samples")
        data_order = {
            "batch_size": int(train_config.get("batch_size", 1)),
            "max_train_samples": (
                None if max_train_samples is None else int(max_train_samples)
            ),
            "shuffle": False,
            "num_workers": num_workers,
            "persistent_workers": (
                None
                if num_workers == 0
                else True
                if persistent_workers is None
                else bool(persistent_workers)
            ),
            "prefetch_factor": (
                None
                if num_workers == 0
                else 2
                if prefetch_factor is None
                else int(prefetch_factor)
            ),
        }
        raw_validate_every = train_config.get("validate_every", 1)
        validate_every = (
            None
            if raw_validate_every is None
            else max(1, int(raw_validate_every))
        )
        expected_fields = {
            "version": 5 if int(train_config.get("batch_size", 1)) == 1 else 6,
            "seed": int(config.get("seed", 2026)),
            "optimizer": optimizer,
            "scheduler": _scheduler_spec(train_config),
            "loss_weights": loss_weights,
            "gradient_clipping": {
                "max_norm": float(train_config.get("grad_clip", 1.0)),
                "norm_type": 2.0,
            },
            "data_order": data_order,
            "validate_every": validate_every,
            "checkpoint_selection": (
                "single_final_epoch"
                if validate_every is None
                else "best_validation_macro_ssim"
            ),
            "terminal_validation": (
                {
                    "mode": "single_final_epoch",
                    "planned_epoch": int(train_config.get("epochs", 40)),
                }
                if validate_every is None
                else None
            ),
        }
        if int(train_config.get("batch_size", 1)) > 1:
            expected_fields["batching"] = batching_contract(int(train_config["batch_size"]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return ["public ANN training config cannot reproduce its protocol"]

    for field, expected in expected_fields.items():
        if protocol.get(field) != expected:
            reasons.append(f"ANN training protocol {field} differs from training config")

    mixed_precision = protocol.get("mixed_precision")
    requested_amp = bool(train_config.get("amp", True))
    runtime = protocol.get("runtime")
    if not isinstance(mixed_precision, dict) or set(mixed_precision) != {
        "requested",
        "effective",
        "autocast_dtype",
        "gradient_scaler",
        "overflow_policy",
    }:
        reasons.append("ANN training mixed-precision protocol is invalid")
    else:
        effective_amp = mixed_precision.get("effective")
        device_type = runtime.get("device_type") if isinstance(runtime, dict) else None
        if (
            mixed_precision.get("requested") != requested_amp
            or not isinstance(effective_amp, bool)
            or effective_amp != mixed_precision.get("gradient_scaler")
            or mixed_precision.get("autocast_dtype")
            != ("float16" if effective_amp else None)
            or effective_amp != (requested_amp and device_type == "cuda")
            or mixed_precision.get("overflow_policy") != _amp_retry_policy(effective_amp)
        ):
            reasons.append("ANN training mixed precision differs from config/runtime")

    runtime_fields = {
        "device_type",
        "torch",
        "cuda_runtime",
        "cudnn",
        "gpu_name",
        "compute_capability",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cudnn_benchmark",
        "deterministic_algorithms",
    }
    if not isinstance(runtime, dict) or set(runtime) != runtime_fields:
        reasons.append("ANN training runtime protocol is invalid")
    else:
        device_type = runtime.get("device_type")
        capability = runtime.get("compute_capability")
        if (
            device_type not in {"cpu", "cuda"}
            or not isinstance(runtime.get("torch"), str)
            or not runtime["torch"].strip()
            or not isinstance(runtime.get("deterministic_algorithms"), bool)
            or (
                device_type == "cuda"
                and (
                    not isinstance(runtime.get("gpu_name"), str)
                    or not runtime["gpu_name"].strip()
                    or not isinstance(runtime.get("cuda_runtime"), str)
                    or not runtime["cuda_runtime"].strip()
                    or not isinstance(capability, list)
                    or len(capability) != 2
                    or any(
                        not isinstance(item, int) or isinstance(item, bool)
                        for item in capability
                    )
                    or any(
                        not isinstance(runtime.get(field), bool)
                        for field in (
                            "cuda_matmul_allow_tf32",
                            "cudnn_allow_tf32",
                            "cudnn_benchmark",
                        )
                    )
                )
            )
            or (
                device_type == "cpu"
                and (
                    any(
                        runtime.get(field) is not None
                        for field in (
                            "cuda_runtime",
                            "cudnn",
                            "gpu_name",
                            "compute_capability",
                            "cuda_matmul_allow_tf32",
                            "cudnn_allow_tf32",
                            "cudnn_benchmark",
                        )
                    )
                )
            )
        ):
            reasons.append("ANN training runtime device identity is invalid")
    return reasons


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


def _current_source_contract() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    return {
        "source_tree_sha256": _source_tree_sha256(project_root),
        **_git_provenance(project_root),
    }


def _training_protocol(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Return every configured choice that can change the optimization trajectory.

    Logging cadence, the resume path, and output paths are deliberately absent.
    ``epochs`` is sealed only for final-only validation because changing that value
    would move the one permitted evaluation. The normalized values below make
    omitted defaults compare equal to explicit defaults.
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
        "version": 5 if int(train_config.get("batch_size", 1)) == 1 else 6,
        **(
            {"batching": batching_contract(int(train_config["batch_size"]))}
            if int(train_config.get("batch_size", 1)) > 1 else {}
        ),
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
            "overflow_policy": _amp_retry_policy(effective_amp),
        },
        "validate_every": validate_every,
        "checkpoint_selection": (
            "single_final_epoch" if validate_every is None else "best_validation_macro_ssim"
        ),
        "terminal_validation": (
            {
                "mode": "single_final_epoch",
                "planned_epoch": int(train_config.get("epochs", 40)),
            }
            if validate_every is None
            else None
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


def _validate_terminal_validation_resume(
    checkpoint: dict[str, Any], expected_protocol: dict[str, Any]
) -> None:
    expected = expected_protocol.get("terminal_validation")
    if expected is None:
        return
    actual_protocol = checkpoint.get("training_protocol")
    actual = (
        actual_protocol.get("terminal_validation")
        if isinstance(actual_protocol, dict)
        else None
    )
    state = checkpoint.get("terminal_validation_state")
    if not isinstance(actual, dict) or not isinstance(state, dict):
        raise TypeError(
            "Final-only resume checkpoint is missing the sealed terminal validation contract"
        )
    old_epoch = int(actual.get("planned_epoch", -1))
    requested_epoch = int(expected.get("planned_epoch", -1))
    if bool(state.get("completed")) and requested_epoch != old_epoch:
        raise ValueError(
            "Final-only validation already completed at the sealed terminal epoch; "
            "extend training in a new run instead of reusing this run"
        )
    if requested_epoch != old_epoch:
        raise ValueError(
            "Final-only validation planned terminal epoch differs from the resume checkpoint"
        )
    completed_epoch = state.get("completed_epoch")
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if bool(state.get("completed")):
        if (
            completed_epoch is None
            or int(completed_epoch) != old_epoch
            or checkpoint_epoch != old_epoch
            or not math.isfinite(_macro_ssim(checkpoint.get("val", {})))
        ):
            raise ValueError("Final-only validation completion state is inconsistent")
    elif completed_epoch is not None or checkpoint_epoch >= old_epoch:
        raise ValueError("Incomplete final-only validation state is inconsistent")


def _ensure_finite_loss(
    loss: torch.Tensor,
    loss_parts: dict[str, torch.Tensor],
    *,
    epoch: int,
    step: int,
    sample_id: Any,
) -> dict[str, float]:
    context = f"epoch={epoch}, step={step}, sample={sample_id}"
    part_names = list(loss_parts)
    tensors = [loss.detach().reshape(())]
    for name in part_names:
        value = loss_parts[name]
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise TypeError(f"Loss component {name!r} must be a scalar tensor")
        tensors.append(value.detach().reshape(()))
    # The packed transfer is also the single synchronization used for fail-fast
    # finite checks and progress logging on this optimization step.
    cpu_values = (
        torch.stack(tensors).to(device="cpu", dtype=torch.float64).tolist()
    )
    invalid_values = [] if math.isfinite(cpu_values[0]) else ["total loss"]
    invalid_values.extend(
        f"{name} component"
        for name, value in zip(part_names, cpu_values[1:], strict=True)
        if not math.isfinite(value)
    )
    if invalid_values:
        raise FloatingPointError(f"Non-finite {', '.join(invalid_values)} at {context}")
    return {
        "total": float(cpu_values[0]),
        **{
            name: float(value)
            for name, value in zip(part_names, cpu_values[1:], strict=True)
        },
    }


def _clip_and_validate_gradients(
    model: torch.nn.Module,
    max_norm: float,
    *,
    epoch: int,
    step: int,
    sample_id: Any,
) -> float:
    """Validate the clipping norm once; collect parameter names only on failure."""
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("train.grad_clip must be finite and greater than zero")
    parameters = list(model.parameters())
    get_norm = getattr(torch.nn.utils, "get_total_norm", None)
    clip_with_norm = getattr(torch.nn.utils, "clip_grads_with_norm_", None)
    if callable(get_norm) and callable(clip_with_norm):
        # These public functions implement the same norm and clipping operations
        # as clip_grad_norm_. A finite L2 norm already proves its gradient
        # elements finite, so rescanning every parameter on success is redundant.
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        total_norm = get_norm(gradients, norm_type=2.0, error_if_nonfinite=False)
    else:
        # Preserve support for older PyTorch releases lacking the split public
        # API. This compatibility path retains the original strict validation.
        invalid = _nonfinite_gradient_names(model)
        if invalid:
            raise FloatingPointError(
                "Non-finite gradients before clipping at "
                f"epoch={epoch}, step={step}, sample={sample_id}; "
                f"parameters={', '.join(invalid)}"
            )
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, norm_type=2.0, error_if_nonfinite=True
        )
    finite_norm = float(total_norm.detach().cpu())
    if not math.isfinite(finite_norm):
        invalid = _nonfinite_gradient_names(model)
        description = "gradients" if invalid else "gradient norm"
        names = f"; parameters={', '.join(invalid)}" if invalid else ""
        raise FloatingPointError(
            f"Non-finite {description} before clipping at "
            f"epoch={epoch}, step={step}, sample={sample_id}{names}"
        )
    if callable(get_norm) and callable(clip_with_norm):
        # No invalid gradient is modified, and unrelated backend failures from
        # either public operation propagate with their original exception type.
        clip_with_norm(parameters, max_norm, total_norm)
    return finite_norm


def _nonfinite_gradient_names(model: torch.nn.Module) -> list[str]:
    """Pack finite checks per device; synchronize once, not once per parameter."""
    by_device: dict[torch.device, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            gradient = parameter.grad
            values = gradient.coalesce().values() if gradient.is_sparse else gradient
            by_device[gradient.device].append((name, torch.isfinite(values).all()))
    invalid: list[str] = []
    for entries in by_device.values():
        flags = torch.stack([flag for _name, flag in entries]).detach().cpu().tolist()
        invalid.extend(name for (name, _flag), finite in zip(entries, flags) if not finite)
    return invalid


@torch.no_grad()
def _snapshot_model_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Pack same-shape buffers into shared storage without changing their values.

    The default model has 58 buffers but only six device/dtype/shape groups. One
    stack per group replaces one allocation/copy per buffer. Restored values are
    views into the packs; all buffers, including non-BatchNorm state, are retained.
    """
    groups: dict[
        tuple[torch.device, torch.dtype, torch.Size], list[tuple[str, torch.Tensor]]
    ] = defaultdict(list)
    snapshot: dict[str, torch.Tensor] = {}
    for name, value in model.named_buffers():
        if value.layout != torch.strided or value.is_quantized:
            # Stacking sparse or differently quantized buffers is not generally
            # value-preserving. Keep the existing clone behavior for these cases.
            snapshot[name] = value.detach().clone()
        else:
            groups[(value.device, value.dtype, value.shape)].append((name, value))
    for entries in groups.values():
        packed = torch.stack([value for _name, value in entries])
        snapshot.update(
            (name, value)
            for (name, _original), value in zip(entries, packed.unbind(), strict=True)
        )
    return snapshot


def _training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    forward_loss: Callable[[], tuple[torch.Tensor, dict[str, torch.Tensor], Any]],
    *,
    optimizer_mode: str,
    max_norm: float,
    epoch: int,
    step: int,
    sample_id: Any,
    max_amp_retries: int = _AMP_MAX_RETRIES,
    timing: Any = None,
) -> tuple[Any, dict[str, float], float, dict[str, float | int]]:
    """Commit one sample, retrying only recoverable AMP gradient overflows.

    ``forward_loss`` must use the same sample and incoming recurrent/temporal
    state on every call. Its payload is published only after a real optimizer
    update. Failed attempts never step the optimizer, never consume a sample,
    and restore model buffers (including BatchNorm counters) and all RNG state.
    """
    if (
        isinstance(max_amp_retries, bool)
        or not isinstance(max_amp_retries, int)
        or max_amp_retries < 0
    ):
        raise ValueError("max_amp_retries must be a nonnegative integer")
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("train.grad_clip must be finite and greater than zero")
    amp_enabled = bool(scaler.is_enabled())
    scale_before = float(scaler.get_scale())
    if not math.isfinite(scale_before) or scale_before <= 0:
        raise FloatingPointError(f"Invalid AMP scale before training step: {scale_before}")
    saved_buffers = _snapshot_model_buffers(model) if amp_enabled else {}
    saved_rng = _capture_rng_state() if amp_enabled else None

    def rollback() -> None:
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        if saved_rng is not None:
            with torch.no_grad():
                buffers = dict(model.named_buffers())
                for name, value in saved_buffers.items():
                    buffers[name].copy_(value)
            _restore_rng_state(saved_rng)

    retries = 0
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            loss, loss_parts, payload = forward_loss()
            loss_values = _ensure_finite_loss(
                loss, loss_parts, epoch=epoch, step=step, sample_id=sample_id
            )
            with timing.scope("backward") if timing is not None else nullcontext():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            with timing.scope("gradient_check") if timing is not None else nullcontext():
                invalid = _nonfinite_gradient_names(model)
            if invalid:
                attempt_scale = float(scaler.get_scale())
                context = (
                    f"epoch={epoch}, step={step}, sample={sample_id}, "
                    f"scale={attempt_scale}, retries={retries}/{max_amp_retries}; "
                    f"parameters={', '.join(invalid)}"
                )
                if not amp_enabled:
                    raise FloatingPointError(f"Non-finite gradients with AMP disabled: {context}")
                if retries >= max_amp_retries:
                    raise FloatingPointError(f"Persistent AMP gradient overflow: {context}")
                # unscale_ recorded the failed optimizer's inf checks. update()
                # consumes those checks, backs off, resets the growth tracker,
                # and clears its per-optimizer stage without stepping weights.
                scaler.update()
                next_scale = float(scaler.get_scale())
                if not math.isfinite(next_scale) or not 0 < next_scale < attempt_scale:
                    raise FloatingPointError(
                        f"AMP scale did not safely back off ({next_scale}): {context}"
                    )
                rollback()
                retries += 1
                # Release the failed graph/payload before allocating the retry.
                del loss, loss_parts, payload
                continue
            with timing.scope("gradient_check") if timing is not None else nullcontext():
                if optimizer_mode == "adam_gc":
                    _centralize_gradients(model)
                gradient_norm = _clip_and_validate_gradients(
                    model, max_norm, epoch=epoch, step=step, sample_id=sample_id
                )
        except Exception as error:
            try:
                rollback()
            except Exception as rollback_error:
                # A poisoned CUDA context can also reject buffer restoration.
                # Preserve the original failure instead of masking its cause.
                raise error from rollback_error
            raise
        # Only finite, centralized and clipped gradients reach the optimizer.
        with timing.scope("optimizer") if timing is not None else nullcontext():
            scaler.step(optimizer)
            scaler.update()
        return payload, loss_values, gradient_norm, {
            "scale_before": scale_before,
            "scale_after": float(scaler.get_scale()),
            "retries": retries,
        }


def _validation_dataset(config: dict[str, Any]):
    return build_dataset(config["dataset"], split="val")


def _dataset_transform_contract(config: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(config["dataset"])
    data.pop("root", None)
    data.pop("val_root", None)
    data.pop("split_manifest", None)
    return data


def _split_manifest_contract(config: dict[str, Any]) -> dict[str, Any] | None:
    manifest_path = config["dataset"].get("split_manifest")
    manifest = load_eventhdr_split_manifest(manifest_path) if manifest_path else None
    if manifest is None:
        return None
    return {
        "status": str(manifest.get("status", "missing")).strip().lower(),
        "split_schema": manifest["split_schema"],
        "group_semantics": manifest["group_semantics"],
        "train_files": manifest["train_files"],
        "val_files": manifest["val_files"],
        "file_to_group": manifest["file_to_group"],
    }


def _validation_protocol(
    config: dict[str, Any],
    val_sampling: dict[str, Any],
    train_dataset,
    val_dataset,
    digest_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    print("Verifying cached hashes or hashing train/validation files for exact resume...")
    return {
        "version": 7,
        "seed": int(config.get("seed", 2026)),
        "recurrent": bool(config["model"].get("recurrent", True)),
        "dataset_transform": _dataset_transform_contract(config),
        "split_manifest": _split_manifest_contract(config),
        "dataset_content": {
            "train": _dataset_content_fingerprint(train_dataset, digest_cache),
            "validation": _dataset_content_fingerprint(val_dataset, digest_cache),
        },
        "dataset_index": {
            "train": _dataset_index_contract(train_dataset),
            "validation": _dataset_index_contract(val_dataset),
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
    load_eventhdr_split_manifest(manifest_path)


@torch.no_grad()
def validate(
    model: ASGCNUNet,
    loader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
    score_positions: set[int] | None = None,
) -> dict[str, Any]:
    model.eval()
    accumulator = MetricAccumulator()
    current_sequence = None
    previous_sequence_index = None
    previous_sensor_size = None
    recurrent_state = None
    for index, batch in enumerate(loader):
        if max_samples is not None and index >= max_samples:
            break
        if len(batch) != 1:
            raise ValueError("Stateful validation currently requires batch_size=1")
        sample = move_sample(batch[0], device)
        sequence_id, sequence_index, sensor_size = _sample_sequence_info(sample)
        if not _continues_sequence(
            sequence_id,
            sequence_index,
            sensor_size,
            current_sequence,
            previous_sequence_index,
            previous_sensor_size,
        ):
            recurrent_state = None
        current_sequence = sequence_id
        previous_sequence_index = sequence_index
        previous_sensor_size = sensor_size
        prediction, diagnostics = model.forward_sample(sample, recurrent_state=recurrent_state)
        recurrent_state = diagnostics["recurrent_state"]
        if recurrent_state is not None:
            recurrent_state = recurrent_state.detach()
        if score_positions is None or index in score_positions:
            target = sample["target"].unsqueeze(0)
            accumulator.update(
                _sample_metric_scene(sample),
                sample["sample_id"],
                frame_metrics(prediction, target),
            )
    return accumulator.summary()


def train(config: dict[str, Any], resume_from: str | Path | None = None) -> Path:
    validate_experiment_config(config)
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
    batch_sampler = (
        SequenceBatchSampler(train_dataset, batch_size, seed=seed) if batch_size > 1 else None
    )
    train_loader = _data_loader(
        train_dataset,
        batch_size,
        int(train_config.get("num_workers", 0)),
        device,
        shuffle=False,
        batch_sampler=batch_sampler,
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
    configured_loss_weights = train_config.get("loss_weights") or {}
    temporal_weight = float(configured_loss_weights.get("temporal", 0.0))

    if resume_checkpoint is not None:
        if resume_checkpoint.get("model_config") != config["model"]:
            raise ValueError(
                "Exact resume requires config.model to match the checkpoint model_config"
            )
        if resume_checkpoint.get("preflight_gate") != config.get("preflight_gate"):
            raise ValueError(
                "Exact resume requires the same verified or explicitly bypassed "
                "preflight gate as the checkpoint"
            )
        if resume_checkpoint.get("validation_protocol") != validation_protocol:
            raise ValueError(
                "Resume validation protocol differs from the checkpoint. Keep the seed, "
                "dataset transforms, split manifest, validation sampling, and SSIM protocol fixed."
            )
        _validate_terminal_validation_resume(resume_checkpoint, training_protocol)
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
    public_config = _public_config(config)

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
    timing_steps = int(train_config.get("timing_steps", 0))
    timing = StageTimer(
        device, enabled=timing_steps > 0,
        warmup_steps=int(train_config.get("timing_warmup", 10)),
        measurement_steps=max(1, timing_steps),
    )
    timing_saved = False
    # Publish run metadata only after every fresh/resume check, including
    # optimizer/scaler/RNG restoration. Rejected attempts must preserve the
    # previous gate, config and hash cache alongside the original checkpoints.
    save_json(hash_cache_path, {"version": 1, "files": digest_cache})
    save_json(run_dir / "config.json", public_config)
    if "preflight_gate" in config:
        save_json(run_dir / "preflight_gate.json", config["preflight_gate"])
    for epoch in range(start_epoch, epochs + 1):
        epoch_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
        model.train()
        _reset_cuda_peak_memory(device)
        state = TrainingState(independent_sequences=batch_size > 1)
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        running_loss = 0.0
        epoch_amp_retries = 0
        epoch_amp_retried_samples = 0
        seen = 0
        optimizer_steps = 0
        epoch_start = time.perf_counter()
        loader_iterator = iter(train_loader)
        frame_total = len(train_dataset)
        if max_train_samples is not None:
            frame_total = min(frame_total, int(max_train_samples))
        progress = tqdm(total=frame_total, desc=f"train {epoch:03d}/{epochs:03d}", unit="frame")
        for step in range(len(train_loader)):
            if max_train_samples is not None and seen >= int(max_train_samples):
                break
            with timing.scope("dataload", gpu=False):
                batch = next(loader_iterator)
                if max_train_samples is not None:
                    batch = batch[: int(max_train_samples) - seen]
            with timing.scope("transfer"):
                samples = [move_sample(sample, device) for sample in batch]
            contexts = state.prepare(samples)

            def forward_loss(current_samples=samples, incoming_contexts=contexts):
                return forward_training_loss(
                    model, criterion, current_samples, incoming_contexts,
                    batch_mode=batch_size > 1, amp_enabled=amp_enabled,
                    temporal_weight=temporal_weight, timing=timing,
                )

            payload, loss_values, _gradient_norm, amp_info = _training_step(
                model,
                optimizer,
                scaler,
                forward_loss,
                optimizer_mode=optimizer_mode,
                max_norm=float(train_config.get("grad_clip", 1.0)),
                epoch=epoch,
                step=step,
                sample_id=[sample.get("sample_id", "unknown") for sample in samples],
                timing=timing,
            )
            prediction, diagnostics, target = payload
            epoch_amp_retries += int(amp_info["retries"])
            epoch_amp_retried_samples += len(samples) * int(amp_info["retries"] > 0)
            state.commit(samples, prediction, diagnostics, target)
            if batch_sampler is not None:
                state.release_finished(samples, batch_sampler.final_sequence_indices)
                if len(state.values) > batch_size:
                    raise RuntimeError("Sequence state storage exceeded the active batch size")
            running_loss += loss_values["total"] * len(samples)
            seen += len(samples)
            optimizer_steps += 1
            progress.update(len(samples))
            if timing.step():
                save_json(run_dir / "timing.json", timing.collect())
                timing_saved = True
            if step % int(train_config.get("log_every", 20)) == 0 or amp_info["retries"]:
                progress.set_postfix(
                    loss=f"{running_loss / max(seen, 1):.4f}",
                    amp_retries=epoch_amp_retries,
                    amp_scale=amp_info["scale_after"],
                    **loss_values,
                )
        progress.close()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start
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
            "performance": {
                "training_seconds": epoch_seconds,
                "frames": seen,
                "optimizer_steps": optimizer_steps,
                "frames_per_second": seen / epoch_seconds if epoch_seconds > 0 else None,
                "batch_size_limit": batch_size,
                "includes_validation": False,
                "timing_instrumentation_requested": timing_steps > 0,
            },
            "amp": {
                "retries": epoch_amp_retries,
                "retried_samples": epoch_amp_retried_samples,
                "scale": float(scaler.get_scale()),
            },
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
            "config": public_config,
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
            "preflight_gate": copy.deepcopy(config.get("preflight_gate")),
            "validation_protocol": validation_protocol,
            "training_protocol": training_protocol,
            "terminal_validation_state": (
                {
                    "planned_epoch": epochs,
                    "completed": bool(should_validate and epoch == epochs),
                    "completed_epoch": epoch if should_validate and epoch == epochs else None,
                }
                if validate_every is None
                else None
            ),
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
                "preflight_gate": checkpoint["preflight_gate"],
                "training_config": checkpoint["config"],
                "validation_protocol": checkpoint["validation_protocol"],
                "training_protocol": checkpoint["training_protocol"],
                "terminal_validation_state": checkpoint["terminal_validation_state"],
            }
            atomic_torch_save(best_checkpoint, run_dir / "best.pt")
        atomic_torch_save(checkpoint, run_dir / "last.pt")
        print(record)
    if timing_steps > 0 and not timing_saved:
        save_json(run_dir / "timing.json", timing.collect())
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
    *,
    allow_unsealed_checkpoint_for_non_reporting: bool = False,
) -> dict[str, Any]:
    _validate_snn_request(inference_mode, simulation_steps)
    validate_experiment_config(config)
    set_seed(int(config.get("seed", 2026)))
    device = resolve_device(config.get("device", "auto"))
    dataset = build_dataset(config["dataset"], split="eval")
    try:
        return _evaluate_dataset(
            config,
            checkpoint_path,
            dataset,
            device,
            inference_mode,
            simulation_steps,
            snn_dynamics,
            allow_unsealed_checkpoint_for_non_reporting=(
                allow_unsealed_checkpoint_for_non_reporting
            ),
        )
    finally:
        if hasattr(dataset, "close"):
            dataset.close()


def _evaluate_dataset(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    dataset,
    device: torch.device,
    inference_mode: str,
    simulation_steps: int,
    snn_dynamics: str | None,
    *,
    allow_unsealed_checkpoint_for_non_reporting: bool,
) -> dict[str, Any]:
    eval_config = config.get("eval", {})
    max_samples = eval_config.get("max_samples")
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
    checkpoint_contract, report_eligible, report_ineligible_reasons = (
        _reporting_checkpoint_contract(
            checkpoint,
            checkpoint_path,
            inference_mode,
            allow_unsealed_checkpoint_for_non_reporting=(
                allow_unsealed_checkpoint_for_non_reporting
            ),
        )
    )
    quality_reasons = _reporting_dataset_coverage_reasons(config, dataset)
    if max_samples is not None:
        quality_reasons.append(
            "quality evaluation uses eval.max_samples; reporting requires "
            "eval.max_samples=null and the complete evaluation dataset"
        )
    report_eligible, report_ineligible_reasons = _merge_reporting_reasons(
        report_eligible,
        report_ineligible_reasons,
        quality_reasons,
        allow_non_reporting=allow_unsealed_checkpoint_for_non_reporting,
        scope="Quality evaluation",
    )
    _set_inference_snn_dynamics(model, inference_mode, snn_dynamics)
    model.eval()
    precision, autocast_dtype = _inference_precision(eval_config, device, model)
    lpips_model = _maybe_lpips(bool(eval_config.get("lpips", False)), device)
    _reset_cuda_peak_memory(device)
    accumulator = MetricAccumulator()
    frame_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    realtime_factors: list[float] = []
    current_sequence = None
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
    evaluation_count = min(
        len(dataset),
        len(dataset) if max_samples is None else int(max_samples),
    )
    evaluation_sampling = _sampling_summary(dataset, list(range(evaluation_count)))
    evaluation_dataset = _evaluation_dataset_provenance(
        config, dataset, evaluation_sampling
    )
    if config["dataset"].get("type") == "eventhdr":
        report_eligible, report_ineligible_reasons = _merge_reporting_reasons(
            report_eligible,
            report_ineligible_reasons,
            _eventhdr_evaluation_lineage_reasons(
                checkpoint,
                inference_mode,
                evaluation_dataset,
            ),
            allow_non_reporting=allow_unsealed_checkpoint_for_non_reporting,
            scope="EventHDR quality evaluation",
        )
    evaluation_protocol = _reporting_protocol(
        kind="quality_evaluation",
        config=config,
        dataset=dataset,
        sampling=evaluation_sampling,
        checkpoint_contract=checkpoint_contract,
        report_eligible=report_eligible,
        report_ineligible_reasons=report_ineligible_reasons,
        device=device,
        precision=precision,
        execution={
            "inference_mode": inference_mode,
            "simulation_steps": simulation_steps if inference_mode == "snn" else None,
            "snn_dynamics": (
                model.snn_dynamics if inference_mode == "snn" else None
            ),
            "scope": "full_dataset_quality_evaluation",
        },
        evaluation_dataset=evaluation_dataset,
    )
    saved = 0
    prediction_stems: set[str] = set()
    for index, batch in enumerate(tqdm(loader, desc=f"evaluate-{inference_mode}")):
        if max_samples is not None and index >= int(max_samples):
            break
        sample = move_sample(batch[0], device)
        sample_id = sample.get("sample_id", index)
        _require_finite_tensor(sample["target"], "target", sample_id)
        sequence_id, sequence_index, sensor_size = _sample_sequence_info(sample)
        if not _continues_sequence(
            sequence_id,
            sequence_index,
            sensor_size,
            current_sequence,
            previous_sequence_index,
            previous_sensor_size,
        ):
            recurrent_state = None
            previous_prediction = None
            previous_target = None
        current_sequence = sequence_id
        previous_sequence_index = sequence_index
        previous_sensor_size = sensor_size
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with _inference_precision_context(device, precision, autocast_dtype):
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
        if not math.isfinite(latency_ms) or latency_ms <= 0:
            raise FloatingPointError(f"Invalid latency: sample={sample_id}")
        _require_finite_tensor(prediction, "prediction", sample_id)
        _require_finite_structure(diagnostics, "diagnostics", sample_id)
        recurrent_state = diagnostics["recurrent_state"]
        if recurrent_state is not None:
            recurrent_state = recurrent_state.detach()
        target = sample["target"].unsqueeze(0).float()
        metric_prediction = prediction.float()
        temporal_tensor = None
        if previous_prediction is not None and previous_target is not None:
            temporal_tensor = temporal_consistency_error(
                metric_prediction,
                previous_prediction,
                target,
                previous_target,
            )
        metrics = frame_metrics(
            metric_prediction,
            target,
            lpips_model,
            extra_metrics=(
                {"temporal_l1": temporal_tensor} if temporal_tensor is not None else None
            ),
        )
        _require_finite_structure(metrics, "metrics", sample_id)
        temporal_l1 = metrics.get("temporal_l1")
        previous_prediction = metric_prediction.detach()
        previous_target = target.detach()
        metric_scene = _sample_metric_scene(sample)
        accumulator.update(metric_scene, sample["sample_id"], metrics)
        dt_us = _positive_interval_us(sample["metadata"].get("dt_us"), sample_id)
        rtf = latency_ms / (dt_us / 1000.0) if dt_us is not None else None
        if rtf is not None:
            realtime_factors.append(rtf)
        row = {
            "scene": metric_scene,
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

    if len(frame_rows) != evaluation_count:
        raise RuntimeError(
            "Quality evaluation did not process its complete declared sampling: "
            f"expected {evaluation_count}, observed {len(frame_rows)}"
        )
    quality = accumulator.summary()
    latency = _latency_summary(latencies)
    latency["deadline_miss_ratio"] = (
        sum(value > 1.0 for value in realtime_factors) / len(realtime_factors)
        if realtime_factors
        else None
    )
    latency["rtf_p95"] = percentile(realtime_factors, 0.95) if realtime_factors else None
    result = {
        "report_eligible": report_eligible,
        "report_ineligible_reasons": report_ineligible_reasons,
        "evaluation_protocol": evaluation_protocol,
        "dataset": config["dataset"]["type"],
        "dataset_coverage": _dataset_coverage_summary(dataset, config["dataset"]),
        "checkpoint": _artifact_path_label(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_model_sha256": checkpoint.get("model_state_sha256"),
        "output_dir": _artifact_path_label(output_dir),
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
        "precision": precision,
    }
    _require_finite_structure(result, "evaluation", "summary")
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


def _evaluation_dataset_transform_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Return data semantics without host paths or manifest locations."""
    data = copy.deepcopy(config["dataset"])
    for key in ("root", "val_root", "split_manifest", "file_manifest"):
        data.pop(key, None)
    return data


def _evaluation_manifest_contract(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    result: dict[str, Any] = {"split": None, "file": None}
    if dataset.get("split_manifest"):
        result["split"] = _split_manifest_contract(config)
    file_manifest = dataset.get("file_manifest")
    if file_manifest:
        result["file"] = _hashed_contract(_public_config(load_json(file_manifest)))
    return result


def _reporting_dataset_coverage_reasons(
    config: dict[str, Any], dataset
) -> list[str]:
    """Require an explicit, exact file-set commitment for reportable runs."""
    data_config = config["dataset"]
    coverage = _dataset_coverage_summary(dataset, data_config)
    reasons: list[str] = []
    expected = data_config.get("expected_file_count")
    if (
        not isinstance(expected, int)
        or isinstance(expected, bool)
        or expected < 1
        or coverage.get("expected_file_count") != expected
        or coverage.get("file_count") != expected
        or coverage.get("complete") is not True
    ):
        reasons.append("evaluation dataset has no exact expected-file-count commitment")

    dataset_type = data_config.get("type")
    if dataset_type == "eventhdr":
        try:
            manifest = _split_manifest_contract(config)
        except (OSError, KeyError, TypeError, ValueError):
            manifest = None
        if not _valid_split_manifest_contract(manifest):
            reasons.append("EventHDR evaluation has no valid final split manifest")
        elif coverage.get("files") != sorted(manifest["val_files"]):
            reasons.append("EventHDR evaluation files differ from the final manifest")
    elif dataset_type == "eventaid_r_zip":
        manifest_path = data_config.get("file_manifest")
        try:
            manifest = load_json(manifest_path) if manifest_path else None
        except (OSError, TypeError, ValueError):
            manifest = None
        entries = manifest.get("files") if isinstance(manifest, dict) else None
        expected_files = (
            sorted(
                f"{item['scene']}.zip"
                for item in entries
                if isinstance(item, dict)
                and isinstance(item.get("scene"), str)
                and item["scene"].strip()
            )
            if isinstance(entries, list)
            else []
        )
        if not expected_files or len(expected_files) != len(set(expected_files)):
            reasons.append("EventAid-R evaluation has no valid fixed file manifest")
        elif coverage.get("files") != expected_files:
            reasons.append("EventAid-R evaluation files differ from the fixed manifest")
    else:
        reasons.append("evaluation dataset type is unsupported for reporting")
    return reasons


def _eventhdr_evaluation_lineage_reasons(
    checkpoint: dict[str, Any],
    inference_mode: str,
    evaluation_dataset: dict[str, Any],
) -> list[str]:
    contract = evaluation_dataset.get("contract")
    if not isinstance(contract, dict):
        return ["EventHDR evaluation dataset identity is invalid"]
    if inference_mode == "ann":
        validation = checkpoint.get("validation_protocol")
    else:
        calibration = checkpoint.get("calibration_protocol")
        identity = (
            calibration.get("source_ann_validation_protocol")
            if isinstance(calibration, dict)
            else None
        )
        validation = identity.get("contract") if isinstance(identity, dict) else None
    if not isinstance(validation, dict):
        return ["EventHDR evaluation has no source ANN validation lineage"]

    reasons: list[str] = []
    dataset_content = validation.get("dataset_content")
    expected_content = (
        dataset_content.get("validation")
        if isinstance(dataset_content, dict)
        else None
    )
    if contract.get("content") != expected_content:
        reasons.append("EventHDR evaluation content differs from source ANN validation data")

    expected_transform = copy.deepcopy(validation.get("dataset_transform"))
    current_transform = copy.deepcopy(contract.get("transform"))
    if isinstance(expected_transform, dict):
        expected_transform.pop("expected_file_count", None)
    if isinstance(current_transform, dict):
        current_transform.pop("expected_file_count", None)
    if current_transform != expected_transform:
        reasons.append("EventHDR evaluation transform differs from source ANN validation")

    manifest_identity = contract.get("manifest")
    split_identity = (
        manifest_identity.get("split")
        if isinstance(manifest_identity, dict)
        else None
    )
    current_manifest = (
        split_identity.get("contract")
        if isinstance(split_identity, dict) and "contract" in split_identity
        else split_identity
    )
    if current_manifest != validation.get("split_manifest"):
        reasons.append("EventHDR evaluation manifest differs from source ANN validation")
    return reasons


def _evaluation_dataset_provenance(
    config: dict[str, Any],
    dataset,
    sampling: dict[str, Any],
) -> dict[str, Any]:
    """Bind evaluation data exactly, reusing a host-local path-token hash cache."""
    eval_config = config.get("eval", {})
    rehash = eval_config.get("rehash_data", False)
    if not isinstance(rehash, bool):
        raise TypeError("eval.rehash_data must be a boolean")
    output_base = Path(eval_config.get("output_dir", "runs/evaluation"))
    cache_path = output_base / ".data_hash_cache.json"
    digest_cache = _load_data_hash_cache(cache_path, rehash)
    content = _dataset_content_fingerprint(dataset, digest_cache)
    # The cache contains only SHA-256 path tokens, stat signatures and file digests.
    # save_json is strict and atomic, so a killed writer cannot poison later runs.
    save_json(cache_path, {"version": 1, "files": digest_cache})
    contract = {
        "content": content,
        "transform": _evaluation_dataset_transform_contract(config),
        "manifest": _evaluation_manifest_contract(config),
        "coverage": _dataset_coverage_summary(dataset, config["dataset"]),
        "sampling": sampling,
    }
    return _hashed_contract(contract)


def _evaluation_runtime_contract(
    device: torch.device,
    precision: dict[str, Any],
) -> dict[str, Any]:
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        compute_capability = list(torch.cuda.get_device_capability(device_index))
    else:
        device_index = None
        gpu_name = None
        compute_capability = None
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "device_type": device.type,
        "device_index": device_index,
        "cuda_runtime": torch.version.cuda if device.type == "cuda" else None,
        "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "precision": copy.deepcopy(precision),
    }


def _reporting_protocol(
    *,
    kind: str,
    config: dict[str, Any],
    dataset,
    sampling: dict[str, Any],
    checkpoint_contract: dict[str, Any],
    report_eligible: bool,
    report_ineligible_reasons: list[str],
    device: torch.device,
    precision: dict[str, Any],
    execution: dict[str, Any],
    evaluation_dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    public_config = _public_config(config)
    protocol: dict[str, Any] = {
        "schema": "asgcn_reporting_protocol_v1",
        "kind": kind,
        "report_eligible": report_eligible,
        "report_ineligible_reasons": list(report_ineligible_reasons),
        "public_config": _hashed_contract(public_config),
        "model_config": _hashed_contract(config["model"]),
        "checkpoint": checkpoint_contract,
        "evaluation_dataset": (
            evaluation_dataset
            if evaluation_dataset is not None
            else _evaluation_dataset_provenance(config, dataset, sampling)
        ),
        "execution": _hashed_contract(execution),
        "source": _hashed_contract(_current_source_contract()),
        "runtime": _hashed_contract(_evaluation_runtime_contract(device, precision)),
        "precision": _hashed_contract(precision),
    }
    protocol["protocol_sha256"] = _canonical_sha256(protocol)
    return protocol


def _reset_benchmark_measurement_window(
    device: torch.device,
    release_warmup_references: Callable[[], None],
) -> None:
    """Start peak-memory accounting after fully releasing warmup state."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    release_warmup_references()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


@torch.no_grad()
def benchmark(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    warmup: int = 10,
    steps: int = 100,
    inference_mode: str = "ann",
    simulation_steps: int = 16,
    snn_dynamics: str | None = None,
    *,
    allow_unsealed_checkpoint_for_non_reporting: bool = False,
) -> dict[str, Any]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    _validate_snn_request(inference_mode, simulation_steps)
    validate_experiment_config(config)
    device = resolve_device(config.get("device", "auto"))
    dataset = build_dataset(config["dataset"], split="eval")
    try:
        return _benchmark_dataset(
            config,
            checkpoint_path,
            dataset,
            device,
            warmup,
            steps,
            inference_mode,
            simulation_steps,
            snn_dynamics,
            allow_unsealed_checkpoint_for_non_reporting=(
                allow_unsealed_checkpoint_for_non_reporting
            ),
        )
    finally:
        if hasattr(dataset, "close"):
            dataset.close()


def _benchmark_dataset(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    dataset,
    device: torch.device,
    warmup: int,
    steps: int,
    inference_mode: str,
    simulation_steps: int,
    snn_dynamics: str | None,
    *,
    allow_unsealed_checkpoint_for_non_reporting: bool,
) -> dict[str, Any]:
    model, checkpoint = load_model_checkpoint(checkpoint_path, device, config["model"])
    _validate_snn_request(inference_mode, simulation_steps, checkpoint, checkpoint_path)
    checkpoint_contract, report_eligible, report_ineligible_reasons = (
        _reporting_checkpoint_contract(
            checkpoint,
            checkpoint_path,
            inference_mode,
            allow_unsealed_checkpoint_for_non_reporting=(
                allow_unsealed_checkpoint_for_non_reporting
            ),
        )
    )
    report_eligible, report_ineligible_reasons = _merge_reporting_reasons(
        report_eligible,
        report_ineligible_reasons,
        _reporting_dataset_coverage_reasons(config, dataset),
        allow_non_reporting=allow_unsealed_checkpoint_for_non_reporting,
        scope="Compute benchmark",
    )
    _set_inference_snn_dynamics(model, inference_mode, snn_dynamics)
    model.eval()
    eval_config = config.get("eval", {})
    precision, autocast_dtype = _inference_precision(eval_config, device, model)
    benchmark_base = Path(eval_config.get("output_dir", "runs/evaluation"))
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
    current_sequence = None
    previous_sequence_index = None
    previous_sensor_size = None
    raw = None
    sample = None
    prediction = None
    diagnostics = None

    def release_warmup_references() -> None:
        nonlocal raw, sample, prediction, diagnostics, recurrent_state
        raw = None
        sample = None
        prediction = None
        diagnostics = None
        recurrent_state = None

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
    benchmark_sampling = {
        "measured": _sampling_summary(dataset, measured_indices),
        "warmup": _sampling_summary(dataset, warmup_indices),
        "forward_schedule": [
            {
                "scored": measured,
                **_dataset_sample_identity(dataset, sample_index),
            }
            for measured, sample_index in schedule
        ],
        "recurrent_context_policy": (
            "full_group_prefix"
            if recurrent and benchmark_context_frames is None
            else "bounded_predecessor"
            if recurrent
            else None
        ),
        "max_recurrent_context_frames_per_group": (
            benchmark_context_frames if recurrent else 0
        ),
        "recurrent_context_frames": context_frames,
    }
    evaluation_dataset = _evaluation_dataset_provenance(
        config, dataset, benchmark_sampling
    )
    if config["dataset"].get("type") == "eventhdr":
        report_eligible, report_ineligible_reasons = _merge_reporting_reasons(
            report_eligible,
            report_ineligible_reasons,
            _eventhdr_evaluation_lineage_reasons(
                checkpoint,
                inference_mode,
                evaluation_dataset,
            ),
            allow_non_reporting=allow_unsealed_checkpoint_for_non_reporting,
            scope="EventHDR compute benchmark",
        )
    benchmark_protocol = _reporting_protocol(
        kind="compute_benchmark",
        config=config,
        dataset=dataset,
        sampling=benchmark_sampling,
        checkpoint_contract=checkpoint_contract,
        report_eligible=report_eligible,
        report_ineligible_reasons=report_ineligible_reasons,
        device=device,
        precision=precision,
        execution={
            "inference_mode": inference_mode,
            "simulation_steps": simulation_steps if inference_mode == "snn" else None,
            "snn_dynamics": (
                model.snn_dynamics if inference_mode == "snn" else None
            ),
            "warmup_steps": warmup,
            "measured_steps": steps,
            "timer_scope": "model_forward_including_graph_excluding_data_and_h2d",
            "memory_scope": "model_and_events_excluding_ground_truth_target",
        },
        evaluation_dataset=evaluation_dataset,
    )
    for iteration, (measured, sample_index) in enumerate(schedule):
        if iteration == len(warmup_indices):
            current_sequence = None
            previous_sequence_index = None
            previous_sensor_size = None
            _reset_benchmark_measurement_window(
                device,
                release_warmup_references,
            )
        raw = dataset[sample_index]  # I/O intentionally outside the timer.
        sample = move_inference_sample(raw, device)
        sample_id = sample.get("sample_id", sample_index)
        _require_finite_tensor(sample["target"], "target", sample_id)
        sequence_id, sequence_index, sensor_size = _sample_sequence_info(sample)
        continuation = _continues_sequence(
            sequence_id,
            sequence_index,
            sensor_size,
            current_sequence,
            previous_sequence_index,
            previous_sensor_size,
        )
        if not continuation:
            recurrent_state = None
            if measured:
                measured_state_resets += 1
        current_sequence = sequence_id
        previous_sequence_index = sequence_index
        previous_sensor_size = sensor_size
        with _inference_precision_context(device, precision, autocast_dtype):
            if measured:
                if cuda_start is not None:
                    cuda_start.record()
                else:
                    start = time.perf_counter()
            prediction, diagnostics = model.forward_sample(
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
                if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                    raise FloatingPointError(f"Invalid latency: sample={sample_id}")
        _require_finite_tensor(prediction, "prediction", sample_id)
        _require_finite_structure(diagnostics, "diagnostics", sample_id)
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
            dt_us = (
                _positive_interval_us(metadata.get("dt_us"), sample_id)
                if isinstance(metadata, dict)
                else None
            )
            if dt_us is not None:
                realtime_factors.append(elapsed_ms / (dt_us / 1000.0))
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
        "report_eligible": report_eligible,
        "report_ineligible_reasons": report_ineligible_reasons,
        "benchmark_protocol": benchmark_protocol,
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
        "checkpoint": _artifact_path_label(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_model_sha256": checkpoint.get("model_state_sha256"),
        "precision": precision,
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
    result["output_path"] = _artifact_path_label(benchmark_path)
    _require_finite_structure(result, "benchmark", "summary")
    save_json(benchmark_path, result)
    return result


def _sealed_calibration_protocol(
    config: dict[str, Any],
    dataset,
    calibration_indices: list[int],
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    device: torch.device,
    *,
    allow_unsealed: bool,
) -> dict[str, Any]:
    validation_protocol = checkpoint.get("validation_protocol")
    training_protocol = checkpoint.get("training_protocol")
    current_transform = _dataset_transform_contract(config)
    current_manifest = _split_manifest_contract(config)
    current_content = _dataset_content_fingerprint(dataset)
    current_source = _current_source_contract()
    mismatches: list[str] = []

    if calibration_indices != list(range(len(dataset))):
        mismatches.append(
            "reporting calibration requires every EventHDR training sample exactly once"
        )

    for reason in _ann_reporting_reasons(checkpoint):
        mismatches.append(f"source ANN reporting protocol: {reason}")

    if checkpoint.get("checkpoint_type") != "ann_inference":
        mismatches.append("source checkpoint is not clean ann_inference")
    if not isinstance(validation_protocol, dict):
        mismatches.append("source checkpoint has no validation_protocol")
        expected_content = None
        expected_transform = None
        expected_manifest = None
    else:
        if validation_protocol.get("version") != 7:
            mismatches.append("source checkpoint has an unsupported validation protocol")
        dataset_content = validation_protocol.get("dataset_content")
        expected_content = (
            dataset_content.get("train") if isinstance(dataset_content, dict) else None
        )
        expected_transform = validation_protocol.get("dataset_transform")
        expected_manifest = validation_protocol.get("split_manifest")
        if expected_content != current_content:
            mismatches.append("calibration data differs from the training dataset content")
        if expected_transform != current_transform:
            mismatches.append("calibration dataset transform differs from training")
        if expected_manifest != current_manifest:
            mismatches.append("calibration split manifest differs from training")

    expected_source = (
        training_protocol.get("source") if isinstance(training_protocol, dict) else None
    )
    if not isinstance(training_protocol, dict):
        mismatches.append("source checkpoint has no training protocol")
    elif training_protocol.get("version") not in {5, 6}:
        mismatches.append("source checkpoint has an unsupported training protocol")
    if not isinstance(expected_source, dict):
        mismatches.append("source checkpoint has no training source contract")
    elif expected_source != current_source:
        mismatches.append("current executable source differs from training")
    terminal_validation = (
        training_protocol.get("terminal_validation")
        if isinstance(training_protocol, dict)
        else None
    )
    if terminal_validation is not None:
        terminal_state = checkpoint.get("terminal_validation_state")
        planned_epoch = (
            terminal_validation.get("planned_epoch")
            if isinstance(terminal_validation, dict)
            else None
        )
        if (
            not isinstance(terminal_state, dict)
            or not bool(terminal_state.get("completed"))
            or terminal_state.get("completed_epoch") != planned_epoch
            or checkpoint.get("epoch") != planned_epoch
        ):
            mismatches.append("source ANN did not complete its sealed final-only validation")

    if allow_unsealed:
        mismatches.insert(0, "explicit unsealed calibration override requested")

    if mismatches and not allow_unsealed:
        raise ValueError(
            "Calibration contract is not sealed: "
            + "; ".join(mismatches)
            + ". Use --allow-unsealed-calibration only for non-reporting tests."
        )

    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        compute_capability = list(torch.cuda.get_device_capability(device_index))
    else:
        gpu_name = None
        compute_capability = None
    source_ann_reporting = _source_ann_reporting_contract(checkpoint)
    return {
        "version": 2,
        "sealed": not mismatches,
        "unsealed_reasons": mismatches,
        "source_checkpoint_type": checkpoint.get("checkpoint_type"),
        "source_checkpoint": _artifact_path_label(checkpoint_path),
        "source_ann_model_sha256": checkpoint.get("model_state_sha256"),
        "source_ann_checkpoint_sha256": _file_sha256(checkpoint_path),
        "source_ann_reporting_contract": _hashed_contract(source_ann_reporting),
        "source_ann_reporting_contract_sha256": _canonical_sha256(
            source_ann_reporting
        ),
        "source_ann_training_protocol": _hashed_contract(
            _public_config(training_protocol)
        ),
        "source_ann_training_protocol_sha256": _canonical_sha256(
            _public_config(training_protocol)
        ),
        "source_ann_validation_protocol": _hashed_contract(
            _public_config(validation_protocol)
        ),
        "source_ann_validation_protocol_sha256": _canonical_sha256(
            _public_config(validation_protocol)
        ),
        "source_ann_training_config": _hashed_contract(
            _public_config(checkpoint.get("training_config"))
        ),
        "source_ann_training_config_sha256": _canonical_sha256(
            _public_config(checkpoint.get("training_config"))
        ),
        "source_preflight_gate": _hashed_contract(
            _public_config(checkpoint.get("preflight_gate"))
        ),
        "source_epoch": checkpoint.get("epoch"),
        "dataset_content": current_content,
        "dataset_content_sha256": current_content["sha256"],
        "dataset_transform": current_transform,
        "split_manifest": current_manifest,
        "selected_sample_ids": [
            _dataset_sample_identity(dataset, index) for index in calibration_indices
        ],
        "training_source": expected_source,
        "calibration_source": current_source,
        "runtime": {
            "device": str(device),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda if device.type == "cuda" else None,
            "gpu_name": gpu_name,
            "compute_capability": compute_capability,
        },
    }


@torch.no_grad()
def calibrate(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path,
    samples: int | None = None,
    overwrite: bool = False,
    allow_unsealed_calibration: bool = False,
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
    validate_experiment_config(config)
    _enforce_training_split_status(config)
    device = resolve_device(config.get("device", "auto"))
    data_config = copy.deepcopy(config["dataset"])
    # Calibration is restricted to EventHDR train, never EventAid-R.
    if data_config["type"] != "eventhdr":
        raise ValueError("SNN calibration must use EventHDR training data")
    dataset = build_dataset(data_config, split="calibration")
    calibration_loader = None
    try:
        model, checkpoint = load_model_checkpoint(checkpoint_path, device, config["model"])
        model.eval()
        model.fold_batch_norm()
        model.reset_activation_maxima()
        calibration_limit = len(dataset) if samples is None else min(int(samples), len(dataset))
        calibration_indices = _balanced_sample_indices(
            dataset, calibration_limit, seed=int(config.get("seed", 2026))
        )
        calibration_protocol = _sealed_calibration_protocol(
            config,
            dataset,
            calibration_indices,
            checkpoint_path,
            checkpoint,
            device,
            allow_unsealed=allow_unsealed_calibration,
        )
        loader_config = config.get("train") or config.get("eval") or {}
        calibration_loader = _data_loader(
            Subset(dataset, calibration_indices),
            batch_size=1,
            num_workers=int(loader_config.get("num_workers", 0)),
            device=device,
            shuffle=False,
            **_loader_kwargs(loader_config),
        )
        calibration_sampling = _sampling_summary(dataset, calibration_indices)
        for batch in tqdm(
            calibration_loader,
            total=len(calibration_indices),
            desc="calibrate-SNN",
        ):
            if len(batch) != 1:
                raise ValueError("SNN calibration requires batch_size=1")
            sample = move_sample(batch[0], device)
            model.calibrate_sample(sample, momentum=-1.0)
        summary_core = _calibration_summary_commitment_core(
            model.calibration_summary()
        )
        model.seal_calibration_commitment(
            len(calibration_indices),
            _calibration_commitment_sha256(
                calibration_protocol,
                len(calibration_indices),
                calibration_sampling,
                summary_core,
            ),
        )
        calibration_summary = model.calibration_summary()
        model.apply_parameter_normalization()
        model_state = model.state_dict()
        inference_checkpoint = {
            "checkpoint_type": "snn_inference",
            "model": model_state,
            "model_state_sha256": _model_state_sha256(model_state),
            "model_config": checkpoint.get("model_config", config["model"]),
            "epoch": checkpoint.get("epoch"),
            "source_checkpoint": _artifact_path_label(checkpoint_path),
            "batch_norm_folded": True,
            "snn_calibrated": True,
            "paper_core_version": PAPER_CORE_VERSION,
            "parameter_normalized": True,
            "snn_calibration_samples": len(calibration_indices),
            "snn_calibration_valid_samples": calibration_summary["minimum_valid_samples"],
            "snn_calibration_summary": calibration_summary,
            "snn_calibration_sampling": calibration_sampling,
            "calibration_protocol": calibration_protocol,
            "preflight_gate": _public_config(checkpoint.get("preflight_gate")),
            "report_eligible": bool(calibration_protocol["sealed"]),
            "report_ineligible_reasons": list(calibration_protocol["unsealed_reasons"]),
        }
    finally:
        calibration_loader = None
        if hasattr(dataset, "close"):
            dataset.close()
    atomic_torch_save(inference_checkpoint, output_path)
    return output_path
