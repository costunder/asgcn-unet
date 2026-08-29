from __future__ import annotations

import copy
import hashlib
import math
import random
import statistics
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


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    fallback_model_config: dict[str, Any],
) -> tuple[ASGCNReconstructor, dict[str, Any]]:
    checkpoint = _load_checkpoint(checkpoint_path)
    model_config = checkpoint.get("model_config", fallback_model_config)
    model = build_model(model_config).to(device)
    if "model" in checkpoint:
        state = checkpoint.pop("model")
        metadata = checkpoint
    else:
        state = checkpoint
        metadata = {}
    model.load_state_dict(state, strict=True)
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


def _balanced_sample_indices(
    dataset, limit: int | None, seed: int = 2026
) -> list[int]:
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
            0
            if max_context_frames is None
            else max(0, first_position - int(max_context_frames))
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
    available_counts = Counter(
        _dataset_group_key(dataset, index) for index in range(len(dataset))
    )
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
    if best_checkpoint.get("validation_protocol") != resume_checkpoint.get(
        "validation_protocol"
    ):
        raise ValueError("Historical best.pt has a different validation protocol")
    if best_checkpoint.get("model_config") != resume_checkpoint.get("model_config"):
        raise ValueError("Historical best.pt has a different model configuration")
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


def _validate_snn_request(
    inference_mode: str,
    simulation_steps: int,
    checkpoint: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
) -> None:
    if inference_mode != "snn":
        return
    if int(simulation_steps) < 1:
        raise ValueError("simulation_steps must be at least 1 for SNN inference")
    if checkpoint is None:
        return
    calibration_samples = int(checkpoint.get("snn_calibration_samples", 0) or 0)
    if not bool(checkpoint.get("batch_norm_folded")) or calibration_samples < 1:
        location = f" {checkpoint_path}" if checkpoint_path is not None else ""
        raise ValueError(
            f"SNN inference requires a calibrated checkpoint;{location} is missing "
            "batch_norm_folded=true or snn_calibration_samples>=1. Run calibrate first."
        )


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


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


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
    manifest_identity = (
        {
            "status": str(manifest.get("status", "missing")).strip().lower(),
            "train_files": manifest["train_files"],
            "val_files": manifest["val_files"],
        }
        if manifest is not None
        else None
    )
    print("Verifying cached hashes or hashing train/validation files for exact resume...")
    return {
        "version": 3,
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
        "selection_metric": "macro_ssim",
        "ssim": "gaussian_valid_11_sigma1.5",
    }


def _enforce_training_split_status(config: dict[str, Any]) -> None:
    manifest_path = config.get("dataset", {}).get("split_manifest")
    if not manifest_path:
        return
    manifest = load_eventhdr_split_manifest(manifest_path)
    status = str(manifest.get("status", "missing")).strip().lower()
    allow_provisional = bool(config.get("train", {}).get("allow_provisional_split", False))
    if status != "final" and not allow_provisional:
        raise ValueError(
            f"Training split manifest {manifest_path} has status='{status}', not 'final'. "
            "Finalize the scene-level split or explicitly set "
            "train.allow_provisional_split=true for a non-reportable smoke run."
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


def train(
    config: dict[str, Any], resume_from: str | Path | None = None
) -> Path:
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
            "full_group_prefix"
            if validation_context_frames is None
            else "bounded_predecessor"
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
        model, resume_checkpoint = load_model_checkpoint(
            resume_path, device, config["model"]
        )
    else:
        model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 2e-4)),
        weight_decay=float(train_config.get("weight_decay", 1e-6)),
    )
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
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
        historical_score = _resume_best_macro_ssim(resume_checkpoint)
        historical_path = run_dir / "best.pt"
        if math.isfinite(historical_score):
            if not historical_path.is_file():
                raise ValueError(
                    f"Exact resume requires the historical best checkpoint: {historical_path}"
                )
            historical_best = _load_checkpoint(historical_path)
            historical_best.pop("model", None)
            _validate_resume_best_pair(resume_checkpoint, historical_best)
            del historical_best
        elif historical_path.exists():
            raise ValueError(
                "Resume checkpoint has no validated best score, but run_dir contains a "
                "best.pt from another or inconsistent run"
            )
    save_json(run_dir / "config.json", config)

    best_ssim = float("-inf")
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
        if "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint.pop("scaler"))
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_ssim = _resume_best_macro_ssim(resume_checkpoint)
        history = list(resume_checkpoint.get("history", []))
        _restore_rng_state(resume_checkpoint.pop("rng_state", None))

    epochs = int(train_config.get("epochs", 40))
    validate_every = max(1, int(train_config.get("validate_every", 1)))
    max_train_samples = train_config.get("max_train_samples")
    for epoch in range(start_epoch, epochs + 1):
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
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.get("grad_clip", 1.0)))
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

        should_validate = epoch % validate_every == 0 or epoch == epochs
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
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val": val_metrics,
            "val_sampling": val_sampling_counts,
            "gpu_memory": _cuda_peak_memory(device),
        }
        history.append(record)
        save_json(run_dir / "history.json", history)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
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
            "best_metric": "macro_ssim",
            "validation_protocol": validation_protocol,
            "history": history,
            "rng_state": _capture_rng_state(),
        }
        validation_ssim = _macro_ssim(val_metrics)
        if validation_ssim > best_ssim:
            best_ssim = validation_ssim
            checkpoint["best_ssim"] = best_ssim
            best_checkpoint = {
                "checkpoint_type": "ann_inference",
                "epoch": checkpoint["epoch"],
                "model": checkpoint["model"],
                "model_config": checkpoint["model_config"],
                "val": checkpoint["val"],
                "val_sampling": checkpoint["val_sampling"],
                "best_ssim": checkpoint["best_ssim"],
                "best_metric": checkpoint["best_metric"],
                "validation_protocol": checkpoint["validation_protocol"],
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
) -> dict[str, Any]:
    _validate_snn_request(inference_mode, simulation_steps)
    set_seed(int(config.get("seed", 2026)))
    device = resolve_device(config.get("device", "auto"))
    dataset = build_dataset(config["dataset"], split="eval")
    eval_config = config.get("eval", {})
    loader = _data_loader(
        dataset,
        1,
        int(eval_config.get("num_workers", 0)),
        device,
        **_loader_kwargs(eval_config),
    )
    model, checkpoint = load_model_checkpoint(checkpoint_path, device, config["model"])
    _validate_snn_request(
        inference_mode, simulation_steps, checkpoint, checkpoint_path
    )
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
    output_dir = Path(eval_config.get("output_dir", "runs/evaluation"))
    save_limit = int(eval_config.get("save_predictions", 0))
    max_samples = eval_config.get("max_samples")
    saved = 0
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
            "events": int(sample["events"].shape[0]),
            "nodes": diagnostics["nodes"],
            "edges": diagnostics["edges"],
        }
        frame_rows.append(row)
        latencies.append(latency_ms)
        if saved < save_limit:
            safe_name = sample["sample_id"].replace("/", "_")
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
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "inference_mode": inference_mode,
        "simulation_steps": simulation_steps if inference_mode == "snn" else None,
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


@torch.no_grad()
def benchmark(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    warmup: int = 10,
    steps: int = 100,
    inference_mode: str = "ann",
    simulation_steps: int = 16,
) -> dict[str, Any]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    _validate_snn_request(inference_mode, simulation_steps)
    device = resolve_device(config.get("device", "auto"))
    dataset = build_dataset(config["dataset"], split="eval")
    model, checkpoint = load_model_checkpoint(checkpoint_path, device, config["model"])
    _validate_snn_request(
        inference_mode, simulation_steps, checkpoint, checkpoint_path
    )
    model.eval()
    cuda_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    cuda_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    latencies: list[float] = []
    event_counts: list[int] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    firing_rates: list[float] = []
    realtime_factors: list[float] = []
    recurrent_state = None
    current_scene = None
    previous_sequence_index = None
    previous_sensor_size = None
    measured_state_resets = 0
    seed = int(config.get("seed", 2026))
    recurrent = model.decoder.recurrent is not None
    warmup_indices = _representative_schedule(
        dataset, warmup, seed, contiguous=False
    )
    measured_indices = _representative_schedule(
        dataset, steps, seed + 1, contiguous=recurrent
    )
    measured_schedule: list[tuple[bool, int]] = []
    context_frames = 0
    benchmark_context_frames = config.get("eval", {}).get(
        "recurrent_context_frames", 32
    )
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
            event_counts.append(int(sample["events"].shape[0]))
            node_counts.append(diagnostics["nodes"])
            edge_counts.append(diagnostics["edges"])
            firing_rates.extend(
                float(value.detach().cpu())
                if torch.is_tensor(value)
                else float(value)
                for value in diagnostics["firing_rates"]
            )
            dt_us = sample["metadata"].get("dt_us")
            if dt_us:
                realtime_factors.append(elapsed_ms / (float(dt_us) / 1000.0))
    result: dict[str, Any] = {
        **_latency_summary(latencies),
        "events_per_second": sum(event_counts) / (sum(latencies) / 1000.0),
        "mean_nodes": statistics.fmean(node_counts),
        "mean_edges": statistics.fmean(edge_counts),
        "mean_firing_rate": statistics.fmean(firing_rates) if firing_rates else None,
        "deadline_miss_ratio": (
            sum(value > 1.0 for value in realtime_factors) / len(realtime_factors)
            if realtime_factors
            else None
        ),
        "rtf_p95": percentile(realtime_factors, 0.95) if realtime_factors else None,
        "inference_mode": inference_mode,
        "simulation_steps": simulation_steps if inference_mode == "snn" else None,
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
        "sampling": _sampling_summary(dataset, measured_indices),
        "warmup_frames": len(warmup_indices),
        "recurrent_context_policy": (
            "full_group_prefix"
            if recurrent and benchmark_context_frames is None
            else "bounded_predecessor"
            if recurrent
            else None
        ),
        "max_recurrent_context_frames_per_group": benchmark_context_frames
        if recurrent
        else 0,
        "recurrent_context_frames": context_frames,
        "state_resets": measured_state_resets,
        "state_reset_ratio": measured_state_resets / len(measured_indices),
    }
    return result


@torch.no_grad()
def calibrate(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path,
    samples: int = 100,
) -> Path:
    if int(samples) < 1:
        raise ValueError("calibration samples must be at least 1")
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
    model.encoder.reset_thresholds()
    calibration_indices = _balanced_sample_indices(
        dataset,
        min(int(samples), len(dataset)),
        seed=int(config.get("seed", 2026)),
    )
    calibration_sampling = _sampling_summary(dataset, calibration_indices)
    for index in tqdm(calibration_indices, desc="calibrate-SNN"):
        sample = move_sample(dataset[index], device)
        model.calibrate_sample(sample, momentum=-1.0)
    inference_checkpoint = {
        "checkpoint_type": "snn_inference",
        "model": model.state_dict(),
        "model_config": checkpoint.get("model_config", config["model"]),
        "epoch": checkpoint.get("epoch"),
        "source_checkpoint": str(checkpoint_path),
        "batch_norm_folded": True,
        "snn_calibrated": True,
        "snn_calibration_samples": len(calibration_indices),
        "snn_calibration_sampling": calibration_sampling,
    }
    output_path = Path(output_path)
    atomic_torch_save(inference_checkpoint, output_path)
    return output_path
