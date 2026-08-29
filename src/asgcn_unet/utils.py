from __future__ import annotations

import copy
import csv
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .data.common import validate_target_normalization
from .losses import validate_loss_weights

_EVALUATION_PRECISIONS = {"fp32", "amp_fp16", "bf16"}


def _validate_optional_positive_integer(value: Any, field: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be null or an integer >= 1")


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Fail fast on cross-stage contracts shared by CLI and engine entry points."""
    if not isinstance(config, dict):
        raise TypeError("Experiment config must be an object")
    dataset = config.get("dataset")
    if dataset is not None:
        if not isinstance(dataset, dict):
            raise TypeError("dataset must be an object")
        validate_target_normalization(dataset.get("target_normalization"))
        if "target_offset" in dataset:
            target_offset = dataset["target_offset"]
            if isinstance(target_offset, bool) or not isinstance(target_offset, int):
                raise TypeError("dataset.target_offset must be an integer and must not be bool")

    train = config.get("train")
    if train is not None:
        if not isinstance(train, dict):
            raise TypeError("train must be an object")
        batch_size = train.get("batch_size", 1)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size != 1:
            raise ValueError("train.batch_size must be 1 for the current sample-wise pipeline")
        _validate_optional_positive_integer(
            train.get("max_train_samples"), "train.max_train_samples"
        )
        _validate_optional_positive_integer(train.get("max_val_samples"), "train.max_val_samples")
        validate_loss_weights(train.get("loss_weights"))

    evaluation = config.get("eval")
    if evaluation is not None:
        if not isinstance(evaluation, dict):
            raise TypeError("eval must be an object")
        batch_size = evaluation.get("batch_size", 1)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size != 1:
            raise ValueError("eval.batch_size must be 1 for the current sample-wise pipeline")
        _validate_optional_positive_integer(evaluation.get("max_samples"), "eval.max_samples")
        precision = evaluation.get("precision", "fp32")
        if not isinstance(precision, str) or precision not in _EVALUATION_PRECISIONS:
            supported = ", ".join(sorted(_EVALUATION_PRECISIONS))
            raise ValueError(f"eval.precision must be one of: {supported}")
        if not isinstance(evaluation.get("tf32", False), bool):
            raise TypeError("eval.tf32 must be a boolean")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def experiment_base_dir(config_path: str | Path) -> Path:
    """Locate the checkout root that owns an experiment configuration.

    Checked-in configs use paths relative to the repository, not relative to the
    shell's current directory.  Falling back to the config directory also keeps
    standalone, externally supplied configs useful.
    """
    config_path = Path(config_path).expanduser().resolve()
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return config_path.parent


def resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    expanded = Path(os.path.expandvars(str(path))).expanduser()
    if not expanded.is_absolute():
        expanded = Path(base_dir) / expanded
    return expanded.resolve()


def resolve_experiment_paths(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    """Return a copy with filesystem paths anchored to the checkout root."""
    validate_experiment_config(config)
    resolved = copy.deepcopy(config)
    base_dir = experiment_base_dir(config_path)
    path_locations = (
        ("dataset", "root"),
        ("dataset", "val_root"),
        ("dataset", "split_manifest"),
        ("dataset", "file_manifest"),
        ("train", "resume"),
        ("output", "run_dir"),
        ("eval", "output_dir"),
    )
    for section, key in path_locations:
        value = resolved.get(section, {}).get(key)
        if value:
            resolved[section][key] = str(resolve_path(value, base_dir))
    return resolved


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def move_sample(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(sample)
    result["events"] = sample["events"].to(device, non_blocking=True)
    result["target"] = sample["target"].to(device, non_blocking=True)
    return result


def move_inference_sample(
    sample: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Move only model inputs for a compute-only inference measurement.

    Quality evaluation uses :func:`move_sample` because its metrics require the
    target on the model device.  A compute benchmark does not consume the target,
    so keeping it on the host prevents ground-truth storage from inflating the
    reported peak GPU memory.
    """
    result = dict(sample)
    result["events"] = sample["events"].to(device, non_blocking=True)
    return result


def save_image(path: str | Path, tensor: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().float().clamp(0, 1).cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    if array.shape[0] == 1:
        image = Image.fromarray((array[0] * 255.0 + 0.5).astype(np.uint8), mode="L")
    else:
        image = Image.fromarray(
            (array[:3].transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8), mode="RGB"
        )
    image.save(path)


def write_frame_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    expected = set(fieldnames)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(f"CSV row {index} keys do not match the first row")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_torch_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
