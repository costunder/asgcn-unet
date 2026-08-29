from __future__ import annotations

import copy
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


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
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def atomic_torch_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
