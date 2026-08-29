from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from .eventaid_r import EventAidRZipDataset
from .eventhdr import EventHDRDataset


def _normalize_eventhdr_file_key(value: str, manifest_path: Path) -> str:
    normalized = value.replace("\\", "/")
    key = PurePosixPath(normalized)
    if (
        key.is_absolute()
        or ".." in key.parts
        or not key.name
        or ":" in key.parts[0]
        or key.suffix.lower() not in {".h5", ".hdf5"}
    ):
        raise ValueError(
            f"EventHDR split manifest {manifest_path} has invalid relative HDF5 path: "
            f"{value!r}"
        )
    return key.as_posix()


def load_eventhdr_split_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"EventHDR split manifest does not exist: {manifest_path}. "
            "Paths in checked-in configs are resolved from the repository root."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for key in ("train_files", "val_files"):
        values = manifest.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"EventHDR split manifest {manifest_path} field '{key}' must be "
                "a non-empty list of HDF5 filenames"
            )
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(
                f"EventHDR split manifest {manifest_path} field '{key}' contains "
                "a non-string or empty filename"
            )
        normalized = [_normalize_eventhdr_file_key(value, manifest_path) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError(
                f"EventHDR split manifest {manifest_path} field '{key}' has duplicates"
            )
        manifest[key] = sorted(normalized)
    overlap = sorted(set(manifest["train_files"]) & set(manifest["val_files"]))
    if overlap:
        raise ValueError(
            f"EventHDR split manifest {manifest_path} leaks files across train/val: "
            + ", ".join(overlap[:8])
            + (" ..." if len(overlap) > 8 else "")
        )
    return manifest


def build_dataset(config: dict[str, Any], split: str = "train"):
    cfg = dict(config)
    dataset_type = cfg.pop("type")
    if split == "val" and cfg.get("val_root"):
        cfg["root"] = cfg["val_root"]
    root = cfg.pop("root")
    cfg.pop("val_root", None)
    split_manifest = cfg.pop("split_manifest", None)
    cfg["random_crop"] = split == "train" and cfg.get("crop_size") is not None
    if dataset_type == "eventhdr":
        if split_manifest and split in {"train", "val", "calibration"}:
            manifest_path = Path(split_manifest)
            manifest = load_eventhdr_split_manifest(manifest_path)
            key = "val_files" if split == "val" else "train_files"
            cfg["allowed_files"] = manifest[key]
        return EventHDRDataset(root=root, **cfg)
    if dataset_type == "eventaid_r_zip":
        return EventAidRZipDataset(root=root, **cfg)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def collate_samples(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Graphs and sensor resolutions are variable-sized; the model loops over this small list.
    return batch
