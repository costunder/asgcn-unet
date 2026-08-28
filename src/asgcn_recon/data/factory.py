from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .eventaid_r import EventAidRZipDataset
from .eventhdr import EventHDRDataset


def build_dataset(config: dict[str, Any], split: str = "train"):
    cfg = dict(config)
    dataset_type = cfg.pop("type")
    root = cfg.pop("root")
    cfg.pop("val_root", None)
    split_manifest = cfg.pop("split_manifest", None)
    cfg["random_crop"] = split == "train" and cfg.get("crop_size") is not None
    if dataset_type == "eventhdr":
        if split_manifest and split in {"train", "val", "calibration"}:
            manifest_path = Path(split_manifest)
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"EventHDR split manifest does not exist: {manifest_path}. "
                    "Paths in checked-in configs are resolved from the repository root."
                )
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            key = "val_files" if split == "val" else "train_files"
            if key not in manifest:
                raise KeyError(
                    f"EventHDR split manifest {manifest_path} has no '{key}' list "
                    f"for split='{split}'"
                )
            if not isinstance(manifest[key], list) or not manifest[key]:
                raise ValueError(
                    f"EventHDR split manifest {manifest_path} field '{key}' must be "
                    "a non-empty list of HDF5 filenames"
                )
            cfg["allowed_files"] = manifest[key]
        return EventHDRDataset(root=root, **cfg)
    if dataset_type == "eventaid_r_zip":
        return EventAidRZipDataset(root=root, **cfg)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def collate_samples(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Graphs and sensor resolutions are variable-sized; the model loops over this small list.
    return batch
