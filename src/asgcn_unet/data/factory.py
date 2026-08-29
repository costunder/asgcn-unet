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
            f"EventHDR split manifest {manifest_path} has invalid relative HDF5 path: {value!r}"
        )
    return key.as_posix()


def _normalize_file_list(
    values: Any,
    *,
    field: str,
    manifest_path: Path,
) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"EventHDR split manifest {manifest_path} field '{field}' must be "
            "a non-empty list of HDF5 filenames"
        )
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(
            f"EventHDR split manifest {manifest_path} field '{field}' contains "
            "a non-string or empty filename"
        )
    normalized = [_normalize_eventhdr_file_key(value, manifest_path) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"EventHDR split manifest {manifest_path} field '{field}' has duplicates")
    return normalized


def load_eventhdr_split_manifest(path: str | Path) -> dict[str, Any]:
    """Load the one supported EventHDR protocol: official train/eval roots.

    Public EventHDR H5 filenames are sequence containers, not verified physical
    scene identifiers.  Keeping one explicit schema prevents an experiment from
    silently treating numeric filenames as scene-disjoint train/validation data.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"EventHDR split manifest does not exist: {manifest_path}. "
            "Paths in checked-in configs are resolved from the repository root."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError(f"EventHDR split manifest {manifest_path} must contain an object")
    raw_status = manifest.get("status")
    if not isinstance(raw_status, str):
        raise TypeError(f"EventHDR split manifest {manifest_path} field 'status' must be a string")
    status = raw_status.strip().lower()
    if status != "final":
        raise ValueError(f"EventHDR split manifest {manifest_path} must have status='final'")
    manifest["status"] = status

    scene_fields = ("scene_groups", "train_scenes", "val_scenes")
    present_scene_fields = [field for field in scene_fields if field in manifest]
    declared_schema = manifest.get("split_schema")
    if not isinstance(declared_schema, str):
        raise TypeError(
            f"EventHDR split manifest {manifest_path} field 'split_schema' must be a string"
        )
    if declared_schema != "official_separate_roots_v1":
        raise ValueError(
            f"EventHDR split manifest {manifest_path} supports only "
            "split_schema='official_separate_roots_v1'"
        )
    if present_scene_fields:
        raise ValueError(
            f"Official EventHDR separate-root manifest {manifest_path} must not "
            "declare physical-scene fields"
        )
    deprecated_group_fields = [
        field for field in ("train_scene_groups", "val_scene_groups") if field in manifest
    ]
    if deprecated_group_fields:
        raise ValueError(
            f"Official EventHDR separate-root manifest {manifest_path} generates "
            "split-local H5 sequence groups automatically; remove: "
            + ", ".join(deprecated_group_fields)
        )
    required_semantics = "h5_sequence_file_not_physical_scene"
    if manifest.get("group_semantics") != required_semantics:
        raise ValueError(
            f"Official EventHDR separate-root manifest {manifest_path} must set "
            f"group_semantics='{required_semantics}'"
        )
    manifest["train_files"] = _normalize_file_list(
        manifest.get("train_files"), field="train_files", manifest_path=manifest_path
    )
    manifest["val_files"] = _normalize_file_list(
        manifest.get("val_files"), field="val_files", manifest_path=manifest_path
    )
    manifest["train_file_to_group"] = {
        file_key: f"official-train-h5::{file_key}" for file_key in manifest["train_files"]
    }
    manifest["val_file_to_group"] = {
        file_key: f"official-eval-h5::{file_key}" for file_key in manifest["val_files"]
    }
    # EventHDRDataset uses the historical file_to_scene argument name.  Values
    # remain explicit sequence-group IDs and are never physical-scene claims.
    manifest["train_file_to_scene"] = manifest["train_file_to_group"]
    manifest["val_file_to_scene"] = manifest["val_file_to_group"]
    manifest["file_to_group"] = {
        "train": manifest["train_file_to_group"],
        "val": manifest["val_file_to_group"],
    }
    manifest["file_to_scene"] = manifest["file_to_group"]
    return manifest


def _discover_eventhdr_files(
    root: Path,
    *,
    exclude_roots: tuple[Path, ...] = (),
) -> set[str]:
    """List H5 keys under one logical root without double-counting nested roots."""
    resolved_root = root.resolve()
    resolved_excludes = tuple(
        candidate.resolve()
        for candidate in exclude_roots
        if candidate.resolve() != resolved_root
        and candidate.resolve().is_relative_to(resolved_root)
    )
    discovered: set[str] = set()
    for pattern in ("*.h5", "*.hdf5"):
        for path in root.rglob(pattern):
            if any(path.resolve().is_relative_to(excluded) for excluded in resolved_excludes):
                continue
            discovered.add(path.relative_to(root).as_posix())
    return discovered


def build_dataset(config: dict[str, Any], split: str = "train"):
    cfg = dict(config)
    dataset_type = cfg.pop("type")
    expected_file_count = cfg.pop("expected_file_count", None)
    file_manifest = cfg.pop("file_manifest", None)
    training_root = Path(cfg["root"])
    validation_root = Path(cfg["val_root"]) if cfg.get("val_root") else training_root
    if split == "val" and cfg.get("val_root"):
        cfg["root"] = cfg["val_root"]
    root = cfg.pop("root")
    cfg.pop("val_root", None)
    split_manifest = cfg.pop("split_manifest", None)
    cfg["random_crop"] = split == "train" and cfg.get("crop_size") is not None
    if dataset_type == "eventhdr":
        eventhdr_group_semantics: str | None = None
        if split_manifest and split in {"train", "val", "calibration"}:
            manifest_path = Path(split_manifest)
            manifest = load_eventhdr_split_manifest(manifest_path)
            eventhdr_group_semantics = manifest["group_semantics"]
            if training_root.resolve() == validation_root.resolve():
                raise ValueError(
                    "Official EventHDR train/eval split requires distinct dataset.root "
                    "and dataset.val_root directories"
                )
            coverage_specs = (
                (
                    "dataset.root train_files",
                    training_root,
                    set(manifest["train_files"]),
                    (validation_root,),
                ),
                (
                    "dataset.val_root val_files",
                    validation_root,
                    set(manifest["val_files"]),
                    (training_root,),
                ),
            )
            for label, coverage_root, declared, excluded_roots in coverage_specs:
                discovered = _discover_eventhdr_files(
                    coverage_root,
                    exclude_roots=excluded_roots,
                )
                undeclared = sorted(discovered - declared)
                missing = sorted(declared - discovered)
                if undeclared or missing:
                    details = []
                    if undeclared:
                        details.append("undeclared: " + ", ".join(undeclared[:8]))
                    if missing:
                        details.append("missing: " + ", ".join(missing[:8]))
                    raise ValueError(
                        f"Final EventHDR manifest must cover every H5 under {label} ("
                        + "; ".join(details)
                        + ")"
                    )
            key = "val_files" if split == "val" else "train_files"
            mapping_key = "val_file_to_group" if split == "val" else "train_file_to_group"
            cfg["allowed_files"] = manifest[key]
            cfg["file_to_scene"] = manifest[mapping_key]
        dataset = EventHDRDataset(root=root, **cfg)
        if eventhdr_group_semantics is not None:
            dataset.group_semantics = eventhdr_group_semantics
        if expected_file_count is not None and len(dataset.files) != int(expected_file_count):
            dataset.close()
            raise ValueError(
                f"EventHDR coverage requires exactly {int(expected_file_count)} H5 files; "
                f"found {len(dataset.files)}"
            )
        return dataset
    if dataset_type == "eventaid_r_zip":
        dataset = EventAidRZipDataset(root=root, **cfg)
        if file_manifest is not None:
            manifest_path = Path(file_manifest)
            if not manifest_path.is_file():
                dataset.close()
                raise FileNotFoundError(f"EventAid-R file manifest does not exist: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                f"{item['scene']}.zip"
                for item in manifest.get("files", [])
                if isinstance(item, dict) and item.get("scene")
            }
            present = {path.name for path in dataset.zip_paths}
            if not expected or present != expected:
                dataset.close()
                missing = sorted(expected - present)
                extra = sorted(present - expected)
                raise ValueError(
                    "EventAid-R coverage does not match the fixed file manifest "
                    f"(missing={missing}, extra={extra})"
                )
        if expected_file_count is not None and len(dataset.zip_paths) != int(expected_file_count):
            dataset.close()
            raise ValueError(
                f"EventAid-R coverage requires exactly {int(expected_file_count)} ZIP files; "
                f"found {len(dataset.zip_paths)}"
            )
        return dataset
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def collate_samples(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Graphs and sensor resolutions are variable-sized; the model loops over this small list.
    return batch
