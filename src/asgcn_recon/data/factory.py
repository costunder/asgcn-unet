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


def _normalize_scene_list(
    values: Any,
    *,
    field: str,
    manifest_path: Path,
) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"EventHDR split manifest {manifest_path} field '{field}' must be "
            "a non-empty list of physical scene IDs"
        )
    if any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in values
    ):
        raise ValueError(
            f"EventHDR split manifest {manifest_path} field '{field}' contains "
            "an invalid physical scene ID"
        )
    if len(values) != len(set(values)):
        raise ValueError(f"EventHDR split manifest {manifest_path} field '{field}' has duplicates")
    return list(values)


def load_eventhdr_split_manifest(path: str | Path) -> dict[str, Any]:
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
    raw_status = manifest.get("status", "provisional")
    if not isinstance(raw_status, str):
        raise TypeError(f"EventHDR split manifest {manifest_path} field 'status' must be a string")
    status = raw_status.strip().lower()
    if status not in {"provisional", "final"}:
        raise ValueError(
            f"EventHDR split manifest {manifest_path} status must be 'provisional' or 'final'"
        )
    manifest["status"] = status

    scene_fields = ("scene_groups", "train_scenes", "val_scenes")
    present_scene_fields = [field for field in scene_fields if field in manifest]
    declared_schema = manifest.get("split_schema")
    if declared_schema is not None and not isinstance(declared_schema, str):
        raise TypeError(
            f"EventHDR split manifest {manifest_path} field 'split_schema' must be a string"
        )
    if declared_schema == "official_separate_roots_v1":
        if status != "final":
            raise ValueError(
                f"Official EventHDR separate-root manifest {manifest_path} must have "
                "status='final'"
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
            file_key: f"official-train-h5::{file_key}"
            for file_key in manifest["train_files"]
        }
        manifest["val_file_to_group"] = {
            file_key: f"official-eval-h5::{file_key}" for file_key in manifest["val_files"]
        }
        # ``EventHDRDataset`` keeps the historical file_to_scene API, but values in
        # this schema are explicitly H5 sequence groups, not physical-scene labels.
        manifest["train_file_to_scene"] = manifest["train_file_to_group"]
        manifest["val_file_to_scene"] = manifest["val_file_to_group"]
        manifest["file_to_group"] = {
            "train": manifest["train_file_to_group"],
            "val": manifest["val_file_to_group"],
        }
        manifest["file_to_scene"] = {
            "train": manifest["train_file_to_scene"],
            "val": manifest["val_file_to_scene"],
        }
        return manifest

    if present_scene_fields and len(present_scene_fields) != len(scene_fields):
        missing = ", ".join(field for field in scene_fields if field not in manifest)
        raise ValueError(
            f"EventHDR split manifest {manifest_path} has an incomplete physical-scene "
            f"schema; missing: {missing}"
        )

    if not present_scene_fields:
        if status == "final":
            raise ValueError(
                f"Final EventHDR split manifest {manifest_path} requires scene_groups, "
                "train_scenes, and val_scenes; legacy train_files/val_files cannot "
                "prove physical-scene separation"
            )
        for field in ("train_files", "val_files"):
            manifest[field] = sorted(
                _normalize_file_list(manifest.get(field), field=field, manifest_path=manifest_path)
            )
        overlap = sorted(set(manifest["train_files"]) & set(manifest["val_files"]))
        if overlap:
            raise ValueError(
                f"EventHDR split manifest {manifest_path} leaks files across train/val: "
                + ", ".join(overlap[:8])
                + (" ..." if len(overlap) > 8 else "")
            )
        manifest["split_schema"] = "legacy_files_v1"
        manifest["train_file_to_scene"] = {
            key: key for key in manifest["train_files"]
        }
        manifest["val_file_to_scene"] = {key: key for key in manifest["val_files"]}
        manifest["file_to_scene"] = {
            **manifest["train_file_to_scene"],
            **manifest["val_file_to_scene"],
        }
        return manifest

    raw_groups = manifest["scene_groups"]
    if not isinstance(raw_groups, dict) or not raw_groups:
        raise ValueError(
            f"EventHDR split manifest {manifest_path} field 'scene_groups' must be "
            "a non-empty object"
        )
    scene_groups: dict[str, list[str]] = {}
    file_to_scene: dict[str, str] = {}
    for scene_id, values in raw_groups.items():
        if not isinstance(scene_id, str) or not scene_id.strip() or scene_id != scene_id.strip():
            raise ValueError(
                f"EventHDR split manifest {manifest_path} has an invalid physical scene ID"
            )
        files = _normalize_file_list(
            values,
            field=f"scene_groups.{scene_id}",
            manifest_path=manifest_path,
        )
        for file_key in files:
            owner = file_to_scene.get(file_key)
            if owner is not None:
                raise ValueError(
                    f"EventHDR split manifest {manifest_path} assigns file {file_key!r} "
                    f"to multiple physical scenes: {owner!r}, {scene_id!r}"
                )
            file_to_scene[file_key] = scene_id
        scene_groups[scene_id] = files

    train_scenes = _normalize_scene_list(
        manifest["train_scenes"], field="train_scenes", manifest_path=manifest_path
    )
    val_scenes = _normalize_scene_list(
        manifest["val_scenes"], field="val_scenes", manifest_path=manifest_path
    )
    unknown_scenes = sorted((set(train_scenes) | set(val_scenes)) - set(scene_groups))
    if unknown_scenes:
        raise ValueError(
            f"EventHDR split manifest {manifest_path} references undefined physical "
            "scenes: " + ", ".join(unknown_scenes)
        )
    scene_overlap = sorted(set(train_scenes) & set(val_scenes))
    if scene_overlap:
        raise ValueError(
            f"EventHDR split manifest {manifest_path} leaks physical scenes across "
            "train/val: " + ", ".join(scene_overlap)
        )
    unassigned_scenes = sorted(set(scene_groups) - set(train_scenes) - set(val_scenes))
    if unassigned_scenes:
        raise ValueError(
            f"EventHDR split manifest {manifest_path} leaves physical scenes unassigned: "
            + ", ".join(unassigned_scenes)
        )

    train_files = [file for scene in train_scenes for file in scene_groups[scene]]
    val_files = [file for scene in val_scenes for file in scene_groups[scene]]
    for field, normalized in (("train_files", train_files), ("val_files", val_files)):
        if field in manifest:
            declared = _normalize_file_list(
                manifest[field], field=field, manifest_path=manifest_path
            )
            if set(declared) != set(normalized):
                raise ValueError(
                    f"EventHDR split manifest {manifest_path} field '{field}' does not "
                    "match the physical-scene assignment"
                )
        manifest[field] = normalized
    manifest["scene_groups"] = dict(sorted(scene_groups.items()))
    manifest["train_scenes"] = train_scenes
    manifest["val_scenes"] = val_scenes
    manifest["file_to_scene"] = dict(sorted(file_to_scene.items()))
    manifest["train_file_to_scene"] = {
        file_key: file_to_scene[file_key] for file_key in train_files
    }
    manifest["val_file_to_scene"] = {
        file_key: file_to_scene[file_key] for file_key in val_files
    }
    manifest["split_schema"] = "physical_scenes_v1"
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
            eventhdr_group_semantics = manifest.get("group_semantics")
            if eventhdr_group_semantics is None:
                eventhdr_group_semantics = (
                    "physical_scene"
                    if manifest["split_schema"] == "physical_scenes_v1"
                    else "h5_sequence_file_not_physical_scene"
                )
            if manifest["status"] == "final":
                roots_match = training_root.resolve() == validation_root.resolve()
                if manifest["split_schema"] == "official_separate_roots_v1" and roots_match:
                    raise ValueError(
                        "Official EventHDR train/eval split requires distinct dataset.root "
                        "and dataset.val_root directories"
                    )
                if roots_match:
                    coverage_specs = (
                        (
                            "dataset.root",
                            training_root,
                            set(manifest["file_to_scene"]),
                            (),
                        ),
                    )
                else:
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
            if manifest["split_schema"] == "official_separate_roots_v1":
                mapping_key = "val_file_to_group" if split == "val" else "train_file_to_group"
            else:
                mapping_key = (
                    "val_file_to_scene" if split == "val" else "train_file_to_scene"
                )
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
