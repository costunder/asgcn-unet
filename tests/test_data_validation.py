from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest

from asgcn_unet.cli import _inspect_one_split
from asgcn_unet.data import (
    EventAidRZipDataset,
    EventHDRDataset,
    build_dataset,
    load_eventhdr_split_manifest,
)
from asgcn_unet.engine import _dataset_coverage_summary
from tests.fixtures import make_eventaid, make_eventhdr


def _remove_events_group(h5: h5py.File) -> None:
    del h5["events"]


def _remove_images_group(h5: h5py.File) -> None:
    del h5["images"]


def _remove_polarity_array(h5: h5py.File) -> None:
    del h5["events/ps"]


def _shorten_y_array(h5: h5py.File) -> None:
    values = h5["events/ys"][:-1]
    del h5["events/ys"]
    h5["events"].create_dataset("ys", data=values)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_remove_events_group, "required group 'events'"),
        (_remove_images_group, "required group 'images'"),
        (_remove_polarity_array, "events/ps"),
        (_shorten_y_array, "equal lengths"),
    ],
)
def test_eventhdr_rejects_missing_or_misaligned_event_arrays(
    tmp_path: Path,
    mutation: Callable[[h5py.File], None],
    message: str,
) -> None:
    path = make_eventhdr(tmp_path / "hdr")
    with h5py.File(path, "a") as h5:
        mutation(h5)

    with pytest.raises(ValueError, match=message):
        EventHDRDataset(path.parent)


def test_eventhdr_requires_image_timestamp(tmp_path: Path) -> None:
    path = make_eventhdr(tmp_path / "hdr")
    with h5py.File(path, "a") as h5:
        del h5["images/image000000000"].attrs["timestamp"]

    with pytest.raises(ValueError, match="missing 'timestamp'"):
        EventHDRDataset(path.parent)


def test_eventhdr_uses_relative_paths_for_nested_duplicate_basenames(tmp_path: Path) -> None:
    make_eventhdr(tmp_path / "one")
    make_eventhdr(tmp_path / "two")

    dataset = EventHDRDataset(tmp_path)
    assert len(dataset) == 8
    assert {dataset[0]["metadata"]["scene"], dataset[4]["metadata"]["scene"]} == {
        "one/test.h5",
        "two/test.h5",
    }
    assert dataset[0]["sample_id"] != dataset[4]["sample_id"]


def test_eventhdr_distinguishes_h5_and_hdf5_stems(tmp_path: Path) -> None:
    first = make_eventhdr(tmp_path / "hdr")
    first.rename(first.with_name("same.h5"))
    second = make_eventhdr(tmp_path / "other")
    second.rename(tmp_path / "hdr" / "same.hdf5")

    dataset = EventHDRDataset(tmp_path / "hdr")
    scenes = {dataset[0]["metadata"]["scene"], dataset[4]["metadata"]["scene"]}
    assert scenes == {"same.h5", "same.hdf5"}


def _write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_official_separate_root_manifest_maps_overlapping_names_to_sequence_groups(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "split_schema": "official_separate_roots_v1",
            "group_semantics": "h5_sequence_file_not_physical_scene",
            "train_files": ["1.h5"],
            "val_files": ["1.h5"],
        },
    )

    manifest = load_eventhdr_split_manifest(manifest_path)

    assert manifest["split_schema"] == "official_separate_roots_v1"
    assert manifest["group_semantics"] == "h5_sequence_file_not_physical_scene"
    assert manifest["train_files"] == ["1.h5"]
    assert manifest["val_files"] == ["1.h5"]
    assert manifest["file_to_group"] == {
        "train": {"1.h5": "official-train-h5::1.h5"},
        "val": {"1.h5": "official-eval-h5::1.h5"},
    }


def test_factory_uses_split_local_sequence_groups_for_overlapping_official_names(
    tmp_path: Path,
) -> None:
    train_root = tmp_path / "train"
    val_root = tmp_path / "eval"
    make_eventhdr(train_root).rename(train_root / "1.h5")
    make_eventhdr(val_root).rename(val_root / "1.h5")
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "split_schema": "official_separate_roots_v1",
            "group_semantics": "h5_sequence_file_not_physical_scene",
            "train_files": ["1.h5"],
            "val_files": ["1.h5"],
        },
    )
    config = {
        "type": "eventhdr",
        "root": str(train_root),
        "val_root": str(val_root),
        "split_manifest": str(manifest_path),
    }

    train_dataset = build_dataset(config, split="train")
    val_dataset = build_dataset(config, split="val")
    try:
        assert train_dataset.group_semantics == "h5_sequence_file_not_physical_scene"
        assert val_dataset.group_semantics == "h5_sequence_file_not_physical_scene"
        assert train_dataset[0]["metadata"]["scene"] == "official-train-h5::1.h5"
        assert val_dataset[0]["metadata"]["scene"] == "official-eval-h5::1.h5"
        coverage = _dataset_coverage_summary(val_dataset, config)
        assert coverage["quality_grouping"] == "source_h5_sequence_file"
    finally:
        train_dataset.close()
        val_dataset.close()

    make_eventhdr(tmp_path / "extra").rename(val_root / "2.h5")
    with pytest.raises(ValueError, match=r"dataset\.val_root.*undeclared: 2\.h5"):
        build_dataset(config, split="val")


def test_official_separate_root_manifest_requires_distinct_roots(tmp_path: Path) -> None:
    root = tmp_path / "eventhdr"
    make_eventhdr(root).rename(root / "1.h5")
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "split_schema": "official_separate_roots_v1",
            "group_semantics": "h5_sequence_file_not_physical_scene",
            "train_files": ["1.h5"],
            "val_files": ["1.h5"],
        },
    )

    with pytest.raises(ValueError, match="requires distinct"):
        build_dataset(
            {
                "type": "eventhdr",
                "root": str(root),
                "val_root": str(root),
                "split_manifest": str(manifest_path),
            },
            split="train",
        )


def test_official_sequence_file_schema_rejects_physical_scene_fields(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "split_schema": "official_separate_roots_v1",
            "group_semantics": "h5_sequence_file_not_physical_scene",
            "train_files": ["1.h5"],
            "val_files": ["1.h5"],
            "scene_groups": {"unsupported-claim": ["1.h5"]},
            "train_scenes": ["unsupported-claim"],
            "val_scenes": ["unsupported-claim"],
        },
    )

    with pytest.raises(ValueError, match="must not declare physical-scene fields"):
        load_eventhdr_split_manifest(manifest_path)


def test_checked_in_full_eventhdr_protocol_uses_official_roots_and_all_frames() -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest = load_eventhdr_split_manifest(repository / "manifests/eventhdr_split.json")
    config = json.loads((repository / "configs/train.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "final"
    assert manifest["split_schema"] == "official_separate_roots_v1"
    assert manifest["group_semantics"] == "h5_sequence_file_not_physical_scene"
    assert set(manifest["train_files"]) == {f"{index}.h5" for index in range(1, 52)}
    assert set(manifest["val_files"]) == {f"{index}.h5" for index in range(1, 20)}
    assert config["dataset"]["root"] == "data/EventHDR/train"
    assert config["dataset"]["val_root"] == "data/EventHDR/eval"
    assert config["dataset"]["frame_stride"] == 1
    assert config["dataset"]["crop_size"] is None
    assert config["train"]["max_train_samples"] is None
    assert config["train"]["max_val_samples"] is None
    assert config["train"]["validate_every"] is None


def test_final_eventhdr_manifest_rejects_legacy_file_lists(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "train_files": ["train.h5"],
            "val_files": ["val.h5"],
        },
    )

    with pytest.raises((TypeError, ValueError), match="split_schema"):
        load_eventhdr_split_manifest(manifest_path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "status": "provisional",
                "split_schema": "official_separate_roots_v1",
                "group_semantics": "h5_sequence_file_not_physical_scene",
                "train_files": ["a.h5"],
                "val_files": ["b.h5"],
            },
            "status='final'",
        ),
        (
            {
                "status": "final",
                "split_schema": "physical_scenes_v1",
                "group_semantics": "h5_sequence_file_not_physical_scene",
                "train_files": ["a.h5"],
                "val_files": ["b.h5"],
            },
            "supports only",
        ),
        (
            {
                "status": "final",
                "split_schema": "official_separate_roots_v1",
                "group_semantics": "physical_scene",
                "train_files": ["a.h5"],
                "val_files": ["b.h5"],
            },
            "group_semantics",
        ),
    ],
)
def test_manifest_rejects_unsupported_protocols(
    tmp_path: Path, payload: dict, message: str
) -> None:
    manifest_path = _write_manifest(tmp_path / "split.json", payload)

    with pytest.raises(ValueError, match=message):
        load_eventhdr_split_manifest(manifest_path)


def test_final_manifest_excludes_nested_validation_root_from_training_coverage(
    tmp_path: Path,
) -> None:
    train_root = tmp_path / "dataset"
    val_root = train_root / "validation"
    make_eventhdr(train_root).rename(train_root / "train.h5")
    make_eventhdr(val_root).rename(val_root / "val.h5")
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "split_schema": "official_separate_roots_v1",
            "group_semantics": "h5_sequence_file_not_physical_scene",
            "train_files": ["train.h5"],
            "val_files": ["val.h5"],
        },
    )
    config = {
        "type": "eventhdr",
        "root": str(train_root),
        "val_root": str(val_root),
        "split_manifest": str(manifest_path),
    }

    train_dataset = build_dataset(config, split="train")
    val_dataset = build_dataset(config, split="val")
    assert [path.name for path in train_dataset.files] == ["train.h5"]
    assert [path.name for path in val_dataset.files] == ["val.h5"]
    train_dataset.close()
    val_dataset.close()


def test_eventaid_fixed_manifest_rejects_partial_external_eval(tmp_path: Path) -> None:
    data_root = tmp_path / "aid"
    make_eventaid(data_root)
    manifest_path = _write_manifest(
        tmp_path / "aid.json",
        {
            "files": [
                {"scene": "R-bear"},
                {"scene": "R-ball"},
            ]
        },
    )

    with pytest.raises(ValueError, match="coverage does not match"):
        build_dataset(
            {
                "type": "eventaid_r_zip",
                "root": str(data_root),
                "file_manifest": str(manifest_path),
            },
            split="eval",
        )


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("event_idx", -1, "outside"),
        ("event_idx", 10_000, "outside"),
        ("event_idx", 1.5, "must be an integer"),
        ("timestamp", np.nan, "must be finite"),
    ],
)
def test_eventhdr_rejects_invalid_image_boundaries(
    tmp_path: Path, attribute: str, value: float, message: str
) -> None:
    path = make_eventhdr(tmp_path / "hdr")
    with h5py.File(path, "a") as h5:
        h5["images/image000000000"].attrs[attribute] = value

    with pytest.raises(ValueError, match=message):
        EventHDRDataset(path.parent)


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("event_idx", "event_idx values must be monotonically"),
        ("timestamp", "image timestamps must be monotonically"),
    ],
)
def test_eventhdr_rejects_nonmonotonic_image_boundaries(
    tmp_path: Path, attribute: str, message: str
) -> None:
    path = make_eventhdr(tmp_path / "hdr")
    with h5py.File(path, "a") as h5:
        first = h5["images/image000000000"].attrs[attribute]
        h5["images/image000000001"].attrs[attribute] = first - 1

    with pytest.raises(ValueError, match=message):
        EventHDRDataset(path.parent)


def _replace_h5_array(path: Path, name: str, update: Callable[[np.ndarray], None]) -> None:
    with h5py.File(path, "a") as h5:
        values = np.asarray(h5[f"events/{name}"][:], dtype=np.float64)
        update(values)
        del h5[f"events/{name}"]
        h5["events"].create_dataset(name, data=values)


def _set_nan(values: np.ndarray) -> None:
    values[1] = np.nan


def _reverse_timestamp(values: np.ndarray) -> None:
    values[1] = values[0] - 1.0


def _set_out_of_range_y(values: np.ndarray) -> None:
    values[1] = 32


def _set_invalid_polarity(values: np.ndarray) -> None:
    values[1] = 2


@pytest.mark.parametrize(
    ("array_name", "update", "message"),
    [
        ("ts", _set_nan, "timestamps must be finite"),
        ("ts", _reverse_timestamp, "timestamps must be monotonically"),
        ("xs", _set_nan, "coordinates must be finite"),
        ("ys", _set_out_of_range_y, "coordinates must lie within"),
        ("ps", _set_invalid_polarity, "polarity values must be"),
    ],
)
def test_eventhdr_rejects_malformed_loaded_event_blocks(
    tmp_path: Path,
    array_name: str,
    update: Callable[[np.ndarray], None],
    message: str,
) -> None:
    path = make_eventhdr(tmp_path / "hdr")
    _replace_h5_array(path, array_name, update)
    dataset = EventHDRDataset(path.parent, max_events=None)

    with pytest.raises(ValueError, match=message):
        dataset[0]


def _replace_zip_member(path: Path, member: str, replacement: bytes) -> None:
    temporary = path.with_suffix(".tmp")
    with zipfile.ZipFile(path, "r") as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist()]
    with zipfile.ZipFile(temporary, "w") as destination:
        for info, content in entries:
            destination.writestr(info, replacement if info.filename == member else content)
    temporary.replace(path)


def _remove_zip_member(path: Path, member: str) -> None:
    temporary = path.with_suffix(".tmp")
    with zipfile.ZipFile(path, "r") as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist()]
    with zipfile.ZipFile(temporary, "w") as destination:
        for info, content in entries:
            if info.filename != member:
                destination.writestr(info, content)
    temporary.replace(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"10 0 0\n", "expected four values"),
        (b"10 0 0 1 junk 0 0 1\n", "every token must be numeric"),
        (b"nan 0 0 1\n", "timestamps must be finite"),
        (b"10 0 0 1\n9 0 0 1\n", "timestamps must be monotonically"),
        (b"10 nan 0 1\n", "coordinates must be finite"),
        (b"10 48 0 1\n", "coordinates must lie within"),
        (b"10 0 0 nan\n", "polarity must be finite"),
        (b"10 0 0 2\n", "polarity values must be"),
    ],
)
def test_eventaid_rejects_malformed_event_text(
    tmp_path: Path, content: bytes, message: str
) -> None:
    path = make_eventaid(tmp_path / "eventaid")
    _replace_zip_member(path, "event/000001.txt", content)
    dataset = EventAidRZipDataset(path.parent, max_events=None)

    with pytest.raises(ValueError, match=message):
        dataset[0]


def test_eventaid_rejects_nonmonotonic_frame_timestamps(tmp_path: Path) -> None:
    path = make_eventaid(tmp_path / "eventaid")
    _replace_zip_member(path, "timestamps.txt", b"100\n99\n101\n102\n")

    with pytest.raises(ValueError, match="timestamps.*strictly increasing"):
        EventAidRZipDataset(path.parent)


def test_eventaid_requires_timestamps_file(tmp_path: Path) -> None:
    path = make_eventaid(tmp_path / "eventaid")
    _remove_zip_member(path, "timestamps.txt")

    with pytest.raises(ValueError, match="timestamps.txt is missing"):
        EventAidRZipDataset(path.parent)


def test_eventaid_rejects_internal_gt_gap(tmp_path: Path) -> None:
    path = make_eventaid(tmp_path / "eventaid")
    _remove_zip_member(path, "gt/000003_img.png")

    with pytest.raises(ValueError, match="GT IDs are not contiguous"):
        EventAidRZipDataset(path.parent)


def test_eventaid_requires_timestamp_for_every_pair(tmp_path: Path) -> None:
    path = make_eventaid(tmp_path / "eventaid")
    _replace_zip_member(path, "timestamps.txt", b"100\n200\n")

    with pytest.raises(ValueError, match="does not cover"):
        EventAidRZipDataset(path.parent)


def test_eventaid_full_inspect_reports_timestamp_relationship_without_assuming_units(
    tmp_path: Path,
) -> None:
    path = make_eventaid(tmp_path / "eventaid")
    dataset = EventAidRZipDataset(path.parent, max_events=None)
    try:
        report = _inspect_one_split(dataset, samples=0, validate_all=True)
    finally:
        dataset.close()

    diagnostics = report["event_timestamp_diagnostics"]
    assert diagnostics["validated_blocks"] == 3
    assert diagnostics["event_count"] == 240
    assert diagnostics["outside_interval_count"] == 0
    assert diagnostics["outside_interval_fraction"] == 0.0
    assert diagnostics["event_to_interval_span_ratio_min"] == pytest.approx(0.95)
    assert diagnostics["event_to_interval_span_ratio_max"] == pytest.approx(0.95)
    assert diagnostics["strict_interval_validation"] is False
