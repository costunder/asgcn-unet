from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest

from asgcn_recon.data import (
    EventAidRZipDataset,
    EventHDRDataset,
    build_dataset,
    load_eventhdr_split_manifest,
)
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


@pytest.mark.parametrize("attribute", ["event_idx", "timestamp"])
def test_eventhdr_requires_image_boundary_attributes(tmp_path: Path, attribute: str) -> None:
    path = make_eventhdr(tmp_path / "hdr")
    with h5py.File(path, "a") as h5:
        del h5["images/image000000000"].attrs[attribute]

    with pytest.raises(ValueError, match=rf"missing '{attribute}'"):
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


def test_final_eventhdr_manifest_normalizes_physical_scene_groups(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "FINAL",
            "scene_groups": {
                "night-drive-a": ["chunk_02.h5", "nested\\chunk_01.hdf5"],
                "night-drive-b": ["validation.h5"],
            },
            "train_scenes": ["night-drive-a"],
            "val_scenes": ["night-drive-b"],
        },
    )

    manifest = load_eventhdr_split_manifest(manifest_path)

    assert manifest["status"] == "final"
    assert manifest["split_schema"] == "physical_scenes_v1"
    assert manifest["train_files"] == ["chunk_02.h5", "nested/chunk_01.hdf5"]
    assert manifest["val_files"] == ["validation.h5"]
    assert manifest["file_to_scene"] == {
        "chunk_02.h5": "night-drive-a",
        "nested/chunk_01.hdf5": "night-drive-a",
        "validation.h5": "night-drive-b",
    }


def test_final_eventhdr_manifest_rejects_legacy_file_lists(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "train_files": ["train.h5"],
            "val_files": ["val.h5"],
        },
    )

    with pytest.raises(ValueError, match="requires scene_groups"):
        load_eventhdr_split_manifest(manifest_path)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"val_scenes": ["scene-a"]}, "leaks physical scenes"),
        (
            {
                "scene_groups": {
                    "scene-a": ["shared.h5"],
                    "scene-b": ["shared.h5"],
                }
            },
            "multiple physical scenes",
        ),
        ({"val_scenes": ["undefined"]}, "undefined physical scenes"),
        ({"scene_groups": {"scene-a": [], "scene-b": ["b.h5"]}}, "non-empty list"),
        (
            {
                "scene_groups": {
                    "scene-a": ["a.h5"],
                    "scene-b": ["b.h5"],
                    "scene-c": ["c.h5"],
                }
            },
            "leaves physical scenes unassigned",
        ),
    ],
)
def test_physical_scene_manifest_rejects_leakage_and_invalid_ownership(
    tmp_path: Path, update: dict, message: str
) -> None:
    payload = {
        "status": "final",
        "scene_groups": {"scene-a": ["a.h5"], "scene-b": ["b.h5"]},
        "train_scenes": ["scene-a"],
        "val_scenes": ["scene-b"],
    }
    payload.update(update)
    manifest_path = _write_manifest(tmp_path / "split.json", payload)

    with pytest.raises(ValueError, match=message):
        load_eventhdr_split_manifest(manifest_path)


def test_physical_scene_manifest_rejects_incomplete_schema(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "provisional",
            "scene_groups": {"scene-a": ["a.h5"]},
            "train_files": ["a.h5"],
            "val_files": ["b.h5"],
        },
    )

    with pytest.raises(ValueError, match=r"incomplete.*train_scenes, val_scenes"):
        load_eventhdr_split_manifest(manifest_path)


def test_factory_assigns_physical_scene_and_retains_source_file(tmp_path: Path) -> None:
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root / "chunk-one")
    make_eventhdr(data_root / "chunk-two")
    make_eventhdr(data_root / "held-out")
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "scene_groups": {
                "physical-night-drive": [
                    "chunk-two/test.h5",
                    "chunk-one/test.h5",
                ],
                "physical-day-drive": ["held-out/test.h5"],
            },
            "train_scenes": ["physical-night-drive"],
            "val_scenes": ["physical-day-drive"],
        },
    )
    config = {
        "type": "eventhdr",
        "root": str(data_root),
        "split_manifest": str(manifest_path),
        "max_events": 8,
    }

    train_dataset = build_dataset(config, split="train")
    first = train_dataset[0]
    second_file = train_dataset[4]

    assert train_dataset.samples[0]["scene"] == "physical-night-drive"
    assert first["metadata"]["scene"] == "physical-night-drive"
    assert first["metadata"]["source_file"] == "chunk-two/test.h5"
    assert second_file["metadata"]["source_file"] == "chunk-one/test.h5"
    assert first["sample_id"] != second_file["sample_id"]

    val_sample = build_dataset(config, split="val")[0]
    assert val_sample["metadata"]["scene"] == "physical-day-drive"
    assert val_sample["metadata"]["source_file"] == "held-out/test.h5"


def test_final_manifest_must_cover_every_h5_under_root(tmp_path: Path) -> None:
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root / "train").rename(data_root / "train" / "a.h5")
    make_eventhdr(data_root / "val").rename(data_root / "val" / "b.h5")
    make_eventhdr(data_root / "extra").rename(data_root / "extra" / "c.h5")
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "scene_groups": {
                "scene-a": ["train/a.h5"],
                "scene-b": ["val/b.h5"],
            },
            "train_scenes": ["scene-a"],
            "val_scenes": ["scene-b"],
        },
    )

    with pytest.raises(ValueError, match="must cover every H5.*extra/c.h5"):
        build_dataset(
            {
                "type": "eventhdr",
                "root": str(data_root),
                "split_manifest": str(manifest_path),
            },
            split="train",
        )


def test_final_manifest_checks_separate_roots_without_collapsing_same_names(
    tmp_path: Path,
) -> None:
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    make_eventhdr(train_root).rename(train_root / "a.h5")
    make_eventhdr(val_root).rename(val_root / "b.h5")
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "final",
            "scene_groups": {
                "training-scene": ["a.h5"],
                "validation-scene": ["b.h5"],
            },
            "train_scenes": ["training-scene"],
            "val_scenes": ["validation-scene"],
        },
    )
    config = {
        "type": "eventhdr",
        "root": str(train_root),
        "val_root": str(val_root),
        "split_manifest": str(manifest_path),
    }

    build_dataset(config, split="train").close()
    build_dataset(config, split="val").close()

    make_eventhdr(tmp_path / "extra").rename(val_root / "a.h5")
    with pytest.raises(ValueError, match=r"dataset\.val_root.*undeclared: a\.h5"):
        build_dataset(config, split="val")


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
            "scene_groups": {
                "training-scene": ["train.h5"],
                "validation-scene": ["val.h5"],
            },
            "train_scenes": ["training-scene"],
            "val_scenes": ["validation-scene"],
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


def test_provisional_legacy_manifest_keeps_file_identity_as_scene(tmp_path: Path) -> None:
    data_root = tmp_path / "hdr"
    make_eventhdr(data_root)
    manifest_path = _write_manifest(
        tmp_path / "split.json",
        {
            "status": "provisional",
            "train_files": ["test.h5"],
            "val_files": ["unused.h5"],
        },
    )

    manifest = load_eventhdr_split_manifest(manifest_path)
    dataset = build_dataset(
        {
            "type": "eventhdr",
            "root": str(data_root),
            "split_manifest": str(manifest_path),
        },
        split="train",
    )

    assert manifest["split_schema"] == "legacy_files_v1"
    assert dataset[0]["metadata"]["scene"] == "test.h5"
    assert dataset[0]["metadata"]["source_file"] == "test.h5"


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
