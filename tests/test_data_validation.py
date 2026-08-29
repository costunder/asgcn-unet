from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest

from asgcn_recon.data import EventAidRZipDataset, EventHDRDataset
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
