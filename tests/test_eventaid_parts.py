from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from asgcn_unet.cli import _inspect_one_split
from asgcn_unet.data import EventAidRZipDataset
from asgcn_unet.engine import _continues_sequence, _sample_sequence_info
from tests.fixtures import make_eventaid

_PARTS = ((1, 3), (4, 6), (11, 14), (21, 21))
_IDS = [frame_id for start, end in _PARTS for frame_id in range(start, end + 1)]


def _upload_members() -> dict[str, bytes]:
    """Represent the official upload layout with tiny temporary encoded images."""
    buffer = io.BytesIO()
    Image.fromarray(np.full((8, 12, 3), 128, dtype=np.uint8)).save(buffer, format="JPEG")
    members = {
        "shape.txt": b"12 8\n",
        "parts.txt": (
            "This group is split into four parts:\n"
            + "\n".join(f"{start}~{end}" for start, end in _PARTS)
            + "\n"
        ).encode(),
        "timestamps_upload.txt": "\n".join(
            str(1000 + rank * 1000) for rank in range(len(_IDS))
        ).encode(),
    }
    for rank, frame_id in enumerate(_IDS):
        t0 = 1000 + rank * 1000
        members[f"event_upload/{frame_id:06d}.txt"] = (
            f"{t0 + 100} 1 2 0\n{t0 + 900} 3 4 1\n".encode()
        )
        members[f"gt_upload/{frame_id:06d}_img.jpg"] = buffer.getvalue()
    return members


def _write_upload(root: Path, members: dict[str, bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "R-parts.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, content in members.items():
            archive.writestr(member, content)
    return path


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_upload_pairs_only_within_parts_and_maps_timestamps_by_sorted_id_rank(
    tmp_path: Path, offset: int
) -> None:
    _write_upload(tmp_path, _upload_members())
    dataset = EventAidRZipDataset(tmp_path, target_offset=offset, max_events=None)
    expected = [
        (part_index, frame_id)
        for part_index, (start, end) in enumerate(_PARTS)
        for frame_id in range(start, end)
        if start <= frame_id + offset <= end
    ]
    try:
        assert len(dataset) == len(expected)
        assert [item["frame_id"] for item in dataset.samples] == [item[1] for item in expected]
        for index, (part_index, frame_id) in enumerate(expected):
            rank = _IDS.index(frame_id)
            sample = dataset[index]
            metadata = sample["metadata"]
            assert sample["sample_id"] == f"R-parts/{frame_id:06d}"
            assert dataset.samples[index]["target_name"] == (
                f"gt_upload/{frame_id + offset:06d}_img.jpg"
            )
            assert metadata["scene"] == "R-parts"
            assert metadata["part_index"] == part_index
            assert metadata["sequence_id"] == f"R-parts/part-{part_index:03d}"
            assert metadata["sequence_index"] == frame_id
            assert metadata["t0_us"] == 1000 + rank * 1000
            assert metadata["t1_us"] == 2000 + rank * 1000
            assert metadata["dt_us"] == 1000
            assert metadata["event_timestamp_diagnostics"]["outside_interval_count"] == 0
            np.testing.assert_allclose(sample["target"].numpy(), 128 / 255, rtol=1e-6)
        report = _inspect_one_split(dataset, samples=0, validate_all=True)
        assert report["validated_samples"] == len(expected)
        assert set(report["scenes"]) == {"R-parts"}
        assert report["scenes"]["R-parts"]["frames"] == len(_IDS)
        assert report["scenes"]["R-parts"]["events"] == len(_IDS)
        assert report["scenes"]["R-parts"]["paired_samples"] == len(expected)
        assert report["scenes"]["R-parts"]["parts"][-1]["paired_samples"] == 0
    finally:
        dataset.close()


def test_upload_accepts_nested_case_insensitive_members_and_backslash_paths(tmp_path: Path) -> None:
    members = {
        "nested\\R-parts\\" + name.upper().replace("/", "\\"): content
        for name, content in _upload_members().items()
    }
    _write_upload(tmp_path, members)
    dataset = EventAidRZipDataset(tmp_path, max_events=None)
    try:
        report = _inspect_one_split(dataset, samples=0, validate_all=True)
        assert report["validated_samples"] == 7
        assert report["scenes"]["R-parts"]["target_formats"] == {"jpg": len(_IDS)}
    finally:
        dataset.close()


def test_part_sequence_identity_resets_recurrence_even_for_adjacent_frame_ids() -> None:
    def sequence_info(part: int, frame_id: int) -> tuple[str, int | None, tuple[int, int]]:
        return _sample_sequence_info(
            {
                "sensor_size": (8, 12),
                "metadata": {
                    "scene": "R-parts",
                    "sequence_id": f"R-parts/part-{part:03d}",
                    "sequence_index": frame_id,
                },
            }
        )

    previous = sequence_info(0, 3)
    adjacent_part = sequence_info(1, 4)
    same_part_next = sequence_info(1, 5)
    assert previous[0] == "R-parts/part-000"
    assert adjacent_part[0] == "R-parts/part-001"
    assert not _continues_sequence(*adjacent_part, *previous)
    assert _continues_sequence(*same_part_next, *adjacent_part)


@pytest.mark.parametrize(
    "parts",
    [
        b"",
        b"This group is split into four parts:\n1-3\n4~6\n11~14\n21~21\n",
        b"This group is split into four parts:\n1~3junk\n4~6\n11~14\n21~21\n",
        b"This group is split into four parts:\n1~4\n4~6\n11~14\n21~21\n",
        b"This group is split into four parts:\n4~6\n1~3\n11~14\n21~21\n",
        b"This group is split into four parts:\n0~3\n4~6\n11~14\n21~21\n",
        b"This group is split into four parts:\n3~1\n4~6\n11~14\n21~21\n",
    ],
)
def test_upload_rejects_invalid_part_declarations(tmp_path: Path, parts: bytes) -> None:
    members = _upload_members()
    members["parts.txt"] = parts
    _write_upload(tmp_path, members)
    with pytest.raises(ValueError, match="(?i)part"):
        EventAidRZipDataset(tmp_path)


def test_upload_requires_parts_metadata(tmp_path: Path) -> None:
    members = _upload_members()
    del members["parts.txt"]
    _write_upload(tmp_path, members)
    with pytest.raises(ValueError, match="parts.txt"):
        EventAidRZipDataset(tmp_path)


def test_upload_does_not_silently_drop_an_entire_unpairable_scene(tmp_path: Path) -> None:
    make_eventaid(tmp_path, frames=8)
    _write_upload(tmp_path, _upload_members())
    # The regular archive can pair this offset, but every upload part is too short.
    with pytest.raises(ValueError, match="R-parts.*pairing"):
        EventAidRZipDataset(tmp_path, target_offset=5)


@pytest.mark.parametrize("missing", ["event_upload/000021.txt", "gt_upload/000021_img.jpg"])
def test_upload_requires_coverage_even_for_unpaired_singleton_part(
    tmp_path: Path, missing: str
) -> None:
    members = _upload_members()
    del members[missing]
    _write_upload(tmp_path, members)
    with pytest.raises(ValueError, match="(?i)part|coverage|IDs"):
        EventAidRZipDataset(tmp_path)


def test_upload_rejects_members_not_declared_in_parts(tmp_path: Path) -> None:
    members = _upload_members()
    members["event_upload/000030.txt"] = b"12000 1 2 1\n"
    members["gt_upload/000030_img.jpg"] = members["gt_upload/000001_img.jpg"]
    _write_upload(tmp_path, members)
    with pytest.raises(ValueError, match="(?i)part|coverage|IDs"):
        EventAidRZipDataset(tmp_path)


@pytest.mark.parametrize("difference", [-1, 1])
def test_upload_requires_exact_timestamp_row_count(tmp_path: Path, difference: int) -> None:
    members = _upload_members()
    members["timestamps_upload.txt"] = "\n".join(
        str(1000 + rank * 1000) for rank in range(len(_IDS) + difference)
    ).encode()
    _write_upload(tmp_path, members)
    with pytest.raises(ValueError, match="(?i)timestamp"):
        EventAidRZipDataset(tmp_path)


@pytest.mark.parametrize("member", ["event/000001.txt", "gt/000001_img.jpg"])
def test_upload_rejects_mixed_regular_and_upload_layouts(tmp_path: Path, member: str) -> None:
    members = _upload_members()
    members[member] = (
        b"1000 1 2 1\n" if member.startswith("event/") else members["gt_upload/000001_img.jpg"]
    )
    _write_upload(tmp_path, members)
    with pytest.raises(ValueError, match="(?i)mix|layout"):
        EventAidRZipDataset(tmp_path)


@pytest.mark.parametrize(
    ("source", "duplicate"),
    [
        ("parts.txt", "nested/PARTS.TXT"),
        ("timestamps_upload.txt", "nested/TIMESTAMPS_UPLOAD.TXT"),
        ("event_upload/000001.txt", "nested/event_upload/1.txt"),
        ("gt_upload/000001_img.jpg", "nested/gt_upload/1_img.jpeg"),
    ],
)
def test_upload_rejects_duplicate_metadata_or_numeric_member_ids(
    tmp_path: Path, source: str, duplicate: str
) -> None:
    members = _upload_members()
    members[duplicate] = members[source]
    _write_upload(tmp_path, members)
    with pytest.raises(ValueError, match="(?i)duplicate"):
        EventAidRZipDataset(tmp_path)
