from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from asgcn_unet.cli import _inspect_one_split
from asgcn_unet.data import EventAidRZipDataset


def _image_bytes(frame_id: int, suffix: str) -> bytes:
    """Encode actual image bytes; test fixtures exist only under pytest tmp_path."""
    pixels = np.full((8, 12, 3), frame_id * 40, dtype=np.uint8)
    buffer = io.BytesIO()
    image_format = "PNG" if suffix.casefold() == "png" else "JPEG"
    Image.fromarray(pixels).save(buffer, format=image_format)
    return buffer.getvalue()


def _scene_members(
    suffixes: tuple[str, ...] = ("jpg", "jpg", "jpg", "jpg"),
    *,
    prefix: str = "",
    separator: str = "/",
) -> dict[str, bytes]:
    def member(relative: str) -> str:
        relative = relative.replace("/", separator)
        return f"{prefix}{separator}{relative}" if prefix else relative

    members = {
        member("shape.txt"): b"12 8\n",
        member("timestamps.txt"): b"1000\n2000\n3000\n4000\n",
    }
    for frame_id, suffix in enumerate(suffixes, 1):
        t0 = frame_id * 1000
        members[member(f"EvEnT/{frame_id:06d}.txt")] = (
            f"{t0 + 100} 1 2 0\n{t0 + 900} 3 4 1\n".encode()
        )
        members[member(f"Gt/{frame_id:06d}_img.{suffix}")] = _image_bytes(frame_id, suffix)
    return members


def _write_scene(root: Path, members: dict[str, bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "R-codec.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, content in members.items():
            archive.writestr(member, content)
    return path


@pytest.mark.parametrize("suffix", ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"])
@pytest.mark.parametrize(
    ("prefix", "separator"),
    [("", "/"), ("nested/R-codec", "/"), (r"nested\R-codec", "\\")],
)
def test_eventaid_decodes_image_formats_with_unchanged_offset_and_timestamps(
    tmp_path: Path, suffix: str, prefix: str, separator: str
) -> None:
    path = _write_scene(
        tmp_path,
        _scene_members((suffix,) * 4, prefix=prefix, separator=separator),
    )
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset = EventAidRZipDataset(tmp_path, max_events=None)
    try:
        assert len(dataset) == 3
        assert dataset.scene_info["R-codec"]["target_formats"] == {suffix.lower(): 4}
        for sample_index in range(3):
            sample = dataset[sample_index]
            frame_id = sample_index + 1
            target_id = frame_id + 1
            assert sample["sample_id"] == f"R-codec/{frame_id:06d}"
            assert sample["target"].shape == (1, 8, 12)
            np.testing.assert_allclose(
                sample["target"].numpy(), target_id * 40 / 255, rtol=1e-6
            )
            assert dataset.samples[sample_index]["target_name"].endswith(
                f"{target_id:06d}_img.{suffix}"
            )
            metadata = sample["metadata"]
            assert metadata["sequence_index"] == frame_id
            assert metadata["t0_us"] == frame_id * 1000
            assert metadata["t1_us"] == (frame_id + 1) * 1000
            assert metadata["dt_us"] == 1000
            assert metadata["raw_event_count"] == 2
            assert metadata["retained_event_count"] == 2
            assert metadata["event_timestamp_diagnostics"]["outside_interval_count"] == 0
            np.testing.assert_array_equal(sample["events"][:, :2], [[1, 2], [3, 4]])
    finally:
        dataset.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash


def test_eventaid_full_inspect_reports_mixed_target_formats_including_boundary_frame(
    tmp_path: Path,
) -> None:
    _write_scene(tmp_path, _scene_members(("PNG", "jpg", "JPEG", "png")))
    dataset = EventAidRZipDataset(tmp_path, max_events=None)
    try:
        report = _inspect_one_split(dataset, samples=0, validate_all=True)
    finally:
        dataset.close()
    assert report["samples"] == 3
    assert report["validated_samples"] == 3
    assert report["scenes"]["R-codec"]["frames"] == 4
    assert report["scenes"]["R-codec"]["events"] == 4
    assert report["scenes"]["R-codec"]["target_formats"] == {"png": 2, "jpg": 1, "jpeg": 1}
    assert report["event_timestamp_diagnostics"]["validated_blocks"] == 3


@pytest.mark.parametrize("suffixes", [("png", "jpg"), ("jpg", "jpeg"), ("PNG", "JPEG")])
def test_eventaid_rejects_duplicate_numeric_gt_ids_across_image_formats(
    tmp_path: Path, suffixes: tuple[str, str]
) -> None:
    original, duplicate = suffixes
    members = _scene_members((original,) * 4)
    members[f"nested/gt/2_img.{duplicate}"] = _image_bytes(2, duplicate)
    _write_scene(tmp_path, members)

    with pytest.raises(ValueError, match="duplicate numeric GT ID 2"):
        EventAidRZipDataset(tmp_path)


@pytest.mark.parametrize(
    "misleading_name",
    [
        "gt/000001_img.jpg.bak",
        "gt/000001_img.png.txt",
        "gt/000001_img.jpeg/extra",
        "gt/000001_img.webp",
        "gt/000001_img.txt",
        "notgt/000001_img.jpg",
        "gt/000001_image.jpg",
    ],
)
def test_eventaid_rejects_scenes_with_only_misleading_or_unsupported_gt_members(
    tmp_path: Path, misleading_name: str
) -> None:
    members = {name: raw for name, raw in _scene_members().items() if not name.startswith("Gt/")}
    members[misleading_name] = _image_bytes(1, "jpg")
    _write_scene(tmp_path, members)

    with pytest.raises(ValueError) as captured:
        EventAidRZipDataset(tmp_path)

    message = str(captured.value).casefold()
    assert "gt" in message
    assert "png" in message
    assert "jpg" in message


def test_eventaid_ignores_unrelated_members_without_counting_or_pairing_them(
    tmp_path: Path,
) -> None:
    members = _scene_members()
    members.update(
        {
            "gt/000002_img.jpg.bak": b"not an image",
            "gt/000002_img.png.txt": b"not an image",
            "preview/000002_img.png": _image_bytes(2, "png"),
            "gt/000002_img.webp": b"not a supported target",
        }
    )
    _write_scene(tmp_path, members)
    dataset = EventAidRZipDataset(tmp_path)
    try:
        assert len(dataset) == 3
        assert dataset.scene_info["R-codec"]["target_formats"] == {"jpg": 4}
        assert dataset.scene_info["R-codec"]["frames"] == 4
        for index in range(len(dataset)):
            assert dataset[index]["target"].shape == (1, 8, 12)
    finally:
        dataset.close()


def test_eventaid_does_not_substitute_or_skip_corrupt_jpeg_targets(tmp_path: Path) -> None:
    members = _scene_members()
    members["Gt/000002_img.jpg"] = b"this is not a JPEG image"
    _write_scene(tmp_path, members)
    dataset = EventAidRZipDataset(tmp_path)
    try:
        assert len(dataset) == 3
        with pytest.raises((OSError, ValueError), match="image|target"):
            dataset[0]
        with pytest.raises((OSError, ValueError), match="image|target"):
            _inspect_one_split(dataset, samples=0, validate_all=True)
    finally:
        dataset.close()
