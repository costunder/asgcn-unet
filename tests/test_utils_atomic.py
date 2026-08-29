from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from asgcn_unet import utils


def _temporary_files(directory: Path, target_name: str) -> list[Path]:
    return list(directory.glob(f".{target_name}.*.tmp"))


def test_save_json_is_strict_atomic_and_preserves_existing_file_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "metrics.json"
    target.write_text('{"old": true}\n', encoding="utf-8")
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def tracked_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(utils.os, "replace", tracked_replace)
    utils.save_json(target, {"value": 1.25})

    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1.25}
    assert replacements and replacements[-1][1] == target
    assert not _temporary_files(tmp_path, target.name)

    with pytest.raises(ValueError, match="Out of range float values"):
        utils.save_json(target, {"invalid": float("nan")})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1.25}
    assert not _temporary_files(tmp_path, target.name)


def test_write_frame_csv_is_atomic_and_rejects_inconsistent_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "frames.csv"
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def tracked_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(utils.os, "replace", tracked_replace)
    utils.write_frame_csv(target, [{"sample": "a", "ssim": 0.5}])

    assert target.read_text(encoding="utf-8").splitlines() == ["sample,ssim", "a,0.5"]
    assert replacements and replacements[-1][1] == target
    assert not _temporary_files(tmp_path, target.name)

    with pytest.raises(ValueError, match="keys do not match"):
        utils.write_frame_csv(target, [{"sample": "a"}, {"sample": "b", "ssim": 0.6}])
    assert target.read_text(encoding="utf-8").splitlines() == ["sample,ssim", "a,0.5"]


def test_atomic_torch_save_replaces_target_without_fixed_temp_name(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.pt"
    utils.atomic_torch_save({"tensor": torch.tensor([1.0, 2.0])}, target)

    payload = torch.load(target, map_location="cpu", weights_only=True)
    assert torch.equal(payload["tensor"], torch.tensor([1.0, 2.0]))
    assert not _temporary_files(tmp_path, target.name)
