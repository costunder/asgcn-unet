from __future__ import annotations

import importlib.util
import os
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "get_hdr.py"
    spec = importlib.util.spec_from_file_location("get_hdr", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


get_hdr = _load_script()


def _write_split(root: Path, split: str, *, missing: str | None = None) -> Path:
    directory = root / split
    directory.mkdir(parents=True)
    for name in get_hdr.EXPECTED[split]:
        if name != missing:
            (directory / name).write_bytes(get_hdr.HDF5_MAGIC + name.encode("ascii"))
    return directory


def _write_complete(root: Path) -> Path:
    _write_split(root, "train")
    _write_split(root, "eval")
    return root


def test_copy_complete_source_and_idempotent_check(tmp_path: Path) -> None:
    source = _write_complete(tmp_path / "download" / "EventHDR")
    destination = tmp_path / "data" / "EventHDR"

    assert get_hdr.main(["--source", str(source), "--destination", str(destination)]) == 0
    assert get_hdr.main(["--check", "--destination", str(destination)]) == 0
    assert get_hdr.main(["--source", str(source), "--destination", str(destination)]) == 0
    assert len(list((destination / "train").glob("*.h5"))) == 51
    assert len(list((destination / "eval").glob("*.h5"))) == 19


def test_missing_source_is_rejected_before_destination_creation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "download"
    _write_split(source, "train", missing="51.h5")
    _write_split(source, "eval")
    destination = tmp_path / "data" / "EventHDR"

    assert get_hdr.main(["--source", str(source), "--destination", str(destination)]) == 1
    assert not destination.exists()
    assert "missing=51.h5" in capsys.readouterr().err


def test_extra_or_nested_h5_is_rejected(tmp_path: Path) -> None:
    source = _write_complete(tmp_path / "download")
    (source / "train" / "52.h5").write_bytes(get_hdr.HDF5_MAGIC)
    with pytest.raises(get_hdr.ImportError, match="extra=52.h5"):
        get_hdr.locate_source(source, ("train", "eval"))

    (source / "train" / "52.h5").unlink()
    nested = source / "train" / "nested"
    nested.mkdir()
    (nested / "1.h5").write_bytes(get_hdr.HDF5_MAGIC)
    with pytest.raises(get_hdr.ImportError, match="Nested HDF5"):
        get_hdr.validate_split_dir(source / "train", "train")


def test_separate_split_source_directory_is_supported(tmp_path: Path) -> None:
    source = _write_split(tmp_path / "browser-download", "eval")
    destination = tmp_path / "data" / "EventHDR"
    assert (
        get_hdr.main(
            [
                "--source",
                str(source),
                "--split",
                "eval",
                "--destination",
                str(destination),
            ]
        )
        == 0
    )
    assert len(list((destination / "eval").glob("*.h5"))) == 19
    assert not (destination / "train").exists()


def test_archive_import_streams_exact_members_without_extract_tree(tmp_path: Path) -> None:
    archive_path = tmp_path / "EventHDR.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for split, names in get_hdr.EXPECTED.items():
            for name in names:
                archive.writestr(f"EventHDR/{split}/{name}", get_hdr.HDF5_MAGIC + b"data")
        archive.writestr("EventHDR/pretrained/model.pt", b"ignored")
    destination = tmp_path / "data" / "EventHDR"

    assert (
        get_hdr.main(
            ["--archive", str(archive_path), "--destination", str(destination)]
        )
        == 0
    )
    assert len(list((destination / "train").glob("*.h5"))) == 51
    assert len(list((destination / "eval").glob("*.h5"))) == 19
    assert not (destination / "pretrained").exists()


def test_archive_rejects_unsafe_or_duplicate_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in get_hdr.EXPECTED["eval"]:
            archive.writestr(f"eval/{name}", get_hdr.HDF5_MAGIC)
        archive.writestr("../eval/other.h5", get_hdr.HDF5_MAGIC)
    with (
        zipfile.ZipFile(archive_path) as archive,
        pytest.raises(get_hdr.ImportError, match="Unsafe archive member"),
    ):
        get_hdr.locate_archive_members(archive, ("eval",))

    duplicate_path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate_path, "w") as archive:
        for name in get_hdr.EXPECTED["eval"]:
            archive.writestr(f"eval/{name}", get_hdr.HDF5_MAGIC)
        archive.writestr("prefix/eval/1.h5", get_hdr.HDF5_MAGIC)
    with (
        zipfile.ZipFile(duplicate_path) as archive,
        pytest.raises(get_hdr.ImportError, match="duplicate"),
    ):
        get_hdr.locate_archive_members(archive, ("eval",))


def test_check_rejects_bad_hdf5_magic(tmp_path: Path) -> None:
    destination = _write_complete(tmp_path / "data" / "EventHDR")
    (destination / "train" / "26.h5").write_bytes(b"not-hdf5")
    with pytest.raises(get_hdr.ImportError, match="Not an HDF5"):
        get_hdr.check_destination(destination, ("train", "eval"))


def test_link_mode_uses_shared_storage_without_copy(tmp_path: Path) -> None:
    source = _write_complete(tmp_path / "shared" / "EventHDR")
    destination = tmp_path / "repo" / "data" / "EventHDR"
    try:
        result = get_hdr.link_source(
            {"train": source / "train", "eval": source / "eval"}, destination
        )
    except OSError as error:
        if os.name == "nt":
            pytest.skip(f"Windows symlink privilege is unavailable: {error}")
        raise
    assert result == {"linked": 2, "kept": 0}
    assert (destination / "train").is_symlink()
    assert (destination / "eval").is_symlink()
    get_hdr.check_destination(destination, ("train", "eval"))
