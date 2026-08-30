#!/usr/bin/env python3
"""Download the official EventHDR release directly, or import existing data."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import BinaryIO

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
MAX_DATASET_BYTES = 100_000_000_000
EXPECTED: dict[str, tuple[str, ...]] = {
    "train": tuple(f"{index}.h5" for index in range(1, 52)),
    "eval": tuple(f"{index}.h5" for index in range(1, 20)),
}


class ImportError(RuntimeError):
    """Raised when an EventHDR source cannot be imported safely."""


def _format_names(names: Iterable[str], limit: int = 8) -> str:
    ordered = sorted(names, key=lambda name: (len(name), name))
    preview = ", ".join(ordered[:limit])
    return preview + (" ..." if len(ordered) > limit else "")


def _validate_exact_names(names: Iterable[str], split: str, source: str) -> None:
    present = set(names)
    expected = set(EXPECTED[split])
    missing = expected - present
    extra = present - expected
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + _format_names(missing))
        if extra:
            details.append("extra=" + _format_names(extra))
        raise ImportError(
            f"{source} does not contain the exact official EventHDR {split} file set "
            f"({'; '.join(details)})"
        )


def _validate_magic(stream: BinaryIO, source: str) -> None:
    magic = stream.read(len(HDF5_MAGIC))
    if magic != HDF5_MAGIC:
        raise ImportError(f"Not an HDF5 file: {source}")


def _h5_files(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}:
            if path.name in files:
                raise ImportError(f"Duplicate HDF5 filename under {directory}: {path.name}")
            files[path.name] = path
    nested = [
        path
        for path in directory.rglob("*")
        if path.parent != directory and path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
    ]
    if nested:
        raise ImportError(
            f"Nested HDF5 files are not allowed under {directory}: "
            + _format_names(path.relative_to(directory).as_posix() for path in nested)
        )
    return files


def validate_split_dir(directory: Path, split: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise ImportError(f"EventHDR {split} directory does not exist: {directory}")
    files = _h5_files(directory)
    _validate_exact_names(files, split, str(directory))
    total_bytes = 0
    for name in EXPECTED[split]:
        path = files[name]
        total_bytes += path.stat().st_size
        with path.open("rb") as stream:
            _validate_magic(stream, str(path))
    if total_bytes >= MAX_DATASET_BYTES:
        raise ImportError(
            f"EventHDR {split} source is {total_bytes} bytes; the accepted dataset must be "
            "smaller than 100 GB"
        )
    return files


def _validate_combined_size(files_by_split: dict[str, dict[str, Path]], source: str) -> None:
    total_bytes = sum(
        path.stat().st_size
        for split_files in files_by_split.values()
        for path in split_files.values()
    )
    if total_bytes >= MAX_DATASET_BYTES:
        raise ImportError(
            f"EventHDR source is {total_bytes} bytes; the complete accepted dataset must be "
            "smaller than 100 GB: " + source
        )


def _candidate_split_dirs(source: Path, split: str) -> list[Path]:
    candidates = (source, source / split, source / "EventHDR" / split)
    result: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir() and any(
            child.is_file() and child.suffix.lower() in {".h5", ".hdf5"}
            for child in candidate.iterdir()
        ):
            resolved = candidate.resolve()
            if resolved not in result:
                result.append(resolved)
    return result


def locate_source(source: Path, splits: tuple[str, ...]) -> dict[str, Path]:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise ImportError(f"Source directory does not exist: {source}")
    located: dict[str, Path] = {}
    for split in splits:
        candidates = _candidate_split_dirs(source, split)
        valid: list[Path] = []
        failures: list[str] = []
        for candidate in candidates:
            try:
                validate_split_dir(candidate, split)
            except ImportError as error:
                failures.append(str(error))
            else:
                valid.append(candidate)
        if len(valid) != 1:
            if len(valid) > 1:
                raise ImportError(
                    f"Ambiguous EventHDR {split} source directories: "
                    + ", ".join(str(path) for path in valid)
                )
            detail = "; ".join(failures) if failures else "no candidate directory found"
            raise ImportError(f"Could not locate exact EventHDR {split} data: {detail}")
        located[split] = valid[0]
    _validate_combined_size(
        {split: validate_split_dir(path, split) for split, path in located.items()},
        str(source),
    )
    return located


def _destination_extras(directory: Path, split: str) -> set[str]:
    if not directory.exists() or directory.is_symlink():
        return set()
    if not directory.is_dir():
        raise ImportError(f"Destination is not a directory: {directory}")
    return set(_h5_files(directory)) - set(EXPECTED[split])


def _prepare_copy_destination(destination: Path, splits: tuple[str, ...]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ImportError(
            f"Copy mode refuses a symlinked EventHDR root; use --check or --link: {destination}"
        )
    for split in splits:
        split_dir = destination / split
        if split_dir.is_symlink():
            raise ImportError(f"Copy mode refuses a symlinked destination: {split_dir}")
        split_dir.mkdir(parents=True, exist_ok=True)
        extras = _destination_extras(split_dir, split)
        if extras:
            raise ImportError(
                f"Destination {split_dir} contains unexpected HDF5 files: " + _format_names(extras)
            )


def _copy_one(source: Path, target: Path) -> str:
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise ImportError(
                f"Refusing to overwrite a different existing file: {target} "
                f"({target.stat().st_size} != {source.stat().st_size} bytes)"
            )
        with target.open("rb") as stream:
            _validate_magic(stream, str(target))
        return "kept"

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".part", dir=target.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
        shutil.copy2(source, temporary)
        with temporary.open("rb") as stream:
            _validate_magic(stream, str(temporary))
        os.replace(temporary, target)
        temporary = None
        return "copied"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def copy_source(source_dirs: dict[str, Path], destination: Path) -> dict[str, int]:
    splits = tuple(source_dirs)
    source_files = {split: validate_split_dir(source_dirs[split], split) for split in splits}
    _prepare_copy_destination(destination, splits)
    counts = {"copied": 0, "kept": 0}
    for split in splits:
        for name in EXPECTED[split]:
            outcome = _copy_one(source_files[split][name], destination / split / name)
            counts[outcome] += 1
        validate_split_dir(destination / split, split)
    return counts


def _safe_zip_parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ImportError(f"Unsafe archive member path: {name!r}")
    return path.parts


def locate_archive_members(
    archive: zipfile.ZipFile, splits: tuple[str, ...]
) -> dict[str, dict[str, zipfile.ZipInfo]]:
    selected: dict[str, dict[str, zipfile.ZipInfo]] = {split: {} for split in splits}
    split_set = set(splits)
    for info in archive.infolist():
        if info.is_dir():
            continue
        parts = _safe_zip_parts(info.filename)
        name = parts[-1]
        if Path(name).suffix.lower() not in {".h5", ".hdf5"}:
            continue

        owner: str | None = None
        if len(splits) == 1 and len(parts) == 1:
            owner = splits[0]
        elif len(parts) >= 2 and parts[-2].lower() in split_set:
            owner = parts[-2].lower()
        if owner is None:
            raise ImportError(f"Cannot assign archive HDF5 member to train/eval: {info.filename}")
        if name in selected[owner]:
            raise ImportError(
                f"Archive contains duplicate EventHDR {owner} filename {name}: "
                f"{selected[owner][name].filename}, {info.filename}"
            )
        selected[owner][name] = info

    for split in splits:
        _validate_exact_names(selected[split], split, "archive")
        total_bytes = sum(info.file_size for info in selected[split].values())
        if total_bytes >= MAX_DATASET_BYTES:
            raise ImportError(
                f"EventHDR {split} archive content is {total_bytes} bytes; the accepted "
                "dataset must be smaller than 100 GB"
            )
        for name in EXPECTED[split]:
            with archive.open(selected[split][name], "r") as stream:
                _validate_magic(stream, f"archive::{selected[split][name].filename}")
    combined_bytes = sum(
        info.file_size for split_members in selected.values() for info in split_members.values()
    )
    if combined_bytes >= MAX_DATASET_BYTES:
        raise ImportError(
            f"EventHDR archive content is {combined_bytes} bytes; the complete accepted "
            "dataset must be smaller than 100 GB"
        )
    return selected


def _copy_archive_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> str:
    if target.exists():
        if target.stat().st_size != info.file_size:
            raise ImportError(
                f"Refusing to overwrite a different existing file: {target} "
                f"({target.stat().st_size} != {info.file_size} bytes)"
            )
        with target.open("rb") as stream:
            _validate_magic(stream, str(target))
        return "kept"

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".part", dir=target.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            with archive.open(info, "r") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        with temporary.open("rb") as stream:
            _validate_magic(stream, str(temporary))
        os.replace(temporary, target)
        temporary = None
        return "copied"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def copy_archive(archive_path: Path, destination: Path, splits: tuple[str, ...]) -> dict[str, int]:
    archive_path = archive_path.expanduser().resolve()
    if not archive_path.is_file():
        raise ImportError(f"Archive does not exist: {archive_path}")
    try:
        archive = zipfile.ZipFile(archive_path, "r")
    except (OSError, zipfile.BadZipFile) as error:
        raise ImportError(f"Invalid ZIP archive {archive_path}: {error}") from error
    with archive:
        members = locate_archive_members(archive, splits)
        _prepare_copy_destination(destination, splits)
        counts = {"copied": 0, "kept": 0}
        for split in splits:
            for name in EXPECTED[split]:
                outcome = _copy_archive_member(
                    archive, members[split][name], destination / split / name
                )
                counts[outcome] += 1
            validate_split_dir(destination / split, split)
        return counts


def link_source(source_dirs: dict[str, Path], destination: Path) -> dict[str, int]:
    for split, source_dir in source_dirs.items():
        validate_split_dir(source_dir, split)
    destination.mkdir(parents=True, exist_ok=True)
    linked = 0
    kept = 0
    for split, source_dir in source_dirs.items():
        target = destination / split
        if target.is_symlink():
            if target.resolve() != source_dir.resolve():
                raise ImportError(f"Destination symlink points elsewhere: {target}")
            kept += 1
            continue
        if target.exists():
            if not target.is_dir() or any(target.iterdir()):
                raise ImportError(
                    f"Refusing to replace a non-empty destination; move it first: {target}"
                )
            target.rmdir()
        target.symlink_to(source_dir.resolve(), target_is_directory=True)
        linked += 1
    for split in source_dirs:
        validate_split_dir(destination / split, split)
    return {"linked": linked, "kept": kept}


def check_destination(destination: Path, splits: tuple[str, ...]) -> None:
    files_by_split = {split: validate_split_dir(destination / split, split) for split in splits}
    _validate_combined_size(files_by_split, str(destination))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download/import/check the complete official EventHDR train (1-51) and "
            "eval (1-19) HDF5 release. --download needs no browser or user login."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--download", action="store_true", help="download directly from the official public share"
    )
    mode.add_argument("--source", type=Path, help="extracted EventHDR/train/eval source")
    mode.add_argument("--archive", type=Path, help="browser-downloaded ZIP archive")
    mode.add_argument("--check", action="store_true", help="check files already in destination")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/EventHDR"),
        help="logical EventHDR destination (default: data/EventHDR)",
    )
    parser.add_argument(
        "--split",
        choices=tuple(EXPECTED),
        help="import/check only one separately downloaded train or eval folder",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="symlink an extracted/shared source instead of copying it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = (
        Path(os.path.abspath(args.destination.expanduser()))
        if args.download
        else args.destination.expanduser().resolve()
    )
    splits = (args.split,) if args.split else tuple(EXPECTED)
    try:
        if args.download:
            if args.link:
                raise ImportError("--link requires --source")
            if __package__:
                from . import hdr_http
            else:
                try:
                    import hdr_http
                except ModuleNotFoundError as error:
                    if error.name != "hdr_http":
                        raise
                    from scripts import hdr_http
            try:
                counts = hdr_http.download_dataset(
                    destination, {split: EXPECTED[split] for split in splits}
                )
                check_destination(destination, splits)
            except ImportError:
                raise ImportError(
                    "EventHDR failed final file-set validation; no incomplete data accepted"
                ) from None
            except (hdr_http.DownloadError, OSError) as error:
                detail = (
                    str(error)
                    if isinstance(error, hdr_http.DownloadError)
                    else type(error).__name__
                )
                raise ImportError(detail) from None
            print(
                "EventHDR download passed: "
                + ", ".join(f"{key}={value}" for key, value in counts.items())
            )
            return 0
        if args.link and args.source is None:
            raise ImportError("--link requires --source")
        if args.check:
            check_destination(destination, splits)
            print(
                f"EventHDR check passed: {destination} "
                + ", ".join(f"{split}={len(EXPECTED[split])}" for split in splits)
            )
            return 0
        if args.source is not None:
            source_dirs = locate_source(args.source, splits)
            counts = (
                link_source(source_dirs, destination)
                if args.link
                else copy_source(source_dirs, destination)
            )
        else:
            counts = copy_archive(args.archive, destination, splits)
        check_destination(destination, splits)
        print(
            f"EventHDR import passed: {destination} "
            + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        )
        return 0
    except (ImportError, OSError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
