#!/usr/bin/env python3
"""Build or verify the deterministic repository code snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
METADATA_START = "<!-- code-summary-metadata\n"
METADATA_END = "\n-->\n"


def _run_git(root: Path, arguments: Sequence[str], *, allow_failure: bool = False) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if process.returncode != 0 and not allow_failure:
        raise ValueError(f"git {' '.join(arguments)} failed")
    return os.fsdecode(process.stdout).strip() if process.returncode == 0 else ""


def _git_root(start: Path) -> Path:
    value = _run_git(start, ["rev-parse", "--show-toplevel"])
    return Path(value).resolve()


def _tracked_paths(root: Path) -> list[Path]:
    process = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        raise ValueError("git ls-files failed")
    paths = [Path(os.fsdecode(item)) for item in process.stdout.split(b"\x00") if item]
    return sorted(paths, key=lambda path: path.as_posix().encode("utf-8", errors="surrogateescape"))


def _decode_text(data: bytes) -> str | None:
    if b"\x00" in data:
        return None
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    if text:
        controls = sum(ord(character) < 32 and character not in "\n\r\t\f" for character in text)
        if controls / len(text) > 0.01:
            return None
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _language(path: Path) -> str:
    by_suffix = {
        ".json": "json",
        ".md": "markdown",
        ".pbs": "bash",
        ".py": "python",
        ".sbatch": "bash",
        ".sh": "bash",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    return by_suffix.get(path.suffix.casefold(), "text")


def _snapshot_digest(files: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256(b"asgcn-unet-code-summary-v1\x00")
    for item in files:
        path_bytes = item["path"].encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(item["bytes"].to_bytes(8, "big"))
        digest.update(bytes.fromhex(item["sha256"]))
    return digest.hexdigest()


def _dirty_without_output(root: Path, output_relative: str) -> bool:
    for arguments in (
        ["diff", "--quiet", "--", ".", f":(exclude){output_relative}"],
        ["diff", "--cached", "--quiet", "--", ".", f":(exclude){output_relative}"],
    ):
        process = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.returncode == 1:
            return True
        if process.returncode != 0:
            raise ValueError("cannot determine tracked dirty state")
    return False


def _generated_timestamp(root: Path) -> tuple[str, str]:
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch is not None:
        try:
            epoch = int(source_date_epoch)
        except ValueError as error:
            raise ValueError("SOURCE_DATE_EPOCH must be an integer") from error
        if epoch < 0:
            raise ValueError("SOURCE_DATE_EPOCH must not be negative")
        source = "SOURCE_DATE_EPOCH"
    else:
        commit_epoch = _run_git(root, ["show", "-s", "--format=%ct", "HEAD"])
        if not commit_epoch:
            raise ValueError("cannot derive a reproducible timestamp without a Git commit")
        epoch = int(commit_epoch)
        source = "source_commit_time"
    generated = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return generated, source


def _provenance(root: Path, output_relative: str) -> dict[str, Any]:
    commit = _run_git(root, ["rev-parse", "HEAD"])
    tree = _run_git(root, ["rev-parse", "HEAD^{tree}"])
    branch = _run_git(root, ["symbolic-ref", "--quiet", "--short", "HEAD"], allow_failure=True)
    if not branch:
        branch = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME") or "DETACHED"
    generated_utc, timestamp_source = _generated_timestamp(root)
    dirty = _dirty_without_output(root, output_relative)
    return {
        # A dirty snapshot cannot truthfully claim the checked-out commit/tree as
        # its content identity. Omitting those SHAs also avoids retaining a pointer
        # to a superseded commit during sensitive-history cleanup.
        "source_commit_at_generation": None if dirty else commit,
        "source_tree_at_generation": None if dirty else tree,
        "branch_at_generation": branch,
        "tracked_tree_dirty_at_generation": dirty,
        "generated_utc": generated_utc,
        "timestamp_source": timestamp_source,
        "note": (
            "Dirty snapshots omit commit/tree identity; snapshot_sha256 is the "
            "verification identity."
        ),
    }


def _validate_preserved_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("summary provenance is missing")
    required = {
        "branch_at_generation": str,
        "tracked_tree_dirty_at_generation": bool,
        "generated_utc": str,
        "timestamp_source": str,
        "note": str,
    }
    for key, expected_type in required.items():
        if not isinstance(value.get(key), expected_type):
            raise TypeError(f"summary provenance field {key!r} is invalid")
    sha_pattern = re.compile(r"[0-9a-f]{40}")
    for key in ("source_commit_at_generation", "source_tree_at_generation"):
        identity = value.get(key)
        if identity is not None and (
            not isinstance(identity, str) or sha_pattern.fullmatch(identity) is None
        ):
            raise ValueError(f"summary provenance field {key!r} is invalid")
    commit_missing = value.get("source_commit_at_generation") is None
    tree_missing = value.get("source_tree_at_generation") is None
    if commit_missing != tree_missing:
        raise ValueError("summary commit/tree provenance must be present or omitted together")
    if value["tracked_tree_dirty_at_generation"] is not commit_missing:
        raise ValueError("dirty summary provenance must omit commit/tree identity")
    try:
        datetime.strptime(value["generated_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ValueError("summary generated_utc is invalid") from error
    return value


def _require_clean_provenance(
    root: Path,
    output_relative: str,
    provenance: dict[str, Any],
) -> None:
    """Require a committed source snapshot, allowing only the summary commit afterward."""

    if provenance["tracked_tree_dirty_at_generation"]:
        raise ValueError("clean code-summary provenance is required")
    commit = provenance.get("source_commit_at_generation")
    tree = provenance.get("source_tree_at_generation")
    if not isinstance(commit, str) or not isinstance(tree, str):
        raise TypeError("clean code-summary provenance must include source commit and tree")
    resolved_tree = _run_git(root, ["rev-parse", f"{commit}^{{tree}}"])
    if resolved_tree != tree:
        raise ValueError("code-summary source commit/tree provenance is inconsistent")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if ancestor.returncode != 0:
        raise ValueError("code-summary source commit is not an ancestor of HEAD")
    source_diff = subprocess.run(
        ["git", "diff", "--quiet", commit, "HEAD", "--", ".", f":(exclude){output_relative}"],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if source_diff.returncode == 1:
        raise ValueError("tracked source changed after the code-summary provenance commit")
    if source_diff.returncode != 0:
        raise ValueError("cannot validate code-summary source provenance")
    if _dirty_without_output(root, output_relative):
        raise ValueError("clean code-summary provenance requires no pending source changes")


def _read_existing_metadata(output: Path) -> dict[str, Any]:
    try:
        text = output.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read {output.name}: {error}") from error
    if not text.startswith(METADATA_START):
        raise ValueError(f"{output.name} has no code-summary metadata")
    end = text.find(METADATA_END, len(METADATA_START))
    if end < 0:
        raise ValueError(f"{output.name} metadata is unterminated")
    try:
        value = json.loads(text[len(METADATA_START) : end])
    except json.JSONDecodeError as error:
        raise ValueError(f"{output.name} metadata is invalid JSON") from error
    if not isinstance(value, dict) or value.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{output.name} metadata version is unsupported")
    _validate_preserved_provenance(value.get("provenance"))
    return value


def _collect_snapshot(root: Path, output_relative: str) -> tuple[list[dict[str, Any]], list[str]]:
    files: list[dict[str, Any]] = []
    skipped_binary: list[str] = []
    for relative in _tracked_paths(root):
        path_label = relative.as_posix()
        if path_label == output_relative:
            continue
        path = root / relative
        try:
            if path.is_symlink():
                data = os.readlink(path).encode("utf-8", errors="surrogateescape")
            else:
                data = path.read_bytes()
        except OSError as error:
            raise ValueError(f"cannot read tracked file {path_label}: {error}") from error
        text = _decode_text(data)
        if text is None:
            skipped_binary.append(path_label)
            continue
        canonical = text.encode("utf-8")
        files.append(
            {
                "path": path_label,
                "bytes": len(canonical),
                "sha256": hashlib.sha256(canonical).hexdigest(),
                "text": text,
                "language": _language(relative),
            }
        )
    return files, skipped_binary


def _render(
    root: Path,
    output_relative: str,
    *,
    preserved_provenance: dict[str, Any] | None = None,
) -> str:
    files, skipped_binary = _collect_snapshot(root, output_relative)
    manifest_files = [
        {"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]}
        for item in files
    ]
    metadata = {
        "format_version": FORMAT_VERSION,
        "generator": "python scripts/build_code_summary.py",
        "provenance": preserved_provenance or _provenance(root, output_relative),
        "snapshot": {
            "snapshot_sha256": _snapshot_digest(manifest_files),
            "included_file_count": len(files),
            "canonicalization": "UTF-8 text with LF newlines",
            "excluded_output": output_relative,
            "skipped_binary_paths": skipped_binary,
            "files": manifest_files,
        },
    }
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)

    max_tilde_run = 0
    for item in files:
        for match in re.finditer(r"~+", item["text"]):
            max_tilde_run = max(max_tilde_run, len(match.group(0)))
    fence = "~" * max(8, max_tilde_run + 1)

    sections = [METADATA_START + metadata_json + METADATA_END, "# Repository code snapshot\n"]
    for item in files:
        body = item["text"]
        if body and not body.endswith("\n"):
            body += "\n"
        sections.append(f"# {item['path']}\n\n{fence}{item['language']}\n{body}{fence}\n")
    return "\n".join(sections)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail unless the snapshot matches tracked text")
    parser.add_argument(
        "--require-clean-provenance",
        action="store_true",
        help=(
            "require committed source provenance; after generation, only the summary file "
            "may differ from its source commit"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("code_summary.md"))
    parser.add_argument("--root", type=Path, help="repository root; defaults to the current work tree")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = args.root.resolve() if args.root else _git_root(Path.cwd())
        output = args.output if args.output.is_absolute() else root / args.output
        output = output.resolve(strict=False)
        try:
            output_relative = output.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("summary output must be inside the repository") from error

        if args.check:
            metadata = _read_existing_metadata(output)
            provenance = _validate_preserved_provenance(metadata["provenance"])
            if args.require_clean_provenance:
                _require_clean_provenance(root, output_relative, provenance)
            expected = _render(root, output_relative, preserved_provenance=provenance)
            actual = output.read_text(encoding="utf-8")
            if actual != expected:
                print(
                    f"{output_relative} is stale; run python scripts/build_code_summary.py",
                    file=sys.stderr,
                )
                return 1
            print(f"{output_relative} matches the tracked text snapshot")
            return 0

        provenance = _provenance(root, output_relative)
        if args.require_clean_provenance:
            _require_clean_provenance(root, output_relative, provenance)
        content = _render(root, output_relative, preserved_provenance=provenance)
        _atomic_write(output, content)
        print(f"wrote {output_relative}")
        return 0
    except (TypeError, ValueError) as error:
        print(f"code summary error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
