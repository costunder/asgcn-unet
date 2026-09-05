"""Exclusive, ownership-checked locks for evaluation artifact writers."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class ArtifactWriterBusyError(RuntimeError):
    """An active writer or an unresolved previous lock owns the output mode."""


class ArtifactWriterOwnershipError(RuntimeError):
    """The lock changed while owned; no foreign lock may be removed."""


def artifact_writer_lock_path(output_dir: str | Path) -> Path:
    """Use a stable sibling so preserving a mode directory cannot move its lock."""
    output = Path(output_dir).expanduser()
    if output.name in ("", ".", ".."):
        raise ValueError("An artifact writer requires a named output mode directory")
    # Resolve the parent only: a final-component symlink is never followed here.
    parent = output.parent.resolve()
    return parent / f".{output.name}.writer.lock"


def _release_owned_lock(path: Path, identity: tuple[int, int], token: str) -> None:
    try:
        current = path.lstat()
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != identity:
            raise ArtifactWriterOwnershipError("Artifact writer lock identity changed; it was preserved")
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("token") != token:
            raise ArtifactWriterOwnershipError("Artifact writer lock token changed; it was preserved")
        current = path.lstat()
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != identity:
            raise ArtifactWriterOwnershipError("Artifact writer lock identity changed; it was preserved")
        path.unlink()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactWriterOwnershipError(
            "Artifact writer lock could not be verified or released; inspect the lock before recovery"
        ) from error


@contextmanager
def exclusive_artifact_writer(output_dir: str | Path) -> Iterator[Path]:
    """Acquire one mode exclusively; never infer that an existing lock is stale.

    Only the configured output base is created. The lock sits beside the mode
    directory and contains its owner PID and an unpredictable ownership token.
    Renaming partial output directories therefore cannot detach a live lock.
    """
    lock_path = artifact_writer_lock_path(output_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError:
        raise ArtifactWriterBusyError(
            f"Artifact writer lock already exists: {lock_path.name}. "
            "Another writer or an unresolved previous lock owns this mode; "
            "no output was overwritten and no lock was automatically removed."
        ) from None
    details = os.fstat(descriptor)
    identity = (details.st_dev, details.st_ino)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema": "asgcn_artifact_writer_lock_v1",
                    "pid": os.getpid(),
                    "token": token,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                },
                stream,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield lock_path
    except BaseException as error:
        try:
            _release_owned_lock(lock_path, identity, token)
        except ArtifactWriterOwnershipError as cleanup_error:
            add_note = getattr(error, "add_note", None)
            if add_note is None:  # Python 3.10 has no BaseException.add_note.
                raise error from cleanup_error
            add_note(str(cleanup_error))
        raise
    else:
        _release_owned_lock(lock_path, identity, token)
