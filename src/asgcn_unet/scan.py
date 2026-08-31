from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .utils import save_json


class ScanInUseError(RuntimeError):
    """A second writer must not modify either the journal or its final report."""


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ScanJournal:
    """Atomically commit bounded blocks without rewriting the entire topology scan.

    A block is durable before its reference is published in ``index.json``. An
    interrupted write can therefore leave an unreferenced block, but never a
    partially committed sample. Explicit resume ignores and may replace such an
    orphan at the next block boundary. Previously committed blocks are immutable.
    """

    def __init__(
        self,
        directory: Path,
        contract: dict[str, Any],
        *,
        resume: bool,
        origin: dict[str, Any] | None = None,
        block_size: int = 128,
        interval_seconds: float = 30.0,
    ) -> None:
        if block_size < 1 or interval_seconds <= 0:
            raise ValueError("Scan checkpoint cadence must be positive")
        self.directory = directory
        self.index_path = directory / "index.json"
        self.contract = contract
        self.origin = {} if origin is None else origin
        self.block_size = block_size
        self.interval_seconds = interval_seconds
        self.records: list[dict[str, Any]] = []
        self.blocks: list[dict[str, Any]] = []
        self.committed = 0
        self.last_commit = time.monotonic()
        self._lock_handle = None
        existed = directory.exists()
        if directory.is_symlink():
            raise ValueError("Topology scan journal directory must not be a symlink")
        if existed and not resume:
            raise FileExistsError(
                "Topology scan journal already exists; use --resume-scan explicitly"
            )
        if not existed and resume:
            raise FileNotFoundError("No topology scan journal exists for --resume-scan")
        if not existed:
            directory.mkdir(parents=True)
        try:
            self._lock()
            if existed:
                self._read()
            else:
                self._write_index()
        except BaseException:
            self.close()
            raise

    def _lock(self) -> None:
        lock_path = self.directory / "writer.lock"
        if lock_path.is_symlink():
            raise ValueError("Topology scan journal lock must not be a symlink")
        handle = lock_path.open("a+b")
        try:
            # Windows byte-range locks also reject reading the locked byte. Use
            # descriptor metadata rather than reading before trying the lock.
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_handle = handle
        except OSError as error:
            handle.close()
            self._lock_handle = None
            raise ScanInUseError("Another process is writing this topology scan journal") from error
        except BaseException:
            handle.close()
            self._lock_handle = None
            raise

    def close(self) -> None:
        handle = self._lock_handle
        if handle is not None:
            self._lock_handle = None
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def _write_index(self) -> None:
        save_json(
            self.index_path,
            {
                "schema": "asgcn_topology_journal_v1",
                "contract": self.contract,
                "contract_sha256": canonical_hash(self.contract),
                "origin": self.origin,
                "origin_sha256": canonical_hash(self.origin),
                "samples_committed": self.committed,
                "blocks": self.blocks,
            },
        )

    def _read(self) -> None:
        if self.index_path.is_symlink() or not self.index_path.is_file():
            raise ValueError("Topology scan journal index must be a regular non-symlink file")
        with self.index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        if not isinstance(index, dict) or index.get("schema") != "asgcn_topology_journal_v1":
            raise ValueError("Unsupported topology scan journal")
        if index.get("contract") != self.contract or index.get("contract_sha256") != canonical_hash(
            self.contract
        ):
            raise ValueError("Topology scan journal data/config/implementation differs")
        blocks = index.get("blocks")
        origin = index.get("origin")
        if not isinstance(origin, dict) or index.get("origin_sha256") != canonical_hash(origin):
            raise ValueError("Topology scan journal origin is invalid")
        if not isinstance(blocks, list):
            raise TypeError("Topology scan journal blocks must be a list")
        for number, entry in enumerate(blocks):
            expected_name = f"{number:06d}.json"
            if (
                not isinstance(entry, dict)
                or entry.get("file") != expected_name
                or isinstance(entry.get("start"), bool)
                or not isinstance(entry.get("start"), int)
                or entry.get("start") != len(self.records)
                or isinstance(entry.get("count"), bool)
                or not isinstance(entry.get("count"), int)
                or entry["count"] < 1
            ):
                raise ValueError("Topology scan journal block ordering is invalid")
            block_path = self.directory / expected_name
            if block_path.is_symlink() or not block_path.is_file():
                raise ValueError("Topology scan journal block must be a regular non-symlink file")
            with block_path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
            if (
                not isinstance(records, list)
                or len(records) != entry["count"]
                or canonical_hash(records) != entry.get("sha256")
            ):
                raise ValueError("Topology scan journal block integrity check failed")
            self.records.extend(records)
        committed = index.get("samples_committed")
        if (
            isinstance(committed, bool)
            or not isinstance(committed, int)
            or committed != len(self.records)
        ):
            raise ValueError("Topology scan journal committed sample count is invalid")
        self.blocks = blocks
        self.origin = origin
        self.committed = len(self.records)

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if (
            len(self.records) - self.committed >= self.block_size
            or time.monotonic() - self.last_commit >= self.interval_seconds
        ):
            self.flush()

    def flush(self) -> None:
        while self.committed < len(self.records):
            stop = min(self.committed + self.block_size, len(self.records))
            records = self.records[self.committed : stop]
            name = f"{len(self.blocks):06d}.json"
            save_json(self.directory / name, records)
            self.blocks.append(
                {
                    "file": name,
                    "start": self.committed,
                    "count": len(records),
                    "sha256": canonical_hash(records),
                }
            )
            previous = self.committed
            self.committed = stop
            try:
                self._write_index()
            except BaseException:
                self.committed = previous
                self.blocks.pop()
                raise
        self.last_commit = time.monotonic()
