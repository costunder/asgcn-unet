"""Smoke-test owned-child cleanup with in-memory process doubles; send no signals."""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

import pytest

from scripts import scan_private_text as scanner


class ReaderProcess:
    def __init__(self, timeouts: int = 0) -> None:
        self.pid = os.getpid() + os.getppid() + 1000
        self.args = ["git", "cat-file", "--batch"]
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(b"invalid-header\n")
        self.timeouts = timeouts
        self.returncode = None
        self.events: list[str] = []

    def wait(self, timeout: float | None = None) -> int:
        assert timeout == 5
        assert self.stdin.closed and self.stdout.closed
        self.events.append("wait")
        if self.timeouts:
            self.timeouts -= 1
            raise subprocess.TimeoutExpired(self.args, timeout)
        self.returncode = 0
        return 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.events.append("terminate")

    def kill(self) -> None:
        self.events.append("kill")


@pytest.mark.parametrize(
    ("timeouts", "events"),
    [(0, ["wait"]), (1, ["wait", "terminate", "wait"]),
     (2, ["wait", "terminate", "wait", "kill", "wait"])],
)
def test_cleanup_prefers_eof_then_targets_only_verified_child(
    timeouts: int, events: list[str], capsys: pytest.CaptureFixture[str],
) -> None:
    process = ReaderProcess(timeouts)
    scanner._finish_owned_history_reader(process, process.pid)
    assert process.events == events
    diagnostics = capsys.readouterr().err
    assert f"PID={process.pid}" in diagnostics
    assert "command=git cat-file --batch" in diagnostics
    assert "closing only this child's pipes" in diagnostics
    if timeouts:
        assert "terminating only this owned child" in diagnostics
    if timeouts == 2:
        assert "forcing only this owned child" in diagnostics


@pytest.mark.parametrize("invalid", ["command", "pid", "caller", "parent"])
def test_cleanup_refuses_unverified_or_calling_process(invalid: str) -> None:
    process = ReaderProcess()
    owned_pid = process.pid
    if invalid == "command":
        process.args = ["unexpected-process"]
    elif invalid == "pid":
        owned_pid += 1
    elif invalid == "caller":
        owned_pid = process.pid = os.getpid()
    else:
        owned_pid = process.pid = os.getppid()
    with pytest.raises(RuntimeError, match="unverified"):
        scanner._finish_owned_history_reader(process, owned_pid)
    assert process.events == []
    assert not process.stdin.closed and not process.stdout.closed


def test_each_signal_is_reported_before_it_is_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    process = ReaderProcess(2)
    reports: list[str] = []
    monkeypatch.setattr(scanner, "print", lambda text, **kwargs: reports.append(text), raising=False)

    def terminate() -> None:
        assert "terminating only this owned child" in reports[-1]

    def kill() -> None:
        assert "forcing only this owned child" in reports[-1]

    monkeypatch.setattr(process, "terminate", terminate)
    monkeypatch.setattr(process, "kill", kill)
    scanner._finish_owned_history_reader(process, process.pid)


def test_scan_keeps_original_error_if_cleanup_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = ReaderProcess(3)
    object_id = "a" * 40
    monkeypatch.setattr(scanner, "_require_complete_history", lambda root: None)
    monkeypatch.setattr(scanner, "_reachable_objects", lambda root: {object_id: "sample.txt"})
    monkeypatch.setattr(scanner, "_reachable_blob_ids", lambda root, objects: [object_id])
    monkeypatch.setattr(scanner.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(ValueError, match="invalid header") as error:
        scanner.scan_history(tmp_path, [])
    assert isinstance(error.value.__cause__, subprocess.TimeoutExpired)
    assert process.events == ["wait", "terminate", "wait", "kill", "wait"]
