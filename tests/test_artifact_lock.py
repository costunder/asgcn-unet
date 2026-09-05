"""Artifact-lock unit tests use only isolated temporary output directories."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from asgcn_unet.artifact_lock import (
    ArtifactWriterBusyError,
    ArtifactWriterOwnershipError,
    artifact_writer_lock_path,
    exclusive_artifact_writer,
)


def test_writer_creates_only_its_sibling_lock_and_preserves_existing_artifacts(tmp_path: Path) -> None:
    mode = tmp_path / "aid" / "ann"
    mode.mkdir(parents=True)
    metrics = mode / "metrics.json"
    metrics.write_text("existing result\n", encoding="utf-8")
    with exclusive_artifact_writer(mode) as lock:
        assert lock == mode.parent / ".ann.writer.lock"
        owner = json.loads(lock.read_text(encoding="utf-8"))
        assert owner["pid"] == os.getpid()
        assert len(owner["token"]) == 32
        assert metrics.read_text(encoding="utf-8") == "existing result\n"
    assert not lock.exists()
    assert metrics.read_text(encoding="utf-8") == "existing result\n"


def test_lock_does_not_create_the_output_mode_itself(tmp_path: Path) -> None:
    mode = tmp_path / "new-output" / "ann"
    with exclusive_artifact_writer(mode):
        assert mode.parent.is_dir()
        assert not mode.exists()
    assert list(mode.parent.iterdir()) == []


def test_existing_lock_is_never_assumed_stale_or_deleted(tmp_path: Path) -> None:
    mode = tmp_path / "ann"
    lock = artifact_writer_lock_path(mode)
    content = json.dumps({"pid": -999, "token": "previous-owner"})
    lock.write_text(content, encoding="utf-8")
    with (
        pytest.raises(ArtifactWriterBusyError, match="already exists"),
        exclusive_artifact_writer(mode),
    ):
        pytest.fail("A second writer acquired an existing lock")
    assert lock.read_text(encoding="utf-8") == content


def test_exception_releases_owned_lock_and_preserves_original_error(tmp_path: Path) -> None:
    mode = tmp_path / "ann"
    original = ValueError("smoke-test failure")
    with pytest.raises(ValueError) as caught, exclusive_artifact_writer(mode):
        raise original
    assert caught.value is original
    assert not artifact_writer_lock_path(mode).exists()


@pytest.mark.parametrize("replace_file", [False, True])
def test_changed_lock_is_preserved_and_ownership_violation_reported(
    tmp_path: Path, replace_file: bool,
) -> None:
    mode = tmp_path / "ann"
    with (
        pytest.raises(ArtifactWriterOwnershipError, match="changed"),
        exclusive_artifact_writer(mode) as lock,
    ):
        if replace_file:
            lock.unlink()
        lock.write_text('{"pid": 123, "token": "foreign"}', encoding="utf-8")
    assert json.loads(lock.read_text(encoding="utf-8"))["token"] == "foreign"


def test_cleanup_violation_does_not_hide_the_original_writer_failure(tmp_path: Path) -> None:
    original = ValueError("original failure")
    with pytest.raises(ValueError) as caught, exclusive_artifact_writer(tmp_path / "ann") as lock:
        lock.write_text('{"token": "foreign"}', encoding="utf-8")
        raise original
    assert caught.value is original
    if sys.version_info >= (3, 11):
        assert any("token changed" in note for note in caught.value.__notes__)
    else:
        assert isinstance(caught.value.__cause__, ArtifactWriterOwnershipError)
        assert "token changed" in str(caught.value.__cause__)
    assert lock.exists()


def test_mode_rename_cannot_detach_the_active_writer_lock(tmp_path: Path) -> None:
    mode = tmp_path / "ann"
    mode.mkdir()
    with exclusive_artifact_writer(mode) as lock:
        mode.rename(tmp_path / "ann.incomplete-smoke-test")
        assert lock.exists()
        with pytest.raises(ArtifactWriterBusyError), exclusive_artifact_writer(mode):
            pytest.fail("A moved mode detached its writer lock")
    assert not lock.exists()


def test_independent_modes_can_hold_independent_locks(tmp_path: Path) -> None:
    with (
        exclusive_artifact_writer(tmp_path / "ann") as ann_lock,
        exclusive_artifact_writer(tmp_path / "snn_literal_eq15_T4") as snn_lock,
    ):
        assert ann_lock != snn_lock
        assert ann_lock.exists() and snn_lock.exists()


def test_separate_process_is_excluded_before_any_output_write(tmp_path: Path) -> None:
    mode = tmp_path / "ann"
    code = (
        "import sys\n"
        "from asgcn_unet.artifact_lock import exclusive_artifact_writer\n"
        "with exclusive_artifact_writer(sys.argv[1]):\n"
        "    raise AssertionError('second writer acquired the lock')\n"
    )
    with exclusive_artifact_writer(mode) as lock:
        before = lock.read_bytes()
        result = subprocess.run(
            [sys.executable, "-c", code, str(mode)],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode != 0
        assert "ArtifactWriterBusyError" in result.stderr
        assert lock.read_bytes() == before
    assert not mode.exists()


@pytest.mark.parametrize("output", ["", ".", "..", Path(Path.cwd().anchor)])
def test_unnamed_or_root_output_is_rejected_before_creating_a_lock(output) -> None:
    with pytest.raises(ValueError, match="named output"):
        artifact_writer_lock_path(output)


def test_existing_symlink_lock_is_preserved_without_following_its_target(tmp_path: Path) -> None:
    mode = tmp_path / "ann"
    target = tmp_path / "untouched.txt"
    target.write_text("untouched", encoding="utf-8")
    lock = artifact_writer_lock_path(mode)
    try:
        lock.symlink_to(target)
    except OSError:
        pytest.skip("This test requires local symlink creation")
    with pytest.raises(ArtifactWriterBusyError), exclusive_artifact_writer(mode):
        pytest.fail("A symlink lock was followed")
    assert lock.is_symlink()
    assert target.read_text(encoding="utf-8") == "untouched"


def test_cleanup_error_preserves_original_failure_without_python311_notes(tmp_path: Path) -> None:
    class LegacyFailure(RuntimeError):
        add_note = None

    run_dir = tmp_path / "legacy-note-smoke-test" / "ann"
    with (
        pytest.raises(LegacyFailure, match="original failure") as caught,
        exclusive_artifact_writer(run_dir) as lock_path,
    ):
        lock_path.write_text('{"token": "foreign"}', encoding="utf-8")
        raise LegacyFailure("original failure")
    assert isinstance(caught.value.__cause__, ArtifactWriterOwnershipError)
    assert lock_path.read_text(encoding="utf-8") == '{"token": "foreign"}'
