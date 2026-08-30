from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import scan_private_text

ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "scripts" / "scan_private_text.py"
SUMMARY_BUILDER = ROOT / "scripts" / "build_code_summary.py"


def _run(command: list[str], cwd: Path, *, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )


def _private_unix_path() -> str:
    separator = chr(47)
    return separator + "home" + separator + "research-node-user" + separator + "project"


def _private_windows_path() -> str:
    separator = chr(92)
    return "D:" + separator + "Users" + separator + "research-node-user" + separator + "project"


def _init_repository(path: Path) -> None:
    assert _run(["git", "init", "-q"], path).returncode == 0
    assert _run(["git", "config", "user.name", "Snapshot Test"], path).returncode == 0
    assert _run(["git", "config", "user.email", "snapshot@example.invalid"], path).returncode == 0


def _environment_without_private_markers() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PRIVATE_MARKERS_B64", None)
    return environment


def test_privacy_scanner_detects_generic_paths_without_echoing_values(tmp_path: Path) -> None:
    secret_values = (_private_unix_path(), _private_windows_path())
    candidate = tmp_path / "candidate.txt"
    candidate.write_text("\n".join(secret_values), encoding="utf-8")
    binary = tmp_path / "payload.bin"
    binary.write_bytes(b"\x00" + secret_values[0].encode("utf-8"))

    result = _run([sys.executable, str(SCANNER), str(candidate), str(binary), "--root", str(tmp_path)], tmp_path)

    assert result.returncode == 1
    assert "generic-unix-home" in result.stderr
    assert "generic-windows-home" in result.stderr
    assert all(value not in result.stderr for value in secret_values)


def test_privacy_scanner_uses_external_and_encoded_markers(tmp_path: Path) -> None:
    marker = "private" + chr(45) + "identity" + chr(45) + "marker"
    left, right = marker[:9], marker[9:]
    encoded_marker = base64.b64encode(marker.encode("utf-8")).decode("ascii")
    encoded_left, encoded_right = encoded_marker[:12], encoded_marker[12:]
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        (
            f'value = f"{{\'{left}\'}}{{\'{right}\'}}"\n'
            f'encoded = "{encoded_left}" + "{encoded_right}"\n'
        ),
        encoding="utf-8",
    )
    denylist = tmp_path / "denylist.txt"
    denylist.write_text(marker + "\n", encoding="utf-8")
    environment = dict(os.environ)
    environment["PRIVATE_MARKERS_B64"] = base64.b64encode((marker + "\n").encode("utf-8")).decode("ascii")

    result = _run(
        [
            sys.executable,
            str(SCANNER),
            str(candidate),
            "--root",
            str(tmp_path),
            "--extra-patterns",
            str(denylist),
        ],
        tmp_path,
        environment=environment,
    )

    assert result.returncode == 1
    assert "python-constant" in result.stderr
    assert "python-constant-base64-decoded" in result.stderr
    assert marker not in result.stderr


def test_privacy_scanner_rejects_invalid_encoded_environment(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.txt"
    candidate.write_text("safe", encoding="utf-8")
    environment = dict(os.environ)
    environment["PRIVATE_MARKERS_B64"] = "not valid base64"

    result = _run(
        [sys.executable, str(SCANNER), str(candidate), "--root", str(tmp_path)],
        tmp_path,
        environment=environment,
    )

    assert result.returncode == 2
    assert "base64-encoded UTF-8" in result.stderr
    assert environment["PRIVATE_MARKERS_B64"] not in result.stderr


def test_privacy_scanner_can_require_a_nonempty_external_denylist(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.txt"
    candidate.write_text("safe\n", encoding="utf-8")
    empty_denylist = tmp_path / "denylist.txt"
    empty_denylist.write_text("# comments do not count as patterns\n", encoding="utf-8")
    environment = _environment_without_private_markers()

    missing = _run(
        [
            sys.executable,
            str(SCANNER),
            str(candidate),
            "--root",
            str(tmp_path),
            "--require-external-patterns",
        ],
        tmp_path,
        environment=environment,
    )
    comments_only = _run(
        [
            sys.executable,
            str(SCANNER),
            str(candidate),
            "--root",
            str(tmp_path),
            "--extra-patterns",
            str(empty_denylist),
            "--require-external-patterns",
        ],
        tmp_path,
        environment=environment,
    )

    assert missing.returncode == 2
    assert comments_only.returncode == 2
    assert "external private-marker denylist is required" in missing.stderr
    assert "external private-marker denylist is required" in comments_only.stderr


@pytest.mark.parametrize("hash_length", [40, 64])
def test_history_enumeration_uses_portable_lf_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hash_length: int,
) -> None:
    commit_id = "a" * hash_length
    tree_id = "b" * hash_length
    blob_id = "c" * hash_length
    other_blob_id = "d" * hash_length
    path_hint = "nested/자료 with spaces\tand\rcarriage.py"
    payload = (
        f"{commit_id}\n{tree_id} \n{blob_id} {path_hint}\n"
        f"{other_blob_id} path=literal-filename.txt\n"
    ).encode()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == tmp_path
        assert kwargs["env"]["GIT_NO_REPLACE_OBJECTS"] == "1"
        assert "PRIVATE_MARKERS_B64" not in kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout=payload, stderr=b"")

    monkeypatch.setattr(scan_private_text.subprocess, "run", run)

    assert scan_private_text._reachable_objects(tmp_path) == {
        commit_id: "",
        tree_id: "",
        blob_id: path_hint,
        other_blob_id: "path=literal-filename.txt",
    }
    assert calls == [["git", "rev-list", "--objects", "--all"]]


@pytest.mark.parametrize("object_bytes", [b"a" * 41, b"g" * 40, b"\xff" * 40])
def test_history_enumeration_rejects_malformed_object_ids_without_echoing_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    object_bytes: bytes,
) -> None:
    monkeypatch.setattr(
        scan_private_text.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=object_bytes + b" file.txt\n", stderr=b""
        ),
    )

    with pytest.raises(ValueError) as error:
        scan_private_text._reachable_objects(tmp_path)

    assert str(error.value) == "git history enumeration returned an invalid object ID"


def test_privacy_scanner_checks_unique_blobs_reachable_from_every_ref(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    safe = tmp_path / "safe.txt"
    safe.write_text("public fixture\n", encoding="utf-8")
    assert _run(["git", "add", "safe.txt"], tmp_path).returncode == 0
    assert _run(["git", "commit", "-qm", "safe base"], tmp_path).returncode == 0
    main_branch = _run(["git", "branch", "--show-current"], tmp_path).stdout.strip()
    safe_commit = _run(["git", "rev-parse", "HEAD"], tmp_path).stdout.strip()

    assert _run(["git", "switch", "-q", "-c", "archived-history"], tmp_path).returncode == 0
    marker = "historical" + chr(45) + "identity" + chr(45) + "marker"
    left, right = marker[:11], marker[11:]
    historical_name = marker + ".py"
    historical = tmp_path / historical_name
    historical.write_text(f'value = "{left}" + "{right}"\n', encoding="utf-8")
    binary = tmp_path / "historical.bin"
    binary.write_bytes(b"\x00" + marker.encode("utf-8"))
    assert _run(["git", "add", historical_name, binary.name], tmp_path).returncode == 0
    assert _run(["git", "commit", "-qm", "historical fixture"], tmp_path).returncode == 0
    assert _run(["git", "tag", "archived-copy"], tmp_path).returncode == 0
    historical_commit = _run(["git", "rev-parse", "HEAD"], tmp_path).stdout.strip()
    object_id = _run(["git", "rev-parse", f"HEAD:{historical_name}"], tmp_path).stdout.strip()
    assert _run(["git", "switch", "-q", main_branch], tmp_path).returncode == 0
    assert _run(["git", "replace", historical_commit, safe_commit], tmp_path).returncode == 0

    environment = _environment_without_private_markers()
    environment["PRIVATE_MARKERS_B64"] = base64.b64encode((marker + "\n").encode()).decode()
    current = _run(
        [
            sys.executable,
            str(SCANNER),
            "--all-tracked",
            "--require-external-patterns",
        ],
        tmp_path,
        environment=environment,
    )
    history = _run(
        [
            sys.executable,
            str(SCANNER),
            "--all-history",
            "--require-external-patterns",
        ],
        tmp_path,
        environment=environment,
    )

    assert current.returncode == 0, current.stderr
    assert history.returncode == 1
    assert f"object {object_id}" in history.stderr
    assert "<redacted>.py:1" in history.stderr
    assert "external-environment-1" in history.stderr
    assert "python-constant" in history.stderr
    assert history.stderr.count("python-constant") == 1
    assert marker not in history.stdout
    assert marker not in history.stderr


def test_ci_history_scan_detects_removed_paths_without_private_secrets(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    source = tmp_path / "example.txt"
    sample_home = chr(47).join(("", "home", "fixture", "data"))
    source.write_text(sample_home + "\n", encoding="utf-8")
    assert _run(["git", "add", source.name], tmp_path).returncode == 0
    assert _run(["git", "commit", "-qm", "historical path fixture"], tmp_path).returncode == 0
    source.write_text("portable data path\n", encoding="utf-8")
    assert _run(["git", "commit", "-qam", "remove path fixture"], tmp_path).returncode == 0
    environment = _environment_without_private_markers()

    current = _run(
        [sys.executable, str(SCANNER), "--all-tracked"],
        tmp_path,
        environment=environment,
    )
    history = _run(
        [sys.executable, str(SCANNER), "--all-tracked", "--all-history"],
        tmp_path,
        environment=environment,
    )

    assert current.returncode == 0, current.stderr
    assert history.returncode == 1
    assert "generic-unix-home" in history.stderr
    assert "external private-marker denylist is required" not in history.stderr
    assert sample_home not in history.stdout + history.stderr


def test_privacy_scanner_rejects_incomplete_shallow_history(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_repository(origin)
    source = origin / "source.txt"
    source.write_text("first\n", encoding="utf-8")
    assert _run(["git", "add", source.name], origin).returncode == 0
    assert _run(["git", "commit", "-qm", "first"], origin).returncode == 0
    source.write_text("second\n", encoding="utf-8")
    assert _run(["git", "commit", "-qam", "second"], origin).returncode == 0

    shallow = tmp_path / "shallow"
    clone = _run(
        ["git", "clone", "-q", "--depth", "1", origin.resolve().as_uri(), str(shallow)],
        tmp_path,
    )
    assert clone.returncode == 0, clone.stderr
    marker = "synthetic" + chr(45) + "history" + chr(45) + "marker"
    environment = _environment_without_private_markers()
    environment["PRIVATE_MARKERS_B64"] = base64.b64encode(marker.encode()).decode()

    result = _run(
        [
            sys.executable,
            str(SCANNER),
            "--all-history",
            "--require-external-patterns",
        ],
        shallow,
        environment=environment,
    )

    assert result.returncode == 2
    assert "complete, non-shallow repository" in result.stderr
    assert marker not in result.stdout
    assert marker not in result.stderr


def test_code_summary_is_deterministic_and_check_detects_changes(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    source = tmp_path / "example.py"
    source.write_text("value = 1\n", encoding="utf-8")
    binary = tmp_path / "weights.bin"
    binary.write_bytes(b"\x00\xff\x10")
    assert _run(["git", "add", "example.py", "weights.bin"], tmp_path).returncode == 0
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_AUTHOR_DATE": "2001-02-03T04:05:06Z",
            "GIT_COMMITTER_DATE": "2001-02-03T04:05:06Z",
        }
    )
    assert _run(["git", "commit", "-qm", "fixture"], tmp_path, environment=environment).returncode == 0

    build = _run([sys.executable, str(SUMMARY_BUILDER)], tmp_path)
    assert build.returncode == 0, build.stderr
    first = (tmp_path / "code_summary.md").read_bytes()
    assert _run([sys.executable, str(SUMMARY_BUILDER)], tmp_path).returncode == 0
    assert (tmp_path / "code_summary.md").read_bytes() == first
    assert _run(["git", "add", "code_summary.md"], tmp_path).returncode == 0

    check = _run([sys.executable, str(SUMMARY_BUILDER), "--check"], tmp_path)
    assert check.returncode == 0, check.stderr
    summary = first.decode("utf-8")
    metadata_text = summary.split("<!-- code-summary-metadata\n", 1)[1].split("\n-->\n", 1)[0]
    metadata = json.loads(metadata_text)
    manifest = metadata["snapshot"]
    assert manifest["included_file_count"] == 1
    assert manifest["skipped_binary_paths"] == ["weights.bin"]
    assert manifest["files"][0]["sha256"] == hashlib.sha256(b"value = 1\n").hexdigest()
    assert metadata["provenance"]["generated_utc"] == "2001-02-03T04:05:06Z"

    source.write_text("value = 2\n", encoding="utf-8")
    stale = _run([sys.executable, str(SUMMARY_BUILDER), "--check"], tmp_path)
    assert stale.returncode == 1
    assert "is stale" in stale.stderr

    assert _run([sys.executable, str(SUMMARY_BUILDER)], tmp_path).returncode == 0
    dirty_summary = (tmp_path / "code_summary.md").read_text(encoding="utf-8")
    dirty_metadata_text = dirty_summary.split("<!-- code-summary-metadata\n", 1)[1].split(
        "\n-->\n", 1
    )[0]
    dirty_provenance = json.loads(dirty_metadata_text)["provenance"]
    assert dirty_provenance["tracked_tree_dirty_at_generation"] is True
    assert dirty_provenance["source_commit_at_generation"] is None
    assert dirty_provenance["source_tree_at_generation"] is None


def test_code_summary_clean_provenance_allows_only_a_followup_summary_commit(
    tmp_path: Path,
) -> None:
    _init_repository(tmp_path)
    source = tmp_path / "example.py"
    source.write_text("value = 1\n", encoding="utf-8")
    assert _run(["git", "add", "example.py"], tmp_path).returncode == 0
    assert _run(["git", "commit", "-qm", "source"], tmp_path).returncode == 0

    generated = _run(
        [sys.executable, str(SUMMARY_BUILDER), "--require-clean-provenance"],
        tmp_path,
    )
    assert generated.returncode == 0, generated.stderr
    assert _run(["git", "add", "code_summary.md"], tmp_path).returncode == 0
    assert _run(["git", "commit", "-qm", "summary"], tmp_path).returncode == 0

    clean = _run(
        [
            sys.executable,
            str(SUMMARY_BUILDER),
            "--check",
            "--require-clean-provenance",
        ],
        tmp_path,
    )
    assert clean.returncode == 0, clean.stderr

    source.write_text("value = 2\n", encoding="utf-8")
    assert _run(["git", "add", "example.py"], tmp_path).returncode == 0
    assert _run(["git", "commit", "-qm", "changed source"], tmp_path).returncode == 0
    stale_provenance = _run(
        [
            sys.executable,
            str(SUMMARY_BUILDER),
            "--check",
            "--require-clean-provenance",
        ],
        tmp_path,
    )
    assert stale_provenance.returncode != 0
    assert "source changed after" in stale_provenance.stderr


def test_ci_pins_actions_and_runs_repository_hygiene_gates() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    uses_lines = [line.strip() for line in workflow.splitlines() if line.strip().startswith("uses:")]

    assert uses_lines
    for line in uses_lines:
        reference = line.split("@", maxsplit=1)[1].split(maxsplit=1)[0]
        assert len(reference) == 40
        assert all(character in "0123456789abcdef" for character in reference)
    assert "python scripts/build_code_summary.py --check" in workflow
    assert "python scripts/build_code_summary.py --check --require-clean-provenance" in workflow
    assert "python scripts/scan_private_text.py --all-tracked" in workflow
    assert "Exercise the Linux server setup entrypoint" in workflow
    assert "run: bash scripts/setup.sh" in workflow
    assert "Run tests with the locked Python 3.12 profile" in workflow
    assert workflow.count("run: python -m pytest -q") >= 2
    assert "fetch-depth: 0" in workflow
    assert "if: github.event_name == 'push' && github.ref == 'refs/heads/main'" in workflow
    assert "Scan trusted main history with generic privacy rules" in workflow
    # Exact identity markers stay in the local release gate, never in CI secrets.
    assert "PRIVATE_MARKERS_B64" not in workflow
    assert "secrets." not in workflow
    assert "--all-history" in workflow
    assert "--require-external-patterns" not in workflow
    assert "for script in scripts/*.sh server/*.sbatch server/*.pbs; do" in workflow
    assert 'bash -n "$script" || exit 1' in workflow
    assert "run: bash -n scripts/*.sh" not in workflow
