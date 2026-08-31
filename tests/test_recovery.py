from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import asgcn_unet.cli as cli_module
from asgcn_unet.recovery import archive_uncheckpointed_run


def _metadata_directory(path: Path) -> dict[str, bytes]:
    path.mkdir(parents=True)
    payloads = {
        "config.json": json.dumps({"model": {}, "dataset": {}, "train": {}}).encode(),
        "preflight_gate.json": b'{"status":"prior-gate"}\r\n',
        ".data_hash_cache.json": b'{"cached":"bytes must remain identical"}\n',
    }
    for name, value in payloads.items():
        (path / name).write_bytes(value)
    return payloads


def _assert_payloads(path: Path, payloads: dict[str, bytes]) -> None:
    assert {entry.name for entry in path.iterdir()} == set(payloads)
    for name, value in payloads.items():
        assert (path / name).read_bytes() == value


def test_archive_preserves_every_metadata_byte_in_unique_sibling_container(tmp_path) -> None:
    project = tmp_path / "project"
    run_dir = project / "runs" / "train"
    payloads = _metadata_directory(run_dir)
    archived = archive_uncheckpointed_run(run_dir, project)
    assert archived is not None
    assert not run_dir.exists()
    assert archived.name == "train"
    assert archived.parent.parent == run_dir.parent
    assert archived.parent.name.startswith("train.failed-")
    _assert_payloads(archived, payloads)


def test_repeated_explicit_restarts_never_overwrite_an_older_archive(tmp_path) -> None:
    project = tmp_path / "project"
    run_dir = project / "runs" / "train"
    original = _metadata_directory(run_dir)
    first = archive_uncheckpointed_run(run_dir, project)
    replacement = _metadata_directory(run_dir)
    replacement["preflight_gate.json"] = b'{"status":"different-failed-run"}\n'
    (run_dir / "preflight_gate.json").write_bytes(replacement["preflight_gate.json"])
    second = archive_uncheckpointed_run(run_dir, project)
    assert first is not None and second is not None and first != second
    _assert_payloads(first, original)
    _assert_payloads(second, replacement)


@pytest.mark.parametrize("exists", [False, True])
def test_missing_or_empty_run_directory_is_an_unchanged_noop(tmp_path, exists: bool) -> None:
    project = tmp_path / "project"
    project.mkdir()
    run_dir = project / "train"
    if exists:
        run_dir.mkdir()
    before = set(project.iterdir())
    assert archive_uncheckpointed_run(run_dir, project) is None
    assert set(project.iterdir()) == before
    assert run_dir.exists() == exists


@pytest.mark.parametrize("name", [
    "best.pt", "last.pt", "checkpoint.pt", "history.json", "history.csv",
    "metrics.json", "notes.txt", "profile.json", ".unexpected",
])
def test_checkpoint_history_and_unknown_entries_are_never_moved(tmp_path, name: str) -> None:
    project = tmp_path / "project"
    run_dir = project / "runs" / "train"
    payloads = _metadata_directory(run_dir)
    payloads[name] = b"irreplaceable-existing-output"
    (run_dir / name).write_bytes(payloads[name])
    with pytest.raises(ValueError, match="checkpoints, history, or unknown"):
        archive_uncheckpointed_run(run_dir, project)
    _assert_payloads(run_dir, payloads)
    assert list(run_dir.parent.iterdir()) == [run_dir]


@pytest.mark.parametrize("name", ["checkpoints", "config.json"])
def test_nested_directory_is_not_treated_as_metadata(tmp_path, name: str) -> None:
    project = tmp_path / "project"
    run_dir = project / "runs" / "train"
    run_dir.mkdir(parents=True)
    child = run_dir / name
    child.mkdir()
    marker = child / "keep.bin"
    marker.write_bytes(b"retain nested data")
    with pytest.raises(ValueError, match="checkpoints, history, or unknown"):
        archive_uncheckpointed_run(run_dir, project)
    assert marker.read_bytes() == b"retain nested data"
    assert list(run_dir.parent.iterdir()) == [run_dir]


@pytest.mark.parametrize("bad_config", [
    b"not json", b"[]", b"null", b"{}", b'{"model":{},"train":{}}',
    b'{"model":null,"dataset":[],"train":1}',
])
def test_malformed_config_cannot_authorize_an_archive(tmp_path, bad_config: bytes) -> None:
    project = tmp_path / "project"
    run_dir = project / "runs" / "train"
    payloads = _metadata_directory(run_dir)
    payloads["config.json"] = bad_config
    (run_dir / "config.json").write_bytes(bad_config)
    with pytest.raises(ValueError):
        archive_uncheckpointed_run(run_dir, project)
    _assert_payloads(run_dir, payloads)
    assert list(run_dir.parent.iterdir()) == [run_dir]


@pytest.mark.parametrize("scope", ["project", "parent", "root", "home"])
def test_broad_directories_are_rejected_before_any_entry_inspection(
    tmp_path, monkeypatch, scope: str
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    fake_home = tmp_path / "user-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    targets = {
        "project": project, "parent": project.parent,
        "root": Path(project.anchor), "home": fake_home,
    }
    with pytest.raises(ValueError, match="broad directory"):
        archive_uncheckpointed_run(targets[scope], project)
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["project", "user-home"]


def test_regular_file_is_not_a_training_directory(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    run_dir = project / "train"
    run_dir.write_bytes(b"not a directory")
    with pytest.raises(ValueError, match="not a directory"):
        archive_uncheckpointed_run(run_dir, project)
    assert run_dir.read_bytes() == b"not a directory"


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Host cannot create test symlinks: {error}")


def test_linked_run_directory_is_rejected_without_moving_its_target(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    original = tmp_path / "original"
    payloads = _metadata_directory(original)
    link = project / "train"
    _symlink_or_skip(link, original, directory=True)
    with pytest.raises(ValueError, match="linked training directory"):
        archive_uncheckpointed_run(link, project)
    assert link.is_symlink()
    _assert_payloads(original, payloads)


def test_linked_metadata_file_is_rejected_without_moving_either_directory(tmp_path) -> None:
    project = tmp_path / "project"
    run_dir = project / "train"
    run_dir.mkdir(parents=True)
    original = tmp_path / "outside.json"
    original.write_text('{"model":{},"dataset":{},"train":{}}', encoding="utf-8")
    linked = run_dir / "config.json"
    _symlink_or_skip(linked, original)
    with pytest.raises(ValueError, match="checkpoints, history, or unknown"):
        archive_uncheckpointed_run(run_dir, project)
    assert linked.is_symlink()
    assert original.is_file()
    assert list(project.iterdir()) == [run_dir]


def test_linked_parent_alias_cannot_bypass_the_broad_project_guard(tmp_path) -> None:
    project = tmp_path / "projects" / "checkout"
    payloads = _metadata_directory(project)
    alias = tmp_path / "alias"
    _symlink_or_skip(alias, project.parent, directory=True)
    # The final component is not a symlink, but resolving its parent reaches the
    # protected project directory. Never trust textual path-prefix comparisons.
    with pytest.raises(ValueError, match="broad directory"):
        archive_uncheckpointed_run(alias / project.name, project)
    _assert_payloads(project, payloads)
    assert list(project.parent.iterdir()) == [project]


def test_failed_rename_preserves_original_bytes_and_existing_archives(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    run_dir = project / "runs" / "train"
    payloads = _metadata_directory(run_dir)
    older = run_dir.parent / "train.failed-existing"
    older.mkdir()
    (older / "keep.bin").write_bytes(b"older archive")
    destinations: list[Path] = []

    def fail_rename(source, destination):
        assert source == run_dir.resolve()
        destinations.append(Path(destination))
        raise PermissionError("simulated rename failure")

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(PermissionError, match="simulated rename failure"):
        archive_uncheckpointed_run(run_dir, project)
    _assert_payloads(run_dir, payloads)
    assert (older / "keep.bin").read_bytes() == b"older archive"
    assert len(destinations) == 1
    assert destinations[0].parent != older
    assert destinations[0].parent.is_dir()
    assert list(destinations[0].parent.iterdir()) == []


def _cli_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    project = tmp_path / "project"
    project.mkdir()
    config = {
        "model": {}, "dataset": {}, "train": {},
        "output": {"run_dir": str(project / "runs" / "train")},
    }
    monkeypatch.setattr(cli_module, "load_json", lambda path: copy.deepcopy(config))
    monkeypatch.setattr(cli_module, "resolve_experiment_paths", lambda value, path: value)
    monkeypatch.setattr(cli_module, "experiment_base_dir", lambda path: project)
    return project, config


def _args(project: Path, command: str, *extra: str):
    return cli_module.build_parser().parse_args([
        command, "--config", str(project / "config.json"), *extra,
    ])


def test_cli_archives_only_after_verified_profile_and_before_new_gate(
    tmp_path, monkeypatch
) -> None:
    project, config = _cli_fixture(tmp_path, monkeypatch)
    run_dir = Path(config["output"]["run_dir"])
    payloads = _metadata_directory(run_dir)
    calls: list[str] = []
    archives: list[Path] = []
    gate = {"status": "passed", "report_eligible": True}

    def verify(value, report):
        calls.append("verify")
        assert value["output"] == config["output"]
        assert report == project / "runs" / "passed.json"
        _assert_payloads(run_dir, payloads)
        return gate

    def archive(path, root):
        calls.append("archive")
        assert calls == ["verify", "archive"]
        result = archive_uncheckpointed_run(path, root)
        assert result is not None
        archives.append(result)
        return result

    def train(value, *, resume_from):
        calls.append("train")
        assert resume_from is None
        assert value["preflight_gate"] == gate
        # Publication belongs to engine.train after its own run/resume checks.
        assert not run_dir.exists()
        return run_dir / "best.pt"

    monkeypatch.setattr(cli_module, "verify_training_preflight", verify)
    monkeypatch.setattr(cli_module, "archive_uncheckpointed_run", archive)
    monkeypatch.setattr(cli_module, "train", train)
    cli_module._execute_command(_args(
        project, "train", "--restart-uncheckpointed", "--preflight-report", "runs/passed.json"
    ))
    assert calls == ["verify", "archive", "train"]
    _assert_payloads(archives[0], payloads)


def test_failed_profile_verification_cannot_move_or_overwrite_old_metadata(
    tmp_path, monkeypatch
) -> None:
    project, config = _cli_fixture(tmp_path, monkeypatch)
    run_dir = Path(config["output"]["run_dir"])
    payloads = _metadata_directory(run_dir)

    def rejected(*args, **kwargs):
        raise RuntimeError("profile is stale")

    def forbidden(*args, **kwargs):
        raise AssertionError("Unverified profile must not alter training output")

    monkeypatch.setattr(cli_module, "verify_training_preflight", rejected)
    for name in ("archive_uncheckpointed_run", "train"):
        monkeypatch.setattr(cli_module, name, forbidden)
    with pytest.raises(RuntimeError, match="profile is stale"):
        cli_module._execute_command(_args(project, "train", "--restart-uncheckpointed"))
    _assert_payloads(run_dir, payloads)


def test_rejected_archive_cannot_overwrite_existing_preflight_gate(tmp_path, monkeypatch) -> None:
    project, config = _cli_fixture(tmp_path, monkeypatch)
    run_dir = Path(config["output"]["run_dir"])
    payloads = _metadata_directory(run_dir)
    payloads["last.pt"] = b"existing checkpoint"
    (run_dir / "last.pt").write_bytes(payloads["last.pt"])
    monkeypatch.setattr(
        cli_module, "verify_training_preflight", lambda *args: {"status": "passed"}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected archive must not write a new gate or train")

    monkeypatch.setattr(cli_module, "train", forbidden)
    with pytest.raises(ValueError, match="checkpoints, history, or unknown"):
        cli_module._execute_command(_args(project, "train", "--restart-uncheckpointed"))
    _assert_payloads(run_dir, payloads)


@pytest.mark.parametrize("resume_source", ["argument", "config"])
def test_restart_and_checkpoint_resume_are_rejected_before_any_mutation(
    tmp_path, monkeypatch, resume_source: str
) -> None:
    project, config = _cli_fixture(tmp_path, monkeypatch)
    extra = ["--restart-uncheckpointed"]
    if resume_source == "argument":
        extra.extend(["--resume", "runs/last.pt"])
    else:
        config["train"]["resume"] = "runs/last.pt"

    def forbidden(*args, **kwargs):
        raise AssertionError("Conflicting resume must fail before any work")

    for name in ("verify_training_preflight", "archive_uncheckpointed_run", "train"):
        monkeypatch.setattr(cli_module, name, forbidden)
    with pytest.raises(ValueError, match="cannot be combined with resume"):
        cli_module._execute_command(_args(project, "train", *extra))


def test_restart_cannot_bypass_the_required_passed_profile(tmp_path, monkeypatch) -> None:
    project, _config = _cli_fixture(tmp_path, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("Restart with a bypass must fail before any mutation")

    for name in ("archive_uncheckpointed_run", "train"):
        monkeypatch.setattr(cli_module, name, forbidden)
    with pytest.raises(ValueError, match="preflight|profile|bypass"):
        cli_module._execute_command(_args(
            project, "train", "--restart-uncheckpointed", "--allow-unverified-preflight"
        ))


@pytest.mark.parametrize("explicit", [False, True])
def test_profile_cli_forwards_resume_reuse_and_cpu_thread_arguments(
    tmp_path, monkeypatch, explicit: bool
) -> None:
    project, config = _cli_fixture(tmp_path, monkeypatch)
    calls: list[tuple] = []
    monkeypatch.setattr(cli_module.torch, "set_num_threads", lambda value: calls.append(("cpu", value)))

    def profile(value, output, **kwargs):
        calls.append(("profile", output, kwargs))
        assert value == config
        return {"passed": True}

    monkeypatch.setattr(cli_module, "training_preflight", profile)
    extra = ["--output", "runs/new.json"]
    if explicit:
        extra.extend([
            "--resume-scan", "--reuse-report", "runs/old.json", "--cpu-threads", "2",
            "--samples", "5", "--top-density", "12",
        ])
    cli_module._execute_command(_args(project, "profile", *extra))
    assert calls == [
        ("cpu", 2 if explicit else 4),
        ("profile", project / "runs" / "new.json", {
            "profile_samples": 5 if explicit else 3,
            "top_density_count": 12 if explicit else 10,
            "require_cuda": True,
            "resume_scan": explicit,
            "reuse_report": project / "runs" / "old.json" if explicit else None,
        }),
    ]


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "many"])
def test_profile_rejects_invalid_cpu_thread_count(value: str) -> None:
    with pytest.raises(SystemExit) as caught:
        cli_module.build_parser().parse_args([
            "profile", "--config", "config.json", "--output", "out.json",
            "--cpu-threads", value,
        ])
    assert caught.value.code == 2
