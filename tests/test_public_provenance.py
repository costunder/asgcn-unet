from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

from asgcn_unet import cli
from scripts import check_env
from tests.fixtures import make_eventhdr


def _run_check_env(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *arguments: str,
) -> dict[str, object]:
    monkeypatch.setattr(sys, "argv", ["check_env.py", *arguments])
    check_env.main()
    return json.loads(capsys.readouterr().out)


def test_check_env_public_report_uses_only_logical_host_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "private-data-mount"
    runs_root = tmp_path / "private-runs-mount"
    data_root.mkdir()

    report = _run_check_env(
        monkeypatch,
        capsys,
        "--data-root",
        str(data_root),
        "--runs-root",
        str(runs_root),
    )

    rendered = json.dumps(report)
    assert report["project_root"] == "$PROJECT_ROOT"
    assert report["data_root"] == "$DATA_ROOT"
    assert report["runs_root"] == "$RUNS_ROOT"
    assert "hostname" not in report
    assert str(tmp_path.resolve()) not in rendered
    assert socket.gethostname() not in rendered
    assert "gpu_devices" in report
    assert "torch_cuda_runtime" in report


def test_check_env_private_provenance_requires_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "data"
    runs_root = tmp_path / "runs"
    data_root.mkdir()

    report = _run_check_env(
        monkeypatch,
        capsys,
        "--data-root",
        str(data_root),
        "--runs-root",
        str(runs_root),
        "--include-private-host-provenance",
    )

    private = report["private_host_provenance"]
    assert isinstance(private, dict)
    assert private["hostname"] == socket.gethostname()
    assert private["data_root"] == str(data_root.resolve())
    assert private["runs_root"] == str(runs_root.resolve())
    assert "do not publish" in private["publication_warning"]


def test_check_env_routine_error_redacts_external_lock_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "data"
    runs_root = tmp_path / "runs"
    lock_path = tmp_path / "private-profile.txt"
    data_root.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_env.py",
            "--data-root",
            str(data_root),
            "--runs-root",
            str(runs_root),
            "--lock",
            str(lock_path),
        ],
    )

    with pytest.raises(SystemExit) as error:
        check_env.main()

    rendered = capsys.readouterr().out
    assert "$LOCK_FILE" in str(error.value)
    assert str(tmp_path.resolve()) not in str(error.value)
    assert str(tmp_path.resolve()) not in rendered


def test_check_env_manifest_error_redacts_repository_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    project_root = Path(check_env.__file__).resolve().parents[1]

    def fail_manifest(_path: Path) -> dict[str, object]:
        raise ValueError(f"invalid manifest at {project_root / 'manifests'}")

    monkeypatch.setattr(check_env, "load_eventhdr_split_manifest", fail_manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_env.py",
            "--data-root",
            str(data_root),
            "--runs-root",
            str(tmp_path / "runs"),
            "--require-eventhdr-train",
        ],
    )

    with pytest.raises(SystemExit) as error:
        check_env.main()

    rendered = capsys.readouterr().out
    assert "$PROJECT_ROOT/manifests" in str(error.value)
    assert str(project_root) not in str(error.value)
    assert str(project_root) not in rendered


def test_inspect_public_result_redacts_root_and_sample_source(tmp_path: Path) -> None:
    data_root = tmp_path / "eventhdr"
    make_eventhdr(data_root)
    config = {
        "dataset": {
            "type": "eventhdr",
            "root": str(data_root.resolve()),
            "max_events": 8,
        }
    }

    result = cli.inspect_dataset(config, samples=1)

    rendered = json.dumps(result)
    assert result["root"] == "$DATA_ROOT"
    assert result["preview"][0]["metadata"]["source"] == "$DATA_ROOT/test.h5"
    assert str(tmp_path.resolve()) not in rendered
    assert "private_host_provenance" not in result


def test_inspect_private_provenance_requires_explicit_opt_in(tmp_path: Path) -> None:
    data_root = tmp_path / "eventhdr"
    make_eventhdr(data_root)
    config = {
        "dataset": {
            "type": "eventhdr",
            "root": str(data_root.resolve()),
            "max_events": 8,
        }
    }

    result = cli.inspect_dataset(
        config,
        samples=1,
        include_private_host_provenance=True,
    )

    assert result["root"] == str(data_root.resolve())
    assert result["preview"][0]["metadata"]["source"].startswith(str(data_root.resolve()))
    assert result["private_host_provenance"]["data_root"] == str(data_root.resolve())
    assert "do not publish" in result["private_host_provenance"]["publication_warning"]


def test_inspect_cli_routine_error_redacts_dataset_and_config_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "inspect.json"
    missing_root = tmp_path / "private-data-mount"
    config_path.write_text(
        json.dumps({"dataset": {"type": "eventhdr", "root": str(missing_root)}}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as error:
        cli.main(["inspect", "--config", str(config_path)])

    message = str(error.value)
    assert "$DATA_ROOT" in message
    assert str(tmp_path.resolve()) not in message


def test_private_provenance_help_warns_against_publication(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as cli_exit:
        cli.build_parser().parse_args(["inspect", "--help"])
    assert cli_exit.value.code == 0
    cli_help = capsys.readouterr().out
    assert "PRIVATE" in cli_help
    assert "do not publish" in cli_help

    monkeypatch.setattr(sys, "argv", ["check_env.py", "--help"])
    with pytest.raises(SystemExit) as check_exit:
        check_env.main()
    assert check_exit.value.code == 0
    check_help = capsys.readouterr().out
    assert "PRIVATE" in check_help
    assert "do not publish" in check_help


def test_train_and_calibrate_cli_results_redact_absolute_checkpoint_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.json"
    run_dir = tmp_path / "private-run"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {},
                "model": {},
                "train": {"batch_size": 1},
                "output": {"run_dir": str(run_dir)},
            }
        ),
        encoding="utf-8",
    )
    private_best = tmp_path / "private-checkpoints" / "best.pt"
    private_snn = tmp_path / "private-checkpoints" / "best_snn.pt"
    monkeypatch.setattr(
        cli,
        "verify_training_preflight",
        lambda config, report: {
            "schema": "asgcn_preflight_verification_v1",
            "status": "verified",
            "report_eligible": True,
        },
    )
    monkeypatch.setattr(cli, "train", lambda config, resume_from=None: private_best)

    cli.main(
        [
            "train",
            "--config",
            str(config_path),
            "--preflight-report",
            str(tmp_path / "profile.json"),
        ]
    )
    train_output = capsys.readouterr().out
    assert json.loads(train_output)["best_checkpoint"] == "$EXTERNAL/best.pt"
    assert str(tmp_path.resolve()) not in train_output

    monkeypatch.setattr(cli, "calibrate", lambda *args, **kwargs: private_snn)
    cli.main(
        [
            "calibrate",
            "--config",
            str(config_path),
            "--checkpoint",
            str(private_best),
            "--output",
            str(private_snn),
        ]
    )
    calibrate_output = capsys.readouterr().out
    assert json.loads(calibrate_output)["calibrated_checkpoint"] == "$EXTERNAL/best_snn.pt"
    assert str(tmp_path.resolve()) not in calibrate_output


@pytest.mark.parametrize(
    ("command", "failure_target"),
    [
        pytest.param("profile", "training_preflight", id="profile"),
        pytest.param("train", "train", id="train"),
        pytest.param("evaluate", "evaluate", id="evaluate"),
        pytest.param("benchmark", "benchmark", id="benchmark"),
        pytest.param("calibrate", "calibrate", id="calibrate"),
    ],
)
def test_noninspect_cli_failure_redacts_paths_and_hostname_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    failure_target: str,
) -> None:
    data_root = tmp_path / "private-data"
    checkpoint = tmp_path / "private-checkpoints" / "model.pt"
    output = tmp_path / "private-output" / "result.pt"
    profile_report = tmp_path / "private-profile" / "profile.json"
    private_hostname = "private-node.internal.example"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {"type": "eventhdr", "root": str(data_root)},
                "model": {},
                "eval": {},
                "output": {"run_dir": str(tmp_path / "private-runs")},
            }
        ),
        encoding="utf-8",
    )

    def fail_command(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            f"failed at {data_root / 'sample.h5'} on {private_hostname.upper()}"
        )

    monkeypatch.delenv("INCLUDE_PRIVATE_HOST_PROVENANCE", raising=False)
    monkeypatch.setattr(cli.socket, "gethostname", lambda: "private-node")
    monkeypatch.setattr(cli.socket, "getfqdn", lambda: private_hostname)
    monkeypatch.setattr(cli, failure_target, fail_command)
    monkeypatch.setattr(
        cli,
        "verify_training_preflight",
        lambda *args, **kwargs: {
            "schema": "asgcn_preflight_verification_v1",
            "status": "verified",
            "report_eligible": True,
        },
    )
    monkeypatch.setattr(cli, "save_json", lambda *args, **kwargs: None)

    extra_arguments = {
        "profile": ["--output", str(profile_report)],
        "train": ["--preflight-report", str(profile_report)],
        "evaluate": ["--checkpoint", str(checkpoint)],
        "benchmark": ["--checkpoint", str(checkpoint)],
        "calibrate": [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
    }[command]

    with pytest.raises(SystemExit) as error:
        cli.main([command, "--config", str(config_path), *extra_arguments])

    message = str(error.value)
    assert f"{command} failed:" in message
    assert "$DATA_ROOT/sample.h5" in message
    assert "$HOST" in message
    assert str(tmp_path.resolve()) not in message
    assert "private-node" not in message.lower()
    assert error.value.__suppress_context__ is True


@pytest.mark.parametrize(
    ("command", "failure_target"),
    [
        pytest.param("profile", "training_preflight", id="profile"),
        pytest.param("train", "train", id="train"),
        pytest.param("evaluate", "evaluate", id="evaluate"),
        pytest.param("benchmark", "benchmark", id="benchmark"),
        pytest.param("calibrate", "calibrate", id="calibrate"),
    ],
)
def test_noninspect_cli_private_error_trace_requires_environment_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    failure_target: str,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {},
                "model": {},
                "eval": {},
                "output": {"run_dir": str(tmp_path / "private-runs")},
            }
        ),
        encoding="utf-8",
    )
    private_message = str(tmp_path / "private-checkpoint.pt")
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "result.pt"
    profile_report = tmp_path / "profile.json"
    monkeypatch.setenv("INCLUDE_PRIVATE_HOST_PROVENANCE", "1")
    monkeypatch.setattr(
        cli,
        failure_target,
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(private_message)),
    )
    monkeypatch.setattr(
        cli,
        "verify_training_preflight",
        lambda *args, **kwargs: {
            "schema": "asgcn_preflight_verification_v1",
            "status": "verified",
            "report_eligible": True,
        },
    )
    monkeypatch.setattr(cli, "save_json", lambda *args, **kwargs: None)

    extra_arguments = {
        "profile": ["--output", str(profile_report)],
        "train": ["--preflight-report", str(profile_report)],
        "evaluate": ["--checkpoint", str(checkpoint)],
        "benchmark": ["--checkpoint", str(checkpoint)],
        "calibrate": [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
    }[command]

    with pytest.raises(RuntimeError, match="private-checkpoint"):
        cli.main([command, "--config", str(config_path), *extra_arguments])


def test_noninspect_cli_redaction_failure_suppresses_details_and_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"dataset": {}, "model": {}, "eval": {}}),
        encoding="utf-8",
    )
    private_message = str(tmp_path / "private-checkpoint.pt")
    monkeypatch.delenv("INCLUDE_PRIVATE_HOST_PROVENANCE", raising=False)
    monkeypatch.setattr(
        cli,
        "evaluate",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(private_message)),
    )
    monkeypatch.setattr(
        cli,
        "_redact_public_error",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(private_message)),
    )

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "evaluate",
                "--config",
                str(config_path),
                "--checkpoint",
                private_message,
            ]
        )

    message = str(error.value)
    assert message == (
        "evaluate failed: details suppressed because public-safe error rendering failed"
    )
    assert str(tmp_path.resolve()) not in message
    assert error.value.__suppress_context__ is True
