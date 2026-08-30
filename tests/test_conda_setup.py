from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILE = {
    "format_version": 1,
    "python": "3.12.14",
    "torch": "2.13.0+cu126",
    "cuda": "12.6",
    "platform": "Linux",
    "machine": "x86_64",
    "environment": "conda",
}


def _script() -> str:
    return (ROOT / "scripts" / "setup.sh").read_text(encoding="utf-8")


def _preflight() -> str:
    return _script().split("<<'PY'\n", maxsplit=1)[1].split("\nPY\n", maxsplit=1)[0]


def _prepare_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    prefix = tmp_path / "private-selected-conda"
    (prefix / "conda-meta").mkdir(parents=True)
    profile = tmp_path / "server.json"
    profile.write_text(json.dumps(PROFILE), encoding="utf-8")
    lock = tmp_path / "server.txt"
    lock.write_text("# synthetic hashed lock for preflight only\n", encoding="utf-8")
    monkeypatch.setenv("CONDA_PREFIX", str(prefix))
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "test-server")
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "base_prefix", str(prefix))
    monkeypatch.setattr(sys, "argv", ["-", str(profile), str(lock)])
    monkeypatch.setattr(platform, "python_version", lambda: "3.12.14")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(platform, "libc_ver", lambda: ("glibc", "2.28"))
    return prefix, profile


def _run_preflight() -> None:
    # Execute the trusted, repository-owned preflight without invoking pip.
    exec(compile(_preflight(), "<setup-preflight>", "exec"), {})  # noqa: S102


def test_conda_preflight_accepts_exact_server_before_any_pip_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _prepare_preflight(tmp_path, monkeypatch)

    _run_preflight()

    output = capsys.readouterr().out
    assert "Conda server preflight passed" in output
    assert str(tmp_path) not in output
    assert "import torch" not in _preflight()
    assert _script().index("<<'PY'") < _script().index("-m pip install")


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("python_version", "3.12.13"),
        ("python_version", "3.13.14"),
        ("system", "Windows"),
        ("machine", "AMD64"),
        ("machine", "aarch64"),
        ("libc_ver", ("glibc", "2.27")),
        ("libc_ver", ("musl", "1.2.5")),
        ("libc_ver", ("", "")),
    ],
)
def test_conda_preflight_rejects_unsupported_host_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value: Any,
) -> None:
    _prepare_preflight(tmp_path, monkeypatch)
    monkeypatch.setattr(platform, attribute, lambda: value)

    with pytest.raises(SystemExit) as error:
        _run_preflight()

    assert str(error.value).startswith("ERROR:")
    assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize("problem", ["venv", "wrong-prefix", "no-conda-meta", "base-name", "base-path"])
def test_conda_preflight_refuses_wrong_or_base_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    prefix, _ = _prepare_preflight(tmp_path, monkeypatch)
    if problem == "venv":
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "other-base"))
    elif problem == "wrong-prefix":
        monkeypatch.setattr(sys, "prefix", str(tmp_path / "other-prefix"))
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "other-prefix"))
    elif problem == "no-conda-meta":
        (prefix / "conda-meta").rmdir()
    elif problem == "base-name":
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
    else:
        monkeypatch.setenv("CONDA_EXE", str(prefix / "bin" / "conda"))

    with pytest.raises(SystemExit) as error:
        _run_preflight()

    assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize(
    "profile",
    [
        [],
        {**PROFILE, "format_version": True},
        {**PROFILE, "python": "3.12.13"},
        {**PROFILE, "torch": "2.13.0+cpu"},
        {**PROFILE, "cuda": "13.0"},
        {**PROFILE, "environment": "venv"},
        {**PROFILE, "packages": "private-secret"},
        {**PROFILE, "private-secret": "unexpected"},
    ],
)
def test_conda_preflight_rejects_changed_or_malformed_server_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: Any
) -> None:
    _, path = _prepare_preflight(tmp_path, monkeypatch)
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        _run_preflight()

    assert "private-secret" not in str(error.value)
    assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize("problem", ["missing-profile", "invalid-json", "missing-lock"])
def test_conda_preflight_metadata_errors_are_public_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    _, profile = _prepare_preflight(tmp_path, monkeypatch)
    if problem == "missing-profile":
        profile.unlink()
    elif problem == "invalid-json":
        profile.write_text("{private-secret", encoding="utf-8")
    else:
        Path(sys.argv[2]).unlink()

    with pytest.raises(SystemExit) as error:
        _run_preflight()

    assert str(tmp_path) not in str(error.value)
    assert "private-secret" not in str(error.value)


def test_installer_has_no_environment_creation_activation_or_unlocked_bootstrap() -> None:
    source = _script()
    for forbidden in ("-m venv", "conda create", "conda activate", "source ", "--upgrade", "rm -"):
        assert forbidden not in source
    assert 'CONDA_PYTHON="${CONDA_PREFIX}/bin/python"' in source
    assert "PIP_CONFIG_FILE=/dev/null" in source
    assert "PIP_REQUIRE_VIRTUALENV=0" in source
    assert "PYTHONNOUSERSITE=1" in source
    assert "unset PYTHONPATH PYTHONHOME" in source
    assert "--no-user --require-hashes --only-binary=:all:" in source
    assert '--no-user --no-deps --no-build-isolation -e "${PROJECT_ROOT}[dev]"' in source
    assert "--lock constraints/py312.txt --runtime-profile constraints/server.json" in source
    assert "PIP_EXTRA_ARGS" not in source
    assert "TORCH_INDEX_URL" not in source
    assert "PYTHON_BIN" not in source


LINUX_SHELL = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="executable installer workflow requires Linux Bash",
)


def _shell_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    setup = scripts / "setup.sh"
    setup.write_text(_script(), encoding="utf-8")
    prefix = tmp_path / "selected-conda"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "conda-meta").mkdir()
    fake_python = prefix / "bin" / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "stage = ('preflight' if args[0] == '-' else 'runtime' if args[0].endswith('check_env.py') "
        "else 'pipcheck' if args == ['-m', 'pip', 'check'] "
        "else 'locked' if '--require-hashes' in args else 'editable')\n"
        "with open(os.environ['MOCK_LOG'], 'a', encoding='utf-8') as log:\n"
        "    log.write(json.dumps({'args': args, 'stage': stage, 'config': os.getenv('PIP_CONFIG_FILE'), "
        "'user': os.getenv('PIP_USER'), 'proxy': os.getenv('HTTPS_PROXY'), "
        "'require_venv': os.getenv('PIP_REQUIRE_VIRTUALENV'), "
        "'no_user_site': os.getenv('PYTHONNOUSERSITE'), 'python_path': os.getenv('PYTHONPATH'), "
        "'python_home': os.getenv('PYTHONHOME')}) + '\\n')\n"
        "if stage == 'preflight':\n"
        "    sys.stdin.read()\n"
        "if os.getenv('MOCK_FAIL_STAGE') == stage:\n"
        "    raise SystemExit(9)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("CONDA_") or name in {
            "PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_PYTHON", "PIP_USER", "REQUIRE_CUDA",
            "MOCK_FAIL_STAGE"
        }:
            del env[name]
    env.update(
        {
            "CONDA_PREFIX": str(prefix),
            "CONDA_DEFAULT_ENV": "test-server",
            "MOCK_LOG": str(tmp_path / "calls.jsonl"),
            "HTTPS_PROXY": "http://proxy.example:8080",
        }
    )
    return project, setup, env


def _shell_run(setup: Path, env: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
    result = subprocess.run(
        ["bash", str(setup)], env=env, capture_output=True, text=True, timeout=30, check=False
    )
    log = Path(env["MOCK_LOG"])
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


@LINUX_SHELL
@pytest.mark.parametrize("require_cuda", [None, "0", "1"])
def test_installer_uses_only_selected_conda_and_locked_commands(
    tmp_path: Path, require_cuda: str | None
) -> None:
    project, setup, env = _shell_fixture(tmp_path)
    marker = tmp_path / "dotenv-was-executed"
    (project / ".env").write_text(f"touch '{marker}'\nREQUIRE_CUDA=invalid\n", encoding="utf-8")
    env.update(
        {
            "PYTHON_BIN": "must-not-run",
            "VENV_DIR": str(tmp_path / "must-not-create"),
            "TORCH_VERSION": "invalid",
            "TORCH_INDEX_URL": "https://must-not-use.example",
            "PIP_EXTRA_ARGS": "--no-deps --target /must-not-use",
            "PIP_CONFIG_FILE": str(tmp_path / "ignored-pip.conf"),
            "PIP_USER": "0",
            "PIP_REQUIRE_VIRTUALENV": "1",
            "PYTHONPATH": str(tmp_path / "foreign-packages"),
            "PYTHONHOME": str(tmp_path / "foreign-python"),
        }
    )
    if require_cuda is not None:
        env["REQUIRE_CUDA"] = require_cuda

    result, calls = _shell_run(setup, env)

    assert result.returncode == 0, result.stderr
    assert [call["stage"] for call in calls] == ["preflight", "locked", "editable", "pipcheck", "runtime"]
    assert calls[1]["args"] == [
        "-m", "pip", "install", "--no-user", "--require-hashes", "--only-binary=:all:",
        "-r", str(project / "constraints" / "server.txt"),
    ]
    assert calls[2]["args"] == [
        "-m", "pip", "install", "--no-user", "--no-deps", "--no-build-isolation", "-e", f"{project}[dev]"
    ]
    assert calls[4]["args"] == [
        "scripts/check_env.py", "--lock", "constraints/py312.txt", "--runtime-profile",
        "constraints/server.json", *(["--require-cuda"] if require_cuda == "1" else []),
    ]
    assert all(call["config"] == "/dev/null" and call["user"] == "0" for call in calls)
    assert all(call["proxy"] == env["HTTPS_PROXY"] for call in calls)
    assert all(call["no_user_site"] == "1" for call in calls)
    assert all(call["require_venv"] == "0" for call in calls)
    assert all(call["python_path"] is None and call["python_home"] is None for call in calls)
    assert not marker.exists()
    assert not (tmp_path / "must-not-create").exists()
    assert not (project / ".venv").exists()
    assert "Conda installation and exact runtime verification complete" in result.stdout


@LINUX_SHELL
@pytest.mark.parametrize("value", ["", "2", "true", "no"])
def test_installer_rejects_invalid_cuda_switch_before_interpreter(tmp_path: Path, value: str) -> None:
    _, setup, env = _shell_fixture(tmp_path)
    env["REQUIRE_CUDA"] = value

    result, calls = _shell_run(setup, env)

    assert result.returncode != 0
    assert "REQUIRE_CUDA must be 0 or 1" in result.stderr
    assert calls == []


@LINUX_SHELL
@pytest.mark.parametrize("name", ["PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_PYTHON", "PIP_USER"])
def test_installer_rejects_pip_destination_environment_overrides(tmp_path: Path, name: str) -> None:
    _, setup, env = _shell_fixture(tmp_path)
    env[name] = "1" if name == "PIP_USER" else str(tmp_path / "other-prefix")

    result, calls = _shell_run(setup, env)

    assert result.returncode != 0
    assert calls == []
    assert str(tmp_path) not in result.stderr


@LINUX_SHELL
@pytest.mark.parametrize("stage", ["preflight", "locked", "editable", "pipcheck", "runtime"])
def test_installer_stops_immediately_when_any_stage_fails(tmp_path: Path, stage: str) -> None:
    _, setup, env = _shell_fixture(tmp_path)
    env["MOCK_FAIL_STAGE"] = stage

    result, calls = _shell_run(setup, env)

    stages = ["preflight", "locked", "editable", "pipcheck", "runtime"]
    assert result.returncode != 0
    assert [call["stage"] for call in calls] == stages[:stages.index(stage) + 1]
    assert "verification complete" not in result.stdout


@LINUX_SHELL
@pytest.mark.parametrize(
    "problem", ["missing-prefix", "relative-prefix", "missing-metadata", "missing-python", "base"]
)
def test_installer_rejects_invalid_conda_selection_before_any_interpreter(
    tmp_path: Path, problem: str
) -> None:
    _, setup, env = _shell_fixture(tmp_path)
    prefix = Path(env["CONDA_PREFIX"])
    if problem == "missing-prefix":
        del env["CONDA_PREFIX"]
    elif problem == "relative-prefix":
        env["CONDA_PREFIX"] = "relative-prefix"
    elif problem == "missing-metadata":
        (prefix / "conda-meta").rmdir()
    elif problem == "missing-python":
        (prefix / "bin" / "python").unlink()
    else:
        env["CONDA_DEFAULT_ENV"] = "base"

    result, calls = _shell_run(setup, env)

    assert result.returncode != 0
    assert calls == []
    assert str(tmp_path) not in result.stderr
