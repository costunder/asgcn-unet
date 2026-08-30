from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import check_env

PROFILE = {
    "format_version": 1,
    "python": "3.12.14",
    "torch": "2.13.0+cu126",
    "cuda": "12.6",
    "platform": "Linux",
    "machine": "x86_64",
    "environment": "conda",
}


def _matching_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    prefix = tmp_path / "private-conda-prefix"
    (prefix / "conda-meta").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "base_prefix", str(prefix))
    monkeypatch.setattr(check_env.platform, "python_version", lambda: PROFILE["python"])
    monkeypatch.setattr(check_env.platform, "system", lambda: "Linux")
    monkeypatch.setattr(check_env.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(check_env.platform, "platform", lambda: "Linux-test-runtime")
    monkeypatch.setattr(check_env.platform, "libc_ver", lambda: ("glibc", "2.28"))
    monkeypatch.setattr(
        check_env,
        "torch",
        SimpleNamespace(__version__=PROFILE["torch"], version=SimpleNamespace(cuda="12.6")),
    )
    monkeypatch.setattr(check_env, "_cuda_inventory", lambda: (False, [], []))
    return prefix


def _profile_file(tmp_path: Path, value: Any = None) -> Path:
    path = tmp_path / "private-runtime-profile.json"
    path.write_text(json.dumps(PROFILE if value is None else value), encoding="utf-8")
    return path


def _arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile_path: Path | None,
    *extra: str,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)
    arguments = [
        "check_env.py",
        "--data-root",
        str(data_root),
        "--runs-root",
        str(tmp_path / "runs"),
        *extra,
    ]
    if profile_path is not None:
        arguments.extend(["--runtime-profile", str(profile_path)])
    monkeypatch.setattr(sys, "argv", arguments)


def test_exact_conda_runtime_matches_public_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _matching_runtime(monkeypatch, tmp_path)
    _arguments(monkeypatch, tmp_path, _profile_file(tmp_path))

    check_env.main()

    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["runtime_profile"] == "$RUNTIME_PROFILE"
    assert report["runtime_profile_match"] is True
    assert report["runtime_profile_mismatches"] == {}
    assert report["runtime_environment"] == "conda"
    assert str(tmp_path) not in output
    assert "private-conda-prefix" not in output


@pytest.mark.parametrize(
    "field,actual",
    [
        ("python", "3.12.13"),
        ("python", "3.13.14"),
        ("torch", "2.13.0"),
        ("torch", "2.13.0+cpu"),
        ("torch", "2.13.0+cu130"),
        ("cuda", "13.0"),
        ("cuda", None),
        ("platform", "Windows"),
        ("machine", "AMD64"),
        ("machine", "aarch64"),
    ],
)
def test_runtime_profile_rejects_each_exact_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
    actual: str | None,
) -> None:
    _matching_runtime(monkeypatch, tmp_path)
    if field == "torch":
        monkeypatch.setattr(check_env.torch, "__version__", actual)
    elif field == "cuda":
        monkeypatch.setattr(check_env.torch.version, "cuda", actual)
    else:
        attribute = {"python": "python_version", "platform": "system", "machine": "machine"}[field]
        monkeypatch.setattr(check_env.platform, attribute, lambda: actual)
    _arguments(monkeypatch, tmp_path, _profile_file(tmp_path))

    with pytest.raises(SystemExit) as error:
        check_env.main()

    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["runtime_profile_match"] is False
    assert report["runtime_profile_mismatches"] == {
        field: {"expected": PROFILE[field], "actual": actual}
    }
    assert f"{field}: expected {PROFILE[field]}, found {actual}" in str(error.value)
    assert str(tmp_path) not in output + str(error.value)


@pytest.mark.parametrize("environment", ["venv", "non-conda"])
def test_runtime_profile_requires_real_conda_interpreter_not_environment_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    environment: str,
) -> None:
    prefix = _matching_runtime(monkeypatch, tmp_path)
    monkeypatch.setenv("CONDA_PREFIX", str(prefix))
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "server")
    if environment == "venv":
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "private-base-prefix"))
    else:
        (prefix / "conda-meta").rmdir()
    _arguments(monkeypatch, tmp_path, _profile_file(tmp_path))

    with pytest.raises(SystemExit) as error:
        check_env.main()

    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["runtime_environment"] == environment
    assert report["runtime_profile_mismatches"] == {
        "environment": {"expected": "conda", "actual": environment}
    }
    assert str(tmp_path) not in output + str(error.value)


def test_runtime_profile_accepts_conda_without_activation_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _matching_runtime(monkeypatch, tmp_path)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)

    assert check_env._check_runtime_profile(PROFILE) == ({}, "conda")


def test_conda_meta_must_be_a_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = _matching_runtime(monkeypatch, tmp_path)
    (prefix / "conda-meta").rmdir()
    (prefix / "conda-meta").write_text("not a directory", encoding="utf-8")

    mismatches, environment = check_env._check_runtime_profile(PROFILE)

    assert environment == "non-conda"
    assert mismatches["environment"] == {"expected": "conda", "actual": "non-conda"}


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        {**PROFILE, "format_version": True},
        {**PROFILE, "format_version": 2},
        {**PROFILE, "python": "3.12"},
        {**PROFILE, "python": None},
        {**PROFILE, "torch": "2.13.0"},
        {**PROFILE, "torch": "2.13.0+*"},
        {**PROFILE, "cuda": 12.6},
        {**PROFILE, "platform": "Windows"},
        {**PROFILE, "machine": "AMD64"},
        {**PROFILE, "environment": "venv"},
        {**PROFILE, "unexpected": "private-secret"},
    ],
)
def test_runtime_profile_rejects_malformed_schema_without_private_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    value: Any,
) -> None:
    _matching_runtime(monkeypatch, tmp_path)
    _arguments(monkeypatch, tmp_path, _profile_file(tmp_path, value))

    with pytest.raises(SystemExit) as error:
        check_env.main()

    output = capsys.readouterr()
    assert "Runtime profile check failed" in str(error.value)
    assert error.value.__suppress_context__ is True
    assert output.out == ""
    assert str(tmp_path) not in str(error.value) + output.err
    assert "private-secret" not in str(error.value) + output.err


@pytest.mark.parametrize("kind", ["missing", "directory", "invalid-json", "invalid-unicode"])
def test_runtime_profile_file_failures_are_fatal_and_public_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    _matching_runtime(monkeypatch, tmp_path)
    path = tmp_path / "private-profile.json"
    if kind == "directory":
        path.mkdir()
    elif kind == "invalid-json":
        path.write_text("{private-secret", encoding="utf-8")
    elif kind == "invalid-unicode":
        path.write_bytes(b"\xffprivate-secret")
    _arguments(monkeypatch, tmp_path, path)

    with pytest.raises(SystemExit) as error:
        check_env.main()

    output = capsys.readouterr()
    assert "Runtime profile check failed" in str(error.value)
    assert "$RUNTIME_PROFILE" in str(error.value)
    assert error.value.__suppress_context__ is True
    assert str(tmp_path) not in str(error.value) + output.out + output.err
    assert "private-secret" not in str(error.value) + output.out + output.err


def test_runtime_profile_paths_only_appear_with_private_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prefix = _matching_runtime(monkeypatch, tmp_path)
    path = _profile_file(tmp_path)
    _arguments(monkeypatch, tmp_path, path, "--include-private-host-provenance")

    check_env.main()

    report = json.loads(capsys.readouterr().out)
    private = report["private_host_provenance"]
    assert private["runtime_profile"] == str(path.resolve())
    assert private["interpreter_prefix"] == str(prefix)


@pytest.mark.parametrize(
    "relative,label",
    [
        ("constraints/server.json", "$PROJECT_ROOT/constraints/server.json"),
        ("runs/private-runtime-profile.json", "$RUNTIME_PROFILE"),
    ],
)
def test_runtime_profile_hides_custom_paths_even_inside_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    relative: str,
    label: str,
) -> None:
    _matching_runtime(monkeypatch, tmp_path)
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    monkeypatch.setattr(check_env, "__file__", str(project / "scripts" / "check_env.py"))
    profile = project / relative
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(json.dumps(PROFILE), encoding="utf-8")
    _arguments(monkeypatch, tmp_path, profile)

    check_env.main()

    output = capsys.readouterr().out
    assert json.loads(output)["runtime_profile"] == label
    assert "private-runtime-profile" not in output
    assert str(tmp_path) not in output


def test_runtime_profile_is_optional_for_existing_cross_platform_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _matching_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    monkeypatch.setattr(check_env.platform, "system", lambda: "Windows")
    monkeypatch.setattr(check_env.torch, "__version__", "2.13.0+cpu")
    _arguments(monkeypatch, tmp_path, None)

    check_env.main()

    report = json.loads(capsys.readouterr().out)
    assert report["runtime_profile"] is None
    assert report["runtime_profile_match"] is None
    assert report["runtime_profile_mismatches"] is None


@pytest.mark.parametrize(
    "expected,actual,matched",
    [
        ("2.13.0", "2.13.0", True),
        ("2.13.0", "2.13.0+cpu", True),
        ("2.13.0", "2.13.0+cu126", True),
        ("2.13.0", "2.14.0+cu126", False),
        ("2.13.0+cu126", "2.13.0+cu126", True),
        ("2.13.0+cu126", "2.13.0", False),
        ("2.13.0+cu126", "2.13.0+cpu", False),
        ("2.13.0+cu126", "2.13.0+cu130", False),
    ],
)
def test_dependency_lock_preserves_explicit_local_build_suffixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected: str,
    actual: str,
    matched: bool,
) -> None:
    path = tmp_path / "versions.txt"
    path.write_text(f"torch=={expected}\n", encoding="utf-8")
    monkeypatch.setattr(check_env.importlib.metadata, "version", lambda _name: actual)

    mismatches = check_env._check_lock(path)

    assert mismatches == ({} if matched else {"torch": {"expected": expected, "actual": actual}})


def test_dependency_lock_missing_exact_build_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "versions.txt"
    path.write_text("torch==2.13.0+cu126\n", encoding="utf-8")

    def missing_package(_name: str) -> str:
        raise check_env.importlib.metadata.PackageNotFoundError("torch")

    monkeypatch.setattr(check_env.importlib.metadata, "version", missing_package)

    assert check_env._check_lock(path) == {"torch": {"expected": "2.13.0+cu126", "actual": None}}


def test_runtime_profile_verifies_full_exact_package_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _matching_runtime(monkeypatch, tmp_path)
    packages = {"torch": "2.13.0+cu126", "nvidia-cuda-runtime-cu12": "12.6.77", "numpy": "2.5.2"}
    path = _profile_file(tmp_path, {**PROFILE, "packages": packages})
    calls: list[str] = []

    def installed_version(name: str) -> str:
        calls.append(name)
        return packages[name]

    monkeypatch.setattr(check_env.importlib.metadata, "version", installed_version)
    _arguments(monkeypatch, tmp_path, path)

    check_env.main()

    report = json.loads(capsys.readouterr().out)
    assert report["runtime_profile_match"] is True
    assert report["runtime_profile_mismatches"] == {}
    assert set(calls) == set(packages)


@pytest.mark.parametrize("actual", [None, "2.13.0", "2.13.0+cpu", "2.13.0+cu130"])
def test_runtime_profile_package_build_mismatch_is_precise_and_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    actual: str | None,
) -> None:
    _matching_runtime(monkeypatch, tmp_path)
    path = _profile_file(tmp_path, {**PROFILE, "packages": {"torch": "2.13.0+cu126"}})

    def installed_version(_name: str) -> str:
        if actual is None:
            raise check_env.importlib.metadata.PackageNotFoundError("torch")
        return actual

    monkeypatch.setattr(check_env.importlib.metadata, "version", installed_version)
    _arguments(monkeypatch, tmp_path, path)

    with pytest.raises(SystemExit) as error:
        check_env.main()

    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["runtime_profile_match"] is False
    assert report["runtime_profile_mismatches"] == {
        "packages.torch": {"expected": "2.13.0+cu126", "actual": actual}
    }
    assert f"packages.torch: expected 2.13.0+cu126, found {actual}" in str(error.value)
    assert str(tmp_path) not in output + str(error.value)


def test_runtime_profile_normalizes_distribution_names_without_relaxing_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _matching_runtime(monkeypatch, tmp_path)
    path = _profile_file(
        tmp_path,
        {**PROFILE, "packages": {"Typing_Extensions": "4.16.0", "Example.Package": "1.2.3+local"}},
    )
    expected = {"typing-extensions": "4.16.0", "example-package": "1.2.3+local"}
    monkeypatch.setattr(check_env.importlib.metadata, "version", expected.__getitem__)

    profile = check_env._runtime_profile(path)

    assert profile["packages"] == expected
    assert check_env._check_runtime_profile(profile) == ({}, "conda")


@pytest.mark.parametrize(
    "packages",
    [
        None,
        [],
        "private-secret",
        {"torch": None},
        {"torch": 2.13},
        {"torch": ""},
        {"torch": ">=2.13.0"},
        {"torch": "2.13.*"},
        {"torch": "2..13.0"},
        {"torch": "https://private.example/torch.whl"},
        {"../private-package": "1.0.0"},
        {"": "1.0.0"},
        {"typing_extensions": "4.16.0", "Typing-Extensions": "4.16.0"},
    ],
)
def test_runtime_profile_rejects_malformed_or_ambiguous_package_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    packages: Any,
) -> None:
    _matching_runtime(monkeypatch, tmp_path)
    path = _profile_file(tmp_path, {**PROFILE, "packages": packages})
    _arguments(monkeypatch, tmp_path, path)

    with pytest.raises(SystemExit) as error:
        check_env.main()

    output = capsys.readouterr()
    assert "Runtime profile check failed" in str(error.value)
    assert output.out == ""
    assert "private" not in str(error.value).lower()
    assert str(tmp_path) not in str(error.value) + output.err
