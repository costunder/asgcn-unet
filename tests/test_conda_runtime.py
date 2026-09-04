from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (ROOT / "scripts/runtime.sh").read_text(encoding="utf-8")
VALIDATOR = RUNTIME.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
REPORTING = ["run", "train", "eval", "calibrate"]
SCHEDULERS = [
    f"server/{stage}.{extension}"
    for extension in ("sbatch", "pbs")
    for stage in ("profile", "train", "calibrate", "eval")
]
ENTRYPOINTS = [f"scripts/{name}.sh" for name in REPORTING + ["get_hdr", "get_aid"]]


def _bash() -> str:
    candidate = shutil.which("bash")
    if candidate:
        return candidate
    git = shutil.which("git")
    if git:
        git_root = Path(git).resolve().parent.parent
        for relative in ("bin/bash.exe", "usr/bin/bash.exe", "usr/bin/sh.exe"):
            candidate_path = git_root / relative
            if candidate_path.is_file():
                return str(candidate_path)
    pytest.skip("Bash is unavailable")


def _shell(*arguments: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    bash = _bash()
    env = os.environ.copy()
    for name in (
        "CONDA_PREFIX", "PYTHON_BIN", "VIRTUAL_ENV", "BASH_ENV", "ENV",
        "SLURM_SUBMIT_DIR", "SLURM_JOB_ID", "PBS_O_WORKDIR", "PBS_JOBID",
        "CUDA_MODULE", "PROJECT_ROOT", "RESUME_CHECKPOINT",
        "ALLOW_UNVERIFIED_PREFLIGHT", "ALLOW_NONREPORTING_EVAL",
    ):
        env.pop(name, None)
    env.update(
        DRY_RUN="1",
        INCLUDE_PRIVATE_HOST_PROVENANCE="0",
        REQUIRE_CUDA="1",
        PATH=str(Path(bash).parent) + os.pathsep + env.get("PATH", ""),
    )
    env.update(overrides)
    return subprocess.run(
        [bash, "--noprofile", "--norc", *arguments],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("relative", ENTRYPOINTS + SCHEDULERS)
def test_every_entrypoint_uses_the_shared_conda_selector(relative: str) -> None:
    source = (ROOT / relative).read_text(encoding="utf-8")
    assert '.venv' not in source
    assert 'source "${PROJECT_ROOT}/scripts/runtime.sh"' in source
    assert "select_conda_python" in source


@pytest.mark.parametrize("relative", ENTRYPOINTS + SCHEDULERS + ["scripts/runtime.sh"])
def test_conda_entrypoint_shell_syntax(relative: str) -> None:
    result = _shell("-n", relative)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("explicit_prefix", [False, True])
def test_validator_accepts_only_the_actual_conda_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], explicit_prefix: bool,
) -> None:
    prefix = tmp_path / "conda"
    (prefix / "conda-meta").mkdir(parents=True)
    executable = prefix / "bin/python"
    with monkeypatch.context() as context:
        context.setattr(sys, "prefix", str(prefix))
        context.setattr(sys, "base_prefix", str(prefix))
        context.setattr(sys, "executable", str(executable))
        context.setattr(sys, "argv", ["-", str(prefix) if explicit_prefix else "", str(executable)])
        # Execute only the fixed validator extracted from this repository.
        exec(compile(VALIDATOR, "<conda-runtime-validator>", "exec"), {})  # noqa: S102
    assert capsys.readouterr().out.strip() == str(prefix.resolve())


@pytest.mark.parametrize("mismatch", ["no_conda", "venv", "prefix", "outside", "executable"])
def test_validator_rejects_non_conda_or_mixed_runtimes_without_private_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str,
) -> None:
    prefix = tmp_path / "selected"
    prefix.mkdir()
    if mismatch != "no_conda":
        (prefix / "conda-meta").mkdir()
    executable = prefix / "bin/python"
    actual = tmp_path / "outside/python" if mismatch == "outside" else executable
    selected = prefix / "bin/other-python" if mismatch == "executable" else actual
    expected = tmp_path / "other-env" if mismatch == "prefix" else prefix
    base = tmp_path / "base-env" if mismatch == "venv" else prefix
    with monkeypatch.context() as context:
        context.setattr(sys, "prefix", str(prefix))
        context.setattr(sys, "base_prefix", str(base))
        context.setattr(sys, "executable", str(actual))
        context.setattr(sys, "argv", ["-", str(expected), str(selected)])
        with pytest.raises(SystemExit, match="selected Conda environment") as error:
            # Execute only the fixed validator extracted from this repository.
            exec(compile(VALIDATOR, "<conda-runtime-validator>", "exec"), {})  # noqa: S102
    assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize("relative", ENTRYPOINTS + SCHEDULERS)
def test_real_execution_requires_explicit_conda_selection(relative: str) -> None:
    result = _shell(relative, DRY_RUN="0")
    assert result.returncode != 0
    assert "select a Conda environment" in result.stderr


@pytest.mark.parametrize("stage", REPORTING)
def test_reporting_dry_run_needs_no_python_gpu_or_checkpoint(stage: str, tmp_path: Path) -> None:
    arguments = [f"scripts/{stage}.sh"]
    if stage == "run":
        arguments.append("all")
    result = _shell(*arguments, OUTPUT_PATH=(tmp_path / "output.pt").as_posix())
    assert result.returncode == 0, result.stderr
    assert "--runtime-profile constraints/server.json" in result.stdout
    assert "--lock constraints/py312.txt" in result.stdout
    assert "--require-cuda" in result.stdout
    assert str(tmp_path) not in result.stdout


@pytest.mark.parametrize("relative", SCHEDULERS)
def test_scheduler_dry_run_needs_no_scheduler_or_conda(relative: str) -> None:
    result = _shell(relative)
    assert result.returncode == 0, result.stderr
    assert "scripts/" in result.stdout
    assert "command not found" not in result.stderr


@pytest.mark.parametrize("script", ["get_hdr", "get_aid"])
def test_download_help_and_dry_run_do_not_need_conda_or_cuda(script: str, tmp_path: Path) -> None:
    help_result = _shell(f"scripts/{script}.sh", "--help", DRY_RUN="0")
    assert help_result.returncode == 0, help_result.stderr
    assert "Usage:" in help_result.stdout
    arguments = [f"scripts/{script}.sh", "--download"] if script == "get_hdr" else [f"scripts/{script}.sh"]
    result = _shell(*arguments, EVENTAID_ROOT=(tmp_path / "missing-data").as_posix())
    assert result.returncode == 0, result.stderr
    assert "--require-cuda" not in result.stdout
    assert "--runtime-profile" not in result.stdout
    assert not (tmp_path / "missing-data").exists()


def test_selector_honors_override_and_disables_foreign_python_paths() -> None:
    result = _shell(
        "-c",
        'source scripts/runtime.sh; select_conda_python; '
        'printf "%s|%s|%s|%s" "$PYTHON_BIN" "$PYTHONNOUSERSITE" "${PYTHONPATH-unset}" "${PYTHONHOME-unset}"',
        PYTHON_BIN="/example/conda/bin/python", CONDA_PREFIX="/example/other",
        PYTHONPATH="/example/foreign", PYTHONHOME="/example/foreign",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "/example/conda/bin/python|1|unset|unset"


def test_dry_run_redacts_external_interpreter_path() -> None:
    result = _shell("scripts/train.sh", PYTHON_BIN="/example/private-conda/bin/python")
    assert result.returncode == 0, result.stderr
    assert "/example/private-conda" not in result.stdout + result.stderr
    assert "EXTERNAL/python" in result.stdout


def test_eval_wrapper_can_run_only_the_missing_half_of_a_mode() -> None:
    benchmark_only = _shell(
        "scripts/eval.sh",
        RUN_EVALUATION="0",
        RUN_BENCHMARK="1",
    )
    assert benchmark_only.returncode == 0, benchmark_only.stderr
    assert "Skipping completed quality evaluation" in benchmark_only.stdout
    assert "asgcn_unet.cli evaluate" not in benchmark_only.stdout
    assert "asgcn_unet.cli benchmark" in benchmark_only.stdout

    quality_only = _shell(
        "scripts/eval.sh",
        RUN_EVALUATION="1",
        RUN_BENCHMARK="0",
    )
    assert quality_only.returncode == 0, quality_only.stderr
    assert "asgcn_unet.cli evaluate" in quality_only.stdout
    assert "asgcn_unet.cli benchmark" not in quality_only.stdout


@pytest.mark.parametrize(
    ("run_evaluation", "run_benchmark", "message"),
    [
        ("invalid", "1", "RUN_EVALUATION must be 0 or 1"),
        ("1", "invalid", "RUN_BENCHMARK must be 0 or 1"),
        ("0", "0", "cannot both be 0"),
    ],
)
def test_eval_wrapper_rejects_invalid_work_selection(
    run_evaluation: str, run_benchmark: str, message: str,
) -> None:
    result = _shell(
        "scripts/eval.sh",
        RUN_EVALUATION=run_evaluation,
        RUN_BENCHMARK=run_benchmark,
    )
    assert result.returncode == 2
    assert message in result.stderr


def test_download_wrappers_do_not_add_a_cuda_or_package_profile_gate() -> None:
    for name in ("get_hdr", "get_aid"):
        source = (ROOT / f"scripts/{name}.sh").read_text(encoding="utf-8")
        assert "check_runtime_profile" not in source
        assert "import torch" not in source
