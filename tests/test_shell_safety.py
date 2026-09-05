"""Smoke-test shell isolation and orchestration with no GPU or dataset execution."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.test_conda_runtime import ENTRYPOINTS, ROOT, SCHEDULERS, _shell

EXECUTABLES = ENTRYPOINTS + SCHEDULERS + ["scripts/setup.sh"]


@pytest.mark.parametrize("relative", EXECUTABLES)
def test_entrypoint_has_no_session_termination_command(relative: str) -> None:
    source = (ROOT / relative).read_text(encoding="utf-8")
    assert not re.search(r"\b(?:exit|logout|shutdown|reboot|poweroff|halt)\b", source)
    assert 'if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then' in source
    assert "_asgcn_entrypoint() (" in source


@pytest.mark.parametrize("relative", EXECUTABLES)
def test_accidental_source_preserves_errexit_traps_directory_and_variables(relative: str) -> None:
    result = _shell(
        "-c",
        'set -Eeuo pipefail; PROJECT_ROOT=caller-project; '
        'trap "printf caller-error >&2" ERR; '
        'before_flags=$-; before_trap=$(trap -p ERR); before_dir=$PWD; '
        'source "$1" --invalid-argument; '
        '[[ "$before_flags" == "$-" && "$before_trap" == "$(trap -p ERR)" '
        '&& "$before_dir" == "$PWD" && "$PROJECT_ROOT" == caller-project ]]; '
        'printf "caller-alive"',
        "source-smoke-test",
        relative,
        DRY_RUN="0",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "caller-alive"
    assert "Source ignored" in result.stderr
    assert "caller-error" not in result.stderr


def test_runtime_command_does_not_replace_the_calling_shell() -> None:
    result = _shell(
        "-c",
        'source scripts/runtime.sh; DRY_RUN=0; '
        'runtime_exec false; command_status=$?; printf "caller-alive:%s" "$command_status"',
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "caller-alive:1"


@pytest.mark.parametrize("visible", ["3,6", "MIG-smoke-test-allocation"])
def test_runtime_preserves_user_or_scheduler_gpu_allocation(visible: str) -> None:
    result = _shell(
        "-c",
        'source scripts/runtime.sh; select_conda_python; printf "%s" "$CUDA_VISIBLE_DEVICES"',
        CUDA_VISIBLE_DEVICES=visible,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == visible


@pytest.mark.parametrize("stage", ["run", "train", "eval", "calibrate"])
def test_measured_fast_defaults_route_existing_checkpoint_and_config(stage: str) -> None:
    result = _shell(f"scripts/{stage}.sh")
    assert result.returncode == 0, result.stderr
    config = "hdr-fast.json" if stage == "eval" else "fast.json"
    assert f"configs/{config}" in result.stdout
    if stage != "train":
        assert "runs/fast/best.pt" in result.stdout
    else:
        assert "runs/fast-profile.json" in result.stdout


def test_explicit_baseline_routes_are_preserved() -> None:
    result = _shell("scripts/run.sh", "all", EXPERIMENT="single")
    assert result.returncode == 0, result.stderr
    assert "configs/train.json" in result.stdout
    assert "runs/train/best.pt" in result.stdout
    assert "runs/fast/best.pt" not in result.stdout


@pytest.mark.parametrize(
    ("config", "override", "expected"),
    [
        ("configs/aid-fast.json", "", ""),
        ("configs/hdr-fast.json", "", ""),
        ("configs/aid.json", "", ""),
        ("other/aid-fast.json", "", ""),
        ("configs/aid-fast.json", "8000000", "8000000"),
        ("configs/hdr-fast.json", "8000000", "8000000"),
    ],
)
def test_measured_edge_guard_is_scoped_and_explicit_override_wins(
    config: str, override: str, expected: str,
) -> None:
    result = _shell(
        "-c",
        'PROJECT_ROOT=$PWD; source scripts/runtime.sh; evaluation_graph_edge_guard "$1"',
        "guard-smoke-test",
        config,
        EVAL_MAX_GRAPH_EDGES=override,
        AID_MAX_GRAPH_EDGES="",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


def test_fast_aid_quality_and_benchmark_receive_the_same_explicit_guard() -> None:
    result = _shell("scripts/eval.sh", "configs/aid-fast.json", EVAL_MAX_GRAPH_EDGES="7475202")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("--max-graph-edges-override 7475202") == 2


def test_unspecified_edge_guard_is_resolved_by_the_config_not_duplicated_in_shell() -> None:
    result = _shell("scripts/eval.sh", "configs/aid-fast.json", EVAL_MAX_GRAPH_EDGES="")
    assert result.returncode == 0, result.stderr
    assert "--max-graph-edges-override" not in result.stdout
    assert result.stdout.count("--config configs/aid-fast.json") >= 2


@pytest.mark.parametrize("config", ["configs/aid-fast.json", "./configs/aid-fast.json"])
def test_explicit_aid_guard_applies_only_to_the_measured_aid_config(config: str) -> None:
    result = _shell(
        "-c",
        'PROJECT_ROOT=$PWD; source scripts/runtime.sh; evaluation_graph_edge_guard "$1"',
        "guard-smoke-test", config, AID_MAX_GRAPH_EDGES="8000000",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "8000000"


@pytest.mark.parametrize("allocation", ["SLURM_CPUS_PER_TASK", "NCPUS"])
def test_runtime_honors_scheduler_cpu_thread_allocation(allocation: str) -> None:
    result = _shell(
        "-c",
        'source scripts/runtime.sh; select_conda_python; printf "%s:%s" "$OMP_NUM_THREADS" "$MKL_NUM_THREADS"',
        **{allocation: "6", "OMP_NUM_THREADS": "", "MKL_NUM_THREADS": ""},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "6:6"


@pytest.mark.parametrize(("status", "state"), [(7, "FAILED"), (75, "PAUSED")])
def test_failed_or_paused_training_preserves_status_without_later_stage(
    tmp_path: Path, status: int, state: str,
) -> None:
    # Isolated smoke fixture: replace only the child-stage launcher with an error.
    project = tmp_path / "shell-smoke-test"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "run.sh").write_text(
        (ROOT / "scripts/run.sh").read_text(encoding="utf-8"), encoding="utf-8",
    )
    (scripts / "runtime.sh").write_text(
        'select_conda_python() { PYTHON_BIN=unused; RUNTIME_PROFILE=constraints/server.json; }\n'
        'env() { return "${MOCK_STAGE_STATUS}"; }\n',
        encoding="utf-8",
    )
    for relative in ("constraints/py312.txt", "configs/fast.json", "configs/hdr-fast.json", "configs/aid-fast.json"):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    result = _shell(
        (scripts / "run.sh").as_posix(), "train", DRY_RUN="0",
        EXPERIMENT="fast", MOCK_STAGE_STATUS=str(status),
        STATUS_DIR="smoke-status",
    )
    assert result.returncode == status, result.stderr
    record = json.loads((project / "smoke-status/train.json").read_text(encoding="utf-8"))
    assert record["state"] == state
    assert record["exit_code"] == status
    assert "completed" not in result.stdout
    assert "[calibrate]" not in result.stdout


@pytest.mark.parametrize("relative", ["scripts/train.sh", "scripts/eval.sh", "scripts/calibrate.sh"])
def test_reporting_wrapper_does_not_probe_cuda_outside_shared_preflight(relative: str) -> None:
    source = (ROOT / relative).read_text(encoding="utf-8")
    assert "check_runtime_profile" in source
    assert "torch.cuda" not in source


@pytest.mark.parametrize("relative", SCHEDULERS)
def test_scheduler_does_not_query_all_physical_gpus(relative: str) -> None:
    assert "nvidia-smi" not in (ROOT / relative).read_text(encoding="utf-8")
