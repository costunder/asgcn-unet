#!/usr/bin/env bash
# Shared Conda-only interpreter selection for interactive and scheduler wrappers.

select_conda_python() {
  DRY_RUN="${DRY_RUN:-0}"
  if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "ERROR: DRY_RUN must be 0 or 1" >&2
    return 2
  fi
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
      PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    elif [[ "${DRY_RUN}" == "1" ]]; then
      PYTHON_BIN=python
    else
      echo "ERROR: select a Conda environment with CONDA_PREFIX or PYTHON_BIN." >&2
      return 1
    fi
  fi
  # Keep user/foreign site packages out of the selected, version-checked runtime.
  export PYTHONNOUSERSITE=1
  # Set before any torch/numpy import, including profile/check entrypoints.
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
  for thread_setting in OMP_NUM_THREADS MKL_NUM_THREADS; do
    if [[ ! "${!thread_setting}" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: ${thread_setting} must be a positive integer." >&2
      return 2
    fi
  done
  unset PYTHONPATH PYTHONHOME
  export DRY_RUN PYTHON_BIN
  export RUNTIME_PROFILE="${RUNTIME_PROFILE:-constraints/server.json}"
  export CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-constraints/py312.txt}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  local executable
  executable="$(command -v -- "${PYTHON_BIN}")" || {
    echo "ERROR: selected Conda Python is not executable." >&2
    return 1
  }
  if [[ ! -x "${executable}" ]]; then
    echo "ERROR: selected Conda Python is not executable." >&2
    return 1
  fi
  PYTHON_BIN="${executable}"
  local selected_prefix
  if ! selected_prefix="$("${PYTHON_BIN}" -I - "${CONDA_PREFIX:-}" "${PYTHON_BIN}" <<'PY'
from pathlib import Path
import sys

try:
    prefix = Path(sys.prefix).resolve()
    base = Path(sys.base_prefix).resolve()
    executable = Path(sys.executable).resolve()
    selected = Path(sys.argv[2]).resolve()
    expected = Path(sys.argv[1]).resolve() if sys.argv[1] else prefix
    valid = (
        prefix == base == expected
        and (prefix / "conda-meta").is_dir()
        and executable == selected
        and executable.is_relative_to(prefix)
    )
except (OSError, RuntimeError, ValueError):
    valid = False
if not valid:
    raise SystemExit("ERROR: selected Python must belong directly to the selected Conda environment, not a venv.")
print(prefix)
PY
)"; then
    return 1
  fi
  CONDA_PREFIX="${selected_prefix}"
  export CONDA_PREFIX PYTHON_BIN
}

runtime_command() {
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "$@"
    return
  fi
  printf ' +'
  local argument display
  for argument in "$@"; do
    display="${argument}"
    if [[ "${INCLUDE_PRIVATE_HOST_PROVENANCE:-0}" != "1" ]]; then
      display="${display//${PROJECT_ROOT}/\$PROJECT_ROOT}"
      if [[ "${display}" == /* || "${display}" == [A-Za-z]:/* ]]; then
        display="\$EXTERNAL/${display##*/}"
      fi
    fi
    printf ' %q' "${display}"
  done
  printf '\n'
}

runtime_exec() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    runtime_command "$@"
  else
    exec "$@"
  fi
}

check_runtime_profile() {
  local arguments=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/check_env.py"
    --runtime-profile "${RUNTIME_PROFILE}"
    --lock "${CONSTRAINTS_FILE}"
  )
  if [[ "${REQUIRE_CUDA:-1}" == "1" ]]; then
    arguments+=(--require-cuda)
  fi
  if [[ "${INCLUDE_PRIVATE_HOST_PROVENANCE:-0}" == "1" ]]; then
    arguments+=(--include-private-host-provenance)
  fi
  runtime_command "${arguments[@]}"
}
