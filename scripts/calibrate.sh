#!/usr/bin/env bash
# Entry scripts must not change an interactive shell through accidental sourcing.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' "Source ignored: run this entrypoint with bash or its scheduler; the current shell is unchanged." >&2
else
_asgcn_entrypoint() (
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  printf '%s\n' \
    "Usage: bash scripts/calibrate.sh [CONFIG [ANN_CHECKPOINT [SNN_CHECKPOINT]]]" \
    "" \
    "Environment:" \
    "  CALIBRATION_SAMPLES=all|N   Default: all; partial N is rejected for reporting" \
    "  OVERWRITE_CALIBRATION=0|1  Default: 0; protect an existing output" \
    "  VALIDATE_DATASET=0|1       Default: 1" \
    "  INSPECT_VALIDATE_ALL=0|1   Default: 0" \
    "  INSPECT_SAMPLES=N          Default: 1" \
    "  REQUIRE_CUDA=0|1           Default: 1" \
    "  PYTHON_BIN=PATH            Default: CONDA_PREFIX/bin/python"
  return 0
fi

EXPERIMENT="${EXPERIMENT:-fast}"
case "${EXPERIMENT}" in
  single) DEFAULT_CONFIG_PATH=configs/train.json; DEFAULT_RUN_DIR=runs/train ;;
  batch) DEFAULT_CONFIG_PATH=configs/batch.json; DEFAULT_RUN_DIR=runs/batch ;;
  fast) DEFAULT_CONFIG_PATH=configs/fast.json; DEFAULT_RUN_DIR=runs/fast ;;
  *) echo "ERROR: EXPERIMENT must be single, batch or fast" >&2; return 2 ;;
esac
CONFIG_PATH="${1:-${CONFIG_PATH:-${DEFAULT_CONFIG_PATH}}}"
CHECKPOINT_PATH="${2:-${CHECKPOINT_PATH:-${DEFAULT_RUN_DIR}/best.pt}}"
OUTPUT_PATH="${3:-${OUTPUT_PATH:-${DEFAULT_RUN_DIR}/best_snn.pt}}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-all}"
OVERWRITE_CALIBRATION="${OVERWRITE_CALIBRATION:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
INSPECT_VALIDATE_ALL="${INSPECT_VALIDATE_ALL:-0}"
export INCLUDE_PRIVATE_HOST_PROVENANCE="${INCLUDE_PRIVATE_HOST_PROVENANCE:-0}"

path_log_label() {
  local path="$1"
  if [[ "${INCLUDE_PRIVATE_HOST_PROVENANCE}" == "1" ]]; then
    printf '%s' "${path}"
  else
    printf '%s' "${path##*/}"
  fi
}

cd "${PROJECT_ROOT}"
if [[ "${INCLUDE_PRIVATE_HOST_PROVENANCE}" != "0" \
  && "${INCLUDE_PRIVATE_HOST_PROVENANCE}" != "1" ]]; then
  echo "ERROR: INCLUDE_PRIVATE_HOST_PROVENANCE must be 0 or 1" >&2
  return 2
fi
# shellcheck source=scripts/runtime.sh
source "${PROJECT_ROOT}/scripts/runtime.sh"
select_conda_python
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: calibration config not found: $(path_log_label "${CONFIG_PATH}")" >&2
  return 1
fi
if [[ "${DRY_RUN}" != "1" && ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: ANN checkpoint not found: $(path_log_label "${CHECKPOINT_PATH}")" >&2
  return 1
fi
for flag_name in REQUIRE_CUDA VALIDATE_DATASET INSPECT_VALIDATE_ALL OVERWRITE_CALIBRATION; do
  flag_value="${!flag_name}"
  if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
    echo "ERROR: ${flag_name} must be 0 or 1" >&2
    return 2
  fi
done

runtime_command "${PYTHON_BIN}" - "${CHECKPOINT_PATH}" "${OUTPUT_PATH}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).expanduser().resolve()
output = Path(sys.argv[2]).expanduser().resolve()
if source == output:
    raise SystemExit("ANN input and calibrated SNN output must be different files")
PY

if [[ -e "${OUTPUT_PATH}" || -L "${OUTPUT_PATH}" ]]; then
  if [[ "${OVERWRITE_CALIBRATION}" != "1" ]]; then
    echo "ERROR: calibrated output already exists: $(path_log_label "${OUTPUT_PATH}")" >&2
    echo "Set OVERWRITE_CALIBRATION=1 only when replacing it is intentional." >&2
    return 1
  fi
  if [[ -d "${OUTPUT_PATH}" ]]; then
    echo "ERROR: calibrated output path is a directory: $(path_log_label "${OUTPUT_PATH}")" >&2
    return 1
  fi
fi

check_runtime_profile
# The shared preflight reports the runtime only after allocation verification.

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

if [[ "${VALIDATE_DATASET}" == "1" ]]; then
  INSPECT_ARGS=(
    --config "${CONFIG_PATH}"
    --samples "${INSPECT_SAMPLES}"
  )
  if [[ "${INSPECT_VALIDATE_ALL}" == "1" ]]; then
    INSPECT_ARGS+=(--validate-all)
  fi
  runtime_command "${PYTHON_BIN}" -m asgcn_unet.cli inspect "${INSPECT_ARGS[@]}"
fi

if [[ "${CALIBRATION_SAMPLES}" != "all" && ! "${CALIBRATION_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CALIBRATION_SAMPLES must be 'all' or a positive integer" >&2
  return 2
fi

echo "Calibrating $(path_log_label "${CHECKPOINT_PATH}") with ${CALIBRATION_SAMPLES} EventHDR samples"
CALIBRATE_ARGS=(
  --config "${CONFIG_PATH}"
  --checkpoint "${CHECKPOINT_PATH}"
  --output "${OUTPUT_PATH}"
  --samples "${CALIBRATION_SAMPLES}"
)
if [[ "${OVERWRITE_CALIBRATION}" == "1" ]]; then
  CALIBRATE_ARGS+=(--overwrite)
fi
runtime_exec "${PYTHON_BIN}" -m asgcn_unet.cli calibrate "${CALIBRATE_ARGS[@]}"
)
_asgcn_entrypoint "$@"
fi
