#!/usr/bin/env bash
#SBATCH --job-name=asgcn-cal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Dependency example:
#   profile_id=$(sbatch --parsable --export=CONDA_PREFIX="${CONDA_PREFIX}" server/profile.sbatch)
#   train_id=$(sbatch --parsable --dependency=afterok:${profile_id} --export=CONDA_PREFIX="${CONDA_PREFIX}" server/train.sbatch)
#   cal_id=$(sbatch --parsable --dependency=afterok:${train_id} --export=CONDA_PREFIX="${CONDA_PREFIX}" server/calibrate.sbatch)
#   sbatch --dependency=afterok:${cal_id} --export=CONDA_PREFIX="${CONDA_PREFIX}",CONFIG_PATH=configs/hdr-fast.json,\
# CHECKPOINT_PATH=runs/fast/best_snn.pt,INFERENCE_MODE=snn server/eval.sbatch

# Entry scripts must not change an interactive shell through accidental sourcing.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' "Source ignored: run this entrypoint with bash or its scheduler; the current shell is unchanged." >&2
else
_asgcn_entrypoint() (
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  elif [[ -z "${SLURM_JOB_ID:-}" ]]; then
    PROJECT_ROOT="${SCRIPT_DIR}/.."
  fi
fi
if [[ -z "${PROJECT_ROOT:-}" ]] \
  || [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]] \
  || [[ ! -f "${PROJECT_ROOT}/scripts/calibrate.sh" ]]; then
  echo "ERROR: PROJECT_ROOT is not a repository checkout." >&2
  echo "Run sbatch from the repository root or use:" >&2
  echo "  sbatch --export=PROJECT_ROOT=/path/to/repo,CONDA_PREFIX=/path/to/conda/env server/calibrate.sbatch" >&2
  return 1
fi
PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${CUDA_MODULE:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: CUDA_MODULE was set, but the module command is unavailable" >&2
    return 1
  fi
  module load "${CUDA_MODULE}"
fi

export EXPERIMENT="${EXPERIMENT:-fast}"
case "${EXPERIMENT}" in
  single) DEFAULT_CONFIG_PATH=configs/train.json; DEFAULT_RUN_DIR=runs/train ;;
  batch) DEFAULT_CONFIG_PATH=configs/batch.json; DEFAULT_RUN_DIR=runs/batch ;;
  fast) DEFAULT_CONFIG_PATH=configs/fast.json; DEFAULT_RUN_DIR=runs/fast ;;
  *) echo "ERROR: EXPERIMENT must be single, batch or fast" >&2; return 2 ;;
esac
export CONFIG_PATH="${CONFIG_PATH:-${DEFAULT_CONFIG_PATH}}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DEFAULT_RUN_DIR}/best.pt}"
export OUTPUT_PATH="${OUTPUT_PATH:-${DEFAULT_RUN_DIR}/best_snn.pt}"
export CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-all}"
export OVERWRITE_CALIBRATION="${OVERWRITE_CALIBRATION:-0}"
export VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
export INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
export INSPECT_VALIDATE_ALL="${INSPECT_VALIDATE_ALL:-0}"
export INCLUDE_PRIVATE_HOST_PROVENANCE="${INCLUDE_PRIVATE_HOST_PROVENANCE:-0}"
# shellcheck source=scripts/runtime.sh
source "${PROJECT_ROOT}/scripts/runtime.sh"
select_conda_python
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTHONUNBUFFERED=1
# Let srun inherit this already filtered/prepared job environment without
# re-importing the submit shell's environment.
export SLURM_EXPORT_ENV=ALL

if [[ "${INCLUDE_PRIVATE_HOST_PROVENANCE}" != "0" \
  && "${INCLUDE_PRIVATE_HOST_PROVENANCE}" != "1" ]]; then
  echo "ERROR: INCLUDE_PRIVATE_HOST_PROVENANCE must be 0 or 1" >&2
  return 2
fi
if [[ "${INCLUDE_PRIVATE_HOST_PROVENANCE}" == "1" ]]; then
  echo "Host: $(hostname)"
  echo "Job: ${SLURM_JOB_ID:-local}"
  echo "Project root: ${PROJECT_ROOT}"
  echo "Config: ${CONFIG_PATH}"
  echo "ANN checkpoint: ${CHECKPOINT_PATH}"
  echo "SNN checkpoint: ${OUTPUT_PATH}"
else
  echo "Host/job provenance: omitted from public log"
  echo "Config: ${CONFIG_PATH##*/}"
  echo "ANN checkpoint: ${CHECKPOINT_PATH##*/}"
  echo "SNN checkpoint: ${OUTPUT_PATH##*/}"
fi
echo "Calibration samples: ${CALIBRATION_SAMPLES}"
# The child preflight reports only devices from the verified allocation.

runtime_command srun bash "${PROJECT_ROOT}/scripts/calibrate.sh" \
  "${CONFIG_PATH}" "${CHECKPOINT_PATH}" "${OUTPUT_PATH}"
)
_asgcn_entrypoint "$@"
fi
