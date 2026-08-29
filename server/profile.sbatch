#!/usr/bin/env bash
#SBATCH --job-name=asgcn-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Dependency example:
#   profile_id=$(sbatch --parsable server/profile.sbatch)
#   sbatch --dependency=afterok:${profile_id} server/train.sbatch

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
  || [[ ! -f "${PROJECT_ROOT}/scripts/run.sh" ]]; then
  echo "ERROR: PROJECT_ROOT is not a repository checkout." >&2
  echo "Submit from the repository root or export PROJECT_ROOT explicitly." >&2
  exit 1
fi
PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${CUDA_MODULE:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: CUDA_MODULE was set, but the module command is unavailable" >&2
    exit 1
  fi
  module load "${CUDA_MODULE}"
fi

export TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train.json}"
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
export REQUIRE_CUDA=1
export PROFILE_OUTPUT="${PROFILE_OUTPUT:-runs/profile.json}"
export PROFILE_SAMPLES="${PROFILE_SAMPLES:-3}"
export PROFILE_TOP_DENSITY="${PROFILE_TOP_DENSITY:-10}"
export INCLUDE_PRIVATE_HOST_PROVENANCE="${INCLUDE_PRIVATE_HOST_PROVENANCE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTHONUNBUFFERED=1
# Let srun inherit this already filtered/prepared job environment without
# re-importing the submit shell's environment.
export SLURM_EXPORT_ENV=ALL

if [[ "${INCLUDE_PRIVATE_HOST_PROVENANCE}" != "0" \
  && "${INCLUDE_PRIVATE_HOST_PROVENANCE}" != "1" ]]; then
  echo "ERROR: INCLUDE_PRIVATE_HOST_PROVENANCE must be 0 or 1" >&2
  exit 2
fi
if [[ "${INCLUDE_PRIVATE_HOST_PROVENANCE}" == "1" ]]; then
  echo "Host: $(hostname)"
  echo "Job: ${SLURM_JOB_ID:-local}"
  echo "Project root: ${PROJECT_ROOT}"
  echo "Config: ${TRAIN_CONFIG}"
  echo "Profile output: ${PROFILE_OUTPUT}"
else
  echo "Host/job provenance: omitted from public log"
  echo "Config: ${TRAIN_CONFIG##*/}"
  echo "Profile output: ${PROFILE_OUTPUT##*/}"
fi
nvidia-smi || true
srun bash "${PROJECT_ROOT}/scripts/run.sh" profile
