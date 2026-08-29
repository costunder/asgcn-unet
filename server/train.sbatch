#!/usr/bin/env bash
#SBATCH --job-name=asgcn-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

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
  || [[ ! -f "${PROJECT_ROOT}/scripts/train.sh" ]]; then
  echo "ERROR: PROJECT_ROOT is not a repository checkout." >&2
  echo "Run sbatch from the repository root or use:" >&2
  echo "  sbatch --export=PROJECT_ROOT=/absolute/path/to/repo server/train.sbatch" >&2
  exit 1
fi
PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"
cd "${PROJECT_ROOT}"

# Cluster-specific modules remain opt-in:
#   sbatch --export=CUDA_MODULE=cuda/12.4 server/train.sbatch
if [[ -n "${CUDA_MODULE:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: CUDA_MODULE was set, but the module command is unavailable" >&2
    exit 1
  fi
  module load "${CUDA_MODULE}"
fi

export CONFIG_PATH="${CONFIG_PATH:-configs/train.json}"
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
export INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
export INSPECT_VALIDATE_ALL="${INSPECT_VALIDATE_ALL:-0}"
export PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-runs/profile.json}"
export ALLOW_UNVERIFIED_PREFLIGHT="${ALLOW_UNVERIFIED_PREFLIGHT:-0}"
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
  echo "Config: ${CONFIG_PATH}"
  if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    echo "Resume: ${RESUME_CHECKPOINT}"
  fi
else
  echo "Host/job provenance: omitted from public log"
  echo "Config: ${CONFIG_PATH##*/}"
  if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    echo "Resume: ${RESUME_CHECKPOINT##*/}"
  fi
fi
nvidia-smi || true

srun bash "${PROJECT_ROOT}/scripts/train.sh" "${CONFIG_PATH}"
