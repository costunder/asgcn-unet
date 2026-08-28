#!/usr/bin/env bash
#SBATCH --job-name=asgcn-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${CUDA_MODULE:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: CUDA_MODULE was set, but the module command is unavailable" >&2
    exit 1
  fi
  module load "${CUDA_MODULE}"
fi

export CONFIG_PATH="${CONFIG_PATH:-configs/hdr_ann.json}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/eventhdr_asgcn/best.pt}"
export INFERENCE_MODE="${INFERENCE_MODE:-ann}"
export SIMULATION_STEPS="${SIMULATION_STEPS:-16}"
export RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
export BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"
export BENCHMARK_STEPS="${BENCHMARK_STEPS:-100}"
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
nvidia-smi || true

srun bash "${PROJECT_ROOT}/scripts/eval.sh" \
  "${CONFIG_PATH}" "${CHECKPOINT_PATH}"
