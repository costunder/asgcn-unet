#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${CONFIG_PATH:-configs/eventhdr_train.json}}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

cd "${PROJECT_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
  echo "Run ./scripts/setup_server.sh first, or set PYTHON_BIN." >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: training config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${REQUIRE_CUDA}" <<'PY'
import sys
import torch

required = sys.argv[1] == "1"
available = torch.cuda.is_available()
print(f"PyTorch {torch.__version__}; CUDA runtime={torch.version.cuda}; available={available}")
if available:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif required:
    raise SystemExit("CUDA GPU is required. Set REQUIRE_CUDA=0 only for a deliberate CPU run.")
PY

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

if [[ "${VALIDATE_DATASET}" == "1" ]]; then
  "${PYTHON_BIN}" -m asgcn_recon.cli inspect \
    --config "${CONFIG_PATH}" --samples "${INSPECT_SAMPLES}"
fi

echo "Starting EventHDR training with ${CONFIG_PATH}"
TRAIN_ARGS=(--config "${CONFIG_PATH}")
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "ERROR: resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
    exit 1
  fi
  echo "Resuming from ${RESUME_CHECKPOINT}"
  TRAIN_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi
exec "${PYTHON_BIN}" -m asgcn_recon.cli train "${TRAIN_ARGS[@]}"
