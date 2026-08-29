#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  printf '%s\n' \
    "Usage: bash scripts/calibrate.sh [CONFIG [ANN_CHECKPOINT [SNN_CHECKPOINT]]]" \
    "" \
    "Environment:" \
    "  CALIBRATION_SAMPLES=all|N   Default: all EventHDR calibration samples" \
    "  OVERWRITE_CALIBRATION=0|1  Default: 0; protect an existing output" \
    "  VALIDATE_DATASET=0|1       Default: 1" \
    "  INSPECT_VALIDATE_ALL=0|1   Default: 0" \
    "  INSPECT_SAMPLES=N          Default: 1" \
    "  REQUIRE_CUDA=0|1           Default: 1" \
    "  PYTHON_BIN=PATH            Default: <repo>/.venv/bin/python"
  exit 0
fi

CONFIG_PATH="${1:-${CONFIG_PATH:-configs/hdr_train.json}}"
CHECKPOINT_PATH="${2:-${CHECKPOINT_PATH:-runs/eventhdr_asgcn/best.pt}}"
OUTPUT_PATH="${3:-${OUTPUT_PATH:-runs/eventhdr_asgcn/best_snn.pt}}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-all}"
OVERWRITE_CALIBRATION="${OVERWRITE_CALIBRATION:-0}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
INSPECT_VALIDATE_ALL="${INSPECT_VALIDATE_ALL:-0}"

cd "${PROJECT_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
  echo "Run ./scripts/setup.sh first, or set PYTHON_BIN." >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: calibration config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: ANN checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
for flag_name in REQUIRE_CUDA VALIDATE_DATASET INSPECT_VALIDATE_ALL OVERWRITE_CALIBRATION; do
  flag_value="${!flag_name}"
  if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
    echo "ERROR: ${flag_name} must be 0 or 1" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" - "${CHECKPOINT_PATH}" "${OUTPUT_PATH}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).expanduser().resolve()
output = Path(sys.argv[2]).expanduser().resolve()
if source == output:
    raise SystemExit("ANN input and calibrated SNN output must be different files")
PY

if [[ -e "${OUTPUT_PATH}" || -L "${OUTPUT_PATH}" ]]; then
  if [[ "${OVERWRITE_CALIBRATION}" != "1" ]]; then
    echo "ERROR: calibrated output already exists: ${OUTPUT_PATH}" >&2
    echo "Set OVERWRITE_CALIBRATION=1 only when replacing it is intentional." >&2
    exit 1
  fi
  if [[ -d "${OUTPUT_PATH}" ]]; then
    echo "ERROR: calibrated output path is a directory: ${OUTPUT_PATH}" >&2
    exit 1
  fi
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
  INSPECT_ARGS=(
    --config "${CONFIG_PATH}"
    --samples "${INSPECT_SAMPLES}"
  )
  if [[ "${INSPECT_VALIDATE_ALL}" == "1" ]]; then
    INSPECT_ARGS+=(--validate-all)
  fi
  "${PYTHON_BIN}" -m asgcn_recon.cli inspect "${INSPECT_ARGS[@]}"
fi

if [[ "${CALIBRATION_SAMPLES}" != "all" && ! "${CALIBRATION_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CALIBRATION_SAMPLES must be 'all' or a positive integer" >&2
  exit 2
fi

echo "Calibrating ${CHECKPOINT_PATH} with ${CALIBRATION_SAMPLES} EventHDR samples"
CALIBRATE_ARGS=(
  --config "${CONFIG_PATH}"
  --checkpoint "${CHECKPOINT_PATH}"
  --output "${OUTPUT_PATH}"
  --samples "${CALIBRATION_SAMPLES}"
)
if [[ "${OVERWRITE_CALIBRATION}" == "1" ]]; then
  CALIBRATE_ARGS+=(--overwrite)
fi
exec "${PYTHON_BIN}" -m asgcn_recon.cli calibrate "${CALIBRATE_ARGS[@]}"
