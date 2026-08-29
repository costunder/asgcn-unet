#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${CONFIG_PATH:-configs/train.json}}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"
INSPECT_VALIDATE_ALL="${INSPECT_VALIDATE_ALL:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-runs/profile.json}"
ALLOW_UNVERIFIED_PREFLIGHT="${ALLOW_UNVERIFIED_PREFLIGHT:-0}"
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
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable: $(path_log_label "${PYTHON_BIN}")" >&2
  echo "Run ./scripts/setup.sh first, or set PYTHON_BIN." >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: training config not found: $(path_log_label "${CONFIG_PATH}")" >&2
  exit 1
fi
if [[ "${VALIDATE_DATASET}" != "0" && "${VALIDATE_DATASET}" != "1" ]]; then
  echo "ERROR: VALIDATE_DATASET must be 0 or 1" >&2
  exit 2
fi
if [[ "${INSPECT_VALIDATE_ALL}" != "0" && "${INSPECT_VALIDATE_ALL}" != "1" ]]; then
  echo "ERROR: INSPECT_VALIDATE_ALL must be 0 or 1" >&2
  exit 2
fi
if [[ "${ALLOW_UNVERIFIED_PREFLIGHT}" != "0" \
  && "${ALLOW_UNVERIFIED_PREFLIGHT}" != "1" ]]; then
  echo "ERROR: ALLOW_UNVERIFIED_PREFLIGHT must be 0 or 1" >&2
  exit 2
fi
if [[ "${ALLOW_UNVERIFIED_PREFLIGHT}" != "1" && ! -f "${PREFLIGHT_REPORT}" ]]; then
  echo "ERROR: passed CUDA preflight report not found: $(path_log_label "${PREFLIGHT_REPORT}")" >&2
  echo "Run bash scripts/run.sh profile first." >&2
  exit 1
fi
if [[ "${ALLOW_UNVERIFIED_PREFLIGHT}" == "1" ]]; then
  echo "WARNING: bypassing CUDA preflight for a non-reporting run." >&2
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
  "${PYTHON_BIN}" -m asgcn_unet.cli inspect "${INSPECT_ARGS[@]}"
fi

echo "Starting EventHDR training with $(path_log_label "${CONFIG_PATH}")"
TRAIN_ARGS=(
  --config "${CONFIG_PATH}"
  --preflight-report "${PREFLIGHT_REPORT}"
)
if [[ "${ALLOW_UNVERIFIED_PREFLIGHT}" == "1" ]]; then
  TRAIN_ARGS+=(--allow-unverified-preflight)
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "ERROR: resume checkpoint not found: $(path_log_label "${RESUME_CHECKPOINT}")" >&2
    exit 1
  fi
  echo "Resuming from checkpoint: $(path_log_label "${RESUME_CHECKPOINT}")"
  TRAIN_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi
exec "${PYTHON_BIN}" -m asgcn_unet.cli train "${TRAIN_ARGS[@]}"
