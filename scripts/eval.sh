#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${CONFIG_PATH:-configs/hdr_ann.json}}"
CHECKPOINT_PATH="${2:-${CHECKPOINT_PATH:-runs/eventhdr_asgcn/best.pt}}"
INFERENCE_MODE="${INFERENCE_MODE:-ann}"
SIMULATION_STEPS="${SIMULATION_STEPS:-16}"
SNN_DYNAMICS="${SNN_DYNAMICS:-}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-100}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
VALIDATE_DATASET="${VALIDATE_DATASET:-1}"
INSPECT_SAMPLES="${INSPECT_SAMPLES:-1}"

cd "${PROJECT_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
  echo "Run ./scripts/setup.sh first, or set PYTHON_BIN." >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: evaluation config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ "${INFERENCE_MODE}" != "ann" && "${INFERENCE_MODE}" != "snn" ]]; then
  echo "ERROR: INFERENCE_MODE must be ann or snn" >&2
  exit 2
fi
if [[ -n "${SNN_DYNAMICS}" ]]; then
  if [[ "${INFERENCE_MODE}" != "snn" ]]; then
    echo "ERROR: SNN_DYNAMICS is only valid when INFERENCE_MODE=snn" >&2
    exit 2
  fi
  if [[ "${SNN_DYNAMICS}" != "literal_eq15" && "${SNN_DYNAMICS}" != "standard_if" ]]; then
    echo "ERROR: SNN_DYNAMICS must be literal_eq15 or standard_if" >&2
    exit 2
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

DYNAMICS_ARGS=()
if [[ -n "${SNN_DYNAMICS}" ]]; then
  DYNAMICS_ARGS=(--snn-dynamics "${SNN_DYNAMICS}")
fi

if [[ "${VALIDATE_DATASET}" == "1" ]]; then
  "${PYTHON_BIN}" -m asgcn_recon.cli inspect \
    --config "${CONFIG_PATH}" --samples "${INSPECT_SAMPLES}"
fi

echo "Evaluating ${CHECKPOINT_PATH} on ${CONFIG_PATH} (${INFERENCE_MODE})"
"${PYTHON_BIN}" -m asgcn_recon.cli evaluate \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --inference-mode "${INFERENCE_MODE}" \
  --simulation-steps "${SIMULATION_STEPS}" \
  "${DYNAMICS_ARGS[@]}"

if [[ "${RUN_BENCHMARK}" == "1" ]]; then
  echo "Running latency benchmark"
  "${PYTHON_BIN}" -m asgcn_recon.cli benchmark \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --warmup "${BENCHMARK_WARMUP}" \
    --steps "${BENCHMARK_STEPS}" \
    --inference-mode "${INFERENCE_MODE}" \
    --simulation-steps "${SIMULATION_STEPS}" \
    "${DYNAMICS_ARGS[@]}"
fi
