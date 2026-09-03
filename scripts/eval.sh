#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${CONFIG_PATH:-configs/hdr.json}}"
CHECKPOINT_PATH="${2:-${CHECKPOINT_PATH:-runs/train/best.pt}}"
INFERENCE_MODE="${INFERENCE_MODE:-ann}"
SIMULATION_STEPS="${SIMULATION_STEPS:-16}"
SNN_DYNAMICS="${SNN_DYNAMICS:-}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-100}"
EVAL_MAX_GRAPH_EDGES="${EVAL_MAX_GRAPH_EDGES:-}"
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
  exit 2
fi
# shellcheck source=scripts/runtime.sh
source "${PROJECT_ROOT}/scripts/runtime.sh"
select_conda_python

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: evaluation config not found: $(path_log_label "${CONFIG_PATH}")" >&2
  exit 1
fi
if [[ "${DRY_RUN}" != "1" && ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: $(path_log_label "${CHECKPOINT_PATH}")" >&2
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
if [[ "${INFERENCE_MODE}" != "ann" && "${INFERENCE_MODE}" != "snn" ]]; then
  echo "ERROR: INFERENCE_MODE must be ann or snn" >&2
  exit 2
fi
if [[ -n "${EVAL_MAX_GRAPH_EDGES}" && ! "${EVAL_MAX_GRAPH_EDGES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EVAL_MAX_GRAPH_EDGES must be a positive integer" >&2
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

check_runtime_profile
runtime_command "${PYTHON_BIN}" - "${REQUIRE_CUDA}" <<'PY'
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
OUTPUT_ARGS=()
GRAPH_EDGE_ARGS=()
if [[ -n "${EVAL_OUTPUT_DIR:-}" ]]; then
  OUTPUT_ARGS=(--output-dir "${EVAL_OUTPUT_DIR}")
fi
if [[ -n "${SNN_DYNAMICS}" ]]; then
  DYNAMICS_ARGS=(--snn-dynamics "${SNN_DYNAMICS}")
fi
if [[ -n "${EVAL_MAX_GRAPH_EDGES}" ]]; then
  GRAPH_EDGE_ARGS=(--max-graph-edges-override "${EVAL_MAX_GRAPH_EDGES}")
fi

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

echo "Evaluating $(path_log_label "${CHECKPOINT_PATH}") on $(path_log_label "${CONFIG_PATH}") (${INFERENCE_MODE})"
runtime_command "${PYTHON_BIN}" -m asgcn_unet.cli evaluate \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --inference-mode "${INFERENCE_MODE}" \
  --simulation-steps "${SIMULATION_STEPS}" \
  "${OUTPUT_ARGS[@]}" \
  "${DYNAMICS_ARGS[@]}" \
  "${GRAPH_EDGE_ARGS[@]}"

if [[ "${RUN_BENCHMARK}" == "1" ]]; then
  echo "Running latency benchmark"
  runtime_command "${PYTHON_BIN}" -m asgcn_unet.cli benchmark \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --warmup "${BENCHMARK_WARMUP}" \
    --steps "${BENCHMARK_STEPS}" \
    --inference-mode "${INFERENCE_MODE}" \
    --simulation-steps "${SIMULATION_STEPS}" \
    "${OUTPUT_ARGS[@]}" \
    "${DYNAMICS_ARGS[@]}" \
    "${GRAPH_EDGE_ARGS[@]}"
fi
