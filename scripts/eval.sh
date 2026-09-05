#!/usr/bin/env bash
# Entry scripts must not change an interactive shell through accidental sourcing.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' "Source ignored: run this entrypoint with bash or its scheduler; the current shell is unchanged." >&2
else
_asgcn_entrypoint() (
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT="${EXPERIMENT:-fast}"
case "${EXPERIMENT}" in
  single) DEFAULT_CONFIG_PATH=configs/hdr.json; DEFAULT_RUN_DIR=runs/train ;;
  batch) DEFAULT_CONFIG_PATH=configs/hdr.json; DEFAULT_RUN_DIR=runs/batch ;;
  fast) DEFAULT_CONFIG_PATH=configs/hdr-fast.json; DEFAULT_RUN_DIR=runs/fast ;;
  *) echo "ERROR: EXPERIMENT must be single, batch or fast" >&2; return 2 ;;
esac
CONFIG_PATH="${1:-${CONFIG_PATH:-${DEFAULT_CONFIG_PATH}}}"
CHECKPOINT_PATH="${2:-${CHECKPOINT_PATH:-${DEFAULT_RUN_DIR}/best.pt}}"
INFERENCE_MODE="${INFERENCE_MODE:-ann}"
SIMULATION_STEPS="${SIMULATION_STEPS:-16}"
SNN_DYNAMICS="${SNN_DYNAMICS:-}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
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
  return 2
fi
# shellcheck source=scripts/runtime.sh
source "${PROJECT_ROOT}/scripts/runtime.sh"
select_conda_python
EVAL_MAX_GRAPH_EDGES="$(evaluation_graph_edge_guard "${CONFIG_PATH}")"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: evaluation config not found: $(path_log_label "${CONFIG_PATH}")" >&2
  return 1
fi
if [[ "${DRY_RUN}" != "1" && ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: $(path_log_label "${CHECKPOINT_PATH}")" >&2
  return 1
fi
if [[ "${VALIDATE_DATASET}" != "0" && "${VALIDATE_DATASET}" != "1" ]]; then
  echo "ERROR: VALIDATE_DATASET must be 0 or 1" >&2
  return 2
fi
if [[ "${RUN_EVALUATION}" != "0" && "${RUN_EVALUATION}" != "1" ]]; then
  echo "ERROR: RUN_EVALUATION must be 0 or 1" >&2
  return 2
fi
if [[ "${RUN_BENCHMARK}" != "0" && "${RUN_BENCHMARK}" != "1" ]]; then
  echo "ERROR: RUN_BENCHMARK must be 0 or 1" >&2
  return 2
fi
if [[ "${RUN_EVALUATION}" == "0" && "${RUN_BENCHMARK}" == "0" ]]; then
  echo "ERROR: RUN_EVALUATION and RUN_BENCHMARK cannot both be 0" >&2
  return 2
fi
if [[ "${INSPECT_VALIDATE_ALL}" != "0" && "${INSPECT_VALIDATE_ALL}" != "1" ]]; then
  echo "ERROR: INSPECT_VALIDATE_ALL must be 0 or 1" >&2
  return 2
fi
if [[ "${INFERENCE_MODE}" != "ann" && "${INFERENCE_MODE}" != "snn" ]]; then
  echo "ERROR: INFERENCE_MODE must be ann or snn" >&2
  return 2
fi
if [[ -n "${EVAL_MAX_GRAPH_EDGES}" && ! "${EVAL_MAX_GRAPH_EDGES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EVAL_MAX_GRAPH_EDGES must be a positive integer" >&2
  return 2
fi
if [[ -n "${SNN_DYNAMICS}" ]]; then
  if [[ "${INFERENCE_MODE}" != "snn" ]]; then
    echo "ERROR: SNN_DYNAMICS is only valid when INFERENCE_MODE=snn" >&2
    return 2
  fi
  if [[ "${SNN_DYNAMICS}" != "literal_eq15" && "${SNN_DYNAMICS}" != "standard_if" ]]; then
    echo "ERROR: SNN_DYNAMICS must be literal_eq15 or standard_if" >&2
    return 2
  fi
fi

check_runtime_profile
# The shared preflight reports the runtime only after allocation verification.

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

if [[ "${RUN_EVALUATION}" == "1" ]]; then
  echo "Evaluating $(path_log_label "${CHECKPOINT_PATH}") on $(path_log_label "${CONFIG_PATH}") (${INFERENCE_MODE})"
  runtime_command "${PYTHON_BIN}" -m asgcn_unet.cli evaluate \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --inference-mode "${INFERENCE_MODE}" \
    --simulation-steps "${SIMULATION_STEPS}" \
    "${OUTPUT_ARGS[@]}" \
    "${DYNAMICS_ARGS[@]}" \
    "${GRAPH_EDGE_ARGS[@]}"
else
  echo "Skipping completed quality evaluation (${INFERENCE_MODE})"
fi

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
)
_asgcn_entrypoint "$@"
fi
