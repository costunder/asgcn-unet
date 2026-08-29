#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  printf '%s\n' \
    "Usage: bash scripts/full.sh" \
    "" \
    "Runs, in order:" \
    "  full environment/data check and complete EventHDR/EventAid-R validation" \
    "  EventHDR ANN train (or RESUME_CHECKPOINT resume)" \
    "  all-sample EventHDR ANN-to-SNN calibration" \
    "  EventHDR and EventAid-R ANN evaluation+benchmark" \
    "  literal_eq15 and standard_if SNN evaluation+benchmark at T=4,8,16,32" \
    "" \
    "Important environment:" \
    "  RESUME_CHECKPOINT=PATH" \
    "  TRAIN_CONFIG / HDR_ANN_CONFIG / HDR_SNN_CONFIG" \
    "  AID_ANN_CONFIG / AID_SNN_CONFIG" \
    "  ANN_CHECKPOINT / SNN_CHECKPOINT" \
    "  PYTHON_BIN=PATH                         Default: <repo>/.venv/bin/python" \
    "  REQUIRE_CUDA=0|1                       Default: 1" \
    "  CALIBRATION_SAMPLES=all|N              Default: all" \
    "  SIMULATION_STEPS_LIST='4 8 16 32'" \
    "  BENCHMARK_WARMUP=N / BENCHMARK_STEPS=N" \
    "  OVERWRITE_CALIBRATION=0|1              Default: 0" \
    "  DRY_RUN=0|1                            Print the complete command schedule"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-constraints/py312.txt}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/hdr_train.json}"
HDR_ANN_CONFIG="${HDR_ANN_CONFIG:-configs/hdr_ann.json}"
HDR_SNN_CONFIG="${HDR_SNN_CONFIG:-configs/hdr_snn.json}"
AID_ANN_CONFIG="${AID_ANN_CONFIG:-configs/aid_ann.json}"
AID_SNN_CONFIG="${AID_SNN_CONFIG:-configs/aid_snn.json}"
ANN_CHECKPOINT="${ANN_CHECKPOINT:-runs/eventhdr_asgcn/best.pt}"
SNN_CHECKPOINT="${SNN_CHECKPOINT:-runs/eventhdr_asgcn/best_snn.pt}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-all}"
OVERWRITE_CALIBRATION="${OVERWRITE_CALIBRATION:-0}"
SIMULATION_STEPS_LIST="${SIMULATION_STEPS_LIST:-4 8 16 32}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-100}"
INSPECT_SAMPLES="${INSPECT_SAMPLES:-2}"
DRY_RUN="${DRY_RUN:-0}"

cd "${PROJECT_ROOT}"
for flag_name in REQUIRE_CUDA OVERWRITE_CALIBRATION DRY_RUN; do
  flag_value="${!flag_name}"
  if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
    echo "ERROR: ${flag_name} must be 0 or 1" >&2
    exit 2
  fi
done
if [[ "${DRY_RUN}" != "1" && ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
  echo "Run ./scripts/setup.sh first, or set PYTHON_BIN." >&2
  exit 1
fi
for required_path in \
  "${CONSTRAINTS_FILE}" \
  "${TRAIN_CONFIG}" \
  "${HDR_ANN_CONFIG}" \
  "${HDR_SNN_CONFIG}" \
  "${AID_ANN_CONFIG}" \
  "${AID_SNN_CONFIG}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "ERROR: required full-run file not found: ${required_path}" >&2
    exit 1
  fi
done
if [[ -n "${RESUME_CHECKPOINT}" && "${DRY_RUN}" != "1" && ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "ERROR: resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

read -r -a SIMULATION_STEPS <<< "${SIMULATION_STEPS_LIST}"
if [[ "${#SIMULATION_STEPS[@]}" -eq 0 ]]; then
  echo "ERROR: SIMULATION_STEPS_LIST must contain at least one positive integer" >&2
  exit 2
fi
for step in "${SIMULATION_STEPS[@]}"; do
  if [[ ! "${step}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: invalid simulation step '${step}' in SIMULATION_STEPS_LIST" >&2
    exit 2
  fi
done

run_cmd() {
  printf ' +'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

run_evaluation() {
  local config_path="$1"
  local checkpoint_path="$2"
  local mode="$3"
  local simulation_steps="$4"
  local dynamics="$5"
  run_cmd env \
    REQUIRE_CUDA="${REQUIRE_CUDA}" \
    VALIDATE_DATASET=0 \
    RUN_BENCHMARK=1 \
    BENCHMARK_WARMUP="${BENCHMARK_WARMUP}" \
    BENCHMARK_STEPS="${BENCHMARK_STEPS}" \
    INFERENCE_MODE="${mode}" \
    SIMULATION_STEPS="${simulation_steps}" \
    SNN_DYNAMICS="${dynamics}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${PROJECT_ROOT}/scripts/eval.sh" "${config_path}" "${checkpoint_path}"
}

echo "[1/5] Full environment and dataset inventory"
CHECK_ARGS=("${PYTHON_BIN}" scripts/check_env.py --require-full-data --lock "${CONSTRAINTS_FILE}")
if [[ "${REQUIRE_CUDA}" == "1" ]]; then
  CHECK_ARGS+=(--require-cuda)
fi
run_cmd "${CHECK_ARGS[@]}"

echo "[2/5] Decode and validate every EventHDR train/eval and EventAid-R sample"
# TRAIN_CONFIG inspection covers both EventHDR roots. Inspecting HDR_ANN_CONFIG
# here would decode the same 19 eval files a second time without adding coverage.
for config_path in "${TRAIN_CONFIG}" "${AID_ANN_CONFIG}"; do
  run_cmd "${PYTHON_BIN}" -m asgcn_recon.cli inspect \
    --config "${config_path}" \
    --samples "${INSPECT_SAMPLES}" \
    --validate-all
done

echo "[3/5] EventHDR ANN training"
run_cmd env \
  REQUIRE_CUDA="${REQUIRE_CUDA}" \
  VALIDATE_DATASET=0 \
  RESUME_CHECKPOINT="${RESUME_CHECKPOINT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${PROJECT_ROOT}/scripts/train.sh" "${TRAIN_CONFIG}"
if [[ "${DRY_RUN}" != "1" && ! -f "${ANN_CHECKPOINT}" ]]; then
  echo "ERROR: training completed without ANN checkpoint: ${ANN_CHECKPOINT}" >&2
  exit 1
fi

echo "[4/5] Full EventHDR ANN-to-SNN calibration"
run_cmd env \
  REQUIRE_CUDA="${REQUIRE_CUDA}" \
  VALIDATE_DATASET=0 \
  CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES}" \
  OVERWRITE_CALIBRATION="${OVERWRITE_CALIBRATION}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${PROJECT_ROOT}/scripts/calibrate.sh" \
    "${TRAIN_CONFIG}" "${ANN_CHECKPOINT}" "${SNN_CHECKPOINT}"
if [[ "${DRY_RUN}" != "1" && ! -f "${SNN_CHECKPOINT}" ]]; then
  echo "ERROR: calibration completed without SNN checkpoint: ${SNN_CHECKPOINT}" >&2
  exit 1
fi

echo "[5/5] Full EventHDR and EventAid-R evaluation and compute benchmark matrix"
for dataset_spec in \
  "${HDR_ANN_CONFIG}|${HDR_SNN_CONFIG}" \
  "${AID_ANN_CONFIG}|${AID_SNN_CONFIG}"; do
  IFS='|' read -r ann_config snn_config <<< "${dataset_spec}"
  run_evaluation "${ann_config}" "${ANN_CHECKPOINT}" ann 16 ""
  for dynamics in literal_eq15 standard_if; do
    for simulation_steps in "${SIMULATION_STEPS[@]}"; do
      run_evaluation \
        "${snn_config}" \
        "${SNN_CHECKPOINT}" \
        snn \
        "${simulation_steps}" \
        "${dynamics}"
    done
  done
done

echo "Full experiment matrix completed."
echo "ANN checkpoint: ${ANN_CHECKPOINT}"
echo "SNN checkpoint: ${SNN_CHECKPOINT}"
