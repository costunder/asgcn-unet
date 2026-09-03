#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/run.sh [check|profile|train|calibrate|eval|eval-hdr|eval-aid|all]

Stages:
  check       Check CUDA/dependencies/full data and decode every selected sample
  profile     Scan all train graphs on CUDA; probe dense and first/empty/sparse samples
  train       Train EventHDR ANN, or resume with RESUME_CHECKPOINT
  calibrate   Convert runs/train/best.pt to runs/train/best_snn.pt
  eval        Run the complete EventHDR/EventAid-R ANN+SNN evaluation matrix
  eval-hdr    Run only the EventHDR ANN+SNN evaluation matrix
  eval-aid    Run only the EventAid-R ANN+SNN evaluation matrix
  all         Run check, profile, train, calibrate and eval in order (default)

Important environment:
  EXPERIMENT=single|batch|fast           Default: single; fast is B16 + Triton
  RESUME_CHECKPOINT=PATH
  MAX_HOURS=N / CHECKPOINT_SECONDS=N     Optional safe pause / checkpoint interval
  TRAIN_CONFIG / HDR_CONFIG / AID_CONFIG
  ANN_CHECKPOINT / SNN_CHECKPOINT
  PYTHON_BIN=PATH                         Default: CONDA_PREFIX/bin/python
  CONDA_PREFIX=PATH                      Selected Conda environment (no nested venv)
  RUNTIME_PROFILE=PATH                   Default: constraints/server.json
  REQUIRE_CUDA=0|1                       Default: 1
  CALIBRATION_SAMPLES=all|N              Default: all; partial N cannot be reporting
  SIMULATION_STEPS_LIST='4 8 16 32'
  BENCHMARK_WARMUP=N / BENCHMARK_STEPS=N
  EVAL_MAX_GRAPH_EDGES=N                 Eval-only guard raise; recorded in outputs
  PROFILE_SAMPLES=N / PROFILE_TOP_DENSITY=N
  PROFILE_OUTPUT=PATH                    Default: runs/profile.json
  PROFILE_RESUME=0|1                     Resume a matching saved scan; default: 0
  PROFILE_REUSE_REPORT=PATH              Reuse topology only; rerun GPU probes
  PROFILE_CPU_THREADS=N                 CPU helpers for CUDA scan; default: 4
  RESTART_TRAIN=0|1                      Archive metadata-only failed run; default: 0
  ALLOW_UNVERIFIED_PREFLIGHT=0|1         Non-reporting train bypass; default: 0
  OVERWRITE_CALIBRATION=0|1              Default: 0
  DRY_RUN=0|1                            Print commands without executing them
  INCLUDE_PRIVATE_HOST_PROVENANCE=0|1    Default: 0; exact host paths are private

Existing artifacts, including PROFILE_OUTPUT, are never silently skipped. Select
a stage explicitly when recovering a partial run; wrappers fail rather than
overwrite protected output.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

STAGE="${1:-all}"
case "${STAGE}" in
  check|profile|train|calibrate|eval|eval-hdr|eval-aid|all) ;;
  *)
    echo "ERROR: unknown stage '${STAGE}'" >&2
    usage >&2
    exit 2
    ;;
esac
if [[ "$#" -gt 1 ]]; then
  echo "ERROR: run.sh accepts at most one stage argument" >&2
  usage >&2
  exit 2
fi

EXPERIMENT="${EXPERIMENT:-single}"
DEFAULT_HDR_CONFIG=configs/hdr.json
DEFAULT_AID_CONFIG=configs/aid.json
case "${EXPERIMENT}" in
  single)
    DEFAULT_TRAIN_CONFIG=configs/train.json
    DEFAULT_TRAIN_RUN=runs/train
    DEFAULT_PROFILE_OUTPUT=runs/profile.json
    DEFAULT_STATUS_DIR=runs/status
    DEFAULT_EVAL_ROOT=""
    ;;
  batch)
    DEFAULT_TRAIN_CONFIG=configs/batch.json
    DEFAULT_TRAIN_RUN=runs/batch
    DEFAULT_PROFILE_OUTPUT=runs/batch-profile.json
    DEFAULT_STATUS_DIR=runs/batch-status
    DEFAULT_EVAL_ROOT=runs/batch/eval
    ;;
  fast)
    DEFAULT_TRAIN_CONFIG=configs/fast.json
    DEFAULT_TRAIN_RUN=runs/fast
    DEFAULT_PROFILE_OUTPUT=runs/fast-profile.json
    DEFAULT_STATUS_DIR=runs/fast-status
    DEFAULT_EVAL_ROOT=runs/fast/eval
    DEFAULT_HDR_CONFIG=configs/hdr-fast.json
    DEFAULT_AID_CONFIG=configs/aid-fast.json
    ;;
  *) echo "ERROR: EXPERIMENT must be single, batch or fast" >&2; exit 2 ;;
esac
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-constraints/py312.txt}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${DEFAULT_TRAIN_CONFIG}}"
HDR_CONFIG="${HDR_CONFIG:-${DEFAULT_HDR_CONFIG}}"
AID_CONFIG="${AID_CONFIG:-${DEFAULT_AID_CONFIG}}"
ANN_CHECKPOINT="${ANN_CHECKPOINT:-${DEFAULT_TRAIN_RUN}/best.pt}"
SNN_CHECKPOINT="${SNN_CHECKPOINT:-${DEFAULT_TRAIN_RUN}/best_snn.pt}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-all}"
OVERWRITE_CALIBRATION="${OVERWRITE_CALIBRATION:-0}"
SIMULATION_STEPS_LIST="${SIMULATION_STEPS_LIST:-4 8 16 32}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-100}"
EVAL_MAX_GRAPH_EDGES="${EVAL_MAX_GRAPH_EDGES:-}"
PROFILE_SAMPLES="${PROFILE_SAMPLES:-3}"
PROFILE_TOP_DENSITY="${PROFILE_TOP_DENSITY:-10}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-${DEFAULT_PROFILE_OUTPUT}}"
PROFILE_RESUME="${PROFILE_RESUME:-0}"
PROFILE_REUSE_REPORT="${PROFILE_REUSE_REPORT:-}"
PROFILE_CPU_THREADS="${PROFILE_CPU_THREADS:-4}"
RESTART_TRAIN="${RESTART_TRAIN:-0}"
ALLOW_UNVERIFIED_PREFLIGHT="${ALLOW_UNVERIFIED_PREFLIGHT:-0}"
INSPECT_SAMPLES="${INSPECT_SAMPLES:-2}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_DIR="${STATUS_DIR:-${DEFAULT_STATUS_DIR}}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${DEFAULT_EVAL_ROOT}}"
export INCLUDE_PRIVATE_HOST_PROVENANCE="${INCLUDE_PRIVATE_HOST_PROVENANCE:-0}"

cd "${PROJECT_ROOT}"

for flag_name in \
  REQUIRE_CUDA \
  OVERWRITE_CALIBRATION \
  DRY_RUN \
  ALLOW_UNVERIFIED_PREFLIGHT \
  PROFILE_RESUME \
  RESTART_TRAIN \
  INCLUDE_PRIVATE_HOST_PROVENANCE; do
  flag_value="${!flag_name}"
  if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
    echo "ERROR: ${flag_name} must be 0 or 1" >&2
    exit 2
  fi
done

path_log_label() {
  local path="$1"
  if [[ "${INCLUDE_PRIVATE_HOST_PROVENANCE}" == "1" ]]; then
    printf '%s' "${path}"
  else
    printf '%s' "${path##*/}"
  fi
}
if [[ "${DRY_RUN}" != "1" \
  && ( "${STAGE}" == "profile" || "${STAGE}" == "all" ) \
  && "${REQUIRE_CUDA}" != "1" ]]; then
  echo "ERROR: profile/all is a CUDA reporting gate and requires REQUIRE_CUDA=1." >&2
  echo "For a deliberate non-reporting train only, select train and set" >&2
  echo "ALLOW_UNVERIFIED_PREFLIGHT=1." >&2
  exit 2
fi
if [[ "${STAGE}" == "all" && "${ALLOW_UNVERIFIED_PREFLIGHT}" == "1" ]]; then
  echo "ERROR: all is a reporting pipeline and cannot bypass its preflight gate." >&2
  exit 2
fi

# shellcheck source=scripts/runtime.sh
source "${PROJECT_ROOT}/scripts/runtime.sh"
select_conda_python
for required_path in \
  "${CONSTRAINTS_FILE}" \
  "${TRAIN_CONFIG}" \
  "${HDR_CONFIG}" \
  "${AID_CONFIG}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "ERROR: required run file not found: $(path_log_label "${required_path}")" >&2
    exit 1
  fi
done
if [[ -n "${RESUME_CHECKPOINT}" && "${DRY_RUN}" != "1" \
  && ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "ERROR: resume checkpoint not found: $(path_log_label "${RESUME_CHECKPOINT}")" >&2
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
for profile_value in "${PROFILE_SAMPLES}" "${PROFILE_TOP_DENSITY}"; do
  if [[ ! "${profile_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: profile sample counts must be positive integers" >&2
    exit 2
  fi
done
if [[ ! "${PROFILE_CPU_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PROFILE_CPU_THREADS must be a positive integer" >&2
  exit 2
fi
if ((PROFILE_TOP_DENSITY < PROFILE_SAMPLES)); then
  echo "ERROR: PROFILE_TOP_DENSITY must be >= PROFILE_SAMPLES" >&2
  exit 2
fi

run_cmd() {
  printf ' +'
  local argument
  local display
  for argument in "$@"; do
    if [[ "${INCLUDE_PRIVATE_HOST_PROVENANCE}" == "1" ]]; then
      display="${argument}"
    else
      display="${argument//${PROJECT_ROOT}/\$PROJECT_ROOT}"
      if [[ "${display}" == /* ]]; then
        display="\$EXTERNAL/${display##*/}"
      elif [[ "${display}" == *=/* ]]; then
        display="${display%%=*}=\$EXTERNAL/${display##*/}"
      fi
    fi
    printf ' %q' "${display}"
  done
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

write_stage_status() {
  local stage_name="$1"
  local state="$2"
  local exit_code="${3:-null}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  mkdir -p "${STATUS_DIR}"
  local status_file="${STATUS_DIR}/${stage_name}.json"
  local temporary
  temporary="$(mktemp "${status_file}.XXXXXX")"
  printf '{"stage":"%s","state":"%s","exit_code":%s,"updated_utc":"%s"}\n' \
    "${stage_name}" \
    "${state}" \
    "${exit_code}" \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" > "${temporary}"
  mv -f -- "${temporary}" "${status_file}"
}

ACTIVE_STAGE=""
record_stage_failure() {
  local exit_code="$?"
  trap - ERR
  if [[ -n "${ACTIVE_STAGE}" ]]; then
    if [[ "${ACTIVE_STAGE}" == "train" && "${exit_code}" == "75" ]]; then
      write_stage_status "${ACTIVE_STAGE}" PAUSED "${exit_code}"
    else
      write_stage_status "${ACTIVE_STAGE}" FAILED "${exit_code}"
    fi
  fi
  exit "${exit_code}"
}
trap record_stage_failure ERR

execute_stage() {
  local stage_name="$1"
  local stage_function="$2"
  ACTIVE_STAGE="${stage_name}"
  write_stage_status "${stage_name}" RUNNING
  "${stage_function}"
  write_stage_status "${stage_name}" COMPLETED 0
  ACTIVE_STAGE=""
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ "${DRY_RUN}" != "1" && ! -f "${path}" ]]; then
    echo "ERROR: ${label} not found: $(path_log_label "${path}")" >&2
    exit 1
  fi
}

run_check() {
  echo "[check] Environment, complete data inventory and full decode validation"
  local check_args=(
    "${PYTHON_BIN}" scripts/check_env.py
    --require-full-data
    --lock "${CONSTRAINTS_FILE}"
    --runtime-profile "${RUNTIME_PROFILE}"
  )
  if [[ "${REQUIRE_CUDA}" == "1" ]]; then
    check_args+=(--require-cuda)
  fi
  if [[ "${INCLUDE_PRIVATE_HOST_PROVENANCE}" == "1" ]]; then
    check_args+=(--include-private-host-provenance)
  fi
  run_cmd "${check_args[@]}"

  # train.json inspection covers both EventHDR train and eval roots.
  for config_path in "${TRAIN_CONFIG}" "${AID_CONFIG}"; do
    local inspect_args=(
      "${PYTHON_BIN}" -m asgcn_unet.cli inspect
      --config "${config_path}"
      --samples "${INSPECT_SAMPLES}"
      --validate-all
    )
    if [[ "${INCLUDE_PRIVATE_HOST_PROVENANCE}" == "1" ]]; then
      inspect_args+=(--include-private-host-provenance)
    fi
    run_cmd "${inspect_args[@]}"
  done
}

run_profile() {
  echo "[profile] Complete CUDA topology scan and dense/first/empty/sparse training probes"
  check_runtime_profile
  local profile_args=("${PYTHON_BIN}" -m asgcn_unet.cli profile \
    --config "${TRAIN_CONFIG}" \
    --output "${PROFILE_OUTPUT}" \
    --samples "${PROFILE_SAMPLES}" \
    --top-density "${PROFILE_TOP_DENSITY}" \
    --cpu-threads "${PROFILE_CPU_THREADS}")
  if [[ "${PROFILE_RESUME}" == "1" ]]; then
    profile_args+=(--resume-scan)
  fi
  if [[ -n "${PROFILE_REUSE_REPORT}" ]]; then
    profile_args+=(--reuse-report "${PROFILE_REUSE_REPORT}")
  fi
  run_cmd "${profile_args[@]}"
  require_file "${PROFILE_OUTPUT}" "training preflight report"
}

run_train() {
  echo "[train] EventHDR ANN training"
  run_cmd env \
    REQUIRE_CUDA="${REQUIRE_CUDA}" \
    VALIDATE_DATASET=0 \
    PREFLIGHT_REPORT="${PROFILE_OUTPUT}" \
    ALLOW_UNVERIFIED_PREFLIGHT="${ALLOW_UNVERIFIED_PREFLIGHT}" \
    RESUME_CHECKPOINT="${RESUME_CHECKPOINT}" \
    RESTART_TRAIN="${RESTART_TRAIN}" \
    MAX_HOURS="${MAX_HOURS:-}" \
    CHECKPOINT_SECONDS="${CHECKPOINT_SECONDS:-}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${PROJECT_ROOT}/scripts/train.sh" "${TRAIN_CONFIG}"
  require_file "${ANN_CHECKPOINT}" "ANN checkpoint"
}

run_calibrate() {
  echo "[calibrate] Full EventHDR ANN-to-SNN calibration"
  require_file "${ANN_CHECKPOINT}" "ANN checkpoint"
  run_cmd env \
    REQUIRE_CUDA="${REQUIRE_CUDA}" \
    VALIDATE_DATASET=0 \
    CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES}" \
    OVERWRITE_CALIBRATION="${OVERWRITE_CALIBRATION}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${PROJECT_ROOT}/scripts/calibrate.sh" \
      "${TRAIN_CONFIG}" "${ANN_CHECKPOINT}" "${SNN_CHECKPOINT}"
  require_file "${SNN_CHECKPOINT}" "SNN checkpoint"
}

run_one_evaluation() {
  local config_path="$1"
  local checkpoint_path="$2"
  local mode="$3"
  local simulation_steps="$4"
  local dynamics="$5"
  local config_name="${config_path##*/}"
  local dataset_name="${config_name%.json}"
  if [[ "${EXPERIMENT}" == "fast" ]]; then
    dataset_name="${dataset_name%-fast}"
  fi
  local output_dir=""
  if [[ -n "${EVAL_OUTPUT_ROOT}" ]]; then
    output_dir="${EVAL_OUTPUT_ROOT}/${dataset_name}"
  fi
  run_cmd env \
    REQUIRE_CUDA="${REQUIRE_CUDA}" \
    VALIDATE_DATASET=0 \
    RUN_BENCHMARK=1 \
    BENCHMARK_WARMUP="${BENCHMARK_WARMUP}" \
    BENCHMARK_STEPS="${BENCHMARK_STEPS}" \
    EVAL_MAX_GRAPH_EDGES="${EVAL_MAX_GRAPH_EDGES}" \
    INFERENCE_MODE="${mode}" \
    SIMULATION_STEPS="${simulation_steps}" \
    SNN_DYNAMICS="${dynamics}" \
    EVAL_OUTPUT_DIR="${output_dir}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${PROJECT_ROOT}/scripts/eval.sh" "${config_path}" "${checkpoint_path}"
}

run_eval_config() {
  local config_path="$1"
  run_one_evaluation "${config_path}" "${ANN_CHECKPOINT}" ann 16 ""
  for dynamics in literal_eq15 standard_if; do
    for simulation_steps in "${SIMULATION_STEPS[@]}"; do
      run_one_evaluation \
        "${config_path}" \
        "${SNN_CHECKPOINT}" \
        snn \
        "${simulation_steps}" \
        "${dynamics}"
    done
  done
}

require_eval_checkpoints() {
  require_file "${ANN_CHECKPOINT}" "ANN checkpoint"
  require_file "${SNN_CHECKPOINT}" "SNN checkpoint"
}

run_eval() {
  echo "[eval] Complete EventHDR and EventAid-R ANN+SNN matrix"
  require_eval_checkpoints
  run_eval_config "${HDR_CONFIG}"
  run_eval_config "${AID_CONFIG}"
}

run_eval_hdr() {
  echo "[eval-hdr] EventHDR ANN+SNN matrix"
  require_eval_checkpoints
  run_eval_config "${HDR_CONFIG}"
}

run_eval_aid() {
  echo "[eval-aid] EventAid-R ANN+SNN matrix"
  require_eval_checkpoints
  run_eval_config "${AID_CONFIG}"
}

case "${STAGE}" in
  check) execute_stage check run_check ;;
  profile) execute_stage profile run_profile ;;
  train) execute_stage train run_train ;;
  calibrate) execute_stage calibrate run_calibrate ;;
  eval) execute_stage eval run_eval ;;
  eval-hdr) execute_stage eval-hdr run_eval_hdr ;;
  eval-aid) execute_stage eval-aid run_eval_aid ;;
  all)
    execute_stage check run_check
    execute_stage profile run_profile
    execute_stage train run_train
    execute_stage calibrate run_calibrate
    execute_stage eval run_eval
    ;;
esac

echo "Stage '${STAGE}' completed."
echo "Training preflight: $(path_log_label "${PROFILE_OUTPUT}")"
echo "ANN checkpoint: $(path_log_label "${ANN_CHECKPOINT}")"
echo "SNN checkpoint: $(path_log_label "${SNN_CHECKPOINT}")"
