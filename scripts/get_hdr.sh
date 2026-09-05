#!/usr/bin/env bash
# Entry scripts must not change an interactive shell through accidental sourcing.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' "Source ignored: run this entrypoint with bash or its scheduler; the current shell is unchanged." >&2
else
_asgcn_entrypoint() (
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  printf '%s\n' \
    "Usage: bash scripts/get_hdr.sh MODE [options]" \
    "" \
    "Modes (choose one):" \
    "  --download       Download the official public EventHDR share directly" \
    "  --source DIR     Import an extracted train/eval source" \
    "  --archive ZIP    Import an existing ZIP archive" \
    "  --check          Check files already in the destination" \
    "" \
    "Options:" \
    "  --destination DIR  Default: data/EventHDR" \
    "  --split train|eval  Select one split (default: both)" \
    "  --link              Symlink an extracted --source instead of copying" \
    "" \
    "Execution requires CONDA_PREFIX or PYTHON_BIN selecting a Conda environment." \
    "The HTTP download needs no CUDA, browser, or user login."
  return 0
fi

cd -- "${PROJECT_ROOT}"
# shellcheck source=scripts/runtime.sh
source "${PROJECT_ROOT}/scripts/runtime.sh"
select_conda_python
runtime_exec "${PYTHON_BIN}" "${SCRIPT_DIR}/get_hdr.py" "$@"
)
_asgcn_entrypoint "$@"
fi
