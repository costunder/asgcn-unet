#!/usr/bin/env bash
set -Eeuo pipefail

# Clone 후 한 번 실행하는 Linux 서버 설치 스크립트.
# https://pytorch.org/get-started/locally/ 에서 서버 드라이버에 맞는 wheel을 고른 뒤:
#   TORCH_INDEX_URL=<official-wheel-index> ./scripts/setup.sh
# 재현용으로 버전을 고정할 때:
#   TORCH_VERSION=<version> TORCH_INDEX_URL=<official-wheel-index> \
#     PROJECT_EXTRAS=eval ./scripts/setup.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  echo "Loading installer settings: ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
TORCH_VERSION="${TORCH_VERSION:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
PROJECT_EXTRAS="${PROJECT_EXTRAS:-}"
REQUIRE_CUDA="${REQUIRE_CUDA:-0}"

if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${PROJECT_ROOT}/${VENV_DIR}"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(f"Python 3.10+ is required; found {sys.version.split()[0]}")
print(f"Using Python {sys.version.split()[0]}")
PY

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating virtual environment: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

# PIP_EXTRA_ARGS is intentionally optional. It is split on spaces, so paths with
# spaces should instead be configured through pip.conf.
PIP_ARGS=()
if [[ -n "${PIP_EXTRA_ARGS:-}" ]]; then
  read -r -a PIP_ARGS <<<"${PIP_EXTRA_ARGS}"
fi

TORCH_SPEC="torch"
if [[ -n "${TORCH_VERSION}" ]]; then
  TORCH_SPEC="torch==${TORCH_VERSION}"
fi

# Installing torch first preserves an explicitly chosen CUDA wheel when the
# editable project (which declares torch>=2.3) is installed below.
if [[ -n "${TORCH_INDEX_URL}" ]]; then
  "${VENV_PYTHON}" -m pip install "${PIP_ARGS[@]}" \
    --index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}"
else
  "${VENV_PYTHON}" -m pip install "${PIP_ARGS[@]}" "${TORCH_SPEC}"
fi

INSTALL_TARGET="${PROJECT_ROOT}"
if [[ -n "${PROJECT_EXTRAS}" ]]; then
  INSTALL_TARGET="${PROJECT_ROOT}[${PROJECT_EXTRAS}]"
fi
"${VENV_PYTHON}" -m pip install "${PIP_ARGS[@]}" -e "${INSTALL_TARGET}"

mkdir -p \
  "${PROJECT_ROOT}/data/EventHDR/train" \
  "${PROJECT_ROOT}/data/EventHDR/eval" \
  "${PROJECT_ROOT}/data/EventAid-R" \
  "${PROJECT_ROOT}/runs"

"${VENV_PYTHON}" - "${REQUIRE_CUDA}" <<'PY'
import sys
import torch

required = sys.argv[1] == "1"
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available now: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif required:
    raise SystemExit(
        "CUDA is required but unavailable. Run this check inside a GPU allocation, "
        "and verify TORCH_INDEX_URL and the NVIDIA driver."
    )
else:
    print("NOTE: no GPU is visible in this shell; login nodes commonly hide GPUs.")
PY

echo
echo "Installation complete."
echo "Python: ${VENV_PYTHON}"
echo "Next: ./scripts/get_aid.sh R-bear"
echo "Then place EventHDR H5 files under data/EventHDR/train and data/EventHDR/eval."
