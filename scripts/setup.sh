#!/usr/bin/env bash
set -Eeuo pipefail

# Install only into the already-selected, non-base Conda server environment.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REQUIRE_CUDA="${REQUIRE_CUDA-0}"

if [[ "$#" -ne 0 ]]; then
  echo "ERROR: setup.sh accepts no positional arguments." >&2
  exit 1
fi
if [[ "${REQUIRE_CUDA}" != "0" && "${REQUIRE_CUDA}" != "1" ]]; then
  echo "ERROR: REQUIRE_CUDA must be 0 or 1." >&2
  exit 1
fi
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_PREFIX}" != /* \
   || ! -d "${CONDA_PREFIX}/conda-meta" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
  echo "ERROR: select an existing non-base Conda environment with its own Python first." >&2
  exit 1
fi
if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
  echo "ERROR: installation into the base Conda environment is not allowed." >&2
  exit 1
fi
if [[ -v PIP_TARGET || -v PIP_PREFIX || -v PIP_ROOT || -v PIP_PYTHON \
   || ( -v PIP_USER && "${PIP_USER}" != "0" ) ]]; then
  echo "ERROR: pip destination overrides are not allowed for the locked Conda installation." >&2
  exit 1
fi

CONDA_PYTHON="${CONDA_PREFIX}/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
export PIP_CONFIG_FILE=/dev/null
export PIP_USER=0
export PIP_REQUIRE_VIRTUALENV=0
echo "Legacy .env and installer version/interpreter overrides are ignored."
echo "Custom pip configuration is disabled; HTTP_PROXY and HTTPS_PROXY remain supported."

# This preflight uses only the standard library and runs before any pip/network work.
"${CONDA_PYTHON}" - "${PROJECT_ROOT}/constraints/server.json" \
  "${PROJECT_ROOT}/constraints/server.txt" <<'PY'
import json
import os
import platform
import re
import sys
from pathlib import Path


def fail(message):
    raise SystemExit("ERROR: " + message)


try:
    prefix = Path(os.environ["CONDA_PREFIX"]).resolve()
    if sys.prefix != sys.base_prefix or Path(sys.prefix).resolve() != prefix:
        fail("the selected interpreter must belong directly to the selected Conda environment")
    if not (prefix / "conda-meta").is_dir():
        fail("the selected interpreter is not a Conda environment")
    conda_executable = os.environ.get("CONDA_EXE")
    if os.environ.get("CONDA_DEFAULT_ENV") == "base" or (
        conda_executable
        and prefix == Path(conda_executable).resolve().parent.parent
    ):
        fail("installation into the base Conda environment is not allowed")
    profile = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    required = {
        "format_version": 1,
        "python": "3.12.14",
        "torch": "2.13.0+cu126",
        "cuda": "12.6",
        "platform": "Linux",
        "machine": "x86_64",
        "environment": "conda",
    }
    if (
        not isinstance(profile, dict)
        or type(profile.get("format_version")) is not int
        or any(profile.get(key) != value for key, value in required.items())
        or set(profile) - set(required) - {"packages"}
    ):
        fail("constraints/server.json does not describe the exact supported Conda server runtime")
    if "packages" in profile and not isinstance(profile["packages"], dict):
        fail("the server profile packages field must be an exact-version object")
    if not Path(sys.argv[2]).is_file():
        fail("the hashed server dependency lock is missing")
    actual = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
    }
    for key, value in actual.items():
        if value != required[key]:
            fail(f"server {key} must be {required[key]}; found {value}")
    libc_name, libc_version = platform.libc_ver()
    numbers = tuple(int(value) for value in re.findall(r"\d+", libc_version))
    if libc_name.lower() != "glibc" or numbers < (2, 28):
        fail("the locked server wheels require Linux glibc>=2.28")
except (OSError, ValueError, KeyError) as error:
    raise SystemExit(
        "ERROR: Conda server preflight could not read valid local runtime metadata "
        f"({type(error).__name__}); no installation was started"
    ) from None

print("Conda server preflight passed: Python 3.12.14, Linux x86_64, glibc>=2.28.")
PY

cd -- "${PROJECT_ROOT}"
"${CONDA_PYTHON}" -m pip install --no-user --require-hashes --only-binary=:all: \
  -r "${PROJECT_ROOT}/constraints/server.txt"
"${CONDA_PYTHON}" -m pip install --no-user --no-deps --no-build-isolation -e "${PROJECT_ROOT}[dev]"
"${CONDA_PYTHON}" -m pip check

CHECK_ARGS=(--lock constraints/py312.txt --runtime-profile constraints/server.json)
if [[ "${REQUIRE_CUDA}" == "1" ]]; then
  CHECK_ARGS+=(--require-cuda)
fi
"${CONDA_PYTHON}" scripts/check_env.py "${CHECK_ARGS[@]}"

echo "Conda installation and exact runtime verification complete."
echo "Next: bash scripts/get_aid.sh --all"
echo "Then: bash scripts/get_hdr.sh --download"
echo "Finally, inside a GPU allocation: bash scripts/run.sh all"
