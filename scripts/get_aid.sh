#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${PROJECT_ROOT}/manifests/eventaid_r.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DESTINATION="${EVENTAID_ROOT:-${PROJECT_ROOT}/data/EventAid-R}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/get_aid.sh [options] [SCENE ...]

With no SCENE, the complete 14-scene release is downloaded. ZIP files stay
compressed because the loader reads them directly.

Options:
  -d, --destination DIR  Download directory (default: data/EventAid-R)
  --all                  Explicitly download all 14 scenes (~24.68 GB)
  -h, --help             Show this help

Examples:
  ./scripts/get_aid.sh                  # all 14 scenes
  ./scripts/get_aid.sh R-bear R-outdoor
  ./scripts/get_aid.sh --all
EOF
}

DOWNLOAD_ALL=0
SCENES=()
while (($#)); do
  case "$1" in
    -d|--destination)
      if (($# < 2)); then
        echo "ERROR: $1 requires a directory" >&2
        exit 2
      fi
      DESTINATION="$2"
      shift 2
      ;;
    --all)
      DOWNLOAD_ALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      SCENES+=("$@")
      break
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      SCENES+=("$1")
      shift
      ;;
  esac
done

if ((DOWNLOAD_ALL == 1)) && ((${#SCENES[@]} > 0)); then
  echo "ERROR: use either --all or explicit scene names, not both" >&2
  exit 2
fi
if ((DOWNLOAD_ALL == 0)) && ((${#SCENES[@]} == 0)); then
  DOWNLOAD_ALL=1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest not found: ${MANIFEST}" >&2
  exit 1
fi

mkdir -p "${DESTINATION}"
DESTINATION="$(cd -- "${DESTINATION}" && pwd)"

REQUESTED=("${SCENES[@]}")
if ((DOWNLOAD_ALL == 1)); then
  REQUESTED=(__ALL__)
fi

echo "EventAid-R destination: ${DESTINATION}"
SELECTION_FILE="$(mktemp)"
trap 'rm -f -- "${SELECTION_FILE}"' EXIT

"${PYTHON_BIN}" - "${MANIFEST}" "${REQUESTED[@]}" >"${SELECTION_FILE}" <<'PY'
import json
import sys

manifest_path, *requested = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as stream:
    files = json.load(stream)["files"]

by_name = {item["scene"]: item for item in files}
if requested == ["__ALL__"]:
    selected = files
else:
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise SystemExit("Unknown scene(s): " + ", ".join(missing))
    selected = [by_name[name] for name in requested]

for item in selected:
    print(item["scene"], item["size"], item["url"], sep="\t")
PY

while IFS=$'\t' read -r scene size url; do
  [[ -n "${scene}" ]] || continue
  target="${DESTINATION}/${scene}.zip"

  if "${PYTHON_BIN}" - "${target}" <<'PY'
import sys
import zipfile

raise SystemExit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)
PY
  then
    echo "Already downloaded and valid: ${target}"
    continue
  fi

  echo "Downloading ${scene} (${size})"
  curl --location --fail --retry 5 --retry-delay 3 --continue-at - \
    --output "${target}" "${url}"

  "${PYTHON_BIN}" - "${target}" <<'PY'
import sys
import zipfile

path = sys.argv[1]
if not zipfile.is_zipfile(path):
    raise SystemExit(f"Downloaded file is not a valid ZIP: {path}")
PY
  echo "Verified ZIP container: ${target}"
done <"${SELECTION_FILE}"

echo "Done. Keep the ZIP files compressed; the EventAid-R loader reads them directly."
