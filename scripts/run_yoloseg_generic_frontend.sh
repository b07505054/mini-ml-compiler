#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${YOLOSEG_ONNX_PATH:-${REPO_ROOT}/models/yolo-seg.onnx}"
OUT_DIR="${YOLOSEG_OUT_DIR:-${REPO_ROOT}/artifacts/yoloseg_generic_frontend}"
PREFIX="${YOLOSEG_PREFIX:-yoloseg}"
PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  cat <<EOF
YOLO-Seg ONNX model not found.

Expected path:
  ${MODEL_PATH}

Place the real model at models/yolo-seg.onnx, then run:
  scripts/run_yoloseg_generic_frontend.sh

No frontend artifacts were generated.
EOF
  exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" \
  "${REPO_ROOT}/tools/run_generic_onnx_frontend.py" \
  "${MODEL_PATH}" \
  "${OUT_DIR}" \
  --prefix "${PREFIX}"

cat <<EOF
YOLO-Seg generic frontend artifacts written to:
  ${OUT_DIR}

Primary reports:
  ${OUT_DIR}/${PREFIX}.frontend_report.json
  ${OUT_DIR}/${PREFIX}.diagnostics_report.json
EOF
