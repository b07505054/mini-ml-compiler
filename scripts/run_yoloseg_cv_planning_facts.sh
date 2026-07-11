#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IN_MLIR="${YOLOSEG_CV_ANNOTATED_MLIR:-${REPO_ROOT}/artifacts/yoloseg_generic_frontend/yoloseg.cv_annotated.mlir}"
SHAPE_IR="${YOLOSEG_SHAPE_IR:-${REPO_ROOT}/artifacts/yoloseg_generic_frontend/yoloseg.shape_generic_graph_ir.json}"
OUT_DIR="${YOLOSEG_OUT_DIR:-${REPO_ROOT}/artifacts/yoloseg_generic_frontend}"
PREFIX="${YOLOSEG_PREFIX:-yoloseg}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

if [[ ! -f "${IN_MLIR}" ]]; then
  cat >&2 <<EOF
error: YOLO-Seg CV-annotated MLIR not found.

Expected:
  ${IN_MLIR}

Run scripts/run_yoloseg_cv_semantic_annotation.sh first or set YOLOSEG_CV_ANNOTATED_MLIR.
EOF
  exit 1
fi

mkdir -p "${OUT_DIR}"

REPORT_JSON="${OUT_DIR}/${PREFIX}.cv_planning_facts.json"
ARGS=(--in "${IN_MLIR}" --out "${REPORT_JSON}")
if [[ -f "${SHAPE_IR}" ]]; then
  ARGS+=(--shape-ir "${SHAPE_IR}")
fi

"${PYTHON}" "${REPO_ROOT}/tools/cv_planning_facts.py" "${ARGS[@]}"

cat <<EOF
YOLO-Seg CV planning facts complete.

Artifacts:
  ${REPORT_JSON}
EOF
