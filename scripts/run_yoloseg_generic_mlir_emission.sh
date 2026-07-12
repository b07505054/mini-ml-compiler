#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${YOLOSEG_ONNX_PATH:-${REPO_ROOT}/models/yolo-seg.onnx}"
OUT_DIR="${YOLOSEG_OUT_DIR:-${REPO_ROOT}/artifacts/yoloseg_generic_frontend}"
PREFIX="${YOLOSEG_PREFIX:-yoloseg}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
MLIR_OPT="${MLIR_OPT:-$(command -v mlir-opt || true)}"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

if [[ -z "${MLIR_OPT}" && -x /opt/homebrew/opt/llvm/bin/mlir-opt ]]; then
  MLIR_OPT="/opt/homebrew/opt/llvm/bin/mlir-opt"
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  cat >&2 <<EOF
YOLO-Seg ONNX model not found.

Expected:
  ${MODEL_PATH}

Set YOLOSEG_ONNX_PATH or place the model at models/yolo-seg.onnx.
EOF
  exit 1
fi

if [[ -z "${MLIR_OPT}" ]]; then
  echo "error: mlir-opt not found; set MLIR_OPT or add it to PATH" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

"${PYTHON}" "${REPO_ROOT}/tools/run_generic_onnx_frontend.py" \
  "${MODEL_PATH}" \
  "${OUT_DIR}" \
  --prefix "${PREFIX}"

SHAPE_IR="${OUT_DIR}/${PREFIX}.shape_generic_graph_ir.json"
CONTRACT_JSON="${OUT_DIR}/${PREFIX}.lowering_contract.json"
MLIR_OUT="${OUT_DIR}/${PREFIX}.generic.mlir"
VERIFIED_MLIR="${OUT_DIR}/${PREFIX}.generic.verified.mlir"
REPORT_JSON="${OUT_DIR}/${PREFIX}.generic_mlir_emission_report.json"

"${PYTHON}" "${REPO_ROOT}/tools/check_generic_lowering_contract.py" \
  --in "${SHAPE_IR}" \
  --out "${CONTRACT_JSON}"

MODEL_ARTIFACT_REF="${MODEL_PATH#"${REPO_ROOT}"/}"

"${PYTHON}" "${REPO_ROOT}/tools/generic_graph_ir_to_mlir.py" \
  --in "${SHAPE_IR}" \
  --out "${MLIR_OUT}" \
  --model-artifact "${MODEL_ARTIFACT_REF}"

"${MLIR_OPT}" "${MLIR_OUT}" > "${VERIFIED_MLIR}"

"${PYTHON}" - "${SHAPE_IR}" "${MLIR_OUT}" "${CONTRACT_JSON}" "${REPORT_JSON}" <<'PY'
import json
import re
import sys
from collections import Counter
from pathlib import Path

shape_path = Path(sys.argv[1])
mlir_path = Path(sys.argv[2])
contract_path = Path(sys.argv[3])
report_path = Path(sys.argv[4])

graph_ir = json.loads(shape_path.read_text(encoding="utf-8"))
contract = json.loads(contract_path.read_text(encoding="utf-8"))
mlir_text = mlir_path.read_text(encoding="utf-8")

node_count = len(graph_ir.get("nodes", []))
op_counts = Counter(node.get("op", "") for node in graph_ir.get("nodes", []))
known_dialects = ["func", "tensor", "linalg", "arith", "math"]
dialects = [dialect for dialect in known_dialects if f"{dialect}." in mlir_text]
unsupported = contract.get("blocking_nodes", [])

report = {
    "model": str(shape_path),
    "mlir": str(mlir_path),
    "total_nodes": node_count,
    "emitted_nodes": node_count if not unsupported else node_count - len(unsupported),
    "unsupported_nodes": unsupported,
    "unsupported_node_count": len(unsupported),
    "generic_op_counts": dict(sorted(op_counts.items())),
    "mlir_verification_status": "verified_with_mlir_opt",
    "emitted_dialects": dialects,
    "truth_boundary": "full_graph_mlir_emission_verified_no_backend_codegen_no_runtime_execution_no_execution_plan_generation",
}
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"generic MLIR emission report: {report_path}")
print(f"  total_nodes: {report['total_nodes']}")
print(f"  emitted_nodes: {report['emitted_nodes']}")
print(f"  unsupported_node_count: {report['unsupported_node_count']}")
print(f"  mlir_verification_status: {report['mlir_verification_status']}")
PY

cat <<EOF
YOLO-Seg GenericGraphIR-to-MLIR emission complete.

Artifacts:
  ${MLIR_OUT}
  ${VERIFIED_MLIR}
  ${REPORT_JSON}
EOF
