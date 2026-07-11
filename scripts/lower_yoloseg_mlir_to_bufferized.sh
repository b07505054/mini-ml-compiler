#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IN_MLIR="${YOLOSEG_GENERIC_MLIR:-${REPO_ROOT}/artifacts/yoloseg_generic_frontend/yoloseg.generic.mlir}"
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

if [[ ! -f "${IN_MLIR}" ]]; then
  cat >&2 <<EOF
error: YOLO-Seg generic MLIR not found.

Expected:
  ${IN_MLIR}

Run scripts/run_yoloseg_generic_mlir_emission.sh first or set YOLOSEG_GENERIC_MLIR.
EOF
  exit 1
fi

if [[ -z "${MLIR_OPT}" ]]; then
  echo "error: mlir-opt not found; set MLIR_OPT or add it to PATH" >&2
  exit 1
fi

HELP_TEXT="$("${MLIR_OPT}" --help-hidden 2>&1)"
for pass_name in one-shot-bufferize buffer-deallocation-pipeline; do
  if [[ "${HELP_TEXT}" != *"--${pass_name}"* ]]; then
    echo "error: required mlir-opt pass unavailable: ${pass_name}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_DIR}"

BUFFERIZED_MLIR="${OUT_DIR}/${PREFIX}.bufferized.mlir"
REPORT_JSON="${OUT_DIR}/${PREFIX}.bufferization_report.json"
PASS_PIPELINE='builtin.module(one-shot-bufferize{bufferize-function-boundaries},buffer-deallocation-pipeline)'

"${MLIR_OPT}" "${IN_MLIR}" --pass-pipeline="${PASS_PIPELINE}" -o "${BUFFERIZED_MLIR}"
"${MLIR_OPT}" "${BUFFERIZED_MLIR}" >/dev/null

"${PYTHON}" - "${IN_MLIR}" "${BUFFERIZED_MLIR}" "${REPORT_JSON}" "${PASS_PIPELINE}" <<'PY'
import json
import re
import sys
from collections import Counter
from pathlib import Path

in_path = Path(sys.argv[1])
bufferized_path = Path(sys.argv[2])
report_path = Path(sys.argv[3])
pipeline = sys.argv[4]

KNOWN_DIALECTS = {
    "affine",
    "arith",
    "bufferization",
    "cf",
    "func",
    "linalg",
    "math",
    "memref",
    "scf",
    "tensor",
}
OP_RE = re.compile(r"\b([A-Za-z_][\w]*\.[A-Za-z_][\w]*)\b")


def op_counts(text: str) -> Counter[str]:
    return Counter(
        op for op in OP_RE.findall(text) if op.split(".", 1)[0] in KNOWN_DIALECTS
    )


def dialect_counts(counts: Counter[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for op, count in counts.items():
        dialect = op.split(".", 1)[0]
        result[dialect] = result.get(dialect, 0) + count
    return dict(sorted(result.items()))


def function_signature_summary(text: str) -> dict[str, object]:
    match = re.search(r"func\.func @([^(]+)\((.*?)\)\s*(?:->\s*([^{]+?))?\s*\{", text, re.S)
    if not match:
        return {"found": False}
    args = match.group(2)
    returns = (match.group(3) or "").strip()
    tensor_args = re.findall(r":\s*tensor<[^>]+>", args)
    memref_args = re.findall(r":\s*memref<[^>]+>", args)
    return {
        "found": True,
        "name": match.group(1),
        "argument_count": len(tensor_args) + len(memref_args),
        "tensor_argument_count": len(tensor_args),
        "memref_argument_count": len(memref_args),
        "return_types": returns,
    }


input_text = in_path.read_text(encoding="utf-8")
bufferized_text = bufferized_path.read_text(encoding="utf-8")
input_counts = op_counts(input_text)
bufferized_counts = op_counts(bufferized_text)

report = {
    "input_mlir": str(in_path),
    "bufferized_mlir": str(bufferized_path),
    "pass_pipeline": pipeline,
    "mlir_verification_status": "verified_with_mlir_opt_after_bufferization",
    "input_operation_count": sum(input_counts.values()),
    "bufferized_operation_count": sum(bufferized_counts.values()),
    "input_dialect_counts": dialect_counts(input_counts),
    "bufferized_dialect_counts": dialect_counts(bufferized_counts),
    "input_selected_ops": {
        op: input_counts.get(op, 0)
        for op in [
            "tensor.empty",
            "tensor.pad",
            "tensor.generate",
            "tensor.extract",
            "tensor.extract_slice",
            "tensor.insert_slice",
            "linalg.generic",
        ]
    },
    "bufferized_selected_ops": {
        op: bufferized_counts.get(op, 0)
        for op in [
            "tensor.empty",
            "tensor.pad",
            "tensor.generate",
            "tensor.extract",
            "tensor.extract_slice",
            "tensor.insert_slice",
            "linalg.generic",
            "memref.alloc",
            "memref.dealloc",
            "memref.copy",
            "memref.subview",
            "memref.load",
            "memref.store",
        ]
    },
    "remaining_tensor_ops": {
        op: count for op, count in sorted(bufferized_counts.items()) if op.startswith("tensor.")
    },
    "remaining_linalg_ops": {
        op: count for op, count in sorted(bufferized_counts.items()) if op.startswith("linalg.")
    },
    "introduced_memref_ops": {
        op: count for op, count in sorted(bufferized_counts.items()) if op.startswith("memref.")
    },
    "input_function_abi": function_signature_summary(input_text),
    "bufferized_function_abi": function_signature_summary(bufferized_text),
    "truth_boundary": "full_graph_bufferization_verified_no_machine_codegen_no_runtime_execution_no_numerical_equivalence_validation_no_execution_plan_generation",
}

report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"bufferization report: {report_path}")
print(f"  input_mlir: {in_path}")
print(f"  bufferized_mlir: {bufferized_path}")
print(f"  pass_pipeline: {pipeline}")
print(f"  remaining_tensor_ops: {sum(report['remaining_tensor_ops'].values())}")
print(f"  remaining_linalg_ops: {sum(report['remaining_linalg_ops'].values())}")
print(f"  memref_alloc: {bufferized_counts.get('memref.alloc', 0)}")
print(f"  memref_dealloc: {bufferized_counts.get('memref.dealloc', 0)}")
print(f"  mlir_verification_status: {report['mlir_verification_status']}")
PY

cat <<EOF
YOLO-Seg tensor/linalg-to-bufferized MLIR boundary complete.

Artifacts:
  ${BUFFERIZED_MLIR}
  ${REPORT_JSON}
EOF
