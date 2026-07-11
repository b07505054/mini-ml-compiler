#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IN_MLIR="${YOLOSEG_GENERIC_MLIR:-${REPO_ROOT}/artifacts/yoloseg_generic_frontend/yoloseg.generic.mlir}"
SHAPE_IR="${YOLOSEG_SHAPE_IR:-${REPO_ROOT}/artifacts/yoloseg_generic_frontend/yoloseg.shape_generic_graph_ir.json}"
OUT_DIR="${YOLOSEG_OUT_DIR:-${REPO_ROOT}/artifacts/yoloseg_generic_frontend}"
PREFIX="${YOLOSEG_PREFIX:-yoloseg}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
MLIR_OPT="${MLIR_OPT:-$(command -v mlir-opt || true)}"
PLUGIN="${MLIR_PASS_PLUGIN:-${REPO_ROOT}/build-mlir/HIRMatMulBiasReluFusionPass.dylib}"

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

if [[ ! -f "${PLUGIN}" ]]; then
  SO_PLUGIN="${PLUGIN%.dylib}.so"
  if [[ -f "${SO_PLUGIN}" ]]; then
    PLUGIN="${SO_PLUGIN}"
  else
    cat >&2 <<EOF
error: MLIR pass plugin not found.

Expected:
  ${PLUGIN}

Build it first, for example:
  cmake --build build-mlir --target HIRMatMulBiasReluFusionPass
EOF
    exit 1
  fi
fi

HELP_TEXT="$("${MLIR_OPT}" --load-pass-plugin="${PLUGIN}" --help-hidden 2>&1)"
if [[ "${HELP_TEXT}" != *"--cv-semantic-annotation"* ]]; then
  echo "error: required mlir-opt pass unavailable from plugin: cv-semantic-annotation" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

ANNOTATED_MLIR="${OUT_DIR}/${PREFIX}.cv_annotated.mlir"
REPORT_JSON="${OUT_DIR}/${PREFIX}.cv_semantic_report.json"
PASS_PIPELINE='builtin.module(cv-semantic-annotation)'

"${MLIR_OPT}" "${IN_MLIR}" \
  --load-pass-plugin="${PLUGIN}" \
  --load-dialect-plugin="${PLUGIN}" \
  --pass-pipeline="${PASS_PIPELINE}" \
  -o "${ANNOTATED_MLIR}"

"${MLIR_OPT}" "${ANNOTATED_MLIR}" >/dev/null

"${PYTHON}" - "${IN_MLIR}" "${ANNOTATED_MLIR}" "${REPORT_JSON}" "${PASS_PIPELINE}" "${SHAPE_IR}" <<'PY'
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

in_path = Path(sys.argv[1])
annotated_path = Path(sys.argv[2])
report_path = Path(sys.argv[3])
pipeline = sys.argv[4]
shape_path = Path(sys.argv[5])

text = annotated_path.read_text(encoding="utf-8")
input_text = in_path.read_text(encoding="utf-8")

KNOWN_DIALECTS = {
    "arith",
    "func",
    "linalg",
    "math",
    "tensor",
}
OP_RE = re.compile(r"\b([A-Za-z_][\w]*\.[A-Za-z_][\w]*)\b")
ATTR_RE = re.compile(r'cv\.([A-Za-z_][\w.]*)\s*=\s*("[^"]*"|\[[^\]]*\]|[0-9]+|true|false)')
TENSOR_RE = re.compile(r"tensor<[^>]+>")


def parse_value(value: str):
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        return re.findall(r'"([^"]*)"', value)
    if value.isdigit():
        return int(value)
    if value == "true":
        return True
    if value == "false":
        return False
    return value


def op_counts(mlir: str) -> Counter[str]:
    return Counter(
        op for op in OP_RE.findall(mlir) if op.split(".", 1)[0] in KNOWN_DIALECTS
    )


def function_attrs(mlir: str) -> dict[str, object]:
    for line in mlir.splitlines():
        if "func.func" in line and "cv." in line:
            return {f"cv.{k}": parse_value(v) for k, v in ATTR_RE.findall(line)}
    return {}


def operation_lines(mlir: str):
    for number, line in enumerate(mlir.splitlines(), start=1):
        if "cv." not in line:
            continue
        op_name = None
        for candidate in OP_RE.findall(line):
            if candidate.split(".", 1)[0] in KNOWN_DIALECTS:
                op_name = candidate
                break
        attrs = {f"cv.{k}": parse_value(v) for k, v in ATTR_RE.findall(line)}
        if op_name and attrs:
            yield number, op_name, attrs, TENSOR_RE.findall(line)


func_attrs = function_attrs(text)
regions: dict[str, dict[str, object]] = {}
output_roles = []
for line, op, attrs, tensor_types in operation_lines(text):
    output_role = attrs.get("cv.output_role")
    if output_role:
        output_roles.append(
            {
                "role": output_role,
                "line": line,
                "operation": op,
                "tensor_types": tensor_types,
                "confidence": attrs.get("cv.recognition_confidence"),
                "evidence": attrs.get("cv.recognition_evidence", []),
            }
        )

    region_id = attrs.get("cv.region_id")
    if not region_id:
        continue
    region = regions.setdefault(
        region_id,
        {
            "region_id": region_id,
            "semantic_roles": Counter(),
            "operation_count": 0,
            "root_operations": [],
            "operations": Counter(),
            "tensor_shapes": Counter(),
            "confidence": Counter(),
            "evidence": [],
        },
    )
    region["operation_count"] += 1
    region["operations"][op] += 1
    if attrs.get("cv.semantic_role"):
        region["semantic_roles"][attrs["cv.semantic_role"]] += 1
    if attrs.get("cv.output_role"):
        region["root_operations"].append({"line": line, "operation": op, "output_role": attrs["cv.output_role"]})
    if attrs.get("cv.recognition_confidence"):
        region["confidence"][attrs["cv.recognition_confidence"]] += 1
    evidence = attrs.get("cv.recognition_evidence", [])
    if isinstance(evidence, list):
        region["evidence"].extend(evidence)
    elif evidence:
        region["evidence"].append(evidence)
    for tensor in tensor_types:
        region["tensor_shapes"][tensor] += 1

normalized_regions = []
for region in regions.values():
    normalized_regions.append(
        {
            "region_id": region["region_id"],
            "semantic_roles": dict(sorted(region["semantic_roles"].items())),
            "operation_count": region["operation_count"],
            "root_operations": region["root_operations"],
            "operations": dict(sorted(region["operations"].items())),
            "tensor_shapes": dict(sorted(region["tensor_shapes"].items())),
            "confidence": dict(sorted(region["confidence"].items())),
            "recognition_evidence": sorted(set(region["evidence"])),
        }
    )

graph_ir_summary = None
if shape_path.exists():
    graph_ir = json.loads(shape_path.read_text(encoding="utf-8"))
    graph_ir_summary = {
        "path": str(shape_path),
        "node_count": len(graph_ir.get("nodes", [])),
        "output_count": len(graph_ir.get("outputs", [])),
    }

counts = op_counts(text)
input_counts = op_counts(input_text)
source_dependency = func_attrs.get("cv.semantic_annotation.source_name_dependency", "unknown")

report = {
    "input_mlir": str(in_path),
    "annotated_mlir": str(annotated_path),
    "shape_generic_graph_ir": graph_ir_summary,
    "pass_pipeline": pipeline,
    "mlir_verification_status": "verified_with_mlir_opt_after_cv_semantic_annotation",
    "model_level_annotation": {
        key: value
        for key, value in sorted(func_attrs.items())
        if key.startswith("cv.model_") or key.startswith("cv.semantic_annotation") or key == "cv.recognition_confidence"
    },
    "recognized_output_roles": output_roles,
    "recognized_regions": sorted(normalized_regions, key=lambda item: item["region_id"]),
    "operation_counts": dict(sorted(counts.items())),
    "input_operation_count": sum(input_counts.values()),
    "annotated_operation_count": sum(counts.values()),
    "source_name_dependence_check": source_dependency,
    "unresolved_semantic_questions": func_attrs.get("cv.semantic_annotation.unresolved", []),
    "truth_boundary": "cv_semantic_annotation_only_no_backend_selection_no_memory_plan_no_kernel_selection_no_execution_plan_generation",
}

report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"CV semantic annotation report: {report_path}")
print(f"  annotated_mlir: {annotated_path}")
print(f"  output_roles: {[item['role'] for item in output_roles]}")
print(f"  regions: {len(normalized_regions)}")
for region in sorted(normalized_regions, key=lambda item: item["region_id"]):
    print(f"  {region['region_id']}: ops={region['operation_count']} roles={region['semantic_roles']}")
print(f"  source_name_dependence_check: {source_dependency}")
print(f"  mlir_verification_status: {report['mlir_verification_status']}")
PY

cat <<EOF
YOLO-Seg CV semantic annotation complete.

Artifacts:
  ${ANNOTATED_MLIR}
  ${REPORT_JSON}
EOF
