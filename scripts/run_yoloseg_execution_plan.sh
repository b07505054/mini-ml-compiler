#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${YOLOSEG_OUT_DIR:-${REPO_ROOT}/artifacts/yoloseg_generic_frontend}"
PREFIX="${YOLOSEG_PREFIX:-yoloseg}"
TARGET_PROFILE="${YOLOSEG_TARGET_PROFILE:-${REPO_ROOT}/configs/target_profiles/apple_a17pro_mobile.json}"
COMPILE_FOR_TARGET="${COMPILE_FOR_TARGET:-${REPO_ROOT}/build-mlir/compile-for-target}"

if [[ ! -f "${TARGET_PROFILE}" ]]; then
  cat >&2 <<EOF
error: target profile not found:
  ${TARGET_PROFILE}

Set YOLOSEG_TARGET_PROFILE to an existing target profile JSON.
EOF
  exit 1
fi

if [[ ! -x "${COMPILE_FOR_TARGET}" ]]; then
  cat >&2 <<EOF
error: compile-for-target not found or not executable:
  ${COMPILE_FOR_TARGET}

Build it first:
  cmake --build build-mlir --target compile-for-target HIRMatMulBiasReluFusionPass
EOF
  exit 1
fi

mkdir -p "${OUT_DIR}"

bash "${REPO_ROOT}/scripts/run_yoloseg_generic_mlir_emission.sh"
bash "${REPO_ROOT}/scripts/run_yoloseg_cv_semantic_annotation.sh"

ANNOTATED_MLIR="${OUT_DIR}/${PREFIX}.cv_annotated.mlir"
PLAN_JSON="${OUT_DIR}/${PREFIX}.execution_plan.json"
PLAN_MLIR="${OUT_DIR}/${PREFIX}.execution_plan_annotated.mlir"
DISPATCH_REPORT_JSON="${OUT_DIR}/${PREFIX}.dispatch_unit_report.json"

"${COMPILE_FOR_TARGET}" \
  --device-profile "${TARGET_PROFILE}" \
  --mlir "${ANNOTATED_MLIR}" \
  --out "${PLAN_JSON}" \
  --dump-annotated-mlir "${PLAN_MLIR}" \
  --dispatch-unit-report "${DISPATCH_REPORT_JSON}"

python3 - "${PLAN_JSON}" "${TARGET_PROFILE}" <<'PY'
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
profile_path = Path(sys.argv[2])
plan = json.loads(plan_path.read_text(encoding="utf-8"))
functions = plan.get("function_plans", [])
cv = plan.get("cv_extension", {})

per_op = sum(len(f.get("per_op_decisions", [])) for f in functions)
outputs = cv.get("outputs", [])
regions = cv.get("semantic_regions", [])

print("YOLO-Seg ExecutionPlan complete.")
print(f"  plan: {plan_path}")
print(f"  target_profile: {profile_path}")
print(f"  function_plans: {len(functions)}")
print(f"  per_op_decisions: {per_op}")
print(f"  cv_outputs: {len(outputs)}")
for output in outputs:
    print(f"    - {output.get('role')}: {output.get('shape')} {output.get('dtype')} {output.get('layout')}")
print(f"  cv_regions: {len(regions)}")
print(f"  truth_boundary: {cv.get('truth_boundary', plan.get('provenance', {}).get('truth_boundary'))}")
PY

cat <<EOF

Artifacts:
  ${PLAN_JSON}
  ${PLAN_MLIR}
EOF
