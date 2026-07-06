#!/usr/bin/env bash
# run_qwen_compiler_pipeline.sh
# Compiler orchestration: Qwen model spec -> serving MLIR -> execution_plan.json
#
# Compiler outputs only:
#   artifacts/qwen/execution_plan.json
#
# Does NOT emit:
#   original_manifest.json
#   benchmark_manifest_no_quant.json
#   calibration_recipe.json (future, quant only)
#
# Usage:
#   bash tools/run_qwen_compiler_pipeline.sh
#
# Env overrides:
#   SPEC         path to QwenModelSpec JSON   (default: configs/models/qwen_0_5b_spec.json)
#   PROFILE      path to TargetDeviceProfile  (default: configs/target_profiles/nvidia_gtx1650_maxq.json)
#   MLIR_OUT     intermediate MLIR path       (default: mlir/qwen_0_5b_serving.mlir)
#   PLAN_OUT     execution plan output path   (default: artifacts/qwen/execution_plan.json)
#   BUILD_DIR    mlir_passes build directory  (default: build-mlir)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-mlir}"
SPEC="${SPEC:-$REPO_ROOT/configs/models/qwen_0_5b_spec.json}"
PROFILE="${PROFILE:-$REPO_ROOT/configs/target_profiles/nvidia_gtx1650_maxq.json}"
MLIR_OUT="${MLIR_OUT:-$REPO_ROOT/mlir/qwen_0_5b_serving.mlir}"
PLAN_OUT="${PLAN_OUT:-$REPO_ROOT/artifacts/qwen/execution_plan.json}"

QWEN_TOOL="$BUILD_DIR/qwen-to-serving-mlir"
CFT_TOOL="$BUILD_DIR/compile-for-target"

# Validate tools are built
if [[ ! -x "$QWEN_TOOL" ]]; then
  echo "error: qwen-to-serving-mlir not found at $QWEN_TOOL" >&2
  echo "  Build first: cmake -S mlir_passes -B build-mlir ... && cmake --build build-mlir" >&2
  exit 1
fi
if [[ ! -x "$CFT_TOOL" ]]; then
  echo "error: compile-for-target not found at $CFT_TOOL" >&2
  echo "  Build first: cmake -S mlir_passes -B build-mlir ... && cmake --build build-mlir" >&2
  exit 1
fi

# Validate inputs
if [[ ! -f "$SPEC" ]]; then
  echo "error: model spec not found: $SPEC" >&2
  exit 1
fi
if [[ ! -f "$PROFILE" ]]; then
  echo "error: device profile not found: $PROFILE" >&2
  exit 1
fi

mkdir -p "$(dirname "$MLIR_OUT")"
mkdir -p "$(dirname "$PLAN_OUT")"

echo "[1/2] qwen-to-serving-mlir"
"$QWEN_TOOL" --model-spec "$SPEC" --out "$MLIR_OUT"

echo "[2/2] compile-for-target"
"$CFT_TOOL" \
  --device-profile "$PROFILE" \
  --mlir "$MLIR_OUT" \
  --out "$PLAN_OUT"

echo ""
echo "Compiler output: $PLAN_OUT"
