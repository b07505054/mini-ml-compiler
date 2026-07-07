#!/usr/bin/env bash
# run_qwen_awq_compiler_pipeline.sh
# Compiler orchestration for Phase C (minimal, AWQ only):
#   Qwen model spec -> serving MLIR -> execution_plan.json (forced AWQ profile)
#
# This mirrors tools/run_qwen_compiler_pipeline.sh exactly, except it uses
# configs/target_profiles/nvidia_gtx1650_maxq_awq_forced.json, which declares
# an experimental forced global quantization override (strategy=weight_only_int4,
# algorithm=awq). Per-op backendCapabilities in that profile are IDENTICAL to
# nvidia_gtx1650_maxq.json -- this does NOT claim GTX 1650 has native INT4
# Tensor Core support. See that profile's sourceNotes/forcedQuantization block.
#
# Compiler outputs only:
#   artifacts/qwen_awq_plan/execution_plan.json
#
# This script does NOT produce the quantized model artifact itself -- that is
# tools/export_qwen_awq.py's job (writes artifacts/qwen_awq/, separate from
# this script's plan output directory).
#
# Usage:
#   bash tools/run_qwen_awq_compiler_pipeline.sh
#
# Env overrides:
#   SPEC         path to QwenModelSpec JSON   (default: configs/models/qwen_0_5b_spec.json)
#   PROFILE      path to TargetDeviceProfile  (default: configs/target_profiles/nvidia_gtx1650_maxq_awq_forced.json)
#   MLIR_OUT     intermediate MLIR path       (default: mlir/qwen_0_5b_serving_awq.mlir)
#   PLAN_OUT     execution plan output path   (default: artifacts/qwen_awq_plan/execution_plan.json)
#   BUILD_DIR    mlir_passes build directory  (default: build-mlir)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-mlir}"
SPEC="${SPEC:-$REPO_ROOT/configs/models/qwen_0_5b_spec.json}"
PROFILE="${PROFILE:-$REPO_ROOT/configs/target_profiles/nvidia_gtx1650_maxq_awq_forced.json}"
MLIR_OUT="${MLIR_OUT:-$REPO_ROOT/mlir/qwen_0_5b_serving_awq.mlir}"
PLAN_OUT="${PLAN_OUT:-$REPO_ROOT/artifacts/qwen_awq_plan/execution_plan.json}"

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

echo "[2/2] compile-for-target (forced AWQ profile)"
"$CFT_TOOL" \
  --device-profile "$PROFILE" \
  --mlir "$MLIR_OUT" \
  --out "$PLAN_OUT"

echo ""
echo "Compiler output: $PLAN_OUT"
echo ""
echo "Note: this plan references quantized_model_artifact_ref but does not"
echo "produce that artifact. Run tools/export_qwen_awq.py separately (requires"
echo "AutoAWQ on a CUDA-capable Linux host) to produce the actual quantized"
echo "checkpoint at artifacts/qwen_awq/."
