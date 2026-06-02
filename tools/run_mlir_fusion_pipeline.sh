#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MLIR_OPT="${MLIR_OPT:-/Users/allen/Developer/llvm-build/bin/mlir-opt}"
PLUGIN="${PLUGIN:-$REPO_ROOT/build-mlir/HIRMatMulBiasReluFusionPass.dylib}"
INPUT="${INPUT:-$REPO_ROOT/mlir_passes/test/matmul_bias_relu.mlir}"
OUTPUT="$REPO_ROOT/trace/mlir_fused_graph.mlir"
KERNEL_PROFILE="${KERNEL_PROFILE:-}"
PASS_PIPELINE="${PASS_PIPELINE:-builtin.module(hir-canonicalize,matmul-bias-relu-fusion,rmsnorm-kernel-selection,hir-fusion-lowering)}"

mkdir -p "$REPO_ROOT/trace"

"$MLIR_OPT" \
  --load-pass-plugin="$PLUGIN" \
  "$INPUT" \
  --allow-unregistered-dialect \
  --pass-pipeline="$PASS_PIPELINE" \
  > "$OUTPUT"

echo "Wrote $OUTPUT"
LOWERING_ARGS=(
  --input "$OUTPUT" \
  --lowered-output "$REPO_ROOT/trace/mlir_lowered_graph.json" \
  --plan-output "$REPO_ROOT/trace/mlir_execution_plan.json"
)

if [[ -n "$KERNEL_PROFILE" ]]; then
  IFS=':' read -r -a PROFILE_PATHS <<< "$KERNEL_PROFILE"
  for PROFILE_PATH in "${PROFILE_PATHS[@]}"; do
    LOWERING_ARGS+=(--kernel-profile "$PROFILE_PATH")
  done
fi

"$REPO_ROOT/tools/mlir_fusion_to_runtime_json.py" "${LOWERING_ARGS[@]}"
