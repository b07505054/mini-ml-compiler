#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MLIR_OPT="${MLIR_OPT:-$(command -v mlir-opt || true)}"
PLUGIN="${PLUGIN:-$REPO_ROOT/build-mlir/HIRMatMulBiasReluFusionPass.dylib}"
DIALECT_PLUGIN="${DIALECT_PLUGIN:-$PLUGIN}"
INPUT="${INPUT:-$REPO_ROOT/mlir_passes/test/matmul_bias_relu.mlir}"
OUTPUT="$REPO_ROOT/trace/mlir_fused_graph.mlir"
KERNEL_PROFILE="${KERNEL_PROFILE:-}"
PASS_PIPELINE="${PASS_PIPELINE:-builtin.module(hir-canonicalize,matmul-bias-relu-fusion,rmsnorm-kernel-selection,hir-fusion-lowering,hir-verify-fused-ops)}"

if [[ -z "$MLIR_OPT" ]]; then
  echo "error: mlir-opt not found; set MLIR_OPT or add it to PATH" >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/trace"

"$MLIR_OPT" \
  --load-pass-plugin="$PLUGIN" \
  --load-dialect-plugin="$DIALECT_PLUGIN" \
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
