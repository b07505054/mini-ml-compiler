#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MLIR_OPT="${MLIR_OPT:-/Users/allen/Developer/llvm-build/bin/mlir-opt}"
PLUGIN="${PLUGIN:-$REPO_ROOT/build-mlir/HIRMatMulBiasReluFusionPass.dylib}"
INPUT="${INPUT:-$REPO_ROOT/mlir_passes/test/matmul_bias_relu.mlir}"
OUTPUT="$REPO_ROOT/trace/mlir_fused_graph.mlir"

mkdir -p "$REPO_ROOT/trace"

"$MLIR_OPT" \
  --load-pass-plugin="$PLUGIN" \
  "$INPUT" \
  --pass-pipeline='builtin.module(matmul-bias-relu-fusion)' \
  > "$OUTPUT"

echo "Wrote $OUTPUT"
"$REPO_ROOT/tools/mlir_fusion_to_runtime_json.py" \
  --input "$OUTPUT" \
  --lowered-output "$REPO_ROOT/trace/mlir_lowered_graph.json" \
  --plan-output "$REPO_ROOT/trace/mlir_execution_plan.json"