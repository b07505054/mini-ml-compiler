#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLIR_OPT="${MLIR_OPT:-$(command -v mlir-opt || true)}"
PLUGIN="${PLUGIN:-$REPO_ROOT/build-mlir/HIRMatMulBiasReluFusionPass.dylib}"

if [[ -z "$MLIR_OPT" ]]; then
  echo "error: mlir-opt not found; set MLIR_OPT or add it to PATH" >&2
  exit 1
fi

python3 "$REPO_ROOT/tools/build_profile_cost_table.py" \
  --profile "$REPO_ROOT/trace/metal_rmsnorm_benchmark.json" \
  --output "$REPO_ROOT/trace/metal_rmsnorm_cost_table.json"

"$MLIR_OPT" \
  --load-pass-plugin="$PLUGIN" \
  --load-dialect-plugin="$PLUGIN" \
  "$REPO_ROOT/mlir_passes/test/rmsnorm_metal_target.mlir" \
  --allow-unregistered-dialect \
  --pass-pipeline='builtin.module(rmsnorm-kernel-selection,hir-fusion-lowering,hir-verify-fused-ops)' \
  > "$REPO_ROOT/trace/metal_rmsnorm_fused_graph.mlir"

python3 "$REPO_ROOT/tools/mlir_fusion_to_runtime_json.py" \
  --input "$REPO_ROOT/trace/metal_rmsnorm_fused_graph.mlir" \
  --lowered-output "$REPO_ROOT/trace/metal_rmsnorm_lowered_graph.json" \
  --plan-output "$REPO_ROOT/trace/metal_rmsnorm_execution_plan.json" \
  --kernel-profile "$REPO_ROOT/trace/metal_rmsnorm_cost_table.json" \
  --rmsnorm-backend Metal

(
  cd "$REPO_ROOT"
  python3 tools/validate_metal_rmsnorm_path.py
)
