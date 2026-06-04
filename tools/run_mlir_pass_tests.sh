#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MLIR_OPT="${MLIR_OPT:-$(command -v mlir-opt || true)}"
FILECHECK="${FILECHECK:-$(command -v FileCheck || true)}"
DEFAULT_PLUGIN="$REPO_ROOT/build-mlir/HIRMatMulBiasReluFusionPass.dylib"
if [[ ! -f "$DEFAULT_PLUGIN" && -f "$REPO_ROOT/build-mlir-codex/HIRMatMulBiasReluFusionPass.dylib" ]]; then
  DEFAULT_PLUGIN="$REPO_ROOT/build-mlir-codex/HIRMatMulBiasReluFusionPass.dylib"
fi
PLUGIN="${PLUGIN:-$DEFAULT_PLUGIN}"
DIALECT_PLUGIN="${DIALECT_PLUGIN:-$PLUGIN}"

if [[ -z "$MLIR_OPT" ]]; then
  echo "error: mlir-opt not found; set MLIR_OPT or add it to PATH" >&2
  exit 1
fi

if [[ -z "$FILECHECK" ]]; then
  echo "error: FileCheck not found; set FILECHECK or add it to PATH" >&2
  exit 1
fi

run_filecheck() {
  local name="$1"
  local input="$2"
  shift 2

  echo "[MLIR test] $name"

  "$MLIR_OPT" "$input" "$@" \
    --load-dialect-plugin="$DIALECT_PLUGIN" \
    | "$FILECHECK" "$input"
}

run_stablehlo_subset_filecheck() {
  local name="$1"
  local input="$2"
  local check_file="$3"
  local pipeline="$4"
  local tmp
  tmp="$(mktemp)"

  echo "[MLIR test] $name"

  python3 "$REPO_ROOT/tools/import_stablehlo_subset.py" "$input" --output "$tmp" >/dev/null
  "$MLIR_OPT" "$tmp" \
    --load-pass-plugin="$PLUGIN" \
    --load-dialect-plugin="$DIALECT_PLUGIN" \
    --pass-pipeline="$pipeline" \
    | "$FILECHECK" "$check_file"
  rm -f "$tmp"
}

run_verify_diagnostics() {
  local name="$1"
  local input="$2"

  echo "[MLIR test] $name"

  "$MLIR_OPT" "$input" \
    --load-dialect-plugin="$DIALECT_PLUGIN" \
    --verify-diagnostics \
    >/dev/null
}

run_filecheck \
  "HIR dialect ops parse and verify" \
  "$REPO_ROOT/mlir_passes/test/hir_dialect_ops.mlir"

run_filecheck \
  "HIR INT8 quant ops parse and verify" \
  "$REPO_ROOT/mlir_passes/test/hir_quant_ops.mlir"

run_verify_diagnostics \
  "HIR dialect verifier rejects invalid metadata" \
  "$REPO_ROOT/mlir_passes/test/hir_dialect_verifier_invalid.mlir"

run_verify_diagnostics \
  "HIR INT8 quant verifier rejects invalid metadata" \
  "$REPO_ROOT/mlir_passes/test/hir_quant_verifier_invalid.mlir"

run_verify_diagnostics \
  "HIR layout verifier rejects invalid alignment" \
  "$REPO_ROOT/mlir_passes/test/hir_layout_verifier_invalid.mlir"

run_verify_diagnostics \
  "HIR target verifier rejects invalid tile shape" \
  "$REPO_ROOT/mlir_passes/test/hir_target_verifier_invalid.mlir"

run_verify_diagnostics \
  "HIR verifier rejects invalid bias broadcast" \
  "$REPO_ROOT/mlir_passes/test/hir_bad_bias_broadcast_invalid.mlir"

run_filecheck \
  "canonicalize add zero" \
  "$REPO_ROOT/mlir_passes/test/canonicalize_add_zero.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-canonicalize)'

run_filecheck \
  "canonicalize nested relu" \
  "$REPO_ROOT/mlir_passes/test/canonicalize_relu_relu.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-canonicalize)'

run_filecheck \
  "matmul-bias-relu fusion annotation" \
  "$REPO_ROOT/mlir_passes/test/matmul_bias_relu.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion)'

run_filecheck \
  "canonicalization enables fusion" \
  "$REPO_ROOT/mlir_passes/test/canonicalization_enables_fusion.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion)'

run_filecheck \
  "no fusion without relu" \
  "$REPO_ROOT/mlir_passes/test/no_fusion_without_relu.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion)'

run_filecheck \
  "no fusion for multi-use matmul result" \
  "$REPO_ROOT/mlir_passes/test/no_fusion_matmul_multi_use.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(matmul-bias-relu-fusion)'

run_filecheck \
  "no fusion for dynamic target shape" \
  "$REPO_ROOT/mlir_passes/test/no_fusion_dynamic_shape_target.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(matmul-bias-relu-fusion)'

run_filecheck \
  "rmsnorm kernel selection annotation" \
  "$REPO_ROOT/mlir_passes/test/rmsnorm_kernel_selection.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(rmsnorm-kernel-selection)'

run_filecheck \
  "matmul-bias-relu HIR lowering" \
  "$REPO_ROOT/mlir_passes/test/matmul_bias_relu_hir_lowering.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering)'

run_filecheck \
  "rmsnorm HIR lowering" \
  "$REPO_ROOT/mlir_passes/test/rmsnorm_hir_lowering.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(rmsnorm-kernel-selection,hir-fusion-lowering)'

run_filecheck \
  "rmsnorm conversion and verifier" \
  "$REPO_ROOT/mlir_passes/test/rmsnorm_conversion_verifier.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(rmsnorm-kernel-selection,hir-fusion-lowering,hir-verify-fused-ops)'

run_filecheck \
  "Apple Silicon RMSNorm HIR lowering" \
  "$REPO_ROOT/mlir_passes/test/rmsnorm_metal_target.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(rmsnorm-kernel-selection,hir-fusion-lowering,hir-verify-fused-ops)'

run_filecheck \
  "HIR RMSNorm lowers to Linalg/Math" \
  "$REPO_ROOT/mlir_passes/test/hir_rmsnorm_to_linalg.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-rmsnorm-to-linalg)'

run_filecheck \
  "HIR RMSNorm lowers to LLVM dialect" \
  "$REPO_ROOT/mlir_passes/test/hir_rmsnorm_to_llvm.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-rmsnorm-to-linalg,one-shot-bufferize{bufferize-function-boundaries},convert-linalg-to-loops,convert-scf-to-cf,convert-index-to-llvm,convert-math-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)'

run_filecheck \
  "StableHLO-compatible MatMul decomposition lowers to HIR" \
  "$REPO_ROOT/mlir_passes/test/stablehlo_compatible_matmul_to_hir.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering,hir-verify-fused-ops)'

run_filecheck \
  "StableHLO-compatible RMSNorm decomposition lowers to HIR" \
  "$REPO_ROOT/mlir_passes/test/stablehlo_compatible_rmsnorm_to_hir.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(stablehlo-compatible-rmsnorm-import)'

run_stablehlo_subset_filecheck \
  "StableHLO textual RMSNorm subset imports and lowers to HIR" \
  "$REPO_ROOT/mlir_passes/test/stablehlo_textual_rmsnorm.mlir" \
  "$REPO_ROOT/mlir_passes/test/stablehlo_compatible_rmsnorm_to_hir.mlir" \
  'builtin.module(stablehlo-compatible-rmsnorm-import)'

run_stablehlo_subset_filecheck \
  "StableHLO textual MatMul subset imports and lowers to HIR" \
  "$REPO_ROOT/mlir_passes/test/stablehlo_textual_matmul_bias_relu.mlir" \
  "$REPO_ROOT/mlir_passes/test/stablehlo_compatible_matmul_to_hir.mlir" \
  'builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering,hir-verify-fused-ops)'

run_filecheck \
  "profile-guided INT8 qmatmul HIR lowering" \
  "$REPO_ROOT/mlir_passes/test/profile_guided_qmatmul_lowering.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering,hir-verify-fused-ops)'

run_filecheck \
  "affine loop tiling" \
  "$REPO_ROOT/mlir_passes/test/matmul_affine_tiling.mlir" \
  --affine-loop-tile="tile-sizes=32,32,32"

run_filecheck \
  "affine vectorization" \
  "$REPO_ROOT/mlir_passes/test/matmul_affine_vectorize.mlir" \
  --affine-super-vectorize="virtual-vector-size=4 test-fastest-varying=0"

echo "[MLIR test] all passed"
