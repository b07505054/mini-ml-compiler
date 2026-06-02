#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MLIR_OPT="${MLIR_OPT:-/Users/allen/Developer/llvm-build/bin/mlir-opt}"
FILECHECK="${FILECHECK:-/Users/allen/Developer/llvm-build/bin/FileCheck}"
PLUGIN="${PLUGIN:-$REPO_ROOT/build-mlir/HIRMatMulBiasReluFusionPass.dylib}"
DIALECT_PLUGIN="${DIALECT_PLUGIN:-$PLUGIN}"

run_filecheck() {
  local name="$1"
  local input="$2"
  shift 2

  echo "[MLIR test] $name"

  "$MLIR_OPT" "$input" "$@" \
    --load-dialect-plugin="$DIALECT_PLUGIN" \
    | "$FILECHECK" "$input"
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
  "affine loop tiling" \
  "$REPO_ROOT/mlir_passes/test/matmul_affine_tiling.mlir" \
  --affine-loop-tile="tile-sizes=32,32,32"

run_filecheck \
  "affine vectorization" \
  "$REPO_ROOT/mlir_passes/test/matmul_affine_vectorize.mlir" \
  --affine-super-vectorize="virtual-vector-size=4 test-fastest-varying=0"

echo "[MLIR test] all passed"
