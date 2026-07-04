#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MLIR_OPT="${MLIR_OPT:-$(command -v mlir-opt || true)}"
FILECHECK="${FILECHECK:-$(command -v FileCheck || true)}"
DEFAULT_PLUGIN=""
PLUGIN_CANDIDATES=(
  "$REPO_ROOT/build-mlir/libHIRMatMulBiasReluFusionPass.so"
  "$REPO_ROOT/build-mlir/HIRMatMulBiasReluFusionPass.so"
  "$REPO_ROOT/build-mlir/HIRMatMulBiasReluFusionPass.dylib"
  "$REPO_ROOT/build-mlir-codex/libHIRMatMulBiasReluFusionPass.so"
  "$REPO_ROOT/build-mlir-codex/HIRMatMulBiasReluFusionPass.so"
  "$REPO_ROOT/build-mlir-codex/HIRMatMulBiasReluFusionPass.dylib"
)
for candidate in "${PLUGIN_CANDIDATES[@]}"; do
  if [[ -f "$candidate" ]]; then
    DEFAULT_PLUGIN="$candidate"
    break
  fi
done
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

if [[ -z "$PLUGIN" || ! -f "$PLUGIN" ]]; then
  echo "error: MLIR pass plugin not found; set PLUGIN or build mlir_passes first" >&2
  printf 'searched:\n' >&2
  printf '  %s\n' "${PLUGIN_CANDIDATES[@]}" >&2
  exit 1
fi

if [[ -z "$DIALECT_PLUGIN" || ! -f "$DIALECT_PLUGIN" ]]; then
  echo "error: MLIR dialect plugin not found; set DIALECT_PLUGIN or build mlir_passes first" >&2
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

run_stablehlo_import_reject() {
  local name="$1"
  local input="$2"
  local expected_reason="$3"
  local tmp
  local err
  tmp="$(mktemp)"
  err="$(mktemp)"
  rm -f "$tmp"

  echo "[MLIR test] $name"

  if python3 "$REPO_ROOT/tools/import_stablehlo_subset.py" "$input" --output "$tmp" 2>"$err"; then
    echo "error: StableHLO importer unexpectedly accepted $input" >&2
    rm -f "$tmp" "$err"
    exit 1
  fi
  if [[ -f "$tmp" ]]; then
    echo "error: StableHLO importer emitted output for rejected input $input" >&2
    rm -f "$tmp" "$err"
    exit 1
  fi
  if ! grep -q "$expected_reason" "$err"; then
    echo "error: StableHLO importer rejection did not contain '$expected_reason'" >&2
    cat "$err" >&2
    rm -f "$tmp" "$err"
    exit 1
  fi
  rm -f "$tmp" "$err"
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

run_filecheck \
  "HIR Q/DQ canonicalization eliminates safe redundant boundaries" \
  "$REPO_ROOT/mlir_passes/test/hir_quant_canonicalize.mlir" \
  --split-input-file \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-quant-canonicalize)'

run_verify_diagnostics \
  "HIR Q/DQ canonicalization rejects invalid quantized dtype before rewrite" \
  "$REPO_ROOT/mlir_passes/test/hir_quant_canonicalize_invalid_dtype.mlir"

run_filecheck \
  "HIR quant propagation forms conservative INT8 islands" \
  "$REPO_ROOT/mlir_passes/test/hir_quant_propagation.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-quant-propagate)'

run_filecheck \
  "HIR INT8 operator selection uses capability/layout/profile gates" \
  "$REPO_ROOT/mlir_passes/test/hir_int8_operator_selection.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-quant-propagate,hir-int8-operator-selection)'

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
  "HIR target verifier rejects invalid padded metadata" \
  "$REPO_ROOT/mlir_passes/test/hir_padded_target_verifier_invalid.mlir"

run_verify_diagnostics \
  "HIR target verifier rejects invalid structured 2:4 metadata" \
  "$REPO_ROOT/mlir_passes/test/hir_sparse_2_4_verifier_invalid.mlir"

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
  "no fusion when tile padding overhead is too high" \
  "$REPO_ROOT/mlir_passes/test/no_fusion_padding_overhead.mlir" \
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
  "matmul-bias-relu padded HIR lowering" \
  "$REPO_ROOT/mlir_passes/test/matmul_bias_relu_padded_hir_lowering.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering)'

run_filecheck \
  "legal structured 2:4 MatMul-Bias-ReLU selects sparse HIR metadata" \
  "$REPO_ROOT/mlir_passes/test/sparse_2_4_matmul_hir_lowering.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering,hir-verify-fused-ops)'

run_filecheck \
  "illegal structured 2:4 RHS falls back to dense HIR metadata" \
  "$REPO_ROOT/mlir_passes/test/sparse_2_4_illegal_fallback.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering,hir-verify-fused-ops)'

run_filecheck \
  "non-constant structured 2:4 RHS falls back to dense HIR metadata" \
  "$REPO_ROOT/mlir_passes/test/sparse_2_4_nonconstant_fallback.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering,hir-verify-fused-ops)'

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
  "HIR MatMul-Bias-ReLU lowers to Linalg" \
  "$REPO_ROOT/mlir_passes/test/hir_matmul_bias_relu_to_linalg.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-matmul-bias-relu-to-linalg)'

run_filecheck \
  "Padded HIR MatMul-Bias-ReLU lowers to Linalg with crop" \
  "$REPO_ROOT/mlir_passes/test/hir_matmul_bias_relu_padded_to_linalg.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-matmul-bias-relu-to-linalg)'

run_filecheck \
  "HIR MatMul-Bias-ReLU bufferizes to memref" \
  "$REPO_ROOT/mlir_passes/test/hir_matmul_bias_relu_bufferize.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-matmul-bias-relu-to-linalg,one-shot-bufferize{bufferize-function-boundaries})'

run_filecheck \
  "HIR MatMul-Bias-ReLU lowers to LLVM dialect" \
  "$REPO_ROOT/mlir_passes/test/hir_matmul_bias_relu_to_llvm.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-matmul-bias-relu-to-linalg,one-shot-bufferize{bufferize-function-boundaries},convert-linalg-to-loops,convert-scf-to-cf,convert-index-to-llvm,convert-math-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)'

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

run_stablehlo_import_reject \
  "StableHLO parser rejects RMSNorm missing rsqrt" \
  "$REPO_ROOT/mlir_passes/test/stablehlo_bad_rmsnorm_missing_rsqrt.mlir" \
  "missing_rsqrt"

run_stablehlo_import_reject \
  "StableHLO parser rejects RMSNorm f16 in parser v1" \
  "$REPO_ROOT/mlir_passes/test/stablehlo_bad_rmsnorm_f16.mlir" \
  "unsupported_dtype"

run_stablehlo_import_reject \
  "StableHLO parser rejects MatMul missing ReLU maximum" \
  "$REPO_ROOT/mlir_passes/test/stablehlo_bad_matmul_missing_maximum.mlir" \
  "missing_maximum_relu"

run_stablehlo_import_reject \
  "StableHLO parser rejects MatMul dynamic shape" \
  "$REPO_ROOT/mlir_passes/test/stablehlo_bad_matmul_dynamic_shape.mlir" \
  "dynamic_shape_unsupported"

run_stablehlo_import_reject \
  "StableHLO parser rejects MatMul multi-use dot result" \
  "$REPO_ROOT/mlir_passes/test/stablehlo_bad_matmul_multi_use.mlir" \
  "dot_result_multi_use"

run_filecheck \
  "profile-guided INT8 qmatmul HIR lowering" \
  "$REPO_ROOT/mlir_passes/test/profile_guided_qmatmul_lowering.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering,hir-verify-fused-ops)'

run_filecheck \
  "quantization-planning: precision selection from target profile and default fallback" \
  "$REPO_ROOT/mlir_passes/test/quantization_planning.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(quantization-planning)'

run_filecheck \
  "serving phase analysis: default constants and target-profile cost attrs" \
  "$REPO_ROOT/mlir_passes/test/serving_phase_analysis.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(serving-phase-analysis)'

run_filecheck \
  "kv-layout-planning annotates paged/contiguous and byte estimate" \
  "$REPO_ROOT/mlir_passes/test/serving/kv_layout_planning.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(kv-layout-planning)'

run_filecheck \
  "replay-eligibility annotates static decode as eligible and prefill as ineligible" \
  "$REPO_ROOT/mlir_passes/test/serving/replay_eligibility.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(replay-eligibility)'

run_filecheck \
  "serving-optimization-pipeline produces serving, kv, and replay attrs" \
  "$REPO_ROOT/mlir_passes/test/serving/serving_optimization_pipeline.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(serving-optimization-pipeline)'

run_filecheck \
  "target constraints: budget forces contiguous layout; static_shape_support=false overrides replay" \
  "$REPO_ROOT/mlir_passes/test/serving/target_constraints.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(serving-optimization-pipeline)'

run_filecheck \
  "execution-provider-planning: target-aware backend selection with fallback chain" \
  "$REPO_ROOT/mlir_passes/test/serving/execution_provider_planning.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(serving-optimization-pipeline)'

run_filecheck \
  "representation-planning: effective dtype and layout from backend capability attrs" \
  "$REPO_ROOT/mlir_passes/test/serving/representation_planning.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(representation-planning-pipeline)'

run_filecheck \
  "layout-planning: layout propagation with agnostic ops and transform boundary detection" \
  "$REPO_ROOT/mlir_passes/test/serving/layout_planning.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(layout-planning-pipeline)'

run_filecheck \
  "boundary-planning: cast, layout transform, dequant, and unsupported boundary detection" \
  "$REPO_ROOT/mlir_passes/test/serving/boundary_planning.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(boundary-planning-pipeline)'

run_filecheck \
  "quantization-strategy: weight_only_int8, fp16 fallback, accuracy-sensitive, dequant boundary" \
  "$REPO_ROOT/mlir_passes/test/serving/quantization_strategy.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(quantization-strategy-planning-pipeline)'

run_filecheck \
  "kernel-availability: exact match lowerable, fallback_required, rewrite_candidate, arm-style match, coreml no-ANE-internals" \
  "$REPO_ROOT/mlir_passes/test/serving/kernel_availability.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(kernel-availability-planning-pipeline)'

run_filecheck \
  "lowering-decision: direct_lower, rewrite_then_lower, dequant_then_lower, fallback_backend, unsupported" \
  "$REPO_ROOT/mlir_passes/test/serving/lowering_decision.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(lowering-decision-planning-pipeline)'

run_filecheck \
  "candidate-generation: dtype variants, layout variants, unsupported-op/static-shape/constant-weight constraint marking" \
  "$REPO_ROOT/mlir_passes/test/serving/candidate_generation.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(candidate-generation-pipeline)'

run_filecheck \
  "llm-frontend-normalization rewrites raw attention graph to canonical serving IR" \
  "$REPO_ROOT/mlir_passes/test/serving/llm_frontend_normalization.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(llm-frontend-normalization)'

run_filecheck \
  "quantization cost effect: plan_dtype drives kv.dtype_bytes and quantization.dtype_bytes" \
  "$REPO_ROOT/mlir_passes/test/serving/quantization_cost_effect.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(quantization-planning,serving-phase-analysis,kv-layout-planning)'

run_filecheck \
  "cv-frontend-normalization: detects YOLOSeg-like pattern and emits cv.frontend.* planning attrs" \
  "$REPO_ROOT/mlir_passes/test/serving/cv_frontend_normalization.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(cv-frontend-normalization)'

run_filecheck \
  "cv-shape-inference: annotates pseudo-CV ops with static shape-derived size metadata" \
  "$REPO_ROOT/mlir_passes/test/serving/cv_shape_inference.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(cv-shape-inference)'

run_filecheck \
  "cv-memory-planning: assigns compiler buffer slots with linear lifetime reuse" \
  "$REPO_ROOT/mlir_passes/test/serving/cv_memory_planning.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(cv-memory-planning)'

run_filecheck \
  "cv-execution-domain-planning: classifies pseudo-CV ops into portable accelerated/host/fallback execution domains" \
  "$REPO_ROOT/mlir_passes/test/serving/cv_execution_domain_planning.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(cv-execution-domain-planning)'

run_filecheck \
  "affine loop tiling" \
  "$REPO_ROOT/mlir_passes/test/matmul_affine_tiling.mlir" \
  --affine-loop-tile="tile-sizes=32,32,32"

run_filecheck \
  "affine vectorization" \
  "$REPO_ROOT/mlir_passes/test/matmul_affine_vectorize.mlir" \
  --affine-super-vectorize="virtual-vector-size=4 test-fastest-varying=0"

echo "[MLIR test] all passed"
