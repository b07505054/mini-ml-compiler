#!/usr/bin/env bash
set -uo pipefail

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

PASSED=0
FAILED=0
FAILED_FILES=()

record_result() {
  local status="$1"
  local file="$2"
  if [[ "$status" -eq 0 ]]; then
    echo "PASS: $file"
    PASSED=$((PASSED + 1))
  else
    echo "FAIL: $file"
    FAILED=$((FAILED + 1))
    FAILED_FILES+=("$file")
  fi
}

run_tracked() {
  local file="$1"
  shift
  ( "$@" )
  local status=$?
  record_result "$status" "$file"
  return 0
}

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

  if "$MLIR_OPT" "$input" "$@" \
       --load-dialect-plugin="$DIALECT_PLUGIN" \
       | "$FILECHECK" "$input"; then
    record_result 0 "$input"
  else
    record_result 1 "$input"
  fi
}

run_filecheck_prefix() {
  local name="$1"
  local input="$2"
  local prefix="$3"
  shift 3
  echo "[MLIR test] $name"
  if "$MLIR_OPT" "$input" "$@" \
       --load-dialect-plugin="$DIALECT_PLUGIN" \
       | "$FILECHECK" "$input" --check-prefix="$prefix"; then
    record_result 0 "$input:$prefix"
  else
    record_result 1 "$input:$prefix"
  fi
}

run_stablehlo_subset_filecheck() {
  local name="$1"
  local input="$2"
  local check_file="$3"
  local pipeline="$4"
  local tmp
  tmp="$(mktemp)"

  echo "[MLIR test] $name"

  local status=0
  python3 "$REPO_ROOT/tools/import_stablehlo_subset.py" "$input" --output "$tmp" >/dev/null &&
    "$MLIR_OPT" "$tmp" \
      --load-pass-plugin="$PLUGIN" \
      --load-dialect-plugin="$DIALECT_PLUGIN" \
      --pass-pipeline="$pipeline" \
      | "$FILECHECK" "$check_file" || status=$?
  rm -f "$tmp"
  record_result "$status" "$input"
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

  local status=0
  if python3 "$REPO_ROOT/tools/import_stablehlo_subset.py" "$input" --output "$tmp" 2>"$err"; then
    echo "error: StableHLO importer unexpectedly accepted $input" >&2
    status=1
  fi
  if [[ -f "$tmp" ]]; then
    echo "error: StableHLO importer emitted output for rejected input $input" >&2
    status=1
  fi
  if ! grep -q "$expected_reason" "$err"; then
    echo "error: StableHLO importer rejection did not contain '$expected_reason'" >&2
    cat "$err" >&2
    status=1
  fi
  rm -f "$tmp" "$err"
  record_result "$status" "$input"
}

run_verify_diagnostics() {
  local name="$1"
  local input="$2"

  echo "[MLIR test] $name"

  if "$MLIR_OPT" "$input" \
       --load-dialect-plugin="$DIALECT_PLUGIN" \
       --verify-diagnostics >/dev/null; then
    record_result 0 "$input"
  else
    record_result 1 "$input"
  fi
}

# Ranking-invariance check for QuantizationCoDesignPass: compiles the input
# with and without the co-design pass ahead of candidate evaluation + plan
# selection, extracts every evaluation.* / selected_plan.* signal from both
# outputs, and requires them to be byte-identical — quant_codesign.est.*
# evidence must never affect ranking inputs or outputs.
run_quant_codesign_ranking_invariant() {
  local input="$REPO_ROOT/mlir_passes/test/serving/quantization_codesign_invariant.mlir"
  local a b
  a="$(mktemp)"
  b="$(mktemp)"

  echo "[MLIR test] quant-codesign ranking invariance (byte-identical with/without pass)"

  "$MLIR_OPT" "$input" \
    --allow-unregistered-dialect \
    --load-pass-plugin="$PLUGIN" \
    --load-dialect-plugin="$DIALECT_PLUGIN" \
    --pass-pipeline='builtin.module(candidate-evaluation-pipeline,plan-selection-pipeline)' \
    | grep -oE '(evaluation|selected_plan)\.[a-z_.0-9]+ = [^,}]+' > "$a"

  "$MLIR_OPT" "$input" \
    --allow-unregistered-dialect \
    --load-pass-plugin="$PLUGIN" \
    --load-dialect-plugin="$DIALECT_PLUGIN" \
    --pass-pipeline='builtin.module(quant-codesign-pipeline,candidate-evaluation-pipeline,plan-selection-pipeline)' \
    | grep -oE '(evaluation|selected_plan)\.[a-z_.0-9]+ = [^,}]+' > "$b"

  local status=0
  if [[ ! -s "$a" ]]; then
    echo "error: no ranking signals extracted — invariant test input is broken" >&2
    status=1
  fi
  if ! diff -u "$a" "$b"; then
    echo "error: ranking signals differ with quant-codesign enabled" >&2
    status=1
  fi
  rm -f "$a" "$b"
  record_result "$status" "$input"
}

# Like run_verify_diagnostics, but runs a pass pipeline so pass-emitted
# diagnostics (errors/warnings/remarks) can be verified against
# expected-* annotations in the input file.
run_pass_verify_diagnostics() {
  local name="$1"
  local input="$2"
  shift 2

  echo "[MLIR test] $name"

  if "$MLIR_OPT" "$input" "$@" \
       --load-dialect-plugin="$DIALECT_PLUGIN" \
       --verify-diagnostics >/dev/null; then
    record_result 0 "$input"
  else
    record_result 1 "$input"
  fi
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
  --pass-pipeline='builtin.module(quantization-planning,matmul-bias-relu-fusion)'

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
  "planned f32 Linalg MatMul-Bias-ReLU reaches LLVM with owned temporary deallocation" \
  "$REPO_ROOT/mlir_passes/test/hir_native_codegen.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-native-codegen)'

run_filecheck \
  "native precision planning selects structurally distinct fused and unfused LLVM paths" \
  "$REPO_ROOT/mlir_passes/test/hir_native_codegen_planning_choices.mlir" \
  --split-input-file \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-native-codegen)'

run_filecheck \
  "planning-guided padded native path reaches pure LLVM dialect" \
  "$REPO_ROOT/mlir_passes/test/hir_native_codegen_padding.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(hir-native-codegen)'

run_filecheck \
  "planning-disabled MatMul vectorizes without fusing Bias/ReLU" \
  "$REPO_ROOT/mlir_passes/test/hir_unfused_vectorization.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline="builtin.module(hir-structured-lowering,transform-preload-library{transform-library-paths=$REPO_ROOT/mlir_passes/transforms/vectorize_unfused_matmul_only.mlir},transform-interpreter{entry-point=__transform_main})"

run_filecheck \
  "native Static Cost Model V2 emits explainable A/B/C/D evidence" \
  "$REPO_ROOT/mlir_passes/test/native_cost_model_v2.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(quantization-planning,matmul-bias-relu-fusion)'

run_filecheck \
  "calibrated K-tail candidates expose materialized, direct, and specialized decisions" \
  "$REPO_ROOT/mlir_passes/test/native_cost_model_k_tail.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(quantization-planning,matmul-bias-relu-fusion)'

run_filecheck_prefix \
  "analytical candidate cost-model mode remains available" \
  "$REPO_ROOT/mlir_passes/test/gbdt_cost_model_modes.mlir" \
  ANALYTICAL \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(matmul-bias-relu-fusion{matmul-cost-model=analytical})'

run_filecheck_prefix \
  "GBDT mode fails closed to analytical for unsupported candidate support" \
  "$REPO_ROOT/mlir_passes/test/gbdt_cost_model_modes.mlir" \
  GBDT \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(matmul-bias-relu-fusion{matmul-cost-model=gbdt})'

run_filecheck_prefix \
  "hybrid mode exposes decomposed schedule semantics and safe fallback" \
  "$REPO_ROOT/mlir_passes/test/gbdt_cost_model_modes.mlir" \
  HYBRID \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(matmul-bias-relu-fusion{matmul-cost-model=hybrid})'

run_filecheck \
  "K-tail transfer keeps the dynamic source dimension out of bounds" \
  "$REPO_ROOT/mlir_passes/test/k_tail_transfer_read_bounds.mlir"

run_filecheck \
  "production direct K-tail shares one accumulator and has no padded source tile" \
  "$REPO_ROOT/mlir_passes/test/hir_direct_k_tail_vector_lowering.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(func.func(hir-direct-k-tail-vector-lowering))'

echo "[MLIR test] native codegen rejects unsupported f16 at HIR legality boundary"
if "$MLIR_OPT" "$REPO_ROOT/mlir_passes/test/hir_native_codegen_unsupported_f16.mlir" \
     --load-pass-plugin="$PLUGIN" \
     --load-dialect-plugin="$DIALECT_PLUGIN" \
     --pass-pipeline='builtin.module(hir-native-codegen)' \
     --verify-diagnostics >/dev/null; then
  record_result 0 "$REPO_ROOT/mlir_passes/test/hir_native_codegen_unsupported_f16.mlir"
else
  record_result 1 "$REPO_ROOT/mlir_passes/test/hir_native_codegen_unsupported_f16.mlir"
fi

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
  "boundary-materialization: materializes required hir.cast with provenance, defers dequant/layout, skips unsupported plans, no-op without requirements" \
  "$REPO_ROOT/mlir_passes/test/serving/boundary_materialization.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(boundary-materialization-pipeline)'

run_pass_verify_diagnostics \
  "boundary-materialization: malformed planning attrs are diagnosed, never silently ignored" \
  "$REPO_ROOT/mlir_passes/test/serving/boundary_materialization_invalid.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(boundary-materialization-pipeline)'

run_filecheck \
  "weight-classification: constant RHS, func-arg RHS, attention RHS, declared constant, unknown producer" \
  "$REPO_ROOT/mlir_passes/test/serving/weight_classification.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(weight-classification-planning-pipeline)'

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
  "quantized-boundary-refinement: weight_dequant_required for fallback backends, direct_lower trust, fp16_fallback, unknown" \
  "$REPO_ROOT/mlir_passes/test/serving/quantized_boundary_refinement.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(quantized-boundary-refinement-pipeline)'

run_filecheck \
  "alternative-lowering-planning: algebraic decomposition, missing kernel, representation conversion, layout conversion, backend fallback" \
  "$REPO_ROOT/mlir_passes/test/serving/alternative_lowering.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(alternative-lowering-planning-pipeline)'

run_filecheck \
  "candidate-generation: dtype variants, layout variants, unsupported-op/static-shape/constant-weight constraint marking" \
  "$REPO_ROOT/mlir_passes/test/serving/candidate_generation.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(candidate-generation-pipeline)'

run_filecheck \
  "candidate-evaluation: direct_lower zero penalty, decomposition penalty, representation_conversion boundary penalty, backend_fallback high penalty, unsupported rejected, rejected_candidates unchanged" \
  "$REPO_ROOT/mlir_passes/test/serving/candidate_evaluation.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(candidate-evaluation-pipeline)'

run_filecheck \
  "tile-planning-v1: tile fits local memory, rejection with reason, dynamic-shape deferral, inert without declared local memory, quant-shrunk footprint, annotation-only cost integration" \
  "$REPO_ROOT/mlir_passes/test/serving/tile_planning.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(tile-planning-pipeline,candidate-evaluation-pipeline)'

run_filecheck \
  "kernel-selection-v1: rmsnorm selected, matmul rejected, dtype/layout/tile mismatches rejected, missing tile plan and dynamic shape and missing registry deferred" \
  "$REPO_ROOT/mlir_passes/test/serving/kernel_selection.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(tile-planning-pipeline,kernel-selection-pipeline)'

run_filecheck \
  "quantization-codesign-v1: inert without policy, weight/backend legality, boundary overhead loses on small shapes, dispatchable kernel wins memory-bound, capability vs dispatchability, accuracy/dynamic/registry deferrals, declared-algorithm evidence, idempotent double-run" \
  "$REPO_ROOT/mlir_passes/test/serving/quantization_codesign.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(quant-codesign-pipeline,quant-codesign-pipeline)'

run_quant_codesign_ranking_invariant

run_filecheck \
  "shape-cost-model-v2: shape-scaled FLOPs/costs, dtype-aware bytes, dynamic-shape fallback, quant weight bytes, shape-aware ranking mode" \
  "$REPO_ROOT/mlir_passes/test/serving/shape_cost_model.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(candidate-evaluation-pipeline,plan-selection-pipeline)'

run_filecheck \
  "plan-selection: direct_lower beats decomposition, repr_conversion beats fallback, fallback last resort, unsupported no valid candidate, tiebreak deterministic" \
  "$REPO_ROOT/mlir_passes/test/serving/plan_selection.mlir" \
  --split-input-file \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(plan-selection-pipeline)'

run_filecheck \
  "llm-frontend-normalization rewrites raw attention graph to canonical serving IR" \
  "$REPO_ROOT/mlir_passes/test/serving/llm_frontend_normalization.mlir" \
  --allow-unregistered-dialect \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(llm-frontend-normalization)'

run_filecheck \
  "llm-frontend-normalization performs a localized per-occurrence rewrite across full per-layer expansion (serving.layer_index)" \
  "$REPO_ROOT/mlir_passes/test/serving/llm_frontend_normalization_layered.mlir" \
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
  "cv-semantic-annotation: annotates real upstream YOLO-Seg-like MLIR without cv ops" \
  "$REPO_ROOT/mlir_passes/test/serving/cv_semantic_annotation.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(cv-semantic-annotation)'

run_filecheck \
  "cv-execution-plan-attrs: attaches minimum real-CV ExecutionPlan attrs" \
  "$REPO_ROOT/mlir_passes/test/serving/cv_execution_plan_attrs.mlir" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline='builtin.module(cv-semantic-annotation,cv-execution-plan-attrs)'

run_filecheck \
  "affine loop tiling" \
  "$REPO_ROOT/mlir_passes/test/matmul_affine_tiling.mlir" \
  --affine-loop-tile="tile-sizes=32,32,32"

run_filecheck \
  "affine vectorization" \
  "$REPO_ROOT/mlir_passes/test/matmul_affine_vectorize.mlir" \
  --affine-super-vectorize="virtual-vector-size=4 test-fastest-varying=0"

run_filecheck \
  "generic nearest 2x resize existing-dialect prototype" \
  "$REPO_ROOT/mlir/generic_resize_nearest_2x_prototype.mlir"

run_filecheck \
  "generic stride-2 transposed convolution existing-dialect prototype" \
  "$REPO_ROOT/mlir/generic_conv_transpose2d_stride2_prototype.mlir"

# ---------------------------------------------------------------------------
# Host-side static checks for the AArch64 native-codegen backend slice
# (compile_hir_matmul_bias_relu_aarch64.sh). These require no network access
# and no Raspberry Pi -- they only exercise mlir-opt, mlir-translate, llc,
# and llvm-objdump on the dev host, cross-generating (never executing)
# AArch64 code. Real hardware execution is a separate, explicitly-labeled
# integration test: tools/run_backend_codegen_pi_integration.sh.
# ---------------------------------------------------------------------------
run_backend_codegen_static_checks() {
  local shape="$1"
  local input="$REPO_ROOT/mlir_passes/test/backend_codegen/matmul_bias_relu_${shape}.mlir"
  local out_dir
  out_dir="$(mktemp -d)"
  local name="matmul_bias_relu_${shape}"

  echo "[MLIR test] backend codegen static checks ($shape)"

  MLIR_BIN="$(dirname "$MLIR_OPT")" PLUGIN="$PLUGIN" \
    bash "$REPO_ROOT/mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh" \
    "$input" "$out_dir" "$name"

  # 1. HIR -> LLVM dialect: real llvm.func with the expected ciface wrapper.
  if ! grep -q "llvm.func @${name}(" "$out_dir/${name}_llvm.mlir"; then
    echo "error: expected llvm.func @${name} not found in ${name}_llvm.mlir" >&2
    exit 1
  fi
  if ! grep -q "llvm.func @_mlir_ciface_${name}(" "$out_dir/${name}_llvm.mlir"; then
    echo "error: expected llvm.func @_mlir_ciface_${name} not found in ${name}_llvm.mlir" >&2
    exit 1
  fi

  # 2. LLVM dialect -> textual LLVM IR.
  if ! grep -q "^define .*@${name}(" "$out_dir/${name}.ll"; then
    echo "error: expected 'define ... @${name}(' not found in ${name}.ll" >&2
    exit 1
  fi

  # 3. AArch64 assembly generation: non-empty, targets the expected symbol.
  if ! grep -q "^${name}:" "$out_dir/${name}.s"; then
    echo "error: expected label '${name}:' not found in ${name}.s" >&2
    exit 1
  fi

  # 4. AArch64 object generation: valid little-endian AArch64 ELF.
  local machine
  machine="$(llvm-readobj -h "$out_dir/${name}.o" | grep -o 'EM_AARCH64' || true)"
  if [[ "$machine" != "EM_AARCH64" ]]; then
    echo "error: ${name}.o is not an EM_AARCH64 object" >&2
    exit 1
  fi

  # 5. Expected exported symbols exist (both the raw kernel and the ciface
  #    wrapper the harness actually calls).
  local symbols
  symbols="$(llvm-objdump -t "$out_dir/${name}.o")"
  if ! grep -q " ${name}\$" <<<"$symbols"; then
    echo "error: symbol ${name} not found in ${name}.o" >&2
    exit 1
  fi
  if ! grep -q " _mlir_ciface_${name}\$" <<<"$symbols"; then
    echo "error: symbol _mlir_ciface_${name} not found in ${name}.o" >&2
    exit 1
  fi

  rm -rf "$out_dir"
}

for shape in 8x8x8 16x16x16 32x32x32; do
  run_tracked "backend_codegen/scalar_${shape}" \
    run_backend_codegen_static_checks "$shape"
done

# ---------------------------------------------------------------------------
# Host-side static checks for the vectorized AArch64 backend variant
# (compile_hir_matmul_bias_relu_aarch64.sh --variant vectorized). Requires no
# network access and no Raspberry Pi. Verifies, in order: (1) the
# project-owned Transform-dialect script actually produces MLIR vector
# ops before LLVM lowering, (2) the final LLVM dialect / LLVM IR still
# contains real LLVM vector types (not silently scalarized), and (3) the
# AArch64 assembly contains real NEON FMLA. Real hardware execution
# (repeated-call and mixed-shape correctness) is a separate, explicitly
# labeled integration test.
# ---------------------------------------------------------------------------
run_backend_codegen_vectorized_static_checks() {
  local shape="$1"
  local input="$REPO_ROOT/mlir_passes/test/backend_codegen/matmul_bias_relu_vectorized_${shape}.mlir"
  local out_dir
  out_dir="$(mktemp -d)"
  local name="matmul_bias_relu_vectorized_${shape}"
  local transform_script="$REPO_ROOT/mlir_passes/transforms/vectorize_matmul_bias_relu.mlir"

  echo "[MLIR test] backend codegen vectorized static checks ($shape)"

  # 1. Vector IR structural test: HIR -> Linalg -> Transform-dialect
  #    vectorization, stopping BEFORE bufferization/LLVM lowering, must
  #    contain at least one of vector.contract / vector.fma /
  #    vector.transfer_read / vector.transfer_write.
  "$MLIR_OPT" "$input" \
    --load-dialect-plugin="$PLUGIN" \
    --load-pass-plugin="$PLUGIN" \
    --pass-pipeline="builtin.module(hir-structured-lowering,transform-preload-library{transform-library-paths=$transform_script},transform-interpreter{entry-point=__transform_main})" \
    -o "$out_dir/${name}_vector.mlir"
  if ! grep -qE "vector\.(contract|fma|transfer_read|transfer_write)" "$out_dir/${name}_vector.mlir"; then
    echo "error: expected at least one of vector.contract/vector.fma/vector.transfer_read/vector.transfer_write in ${name}_vector.mlir" >&2
    exit 1
  fi
  if [[ "$shape" == "15x31x15" ]]; then
    grep -q "vector<16x32xf32>" "$out_dir/${name}_vector.mlir" ||
      { echo "error: padded D did not vectorize the 16x32 compute input" >&2; exit 1; }
    grep -q "tensor.extract_slice.*\\[15, 15\\]" "$out_dir/${name}_vector.mlir" ||
      { echo "error: padded D lost the 15x15 output crop" >&2; exit 1; }
  fi

  # 2. Full pipeline (reuses the exact reproducible script).
  MLIR_BIN="$(dirname "$MLIR_OPT")" PLUGIN="$PLUGIN" \
    bash "$REPO_ROOT/mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh" \
    --variant vectorized "$input" "$out_dir" "$name"

  # 3. LLVM vector IR structural test: the textual LLVM IR must contain a
  #    real LLVM vector type (e.g. "<16 x float>"), not scalarized fallback.
  if ! grep -qE "<[0-9]+ x float>" "$out_dir/${name}.ll"; then
    echo "error: expected an LLVM vector type (e.g. '<N x float>') in ${name}.ll -- vectorization silently scalarized" >&2
    exit 1
  fi
  if grep -qE '(^|[[:space:]])(linalg|tensor|vector)\.' "$out_dir/${name}_llvm.mlir"; then
    echo "error: unsupported structured/vector op survived the LLVM boundary" >&2
    exit 1
  fi

  # 4. NEON/FMLA assembly test: must contain real AArch64 NEON fmla on a
  #    vector register (v*.4s form), not scalar-only fmul/fadd.
  if ! grep -qE '^\s+fmla\s+v[0-9]+\.4s' "$out_dir/${name}.s"; then
    echo "error: expected 'fmla v*.4s' in ${name}.s -- vectorized build did not select fused NEON FMLA" >&2
    exit 1
  fi

  # 5. Same ABI/exported-symbol checks as the generic variant.
  if ! grep -q "llvm.func @${name}(" "$out_dir/${name}_llvm.mlir"; then
    echo "error: expected llvm.func @${name} not found in ${name}_llvm.mlir" >&2
    exit 1
  fi
  if ! grep -q "llvm.func @_mlir_ciface_${name}(" "$out_dir/${name}_llvm.mlir"; then
    echo "error: expected llvm.func @_mlir_ciface_${name} not found in ${name}_llvm.mlir" >&2
    exit 1
  fi
  local machine
  machine="$(llvm-readobj -h "$out_dir/${name}.o" | grep -o 'EM_AARCH64' || true)"
  if [[ "$machine" != "EM_AARCH64" ]]; then
    echo "error: ${name}.o is not an EM_AARCH64 object" >&2
    exit 1
  fi

  rm -rf "$out_dir"
}

for shape in 8x8x8 16x16x16 32x32x32 15x31x15; do
  run_tracked "backend_codegen/vectorized_${shape}" \
    run_backend_codegen_vectorized_static_checks "$shape"
done

run_backend_codegen_tiled_tail_static_checks() {
  local input="$REPO_ROOT/mlir_passes/test/backend_codegen/matmul_bias_relu_tiled_tail_15x31x15.mlir"
  local out_dir
  out_dir="$(mktemp -d)"
  local name="matmul_bias_relu_tiled_tail_15x31x15"
  echo "[MLIR test] bounded tiled-vector tail static checks (15x31x15)"
  MLIR_BIN="$(dirname "$MLIR_OPT")" PLUGIN="$PLUGIN" \
    bash "$REPO_ROOT/mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh" \
      --variant tiled-vectorized-materialized-tail --tile-m 8 --tile-n 8 --tile-k 8 \
      "$input" "$out_dir" "$name"
  grep -qE '<(4|8|16|32|64) x float>' "$out_dir/${name}.ll" ||
    { echo "error: tiled tail LLVM IR has no bounded vector compute" >&2; return 1; }
  if grep -qE '<(225|465|512) x float>' "$out_dir/${name}.ll"; then
    echo "error: whole-shape giant vector survived tiled tail lowering" >&2
    return 1
  fi
  if grep -qE '(^|[[:space:]])(linalg|tensor|vector)\.' "$out_dir/${name}_llvm.mlir"; then
    echo "error: tiled tail LLVM boundary contains unsupported dialect ops" >&2
    return 1
  fi
  local malloc_calls
  malloc_calls="$(grep -c 'call.*@malloc' "$out_dir/${name}.ll")"
  [[ "$malloc_calls" -eq 1 ]] ||
    { echo "error: tiled tail path materialized full padded heap buffers" >&2; return 1; }
  grep -qE '^\s+fmla\s+v[0-9]+\.4s' "$out_dir/${name}.s" ||
    { echo "error: tiled tail object has no NEON FMLA" >&2; return 1; }
  local text_size
  text_size="$(llvm-size "$out_dir/${name}.o" | tail -1 | awk '{print $1}')"
  [[ "$text_size" -lt 10000 ]] ||
    { echo "error: tiled tail text size $text_size exceeds 10000 bytes" >&2; return 1; }
  rm -rf "$out_dir"
}

run_tracked "backend_codegen/tiled_tail_15x31x15" \
  run_backend_codegen_tiled_tail_static_checks

run_backend_codegen_direct_k_tail_static_checks() {
  local input="$REPO_ROOT/mlir_passes/test/backend_codegen/matmul_bias_relu_direct_8x8x15.mlir"
  local out_dir
  out_dir="$(mktemp -d)"
  local name="matmul_bias_relu_direct_8x8x15"
  echo "[MLIR test] production direct K-tail static checks (8x8x15)"
  MLIR_BIN="$(dirname "$MLIR_OPT")" PLUGIN="$PLUGIN" \
    bash "$REPO_ROOT/mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh" \
      --variant tiled-vectorized-direct-k-tail \
      --tile-m 8 --tile-n 8 --tile-k 8 "$input" "$out_dir" "$name"
  if grep -qE '(^|[[:space:]])(hir|linalg|tensor|vector|scf|memref)\.' \
      "$out_dir/${name}_llvm.mlir"; then
    echo "error: direct K-tail LLVM boundary contains unsupported dialect ops" >&2
    return 1
  fi
  grep -qE '^\s+fmla\s+v[0-9]+\.4s' "$out_dir/${name}.s" ||
    { echo "error: direct K-tail object has no NEON FMLA" >&2; return 1; }
  local text_size
  text_size="$(llvm-size "$out_dir/${name}.o" | tail -1 | awk '{print $1}')"
  [[ "$text_size" -lt 4096 ]] ||
    { echo "error: direct K-tail text size $text_size exceeds 4096 bytes" >&2; return 1; }
  rm -rf "$out_dir"
}

run_tracked "backend_codegen/direct_k_tail_8x8x15" \
  run_backend_codegen_direct_k_tail_static_checks

run_k_tail_calibration_fit_check() {
  local out_json out_csv
  out_json="$(mktemp)"
  out_csv="$(mktemp)"
  python3 "$REPO_ROOT/tools/fit_k_tail_calibration.py" \
    --input "$REPO_ROOT/mlir_passes/test/data/k_tail_calibration_sample.txt" \
    --base-config "$REPO_ROOT/configs/calibration/cortex_a76_fp32_matmul_bias_relu_v1.json" \
    --output "$out_json" --csv-output "$out_csv" >/dev/null
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["fit_sample_count"] == 14; assert d["direct_vector_k_tail_per_k_ns"] > 0; assert "shape" not in d' "$out_json"
  local status=$?
  rm -f "$out_json" "$out_csv"
  return "$status"
}

run_tracked "calibration/k_tail_fit_deterministic" \
  run_k_tail_calibration_fit_check

# ---------------------------------------------------------------------------
# Host-side static checks for the tiled-vectorized AArch64 backend variant
# (compile_hir_matmul_bias_relu_aarch64.sh --variant tiled-vectorized).
# Requires no network access and no Raspberry Pi. Verifies, in order:
# (1) the project-owned tiling+vectorization Transform-dialect script
#     actually produces real scf.for loops (not a single unrolled
#     vector.contract) plus vector transfer/fma ops before LLVM lowering,
# (2) no vector type wider than 64 FP32 lanes appears anywhere in that
#     pre-bufferization intermediate -- the direct structural evidence that
#     this is NOT whole-shape vectorization,
# (3) the final LLVM IR still contains a real (narrow) LLVM vector type,
# (4) the AArch64 assembly contains real NEON FMLA AND real loop branches
#     (not just the ~2 branches of the fully-unrolled variant),
# (5) the 32x32x32 object stays under the 20,000-byte code-size gate this
#     slice exists to satisfy (artifacts/backend_codegen/
#     aarch64_matmul_bias_relu_tiled/README.md has the full before/after).
# Real hardware execution (repeated-call and mixed-shape correctness) is a
# separate, explicitly labeled integration test:
# tools/run_backend_codegen_tiled_pi_integration.sh.
# ---------------------------------------------------------------------------
run_backend_codegen_tiled_static_checks() {
  local shape="$1"
  local input="$REPO_ROOT/mlir_passes/test/backend_codegen/matmul_bias_relu_tiled_${shape}.mlir"
  local out_dir
  out_dir="$(mktemp -d)"
  local name="matmul_bias_relu_tiled_${shape}"
  # Default tile (4, 8, 8) -- since the tile-candidate slice, the fixed
  # transform file this test used to point at directly is gone; a concrete
  # instance is generated from the parameterized template instead (same
  # mechanism compile_hir_matmul_bias_relu_aarch64.sh --variant
  # tiled-vectorized uses internally with no --tile-m/--tile-n/--tile-k
  # given).
  local transform_script="$out_dir/transform.mlir"
  bash "$REPO_ROOT/mlir_passes/tools/generate_tiled_transform.sh" \
    --tile-m 4 --tile-n 8 --tile-k 8 --output "$transform_script" >/dev/null

  echo "[MLIR test] backend codegen tiled-vectorized static checks ($shape)"

  # 1. Tiled vector IR structural test: HIR -> Linalg -> tile+fuse+vectorize,
  #    stopping BEFORE bufferization/LLVM lowering, must contain real
  #    scf.for loops (the outer M/N/K tiling) plus vector transfer/fma ops
  #    (the microkernel body).
  "$MLIR_OPT" "$input" \
    --load-dialect-plugin="$PLUGIN" \
    --load-pass-plugin="$PLUGIN" \
    --pass-pipeline="builtin.module(hir-structured-lowering,transform-preload-library{transform-library-paths=$transform_script},transform-interpreter{entry-point=__transform_main})" \
    -o "$out_dir/${name}_vector.mlir"
  if ! grep -q "scf.for" "$out_dir/${name}_vector.mlir"; then
    echo "error: expected scf.for (outer tiling loops) in ${name}_vector.mlir -- tiling did not produce real loops" >&2
    exit 1
  fi
  if ! grep -qE "vector\.(contract|fma|transfer_read|transfer_write|broadcast|splat)" "$out_dir/${name}_vector.mlir"; then
    echo "error: expected at least one vector.{contract,fma,transfer_read,transfer_write,broadcast,splat} in ${name}_vector.mlir" >&2
    exit 1
  fi

  # 2. Bounded vector-width test: no vector type in this intermediate may
  #    exceed 64 FP32 lanes (the direct opposite of whole-shape
  #    vectorization, where e.g. 32x32x32 produces vector<32x32xf32> ==
  #    1024 lanes). Checks both 1-D (vector<Nxf32>) and 2-D
  #    (vector<AxBxf32>) forms.
  local bad_vec
  bad_vec="$(grep -oE 'vector<[0-9]+(x[0-9]+)*xf32>' "$out_dir/${name}_vector.mlir" | sort -u | while read -r v; do
    dims="$(echo "$v" | grep -oE '[0-9]+' | head -n -1)"
    [[ -z "$dims" ]] && dims="$(echo "$v" | grep -oE '[0-9]+')"
    lanes=1
    for d in $dims; do lanes=$((lanes * d)); done
    if [[ "$lanes" -gt 64 ]]; then echo "$v (lanes=$lanes)"; fi
  done)"
  if [[ -n "$bad_vec" ]]; then
    echo "error: found vector type(s) exceeding 64 FP32 lanes in ${name}_vector.mlir (whole-shape vectorization regression):" >&2
    echo "$bad_vec" >&2
    exit 1
  fi

  # 3. Full pipeline (reuses the exact reproducible script).
  MLIR_BIN="$(dirname "$MLIR_OPT")" PLUGIN="$PLUGIN" \
    bash "$REPO_ROOT/mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh" \
    --variant tiled-vectorized "$input" "$out_dir" "$name"

  # 4. LLVM vector IR structural test: real (narrow) LLVM vector type.
  if ! grep -qE "<[0-9]+ x float>" "$out_dir/${name}.ll"; then
    echo "error: expected an LLVM vector type (e.g. '<N x float>') in ${name}.ll -- tiled vectorization silently scalarized" >&2
    exit 1
  fi

  # 5. NEON/FMLA assembly test.
  if ! grep -qE '^\s+fmla\s+v[0-9]+\.4s' "$out_dir/${name}.s"; then
    echo "error: expected 'fmla v*.4s' in ${name}.s -- tiled build did not select fused NEON FMLA" >&2
    exit 1
  fi

  # 6. Real loop branches present: at least one conditional branch
  #    (b.<cond>/cbz/cbnz), distinguishing this from the fully-unrolled
  #    vectorized variant's ~2 total branches (function entry/exit only).
  local cond_branches
  cond_branches="$(grep -cE '^\s+(b\.[a-z]+|cbz|cbnz)\s' "$out_dir/${name}.s" || true)"
  if [[ "$cond_branches" -lt 1 ]]; then
    echo "error: expected at least one conditional branch (loop) in ${name}.s, found $cond_branches" >&2
    exit 1
  fi

  # 7. ABI/exported-symbol checks, same convention as the other variants.
  if ! grep -q "llvm.func @${name}(" "$out_dir/${name}_llvm.mlir"; then
    echo "error: expected llvm.func @${name} not found in ${name}_llvm.mlir" >&2
    exit 1
  fi
  if ! grep -q "llvm.func @_mlir_ciface_${name}(" "$out_dir/${name}_llvm.mlir"; then
    echo "error: expected llvm.func @_mlir_ciface_${name} not found in ${name}_llvm.mlir" >&2
    exit 1
  fi
  local machine
  machine="$(llvm-readobj -h "$out_dir/${name}.o" | grep -o 'EM_AARCH64' || true)"
  if [[ "$machine" != "EM_AARCH64" ]]; then
    echo "error: ${name}.o is not an EM_AARCH64 object" >&2
    exit 1
  fi

  # 8. Code-size regression gate: this slice's whole reason for existing.
  #    32x32x32 must stay well under the fully-unrolled variant's 113,216
  #    bytes -- gated at 20,000 bytes (the primary achievement target).
  if [[ "$shape" == "32x32x32" ]]; then
    local obj_size
    obj_size="$(stat -c%s "$out_dir/${name}.o")"
    if [[ "$obj_size" -ge 20000 ]]; then
      echo "error: ${name}.o is $obj_size bytes, expected < 20000 (code-size gate regression)" >&2
      exit 1
    fi
  fi

  rm -rf "$out_dir"
}

for shape in 8x8x8 16x16x16 32x32x32 64x64x64 32x64x32 64x32x64; do
  run_tracked "backend_codegen/tiled_${shape}" \
    run_backend_codegen_tiled_static_checks "$shape"
done

# ---------------------------------------------------------------------------
# Host-side static checks for the PARAMETERIZED tiled-vectorized variant
# (compile_hir_matmul_bias_relu_aarch64.sh --variant tiled-vectorized
# --tile-m/--tile-n/--tile-k), added for the tile-candidate selection slice.
# Requires no network access and no Raspberry Pi. Exercises a sample of
# non-default tile candidates (not all 42 -- see
# artifacts/backend_codegen/aarch64_matmul_bias_relu_tile_candidates/
# candidate_results.json for the full evaluated set) to confirm the
# parameterized transform template produces the same class of evidence
# (bounded vector width, real FMLA, real loop branches) as the fixed 4x8x8
# tile already covered by run_backend_codegen_tiled_static_checks above.
# ---------------------------------------------------------------------------
run_backend_codegen_tile_candidate_static_checks() {
  local shape="$1" tm="$2" tn="$3" tk="$4"
  local input="$REPO_ROOT/mlir_passes/test/backend_codegen/matmul_bias_relu_tiled_${shape}.mlir"
  local out_dir
  out_dir="$(mktemp -d)"
  local name="matmul_bias_relu_tiled_${shape}_tm${tm}_tn${tn}_tk${tk}"
  local transform_template="$REPO_ROOT/mlir_passes/transforms/tile_vectorize_matmul_bias_relu.template.mlir"
  local transform_instance="$out_dir/transform.mlir"

  echo "[MLIR test] tile candidate static checks (shape=$shape tile=${tm}x${tn}x${tk})"

  bash "$REPO_ROOT/mlir_passes/tools/generate_tiled_transform.sh" \
    --tile-m "$tm" --tile-n "$tn" --tile-k "$tk" --output "$transform_instance" >/dev/null

  # 1. Tiled vector IR structural test (pre-bufferization): real scf.for
  #    loops plus vector transfer/fma ops, bounded vector width.
  "$MLIR_OPT" "$input" \
    --load-dialect-plugin="$PLUGIN" \
    --load-pass-plugin="$PLUGIN" \
    --pass-pipeline="builtin.module(hir-structured-lowering,transform-preload-library{transform-library-paths=$transform_instance},transform-interpreter{entry-point=__transform_main})" \
    -o "$out_dir/${name}_vector.mlir"
  if ! grep -q "scf.for" "$out_dir/${name}_vector.mlir"; then
    echo "error: expected scf.for in ${name}_vector.mlir" >&2
    exit 1
  fi
  if ! grep -qE "vector\.(contract|fma|transfer_read|transfer_write)" "$out_dir/${name}_vector.mlir"; then
    echo "error: expected vector.{contract,fma,transfer_read,transfer_write} in ${name}_vector.mlir" >&2
    exit 1
  fi
  local bad_vec
  bad_vec="$(grep -oE 'vector<[0-9]+(x[0-9]+)*xf32>' "$out_dir/${name}_vector.mlir" | sort -u | while read -r v; do
    dims="$(echo "$v" | grep -oE '[0-9]+' | head -n -1)"
    [[ -z "$dims" ]] && dims="$(echo "$v" | grep -oE '[0-9]+')"
    lanes=1
    for d in $dims; do lanes=$((lanes * d)); done
    if [[ "$lanes" -gt 64 ]]; then echo "$v (lanes=$lanes)"; fi
  done)"
  if [[ -n "$bad_vec" ]]; then
    echo "error: found vector type(s) exceeding 64 FP32 lanes in ${name}_vector.mlir:" >&2
    echo "$bad_vec" >&2
    exit 1
  fi

  # 2. Full pipeline via the parameterized compile-script interface.
  MLIR_BIN="$(dirname "$MLIR_OPT")" PLUGIN="$PLUGIN" \
    bash "$REPO_ROOT/mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh" \
    --variant tiled-vectorized --tile-m "$tm" --tile-n "$tn" --tile-k "$tk" \
    "$input" "$out_dir" "$name"

  # 3. Real NEON FMLA.
  if ! grep -qE '^\s+fmla\s+v[0-9]+\.4s' "$out_dir/${name}.s"; then
    echo "error: expected 'fmla v*.4s' in ${name}.s" >&2
    exit 1
  fi

  # 4. Real loop branch present (at least one conditional branch/back-edge).
  local cond_branches
  cond_branches="$(grep -cE '^\s+(b\.[a-z]+|cbz|cbnz)\s' "$out_dir/${name}.s" || true)"
  if [[ "$cond_branches" -lt 1 ]]; then
    echo "error: expected at least one conditional branch in ${name}.s, found $cond_branches" >&2
    exit 1
  fi

  rm -rf "$out_dir"
}

# Sample of non-default candidates: the two tiles this slice's scoring
# policy actually selected for at least one shape (4x8x8, 8x8x8) plus one
# never-selected candidate (4x4x4), each on a distinct shape, to keep this
# host-suite addition fast while still exercising the parameterized path
# beyond the already-covered fixed 4x8x8 default.
run_tracked "backend_codegen/tile_candidate_32x32x32_8x8x8" \
  run_backend_codegen_tile_candidate_static_checks "32x32x32" 8 8 8
run_tracked "backend_codegen/tile_candidate_64x64x64_4x8x8" \
  run_backend_codegen_tile_candidate_static_checks "64x64x64" 4 8 8
run_tracked "backend_codegen/tile_candidate_16x16x16_4x4x4" \
  run_backend_codegen_tile_candidate_static_checks "16x16x16" 4 4 4

echo
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "Total: $((PASSED + FAILED))"
if [[ "$FAILED" -gt 0 ]]; then
  echo "Failed files:"
  printf '  %s\n' "${FAILED_FILES[@]}"
  exit 1
fi
