#!/usr/bin/env bash
#
# compile_hir_matmul_bias_relu_aarch64.sh
#
# Reproducible native-codegen pipeline for hir.fused_matmul_bias_relu, with
# two independently reproducible variants:
#
#   --variant generic (default)
#     Input HIR MLIR
#       -> mlir-opt (hir-matmul-bias-relu-to-linalg + stock MLIR->LLVM dialect
#                    conversion passes, the exact pipeline verified by
#                    mlir_passes/test/hir_matmul_bias_relu_to_llvm.mlir)
#       -> LLVM dialect MLIR -> mlir-translate -> LLVM IR -> llc -> AArch64
#          assembly/object. No unrolling or vectorization; llc's default
#          scalar codegen from unmodified Linalg-derived loops.
#
#   --variant vectorized
#     Input HIR MLIR
#       -> mlir-opt (hir-matmul-bias-relu-to-linalg, UNCHANGED, same pass as
#                    the generic variant)
#       -> mlir-opt (project-owned Transform-dialect script,
#                    mlir_passes/transforms/vectorize_matmul_bias_relu.mlir,
#                    applied via transform-preload-library + transform-interpreter)
#          rewrites the tensor-level Linalg form into MLIR Vector-dialect ops
#          (vector.transfer_read / vector.contract / vector.transfer_write,
#          plus vectorized arith for the bias-add+ReLU stage). This is
#          project-owned instruction-selection PREPARATION, not machine
#          instruction selection.
#       -> stock MLIR->LLVM dialect conversion (convert-vector-to-llvm with
#          vector-contract-lowering=outerproduct, which is what causes LLVM's
#          own AArch64 backend to select fused NEON FMLA instructions --
#          that final selection step is LLVM-owned, not implemented here)
#       -> LLVM dialect MLIR -> mlir-translate -> LLVM IR -> llc -> AArch64
#          assembly/object containing real NEON vector instructions.
#
# Neither variant hand-writes NEON intrinsics anywhere in this script or in
# the C++ harness that calls the resulting objects -- both are compiler
# output, not handwritten kernels.
#
# Usage:
#   compile_hir_matmul_bias_relu_aarch64.sh [--variant generic|vectorized] \
#     <input.mlir> <output_dir> [artifact_name]
#
# Environment overrides:
#   MLIR_BIN       Directory containing mlir-opt / mlir-translate.
#                  Default: the project-local MLIR 21 toolchain used to build
#                  mlir_passes/ (see mlir_passes/README.md / build-mlir/CMakeCache.txt).
#   PLUGIN         Path to libHIRMatMulBiasReluFusionPass.so.
#                  Default: <repo>/build-mlir/libHIRMatMulBiasReluFusionPass.so
#   TRANSFORM_SCRIPT  Path to the vectorization Transform-dialect script.
#                  Default: <repo>/mlir_passes/transforms/vectorize_matmul_bias_relu.mlir
#                  (only read when --variant vectorized).
#   LLC            llc binary. Default: llc on PATH (expected LLVM 21, matching MLIR_BIN).
#   TARGET_TRIPLE  Default: aarch64-linux-gnu
#   TARGET_CPU     Default: cortex-a76 (Raspberry Pi 5 CPU, confirmed via the
#                  backend-compiler audit's live `lscpu`/`/proc/cpuinfo` inspection
#                  of the real Raspberry Pi target).
#
# Outputs (written to <output_dir>, all prefixed with [artifact_name]):
#   <name>_llvm.mlir   LLVM dialect MLIR (mlir-opt output)
#   <name>.ll           Textual LLVM IR (mlir-translate output)
#   <name>.s             AArch64 assembly (llc -filetype=asm)
#   <name>.o             AArch64 ELF object (llc -filetype=obj)

set -euo pipefail

usage() {
  echo "usage: $0 [--variant generic|vectorized] <input.mlir> <output_dir> [artifact_name]" >&2
  exit 1
}

VARIANT="generic"
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --variant=*)
      VARIANT="${1#--variant=}"
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

[[ "$VARIANT" == "generic" || "$VARIANT" == "vectorized" ]] || {
  echo "error: --variant must be 'generic' or 'vectorized' (got '$VARIANT')" >&2
  exit 1
}

[[ $# -ge 2 ]] || usage

INPUT_MLIR="$1"
OUTPUT_DIR="$2"
NAME="${3:-$(basename "${INPUT_MLIR%.mlir}")}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MLIR_BIN="${MLIR_BIN:-/home/allen/Desktop/Project/.deps/mlir21-root/usr/lib/llvm-21/bin}"
PLUGIN="${PLUGIN:-$REPO_ROOT/build-mlir/libHIRMatMulBiasReluFusionPass.so}"
TRANSFORM_SCRIPT="${TRANSFORM_SCRIPT:-$REPO_ROOT/mlir_passes/transforms/vectorize_matmul_bias_relu.mlir}"
LLC="${LLC:-llc}"
TARGET_TRIPLE="${TARGET_TRIPLE:-aarch64-linux-gnu}"
TARGET_CPU="${TARGET_CPU:-cortex-a76}"

MLIR_OPT="$MLIR_BIN/mlir-opt"
MLIR_TRANSLATE="$MLIR_BIN/mlir-translate"

[[ -f "$INPUT_MLIR" ]] || { echo "error: input MLIR not found: $INPUT_MLIR" >&2; exit 1; }
[[ -x "$MLIR_OPT" ]] || { echo "error: mlir-opt not found/executable at $MLIR_OPT" >&2; exit 1; }
[[ -x "$MLIR_TRANSLATE" ]] || { echo "error: mlir-translate not found/executable at $MLIR_TRANSLATE" >&2; exit 1; }
[[ -f "$PLUGIN" ]] || { echo "error: pass plugin not found at $PLUGIN (build mlir_passes first)" >&2; exit 1; }
command -v "$LLC" >/dev/null 2>&1 || { echo "error: llc not found on PATH" >&2; exit 1; }
if [[ "$VARIANT" == "vectorized" ]]; then
  [[ -f "$TRANSFORM_SCRIPT" ]] || { echo "error: vectorization transform script not found at $TRANSFORM_SCRIPT" >&2; exit 1; }
fi

mkdir -p "$OUTPUT_DIR"

KERNEL_LLVM_MLIR="$OUTPUT_DIR/${NAME}_llvm.mlir"
KERNEL_LL="$OUTPUT_DIR/${NAME}.ll"
KERNEL_S="$OUTPUT_DIR/${NAME}.s"
KERNEL_O="$OUTPUT_DIR/${NAME}.o"

# Generic variant: identical pass list to mlir_passes/test/hir_matmul_bias_relu_to_llvm.mlir,
# with two additions required specifically for standalone execution (that
# FileCheck test only ever inspects static IR text -- it never runs the
# result, so it never needed these):
#
#   1. `buffer-deallocation-pipeline`, inserted right after bufferization.
#      Without it, the intermediate matmul-only buffer allocated inside the
#      lowered function (before the bias-add+relu stage consumes it) is never
#      freed -- a real per-call memory leak, found and fixed during this
#      slice's own hardware validation. The function's *returned* buffer is
#      correctly left un-freed by this pass, since its ownership transfers
#      to the caller.
#
#   2. The input function carries `llvm.emit_c_interface`, so
#      convert-func-to-llvm additionally emits a `_mlir_ciface_<fn>` wrapper.
#      That wrapper is a stock MLIR mechanism, self-contained in the emitted
#      IR -- it requires no external MLIR runtime library (e.g.
#      mlir_c_runner_utils) to link or run.
GENERIC_PIPELINE='builtin.module(hir-matmul-bias-relu-to-linalg,one-shot-bufferize{bufferize-function-boundaries},buffer-deallocation-pipeline,convert-linalg-to-loops,convert-scf-to-cf,convert-index-to-llvm,convert-math-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)'

# Vectorized variant: same hir-matmul-bias-relu-to-linalg pass, then the
# project-owned Transform-dialect vectorization script (see
# mlir_passes/transforms/vectorize_matmul_bias_relu.mlir), then a
# vector-aware lowering path instead of convert-linalg-to-loops:
#
#   - function-boundary-type-conversion=identity-layout-map on
#     one-shot-bufferize: without this, argument memrefs get a fully dynamic
#     stride layout (needed in general for ABI genericity), which blocks
#     MLIR's vector-transfer lowering patterns from proving the innermost
#     dimension is unit-stride, so they refuse to lower 2-D vector transfers
#     into real vector loads. Forcing an identity (static, contiguous) layout
#     at the boundary resolves this. Verified this does NOT change the
#     _mlir_ciface_ ABI: the generated llvm.func / _mlir_ciface_ signatures
#     are byte-identical in shape to the generic variant's, so the same
#     harness code calls both variants unmodified.
#   - test-vector-transfer-flatten-patterns + expand-strided-metadata:
#     rewrite contiguous N-D vector.transfer ops into 1-D transfers (and
#     lower the memref.collapse_shape this introduces) before
#     convert-vector-to-llvm, which only handles 1-D vector transfers
#     directly.
#   - convert-vector-to-llvm{vector-contract-lowering=outerproduct}: lowers
#     vector.contract via vector.outerproduct, which is what causes LLVM's
#     AArch64 instruction selector to choose fused `fmla` (verified in
#     disassembly) instead of separate fmul/fadd.
#   - convert-ub-to-llvm: lowers the `ub.poison` padding-value placeholder
#     that vector.transfer_read's padding operand introduces.
VECTORIZED_PIPELINE="builtin.module(hir-matmul-bias-relu-to-linalg,transform-preload-library{transform-library-paths=$TRANSFORM_SCRIPT},transform-interpreter{entry-point=__transform_main},one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},buffer-deallocation-pipeline,func.func(test-vector-transfer-flatten-patterns),expand-strided-metadata,convert-vector-to-llvm{vector-contract-lowering=outerproduct},convert-ub-to-llvm,convert-scf-to-cf,convert-index-to-llvm,convert-math-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)"

if [[ "$VARIANT" == "generic" ]]; then
  PASS_PIPELINE="$GENERIC_PIPELINE"
else
  PASS_PIPELINE="$VECTORIZED_PIPELINE"
fi

echo "[1/4] mlir-opt: HIR -> LLVM dialect ($NAME, variant=$VARIANT)"
"$MLIR_OPT" "$INPUT_MLIR" \
  --load-dialect-plugin="$PLUGIN" \
  --load-pass-plugin="$PLUGIN" \
  --pass-pipeline="$PASS_PIPELINE" \
  -o "$KERNEL_LLVM_MLIR"

echo "[2/4] mlir-translate: LLVM dialect -> LLVM IR text"
"$MLIR_TRANSLATE" --mlir-to-llvmir "$KERNEL_LLVM_MLIR" -o "$KERNEL_LL"

echo "[3/4] llc: LLVM IR -> AArch64 assembly ($TARGET_TRIPLE, $TARGET_CPU)"
"$LLC" -mtriple="$TARGET_TRIPLE" -mcpu="$TARGET_CPU" -filetype=asm "$KERNEL_LL" -o "$KERNEL_S"

echo "[4/4] llc: LLVM IR -> AArch64 object"
"$LLC" -mtriple="$TARGET_TRIPLE" -mcpu="$TARGET_CPU" -filetype=obj "$KERNEL_LL" -o "$KERNEL_O"

echo "done:"
echo "  $KERNEL_LLVM_MLIR"
echo "  $KERNEL_LL"
echo "  $KERNEL_S"
echo "  $KERNEL_O"
