#!/usr/bin/env bash
#
# compile_hir_matmul_bias_relu_aarch64.sh
#
# Reproducible native-codegen pipeline for hir.fused_matmul_bias_relu:
#
#   Input HIR MLIR
#     -> mlir-opt (hir-matmul-bias-relu-to-linalg + stock MLIR->LLVM dialect
#                  conversion passes, the exact pipeline verified by
#                  mlir_passes/test/hir_matmul_bias_relu_to_llvm.mlir)
#     -> LLVM dialect MLIR
#     -> mlir-translate --mlir-to-llvmir
#     -> LLVM IR (.ll)
#     -> llc (AArch64 assembly + object)
#
# This script only reuses the existing, FileCheck-verified
# hir-matmul-bias-relu-to-linalg pass (mlir_passes/lib/MatMulBiasReluFusionPass.cpp)
# and stock upstream MLIR conversion passes already present in the plugin.
# It adds no project-owned target-specific instruction selection, scheduling,
# or register allocation -- this is the generic LLVM AArch64 backend path.
#
# Usage:
#   compile_hir_matmul_bias_relu_aarch64.sh <input.mlir> <output_dir> [artifact_name]
#
# Environment overrides:
#   MLIR_BIN       Directory containing mlir-opt / mlir-translate.
#                  Default: the project-local MLIR 21 toolchain used to build
#                  mlir_passes/ (see mlir_passes/README.md / build-mlir/CMakeCache.txt).
#   PLUGIN         Path to libHIRMatMulBiasReluFusionPass.so.
#                  Default: <repo>/build-mlir/libHIRMatMulBiasReluFusionPass.so
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
  echo "usage: $0 <input.mlir> <output_dir> [artifact_name]" >&2
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

mkdir -p "$OUTPUT_DIR"

KERNEL_LLVM_MLIR="$OUTPUT_DIR/${NAME}_llvm.mlir"
KERNEL_LL="$OUTPUT_DIR/${NAME}.ll"
KERNEL_S="$OUTPUT_DIR/${NAME}.s"
KERNEL_O="$OUTPUT_DIR/${NAME}.o"

# Identical pass list to mlir_passes/test/hir_matmul_bias_relu_to_llvm.mlir,
# with two additions required specifically for standalone execution (that
# FileCheck test only ever inspects static IR text -- it never runs the
# result, so it never needed these):
#
#   1. `buffer-deallocation-pipeline`, inserted right after bufferization.
#      Without it, the intermediate matmul-only buffer allocated inside the
#      lowered function (before the bias-add+relu stage consumes it) is never
#      freed -- a real per-call memory leak. This was found during this
#      slice's own hardware validation: repeated invocation under
#      Raspberry Pi correctness/benchmark testing produced incorrect output
#      on larger shapes after enough calls, traced (via a from-scratch
#      malloc/free-count check on the emitted LLVM IR) to this missing pass.
#      The function's *returned* buffer is correctly left un-freed by this
#      pass, since its ownership transfers to the caller.
#
#   2. The input function still carries `llvm.emit_c_interface`, so
#      convert-func-to-llvm additionally emits a `_mlir_ciface_<fn>` wrapper.
#      That wrapper is a stock MLIR mechanism, self-contained in the emitted
#      IR -- it requires no external MLIR runtime library (e.g.
#      mlir_c_runner_utils) to link or run.
PASS_PIPELINE='builtin.module(hir-matmul-bias-relu-to-linalg,one-shot-bufferize{bufferize-function-boundaries},buffer-deallocation-pipeline,convert-linalg-to-loops,convert-scf-to-cf,convert-index-to-llvm,convert-math-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)'

echo "[1/4] mlir-opt: HIR -> LLVM dialect ($NAME)"
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
