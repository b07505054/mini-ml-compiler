#!/usr/bin/env bash
#
# compile_hir_matmul_bias_relu_aarch64.sh
#
# Reproducible native-codegen pipeline for hir.fused_matmul_bias_relu, with
# three independently reproducible variants:
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
#          instruction selection. NO tiling: the whole static M/N/K shape
#          becomes one vector.contract, which is why static code size (and
#          static FMLA count) scales with M*N*K -- see
#          artifacts/backend_codegen/aarch64_matmul_bias_relu_tiled/README.md
#          for why this does not scale to larger shapes.
#       -> stock MLIR->LLVM dialect conversion (convert-vector-to-llvm with
#          vector-contract-lowering=outerproduct, which is what causes LLVM's
#          own AArch64 backend to select fused NEON FMLA instructions --
#          that final selection step is LLVM-owned, not implemented here)
#       -> LLVM dialect MLIR -> mlir-translate -> LLVM IR -> llc -> AArch64
#          assembly/object containing real NEON vector instructions.
#
#   --variant tiled-vectorized [--tile-m M] [--tile-n N] [--tile-k K]
#     Input HIR MLIR
#       -> mlir-opt (hir-matmul-bias-relu-to-linalg, UNCHANGED)
#       -> mlir_passes/tools/generate_tiled_transform.sh instantiates
#          mlir_passes/transforms/tile_vectorize_matmul_bias_relu.template.mlir
#          with the requested --tile-m/--tile-n/--tile-k (default 4/8/8,
#          matching the original fixed-tile slice byte-for-byte -- see
#          artifacts/backend_codegen/aarch64_matmul_bias_relu_tile_candidates/README.md
#          for why a template+substitution approach was chosen over
#          committing one Transform file per candidate tile)
#       -> mlir-opt applies that generated Transform-dialect script, which
#          tiles the matmul/bias/relu chain into the requested MxNxK
#          register-tile microkernel using STOCK Transform-dialect ops
#          (transform.structured.tile_using_for + fuse_into_containing_op),
#          then vectorizes only the small tiled ops -- producing three nested
#          scf.for loops (M, N, K) around a fixed-size vector.contract
#          (vector<4x8xf32>, vector<8x8xf32> operands, vector<4x8xf32>
#          accumulator; largest vector type after full lowering is <8 x
#          float>, i.e. one native 128-bit NEON register). Outer loops
#          remain in the generated code; only the microkernel body is
#          vectorized. Same project-owned/LLVM-owned truth boundary as
#          --variant vectorized -- see the transform script's header comment
#          and artifacts/backend_codegen/aarch64_matmul_bias_relu_tiled/README.md.
#       -> stock MLIR->LLVM dialect conversion. This variant additionally
#          requires convert-vector-to-scf{full-unroll target-rank=1} before
#          convert-vector-to-llvm: the microkernel's accumulator tile
#          bufferizes to a memref.subview of the shared output buffer (a
#          real slice, not a fresh allocation -- one-shot-bufferize
#          correctly avoids an extra copy here), and convert-vector-to-llvm
#          only lowers 1-D vector transfers directly; convert-vector-to-scf
#          statically unrolls the small (<=8-row) N-D transfers on that
#          subview into a sequence of 1-D transfers instead. (The whole-shape
#          `vectorized` variant does not need this: its transfers are on a
#          fully contiguous, non-sliced memref, which
#          test-vector-transfer-flatten-patterns can flatten directly.)
#          Each stage that introduces `affine.apply` (tiling, and
#          convert-vector-to-scf's index arithmetic) is followed by
#          `lower-affine` before the next stage, since nothing later in the
#          pipeline understands the affine dialect.
#       -> LLVM dialect MLIR -> mlir-translate -> LLVM IR -> llc -> AArch64
#          assembly/object: a compact, reusable microkernel body (real NEON
#          `fmla`) invoked repeatedly by real loop branches, instead of one
#          giant unrolled instruction stream.
#
#     Legality: tiled-vectorized requires M%TILE_M==0, N%TILE_N==0,
#     K%TILE_K==0 for the requested tile (default 4/8/8, the tile chosen in
#     the original single-tile slice -- see
#     artifacts/backend_codegen/aarch64_matmul_bias_relu_tiled/README.md for
#     the register-pressure analysis behind that specific choice; other
#     tiles are evaluated on their own evidence by the tile-candidate slice,
#     see artifacts/backend_codegen/aarch64_matmul_bias_relu_tile_candidates/README.md).
#     Shapes that do not divide evenly are REJECTED with a nonzero exit and
#     an explicit error message (no tail handling) -- use --variant generic
#     or vectorized for such shapes instead.
#
#   --variant tiled-scheduled [--tile-m M] [--tile-n N] [--tile-k K] [--schedule-unroll-k F]
#     Machine-scheduling analysis slice (see
#     artifacts/backend_codegen/aarch64_matmul_bias_relu_scheduling/README.md).
#     Identical to --variant tiled-vectorized in every respect (same
#     legality rule, same downstream lowering pipeline) EXCEPT the
#     Transform-dialect script additionally applies stock
#     `transform.loop.unroll` (factor F, default 1 -- a verified true
#     no-op producing byte-identical .text to tiled-vectorized) to the
#     K-reduction scf.for loop, right after K-tiling and before
#     vectorization. This is Option A ("K-loop unroll and interleave")
#     from that slice's Transformation Design Gate: halving the K-loop's
#     dynamic trip count (for F=2) gives LLVM's own machine scheduler a
#     larger static loop body -- more independent per-static-body
#     accumulator-chain material -- to interleave, without any
#     project-owned instruction reordering or custom scheduling logic.
#     Uses mlir_passes/transforms/tile_schedule_matmul_bias_relu.template.mlir
#     and mlir_passes/tools/generate_scheduled_transform.sh (siblings of
#     the tiled-vectorized template/generator, not the same files -- kept
#     separate so tiled-vectorized's own output can never be affected by
#     this variant's existence).
#
# No variant hand-writes NEON intrinsics anywhere in this script or in the
# C++ harnesses that call the resulting objects -- all are compiler output,
# not handwritten kernels.
#
# Usage:
#   compile_hir_matmul_bias_relu_aarch64.sh \
#     [--variant generic|vectorized|tiled-vectorized] \
#     [--tile-m M --tile-n N --tile-k K]  (tiled-vectorized only; default 4 8 8) \
#     <input.mlir> <output_dir> [artifact_name]
#
# Environment overrides:
#   MLIR_BIN       Directory containing mlir-opt / mlir-translate.
#                  Default: the project-local MLIR 21 toolchain used to build
#                  mlir_passes/ (see mlir_passes/README.md / build-mlir/CMakeCache.txt).
#   PLUGIN         Path to libHIRMatMulBiasReluFusionPass.so.
#                  Default: <repo>/build-mlir/libHIRMatMulBiasReluFusionPass.so
#   TRANSFORM_SCRIPT  Path to the whole-shape vectorization Transform-dialect
#                  script. Default: <repo>/mlir_passes/transforms/vectorize_matmul_bias_relu.mlir
#                  (only read when --variant vectorized).
#   TILE_TRANSFORM_TEMPLATE  Path to the parameterized tiled-microkernel
#                  Transform-dialect template. Default:
#                  <repo>/mlir_passes/transforms/tile_vectorize_matmul_bias_relu.template.mlir
#                  (only read when --variant tiled-vectorized). A concrete
#                  instance for the requested --tile-m/--tile-n/--tile-k is
#                  generated into a scratch temp file via
#                  mlir_passes/tools/generate_tiled_transform.sh -- never
#                  written into the repository tree.
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
  echo "usage: $0 [--variant generic|vectorized|tiled-vectorized|tiled-scheduled] [--tile-m M --tile-n N --tile-k K] [--schedule-unroll-k F] <input.mlir> <output_dir> [artifact_name]" >&2
  exit 1
}

VARIANT="generic"
ARG_TILE_M=""
ARG_TILE_N=""
ARG_TILE_K=""
ARG_SCHEDULE_UNROLL_K=""
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
    --tile-m)
      ARG_TILE_M="$2"
      shift 2
      ;;
    --tile-n)
      ARG_TILE_N="$2"
      shift 2
      ;;
    --tile-k)
      ARG_TILE_K="$2"
      shift 2
      ;;
    --schedule-unroll-k)
      ARG_SCHEDULE_UNROLL_K="$2"
      shift 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

[[ "$VARIANT" == "generic" || "$VARIANT" == "vectorized" || "$VARIANT" == "tiled-vectorized" || "$VARIANT" == "tiled-scheduled" ]] || {
  echo "error: --variant must be 'generic', 'vectorized', 'tiled-vectorized', or 'tiled-scheduled' (got '$VARIANT')" >&2
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
TILE_TRANSFORM_TEMPLATE="${TILE_TRANSFORM_TEMPLATE:-$REPO_ROOT/mlir_passes/transforms/tile_vectorize_matmul_bias_relu.template.mlir}"
GENERATE_TILED_TRANSFORM="$REPO_ROOT/mlir_passes/tools/generate_tiled_transform.sh"
SCHEDULE_TRANSFORM_TEMPLATE="${SCHEDULE_TRANSFORM_TEMPLATE:-$REPO_ROOT/mlir_passes/transforms/tile_schedule_matmul_bias_relu.template.mlir}"
GENERATE_SCHEDULED_TRANSFORM="$REPO_ROOT/mlir_passes/tools/generate_scheduled_transform.sh"
LLC="${LLC:-llc}"
TARGET_TRIPLE="${TARGET_TRIPLE:-aarch64-linux-gnu}"
TARGET_CPU="${TARGET_CPU:-cortex-a76}"

# Microkernel tile size for --variant tiled-vectorized, overridable via
# --tile-m/--tile-n/--tile-k. Default 4/8/8 is the tile chosen (and
# register-pressure-justified) in the original single-tile slice -- see
# artifacts/backend_codegen/aarch64_matmul_bias_relu_tiled/README.md. This
# default is a documented backward-compatible choice: omitting the tile
# flags reproduces that slice's committed objects byte-for-byte (verified
# in artifacts/backend_codegen/aarch64_matmul_bias_relu_tile_candidates/README.md).
TILE_M="${ARG_TILE_M:-4}"
TILE_N="${ARG_TILE_N:-8}"
TILE_K="${ARG_TILE_K:-8}"
# K-loop unroll factor for --variant tiled-scheduled only. Default 1 is a
# verified true no-op (byte-identical .text to tiled-vectorized -- see
# artifacts/backend_codegen/aarch64_matmul_bias_relu_scheduling/README.md).
SCHEDULE_UNROLL_K="${ARG_SCHEDULE_UNROLL_K:-1}"

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
if [[ "$VARIANT" == "tiled-vectorized" || "$VARIANT" == "tiled-scheduled" ]]; then
  [[ -f "$TILE_TRANSFORM_TEMPLATE" ]] || { echo "error: tiled transform template not found at $TILE_TRANSFORM_TEMPLATE" >&2; exit 1; }
  [[ -x "$GENERATE_TILED_TRANSFORM" ]] || { echo "error: $GENERATE_TILED_TRANSFORM not found/executable" >&2; exit 1; }
  for name_val in "TILE_M:$TILE_M" "TILE_N:$TILE_N" "TILE_K:$TILE_K"; do
    val="${name_val##*:}"
    [[ "$val" =~ ^[0-9]+$ && "$val" -ge 1 ]] || { echo "error: ${name_val%%:*} must be a positive integer (got '$val')" >&2; exit 1; }
  done
  if [[ "$VARIANT" == "tiled-scheduled" ]]; then
    [[ -f "$SCHEDULE_TRANSFORM_TEMPLATE" ]] || { echo "error: scheduled transform template not found at $SCHEDULE_TRANSFORM_TEMPLATE" >&2; exit 1; }
    [[ -x "$GENERATE_SCHEDULED_TRANSFORM" ]] || { echo "error: $GENERATE_SCHEDULED_TRANSFORM not found/executable" >&2; exit 1; }
    [[ "$SCHEDULE_UNROLL_K" =~ ^[0-9]+$ && "$SCHEDULE_UNROLL_K" -ge 1 ]] || { echo "error: --schedule-unroll-k must be a positive integer (got '$SCHEDULE_UNROLL_K')" >&2; exit 1; }
  fi

  # Legality check: reject shapes that do not divide evenly by the
  # requested tile size (no tail handling -- see the header comment).
  # Parses M/K from the lhs arg's tensor<MxKxf32> and N from the rhs arg's
  # tensor<KxNxf32>, in argument order, from the raw HIR text.
  mapfile -t DIMS < <(grep -oE 'tensor<[0-9]+x[0-9]+xf32>' "$INPUT_MLIR" | head -2 | grep -oE '[0-9]+x[0-9]+')
  [[ "${#DIMS[@]}" -eq 2 ]] || { echo "error: could not parse M/N/K from $INPUT_MLIR for the tiled-vectorized legality check" >&2; exit 1; }
  SHAPE_M="${DIMS[0]%%x*}"
  SHAPE_K="${DIMS[0]##*x}"
  SHAPE_N="${DIMS[1]##*x}"
  if (( SHAPE_M % TILE_M != 0 || SHAPE_N % TILE_N != 0 || SHAPE_K % TILE_K != 0 )); then
    echo "error: --variant $VARIANT requires M%${TILE_M}==0, N%${TILE_N}==0, K%${TILE_K}==0" >&2
    echo "       got M=$SHAPE_M N=$SHAPE_N K=$SHAPE_K (from $INPUT_MLIR) -- no tail handling in this version" >&2
    echo "       use --variant generic or --variant vectorized for this shape instead" >&2
    exit 1
  fi
  if [[ "$VARIANT" == "tiled-scheduled" ]]; then
    # The K-loop's dynamic trip count (K/TILE_K) must itself be evenly
    # divisible by the unroll factor -- transform.loop.unroll silently
    # clamps an oversized factor down to the trip count, but does not
    # guarantee a clean result for a factor that divides neither the trip
    # count nor 1, so reject explicitly here instead of trusting that.
    K_TRIP_COUNT=$(( SHAPE_K / TILE_K ))
    if (( K_TRIP_COUNT % SCHEDULE_UNROLL_K != 0 )); then
      echo "error: --schedule-unroll-k=$SCHEDULE_UNROLL_K must evenly divide the K-loop trip count (K/TILE_K = $K_TRIP_COUNT)" >&2
      exit 1
    fi
  fi
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

# Tiled-vectorized variant: a concrete Transform-dialect script is
# generated on demand from mlir_passes/transforms/
# tile_vectorize_matmul_bias_relu.template.mlir for the requested
# TILE_M x TILE_N x TILE_K (see mlir_passes/tools/generate_tiled_transform.sh),
# written to a scratch temp file (cleaned up on exit, never committed), then
# tiles+fuses+vectorizes into that microkernel (see that template's header
# and the variant's own header comment above for the full pipeline
# explanation, including why lower-affine appears three times and why
# convert-vector-to-scf{full-unroll} is required here but not for the
# whole-shape vectorized variant).
TILED_PIPELINE=""
GENERATED_TRANSFORM_SCRIPT=""
if [[ "$VARIANT" == "tiled-vectorized" ]]; then
  GENERATED_TRANSFORM_SCRIPT="$(mktemp -t tile_transform_XXXXXX.mlir)"
  trap 'rm -f "$GENERATED_TRANSFORM_SCRIPT"' EXIT
  TEMPLATE="$TILE_TRANSFORM_TEMPLATE" bash "$GENERATE_TILED_TRANSFORM" \
    --tile-m "$TILE_M" --tile-n "$TILE_N" --tile-k "$TILE_K" \
    --output "$GENERATED_TRANSFORM_SCRIPT" >/dev/null
  TILED_PIPELINE="builtin.module(hir-matmul-bias-relu-to-linalg,transform-preload-library{transform-library-paths=$GENERATED_TRANSFORM_SCRIPT},transform-interpreter{entry-point=__transform_main},lower-affine,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},buffer-deallocation-pipeline,expand-strided-metadata,lower-affine,convert-vector-to-scf{full-unroll target-rank=1},lower-affine,convert-vector-to-llvm{vector-contract-lowering=outerproduct},convert-ub-to-llvm,convert-scf-to-cf,convert-index-to-llvm,convert-math-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)"
fi

# Tiled-scheduled variant: same tile-and-fuse structure as tiled-vectorized,
# generated from tile_schedule_matmul_bias_relu.template.mlir instead, which
# adds one transform.loop.unroll on the K-reduction loop (Stage 8/9/10 of the
# machine-scheduling analysis slice -- see that template's header comment for
# the full rationale). Downstream lowering stages are identical to
# TILED_PIPELINE; only the preloaded transform-library path differs.
TILED_SCHEDULED_PIPELINE=""
GENERATED_SCHEDULE_TRANSFORM_SCRIPT=""
if [[ "$VARIANT" == "tiled-scheduled" ]]; then
  GENERATED_SCHEDULE_TRANSFORM_SCRIPT="$(mktemp -t schedule_transform_XXXXXX.mlir)"
  trap 'rm -f "$GENERATED_SCHEDULE_TRANSFORM_SCRIPT"' EXIT
  TEMPLATE="$SCHEDULE_TRANSFORM_TEMPLATE" bash "$GENERATE_SCHEDULED_TRANSFORM" \
    --tile-m "$TILE_M" --tile-n "$TILE_N" --tile-k "$TILE_K" \
    --schedule-unroll-k "$SCHEDULE_UNROLL_K" \
    --output "$GENERATED_SCHEDULE_TRANSFORM_SCRIPT" >/dev/null
  TILED_SCHEDULED_PIPELINE="builtin.module(hir-matmul-bias-relu-to-linalg,transform-preload-library{transform-library-paths=$GENERATED_SCHEDULE_TRANSFORM_SCRIPT},transform-interpreter{entry-point=__transform_main},lower-affine,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},buffer-deallocation-pipeline,expand-strided-metadata,lower-affine,convert-vector-to-scf{full-unroll target-rank=1},lower-affine,convert-vector-to-llvm{vector-contract-lowering=outerproduct},convert-ub-to-llvm,convert-scf-to-cf,convert-index-to-llvm,convert-math-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)"
fi

if [[ "$VARIANT" == "generic" ]]; then
  PASS_PIPELINE="$GENERIC_PIPELINE"
elif [[ "$VARIANT" == "vectorized" ]]; then
  PASS_PIPELINE="$VECTORIZED_PIPELINE"
elif [[ "$VARIANT" == "tiled-scheduled" ]]; then
  PASS_PIPELINE="$TILED_SCHEDULED_PIPELINE"
  echo "[tile] TM=$TILE_M TN=$TILE_N TK=$TILE_K SCHEDULE_UNROLL_K=$SCHEDULE_UNROLL_K (transform script: $GENERATED_SCHEDULE_TRANSFORM_SCRIPT)"
else
  PASS_PIPELINE="$TILED_PIPELINE"
  echo "[tile] TM=$TILE_M TN=$TILE_N TK=$TILE_K (transform script: $GENERATED_TRANSFORM_SCRIPT)"
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
