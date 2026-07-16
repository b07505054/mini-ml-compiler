#!/usr/bin/env bash
#
# run_backend_codegen_pi_integration.sh
#
# HARDWARE INTEGRATION TEST -- requires SSH access to the real Raspberry Pi
# target. This is intentionally kept separate from tools/run_mlir_pass_tests.sh
# (which is host-only and requires no network access / real device).
#
# End-to-end validation of the project's first native AArch64 code
# generation path:
#
#   HIR MLIR -> LLVM dialect -> LLVM IR -> AArch64 object (dev host)
#     -> scp to Raspberry Pi -> g++ link with a correctness/benchmark harness
#     -> execute on the real Raspberry Pi -> correctness + latency evidence
#
# For each of three shapes (8x8x8, 16x16x16, 32x32x32), this script:
#   1. Rebuilds the AArch64 object via compile_hir_matmul_bias_relu_aarch64.sh
#   2. Transfers it to the Raspberry Pi (does NOT install anything there)
#   3. Links it there against the harness, in its own process
#      (see aarch64_matmul_bias_relu_harness.cpp for why shapes are run as
#      separate process invocations rather than all three in one process)
#   4. Runs correctness + benchmark on the real device
#   5. Collects static backend metrics (LLVM IR / AArch64 instruction counts,
#      object size) on the dev host via llvm-objdump
#
# Usage:
#   run_backend_codegen_pi_integration.sh <output_dir>
#
# Environment overrides:
#   DEV_HOST   SSH target for the development host running this script's
#              relative paths against the repo. Default: run locally (assumes
#              this script itself already runs on the dev host).
#   PI_HOST    SSH target for the Raspberry Pi. Default: allen@100.110.37.6
#   ITERATIONS Timed iterations per shape. Default: 2000
#   WARMUP     Warmup iterations per shape. Default: 200

set -euo pipefail

usage() {
  echo "usage: $0 <output_dir>" >&2
  exit 1
}

[[ $# -ge 1 ]] || usage
OUTPUT_DIR="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PI_HOST="${PI_HOST:-allen@100.110.37.6}"
ITERATIONS="${ITERATIONS:-2000}"
WARMUP="${WARMUP:-200}"
SHAPES=(8x8x8 16x16x16 32x32x32)

mkdir -p "$OUTPUT_DIR"

echo "== [1/5] Rebuild AArch64 objects for all shapes =="
for shape in "${SHAPES[@]}"; do
  bash "$REPO_ROOT/mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh" \
    "$REPO_ROOT/mlir_passes/test/backend_codegen/matmul_bias_relu_${shape}.mlir" \
    "$OUTPUT_DIR" \
    "matmul_bias_relu_${shape}"
done

echo "== [2/5] Compile harness object (host-side, x86_64, for reference only -- not executed here) =="
# The harness itself is only ever executed on the Pi (it links against
# AArch64 objects); this repo does not attempt to run AArch64 code on the
# x86_64 dev host.

echo "== [3/5] Transfer objects + harness to the Raspberry Pi =="
PI_DIR="/tmp/backend_codegen_$(date +%s)"
ssh "$PI_HOST" "mkdir -p '$PI_DIR'"
for shape in "${SHAPES[@]}"; do
  scp -q "$OUTPUT_DIR/matmul_bias_relu_${shape}.o" "$PI_HOST:$PI_DIR/"
done
scp -q "$REPO_ROOT/mlir_passes/tools/aarch64_matmul_bias_relu_harness.cpp" "$PI_HOST:$PI_DIR/"

echo "== [4/5] Build + run on the Raspberry Pi (record device state, correctness, benchmark) =="
ssh "$PI_HOST" "hostname; uname -a; lscpu; gcc --version | head -1" > "$OUTPUT_DIR/pi_device_state.txt"

ssh "$PI_HOST" "cd '$PI_DIR' && \
  g++ -O2 -c aarch64_matmul_bias_relu_harness.cpp -o harness.o && \
  g++ -O2 harness.o matmul_bias_relu_8x8x8.o matmul_bias_relu_16x16x16.o matmul_bias_relu_32x32x32.o -o generated_kernel_benchmark && \
  file generated_kernel_benchmark" > "$OUTPUT_DIR/pi_build_log.txt"

# Each shape runs as its own process (see aarch64_matmul_bias_relu_harness.cpp
# header comment for why: process-per-shape isolation was required to avoid
# a cross-call heap-state interaction found during this slice's validation).
SHAPE_JSON_FILES=()
for shape in "${SHAPES[@]}"; do
  OUT_JSON="$OUTPUT_DIR/pi_result_${shape}.json"
  ssh "$PI_HOST" "cd '$PI_DIR' && ./generated_kernel_benchmark $ITERATIONS $WARMUP $shape" > "$OUT_JSON"
  SHAPE_JSON_FILES+=("$OUT_JSON")
done

python3 "$SCRIPT_DIR/merge_backend_codegen_shape_results.py" \
  "${SHAPE_JSON_FILES[@]}" \
  --out "$OUTPUT_DIR/benchmark_results.json" \
  --correctness-out "$OUTPUT_DIR/correctness_results.json"

echo "== [5/5] Static backend metrics (dev host, llvm-objdump cross-disassembly) =="
python3 "$SCRIPT_DIR/collect_backend_codegen_static_metrics.py" \
  --output-dir "$OUTPUT_DIR" \
  --shapes "${SHAPES[@]}" \
  > "$OUTPUT_DIR/backend_metrics.json"

echo
echo "done. Artifacts in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR"
