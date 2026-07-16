#!/usr/bin/env bash
#
# run_backend_codegen_vectorized_pi_integration.sh
#
# HARDWARE INTEGRATION TEST -- requires SSH access to the real Raspberry Pi
# target. Separate from tools/run_mlir_pass_tests.sh (host-only, no network).
#
# Validates the vectorized AArch64 backend slice end-to-end on real hardware:
#   1. Rebuilds generic + vectorized objects for all three shapes.
#   2. Transfers objects, test tools, and the handwritten-kernel dependency
#      sources to the Raspberry Pi.
#   3. Runs the repeated-call correctness test (1000 calls) for BOTH variants,
#      all three shapes.
#   4. Runs the mixed-shape/mixed-variant same-process stress test
#      (500 cycles = 6000 calls, generic+vectorized interleaved, with
#      unrelated allocator noise between calls).
#   5. Runs the 4-way benchmark (scalar / generic / vectorized / handwritten)
#      for all three shapes.
#
# All correctness testing here exercises the FIXED lowering (explicit
# linalg.fill zero-init of the matmul accumulator -- see
# mlir_passes/lib/MatMulBiasReluFusionPass.cpp and
# aarch64_matmul_bias_relu_repeated_call_test.cpp for the full history).
#
# Usage:
#   run_backend_codegen_vectorized_pi_integration.sh <output_dir>
#
# Environment overrides:
#   PI_HOST    SSH target for the Raspberry Pi. Default: allen@100.110.37.6
#   ITERATIONS Timed benchmark iterations per shape. Default: 2000
#   WARMUP     Warmup iterations per shape. Default: 200
#   REPEATED_CALLS  Calls per shape/variant for the repeated-call test. Default: 1000
#   MIXED_CYCLES    Cycles for the mixed-shape stress test. Default: 500

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
REPEATED_CALLS="${REPEATED_CALLS:-1000}"
MIXED_CYCLES="${MIXED_CYCLES:-500}"
SHAPES=(8x8x8 16x16x16 32x32x32)

mkdir -p "$OUTPUT_DIR"

echo "== [1/5] Rebuild generic + vectorized AArch64 objects for all shapes =="
for shape in "${SHAPES[@]}"; do
  bash "$REPO_ROOT/mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh" \
    --variant generic \
    "$REPO_ROOT/mlir_passes/test/backend_codegen/matmul_bias_relu_${shape}.mlir" \
    "$OUTPUT_DIR" "matmul_bias_relu_${shape}"
  bash "$REPO_ROOT/mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh" \
    --variant vectorized \
    "$REPO_ROOT/mlir_passes/test/backend_codegen/matmul_bias_relu_vectorized_${shape}.mlir" \
    "$OUTPUT_DIR" "matmul_bias_relu_vectorized_${shape}"
done

echo "== [2/5] Transfer objects, test tools, and handwritten-kernel deps to the Raspberry Pi =="
PI_DIR="/tmp/backend_codegen_vectorized_$(date +%s)"
ssh "$PI_HOST" "mkdir -p '$PI_DIR/include/kernels' '$PI_DIR/include/ir' '$PI_DIR/include/runtime' '$PI_DIR/src'"

for shape in "${SHAPES[@]}"; do
  scp -q "$OUTPUT_DIR/matmul_bias_relu_${shape}.o" "$OUTPUT_DIR/matmul_bias_relu_vectorized_${shape}.o" "$PI_HOST:$PI_DIR/"
done
scp -q "$REPO_ROOT/mlir_passes/tools/aarch64_matmul_bias_relu_repeated_call_test.cpp" \
       "$REPO_ROOT/mlir_passes/tools/aarch64_matmul_bias_relu_mixed_shape_test.cpp" \
       "$REPO_ROOT/mlir_passes/tools/aarch64_matmul_bias_relu_vectorized_harness.cpp" \
       "$PI_HOST:$PI_DIR/"
scp -q "$REPO_ROOT/include/kernels/cpu_kernels.h" "$PI_HOST:$PI_DIR/include/kernels/"
scp -q "$REPO_ROOT/include/ir/tensor.h" "$REPO_ROOT/include/ir/quant_tensor.h" "$PI_HOST:$PI_DIR/include/ir/"
scp -q "$REPO_ROOT/include/runtime/thread_pool.h" "$PI_HOST:$PI_DIR/include/runtime/"
scp -q "$REPO_ROOT/src/kernels/cpu_kernels.cpp" "$REPO_ROOT/src/runtime/thread_pool.cpp" "$PI_HOST:$PI_DIR/src/"

echo "== [3/5] Build test binaries on the Raspberry Pi =="
ssh "$PI_HOST" "hostname; uname -a; lscpu; gcc --version | head -1" > "$OUTPUT_DIR/pi_device_state.txt"

ssh "$PI_HOST" "cd '$PI_DIR' && \
  g++ -O2 -std=c++17 -c aarch64_matmul_bias_relu_repeated_call_test.cpp -o repeated_call_test.o && \
  g++ -O2 repeated_call_test.o matmul_bias_relu_*.o -o repeated_call_test && \
  g++ -O2 -std=c++17 -c aarch64_matmul_bias_relu_mixed_shape_test.cpp -o mixed_shape_test.o && \
  g++ -O2 mixed_shape_test.o matmul_bias_relu_*.o -o mixed_shape_test && \
  g++ -O2 -std=c++17 -Iinclude -c aarch64_matmul_bias_relu_vectorized_harness.cpp -o vec_harness.o && \
  g++ -O2 -std=c++17 -Iinclude -c src/cpu_kernels.cpp -o cpu_kernels.o && \
  g++ -O2 -std=c++17 -Iinclude -c src/thread_pool.cpp -o thread_pool.o && \
  g++ -O2 vec_harness.o cpu_kernels.o thread_pool.o matmul_bias_relu_*.o -o vectorized_benchmark -lpthread" \
  > "$OUTPUT_DIR/pi_build_log.txt" 2>&1

echo "== [4/5] Repeated-call correctness (${REPEATED_CALLS} calls x 2 variants x 3 shapes) + mixed-shape stress (${MIXED_CYCLES} cycles) =="
REPEATED_LOG="$OUTPUT_DIR/repeated_call_results.txt"
: > "$REPEATED_LOG"
REPEATED_FAIL=0
for shape in "${SHAPES[@]}"; do
  for variant in generic vectorized; do
    if ! ssh "$PI_HOST" "cd '$PI_DIR' && ./repeated_call_test '$shape' '$REPEATED_CALLS' '$variant'" >> "$REPEATED_LOG" 2>&1; then
      REPEATED_FAIL=1
    fi
  done
done
cat "$REPEATED_LOG"

MIXED_LOG="$OUTPUT_DIR/mixed_shape_results.txt"
MIXED_FAIL=0
if ! ssh "$PI_HOST" "cd '$PI_DIR' && ./mixed_shape_test '$MIXED_CYCLES'" > "$MIXED_LOG" 2>&1; then
  MIXED_FAIL=1
fi
cat "$MIXED_LOG"

if [[ "$REPEATED_FAIL" -ne 0 || "$MIXED_FAIL" -ne 0 ]]; then
  echo "error: correctness stress tests failed -- see $REPEATED_LOG / $MIXED_LOG" >&2
  exit 1
fi

echo "== [5/5] 4-way benchmark (scalar / generic / vectorized / handwritten) =="
for shape in "${SHAPES[@]}"; do
  ssh "$PI_HOST" "cd '$PI_DIR' && ./vectorized_benchmark $ITERATIONS $WARMUP $shape" \
    > "$OUTPUT_DIR/benchmark_${shape}.json"
  cat "$OUTPUT_DIR/benchmark_${shape}.json"
done

echo
echo "done. Artifacts in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR"
