#!/usr/bin/env bash
#
# run_tile_candidates_pi_integration.sh
#
# HARDWARE INTEGRATION for the AArch64 tile-candidate selection slice.
# Transfers all compiled candidate objects (already built on the dev host
# by tools/generate_aarch64_matmul_tile_candidates.py) plus the generated
# harness sources (tools/generate_tile_candidate_harness.py) to the real
# Raspberry Pi, builds there (native aarch64 g++ -- cross-compiled .o
# objects cannot be linked on the x86_64 dev host), and runs:
#   1. Repeated-call correctness (1 call, then 1000 calls) with guard
#      checks, for every legal candidate.
#   2. Mixed-candidate stress (the 5-step sequence from the task brief,
#      >=200 cycles, allocator noise).
#   3. Per-shape benchmark (every candidate for that shape + scalar ref).
#
# Usage:
#   PI_HOST=allen@100.110.37.6 \
#   bash tools/run_tile_candidates_pi_integration.sh \
#     <objects_dir> <harness_dir> <candidates_json> <output_dir>

set -euo pipefail

[[ $# -ge 4 ]] || { echo "usage: $0 <objects_dir> <harness_dir> <candidates_json> <output_dir>" >&2; exit 1; }
OBJECTS_DIR="$1"
HARNESS_DIR="$2"
CANDIDATES_JSON="$3"
OUTPUT_DIR="$4"

PI_HOST="${PI_HOST:-allen@100.110.37.6}"
REPEATED_CALLS="${REPEATED_CALLS:-1000}"
MIXED_CYCLES="${MIXED_CYCLES:-200}"
ITERATIONS="${ITERATIONS:-2000}"
ITERATIONS_LARGE="${ITERATIONS_LARGE:-500}"
WARMUP="${WARMUP:-200}"

mkdir -p "$OUTPUT_DIR"

echo "== [1/4] Transfer candidate objects + harness sources to the Pi =="
PI_DIR="/tmp/tile_candidates_$(date +%s)"
ssh "$PI_HOST" "mkdir -p '$PI_DIR'"
scp -q "$OBJECTS_DIR"/*.o "$PI_HOST:$PI_DIR/"
scp -q "$HARNESS_DIR"/*.cpp "$PI_HOST:$PI_DIR/"

echo "== [2/4] Build on the Raspberry Pi (native g++) =="
ssh "$PI_HOST" "hostname; uname -a; lscpu; gcc --version | head -1" > "$OUTPUT_DIR/pi_device_state.txt"
ssh "$PI_HOST" "cd '$PI_DIR' && \
  OBJS=\$(ls matmul_*.o) && \
  g++ -O2 -std=c++17 -c aarch64_matmul_tile_candidate_repeated_call_test.cpp -o repeated_call_test.o && \
  g++ -O2 repeated_call_test.o \$OBJS -o repeated_call_test && \
  g++ -O2 -std=c++17 -c aarch64_matmul_tile_candidate_mixed_stress_test.cpp -o mixed_stress_test.o && \
  g++ -O2 mixed_stress_test.o \$OBJS -o mixed_stress_test && \
  g++ -O2 -std=c++17 -c aarch64_matmul_tile_candidate_benchmark.cpp -o benchmark.o && \
  g++ -O2 benchmark.o \$OBJS -o benchmark" \
  > "$OUTPUT_DIR/pi_build_log.txt" 2>&1
echo "build log:"; cat "$OUTPUT_DIR/pi_build_log.txt"

echo "== [3/4] Repeated-call correctness (1 call then ${REPEATED_CALLS} calls, ALL candidates, one process) + mixed stress (${MIXED_CYCLES} cycles) =="
REPEATED_LOG="$OUTPUT_DIR/repeated_call_results.txt"
REPEATED_FAIL=0
if ! ssh "$PI_HOST" "cd '$PI_DIR' && ./repeated_call_test '$REPEATED_CALLS'" > "$REPEATED_LOG" 2>&1; then
  REPEATED_FAIL=1
fi
cat "$REPEATED_LOG"

MIXED_LOG="$OUTPUT_DIR/mixed_stress_results.txt"
MIXED_FAIL=0
if ! ssh "$PI_HOST" "cd '$PI_DIR' && ./mixed_stress_test '$MIXED_CYCLES'" > "$MIXED_LOG" 2>&1; then
  MIXED_FAIL=1
fi
cat "$MIXED_LOG"

if [[ "$REPEATED_FAIL" -ne 0 || "$MIXED_FAIL" -ne 0 ]]; then
  echo "error: correctness stress tests failed -- see $REPEATED_LOG / $MIXED_LOG" >&2
  exit 1
fi

echo "== [4/4] Per-shape benchmark (all candidates for that shape + scalar) =="
for shape_key in 16x16x16 32x32x32 64x64x64 32x64x32 64x32x64 8x8x8; do
  iters="$ITERATIONS"
  case "$shape_key" in
    64x64x64|32x64x32|64x32x64) iters="$ITERATIONS_LARGE" ;;
  esac
  ssh "$PI_HOST" "cd '$PI_DIR' && ./benchmark '$shape_key' '$iters' '$WARMUP'" > "$OUTPUT_DIR/benchmark_${shape_key}.json"
  cat "$OUTPUT_DIR/benchmark_${shape_key}.json"
done

echo
echo "done. Artifacts in $OUTPUT_DIR"
