#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-metal-rmsnorm-e2e}"
PLUGIN="${PLUGIN:-$REPO_ROOT/build-mlir/HIRMatMulBiasReluFusionPass.dylib}"

cmake -S "$REPO_ROOT" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --target benchmark_metal_rmsnorm run_metal_rmsnorm_plan

"$BUILD_DIR/benchmark_metal_rmsnorm"
PLUGIN="$PLUGIN" "$REPO_ROOT/tools/run_metal_rmsnorm_compiler_pipeline.sh"
"$BUILD_DIR/run_metal_rmsnorm_plan"

(
  cd "$REPO_ROOT"
  python3 tools/validate_metal_rmsnorm_path.py
)
