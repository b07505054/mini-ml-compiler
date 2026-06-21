#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"

cmake -S "$REPO_ROOT" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR"

PLAN="$REPO_ROOT/trace/metal_rmsnorm_execution_plan.json"
if [[ -f "$PLAN" ]]; then
  echo "check.sh: found $PLAN; metal_rmsnorm_plan_dispatch will run for real."
else
  echo "check.sh: $PLAN not found; metal_rmsnorm_plan_dispatch will report as skipped." \
       "Run tools/run_metal_rmsnorm_compiler_pipeline.sh (requires an MLIR build) to generate it."
fi

ctest --test-dir "$BUILD_DIR" --output-on-failure
