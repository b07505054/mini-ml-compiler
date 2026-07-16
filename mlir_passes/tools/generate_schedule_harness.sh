#!/usr/bin/env bash
#
# generate_schedule_harness.sh
#
# Substitutes the SHAPE_TOKEN placeholder in
# mlir_passes/tools/aarch64_matmul_bias_relu_schedule_harness.cpp.template
# with a concrete shape string (e.g. "32x32x32"), writing the result to a
# caller-specified scratch output path (NOT committed -- same convention as
# generate_tiled_transform.sh / generate_scheduled_transform.sh).
#
# Usage:
#   generate_schedule_harness.sh --shape 32x32x32 --output PATH
#
# Environment overrides:
#   TEMPLATE  Path to the template. Default:
#             <repo>/mlir_passes/tools/aarch64_matmul_bias_relu_schedule_harness.cpp.template

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEMPLATE="${TEMPLATE:-$REPO_ROOT/mlir_passes/tools/aarch64_matmul_bias_relu_schedule_harness.cpp.template}"

usage() {
  echo "usage: $0 --shape 32x32x32 --output PATH" >&2
  exit 1
}

SHAPE=""
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --shape) SHAPE="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    *) echo "error: unrecognized argument '$1'" >&2; usage ;;
  esac
done

[[ -n "$SHAPE" && -n "$OUTPUT" ]] || usage
[[ "$SHAPE" =~ ^[0-9]+x[0-9]+x[0-9]+$ ]] || { echo "error: --shape must look like MxNxK (got '$SHAPE')" >&2; exit 1; }
[[ -f "$TEMPLATE" ]] || { echo "error: template not found at $TEMPLATE" >&2; exit 1; }

mkdir -p "$(dirname "$OUTPUT")"
sed -e "s/SHAPE_TOKEN/${SHAPE}/g" "$TEMPLATE" > "$OUTPUT"
echo "$OUTPUT"
