#!/usr/bin/env bash
#
# generate_tiled_transform.sh
#
# Substitutes TILE_M / TILE_N / TILE_K placeholders in
# mlir_passes/transforms/tile_vectorize_matmul_bias_relu.template.mlir with
# concrete positive integers, writing the result to a caller-specified
# output path (expected to be a scratch/temp location, e.g.
# /tmp/tile_candidates/... -- NOT committed; see the tile-candidate slice's
# README.md for why seven-plus near-identical Transform files are not
# checked into the repository).
#
# This is the "Option B: one parameterized template, substitute validated
# integer values" interface from the tile-candidate task brief.
#
# Usage:
#   generate_tiled_transform.sh --tile-m M --tile-n N --tile-k K --output PATH
#
# Environment overrides:
#   TEMPLATE  Path to the template. Default:
#             <repo>/mlir_passes/transforms/tile_vectorize_matmul_bias_relu.template.mlir

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEMPLATE="${TEMPLATE:-$REPO_ROOT/mlir_passes/transforms/tile_vectorize_matmul_bias_relu.template.mlir}"

usage() {
  echo "usage: $0 --tile-m M --tile-n N --tile-k K --output PATH" >&2
  exit 1
}

TILE_M=""
TILE_N=""
TILE_K=""
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tile-m) TILE_M="$2"; shift 2 ;;
    --tile-n) TILE_N="$2"; shift 2 ;;
    --tile-k) TILE_K="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    *) echo "error: unrecognized argument '$1'" >&2; usage ;;
  esac
done

[[ -n "$TILE_M" && -n "$TILE_N" && -n "$TILE_K" && -n "$OUTPUT" ]] || usage

# Validate positive integers -- a bad substitution here would silently
# produce malformed MLIR (e.g. an empty tile_sizes entry) rather than a
# clear error.
for name_val in "TILE_M:$TILE_M" "TILE_N:$TILE_N" "TILE_K:$TILE_K"; do
  name="${name_val%%:*}"; val="${name_val##*:}"
  if ! [[ "$val" =~ ^[0-9]+$ ]] || [[ "$val" -lt 1 ]]; then
    echo "error: $name must be a positive integer (got '$val')" >&2
    exit 1
  fi
done

[[ -f "$TEMPLATE" ]] || { echo "error: template not found at $TEMPLATE" >&2; exit 1; }

mkdir -p "$(dirname "$OUTPUT")"

sed -e "s/TILE_M/${TILE_M}/g" -e "s/TILE_N/${TILE_N}/g" -e "s/TILE_K/${TILE_K}/g" \
  "$TEMPLATE" > "$OUTPUT"

echo "$OUTPUT"
