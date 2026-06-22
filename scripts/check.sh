#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"

echo "[check] configuring ($BUILD_DIR)"
cmake -S "$REPO_ROOT" -B "$BUILD_DIR"

echo "[check] building"
cmake --build "$BUILD_DIR"

echo "[check] running ctest"
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo "[check] done"
