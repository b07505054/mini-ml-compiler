#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "Project ONNX files:"
find . \
  \( \
    -path './.venv' \
    -o -path './venv' \
    -o -path './build' \
    -o -path './build-*' \
    -o -path './cmake-build-*' \
    -o -path './mlir_passes/build' \
    -o -path './third_party' \
    -o -path './external' \
    -o -path './artifacts' \
    -o -path './integration_bundle' \
    -o -path './.git' \
  \) -prune \
  -o -type f -iname '*.onnx' -print \
  | sed 's#^\./##' \
  | sort

echo
echo "Expected ONNX paths:"
for path in \
  models/yolo-seg.onnx \
  models/yolo-seg.onnx.data \
  models/bert_tiny.onnx \
  models/tiny_mlp.onnx \
  models/matmul_add_relu.onnx
do
  if [[ -e "${path}" ]]; then
    printf 'FOUND   %s\n' "${path}"
  else
    printf 'MISSING %s\n' "${path}"
  fi
done
