#!/usr/bin/env bash
set -euo pipefail

MLIR_INPUT="${MLIR_INPUT:-mlir/tiny_gpt_serving.mlir}"
CONFIG_INPUT="${CONFIG_INPUT:-configs/tiny_gpt_llm_config.json}"
ANALYSIS_OUT="${ANALYSIS_OUT:-trace/llm_serving_compiler_analysis.json}"
ARTIFACT_OUT="${ARTIFACT_OUT:-artifacts/apple_demo}"
VALIDATION_OUT="${VALIDATION_OUT:-trace/llm_artifact_validation_report.json}"
BUNDLE_OUT="${BUNDLE_OUT:-integration_bundle/apple_demo_artifacts}"

echo "[1/4] Analyze LLM serving MLIR"
python3 tools/analyze_llm_serving_mlir.py \
  --mlir "$MLIR_INPUT" \
  --out "$ANALYSIS_OUT"

echo "[2/4] Emit serving runtime artifacts"
python3 tools/emit_llm_artifacts_from_analysis.py \
  --analysis "$ANALYSIS_OUT" \
  --config "$CONFIG_INPUT" \
  --out "$ARTIFACT_OUT"

echo "[3/4] Validate artifacts"
python3 tools/validate_llm_serving_artifacts.py \
  --artifacts "$ARTIFACT_OUT" \
  --out "$VALIDATION_OUT"

echo "[4/4] Copy artifacts to integration bundle"
mkdir -p "$BUNDLE_OUT"
cp "$ARTIFACT_OUT"/*.json "$BUNDLE_OUT"/
cp "$VALIDATION_OUT" "$BUNDLE_OUT"/llm_artifact_validation_report.json

echo
echo "LLM serving artifact pipeline complete."
echo "Artifacts: $ARTIFACT_OUT"
echo "Bundle:    $BUNDLE_OUT"
echo "Report:    $VALIDATION_OUT"