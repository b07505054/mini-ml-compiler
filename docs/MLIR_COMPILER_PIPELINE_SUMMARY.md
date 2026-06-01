# MLIR Compiler Pipeline Summary

## What Was Added

- Real MLIR C++ pass plugin infrastructure in `mlir_passes/`
- HIR-focused canonicalization pass for tensor-level cleanup
- MatMul + Bias Add + ReLU fusion candidate detection
- Fusion group and role annotations across producer and consumer ops
- FileCheck tests for canonicalization, positive fusion, and negative fusion cases
- MLIR Affine loop tiling test
- MLIR Affine vectorization test
- Pipeline script for generating runtime-facing artifacts
- JSON bridge from annotated MLIR to lowered graph and execution plan artifacts

## Pipeline

```text
MLIR input
  -> HIRCanonicalizationPass
  -> MatMulBiasReluFusionPass
  -> trace/mlir_fused_graph.mlir
  -> tools/mlir_fusion_to_runtime_json.py
  -> trace/mlir_lowered_graph.json
  -> trace/mlir_execution_plan.json
```

## Verification

```bash
cmake --build build-mlir
tools/run_mlir_pass_tests.sh
tools/run_mlir_fusion_pipeline.sh
```

Expected signals:

```text
add(x, 0.0) canonicalized away
relu(relu(x)) canonicalized to relu(x)
fusion.candidate = "matmul_bias_relu"
fusion.group = "matmul_bias_relu_0"
FusedMatMulBiasReLU
dispatch_fused_kernel
```

## Engineering Relevance

This demonstrates:

- MLIR C++ pass/plugin development
- Canonicalization rewrites before fusion
- Linalg pattern analysis
- Fusion candidate detection
- False-positive prevention with negative tests
- Affine loop tiling
- Affine vectorization
- Runtime lowering bridge
- Cost-model metadata for backend scheduling
