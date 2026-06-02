# MLIR Compiler Pipeline Summary

## What Was Added

- Real MLIR C++ pass plugin infrastructure in `mlir_passes/`
- HIR-focused canonicalization pass for tensor-level cleanup
- MatMul + Bias Add + ReLU fusion candidate detection
- Fusion group and role annotations across producer and consumer ops
- FileCheck tests for canonicalization, positive fusion, and negative fusion cases
- A canonicalization-enables-fusion test where `add(x, 0.0)` cleanup exposes
  the MatMul -> BiasAdd -> ReLU chain to the fusion detector
- MLIR Affine loop tiling test
- MLIR Affine vectorization test
- Pipeline script for generating runtime-facing artifacts
- MLIR HIR lowering pass that rewrites fusion candidates to generic HIR ops:
  `hir.fused_matmul_bias_relu` and `hir.fused_rmsnorm`
- JSON bridge from HIR MLIR ops to lowered graph and execution plan artifacts
- Runtime execution plan metadata that maps the lowered fused op to
  `OpType::FusedMatMulAddReLU` and the `fused_matmul_add_relu` custom kernel
- A C++ benchmark that checks the MLIR execution plan and dispatches the fused
  op through the runtime registry for correctness and latency comparison
- Runtime-aware kernel-selection metadata for fusion candidates. The compiler
  now records selected, candidate, and fallback kernels plus runtime benchmark
  evidence instead of assuming that every fused candidate should dispatch to a
  custom kernel.
- RMSNorm compiler annotation and lowering support:
  `llm.rmsnorm -> hir.fused_rmsnorm`, with kernel selection driven by
  `heterogeneous-inference-runtime` RMSNorm benchmark artifacts.

## Pipeline

```text
MLIR input
  -> HIRCanonicalizationPass
  -> MatMulBiasReluFusionPass
  -> RMSNormKernelSelectionPass
  -> HIRFusionLoweringPass
  -> trace/mlir_fused_graph.mlir
  -> tools/mlir_fusion_to_runtime_json.py
  -> trace/mlir_lowered_graph.json
  -> trace/mlir_execution_plan.json
  -> runtime-aware dispatch to selected kernel or fallback
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
canonicalization_enables_fusion.mlir gains fusion.candidate after cleanup
fusion.candidate = "matmul_bias_relu"
fusion.group = "matmul_bias_relu_0"
hir.fused_matmul_bias_relu emitted by HIRFusionLoweringPass
FusedMatMulBiasReLU
dispatch_selected_kernel
runtime_kernel = fused_matmul_add_relu
runtime_op_type = FusedMatMulAddReLU
kernel_selection.selected_kernel
kernel_selection.fallback_kernel
kernel_selection.evidence
llm.rmsnorm annotated as fusion.candidate = "rmsnorm"
hir.fused_rmsnorm emitted with fused_rmsnorm_cuda candidate and torch_rmsnorm fallback
```

## Engineering Relevance

This demonstrates:

- MLIR C++ pass/plugin development
- Canonicalization rewrites before fusion
- Linalg pattern analysis
- Fusion candidate detection
- MLIR lowering from source/fusion dialect patterns into HIR-stage generic ops
- False-positive prevention with negative tests
- Affine loop tiling
- Affine vectorization
- Runtime lowering bridge
- Cost-model metadata for backend scheduling
- Custom fused-kernel dispatch path from MLIR lowering metadata to C++ runtime
- Runtime-aware compiler behavior: kernel selection is based on benchmark
  evidence when available and conservatively falls back when evidence is absent
