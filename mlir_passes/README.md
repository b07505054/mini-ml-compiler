# MLIR Fusion Passes

This directory contains real MLIR C++ pass infrastructure, separate from the
project's custom toy graph IR.

The first pass detects a tensor-level MatMul + Bias Add + ReLU pattern:

```text
linalg.matmul
  -> linalg.map arith.addf
  -> linalg.map arith.maximumf with zero
```

The initial implementation is detect-and-annotate:

```mlir
linalg.matmul {fusion.candidate = "matmul_bias_relu"}
```

## Requirements

Tested target: LLVM/MLIR built from `/Users/allen/Developer/llvm-build`.

Use the same LLVM/MLIR build for CMake, `mlir-opt`, and `FileCheck`.
Mixing a Homebrew `mlir-opt` with a locally built plugin can cause plugin
registration failures.

Required CMake packages:

```text
/Users/allen/Developer/llvm-build/lib/cmake/mlir
/Users/allen/Developer/llvm-build/lib/cmake/llvm
```

## Build

From the `ml-graph-compiler-runtime` repo root:

```bash
cmake -S mlir_passes -B build-mlir \
  -DMLIR_DIR=/Users/allen/Developer/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=/Users/allen/Developer/llvm-build/lib/cmake/llvm

cmake --build build-mlir
```

On macOS this builds:

```text
build-mlir/HIRMatMulBiasReluFusionPass.dylib
```

## Run The Fusion Pipeline

This plugin is registered as an MLIR pass pipeline. Use the same `mlir-opt`
binary from the LLVM/MLIR build used by CMake.

```bash
/Users/allen/Developer/llvm-build/bin/mlir-opt \
  --load-pass-plugin=build-mlir/HIRMatMulBiasReluFusionPass.dylib \
  mlir_passes/test/matmul_bias_relu.mlir \
  --pass-pipeline='builtin.module(matmul-bias-relu-fusion)'
```

Expected output includes:

```mlir
linalg.matmul {fusion.candidate = "matmul_bias_relu"}
```

## FileCheck

```bash
/Users/allen/Developer/llvm-build/bin/mlir-opt \
  --load-pass-plugin=build-mlir/HIRMatMulBiasReluFusionPass.dylib \
  mlir_passes/test/matmul_bias_relu.mlir \
  --pass-pipeline='builtin.module(matmul-bias-relu-fusion)' \
  | /Users/allen/Developer/llvm-build/bin/FileCheck mlir_passes/test/matmul_bias_relu.mlir
```

## Runtime Bridge

The fusion pipeline can emit runtime-facing artifacts:

```bash
tools/run_mlir_fusion_pipeline.sh
```

Outputs:

```text
trace/mlir_fused_graph.mlir
trace/mlir_lowered_graph.json
trace/mlir_execution_plan.json
```

This connects the real MLIR pass output to the existing C++ heterogeneous
runtime planning story by mapping an annotated `linalg.matmul` into a
`hir.fused_matmul_bias_relu` lowered runtime op.

Quick verification:

```bash
grep 'fusion.candidate = "matmul_bias_relu"' trace/mlir_fused_graph.mlir
grep 'FusedMatMulBiasReLU' trace/mlir_lowered_graph.json
grep 'dispatch_fused_kernel' trace/mlir_execution_plan.json
```

## Resume Bullet

Added an MLIR C++ pass pipeline that detects MatMul-Bias-ReLU patterns,
annotates `linalg.matmul` fusion candidates, and exports lowered runtime
planning artifacts for a heterogeneous C++ execution planner.