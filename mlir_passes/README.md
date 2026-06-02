# MLIR Compiler Passes

This directory contains real MLIR C++ pass infrastructure, separate from the
project's custom toy graph IR.

The plugin currently contains MLIR compiler passes for canonicalization,
fusion detection, lowering, and verification:

- `hir` dialect: defines typed runtime-facing fused ops such as
  `hir.fused_matmul_bias_relu` and `hir.fused_rmsnorm`.

- `hir-canonicalize`: rewrites small tensor-level canonical forms before
  optimization.
- `matmul-bias-relu-fusion`: detects a tensor-level MatMul + Bias Add + ReLU
  fusion candidate.
- `rmsnorm-kernel-selection`: marks `llm.rmsnorm` ops for runtime-aware HIR
  lowering.
- `hir-fusion-lowering`: lowers annotated fusion candidates into typed HIR
  dialect ops.
- `hir-verify-fused-ops`: verifies invariants on emitted HIR fused ops.

The canonicalization pass performs real IR rewrites with MLIR
`OpRewritePattern`, `PatternRewriter`, `RewritePatternSet`, and the greedy
rewrite driver:

```text
linalg.map addf(x, 0.0)
  -> x

linalg.map maximumf(linalg.map maximumf(x, 0.0), 0.0)
  -> linalg.map maximumf(x, 0.0)
```

The fusion pass then detects:

```text
linalg.matmul
  -> linalg.map arith.addf
  -> linalg.map arith.maximumf with zero
```

The initial implementation is detect-and-annotate:

```mlir
linalg.matmul {fusion.candidate = "matmul_bias_relu"}
```

RMSNorm lowering uses MLIR dialect-conversion infrastructure:

```text
ConversionTarget marks "llm.rmsnorm" illegal
TypeConverter provides the HIR result-type conversion hook
ConversionPattern rewrites "llm.rmsnorm" -> hir.fused_rmsnorm
hir.fused_rmsnorm::verify checks op invariants
hir-verify-fused-ops provides a pipeline-level verification stage
```

The test `canonicalization_enables_fusion.mlir` demonstrates why the passes run
in that order: the input graph contains an identity `add(x, 0.0)` between
MatMul and BiasAdd. `hir-canonicalize` removes the identity map first, then
`matmul-bias-relu-fusion` can see and annotate the cleaned fusion chain.

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
  --load-dialect-plugin=build-mlir/HIRMatMulBiasReluFusionPass.dylib \
  --load-pass-plugin=build-mlir/HIRMatMulBiasReluFusionPass.dylib \
  mlir_passes/test/matmul_bias_relu.mlir \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion)'
```

Expected output includes:

```mlir
linalg.matmul {fusion.candidate = "matmul_bias_relu"}
```

## FileCheck

```bash
/Users/allen/Developer/llvm-build/bin/mlir-opt \
  --load-dialect-plugin=build-mlir/HIRMatMulBiasReluFusionPass.dylib \
  --load-pass-plugin=build-mlir/HIRMatMulBiasReluFusionPass.dylib \
  mlir_passes/test/matmul_bias_relu.mlir \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion)' \
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
runtime planning story by first lowering an annotated `linalg.matmul` chain into
the typed MLIR op `hir.fused_matmul_bias_relu`, then exporting that HIR op to
runtime-facing JSON.

The default bridge pipeline runs:

```text
hir-canonicalize
  -> matmul-bias-relu-fusion
  -> rmsnorm-kernel-selection
  -> hir-fusion-lowering
  -> hir-verify-fused-ops
```

Quick verification:

```bash
grep 'hir.fused_matmul_bias_relu' trace/mlir_fused_graph.mlir
grep 'FusedMatMulBiasReLU' trace/mlir_lowered_graph.json
grep 'dispatch_selected_kernel' trace/mlir_execution_plan.json
grep 'fused_matmul_add_relu' trace/mlir_execution_plan.json
```

The JSON bridge also attaches a lightweight cost model with estimated FLOPs,
memory traffic, and arithmetic intensity so the fused op can be consumed by
backend placement or scheduling heuristics.

The pass also assigns a `fusion.group` and per-op `fusion.role` metadata to
the MatMul, bias-add, and ReLU operations, making the detected producer and
consumer chain explicit before runtime lowering.

The runtime bridge records the concrete selected C++ dispatch path:

```text
hir.fused_matmul_bias_relu
  -> OpType::FusedMatMulAddReLU
  -> selected kernel from runtime benchmark evidence
```

`run_mlir_fused_kernel_benchmark` verifies that the emitted execution plan names
that custom fused kernel path, dispatches the fused op through `OpRegistry`, and
checks fused-vs-unfused numerical correctness. The benchmark is used as a
compiler/runtime integration check; CPU speedup depends on shape and local
microarchitecture.

## Resume Bullet

Added an MLIR C++ compiler pipeline with `RewritePattern` canonicalization,
typed HIR dialect ops, MatMul-Bias-ReLU fusion detection,
`ConversionTarget`/`TypeConverter`-aware RMSNorm lowering, op verification,
FileCheck coverage, and runtime-facing lowering artifacts for a heterogeneous
C++ execution planner.
