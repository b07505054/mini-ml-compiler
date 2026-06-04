# StableHLO Textual Subset Pipeline Report

Status: `ok`

## Pipeline

`stablehlo textual subset -> linalg/arith/math -> HIR -> LLVM dialect -> mlir-runner`

## Results

- RMSNorm HIR contains `hir.fused_rmsnorm`: `True`
- MatMul HIR contains `hir.fused_matmul_bias_relu`: `True`
- RMSNorm LLVM contains `llvm.func`: `True`
- Expected: `0.7302966946537768`
- Actual: `0.7302967`
- Abs error: `5.346223241886605e-09`

## Tools

- mlir-opt: `/opt/homebrew/opt/llvm/bin/mlir-opt`
- mlir-runner: `/opt/homebrew/opt/llvm/bin/mlir-runner`
- plugin: `/Users/allen/Documents/Codex/project/ml-graph-compiler-runtime/build-mlir-codex/HIRMatMulBiasReluFusionPass.dylib`
