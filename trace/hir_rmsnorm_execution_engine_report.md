# HIR RMSNorm ExecutionEngine Report

Status: `ok`

## Pipeline

`hir.fused_rmsnorm -> linalg.generic/math.rsqrt -> LLVM dialect -> mlir-runner`

## Result

- Expected: `0.7302966946537768`
- Actual: `0.7302967`
- Absolute error: `5.346223241886605e-09`
- Tolerance: `1e-05`

## Tools

- mlir-opt: `/opt/homebrew/opt/llvm/bin/mlir-opt`
- mlir-runner: `/opt/homebrew/opt/llvm/bin/mlir-runner`
- plugin: `/Users/allen/Documents/Codex/project/ml-graph-compiler-runtime/build-mlir-codex/HIRMatMulBiasReluFusionPass.dylib`
