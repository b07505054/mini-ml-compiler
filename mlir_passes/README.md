# MLIR Fusion Passes

This directory contains real MLIR C++ pass infrastructure, separate from the
project's custom toy graph IR.

The first pass detects a tensor-level MatMul + Bias Add + ReLU pattern:

```text
linalg.matmul
  -> linalg.map arith.addf
  -> linalg.map arith.maximumf with zero