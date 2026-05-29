# ML Compiler Resume Bullets

## Strong Resume Bullets

- Built a real MLIR C++ pass plugin that detects MatMul-Bias-ReLU patterns in `linalg` IR and annotates fusion candidates with producer/consumer fusion metadata.
- Added MLIR FileCheck coverage for positive and negative fusion cases, preventing false-positive fusion when the ReLU consumer is absent.
- Connected annotated MLIR output into runtime-facing artifacts by exporting fused graph IR, lowered graph JSON, and execution plan JSON for a heterogeneous C++ runtime planner.
- Added lightweight cost-model metadata for fused MatMul-Bias-ReLU ops, including estimated FLOPs, memory traffic, and arithmetic intensity for backend scheduling.
- Added MLIR Affine loop tiling and vectorization tests to demonstrate loop transformation, vector IR generation, and compiler optimization workflows.

## Short Version

- Implemented an MLIR C++ fusion pass pipeline for MatMul-Bias-ReLU detection, FileCheck validation, runtime lowering artifacts, and cost-model metadata.

## Interview Talking Points

- The pass currently performs detect-and-annotate instead of rewrite, which keeps the first version robust and easy to validate.
- The pipeline separates MLIR frontend analysis from the existing custom C++ runtime planner.
- The negative test demonstrates that the pass avoids annotating incomplete fusion patterns.
- The runtime bridge shows how compiler annotations can feed backend placement, scheduling, and dispatch decisions.
- The Affine tiling and vectorization tests show familiarity with MLIR loop transformation workflows beyond graph-level pattern matching.

## One-Line Project Summary

Added a real MLIR compiler pass pipeline that detects MatMul-Bias-ReLU fusion candidates, validates the transformation with FileCheck, and exports lowered runtime planning artifacts for heterogeneous execution.