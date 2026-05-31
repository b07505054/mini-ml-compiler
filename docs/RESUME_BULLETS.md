# ML Compiler Resume Bullets

## Strong Resume Bullets

- Built an LLM compiler/runtime planning pipeline that analyzes a tiny MLIR-style LLM graph and emits runtime-facing artifacts for graph IR, prefill/decode execution, KV-cache layout, memory planning, scheduling metadata, and validation.
- Implemented a serving-aware compiler analysis pass that extracts prefill/decode phase partitioning, KV-cache producer/consumer roles, and runtime constraints from LLM graph IR.
- Added an artifact-lowering pipeline that converts compiler analysis output into Apple-demo-ready JSON contracts for execution planning, KV-cache layout, memory budget validation, and scheduler visualization.
- Added validation tooling for LLM compiler artifacts, checking required outputs, prefill/decode phases, KV-cache capacity consistency, memory budget status, scheduling queues, and manifest completeness.
- Built a real MLIR C++ pass plugin that detects MatMul-Bias-ReLU patterns in `linalg` IR and annotates fusion candidates with producer/consumer fusion metadata.
- Added MLIR FileCheck coverage for positive and negative fusion cases, preventing false-positive fusion when the ReLU consumer is absent.
- Connected annotated MLIR output into runtime-facing artifacts by exporting fused graph IR, lowered graph JSON, and execution plan JSON for a heterogeneous C++ runtime planner.
- Added lightweight cost-model metadata for fused MatMul-Bias-ReLU ops, including estimated FLOPs, memory traffic, and arithmetic intensity for backend scheduling.
- Added MLIR Affine loop tiling and vectorization tests to demonstrate loop transformation, vector IR generation, and compiler optimization workflows.

## Short Version

- Implemented an LLM compiler/runtime planning pipeline from tiny MLIR graph analysis to validated execution, KV-cache, memory, scheduling, and Apple-demo artifacts.
- Implemented an MLIR C++ fusion pass pipeline for MatMul-Bias-ReLU detection, FileCheck validation, runtime lowering artifacts, and cost-model metadata.

## Interview Talking Points

- The LLM pipeline is positioned as compiler/runtime planning, not a production serving engine: it emits execution, memory, KV-cache layout, scheduling, and validation contracts consumed by downstream demos.
- The lightweight analysis pass separates frontend graph analysis from artifact lowering, mirroring a compiler pipeline where analysis results feed runtime planning.
- KV-cache work is scoped to layout and memory planning metadata, while dynamic allocation, eviction, token sampling, and request serving remain runtime responsibilities.
- The validation report turns generated JSON into a testable contract, checking phase structure, KV-cache capacity, memory budget, scheduling queues, and manifest completeness.
- The pass currently performs detect-and-annotate instead of rewrite, which keeps the first version robust and easy to validate.
- The pipeline separates MLIR frontend analysis from the existing custom C++ runtime planner.
- The negative test demonstrates that the pass avoids annotating incomplete fusion patterns.
- The runtime bridge shows how compiler annotations can feed backend placement, scheduling, and dispatch decisions.
- The Affine tiling and vectorization tests show familiarity with MLIR loop transformation workflows beyond graph-level pattern matching.

## One-Line Project Summary

Built an LLM compiler/runtime planning demo that analyzes a tiny MLIR-style LLM graph, extracts prefill/decode and KV-cache planning metadata, emits validated runtime-facing artifacts, and connects MLIR compiler analysis to dashboard-ready execution, memory, scheduling, and validation contracts.
