# Project Status 2026

Last verified: 2026-07-14.

## Thesis

Build an IR-centered, hardware-aware implementation-decision compiler for Edge AI backends. The compiler chooses legal complete implementations from IR semantics, capability constraints, and evidence. Runtime validates and executes the exact contract.

## Current Canonical Narrative

Epoch 1 establishes one strong production/canonical path and one repaired same-stack comparison path.

The production/canonical path is Raspberry Pi portable CPU FP32 fused MatMul + Bias + ReLU. It uses complete portable candidates, feasibility, calibrated policy, selected-candidate materialization, ExecutionPlan, Runtime validation, and native kernel execution.

The repaired comparison path is E3. It invokes the live Compiler to select among XNNPACK X1/X4 candidates, emits a compiler-owned comparison contract, and executes through the same ExecuTorch/XNNPACK runner and `.pte` as ExecuTorch default.

## Current Metrics

- P1D.1 accepted threshold: `262144` over `M*N*K`.
- P1D.1 held-out exact match: `86.6667%`.
- P1D.1 mean / P95 / max regret: `0.067392%` / `0.489076%` / `0.768578%`.
- E2.1: `324` records, `0` correctness failures, implementation-stack comparison, project portable stack geomean speedup `0.380026x` versus ExecuTorch/XNNPACK default.
- E3 discovery: `162` records, `18` workloads, candidate verdict `XNNPACK_ONE_STATIC_WINNER`, selected policy `static_X1`, X1 max regret `0.415903%`.
- E3 formal: `60` records, `2` project-policy wins, `8` ties, `0` default wins, geomean default/project ratio `1.031686x`.

## Interpretation

E3 is not a complex shape-aware policy victory. It found one static XNNPACK winner for the target/workload scope. The Compiler value is that it exposes XNNPACK configurations as candidates, validates feasibility/provenance, calibrates the candidate space, selects the low-regret static option, emits a contract, and executes through the exact same XNNPACK stack.

## What Not To Claim

Do not claim general superiority over ExecuTorch, XNNPACK, vLLM, TVM, ONNX Runtime, TensorRT, or any device family. Do not claim Triton production integration. Do not claim quantization co-design is complete. Do not claim Capability DB is already the sole source of truth.
