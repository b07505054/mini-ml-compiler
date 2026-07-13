# Project Status 2026

Last verified: 2026-07-13\nSource host: GPU Linux /home/allen/Desktop/Project/ml-graph-compiler-runtime\nVerified compiler HEAD: e30c54cc477aab771525661d4dfc3c53419cd8a9 (master, ahead 1 of origin/master)\nVerified runtime HEAD: f4cc98bc93e1e8e5ecea32ffb0779b0a5c801097 (main, ahead 1 of origin/main)\nVerified capabilities HEAD: 84cf1d229788390f3b95254416636672fabe8d20 (main, origin-aligned)\nIVP source: Mac-only divergent checkout at 3f11a0422123e88eab7f90cff06d8ab7a7d48f24, ahead 1 / behind 2\nRaspberry Pi: execution/evidence target only; no canonical source repositories verified there\n

## Thesis Objective

Build an IR-centered, hardware-aware implementation-decision compiler for Edge AI backends. The compiler should choose legal complete implementations from IR semantics, capability constraints, and calibrated evidence, then export an exact runtime contract.

## Repository Responsibilities

- `ml-graph-compiler-runtime`: compiler, GenericGraphIR, MLIR/HIR, analyses, legality, candidate providers, policy, Implementation IR materialization, ExecutionPlan generation.
- `heterogeneous-inference-runtime`: ExecutionPlan parsing/validation, artifact resolution, exact backend/kernel/runtime dispatch, runtime provenance and telemetry, compiler-authorized fallback.
- `ml-platform-capabilities`: intended canonical home for declared hardware/platform/backend/kernel/model/workload/deployment capability facts; not a benchmark-results database.
- `Inference-Validation-Platform`: correctness, contract validation, latency/throughput/memory/accuracy evaluation, oracle/regret, regression, reports, evidence quality checks.

## Architecture Diagram

```text
Model / GenericGraphIR
  -> Semantic IR
  -> Program Analysis
  -> Legality Analysis
  -> Candidate Providers
  -> Feasible Implementation Candidates
  -> Evidence Attachment
  -> Objective + Policy
  -> Selected Implementation
  -> Implementation IR Materialization
  -> Execution Contract (ExecutionPlan today)
  -> Runtime Exact Validation and Dispatch
  -> Execution Evidence / Telemetry
  -> Offline Calibration / Policy Update
```

## Supported Hardware / Environments

- GPU Linux: canonical compiler/runtime/capability repos and CUDA/Triton/AWQ/vLLM evidence paths.
- Raspberry Pi 5 Cortex-A76: execution/evidence target for portable CPU kernel/tile/thread path; no canonical source repos.
- Mac: local repositories, including divergent IVP source; use only when a repo is absent from GPU Linux.

## Compiler Capabilities

Real: GenericGraphIR/MLIR/HIR paths, fusion legality, HIR lowering, target constraint attrs, ExecutionPlan export, kernel selection, tile planning, thread schedule planning, quantization planning, limited boundary materialization.

Parallel/unintegrated: Triton measured selector, AWQ/vLLM deployment, multiple candidate schemas.

Not yet mature: canonical `ImplementationCandidate`, unified policy engine, DMA/memory-space/synchronization/NPU Implementation IR.

## Runtime Capabilities

Real: ExecutionPlan parsing/validation, strict portable CPU adapter, native CPU fused kernel dispatch, vLLM config materialization, runtime/evidence artifacts.

Caution: generic fallback dispatch exists and must be treated as explicit runtime failure/fallback policy, not compiler implementation selection.

## Capability DB Status

`ml-platform-capabilities` exists and is partially consumed. It is the intended canonical direction, but compiler-local profiles are currently richer/newer. Measured performance belongs in evidence/runtime/evaluation artifacts, not capability profiles.

## Evaluation Status

IVP validates runtime artifacts, compiler/runtime consistency, and generated reports. Some control-plane flows are simulated. Standalone comparison paths exist for ExecuTorch, TVM TensorIR, TensorRT, ONNX Runtime, and PyTorch, but they are not all compiler-selected production candidates.

## Real Execution Evidence

- Raspberry Pi P1B/P1C/P1D portable CPU execution evidence.
- Triton/CUDA fused MatMul + Bias + ReLU measurement/calibration reports.
- AWQ/vLLM materialization and serving evidence, without complete accuracy/perplexity validation.
- Runtime artifact validation reports in IVP.

## Completed Milestone History

P1A target profile -> P1B exact CPU dispatch -> P1C eight tile candidates -> P1C.1 low-regret tile default -> P1D thread schedule planning/runtime execution.

## Current Research Conclusions

- The primary weakness is fragmentation and integration debt, not absence of meaningful implementation.
- Measurement is valuable only after legality.
- Edge backend credibility requires memory hierarchy and data movement to become compiler IR concepts.
- P1D evidence shows a real multi-region thread decision, but the threshold policy is not shipped.

## Remaining Research Questions

- What is the canonical candidate/provider/policy abstraction?
- How should Triton/AWQ/CPU/NPU candidates share feasibility and evidence?
- What is the first complete Implementation IR path for memory spaces, DMA, and synchronization?
- What narrow ExecuTorch comparison fairly tests compiler decision quality?
- What accuracy evidence is required for quantization policy?

## Truth Boundary Summary

Declared capability is not measurement. Static estimate is not runtime latency. Measurement ranks legal candidates but does not define legality. ExecutionPlan is compiler intent and runtime contract, not semantic IR. Runtime telemetry is evidence, not compiler policy by itself.
