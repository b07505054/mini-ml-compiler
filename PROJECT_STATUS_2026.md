# Project Status 2026

Last verified: 2026-07-13\nSource host: GPU Linux /home/allen/Desktop/Project/ml-graph-compiler-runtime\nVerified compiler HEAD before A5: b81830e6883d6284e867fe5e19cc44ccd85f0e23 (master, ahead 7 of origin/master)\nVerified runtime HEAD: a6e2ae8648ee27d8e73396218266e98a0ea0cbc6 (main, ahead 3 of origin/main)\nVerified capabilities HEAD: aac593da0bdde7a95c38c03920fc4d00b73011db (main, ahead 1 of origin/main)\nIVP source: Mac-only divergent checkout at 3f11a0422123e88eab7f90cff06d8ab7a7d48f24, ahead 1 / behind 2\nRaspberry Pi: execution/evidence target only; no canonical source repositories verified there\n

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

A1 provides a narrow compiler-internal `ImplementationCandidate` core for the active op-candidate generation/evaluation/selection path. A2 routes the P1D.1 Raspberry Pi thread-schedule decision through serial/parallel implementation candidates before exporting the unchanged ExecutionPlan contract. A3 makes those two live Raspberry Pi portable CPU candidates complete for backend, opaque Runtime contract, kernel ID, tile identity, dtype, and thread schedule without changing policy or Runtime behavior. A4 extracts their enumeration into `PortableCPUProvider`. A5 separates provider enumeration, target/workload feasibility evaluation, policy input, and selected-candidate materialization for that fixed portable CPU path. Not yet mature: project-wide candidate/provider unification, unified policy engine, DMA/memory-space/synchronization/NPU Implementation IR.

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

P1A target profile -> P1B exact CPU dispatch -> P1C eight tile candidates -> P1C.1 low-regret tile default -> P1D thread schedule planning/runtime execution -> P1D.1 offline-calibrated IR-derived thread-schedule policy -> A1 minimal compiler-internal `ImplementationCandidate` foundation -> A2 P1D.1 candidate migration -> A3 complete portable CPU candidate identity for the active fixed kernel/tile/dtype/thread path -> A4 dedicated in-process `PortableCPUProvider` extraction -> A5 provider contract hardening with explicit feasibility and materialization boundaries.

## Current Research Conclusions

- The primary weakness is fragmentation and integration debt, not absence of meaningful implementation.
- Measurement is valuable only after legality.
- Edge backend credibility requires memory hierarchy and data movement to become compiler IR concepts.
- P1D.1 ships the first narrow offline-calibrated policy edge: Raspberry Pi fused MatMul + Bias + ReLU selects serial below `M*N*K=262144` and 4-thread split-M at/above the threshold.
- A1 reduces duplicated candidate parsing in the active compiler-internal candidate path without changing Runtime behavior or claiming Triton/AWQ/thread/tile unification.
- A2 proves one real calibrated implementation decision can use the candidate core without changing the threshold, policy metric, Runtime code, or ExecutionPlan semantics.
- A3 proves the active portable CPU path can select between complete opaque implementation candidates that include kernel, tile, dtype, and thread identity, while still leaving inactive tile alternatives and external provider paths outside the candidate core.
- A4 proves candidate enumeration can be separated from policy for the active portable CPU path without adding a global registry, changing Runtime behavior, or changing plan semantics.
- A5 proves the provider does not need to own feasibility, policy, or materialization: provider output is enumeration, `PortableCPUFeasibilityEvaluator` owns target/workload satisfaction, policy remains the only selector, and selected-candidate materialization derives the Execution Contract fields without changing plan hashes.

## Remaining Research Questions

- What is the canonical candidate/provider/policy abstraction?
- How should Triton/AWQ/CPU/NPU candidates share feasibility and evidence?
- What is the first complete Implementation IR path for memory spaces, DMA, and synchronization?
- What narrow ExecuTorch comparison fairly tests compiler decision quality?
- What accuracy evidence is required for quantization policy?

## Truth Boundary Summary

Declared capability is not measurement. Static estimate is not runtime latency. Measurement ranks legal candidates but does not define legality. ExecutionPlan is compiler intent and runtime contract, not semantic IR. Runtime telemetry is evidence, not compiler policy by itself.
