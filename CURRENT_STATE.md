# Current State

Last verified: 2026-07-13\nSource host: GPU Linux /home/allen/Desktop/Project/ml-graph-compiler-runtime\nVerified compiler HEAD: e30c54cc477aab771525661d4dfc3c53419cd8a9 (master, ahead 1 of origin/master)\nVerified runtime HEAD: f4cc98bc93e1e8e5ecea32ffb0779b0a5c801097 (main, ahead 1 of origin/main)\nVerified capabilities HEAD: 84cf1d229788390f3b95254416636672fabe8d20 (main, origin-aligned)\nIVP source: Mac-only divergent checkout at 3f11a0422123e88eab7f90cff06d8ab7a7d48f24, ahead 1 / behind 2\nRaspberry Pi: execution/evidence target only; no canonical source repositories verified there\n

## Five-Minute Summary

The system is an IR-centered, hardware-aware implementation-decision compiler for Edge AI backends. The compiler owns semantic IR, analysis, legality, candidate generation, policy, implementation selection, implementation IR materialization, and ExecutionPlan generation. Runtime validates and executes exact compiler contracts. Evidence and calibration inform ranking after legality.

## Repository and Host Map

| Repository | Host/path | Branch/head | Status | Ownership |
|---|---|---|---|---|
| `ml-graph-compiler-runtime` | GPU Linux `/home/allen/Desktop/Project/ml-graph-compiler-runtime` | `master` `e30c54cc477aab771525661d4dfc3c53419cd8a9` | canonical source, ahead 1 | compiler, IR, legality, candidates, policy, ExecutionPlan |
| `heterogeneous-inference-runtime` | GPU Linux `/home/allen/Desktop/Project/heterogeneous-inference-runtime` | `main` `f4cc98bc93e1e8e5ecea32ffb0779b0a5c801097` | canonical source, ahead 1 | runtime contract validation and exact dispatch |
| `ml-platform-capabilities` | GPU Linux `/home/allen/Desktop/Project/ml-platform-capabilities` | `main` `84cf1d229788390f3b95254416636672fabe8d20` | canonicalization target, origin-aligned | declared capability profiles |
| `Inference-Validation-Platform` | Mac `/Users/allen/Documents/Codex/project/systems-portfolio/Inference-Validation-Platform` | `main` `3f11a0422123e88eab7f90cff06d8ab7a7d48f24` | divergent, ahead 1 / behind 2 | validation/evaluation/reporting |
| Raspberry Pi | `allen@100.110.37.6` | no source repos | evidence-only target | deployment bundles and execution evidence |

## Current Phase

P1D and P1D.1 are complete locally. P1D.1 implemented an IR-derived, offline-calibrated Raspberry Pi thread-schedule policy. No next phase has started.

Phase D0 is documentation-only canonicalization. No compiler passes, runtime behavior, schemas, target profiles, tests, evidence JSON, or generated artifacts are changed by this phase.

## Completed Milestones

- P1A: evidence-based Raspberry Pi 5 Cortex-A76 CPU target profile.
- P1B: HardwareProfile -> compiler-selected CPU kernel -> ExecutionPlan -> heterogeneous runtime -> native portable C++ ARM execution -> real Raspberry Pi correctness/timing evidence.
- P1C: portable fused MatMul + Bias + ReLU expanded to eight tile candidates.
- P1C.1: `portable_fused_matmul_bias_relu_bm32_bn128_bk32` adopted as calibration-only low-regret Raspberry Pi static default.
- P1D: backend-neutral ThreadSchedule planning and runtime execution for serial, 2/4-thread split-M, and 2/4-thread split-N schedules.
- P1D.1: compiler-side offline-calibrated Raspberry Pi policy selects serial below `M*N*K=262144` and 4-thread split-M at/above the threshold for the fixed fused MatMul + Bias + ReLU portable CPU kernel.

## What Is Real and Executable

- MLIR/HIR fusion and lowering for fused MatMul + Bias + ReLU and fused RMSNorm paths.
- ExecutionPlan generation from compiler pipeline.
- Runtime ExecutionPlan parsing/validation.
- Portable CPU fused MatMul + Bias + ReLU native execution on Raspberry Pi.
- Runtime validation of exact CPU kernel ID and thread schedule.
- AWQ Qwen artifact and compiler-plan-to-vLLM materialization path that can invoke `vLLM --quantization awq`.
- Triton/CUDA measured fused MatMul + Bias + ReLU decision pipeline as a parallel measured path.

## What Is Measured

- Raspberry Pi P1B/P1C/P1D correctness and timing evidence.
- P1D evidence: always-serial mean regret is approximately 231%; offline calibration-derived size-threshold mean regret is approximately 0.14%.
- Triton/CUDA fused MatMul + Bias + ReLU candidate measurements and selection reports.
- Some AWQ/vLLM serving measurements exist, but complete accuracy/perplexity validation does not.

## What Is Simulated or Planning-Only

- Some IVP control-plane metrics and mock-worker results are simulations.
- Some runtime distributed serving, scheduling, and trace artifacts are simulation or artifact validation rather than live cluster execution.
- NPU path is planning-only.
- Many capability profiles are declared facts, not measured support.

## Parallel / Unintegrated Paths

- Triton/CUDA measured selector uses a private schema and is not yet canonical ExecutionPlan/Runtime dispatch.
- AWQ/vLLM is executable but not integrated into one unified candidate/policy architecture.
- CandidateGenerationPass, ServingCostModelPass, PlanSelectionPass, KernelSelection, TilePlanning, ThreadSchedule, Triton selection, and AWQ deployment use separate candidate/decision representations.
- Compiler-local target profiles are richer than `ml-platform-capabilities`.

## Not Implemented

- Canonical `ImplementationCandidate` type.
- Unified policy engine across CPU, Triton, AWQ, vLLM, and future NPU paths.
- Complete dequant/layout-transform IR materialization.
- Mature memory-space/DMA/synchronization/NPU Implementation IR.
- Complete AWQ accuracy/perplexity validation.
- Complete narrow fair ExecuTorch head-to-head compiler-decision comparison.
