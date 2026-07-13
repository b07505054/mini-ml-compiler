# Current State

Last verified: 2026-07-13\nSource host: GPU Linux /home/allen/Desktop/Project/ml-graph-compiler-runtime\nVerified compiler base before A1: 5822a04aa600ec41fcca5ef00619cc27d3e37c40 (master, ahead 3 of origin/master)\nVerified runtime HEAD: a6e2ae8648ee27d8e73396218266e98a0ea0cbc6 (main, ahead 3 of origin/main)\nVerified capabilities HEAD: aac593da0bdde7a95c38c03920fc4d00b73011db (main, ahead 1 of origin/main)\nIVP source: Mac-only divergent checkout; not modified in A1\nRaspberry Pi: execution/evidence target only; no canonical source repositories verified there\n

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

P1D, P1D.1, A1, A2, and A3 are complete locally. A1 added a minimal compiler-internal `ImplementationCandidate` foundation for the existing CandidateGeneration -> ServingCostModel -> PlanSelection path. A2 migrated the P1D.1 Raspberry Pi thread-schedule decision into that candidate core. A3 makes the two active Raspberry Pi portable CPU candidates complete for backend, opaque Runtime contract, kernel, tile, dtype, and thread-schedule identity without changing policy or Runtime behavior. No Triton, AWQ, NPU, DMA, NEON, ExecuTorch comparison, or new policy phase has started.

Phase D0 is documentation-only canonicalization. No compiler passes, runtime behavior, schemas, target profiles, tests, evidence JSON, or generated artifacts are changed by this phase.

## Completed Milestones

- P1A: evidence-based Raspberry Pi 5 Cortex-A76 CPU target profile.
- P1B: HardwareProfile -> compiler-selected CPU kernel -> ExecutionPlan -> heterogeneous runtime -> native portable C++ ARM execution -> real Raspberry Pi correctness/timing evidence.
- P1C: portable fused MatMul + Bias + ReLU expanded to eight tile candidates.
- P1C.1: `portable_fused_matmul_bias_relu_bm32_bn128_bk32` adopted as calibration-only low-regret Raspberry Pi static default.
- P1D: backend-neutral ThreadSchedule planning and runtime execution for serial, 2/4-thread split-M, and 2/4-thread split-N schedules.
- P1D.1: compiler-side offline-calibrated Raspberry Pi policy selects serial below `M*N*K=262144` and 4-thread split-M at/above the threshold for the fixed fused MatMul + Bias + ReLU portable CPU kernel.
- A1: compiler-internal op-scoped `ImplementationCandidate` type, shared encode/decode helpers, minimal feasibility summary, and `PolicyResult` separation are implemented for the active compiler candidate/evaluation/selection path.
- A2: P1D.1 now enumerates serial and 4-thread split-M thread-schedule `ImplementationCandidate`s, evaluates typed feasibility, applies the unchanged policy, and materializes the selected candidate into the existing `thread_schedule` ExecutionPlan contract.
- A3: the active Raspberry Pi portable CPU candidates now include complete executable identity for the fixed fused MatMul + Bias + ReLU path: CPU backend, opaque portable native kernel contract, kernel ID `portable_fused_matmul_bias_relu_bm32_bn128_bk32`, tile `BM=32, BN=128, BK=32`, compiler-normalized dtype `fp32`, and serial/4-thread split-M schedule variants.

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
- AWQ/vLLM is executable but not integrated into the A1/A2/A3 compiler-internal candidate core or one unified candidate/policy architecture.
- CandidateGenerationPass, ServingCostModelPass, PlanSelectionPass, KernelSelection, TilePlanning, ThreadSchedule, Triton selection, and AWQ deployment use separate candidate/decision representations.
- Compiler-local target profiles are richer than `ml-platform-capabilities`.

## Not Implemented

- Project-wide `ImplementationCandidate` unification across inactive tile alternatives, generic TilePlan, unrelated KernelSelection paths, Triton, AWQ/vLLM, deployment candidates, serving candidates, and external providers.
- Unified policy engine across CPU, Triton, AWQ, vLLM, and future NPU paths.
- Complete dequant/layout-transform IR materialization.
- Mature memory-space/DMA/synchronization/NPU Implementation IR.
- Complete AWQ accuracy/perplexity validation.
- Complete narrow fair ExecuTorch head-to-head compiler-decision comparison.
