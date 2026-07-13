# Architecture Constitution

Last verified: 2026-07-13\nSource host: GPU Linux /home/allen/Desktop/Project/ml-graph-compiler-runtime\nVerified compiler HEAD: e30c54cc477aab771525661d4dfc3c53419cd8a9 (master, ahead 1 of origin/master)\nVerified runtime HEAD: f4cc98bc93e1e8e5ecea32ffb0779b0a5c801097 (main, ahead 1 of origin/main)\nVerified capabilities HEAD: 84cf1d229788390f3b95254416636672fabe8d20 (main, origin-aligned)\nIVP source: Mac-only divergent checkout at 3f11a0422123e88eab7f90cff06d8ab7a7d48f24, ahead 1 / behind 2\nRaspberry Pi: execution/evidence target only; no canonical source repositories verified there\n

## Canonical Identity

This project is an IR-centered, hardware-aware implementation-decision compiler for Edge AI backends.

It is not merely an MLIR lowering demo, an auto-tuning framework, a deployment planner, a runtime framework, or a benchmark collection. The compiler uses IR semantics and legality to define valid implementation choices, evidence-backed policy to select among legal candidates, implementation IR materialization for representation-changing decisions, and an exact runtime contract for execution.

## Canonical Pipeline

Model / GenericGraphIR -> Semantic IR -> Program Analysis -> Legality Analysis -> Candidate Providers -> Feasible Implementation Candidates -> Evidence Attachment -> Objective + Policy -> Selected Implementation -> Implementation IR Materialization -> Execution Contract -> Runtime Exact Validation and Dispatch -> Execution Evidence / Telemetry -> Offline Calibration / Policy Update.

## Mandatory Layers

### 1. Semantic IR

Semantic IR represents what the program computes. It owns operator semantics, tensor ranks/shapes/types, constants, regions, graph dependencies, and model-level structure. IR is the semantic authority. ExecutionPlan is derived from IR and is not semantic IR.

### 2. Implementation IR

Implementation IR represents how the compiler has chosen to implement the program. It covers fusion, lowering, layout, precision representation, tiling, parallel decomposition, memory spaces, DMA, synchronization, graph partitioning, backend regions, and target-visible dispatch regions. A decision that changes representation, data movement, control structure, or generated code shape must materialize into IR.

Current verified state: HIR lowering exists for fused MatMul + Bias + ReLU and RMSNorm paths; BoundaryMaterialization currently inserts limited real IR including `hir.cast`; dequantization, layout-transform, memory-space, DMA, synchronization, graph-partition boundary, and NPU command-region materialization are incomplete.

### 3. Execution Contract

The Execution Contract is the exact compiler-to-runtime contract. The current serialized production form is `ExecutionPlan`; this documentation does not rename production schemas. The contract records backend, kernel or artifact identity, shapes, dtype, layout requirements, thread schedule, runtime requirements, validation obligations, truth boundaries, and explicit fallback permissions.

Selection of an opaque prebuilt kernel, runtime, or artifact may remain in the Execution Contract when the compiler does not own its internal lowering.

### 4. Runtime

Runtime validates and executes the exact contract. Runtime must not perform compiler candidate search, online benchmarking for implementation selection, silent kernel substitution, silent backend change, silent precision change, or hidden fallback. Runtime may only use fallback explicitly permitted by the compiler contract.

### 5. Evidence / Calibration

Evidence stores static estimates, measurements, predictions, correctness, accuracy, oracle, regret, telemetry, confidence, provenance, and truth boundaries. Evidence may influence policy only after legality. Evidence must not define legality.

## Core Abstractions

### ImplementationCandidate

A candidate is a selectable implementation option rooted in one explicit scope: an IR operation, fused IR region, graph partition, deployment boundary, or serving boundary. There is no single canonical `ImplementationCandidate` type in source today; creating one is an architectural requirement, not current implementation.

### CandidateProvider

Candidate Providers enumerate candidates. Providers do not own global policy. CPU, CUDA/Triton, AWQ/vLLM, CoreML, ExecuTorch, TVM, TensorRT, ONNX Runtime, PyTorch fallback, and future NPU paths can provide candidates only at explicit scopes.

### Feasibility

Feasibility answers whether a candidate can be used under hardware, runtime, shape, dtype, layout, memory, artifact, accuracy, and contract constraints. Feasibility is separate from evidence and ranking.

### Policy

Policy chooses among feasible candidates under an objective. Measurement may rank legal candidates. Measurement may never legalize an illegal candidate.

## Decision Scope Hierarchy

Every decision must state scope:

- operator
- fused region
- graph partition
- model deployment
- serving configuration

Backend, kernel, tile, thread, fusion, layout, memory, precision, quantization, DMA, graph partition, runtime, serving, and artifact selection must be classified by scope before implementation begins.

## Compiler / Runtime Boundary

Compiler owns semantic analysis, legality, candidate generation, feasibility filtering, evidence references, policy, implementation selection, Implementation IR materialization, and ExecutionPlan generation.

Runtime owns ExecutionPlan parsing and validation, artifact resolution, exact backend/kernel/runtime dispatch, memory/resource execution, provenance, telemetry, explicit failure, and compiler-authorized fallback.

Runtime does not own global implementation selection.

## Edge AI Backend Requirements

For Edge AI, memory hierarchy, memory spaces, DMA, synchronization, local storage, graph partitioning, boundary movement, CPU/GPU/NPU feasibility, deployment constraints, thermal/power evidence, and strict runtime contracts are compiler concepts. They cannot be reduced to runtime flags once the compiler owns backend lowering.

## Architectural Invariants

1. IR is the semantic authority.
2. Every implementation candidate is rooted in IR, graph partition, deployment boundary, or serving boundary.
3. Measurement may rank legal candidates; it may never legalize an illegal candidate.
4. Representation, data movement, control-structure, or code-shape decisions must materialize into IR.
5. Opaque prebuilt kernel/runtime/artifact selection may remain in the Execution Contract.
6. Feasibility, evidence, policy, and selection are separate concepts.
7. Candidate Providers enumerate; they do not own global policy.
8. The compiler chooses; Runtime validates and executes.
9. Runtime fallback must be explicitly permitted by the compiler contract.
10. Truth boundaries are mandatory for declared capability, static estimate, measurement, prediction, calibration, oracle, regret, and runtime trace.
11. Decision scope must always be explicit.
12. Hardware capability, Runtime capability, deployment capability, telemetry, and measured evidence have separate ownership.
13. ExecutionPlan is derived from IR; ExecutionPlan is not semantic IR.
14. Edge memory hierarchy, DMA, synchronization, local storage, partitioning, and boundary movement are compiler concepts.
15. Every future feature must identify its architectural layer before implementation begins.

## Audit Hygiene

Capability-first audit protocol: capability -> enumerate every path -> find producer -> find consumer -> verify tests -> verify execution -> verify measured evidence -> classify canonical vs parallel vs historical -> then conclude.

Known audit failure modes to avoid:

- Absence from `runtimeKernels[]` must not be generalized into absence of executable AWQ. AWQ/vLLM execution uses an artifact/runtime deployment path, not custom INT4 runtime kernel descriptors.
- Absence of `createTritonSelectionPass` from `compile-for-target` must not be generalized into absence of real Triton decision-making. The Triton path has real measured parallel selection, but it is not canonical ExecutionPlan/Runtime integration yet.
