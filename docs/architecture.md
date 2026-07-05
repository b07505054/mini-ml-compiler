# Architecture

## Purpose

This repository is a prototype **execution-planning ML compiler**. It reuses
existing MLIR infrastructure for generic graph semantics, lifts backend-relevant
regions into a decision-oriented HIR, and produces an explainable
hardware-aware execution plan consumed by heterogeneous runtimes.

It is not a production inference runtime and not a full backend codegen compiler.

## Compiler Pipeline

```text
ONNX / model graph
  ->
Frontend (import / normalize into MLIR)
  ->
Shared capability profiles
  (HardwareCapability / BackendCapability / KernelLibraryCapability)
  ->
Static optimization
  Decision Engine: 15-pass serving pipeline
  ->
ExecutionPlan  [compiler deliverable]
```

`ExecutionPlan` is produced by `PlanSelectionPass` and exported as
`execution_plan.json`. It is the compiler's final output.

Frontend, Shared Capability Profiles, and Static Optimization are **sequential
modules in one compiler pipeline** — not parallel tracks.

The compiler produces the **theoretical best static solution** given declared
capability profiles.

## Two Project Components

This repository contains two components. They are not parallel tracks; one is
the primary compiler, the other is a local demo harness.

1. **MLIR compiler** (primary): HIR/CV/Serving dialects, the 15-pass
   hardware-aware planning pipeline, quantization planning, kernel availability
   checking, per-op plan selection, and execution-plan artifact export. This is
   the execution-planning compiler described throughout this document.

2. **Custom C++ runtime demo harness** (secondary): a local toy graph runtime
   used for demos, benchmark bridges, memory-planning experiments, and
   backend-sandbox experiments. It is useful evidence for compiler/runtime
   contracts, but it is not the production distributed runtime. New
   runtime/deployment features belong in the sibling
   `heterogeneous-inference-runtime` project.

## Compiler Responsibility

The compiler does:

- Import / normalize model graphs into MLIR
- Lift backend-relevant regions into HIR
- Read shared hardware, backend, and kernel capability profiles
- Run hardware-aware static optimization passes (the Decision Engine)
- Select per-op execution plans using static penalty scoring
- Export `execution_plan.json` — the compiler deliverable

The compiler does NOT:

- Perform runtime scheduling or dynamic execution ordering
- Manage memory at runtime (no allocation, deallocation, or arena management)
- Execute kernels or dispatch CUDA/Metal/CoreML APIs
- Perform speculative decoding or KV cache management
- Make deployment policy decisions (batching, prefill/decode split, SLO targets)
- Claim measured latency or runtime speedup unless a benchmark produced them
- Materialize IR unless a pass explicitly states it does so

## Hardware Model Layers

Three distinct layers with different scopes and truth_boundary values.

### HardwareCapability

Theoretical hardware support sourced from public documentation or declared
device profiles.

- `truth_boundary = public_docs | declared_profile`
- Does not imply a specific backend or kernel exists for a given op
- Used to gate whether a backend can be attempted at all

### BackendCapability

What the backend, API, or compiler actually exposes. May be more restrictive
than raw hardware capability. Declares supported dtypes, layouts, quant modes,
cast/dequant/layout-transform support, and fallback paths.

- `truth_boundary = declared_profile`
- Sourced from target device profile JSON (`configs/target_profiles/`)
- Does not imply a specific kernel exists for a given (op, dtype, layout) tuple

### KernelLibraryCapability

Actual kernel availability for a specific (op, dtype, layout, quant_mode)
combination as declared in a kernel library profile.

- `truth_boundary = declared_profile | measured_profile`
- Sourced from `target.kernel_libraries.{backend}` module attrs in MLIR
- The most specific layer: a kernel either exists or doesn't for a given tuple
- When `truth_boundary = measured_profile`, a benchmark produced the data

No layer claims measured hardware performance unless a benchmark explicitly
produced it and the `truth_boundary` reflects that.

## Shared Capability Profiles

The compiler and runtime intentionally read from the same capability profile
data. Both repositories are currently centered on the same development machine.
Neither repository maintains separate hardware truths.

```text
Shared (compiler reads, runtime reads):
  HardwareCapability    — theoretical hardware support
  BackendCapability     — what the backend/API actually exposes
  KernelLibraryCapability — actual kernel availability per (op, dtype, layout, quant_mode)

Runtime-only extension:
  MeasuredSupport       — runtime-measured latency / throughput evidence
                          truth_boundary = measured_profile
                          The compiler never writes this layer.
```

When the runtime collects measured evidence and writes `MeasuredSupport`, it
extends the shared profile in a way the compiler can read in a subsequent
compilation to make better static decisions. This is the intended feedback path:
runtime measurement → updated profile → compiler re-optimization.

## Decision Flow

```text
Backend selected       (ExecutionProviderPlanningPass)
  ->
Representation chosen  (RepresentationPlanningPass)
  ->
Layout planned         (LayoutPlanningPass)
  ->
Boundary needs identified   (BoundaryPlanningPass)
  ->
Weights classified     (WeightClassificationPlanningPass)
  ->
Quantization strategy chosen  (QuantizationStrategyPlanningPass)
  ->
Kernel availability checked   (KernelAvailabilityPlanningPass)
  ->
Lowering decision made        (LoweringDecisionPlanningPass)
  ->
Quantized boundaries refined  (QuantizedBoundaryRefinementPass)
  ->
Alternative lowering candidates generated  (AlternativeLoweringPlanningPass)
  ->
Candidates generated   (CandidateGenerationPass)
  ->
Candidates evaluated (static penalty, no measured latency)
                       (CandidateEvaluationPass)
  ->
Plan selected          (PlanSelectionPass)
  ->
Execution plan exported
```

### Fallback is Last Resort, Not First Response

Fallback is **not** chosen immediately after a kernel miss. When
`KernelAvailabilityPlanningPass` finds no exact kernel, `AlternativeLoweringPlanningPass`
generates and validates alternatives in this priority order:

1. Algebraic decomposition (e.g., gelu → mul + sigmoid + mul, if those kernels exist)
2. Representation conversion (e.g., weight_only_int8 with dequant boundary)
3. Layout conversion (if backend supports layout transforms)
4. Cast conversion (if backend supports cast boundary ops)
5. Backend fallback (last resort: another backend handles the op)

Only when none of steps 1–4 produce a valid candidate does
`PlanSelectionPass` select the `backend_fallback` candidate. The unsupported
sentinel is emitted only when no viable path exists at all.

## 15-Pass Serving Pipeline

Each pass annotates ops with structured attrs and does not modify IR structure.
No pass selects a winner until `PlanSelectionPass`.

### ServingPhaseAnalysis

- **Question**: Is this function a prefill or decode serving phase?
- **Input attrs**: function name / signature patterns
- **Output attrs**: `serving.phase`, `serving.truth_boundary`
- **Modifies IR**: no

### KVLayoutPlanningPass

- **Question**: What KV cache layout, block size, and byte estimate should this function use?
- **Input attrs**: `serving.phase`, target profile
- **Output attrs**: `kv.layout`, `kv.block_size`, `kv.dtype_bytes`, `kv.truth_boundary`
- **Modifies IR**: no

### ReplayEligibilityPass

- **Question**: Can this function be replayed as a CUDA graph?
- **Input attrs**: `serving.phase`, function structure
- **Output attrs**: `replay.eligible`, `replay.reason`, `replay.truth_boundary`
- **Modifies IR**: no

### ExecutionProviderPlanningPass

- **Question**: Which backend/API should execute this function?
- **Input attrs**: `target.backend_capability_names`, function attrs
- **Output attrs**: `execution_provider.backend`, `execution_provider.api`, `execution_provider.truth_boundary`
- **Modifies IR**: no

### RepresentationPlanningPass

- **Question**: What effective dtype and tensor layout does the selected backend expect?
- **Input attrs**: `target.backend_capabilities.{backend}.*`, `execution_provider.backend`
- **Output attrs**: `representation.effective_dtype`, `representation.effective_layout`, `representation.truth_boundary`
- **Modifies IR**: no

### LayoutPlanningPass

- **Question**: Which specific tensor layout applies per op?
- **Input attrs**: `representation.effective_layout`, backend capability layout fields
- **Output attrs**: per-op `layout.effective_layout`, `layout.transform_required`, `layout.truth_boundary`
- **Modifies IR**: no

### BoundaryPlanningPass

- **Question**: Which boundary materialization ops (cast, dequant, layout_transform) are required?
- **Input attrs**: `representation.effective_dtype`, `layout.transform_required`, backend capability boundary support flags
- **Output attrs**: per-op `boundary.cast_required`, `boundary.dequant_required`, `boundary.layout_transform_required`, `boundary.truth_boundary`
- **Modifies IR**: no

### WeightClassificationPlanningPass

- **Question**: Are weight tensors constant (compile-time known) or runtime-variable?
- **Input attrs**: op operand producers, `representation.weights_are_constant`
- **Output attrs**: per-op `weight.classification`, `weight.truth_boundary`
- **Modifies IR**: no

### QuantizationStrategyPlanningPass

- **Question**: What quantization strategy applies per op (fp16_fallback, weight_only_int8, accuracy_sensitive)?
- **Input attrs**: `representation.effective_dtype`, `weight.classification`, backend quant mode support
- **Output attrs**: per-op `quant.strategy`, `quant.activation_dtype`, `quant.weight_dtype`, `quant.truth_boundary`
- **Modifies IR**: no

### KernelAvailabilityPlanningPass

- **Question**: Does a kernel exist in the declared library for this (op, dtype, layout, quant_mode) combination?
- **Input attrs**: `target.kernel_libraries.{backend}`, per-op `quant.*`, `layout.*`
- **Output attrs**: per-op `kernel.exists`, `kernel.lowering_status`, `kernel.fallback_backend`, `kernel.truth_boundary`
- **Modifies IR**: no

### LoweringDecisionPlanningPass

- **Question**: What is the final lowering path (direct_lower / rewrite_then_lower / dequant_then_lower / fallback_backend / unsupported)?
- **Input attrs**: `kernel.*`, `boundary.*`, `quant.*`
- **Output attrs**: per-op `lowering.decision`, `lowering.requires_cast`, `lowering.requires_dequant`, `lowering.requires_layout_transform`, `lowering.truth_boundary`
- **Modifies IR**: no

### QuantizedBoundaryRefinementPass

- **Question**: After lowering decisions are known, does weight dequant actually need a boundary op?
- **Input attrs**: `lowering.decision`, `quant.strategy`, `kernel.*`, backend `supports_dequant_boundary`
- **Output attrs**: per-op `boundary.weight_dequant_required`, `boundary.truth_boundary` (refined)
- **Modifies IR**: no

### AlternativeLoweringPlanningPass

- **Question**: When exact kernel lowering is unavailable, what legal alternatives exist?
- **Input attrs**: `kernel.*`, `quant.*`, `layout.*`, `boundary.*`, `target.kernel_libraries.*`
- **Output attrs**: per-op `alternative.available`, `alternative.candidates` (ArrayAttr), `alternative.truth_boundary`
- **truth_boundary**: `alternative_lowering_static_not_materialized_not_cost_evaluated`
- **Modifies IR**: no; candidates are annotations only, not materialized IR

### CandidateGenerationPass

- **Question**: What executable candidates (from both kernel availability and valid alternatives) exist for each op?
- **Input attrs**: `kernel.*`, `lowering.*`, `alternative.*`, `target.backend_capability_names`
- **Output attrs**:
  - per-op `compiler.candidates`, `compiler.candidates.count`, `compiler.rejected_candidates`
  - func-level `candidates` (backend × dtype × layout matrix), `candidates.count`
- **truth_boundary**: `candidate_generation_static_constraints_not_cost_evaluated`
- **Modifies IR**: no

### CandidateEvaluationPass

- **Question**: What is the static relative penalty score for each candidate?
- **Input attrs**: per-op `compiler.candidates`
- **Output attrs**: per-op `compiler.evaluated_candidates` (candidates augmented with `evaluation.*` fields)
- **Penalty model** (relative, no latency unit):
  - `direct_lower`: 0
  - `algebraic_decomposition`: 5
  - `representation_conversion`: 3 + boundary penalties (dequant +3, cast +2, layout +2)
  - `layout_conversion`: 4
  - `cast_conversion`: 2
  - `backend_fallback`: 20
  - `unsupported`: 100
  - unknown type: 10 (→ `partially_evaluated`)
- **truth_boundary**: `candidate_evaluation_static_penalty_not_measured_latency`
- **Modifies IR**: no. Does not emit fake latency ms.

### PlanSelectionPass

- **Question**: Which candidate wins for each op?
- **Input attrs**: per-op `compiler.evaluated_candidates`
- **Selection rules**:
  1. Prefer lowest penalty among `evaluation.status = "evaluated"` candidates.
  2. Prefer non-fallback over backend_fallback (rule 4: fallback only when no non-fallback exists).
  3. Tiebreak: `direct_lower` > `representation_conversion` > `cast_conversion` > `layout_conversion` > `algebraic_decomposition` > `backend_fallback` > `unsupported`.
  4. If no evaluated candidates, consider `partially_evaluated`; if none, select `unsupported`.
- **Output attrs**:
  - per-op `selected_plan.*` (9 flat attrs: candidate_type, penalty_score, reason, required_boundary_ops, backend, kernel_library, kernel_name, candidate_id, truth_boundary)
  - per-op `compiler.selected_candidates` (1-element ArrayAttr), `compiler.selection_rejections`
- **truth_boundary**: `plan_selection_static_penalty_not_measured_runtime`
- **Modifies IR**: no. Does not insert boundary ops.

## Materialization Status

**Planning is implemented. IR materialization is intentionally deferred.**

Planning (implemented by the 15-pass pipeline):
- Annotate `selected_plan.*` per op
- Annotate `quant.*`, `kernel.*`, `lowering.*`, `alternative.*`
- Export `execution_plan.json` — the compiler deliverable

IR materialization (intentionally deferred — not yet implemented):
- Inserting `hir.cast` where `boundary.cast_required = true`
- Inserting `hir.dequantize` / `hir.requantize` where `boundary.weight_dequant_required = true`
- Inserting `hir.layout_transform` where `layout.transform_required = true`
- Replacing `gelu` with primitive ops per an algebraic decomposition plan
- Rewriting IR graph structure based on selected plan

No pass in the current pipeline materializes IR. All passes are annotation-only.
IR structure is read-only until a future materialization pass explicitly states otherwise.

## Apple/CoreML Quantization Demo

The Apple Silicon / CoreML demo path demonstrates the compiler's decision-making
for a mobile accelerator profile. It is a demonstration of planning decisions,
not a measurement of ANE performance.

**What it demonstrates** (source: `declared_profile`, truth_boundary verified):
- Apple/CoreML-style quantization lowering decisions derived from a declared device profile
- Weight classification preventing incorrect `weight_only_int8` on runtime activations
- Kernel availability vs backend capability separation — a backend may declare
  support for a dtype without having a kernel for a specific op
- Per-op plan selection from evaluated candidates

**What it does NOT claim**:
- Real ANE kernel internals or layout proofs (`truth_boundary ≠ measured_profile`)
- Measured Core ML latency or throughput
- A CoreML declared profile as evidence of ANE internal execution paths
- Fallback is selected immediately after a kernel miss

## Truth Boundary Values

Every planning attr carries a `truth_boundary` field:

- `public_docs` — sourced from public hardware documentation
- `declared_profile` — from a declared device/backend profile JSON
- `measured_profile` — from an actual benchmark run on the target hardware
- `unknown` — insufficient information to classify

No compiler claim should be taken as measured performance unless
`truth_boundary = measured_profile` and a benchmark script produced the data.
The compiler makes only static optimization claims. Runtime performance
evidence lives in the runtime's `MeasuredSupport` layer.

## Module Reference

### Core Graph IR (custom C++ demo harness)

Implemented in `include/ir/` and `src/ir/`.

- `Graph`, `Tensor`, `Node`, `OpType`.
- Custom toy IR, separate from the MLIR plugin in `mlir_passes/`.
- Used for local demos, benchmark bridges, and backend-sandbox experiments.

### Compiler Passes Over Toy IR

Implemented in `include/pass/` and `src/pass/`.

- `PassManager`, canonicalization, fusion, cost reporting, memory planning.
- Educational/prototype passes, not a production optimizer.

### Runtime (local demo harness)

Implemented in `include/runtime/` and `src/runtime/`.

- `CPUBackend` dispatches real CPU kernels.
- `MockGPUBackend` logs simulated GPU execution and dispatches CPU kernels.
- `MetalBackend` logs Metal device discovery; does not execute graph kernels.
- `MemoryPlanner`, `ArenaAllocator`, `ExecutionPlanBuilder`.

This is a local demo harness and benchmark bridge. Runtime artifact consumption,
backend dispatch, distributed scheduling, and serving simulation belong in
`heterogeneous-inference-runtime`.

### MLIR Pass Plugin (primary compiler)

Implemented in `mlir_passes/`.

- `hir` dialect: typed runtime-facing fused ops, quantization ops.
- `cv` dialect: CV compiler IR.
- Serving pipeline: 15-pass hardware-aware planning pipeline.
- Fusion passes: canonicalization, MatMul-Bias-ReLU fusion, RMSNorm lowering.
- Quantization passes: planning, INT8 island propagation, Q/DQ canonicalization, operator selection.
- CV passes: frontend normalization, shape inference, memory planning, execution-domain planning.
- FileCheck tests under `mlir_passes/test/`.

This module is independent of the custom toy `Graph` IR. The bridge to the
runtime story is through generated MLIR and JSON artifacts.

### Python Tooling

Implemented in `tools/` and `src/ml_graph_compiler_runtime/`.

- Artifact generation, validation, and visualization for the compiler/runtime story.
- LLM serving analysis, execution plan export, validation, and integration bundle generation.
- All Python tooling is at the edges (validation, debug, demo). It is not the source of truth for compiler functionality.

## Implemented vs Simulated

Implemented:
- C++17 library build through CMake.
- Custom `Graph`/`Tensor`/`Node` IR, CPU kernels, op registry, pass manager.
- Real MLIR plugin: HIR/CV/Serving dialects, 15-pass serving pipeline, quantization passes, CV passes, FileCheck tests.
- Memory lifetime analysis, arena offset assignment, execution plan data structures.
- Python artifact generation and validation for LLM serving plans.

Simulated, partial, or demo-only:
- `MockGPUBackend` uses CPU kernels.
- Generic `MetalBackend` logs dispatch info, does not execute graph kernels.
- Runtime replanning uses predefined fallback assignments, not real feedback.
- CV and LLM timeline artifacts are demo snapshots, not traces from a production scheduler.
- Candidate serving plan latency and throughput values are estimated demo values (`declared_profile` or `estimated`).
- OpenXLA/IREE/Torch-MLIR paths are optional probes that may skip if tools are absent.
