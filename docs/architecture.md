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
  Decision Engine: 16-pass serving pipeline
  ->
ExecutionPlan  [compiler deliverable]
```

`ExecutionPlan` is produced by `PlanSelectionPass` and exported as
`execution_plan.json`. It is the compiler's final output.

Frontend, Shared Capability Profiles, and Static Optimization are **sequential
modules in one compiler pipeline** — not parallel tracks.

The compiler produces the **theoretical best static solution** given declared
capability profiles.

## Current Compiler Pipeline (Qwen / LLM Serving)

The abstract pipeline above is concretely realized today, for the Qwen/LLM
serving path, as:

```text
Model
    │
    ▼
ONNX Export
    │
    ▼
Real ONNX Graph
    │
    ▼
Python Frontend Adapter
    │
    ▼
GraphFacts (Frontend Boundary)
    │
    ▼
Serving MLIR
    │
    ▼
Semantic / Planning Passes
    │
    ▼
ExecutionPlan
```

Concretely: an HF model → `tools/export_qwen_onnx.py` (optional HF Optimum
export) → a real `.onnx` protobuf file → `tools/onnx_graph_to_facts.py`
(Python frontend adapter, pattern-matched to Qwen2's HuggingFace naming
convention) → `GraphFacts` JSON → `mlir_passes/tools/qwen-onnx-to-serving-mlir`
(C++) → full per-layer-expanded Serving MLIR → `LLMFrontendNormalizationPass`
+ the 16-pass serving pipeline described below → `execution_plan.json`. See
`README.md`'s "ONNX Graph Import Path" and "Real ONNX Protobuf Bridge"
sections and `CLAUDE.md` for full detail and truth boundary of every stage.

The CV pipeline (`## Current CV Compiler Pipeline` and related sections in
`README.md`) has its own, separate, not-yet-implemented ONNX import — the
frontend adapter described here is specific to the Qwen/LLM serving path.

## What GraphFacts Is (and Is Not)

`GraphFacts` is the JSON contract between the frontend layer and the compiler
layer for the Qwen/LLM serving path.

`GraphFacts` is **NOT**:
- a model specification — the separate, legacy `qwen-to-serving-mlir` /
  `configs/models/qwen_0_5b_spec.json` path this repo is moving away from,
  not toward.
- fake graph generation. One specific `GraphFacts` document (the
  hand-authored fixture at `configs/models/qwen_0_5b_onnx_graph_facts.json`)
  is honestly labeled as such and kept for regression coverage — that does
  not make `GraphFacts` itself a fake-graph mechanism.

`GraphFacts` **IS**:
- a frontend adapter representation — the contract any frontend (ONNX today;
  potentially Torch FX / StableHLO later) must produce.
- a compiler input boundary — everything downstream of it (the C++ importer,
  `LLMFrontendNormalizationPass`, the 16-pass pipeline, `ExecutionPlan`
  export) needs no changes when the source of `GraphFacts` changes.
- derived from a real ONNX graph when produced by the real bridge — real
  protobuf parsing, real initializer names/shapes/dtypes, not fabricated
  (though role assignment within it is still Qwen2-pattern-matched, not a
  general graph interpreter).
- intentionally isolated from compiler policy — it carries structural facts
  and declared metadata only; quantization, layout, and lowering policy live
  entirely downstream in the 16-pass pipeline.

## Architecture Philosophy

The compiler is divided into three layers:

- **Frontend** — parses foreign model formats. Currently ONNX, through a
  Python frontend adapter pattern-matched to Qwen2's HuggingFace naming
  convention. Future: Torch FX / StableHLO adapters (not implemented).
- **Compiler** — semantic canonicalization (recognizing raw graph structure
  as the `llm.*`/`serving.*` vocabulary), planning (the 16-pass serving
  pipeline described below), and execution plan generation.
- **Runtime** — executes the produced plan. This belongs entirely to the
  sibling `heterogeneous-inference-runtime` project; this repo never
  executes a plan, only produces one.

## Current Capability

Current:
- Reads real ONNX graphs.
- Extracts graph structure and model metadata (per-layer role presence,
  dimensions, dtype, RoPE/lm_head-tying signals — from real initializer
  names and shapes, never guessed; hard failure on any gap).
- Converts ONNX-derived information into `GraphFacts`.
- Generates Serving MLIR.
- Generates `ExecutionPlan`.

Not yet:
- A general ONNX compiler — only Qwen2's HuggingFace naming convention is
  recognized.
- General ONNX→MLIR lowering — no generic ONNX-dialect import.
- ONNX-MLIR integration.
- A Torch FX frontend.
- A StableHLO frontend (as a source for this pipeline — the unrelated
  "StableHLO-compatible" Linalg/Arith decomposition demo elsewhere in this
  repo is a separate path).

## Future Architecture

The legacy Qwen path remains supported:

```text
Qwen ONNX
  -> Qwen GraphFacts
  -> qwen-onnx-to-serving-mlir
  -> LLM dialect
  -> ExecutionPlan
```

The new generic importer target starts with a model-agnostic ONNX boundary:

```text
ONNX
  -> ImportedGraphIR
  -> GenericGraphIR
  -> Canonical GenericGraphIR
  -> Shape/Type Annotated GenericGraphIR
  -> Diagnostics Report
  -> Domain Recognition
      -> LLM dialect
      -> CV dialect
  -> Planning
  -> ExecutionPlan
```

- `tools/onnx_import_to_graph_ir.py` implements the Phase 1 importer
  boundary: ONNX protobuf metadata to `ImportedGraphIR` JSON. It preserves
  graph inputs, outputs, nodes, attributes, values, initializers, shapes,
  dtypes, source ONNX names, opset imports, provenance, and bounded small
  numeric tensor literals for shape-bearing metadata. It does not perform
  model-family recognition.
- `tools/imported_graph_ir_to_generic_graph_ir.py` implements the Phase 2
  normalization boundary: `ImportedGraphIR` to compiler-owned
  `GenericGraphIR` with a small model-agnostic `nn.*` op vocabulary and
  source mapping back to imported ONNX nodes.
- `tools/canonicalize_generic_graph_ir.py` implements the Phase 4
  canonicalization boundary: selected `nn.*` op attributes are normalized into
  compiler-owned `canonical_attrs` while original source attributes and
  provenance remain available. Canonicalization may use small static tensor
  literals to recover model-agnostic shape operands for ops such as `Reshape`,
  `Slice`, `Resize`, and `Split`.
- `tools/infer_generic_graph_shapes.py` implements the Phase 5
  shape/type-consistency boundary: canonical `nn.*` ops receive conservative
  `shape_inference_status`, `inferred_outputs`, and explanatory notes without
  requiring full symbolic solving.
- `tools/run_generic_onnx_frontend.py` implements the Phase 6 driver:
  ONNX to verified ImportedGraphIR, GenericGraphIR, canonical GenericGraphIR,
  and shape/type annotated GenericGraphIR, with a frontend report and no
  domain recognition or lowering.
- `tools/diagnose_generic_graph_ir.py` implements the Phase 7 diagnostics
  boundary: shape-annotated `GenericGraphIR` to a model-agnostic readiness
  report with op coverage, shape inference status, metadata gaps, largest
  initializers, verifier status, and no domain recognition or lowering.
- `tools/check_generic_lowering_contract.py` implements the Phase 13
  non-emitting contract boundary: shape-annotated `GenericGraphIR` is checked
  for required attrs, shapes, dtypes, and selected lowering strategies using
  existing `func`, `tensor`, `arith`, `math`, and `linalg` infrastructure.
  It does not introduce a generic custom MLIR dialect.
- `GraphFacts` is now explicitly a legacy Qwen/LLM adapter contract, not the
  generic ONNX importer schema. It is retained so existing Qwen behavior keeps
  working while the generic path is introduced alongside it.
- Semantic recognition should eventually move out of Python and into a
  compiler-side MLIR pass, following the pattern-matching technique
  `StableHLOCompatibleRMSNormPattern` already demonstrates in
  `mlir_passes/lib/MatMulBiasReluFusionPass.cpp` for a different pattern.
- Frontend adapters should eventually become thin format parsers only — pure
  parsing and graph traversal, no architecture knowledge.
- `GraphFacts` is the current transition layer: it already isolates planning
  passes from frontend format details; what's not yet true is that role
  assignment happens in the compiler rather than in the frontend adapter.

This is a target, not a current capability. See `docs/future_work.md`.

## Two Project Components

This repository contains two components. They are not parallel tracks; one is
the primary compiler, the other is a local demo harness.

1. **MLIR compiler** (primary): HIR/CV/Serving dialects, the 16-pass
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

## 16-Pass Serving Pipeline

Each pass annotates ops with structured attrs and does not modify IR structure.
No pass selects a winner until `PlanSelectionPass`. After the 16 planning
passes, the separate `BoundaryMaterializationPass` (see Materialization
Status below) is the only IR-transforming stage.

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
- **Shape-aware layer** (`shape_cost_model_v2`, `ShapeCostModel.h`): for
  supported op kinds (`matmul_like` 2·M·K·N FLOPs with inferred K×N weight,
  `normalization` 4 ops/elem, `elementwise` 1 op/elem) with fully static
  shapes, each candidate additionally gets `evaluation.shape_cost.*`:
  FLOPs, dtype-aware input/output/weight/total bytes
  (fp32/fp16/bf16/int8/int4, using candidate dtype →
  `quant.activation_dtype` → result element type → effective dtype, and
  `quant.weight_dtype` for weights), arithmetic intensity (milli), and —
  only when the module declares `target.static_cost_profile.*` peak numbers
  from the target profile's `staticCostProfile` block — roofline time
  estimates `max(flops/peak, bytes/bandwidth) + boundary_bytes/bandwidth`
  in integer nanoseconds. Unknown op kinds and dynamic shapes fall back to
  the fixed model above, recorded in `compiler.shape_profile.{op_kind,
  status, ranking_mode}`. These are **static compiler estimates from
  declared theoretical peaks — not measured benchmarks, not runtime latency
  guarantees**; they are the first step toward shape × dtype × quantization
  × hardware co-design decisions.
- **Ranking**: `evaluation.penalty_score` keeps the fixed heuristic by
  default. Opt-in module attr `serving.cost_model.mode = "shape_aware_v2"`
  ranks an op's evaluated candidates by `estimated_total_cost_nanos`
  instead, only when every evaluated candidate has a time estimate (the
  heuristic score is preserved as `evaluation.penalty_score_v0`).
  PlanSelection tier rules (fallback last resort, unsupported never) are
  unaffected either way.
- **truth_boundary**: `candidate_evaluation_static_penalty_not_measured_latency`;
  shape layer: `static_shape_derived_declared_profile_not_measured_not_runtime_validated`
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
  - per-op `selected_plan.shape_cost.*` — the winning candidate's
    `evaluation.shape_cost.*` evidence, promoted only when present; the
    builder exports it as the per-op `shape_cost` object in
    `execution_plan.json`
  - per-op `compiler.selected_candidates` (1-element ArrayAttr), `compiler.selection_rejections`
- **truth_boundary**: `plan_selection_static_penalty_not_measured_runtime`
- **Modifies IR**: no. Does not insert boundary ops.

## Memory-Hierarchy-Aware Tile Planning (tile_planning_v1)

`TilePlanningPass` is an additional planning stage (run by
`compile-for-target` after quantization strategy and before candidate
evaluation; standalone as `tile-planning-pipeline`). It extends the static
cost model into the memory hierarchy:

- **Declared memory hierarchy** — the target profile's `staticCostProfile`
  block gains `localMemoryBytes` (SRAM / shared memory / scratchpad
  capacity), `cacheLineBytes`, and optional `supportsAsyncCopy` /
  `supportsDma` capability flags, attached as
  `target.static_cost_profile.*` module attrs. These are declared
  capacities and capabilities (public docs or declared profile), never
  measured behavior. `nvidia_gtx1650_maxq.json` declares 64 KB shared
  memory per SM, 128 B cache lines, and `supportsAsyncCopy: false`
  (cp.async requires Ampere).
- **Tile feasibility** — for matmul-like ops with static shapes, the pass
  selects the largest tile from a fixed conservative menu whose
  A (Mt×Kt) + B (Kt×Nt) + C (Mt×Nt) working set fits the declared local
  memory, using existing dtype/quantization metadata (int8 weights shrink
  the B tile). It records the footprint, double-buffer feasibility
  (2×(A+B) + C), the declared staging capability, a reuse-limited global
  traffic estimate (A read ⌈N/Nt⌉ times, B read ⌈M/Mt⌉ times, C written
  once), and — when no tile fits — the rejection reason. Dynamic shapes
  defer with an explicit status. The pass is inert when no local memory is
  declared.
- **Cost-model integration (conservative)** — `ServingCostModelPass`
  converts the planned traffic into
  `compiler.shape_profile.estimated_tiled_memory_cost_nanos` when
  bandwidth is declared. This is an annotation reported alongside the
  ideal each-byte-once estimate; **candidate ranking is unchanged**.
- **Export** — per-op `tile_plan` object in `execution_plan.json`, and
  per-op `layout` decisions (from the existing `LayoutPlanningPass` attrs,
  including the `required_input_layout` → `selected_layout` transition)
  are now exported as well.

Truth boundary:
`tile_planning_static_local_memory_model_not_measured_not_codegen` — this
is memory-hierarchy-aware **static planning**. It is not measured
performance, performs no DMA or async copies, generates no code, and does
not claim the backend kernel uses the planned tiling. Layout transforms
remain planning-only (deferred by `BoundaryMaterializationPass` until a
layout-transform op exists).

## Materialization Status

**Planning is implemented. Cast-boundary materialization is implemented.
All other IR materialization is intentionally deferred.**

Planning (implemented by the 16-pass pipeline):
- Annotate `selected_plan.*` per op
- Annotate `quant.*`, `kernel.*`, `lowering.*`, `alternative.*`
- Export `execution_plan.json` — the compiler deliverable

IR materialization implemented (`BoundaryMaterializationPass`, runs after
`PlanSelectionPass` and before plan export in `compile-for-target`; also
registered standalone as `boundary-materialization-pipeline`):
- Inserting `hir.cast` where `boundary.cast_required = true` — float-to-float
  precision casts only, converting the op's result to the planned
  `representation.effective_dtype`, redirecting uses, and updating function
  result types. Every inserted op carries provenance
  (`compiler.materialized`, `materialized.by`, `materialized.from_decision`,
  `materialized.of_op`) and `truth_boundary =
  compiler_materialized_boundary_op_not_runtime_executed`; the `hir.cast`
  verifier rejects casts without this provenance. Ops whose selected plan is
  `unsupported` are never materialized; malformed planning attrs are
  diagnosed as errors, not silently skipped. The exported plan reports
  `materialized_boundary_ops` next to the planned requirements.

IR materialization intentionally deferred (recorded per-op in
`boundary.materialization.deferred` and exported as
`deferred_boundary_ops`, never faked):
- Inserting `hir.dequantize` / `hir.requantize` where
  `boundary.weight_dequant_required = true` — requires scale/zero-point
  metadata the planning pipeline does not produce
- Inserting `hir.layout_transform` where `layout.transform_required = true`
  — no layout-transform op exists in the hir dialect yet
- Replacing `gelu` with primitive ops per an algebraic decomposition plan
- Rewriting IR graph structure based on selected plan

The 16 planning passes remain annotation-only; `BoundaryMaterializationPass`
is the only pass that modifies IR structure, and only for the cast subset
above.

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
- Serving pipeline: 16-pass hardware-aware planning pipeline.
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
- Real MLIR plugin: HIR/CV/Serving dialects, 16-pass serving pipeline, quantization passes, CV passes, FileCheck tests.
- Memory lifetime analysis, arena offset assignment, execution plan data structures.
- Python artifact generation and validation for LLM serving plans.

Simulated, partial, or demo-only:
- `MockGPUBackend` uses CPU kernels.
- Generic `MetalBackend` logs dispatch info, does not execute graph kernels.
- Runtime replanning uses predefined fallback assignments, not real feedback.
- CV and LLM timeline artifacts are demo snapshots, not traces from a production scheduler.
- Candidate serving plan latency and throughput values are estimated demo values (`declared_profile` or `estimated`).
- OpenXLA/IREE/Torch-MLIR paths are optional probes that may skip if tools are absent.
