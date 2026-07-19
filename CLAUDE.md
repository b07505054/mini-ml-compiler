# Claude Code Handoff

## Repository Summary

This repository is a prototype **execution-planning ML compiler**. It reuses existing MLIR infrastructure for generic graph semantics, lifts backend-relevant regions into a decision-oriented HIR, and produces an explainable hardware-aware execution plan consumed by heterogeneous runtimes.

The primary compiler story lives in `mlir_passes/`. The custom C++ runtime in `src/` and `apps/` is a local demo harness and benchmark bridge — it is not the production distributed runtime layer. New runtime/deployment features belong in the sibling `heterogeneous-inference-runtime` project.

Be careful to distinguish implemented behavior from simulations:

- Implemented: HIR/CV/Serving MLIR dialects, 16-pass serving pipeline, custom C++ `Graph`/`Tensor`/`Node` IR, CPU kernels, op registry, pass manager, memory planner, execution plan structures, cost planner, Python artifact generation/validation.
- Simulated or partial: `MockGPUBackend`, generic `MetalBackend` graph execution, runtime replanning, many timeline artifacts, and serving latency/throughput values in generated plans.

High-value implemented compiler evidence:

- HIR dialect ops, verifiers, canonicalization, fusion, conversion, and lowering tests under `mlir_passes/test/`.
- 16-pass hardware-aware serving pipeline (serving phase → KV layout → replay eligibility → execution provider → representation → layout → boundary → weight classification → quantization strategy → kernel availability → lowering decision → quantized boundary refinement → alternative lowering → candidate generation → candidate evaluation → plan selection).
- StableHLO-compatible textual subset import for RMSNorm and MatMul-Bias-ReLU patterns.
- HIR RMSNorm executable CPU path via the MLIR execution engine.
- Apple Silicon MLIR-to-Metal RMSNorm path with a real Metal kernel, generated execution plan, and dispatch validation when the MLIR pipeline has produced the required trace.
- CPU software-prefetch MatMul-Bias-ReLU backend candidate and benchmark executable.

Assume checked-in traces and artifacts may be stale unless regenerated.

## Documentation Hygiene

Write portfolio documentation as achievement/result evidence, not as a
step-by-step work diary.

- Do not create root-level `FINAL_*`, `STAGE_*`, `CHANGELOG`, release-note, or
  audit documents for individual slices.
- Root documentation should stay minimal: `README.md` is the public entry point
  and `CLAUDE.md` is the handoff/instruction file. Project status, architecture,
  maturity, publication, and gap documents belong under `docs/project/`.
- Result reports belong under `DOC/result/`. Artifact-local reproduction notes,
  manifests, checksums, commands, and raw evidence belong under the relevant
  `artifacts/` directory.
- Public-facing docs should summarize the achievement, measured result, scope,
  evidence path, and truth boundary. Avoid listing every stage, prompt, file
  added, or implementation step unless it is necessary for reproduction.
- If a new slice creates several overlapping writeups, consolidate them into one
  existing README or one canonical result report, then remove duplicated
  root-level summaries.
- Keep source, scripts, tests, and generated evidence separate. Do not delete
  program files when curating portfolio docs unless explicitly requested.

## Architecture

### MLIR Design Principles

#### MLIR Dialect Design Principle

Always prefer existing upstream MLIR dialects over introducing new custom dialects.

When proposing a compiler architecture or implementation plan:

1. First determine whether the operation can be represented using existing MLIR dialects, including but not limited to:
   - func
   - tensor
   - arith
   - math
   - linalg
   - memref
   - scf
   - affine
   - vector
   - tosa
   - stablehlo (if available in the project)

2. Only introduce a custom dialect when:
   - the semantics cannot be expressed cleanly by existing upstream dialects,
   - or a custom operation carries meaningful compiler/domain semantics that would otherwise be lost,
   - or the custom dialect enables important analyses or optimizations that cannot reasonably be implemented on existing dialects.

3. Never introduce a custom dialect solely because an operation has a different name from ONNX or because creating a new op appears simpler.

4. Every proposal for a new custom dialect or operation must include:
   - why existing MLIR dialects are insufficient,
   - what compiler semantics the custom op represents,
   - what optimization or analysis becomes possible because of it,
   - why this design is preferable to lowering directly into existing MLIR dialects.

5. If no strong justification exists, lower into existing MLIR dialects instead of inventing a new dialect.

#### Architecture Planning Rule

When proposing future compiler phases, always evaluate the following order first:

1. Can this be expressed using an existing MLIR dialect?
2. Can this be represented as metadata or attributes instead of a new operation?
3. Can this be implemented as a compiler pass rather than a new dialect?
4. Only if all of the above are insufficient, propose a new custom dialect or operation.

Custom dialects should be considered the last architectural option, not the default one.

#### Existing Infrastructure First Rule

Before proposing any new compiler component (IR, dialect, pass, schema, runtime abstraction, metadata, planner, etc.), always evaluate whether an existing component can be extended instead.

The preferred order is:

1. Reuse an existing upstream MLIR dialect.
2. Extend an existing project component.
3. Add a compiler pass.
4. Add metadata or attributes.
5. Add a new IR abstraction.
6. Add a new custom dialect.

Every proposal for a new component must explain why each earlier option is insufficient.

Avoid introducing new abstractions solely for organizational convenience.

### Execution-Planning Compiler Pipeline

```text
ONNX / model graph
  ->
Frontend (import / normalize)
  ->
Shared capability profiles
  (HardwareCapability / BackendCapability / KernelLibraryCapability)
  ->
Static optimization
  (Decision Engine: 16-pass serving pipeline)
  ->
ExecutionPlan  [compiler deliverable]
```

Hardware Model and Decision Engine are **sequential modules** in one compiler
pipeline — not parallel tracks.

### Compiler Responsibility

The compiler owns:

- Import / normalize model graph into MLIR
- Lift backend-relevant regions into HIR
- Read shared hardware/backend/kernel capability profiles
- Run hardware-aware static optimization passes (the Decision Engine)
- Select per-op execution plans using static penalty scoring
- Export `execution_plan.json` — the compiler deliverable

The compiler produces the **theoretical best static solution** given declared
capability profiles. It has no visibility into runtime state.

### Compiler Does NOT

- Perform runtime scheduling or dynamic execution ordering
- Manage memory at runtime (no allocation, deallocation, or arena management)
- Execute kernels or dispatch CUDA/Metal/CoreML APIs
- Perform speculative decoding or KV cache management
- Make deployment policy decisions (batching, prefill/decode split, SLO targets)
- Claim measured latency or runtime speedup unless a benchmark produced them
- Materialize IR unless a pass explicitly states it does so

Those responsibilities belong entirely to `heterogeneous-inference-runtime`.

### 16-Pass Serving Pipeline

Each pass answers one planning question and writes structured attrs. No pass
selects a winner until `PlanSelectionPass`. None of the 16 planning passes
materializes IR; the separate `BoundaryMaterializationPass` (run after
`PlanSelectionPass`, before plan export) is the only IR-transforming stage —
see Materialization Status below.

| Pass | Question answered | truth_boundary |
|---|---|---|
| ServingPhaseAnalysis | Prefill or decode? | declared_profile |
| KVLayoutPlanning | KV cache layout and block size? | declared_profile |
| ReplayEligibilityPass | CUDA-graph replayable? | declared_profile |
| ExecutionProviderPlanning | Which backend/API? | declared_profile |
| RepresentationPlanning | What dtype/layout does the backend expect? | declared_profile |
| LayoutPlanning | Which specific tensor layout per op? | declared_profile |
| BoundaryPlanning | Which boundary ops (cast/dequant/layout_transform) required? | declared_profile |
| WeightClassificationPlanning | Weight tensors constant or runtime-variable? | declared_profile |
| QuantizationStrategyPlanning | What quantization strategy per op? | declared_profile |
| KernelAvailabilityPlanning | Does a kernel exist for (op, dtype, layout, quant)? | declared_profile |
| LoweringDecisionPlanning | Direct lower / fallback / unsupported? | declared_profile |
| QuantizedBoundaryRefinement | Refine weight dequant after lowering decisions. | declared_profile |
| AlternativeLoweringPlanning | Legal alternatives when exact kernel is missing? | alternative_lowering_static_not_materialized_not_cost_evaluated |
| CandidateGenerationPass | What executable candidates exist per op? | candidate_generation_static_constraints_not_cost_evaluated |
| CandidateEvaluation | Static relative penalty score per candidate? | candidate_evaluation_static_penalty_not_measured_latency |
| PlanSelection | Which candidate wins? | plan_selection_static_penalty_not_measured_runtime |

### Decision Flow

```text
Backend selected
  -> Representation chosen
  -> Layout planned
  -> Boundary needs identified
  -> Weights classified
  -> Quantization strategy chosen
  -> Kernel availability checked
  -> Lowering decision made
  -> Quantized boundaries refined
  -> Alternative lowering candidates generated (if exact kernel missing)
  -> Candidates evaluated with static penalty (no measured latency)
  -> Plan selected (lowest-penalty evaluated candidate wins)
  -> Execution plan exported
```

Fallback is **not** chosen immediately after a kernel miss. It is a last-resort
candidate considered only after direct kernel, algebraic decomposition,
representation conversion, and other alternative paths have been found unavailable
or invalid.

### Shape-Aware Static Cost Model (shape_cost_model_v2)

CandidateEvaluation (ServingCostModelPass) additionally computes a
shape-aware, dtype-aware, profile-aware estimate per candidate
(`evaluation.shape_cost.*`, promoted to `selected_plan.shape_cost.*` and
exported as the per-op `shape_cost` object): FLOPs from static tensor shapes
(matmul-like 2·M·K·N with inferred K×N weight; normalization 4 ops/elem;
elementwise 1 op/elem), dtype-aware bytes (fp32/fp16/bf16/int8/int4, using
existing `quant.activation_dtype`/`quant.weight_dtype` metadata), arithmetic
intensity, and — only when the target profile declares a `staticCostProfile`
block (peak FLOPs, memory bandwidth; e.g. `nvidia_gtx1650_maxq.json` carries
public-docs theoretical peaks) — roofline time estimates in integer
nanoseconds. Truth boundary:
`static_shape_derived_declared_profile_not_measured_not_runtime_validated` —
a static compiler estimate, never a measured benchmark or latency guarantee.

Unknown op kinds and dynamic shapes fall back to the fixed V1 penalty model,
recorded in `compiler.shape_profile.{op_kind, status, ranking_mode}`.
Ranking is unchanged by default; the opt-in module attr
`serving.cost_model.mode = "shape_aware_v2"` ranks an op's evaluated
candidates by `estimated_total_cost_nanos` when every evaluated candidate
has an estimate (V0 preserved as `evaluation.penalty_score_v0`). Fallback
last-resort and unsupported-never invariants are tier rules and unaffected.

### Memory-Hierarchy-Aware Tile Planning (tile_planning_v1)

`TilePlanningPass` (run by `compile-for-target` before candidate
evaluation; standalone `tile-planning-pipeline`) plans local-memory tile
feasibility for matmul-like ops with static shapes against the profile's
declared `staticCostProfile.localMemoryBytes` (SRAM/shared
memory/scratchpad; optional `cacheLineBytes`, `supportsAsyncCopy`,
`supportsDma`). The `MemoryHierarchyProfile` is **optional declared
metadata** — not every backend exposes local memory or DMA details.
Feasibility runs only when the op kind is supported, the capacity is
declared, and shapes are fully static; when `localMemoryBytes` is missing,
matmul-like ops are stamped `tile.plan.status =
"deferred_missing_memory_hierarchy"` with a `deferred_reason` (exported;
the plan stays valid) — no capacity is ever invented. Otherwise the pass
selects the largest tile from a fixed menu whose A+B+C working set fits
(quant metadata shrinks the B tile), records double-buffer feasibility, the
staging capability as a declared fact (`async_copy_declared` /
`dma_declared` / `declared_unavailable` when declared false /
`unknown_not_declared` when nothing is declared), a reuse-limited
global-traffic estimate, and explains rejections; dynamic
shapes defer with an explicit status. `ServingCostModelPass` annotates the
planned traffic as `compiler.shape_profile.estimated_tiled_memory_cost_nanos`
— annotation only, ranking unchanged. Exported per op as `tile_plan`;
per-op `layout` decisions (existing `LayoutPlanningPass` attrs, including
the `required_input_layout` → `selected_layout` transition) are exported
too. Truth boundary:
`tile_planning_static_local_memory_model_not_measured_not_codegen` — static
planning only: no measured performance, no DMA/async-copy execution, no
codegen, no claim the backend kernel uses this tiling.

### Kernel Selection Framework (kernel_selection_contract_v1)

`KernelSelectionPass` (after tile planning in `compile-for-target`;
standalone `kernel-selection-pipeline`) selects a concrete runtime kernel
per op by matching `RuntimeKernelDescriptor` entries (profile
`runtimeKernels` → `target.runtime_kernels`) on op name × backend × dtype
× quant mode × layout × shape staticness × tile plan × local memory.
Distinct from `KernelAvailabilityPlanningPass` (third-party library
coverage). Non-matches record explicit `rejected_*`/`deferred_*` statuses
with per-descriptor reasons, including
`deferred_no_kernel_library_declared` — never silent. Exported per op as
`kernel_selection`. The registry is honest: exactly ONE kernel is declared
today (Metal RMSNorm f32, `handwritten_runtime`; on the CoreML-primary
A17 Pro plan it rejects with `backend_mismatch`). This is a selection
framework, not kernel coverage; a selection is a runtime contract, never
an execution or performance claim. See `docs/RUNTIME_KERNEL_CONTRACT.md`
for the add-a-kernel checklist.

### Quantization Co-Design (quantization_codesign_contract_v1)

`QuantizationCoDesignPass` (after kernel selection in `compile-for-target`;
standalone `quant-codesign-pipeline`; **inert unless**
`quant.codesign.policy` is set — no existing profile sets it) evaluates
matmul-like constant-weight ops and reports SEPARATE facts: numeric
representation, algorithm declaration (none implemented; forced-AWQ only
*declares* an external artifact), backend legality, concrete runtime-kernel
support (library capability never counts as dispatchable), static
systems-cost comparison, honest accuracy evidence (`no_accuracy_evidence` /
`algorithm_declared_not_calibrated` — the only truthful values), and
materialization status. Policies: `planning_only` / `systems_cost_only` /
`require_dispatchable_kernel` / `require_accuracy_evidence` (always defers
today). Unknown granularity/group/axis/symmetric/scale/zero-point metadata
is omitted, never defaulted. `quant.strategy` semantics unchanged;
`quant_codesign.est.*` never affects CandidateEvaluation/PlanSelection
(ranking-invariance test enforces byte-identical signals). Exported per op
as `quantization_codesign`. Key finding the model surfaces honestly:
without a kernel that consumes quantized weights, the materialized dequant
intermediate makes weight-only quantization LOSE on traffic. See
`docs/QUANTIZATION_CODESIGN.md` for the full cost-term table.

### Hardware Model Layers

Three distinct layers with different scopes and truth_boundary values:

- **HardwareCapability**: theoretical hardware support from public docs or declared device profiles. `truth_boundary = public_docs | declared_profile`.
- **BackendCapability**: what the backend/API/compiler actually exposes, which may be more restrictive than hardware. `truth_boundary = declared_profile`.
- **KernelLibraryCapability**: actual kernel availability for a specific (op, dtype, layout, quant_mode) tuple. `truth_boundary = declared_profile | measured_profile`.

No layer claims measured hardware performance unless a benchmark explicitly produced it.

### Materialization Status

Planning is implemented. Cast-boundary materialization is implemented.
All other IR materialization is intentionally deferred.

Planning means (implemented):
- annotate `selected_plan.*` per op
- annotate `quant.*`, `kernel.*`, `lowering.*`, `alternative.*`
- export execution plan JSON / runtime contract

Materialization implemented (`BoundaryMaterializationPass`, after
`PlanSelectionPass` and before plan export in `compile-for-target`;
standalone pipeline `boundary-materialization-pipeline`):
- inserting `hir.cast` where the plan set `boundary.cast_required = true`
  (float-to-float precision only), with provenance attrs
  (`compiler.materialized`, `materialized.by/from_decision/of_op`) and
  `truth_boundary = compiler_materialized_boundary_op_not_runtime_executed`.
  Uses are redirected, function result types updated; ops with an
  `unsupported` selected plan are never materialized; malformed planning
  attrs raise diagnostics, never silent no-ops. The exported plan reports
  `materialized_boundary_ops` / `deferred_boundary_ops` per op.

Materialization still deferred (recorded as deferred, never faked):
- inserting `hir.dequantize` / `hir.requantize` (needs scale/zero-point
  metadata the planner does not produce)
- inserting `hir.layout_transform` (no such hir op exists yet)
- replacing `gelu` with primitive ops
- rewriting graph structure

The 16 planning passes remain annotation-only; `BoundaryMaterializationPass`
is the only pass that modifies IR structure, and only for the cast subset.

### Apple/CoreML Quantization Demo

The Apple/CoreML demo demonstrates:
- Apple/CoreML-style quantization lowering decisions (`declared_profile`)
- weight classification preventing incorrect `weight_only_int8` on runtime activations
- kernel availability vs backend capability separation
- per-op plan selection and export

The demo does NOT claim:
- real ANE kernel internals or layout proofs
- measured Core ML performance
- a CoreML profile as evidence of ANE internal execution
- fallback is selected immediately after a kernel miss

### Current Compiler Pipeline (Qwen / LLM Serving)

The abstract pipeline above (Frontend → Shared Capability Profiles → Static
Optimization → ExecutionPlan) is concretely realized today, for the Qwen/LLM
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

Each stage, concretely: HF model → `tools/export_qwen_onnx.py` (optional, HF
Optimum export) → a real `.onnx` protobuf file → `tools/onnx_graph_to_facts.py`
(Python frontend adapter) → `GraphFacts` JSON → `qwen-onnx-to-serving-mlir`
(C++) → full per-layer-expanded Serving MLIR → `LLMFrontendNormalizationPass`
+ the 16-pass serving pipeline → `execution_plan.json`. See "GTX 1650 / Qwen
vLLM Serving Demo," "ONNX Graph Import Path (Phase 1)," and "Real ONNX
Protobuf Bridge (Phase 2)" below for the full detail and truth boundary of
every stage.

### What GraphFacts Is (and Is Not)

`GraphFacts` is NOT:
- a model specification (that is the separate, legacy `qwen-to-serving-mlir`
  / `configs/models/qwen_0_5b_spec.json` path — see below).
- fake graph generation. The hand-authored fixture
  (`configs/models/qwen_0_5b_onnx_graph_facts.json`) is one specific,
  honestly-labeled instance of a `GraphFacts` document that predates the
  bridge and remains as regression coverage — it is not what `GraphFacts` as
  a schema/boundary concept inherently is.

`GraphFacts` IS:
- a frontend adapter representation — the output contract every frontend
  (today: the ONNX bridge; potentially, later: Torch FX / StableHLO
  adapters) must produce.
- the compiler input boundary — `qwen-onnx-to-serving-mlir` and everything
  downstream (`LLMFrontendNormalizationPass`, the 16-pass pipeline,
  `ExecutionPlan` export) consumes only this schema and needs no changes
  when the source of `GraphFacts` changes.
- derived from a real ONNX graph when produced by
  `tools/onnx_graph_to_facts.py` (real protobuf parsing, real initializer
  names/shapes/dtypes) — not fabricated, though role assignment within it is
  still Qwen2-pattern-matched, not a general graph interpreter (see below).
- intentionally isolated from compiler policy — it carries structural facts
  (layer count, dims, dtype, per-role presence) and declared metadata
  (`positional_encoding`), never planning decisions; quantization, layout,
  and lowering policy live entirely downstream in the 16-pass pipeline.

### Architecture Philosophy

The compiler is divided into three layers:

- **Frontend** — parses foreign model formats. Currently: ONNX, through a
  Python frontend adapter (`tools/onnx_graph_to_facts.py`) pattern-matched to
  Qwen2's HuggingFace naming convention. Future: Torch FX / StableHLO
  adapters (not implemented — see "Future Architecture" below).
- **Compiler** — semantic canonicalization using existing upstream MLIR
  dialects by default. Existing `llm.*`/`serving.*` vocabulary may be used
  only where it carries justified compiler/domain semantics under the MLIR
  Design Principles above. Today recognition is done partly in the Python
  frontend adapter and partly in `LLMFrontendNormalizationPass`; planning
  uses the 16-pass serving pipeline, followed by execution plan generation
  (`PlanSelectionPass` + exporter).
- **Runtime** — executes the produced plan. This belongs entirely to the
  sibling `heterogeneous-inference-runtime` project; this repo never
  executes a plan, only produces one.

### Current Capability

Current:
- Reads real ONNX graphs (`tools/onnx_graph_to_facts.py`, real `.onnx`
  protobuf parsed via the `onnx` Python package).
- Extracts graph structure and model metadata (per-layer role presence,
  dimensions, dtype, RoPE/lm_head-tying signals — from real initializer
  names and shapes, never guessed; hard failure on any gap).
- Converts ONNX-derived information into `GraphFacts` (the frontend
  boundary).
- Generates Serving MLIR (`qwen-onnx-to-serving-mlir`, full per-layer
  expansion).
- Generates `ExecutionPlan` (the unchanged 16-pass planning pipeline and
  exporter).

Not yet:
- A general ONNX compiler — only Qwen2's specific HuggingFace naming
  convention is recognized; any other model family, or a renamed/fused
  parameter, is out of scope and fails rather than guessing.
- General ONNX→MLIR lowering — there is no generic ONNX-dialect import; role
  recognition is Python name-pattern-matching, not structural graph-edge
  tracing over an imported IR.
- ONNX-MLIR integration — no `onnx-mlir` dependency exists in this repo.
- A Torch FX frontend — not started.
- A StableHLO frontend — not started. (The existing "StableHLO-compatible"
  Linalg/Arith decomposition support described elsewhere in this file is a
  separate, unrelated demo path, not a StableHLO import for this pipeline.)

### Future Architecture

Long-term target, if a second/third frontend is ever added:

```text
ONNX / Torch FX / StableHLO
            │
            ▼
Frontend Parser
            │
            ▼
Raw Import Graph
            │
            ▼
Semantic Canonicalization
            │
            ▼
Serving MLIR
            │
            ▼
Planning Passes
            │
            ▼
ExecutionPlan
```

- Semantic recognition (role classification, RoPE/tied-embedding detection)
  should eventually move out of Python and into a compiler-side MLIR pass,
  following the pattern-matching technique `StableHLOCompatibleRMSNormPattern`
  already demonstrates in `mlir_passes/lib/MatMulBiasReluFusionPass.cpp` for
  a different pattern.
- Frontend adapters should eventually become thin format parsers only —
  pure parsing and graph traversal, no architecture knowledge — once that
  compiler-side recognition pass exists.
- `GraphFacts` is the **current transition layer**: it already isolates
  planning passes from frontend format details, which is the part of this
  target that is already true. What is not yet true is that role assignment
  happens in the compiler rather than in the Python adapter that produces
  `GraphFacts`.

This is a target architecture, not a current capability — see "Current
Capability" above and `docs/future_work.md` for status.

### GTX 1650 / Qwen vLLM Serving Demo

`tools/run_qwen_compiler_pipeline.sh` produces
`artifacts/qwen/execution_plan.json` from
`configs/models/qwen_0_5b_spec.json` (architecture-only: layer count, hidden
size, heads — not imported model weights, `truth_boundary =
declared_model_config_not_full_graph_import`) and
`configs/target_profiles/nvidia_gtx1650_maxq.json`. This is the compiler-side
source artifact consumed by `heterogeneous-inference-runtime`'s measured A/B
vLLM benchmark (`results/qwen_no_quant/` in that repo):

- Current per-op quantization decision for this target is `fp16_fallback`
  (quantization `none`), matching `nvidia_gtx1650_maxq.json`'s declared
  `supportedQuantModes: ["none"]` on both backends — Turing (cc 7.5) has no
  native INT4 tensor-core path, so this profile cannot honestly claim AWQ/GPTQ
  support today.
- Compiler-guided no-quant (B) uses the same original Qwen weights as the
  manual baseline (A); measured E2E delta is within ~1% across repeatability
  trials, i.e. benchmark noise, not a speedup claim.
- A minimal quantized Phase C (AWQ only; GPTQ not implemented) now exists:
  `tools/export_qwen_awq.py` (real AutoAWQ export edge tool, fails clearly
  without AutoAWQ), `configs/target_profiles/nvidia_gtx1650_maxq_awq_forced.json`
  (an experimental forced-quant profile variant — per-op `backendCapabilities`
  are unchanged from the no-quant profile; only a new `forcedQuantization`
  block is added), and `tools/run_qwen_awq_compiler_pipeline.sh` (produces
  `artifacts/qwen_awq_plan/execution_plan.json`, whose
  `global_decisions.quantization` carries `strategy: weight_only_int4`,
  `algorithm: awq`, `quantized_model_artifact_ref`, and
  `truth_boundary: experimental_forced_quant_not_native_int4_support_on_gtx1650`).
  The runtime-side materializer (`heterogeneous-inference-runtime`) emits
  `--quantization awq` and points `--model`/`--tokenizer` at the quantized
  artifact path. No measured C results exist yet — this development machine
  has no CUDA and no AutoAWQ; `scripts/run_qwen_quant_benchmark.sh` in that
  repo materializes all three (A/B/C) server commands but stops short of
  running C until the AWQ artifact and a CUDA host are available. See
  `docs/future_work.md` for what remains (GPTQ, accuracy evaluation, measured
  results).
- `docs/EXECUTION_PLAN_SCHEMA.md`'s AWQ JSON example is a schema illustration
  that predates this Phase C work; it happens to match the real
  `strategy`/`algorithm` field values now produced for the forced-AWQ profile,
  but do not read it as evidence of measured behavior — it is still not a
  real execution trace.
- `qwen-to-serving-mlir` (above) is the **legacy/scaffold ModelSpec path**: a
  hand-templated single op block per phase, never looped over `num_layers`.
  It does not represent a real per-layer graph. See below for the real
  graph-import path.

### ONNX Graph Import Path (Phase 1)

Compiler philosophy: input is moving toward real, per-layer-expanded graphs
(ONNX-shaped today, a real ONNX/HF import later) rather than hand-authored
model specifications. The ModelSpec path (`qwen-to-serving-mlir`, previous
section) is the legacy/scaffold generator being phased out, not the long-term
architecture.

`mlir_passes/tools/qwen-onnx-to-serving-mlir` is the target-architecture
frontend: it reads `configs/models/qwen_0_5b_onnx_graph_facts.json` and emits
full per-layer-expanded, flat, unrolled serving MLIR (`serving.layer_index =
0..num_layers-1`, real SSA chaining), not a single hand-templated block. It
emits the raw pre-canonicalization attention pattern per layer;
`LLMFrontendNormalizationPass` was generalized to do a localized
per-occurrence rewrite (real q/k/v operands, not a dummy placeholder) instead
of a whole-function erase, so it canonicalizes each layer's occurrence
independently.

Truth boundary — do not overclaim this:
- `configs/models/qwen_0_5b_onnx_graph_facts.json` is a **hand-authored
  fixture**, not a real ONNX protobuf parse (`truth_boundary =
  onnx_shaped_fixture_not_real_onnx_protobuf_import`).
- `tools/export_qwen_onnx.py` (Python edge tooling, not compiler-core)
  attempts a real HF Optimum export + real `onnx`-package introspection when
  installed, but its output is a diagnostic report only, **not yet consumed**
  by the C++ importer.
- Linear/weight-bearing ops (`q_proj`/`k_proj`/`v_proj`/`o_proj`/`mlp`/
  `lm_head_proj` here; `qkv_projection`/`mlp` from the legacy ModelSpec path)
  are stamped with an explicit `serving.quantizable = true` attribute at
  emission time. `QuantizationStrategyPlanningPass`/
  `WeightClassificationPlanningPass` key off that attribute first, falling
  back to a small generic name-fragment match (`matmul`, `conv`, `gemm`,
  etc.) only for ops with no explicit marker — frontend-specific naming stays
  behind the importer instead of leaking into generic planning passes as
  op-name substring matches.
- The exported `ExecutionPlan` is **verbose** in Phase 1 (no
  `layer_range`/`layer_count` compression, no JSON schema change) — ~170
  `per_op_decisions` entries per phase for 24 layers, vs. the legacy path's
  single entry. Compression is future work and export-time only; the
  compiler's internal IR model is always full expansion. See
  `docs/future_work.md`.

### Real ONNX Protobuf Bridge (Phase 2 — frontend adapter, not a general importer)

`GraphFacts` is the frontend boundary / adapter seam: `qwen-onnx-to-serving-mlir`
never changed to support this milestone, because the adapter's whole job is
to emit the same `GraphFacts` JSON schema Phase 1 already established.

```text
HF / external model source -> ONNX protobuf
  -> tools/onnx_graph_to_facts.py (Python frontend adapter)
  -> GraphFacts JSON -> qwen-onnx-to-serving-mlir (unchanged) -> Serving MLIR
```

`tools/onnx_graph_to_facts.py` loads a real `.onnx` file with the `onnx`
package and reads real protobuf structure (node op types, initializer
names/shapes/dtypes — never tensor values). It classifies per-layer roles
by matching real initializer names against **Qwen2's specific HuggingFace
naming convention** (`model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`,
`model.layers.{i}.mlp.{gate,up,down}_proj.weight`, `*layernorm*`,
`model.embed_tokens.weight`, `lm_head.weight` or tied embedding), derives
`num_layers`/`hidden_size`/`intermediate_size`/`vocab_size`/`dtype` from
real shapes/dtypes, and detects RoPE and lm_head tying from real graph
signals. `num_attention_heads`/`num_key_value_heads`/
`max_position_embeddings` are not recoverable from graph structure alone —
read from an HF `config.json` next to the `.onnx` file, or explicit CLI
overrides, never guessed. Any missing per-layer role is a hard failure
(`OnnxGraphToFactsError`), not a silent guess or omission.

**Not a general ONNX importer** — only Qwen2's exact naming convention is
recognized; the per-layer *operator sequence* is still a declared Qwen2
architecture template (matching the fixture), not derived from raw graph
edges. Truth boundary emitted:
`"onnx_protobuf_parsed_pattern_matched_not_general_graph_interpreter"`.

`tools/validate_onnx_graph_facts.py` validates any GraphFacts document:
real bridge output (has a `provenance` field) gets full per-layer
completeness checks (num_layers matches parsed layer count, every layer has
q/k/v/o_proj + mlp roles, embedding/final_norm/lm_head detected or cleanly
reported as tied) that fail hard on any gap; the hand-authored fixture (no
`provenance` field) gets schema-only checks with per-layer completeness
explicitly reported as skipped, not silently passed.

**Current status:** the hand-authored fixture and the real bridge coexist —
the fixture is kept as fast, deterministic, network-free regression
coverage (`tests/test_onnx_graph_facts_fixture_regression.py`), not
replaced. RoPE stays absorbed (not a distinct op); when detected it is
stamped as a function-level `serving.positional_encoding = "rope"` attribute
on `qwen_prefill`/`qwen_decode` — an optional GraphFacts field the C++
importer reads additively (absent for the fixture, which emits identical
MLIR as before this change).

Not implemented (do not claim otherwise): general ONNX import for arbitrary
model families, ONNX-MLIR (or equivalent) frontend integration, a Torch FX
adapter, a StableHLO adapter, decode-with-past graph handling, and
layer-range/export-time compression. See `docs/future_work.md`.

### Shared Capability Profiles

The compiler and runtime intentionally read from the same capability data. Both
repositories are currently centered on the same development machine. The three
shared layers:

- **HardwareCapability** — theoretical hardware support. `truth_boundary = public_docs | declared_profile`. Neither repo may claim measured performance here unless a benchmark produced it.
- **BackendCapability** — what the backend/API actually exposes, which may be more restrictive than hardware. `truth_boundary = declared_profile`.
- **KernelLibraryCapability** — actual kernel availability for (op, dtype, layout, quant_mode). `truth_boundary = declared_profile | measured_profile`.

The runtime may extend these with a fourth layer:

- **MeasuredSupport** — runtime-measured latency/throughput evidence collected during execution. `truth_boundary = measured_profile`. This layer belongs to the runtime only; the compiler never writes it.

Neither repository maintains separate hardware truths. All shared profile data
must be consistent across both.

### Truth Boundary

The compiler makes only static optimization claims:

- No measured latency values unless an explicit benchmark produced them.
- No runtime speedup claims.
- No deployment performance claims.
- No claims about dynamic execution quality (batching efficiency, SLO hit rate, etc.).

Every planning annotation carries a `truth_boundary` field. If no benchmark
exists, the claim is `declared_profile` or `public_docs` — not `measured_profile`.

## Environment Policy

- Use the repo's `.venv` for all Python tooling; do not rely on a system Python.
- Install optional Python dependencies (e.g. `jax[cpu]`, `torch_mlir`) into `.venv` only when a specific demo requires them.
- For MLIR work, use a single consistent LLVM/MLIR build for CMake, `mlir-opt`, and `FileCheck` (see MLIR Notes).
- Treat missing toolchain/environment dependencies as expected in a fresh checkout; report them clearly rather than working around them silently.

## Working Rules

- Do not modify source code unless explicitly asked.
- Preserve user changes in the working tree.
- Do not refactor opportunistically.
- Do not change tests unless the task explicitly calls for it.
- Do not invent benchmark numbers.
- If metrics are estimated, label them estimated.
- Explain assumptions explicitly.
- Explain changes after implementation.

## Coding Preferences

- Python 3.11.
- Prefer dataclasses.
- Avoid unnecessary classes.
- Keep functions under 100 lines when practical.
- Use type hints.
- Prefer simple modular design.
- Avoid over-engineering.
- Composition over inheritance.
- No giant classes.
- Write tests for non-trivial logic.
- Run tests after changes.

## C++ Notes

- The main project is C++17 and built with CMake.
- The core library target is `mlcompiler`.
- Platform-specific paths exist for Apple Metal and optional CUDA.
- The generic backend API is intentionally small and not sufficient for full device runtime semantics.
- `MockGPUBackend` dispatches CPU kernels.
- Generic `MetalBackend` logs dispatch/device information; do not describe it as full graph-kernel execution.

## MLIR Notes

- `mlir_passes/` is separate from the custom toy graph IR.
- It uses LLVM/MLIR CMake packages, TableGen, a `hir` dialect, pass registration, and FileCheck tests.
- Use the same LLVM/MLIR build for CMake, `mlir-opt`, and `FileCheck`.
- Environmental toolchain failures are common; report them clearly.

## Artifact Notes

- `trace/`, `artifacts/apple_demo/`, and `integration_bundle/apple_demo_artifacts/` contain generated JSON outputs.
- Treat candidate serving latency/throughput and planner values as estimated unless a current benchmark produced them.
- Prefer regenerating artifacts with the relevant tool script before relying on them.
- Keep schema/version changes explicit.

## Suggested Verification

`scripts/check.sh` is the canonical baseline validation entrypoint. It runs:

```bash
scripts/check.sh
```

It does not install any system dependencies (no `brew`/`apt-get` calls) — if a
required tool is missing, the relevant step fails and that failure should be
reported as-is, not worked around.

### Known baseline validation gap: `metal_rmsnorm_plan_dispatch`

On Apple builds, `bash scripts/check.sh` currently surfaces a known issue in
the `metal_rmsnorm_plan_dispatch` test (`apps/run_metal_rmsnorm_plan.mm`,
registered in `CMakeLists.txt`): it requires
`trace/metal_rmsnorm_execution_plan.json`, which is generated only by the MLIR
pipeline (`tools/run_metal_rmsnorm_compiler_pipeline.sh`), not by the baseline
CMake build. In a baseline checkout that file is absent, so this test
is reported as skipped rather than exercising the dispatch path.

- Do not fabricate `trace/metal_rmsnorm_execution_plan.json` to make this test
  pass — it is an MLIR-pipeline-generated artifact, not baseline CMake output.
- Do not describe the dispatch logic as verified unless the MLIR pipeline has
  produced `trace/metal_rmsnorm_execution_plan.json` and the dispatch test has
  run against that artifact.

For MLIR changes, when LLVM/MLIR tools are installed:

```bash
cmake -S mlir_passes -B build-mlir \
  -DMLIR_DIR="$(brew --prefix llvm)/lib/cmake/mlir" \
  -DLLVM_DIR="$(brew --prefix llvm)/lib/cmake/llvm"
cmake --build build-mlir
tools/run_mlir_pass_tests.sh
```

For LLM serving artifacts:

```bash
tools/run_llm_serving_artifact_pipeline.sh
```

If a verification command cannot run because dependencies are missing, state that directly.

## Common Commands

- `scripts/check.sh` — canonical CMake build + CTest baseline check (see Suggested Verification above).
- `tools/run_metal_rmsnorm_compiler_pipeline.sh` — produces `trace/metal_rmsnorm_execution_plan.json`, required by the `metal_rmsnorm_plan_dispatch` CTest target.
- `tools/run_metal_rmsnorm_end_to_end.sh` — runs the Apple Silicon MLIR-to-Metal RMSNorm path end to end when the toolchain is available.
- `python3 tools/validate_metal_rmsnorm_path.py` — validates Metal RMSNorm generated artifacts and dispatch evidence.
- `python3 tools/run_stablehlo_subset_pipeline.py` — exercises the StableHLO-compatible textual subset importer.
- `.venv/bin/python tools/run_jax_stablehlo_pipeline.py` — exports supported JAX functions to StableHLO and runs the local HIR/LLVM path when JAX is installed.
- `.venv/bin/python tools/run_torch_mlir_tiny_transformer_probe.py` — probes Torch-MLIR export when `torch_mlir` is installed.
- `.venv/bin/python tools/run_iree_stablehlo_subset_comparison.py` — runs the optional IREE comparison path.
- `build/benchmark_prefetch_matmul` — benchmarks the CPU software-prefetch MatMul-Bias-ReLU candidate after CMake build.
- `tools/run_mlir_pass_tests.sh` — runs the MLIR pass plugin FileCheck tests against `build-mlir`.
- `tools/run_llm_serving_artifact_pipeline.sh` — runs the full MLIR-to-LLM-artifacts pipeline end to end.
- `python3 tools/validate_compiler_artifacts.py` — validates generated `trace/cv_*.json` compiler/runtime artifacts.
- `python3 tools/validate_llm_serving_artifacts.py` — validates generated `artifacts/apple_demo` LLM serving artifacts.
- `tools/check_openxla_toolchain.py` — checks optional StableHLO tooling availability.

## Compiler Core Policy: Zero Python / Zero JSON

The compiler core must be C++/MLIR-first.

Python and JSON are allowed only at the edges:
- Python: legacy prototypes, validation tooling, debug scripts, regression comparison, runtime/demo helpers.
- JSON: debug dumps, validation artifacts, temporary runtime/demo interchange.

They must not be the source of truth for new compiler functionality.

New compiler-core work should be implemented in:
- C++
- MLIR passes
- TableGen where appropriate
- FileCheck / CTest tests

Runtime metadata must originate from:
MLIR attributes or C++ RuntimeMetadataContract.

If JSON is emitted, it is a serialization/debug format only.

Do not add new Python+JSON planner logic as the primary compiler implementation.

## Portfolio-Level Policy

When this repository is maintained inside the `systems-portfolio` wrapper, follow the root `CLAUDE.md` for shared documentation hierarchy, benchmark honesty, and Git authorship rules. Keep this file focused on repository-specific capabilities, truth boundaries, and validation commands.
