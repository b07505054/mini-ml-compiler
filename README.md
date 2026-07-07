## ML Graph Compiler and Runtime Infrastructure

This repository is a prototype **execution-planning ML compiler**. It reuses
existing MLIR infrastructure for generic graph semantics, lifts
backend-relevant regions into a decision-oriented HIR, and produces an
explainable hardware-aware execution plan consumed by heterogeneous runtimes.

The repository contains two components — one primary, one secondary:

1. **MLIR compiler** (primary): HIR/CV/Serving dialects, the 16-pass
   hardware-aware serving pipeline, quantization planning, INT8 island
   propagation, Q/DQ canonicalization, fusion/lowering,
   capability-gated operator-selection, verification, and execution-plan
   or artifact export. This is the execution-planning compiler.
2. **C++ runtime demo harness** (secondary): a small custom graph runtime used
   for demos, benchmark bridges, memory-planning experiments, kernel-registry
   examples, and backend-sandbox experiments. It is not a parallel compiler
   track — it is a local demo harness and benchmark bridge.

These are sequential modules in one pipeline, not parallel tracks:
Frontend → Shared Capability Profiles → Static Optimization → Execution Plan.
The optimization step is the 16-pass serving pipeline described below.

The C++ runtime code in this repository is not the production distributed
runtime story. New runtime/deployment features should go to the sibling
`heterogeneous-inference-runtime` project.

**The compiler produces the theoretical best static solution.
The runtime produces the best dynamic execution.**

The compiler never performs:
- runtime scheduling or dynamic execution ordering
- memory management (allocation, deallocation, arena management)
- speculative decoding or KV cache management
- deployment policy (batching, prefill/decode split, SLO enforcement)

Those belong entirely to `heterogeneous-inference-runtime`.

Architecture:

```text
ONNX / model graph
  ->
Frontend (import / normalize)
  ->
Shared capability profiles
  (HardwareCapability / BackendCapability / KernelLibraryCapability)
  ->
Static optimization — Decision Engine (16-pass serving pipeline)
  ->
ExecutionPlan  [compiler deliverable]
```

For the **CV pipeline** specifically, ONNX import, backend kernel mapping,
dynamic-shape support, and PocketChef visualization of the CV plan are future
work — see "CV Compiler Pipeline" below. For the **Qwen/LLM serving
pipeline**, a real ONNX frontend adapter is implemented (not a general ONNX
importer) — see "Current Compiler Pipeline" and "Real ONNX Protobuf Bridge"
below.

### Honest Claims / Not Claimed

Honest claims:

- The compiler performs static optimization and produces `execution_plan.json`
  as its deliverable — a hardware-aware planning artifact.
- Quantization support is implemented at the compiler-pass metadata/legality
  level (static planning, `declared_profile` truth boundary).
- The local C++ runtime is a demo harness and benchmark bridge.
- Compiler and runtime read from the same shared capability profile data.

Not claimed:

- Measured latency, runtime speedup, or deployment performance.
- Full INT8 graph runtime execution.
- Production calibration or automatic quantization-parameter generation.
- Complete ONNX import.
- Runtime scheduling, memory management, speculative decoding, or KV cache
  management — these belong to `heterogeneous-inference-runtime`.
- The C++ runtime harness and the Python runtime project are one production
  system.

### Current Compiler Pipeline (Qwen / LLM Serving)

The abstract diagram above (Frontend → Shared Capability Profiles → Static
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

Concretely: an HF model → `tools/export_qwen_onnx.py` (optional HF Optimum
export) → a real `.onnx` protobuf file → `tools/onnx_graph_to_facts.py`
(Python frontend adapter) → `GraphFacts` JSON → `mlir_passes/tools/qwen-onnx-to-serving-mlir`
(C++, unchanged since Phase 1) → full per-layer-expanded Serving MLIR →
`LLMFrontendNormalizationPass` + the 16-pass serving pipeline →
`execution_plan.json`. See "ONNX Graph Import Path (Phase 1)" and "Real ONNX
Protobuf Bridge (Phase 2)" below for the full detail and truth boundary of
every stage.

### What GraphFacts Is (and Is Not)

`GraphFacts` is **NOT**:
- a model specification — that's the separate, legacy `qwen-to-serving-mlir`
  / `configs/models/qwen_0_5b_spec.json` path (see below), which this repo
  is moving away from, not toward.
- fake graph generation. The hand-authored fixture
  (`configs/models/qwen_0_5b_onnx_graph_facts.json`) is one specific,
  honestly-labeled instance of a `GraphFacts` document that predates the
  real bridge and remains as regression coverage — it is not what
  `GraphFacts` as a schema/boundary concept inherently is.

`GraphFacts` **IS**:
- a frontend adapter representation — the output contract any frontend
  (today: the ONNX bridge; potentially later: Torch FX / StableHLO adapters)
  must produce.
- the compiler input boundary — `qwen-onnx-to-serving-mlir` and everything
  downstream needs no changes when the source of `GraphFacts` changes.
- derived from a real ONNX graph when produced by
  `tools/onnx_graph_to_facts.py` — real protobuf parsing, real initializer
  names/shapes/dtypes, not fabricated (though role assignment within it is
  still Qwen2-pattern-matched, not a general graph interpreter — see below).
- intentionally isolated from compiler policy — it carries structural facts
  and declared metadata, never planning decisions; quantization, layout, and
  lowering policy live entirely downstream in the 16-pass pipeline.

### Architecture Philosophy

The compiler is divided into three layers:

- **Frontend** — parses foreign model formats. Currently ONNX, through a
  Python frontend adapter (`tools/onnx_graph_to_facts.py`) pattern-matched to
  Qwen2's HuggingFace naming convention. Future: Torch FX / StableHLO
  adapters (not implemented — see "Future Architecture" below).
- **Compiler** — semantic canonicalization (recognizing raw graph structure
  as the `llm.*`/`serving.*` vocabulary), planning (the 16-pass serving
  pipeline), and execution plan generation.
- **Runtime** — executes the produced plan. This belongs entirely to the
  sibling `heterogeneous-inference-runtime` project; this repo never
  executes a plan, only produces one.

### Current Capability

Current:
- ✓ Reads real ONNX graphs (`tools/onnx_graph_to_facts.py`, real `.onnx`
  protobuf parsed via the `onnx` Python package).
- ✓ Extracts graph structure and model metadata (per-layer role presence,
  dimensions, dtype, RoPE/lm_head-tying signals — from real initializer
  names and shapes, never guessed; hard failure on any gap).
- ✓ Converts ONNX-derived information into `GraphFacts` (the frontend
  boundary).
- ✓ Generates Serving MLIR (`qwen-onnx-to-serving-mlir`, full per-layer
  expansion).
- ✓ Generates `ExecutionPlan` (the unchanged 16-pass planning pipeline and
  exporter).

Not yet:
- ✗ A general ONNX compiler — only Qwen2's specific HuggingFace naming
  convention is recognized; other model families, or renamed/fused
  parameters, are out of scope and fail rather than guess.
- ✗ General ONNX→MLIR lowering — no generic ONNX-dialect import; role
  recognition is Python name-pattern-matching, not structural graph-edge
  tracing over an imported IR.
- ✗ ONNX-MLIR integration — no `onnx-mlir` dependency exists in this repo.
- ✗ A Torch FX frontend — not started.
- ✗ A StableHLO frontend — not started (the existing "StableHLO-compatible"
  Linalg/Arith decomposition demo elsewhere in this repo is unrelated to this
  pipeline).

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
- Frontend adapters should eventually become thin format parsers only — pure
  parsing and graph traversal, no architecture knowledge — once that
  compiler-side recognition pass exists.
- `GraphFacts` is the **current transition layer**: it already isolates
  planning passes from frontend format details; what's not yet true is that
  role assignment happens in the compiler rather than in the Python adapter
  that produces `GraphFacts`.

This is a target architecture, not a current capability — see "Current
Capability" above and `docs/future_work.md` for status.

## Latest Milestones

### 16-Pass Hardware-Aware Serving Pipeline (primary current work)

A complete 16-pass execution-planning pipeline in `mlir_passes/lib/serving/`,
with related HIR/quantization helpers in `mlir_passes/lib/`. Each pass
annotates ops with structured MLIR attrs without modifying IR structure. No
pass selects a winner until `PlanSelectionPass`.

| Pass | Attr prefix(es) written |
|---|---|
| ServingPhaseAnalysis | `serving.*` |
| KVLayoutPlanningPass | `kv.*` |
| ReplayEligibilityPass | `replay.*` |
| ExecutionProviderPlanningPass | `execution_provider.*` |
| RepresentationPlanningPass | `representation.*` |
| LayoutPlanningPass | `layout.*` |
| BoundaryPlanningPass | `boundary.*` |
| WeightClassificationPlanningPass | `weight.*` |
| QuantizationStrategyPlanningPass | `quant.*` |
| KernelAvailabilityPlanningPass | `kernel.*` |
| LoweringDecisionPlanningPass | `lowering.*` |
| QuantizedBoundaryRefinementPass | `boundary.*` (refined) |
| AlternativeLoweringPlanningPass | `alternative.*` |
| CandidateGenerationPass | `compiler.candidates.*` |
| CandidateEvaluationPass | `compiler.evaluated_candidates.*` |
| PlanSelectionPass | `selected_plan.*`, `compiler.selected_candidates` |

Truth boundaries: every planning attr carries a `truth_boundary` field
(`public_docs`, `declared_profile`, `measured_profile`, or
pass-specific static-constraint labels). No claim presents static penalty
scores as measured hardware latency.

### Qwen 2.5-0.5B / GTX 1650 vLLM Serving Demo

`tools/run_qwen_compiler_pipeline.sh` runs `qwen-to-serving-mlir` then
`compile-for-target` against `configs/models/qwen_0_5b_spec.json` (an
architecture-only spec — layer count, hidden size, heads; not imported model
weights) and `configs/target_profiles/nvidia_gtx1650_maxq.json`, producing
`artifacts/qwen/execution_plan.json`. That file is the compiler-side source
artifact for the sibling `heterogeneous-inference-runtime` repo's measured A/B
vLLM benchmark:

- **A (baseline):** original Qwen weights, hand-written vLLM config,
  quantization none. Default vLLM config OOMs on this 4 GB card; a manually
  tuned conservative config is the measured baseline.
- **B (compiler no-quant):** same original Qwen weights, but vLLM runtime
  flags are materialized from `execution_plan.json` via
  `deployment/execution_plan/path_builder.py` ->
  `deployment/vllm_adapter/config_materializer.py` in the runtime repo.
  Quantization is `none` (per-op `fp16_fallback`, matching this target
  profile's declared `supportedQuantModes: ["none"]`). Measured E2E delta vs.
  the conservative manual baseline is within ~1% across three repeatability
  trials — treated as benchmark noise, not a speedup claim.
- **C (compiler quant, minimal, AWQ only — not yet measured):** an
  AWQ Qwen checkpoint (`tools/export_qwen_awq.py`, real AutoAWQ export,
  fails clearly without the AutoAWQ dependency) plus an experimental
  forced-quant target profile
  (`configs/target_profiles/nvidia_gtx1650_maxq_awq_forced.json`) whose
  per-op `backendCapabilities` are unchanged from the no-quant profile — it
  does not claim GTX 1650 (Turing, cc 7.5) gained native INT4 Tensor Core
  support. `tools/run_qwen_awq_compiler_pipeline.sh` produces
  `artifacts/qwen_awq_plan/execution_plan.json` with
  `global_decisions.quantization = {strategy: weight_only_int4, algorithm:
  awq, quantized_model_artifact_ref, truth_boundary:
  experimental_forced_quant_not_native_int4_support_on_gtx1650}`. GPTQ is not
  implemented. No measured C results exist yet — this repo's development
  machine has no CUDA and no AutoAWQ, so `scripts/run_qwen_quant_benchmark.sh`
  (runtime repo) materializes all three (A/B/C) vLLM server commands but
  stops after materialization until run on a CUDA-capable host with the AWQ
  artifact present.

Measured evidence for A/B lives in `heterogeneous-inference-runtime` under
`results/qwen_no_quant/`, not in this repo; materialized (not yet measured)
C commands live under that repo's `results/qwen_quant/`. See
`docs/EXECUTION_PLAN_SCHEMA.md` and `docs/future_work.md` for the schema and
what remains for Phase C.

`qwen-to-serving-mlir` (above) is now the **legacy/scaffold ModelSpec path**:
it hand-templates a single fixed op block per phase from 9 declared scalar
fields and never loops over `num_layers`, so it cannot represent a real
per-layer graph at any layer count. The target architecture is real graph
import — see below.

### ONNX Graph Import Path (Phase 1 — real per-layer expansion, not real ONNX parsing yet)

`mlir_passes/tools/qwen-onnx-to-serving-mlir` is a new, separate frontend
tool that reads `configs/models/qwen_0_5b_onnx_graph_facts.json` and emits
**full per-layer-expanded, flat, unrolled** serving MLIR — one real op
sequence per decoder layer (`serving.layer_index = 0..num_layers-1`, real SSA
chaining layer-to-layer), not qwen-to-serving-mlir's single hand-templated
block. It emits the **raw** (pre-canonicalization) attention pattern
(`llm.q_proj`/`k_proj`/`v_proj`/`attention_scores`/`softmax`/`attention_output`
+ `kv_cache_write`/`read`) per layer, which `LLMFrontendNormalizationPass`
(generalized in this change to do a localized per-occurrence rewrite instead
of a whole-function erase) canonicalizes into one real `llm.attention_prefill`
/`llm.attention_decode` op per layer, wired to that layer's real q/k/v
values — not a dummy placeholder.

**Compiler philosophy:** compiler input is moving toward real, per-layer-expanded
graphs (ONNX-shaped today, a real ONNX/HF import tomorrow) rather than
hand-authored model specifications. `qwen-to-serving-mlir`'s ModelSpec path is
the legacy/scaffold generator being phased out, not the long-term architecture
— it is kept only for the existing GTX 1650 A/B benchmark artifact above, and
gets no new capability.

**Be precise about what Phase 1 is and is not:**
- The graph-facts JSON is a **hand-authored fixture** standing in for facts a
  real ONNX importer would extract — it is explicitly labeled
  `"truth_boundary": "onnx_shaped_fixture_not_real_onnx_protobuf_import"`.
  This tool does **not** parse a real ONNX protobuf file.
- `tools/export_qwen_onnx.py` is a Python edge script (not compiler-core) that
  attempts a real HF Optimum → ONNX export and real `onnx`-package graph
  introspection when that optional toolchain is installed; its output is a
  diagnostic report only and is **not yet consumed** by the C++ importer in
  this change — see that script's module docstring.
- Linear/weight-bearing ops (`q_proj`/`k_proj`/`v_proj`/`o_proj`/`mlp`/
  `lm_head_proj` from this importer, `qkv_projection`/`mlp` from the legacy
  ModelSpec path) are marked with an explicit `serving.quantizable = true`
  attribute at emission time, so `QuantizationStrategyPlanningPass`/
  `WeightClassificationPlanningPass` key off that attribute first and only
  fall back to a small generic name-fragment match (`matmul`, `conv`, `gemm`,
  etc.) for ops with no explicit marker. This keeps frontend-specific naming
  conventions out of the generic planning passes instead of pattern-matching
  op-name substrings like `"proj"`/`"mlp"` directly in those passes.
- The exported `ExecutionPlan` stays **verbose** in Phase 1: no
  `layer_range`/`layer_count` compression, no JSON schema change. A 24-layer
  plan has ~170 `per_op_decisions` entries per phase (real, one per real op),
  not the legacy path's single `op_1` (`llm.rmsnorm`) entry. Compression is
  deferred to a future phase and will only ever be an export-time
  optimization over verified-identical decisions, never the compiler's
  internal IR model — see `docs/future_work.md`.

Run `mlir_passes/test/serving/RunQwenOnnxServingPlanExportTest.cmake` (CTest
`QwenOnnxServingPlanExportTest`) for the full pipeline, or
`mlir_passes/test/serving/llm_frontend_normalization_layered.mlir` (FileCheck)
for the localized per-occurrence rewrite in isolation.

### Real ONNX Protobuf Bridge (Phase 2 — frontend adapter, not a general importer)

`GraphFacts` (the JSON schema above) is the **frontend boundary / adapter
seam** in this architecture:

```text
HF / external model source
  -> ONNX protobuf
  -> Python frontend adapter (tools/onnx_graph_to_facts.py)
  -> GraphFacts JSON
  -> qwen-onnx-to-serving-mlir  [C++, unchanged from Phase 1]
  -> Serving MLIR -> existing compiler passes -> ExecutionPlan
```

`tools/onnx_graph_to_facts.py` loads a real `.onnx` file with the Python
`onnx` package and reads real protobuf structure — node op types,
initializer names, initializer shapes/dtypes (metadata only; tensor values
are never materialized). It classifies per-layer roles
(`q_proj`/`k_proj`/`v_proj`/`o_proj`, mlp gate/up/down, both layernorms) by
matching real initializer names against **Qwen2's specific HuggingFace
parameter-naming convention** (e.g.
`model.layers.{i}.self_attn.q_proj.weight`), derives `num_layers`,
`hidden_size`, `intermediate_size`, `vocab_size`, and `dtype` from real
initializer shapes/dtypes, and detects RoPE presence and lm_head/embedding
weight tying from real graph signals. It emits the same `GraphFacts` JSON
schema Phase 1 already established — `qwen-onnx-to-serving-mlir` needed
**no changes** to consume it.

**This is not a general ONNX importer.** It only recognizes Qwen2's
decoder-only architecture (RMSNorm, GQA, SwiGLU MLP, separate q/k/v/o Linear
layers) via this one naming convention. Any other model family, or a Qwen
graph whose parameter names have been renamed/fused by an ONNX
graph-optimization pass, is out of scope and fails the role-classification
checks rather than guessing. The per-layer *operator sequence*
(rmsnorm → q/k/v_proj → attention_scores → softmax → attention_output →
kv_cache_boundary → o_proj → rmsnorm → mlp) is a **declared Qwen2
architecture template**, matching the existing hand-authored fixture, not
derived by tracing computational edges through the raw ONNX graph — what
*is* derived from the real graph is layer count, per-layer role presence,
real dimensions/dtype, and RoPE-ness.

Truth boundary emitted in the bridge's output:
`"onnx_protobuf_parsed_pattern_matched_not_general_graph_interpreter"` — a
real ONNX protobuf was parsed, but role assignment is heuristic pattern
matching tuned to Qwen2, not a general graph interpreter, and no numeric
weight-value verification is performed.

Scalar facts not recoverable from ONNX graph structure alone
(`num_attention_heads`, `num_key_value_heads`, `max_position_embeddings`)
are read from an HF `config.json` sitting next to the `.onnx` file (the
layout HF Optimum's own export already produces) or from explicit CLI
overrides — never guessed.

`tools/validate_onnx_graph_facts.py` validates any `GraphFacts` document:
for real bridge output (which carries an additive `provenance` field) it
verifies `num_layers` matches the number of distinct parsed layer indices
and that every layer has all required roles, failing hard on any gap. For
the hand-authored fixture (no `provenance` field), it validates the
top-level schema only and explicitly reports per-layer completeness checks
as skipped, not silently passed.

**Current status:** the hand-authored fixture
(`configs/models/qwen_0_5b_onnx_graph_facts.json`) and the real bridge
coexist. The fixture remains as fast, deterministic, network-free
regression coverage (`tests/test_onnx_graph_facts_fixture_regression.py`);
it is not replaced. RoPE is absorbed during pattern recognition rather than
materialized as a distinct op — when detected, it is stamped as a
function-level `serving.positional_encoding = "rope"` attribute on
`qwen_prefill`/`qwen_decode` (an additive, harmless read the C++ importer
performs only when the field is present; the hand-authored fixture has no
such field and emits identical MLIR as before).

Not implemented (do not claim otherwise): general ONNX import for arbitrary
model families, ONNX-MLIR (or equivalent) frontend integration, a Torch FX
adapter, a StableHLO adapter, decode-with-past graph handling, and
layer-range/export-time compression. See `docs/future_work.md`.

Tests: `tests/test_onnx_graph_to_facts.py` (bridge unit tests plus an
end-to-end test through the real `qwen-onnx-to-serving-mlir` binary; skips
cleanly if `onnx` isn't installed or the binary isn't built) and
`tests/test_onnx_graph_facts_fixture_regression.py` (fixture regression,
no optional dependencies).

### CV Compiler Pipeline

- Registered `cv` MLIR dialect with seven CV operations.
- Added variadic `cv.detect_head` parsing/printing support.
- Added Phase 1 CV compiler passes:
  `CVFrontendNormalizationPass`, `CVShapeInferencePass`,
  `CVMemoryPlanningPass`, and `CVExecutionDomainPlanningPass`.
- Added `CVExecutionPlanBuilder` and `CVExecutionPlanExporter`.
- Added `emit-cv-execution-plan`.
- Added the checked-in `artifacts/apple_demo/cv_execution_plan.json` artifact.

Future work remains: ONNX importer, more CV operators, dynamic shape support,
backend mapping, and PocketChef visualization.

## Repository Layout

- `include/`, `src/`: custom C++ graph IR, runtime, kernels, planners, and
  toy compiler passes.
- `apps/`: C++ demos, runtime experiments, and platform-specific harnesses.
- `mlir_passes/`: real MLIR C++ dialects, passes, tools, and FileCheck/CTest
  coverage. This includes both HIR/LLM work and the CV dialect/compiler path.
- `mlir/`: small MLIR input examples, including raw CV and LLM inputs.
- `tools/`: artifact generation, validation, and optional frontend probes.
- `artifacts/apple_demo/`: generated or checked-in demo artifacts, including
  `cv_execution_plan.json` and LLM serving plans.
- `integration_bundle/apple_demo_artifacts/`: copied artifacts for downstream
  demo consumption.
- `docs/`: architecture, data-flow, design, future-work, and case-study notes.

The LLM artifact path now emits a `serving_framework_contract.json` alongside
the execution plan. This contract names the serving policies and metrics a
runtime should honor: continuous batching, prefill/decode split, paged KV-cache
pressure, dynamic batching/backend routing, TensorRT-style engine profile
dispatch, TTFT, TPOT, throughput, queue wait, memory pressure, and SLO signals.

### Baseline Validation

`bash scripts/check.sh` is the canonical baseline validation command. It runs:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

It does not install any system dependencies — missing tools should surface as
a failed step, not be silently worked around.

On Apple builds, `ctest` currently reports a known failure in
`metal_rmsnorm_plan_dispatch`. That test requires
`trace/metal_rmsnorm_execution_plan.json`, which is generated only by the MLIR
pipeline (`tools/run_metal_rmsnorm_compiler_pipeline.sh`), not by the baseline
CMake build, so the file is absent in a plain checkout. The CTest entry reports
that case as skipped rather than fabricating the artifact or failing baseline
validation. Run the MLIR pipeline first when validating the Metal RMSNorm
dispatch path itself.

### Current CV Compiler Pipeline

The current MLIR CV path is:

```text
ONNX (future)
  ->
CV Dialect
  ->
CVFrontendNormalizationPass
  ->
CVShapeInferencePass
  ->
CVMemoryPlanningPass
  ->
CVExecutionDomainPlanningPass
  ->
CVExecutionPlanBuilder
  ->
CVExecutionPlanExporter
  ->
emit-cv-execution-plan
  ->
artifacts/apple_demo/cv_execution_plan.json
```

Implemented CV dialect/compiler components:

- `cv` dialect registration.
- Seven CV operations covering the raw CV graph shape used by this milestone.
- Variadic `detect_head` support.
- Frontend normalization metadata.
- Static shape metadata for supported CV ops.
- Tensor lifetime and memory-planning metadata.
- Execution-domain planning metadata.
- Runtime-facing JSON export through the `emit-cv-execution-plan` tool.

Not currently claimed:

- ONNX import.
- A complete CV operator set.
- Dynamic-shape compilation.
- Final backend/kernel dispatch mapping for the CV plan.
- PocketChef visualization of `cv_execution_plan.json`.

### Quantization Compiler Pipeline Status

The MLIR/HIR compiler path now includes a small, test-covered quantization
pipeline:

```text
import / canonicalize
  ->
quantization-planning
  ->
hir-quant-propagate
  ->
hir-quant-canonicalize
  ->
fusion / lowering
  ->
layout legality boundary
  ->
hir-int8-operator-selection
  ->
verification / execution-plan export
```

Implemented:

- Conservative INT8 island metadata propagation through INT8-candidate
  MatMul, ReLU, and reshape/cast ops.
- Safe Q/DQ and DQ/Q elimination when quantization metadata is identical.
- A deterministic capability-table pass for INT8 selection metadata, with
  fallback reasons for unsupported backend, illegal layout, illegal shape, or
  profile metadata that does not favor INT8.
- FileCheck coverage for positive and negative cases.

Still not claimed:

- Calibration or automatic quantization-parameter generation.
- Full layout rewrite such as NCHW/NHWC conversion.
- Runtime INT8 dispatch through the custom C++ graph executor.
- Arbitrary graph-wide quantization propagation.

Honest interview wording:

> I implemented a real MLIR/HIR quantization compiler slice: quantization
> planning, conservative INT8 island propagation, safe Q/DQ cleanup, and
> capability-gated INT8 operator-selection metadata with negative tests. It is
> still a compiler pipeline prototype: the runtime-facing artifacts can carry
> INT8 decisions, but the custom C++ runtime does not yet execute a full INT8
> graph.

### Legacy Toy CV Graph Pipeline

The custom C++ graph runtime also includes a CV inference graph demo:

```text
Conv2D
    →
BatchNorm
    →
ReLU
    →
MaxPool
    →
Flatten
    →
Linear
```

Implemented graph-level compiler optimization passes including:

- ShapeInferencePass
- CanonicalizationPass
- DTypePropagationPass
- FusionCandidatePass
- MemoryPlanningPass
- BackendPlacementPass
- SchedulingPass

Implemented backend-aware graph lowering into:

- LoweredGraph IR
- ExecutionPlan IR
- StaticExecutionSchedule

Generated artifacts:

- [cv_lowered_graph.json](trace/cv_lowered_graph.json)
- [cv_execution_plan_v2.json](trace/cv_execution_plan_v2.json)
- [cv_static_schedule.json](trace/cv_static_schedule.json)

### CV Graph Fusion Analysis

Implemented compiler-side fusion analysis and graph rewrite infrastructure for CV inference optimization.

Implemented fusion rewrite:

```text
Conv2D + BatchNorm + ReLU
    →
FusedConvBatchNormReLU
```

Implemented:

- fusion-candidate analysis
- graph rewrite infrastructure
- fused-op lowering
- fused execution scheduling
- fusion-aware runtime planning

Example compiler rewrite:

```text
rewriting:
Conv2D + BatchNorm + ReLU
    →
FusedConvBatchNormReLU
```

This simulates lightweight fusion infrastructure used in production ML compilers and inference runtimes.

### Tensor Lifetime and Memory Planning

Implemented compiler-side tensor lifetime analysis and runtime memory-reuse planning.

Implemented:

- activation lifetime tracking
- persistent tensor analysis
- buffer reuse planning
- memory-offset assignment
- runtime memory reuse
- peak-memory estimation
- activation reuse analysis

Example memory reuse behavior:

```text
relu_out reuses buffer from conv_out
pool_out reuses buffer from conv_out
flat_out reuses buffer from conv_out
logits reuses buffer from conv_out
```

Example memory-planning result:

```text
Naive memory:
4882250 float elements

Planned peak memory:
3699424 float elements

Saved memory:
1182826 float elements
```

Generated artifacts:

- Memory planning metadata is emitted with the generated CV trace artifacts.

This simulates lightweight runtime-memory planning infrastructure used in heterogeneous inference runtimes and serving systems.

### Backend Placement and Heterogeneous Scheduling

Implemented heterogeneous backend-placement analysis and runtime execution scheduling.

Implemented backend placement including:

- CPU execution
- Metal execution
- MockGPU execution

Implemented:

- backend-aware lowering
- dependency-aware scheduling
- backend transition analysis
- execution-plan generation
- execution timeline simulation
- heterogeneous runtime orchestration

Example execution schedule:

```text
[0] conv1 | FusedConvBatchNormReLU | backend=Metal
[1] pool1 | MaxPool | backend=CPU
[2] flatten | Flatten | backend=CPU
[3] linear | Linear | backend=Metal
```

Generated artifacts:

- [cv_static_schedule.json](trace/cv_static_schedule.json)
- [cv_runtime_timeline.json](trace/cv_runtime_timeline.json)

### Subgraph Partitioning

Implemented backend-aware subgraph partitioning for heterogeneous inference execution.

Implemented:

- backend-oriented graph partitioning
- execution-region grouping
- backend execution segmentation
- runtime migration-region analysis

Example partitioning result:

```text
subgraph 0 | backend=Metal | ops=conv1
subgraph 1 | backend=CPU | ops=pool1 flatten
subgraph 2 | backend=Metal | ops=linear
```

Generated artifacts:

- [cv_subgraph_partition.json](trace/cv_subgraph_partition.json)

This simulates heterogeneous execution partitioning used in production inference runtimes and compiler-runtime systems.

### Compiler Cost Analysis

Implemented compiler-side cost-analysis infrastructure with runtime-aware scheduling metadata.

Implemented:

- estimated memory-read analysis
- estimated memory-write analysis
- FLOPs estimation
- arithmetic-intensity analysis
- backend-switch overhead estimation
- launch-cost estimation
- fusion-aware execution analysis

Example runtime-aware cost report:

```text
conv1 | FusedConvBatchNormReLU
backend=Metal
read_bytes=603840
write_bytes=3154176
flops=42581376
intensity=11.3308
```

Generated artifacts:

- [cv_cost_report.json](trace/cv_cost_report.json)

This simulates lightweight compiler-side cost modeling and runtime execution analysis used in ML compiler/runtime systems.

## Adaptive Runtime Planning and Orchestration

Implemented adaptive runtime-planning infrastructure for heterogeneous execution scheduling, runtime feedback analysis, backend migration, and dynamic runtime recovery orchestration.

Implemented:

- runtime feedback-driven backend replanning
- heterogeneous execution-plan comparison
- runtime latency-aware backend migration
- runtime overload detection
- adaptive CPU fallback orchestration
- GPU recovery-state management
- runtime state-machine simulation
- runtime orchestration visualization tooling

### Timeline Optimization Simulation

Implemented runtime what-if execution-plan analysis for heterogeneous backend scheduling.

Implemented execution-plan comparisons including:

- current heterogeneous execution plan
- all-Metal execution plan
- Metal-pool optimized execution
- CPU-middle fallback execution

Compared runtime-planning metrics including:

- total execution latency
- backend-switch overhead
- memory pressure estimation
- GPU occupancy proxy
- runtime orchestration efficiency

Example runtime-planning analysis:

```text
Current:
Metal conv
↓ switch
CPU pool
CPU flatten
↓ switch
Metal linear

All-Metal:
Metal conv
Metal pool
Metal flatten
Metal linear
```

Generated artifacts:

![Timeline Optimization](cv_timeline_optimization.png)

This simulates lightweight runtime-planning analysis and heterogeneous execution optimization used in production ML runtimes.

### Cost-Based Backend Planner

Implemented a lightweight cost-based backend planner for heterogeneous runtime execution optimization.

Implemented:

- candidate backend-plan evaluation
- latency-aware plan selection
- backend-switch cost estimation
- GPU occupancy-aware scheduling heuristics
- runtime memory-pressure estimation
- execution-plan ranking
- best-plan selection infrastructure

Implemented runtime-planning candidates including:

- current heterogeneous plan
- all-Metal plan
- Metal-pool-only plan

Example planner output:

```text
current:
latency=1.49 ms
switch_cost=0.04 ms
gpu_occupancy=0.36

all_metal:
latency=0.76 ms
switch_cost=0.00 ms
gpu_occupancy=1.00
BEST
```

Generated artifacts:

![Cost-Based Planner](cv_cost_based_planner.png)

This simulates lightweight cost-based runtime scheduling infrastructure used in modern inference runtimes and compiler-runtime systems.

### Offline Compiler Artifact Validation Gate

Added an offline compiler artifact gate for catching compiler/runtime contract regressions before expensive backend or accelerator benchmarking.

The gate checks execution-plan dependency contracts, cost-planner consistency, backend-placement sanity, and optional memory-plan regressions across the generated `trace/cv_*.json` artifacts.

### Runtime Adaptive Replanning

Implemented runtime-feedback-driven adaptive replanning simulation for heterogeneous inference execution.

Implemented:

- runtime latency monitoring
- backend overload detection
- runtime backend migration
- adaptive CPU fallback orchestration
- runtime-plan replacement
- runtime execution recovery modeling

Example runtime replanning scenario:

```text
Initial Plan:
all_metal
latency=0.76 ms

Runtime Feedback Trigger:
Metal observed 2.84 ms overload

Replanned:
runtime_replanned_cpu_fallback
latency=2.10 ms
```

Generated artifacts:

![Runtime Adaptive Replanning](cv_runtime_replan.png)

This simulates runtime-feedback orchestration and adaptive heterogeneous backend migration systems used in serving runtimes and edge inference systems.

### Adaptive Runtime State Machine

Implemented adaptive runtime state-machine simulation for dynamic backend orchestration and runtime recovery pipelines.

Implemented runtime states including:

- NORMAL
- OVERLOAD_DETECTED
- REPLANNING
- CPU_FALLBACK
- RECOVERY_CHECK
- RESTORE_GPU_PLAN

Implemented runtime transitions including:

- Metal latency-spike detection
- planner invocation
- backend migration
- GPU health probing
- latency normalization recovery

Example runtime orchestration flow:

```text
NORMAL
    →
OVERLOAD_DETECTED
    →
REPLANNING
    →
CPU_FALLBACK
    →
RECOVERY_CHECK
    →
RESTORE_GPU_PLAN
```

Generated artifacts:

![Runtime State Machine](cv_runtime_state_machine.png)

This simulates adaptive runtime orchestration systems used in heterogeneous inference runtimes, edge inference systems, and serving-oriented runtime infrastructures.

## LLM Compiler/Runtime Planning Infrastructure

Implemented a lightweight LLM compiler/runtime planning path that turns a tiny
LLM graph into runtime-facing artifacts for prefill/decode execution, KV-cache
layout, memory planning, scheduling metadata, validation, and Apple dashboard
integration.

Implemented:

- MLIR-style tiny LLM graph input
- serving-aware compiler analysis
- prefill/decode execution planning
- KV-cache layout and memory planning
- scheduler metadata generation
- artifact validation and integration bundle generation

### KV Cache Infrastructure

Implemented KV-cache layout and memory-planning infrastructure for Transformer
compiler/runtime planning.

Implemented:

- KV block-size planning
- KV token-capacity estimation
- bytes-per-token and bytes-per-block estimation
- paged-attention metadata
- block-table metadata
- prefix-cache policy contract
- capacity-aware admission policy metadata
- LRU finished-prefix eviction policy metadata
- runtime memory contract generation

Generated artifacts:

- [kv_cache_plan.json](artifacts/apple_demo/kv_cache_plan.json)

This is not a full KV-cache manager. It emits a compiler/runtime planning
contract that a serving runtime or dashboard can consume. The
`kv_cache_plan.json` contract now includes prefix-cache, eviction, and admission
policy fields that downstream serving demos can enforce at request time.

### Transformer Attention Planning

Implemented Transformer attention planning and runtime-facing metadata for
prefill/decode execution.

Implemented:

- fused attention simulation
- tiled attention execution
- causal attention execution
- paged-attention planning metadata
- attention execution metadata
- backend-aware attention execution

Implemented demos including:

- run_attention_demo
- run_fused_attention_demo
- run_tiled_attention_demo
- run_causal_attention_demo

This provides attention-oriented compiler/runtime context for the LLM artifact
pipeline.

### LLM Scheduling Metadata

Implemented scheduling metadata generation for LLM compiler/runtime planning.

Implemented:

- prefill/decode queue metadata
- continuous-batching scheduler metadata
- decode-step token metadata
- workload-shape metadata
- dashboard signal definitions
- validation checks for scheduling artifacts

Implemented planning flow including:

```text
Tiny LLM MLIR Graph
    →
Serving Analysis
    →
Execution Plan
    →
KV-Cache Layout Plan
    →
Memory Plan
    →
Scheduling Plan
    →
Validation Report
    →
Apple Demo Bundle
```

This keeps the project positioned as an LLM compiler/runtime planning demo
rather than a full request-serving runtime implementation.

### MLIR Compiler Pass Pipeline

This project now includes a real MLIR C++ pass plugin under `mlir_passes/`.
The pass detects a tensor-level MatMul + Bias Add + ReLU pattern:

```text
linalg.matmul
  -> linalg.map arith.addf
  -> linalg.map arith.maximumf
```

The pass annotates fusion candidates and assigns fusion metadata:

```mlir
linalg.matmul {
  fusion.candidate = "matmul_bias_relu",
  fusion.group = "matmul_bias_relu_0",
  fusion.role = "producer"
}
```

The MLIR pipeline is connected to runtime-facing artifacts:

```text
trace/mlir_fused_graph.mlir
trace/mlir_lowered_graph.json
trace/mlir_execution_plan.json
```

### Demo Integration Artifacts

This repo is the compiler producer for the external demo project. It does not
host the dashboard itself; it emits compiler artifacts that a runtime workbench
can consume.

Current demo artifact directory:

```text
integration_bundle/apple_demo_artifacts/
```

Key outputs:

- `artifact_provenance.json`: compiler version, pass pipeline, source artifact
  hashes, and emitted artifact hashes
- `tiny_gpt_serving.mlir`: LLM-shaped MLIR workload used to exercise the pass
  pipeline
- `mlir_fused_graph.mlir`: annotated MLIR after fusion-candidate detection
- `mlir_lowered_graph.json`: runtime-facing HIR JSON
- `serving_execution_plan.json`: compiler-produced prefill/decode execution
  contract
- `candidate_execution_plans.json`: Metal, CPU, and hybrid plan candidates
- `memory_timeline.json`: allocation, reuse, and free events for memory-planning
  inspection
- `validation_manifest.json`: artifact-level validation and integration
  manifest

The intended integration path is:

```text
MLIR source
  -> fusion annotation
  -> HIR JSON
  -> execution-plan JSON
  -> runtime planner
  -> validation/dashboard artifacts
```

This makes the compiler the source of truth for the demo. The dashboard should
show the emitted compiler contract, not invent optimization claims inside the
frontend.

Run the pipeline and tests:

```bash
cmake --build build-mlir
tools/run_mlir_pass_tests.sh
tools/run_mlir_fusion_pipeline.sh
```

This adds a real MLIR frontend pass stage before the existing custom
LoweredGraph / ExecutionPlan / heterogeneous runtime planning flow.

### HIR-to-LLVM Executable CPU Path

The compiler now has a native MLIR backend lowering path in addition to the
runtime JSON bridge:

```text
hir.fused_rmsnorm
  -> linalg.generic + math.rsqrt
  -> one-shot bufferization
  -> LLVM dialect
  -> mlir-runner executable CPU function
```

Run the correctness harness:

```bash
PLUGIN=$PWD/build-mlir-codex/HIRMatMulBiasReluFusionPass.dylib \
python3 tools/run_hir_rmsnorm_execution_engine.py
```

The report is written to
`trace/hir_rmsnorm_execution_engine_report.json` and `.md`.

### OpenXLA / StableHLO Alignment

StableHLO tooling is optional and checked explicitly:

```bash
python3 tools/check_openxla_toolchain.py
```

Until `stablehlo-opt` or StableHLO Python tooling is installed, native
StableHLO tests are skipped. The current FileCheck coverage uses
StableHLO-compatible decompositions represented in standard MLIR
`linalg/arith/tensor/math` form, then lowers MatMul-Bias-ReLU and RMSNorm
patterns into HIR.

For a CI-friendly frontend proof that still starts from `stablehlo.*` op names,
run the textual subset pipeline:

```bash
PLUGIN=$PWD/build-mlir-codex/HIRMatMulBiasReluFusionPass.dylib \
python3 tools/run_stablehlo_subset_pipeline.py
```

This imports the supported StableHLO textual subset into standard MLIR, lowers
RMSNorm and MatMul-Bias-ReLU into HIR, then lowers RMSNorm to LLVM dialect and
executes it with `mlir-runner`.

For the real JAX frontend path, install CPU JAX in the repo venv and run:

```bash
.venv/bin/python -m pip install -U "jax[cpu]"
.venv/bin/python tools/run_jax_stablehlo_pipeline.py
```

This exports StableHLO from JAX RMSNorm and MatMul-Bias-ReLU functions, imports
the supported patterns through a lightweight StableHLO parser/legality gate,
lowers legal imports into HIR, and executes the RMSNorm HIR-to-LLVM CPU path
against the JAX/PJRT-backed compiled CPU reference. Illegal patterns are
rejected before Linalg/HIR import. The result records correctness, compile
latency, first-run latency, warm-run latency, host/device-buffer timing,
executable metadata, and local `mlir-runner` latency. The same artifact records a JAX tiny block
`RMSNorm -> MatMul -> Bias -> ReLU` and shape-specialized RMSNorm compile/reuse
costs to show the frontend/runtime boundary for multi-op and shape-bucketed
workloads. MatMul-Bias-ReLU also has a native HIR-to-Linalg-to-LLVM executable
path, dynamic-shape input falls back before lowering, and measured tile
autotuning selects the fastest correct tile candidate for supported shape
buckets. It is intentionally scoped to the local HIR/LLVM pipeline and a
JAX-backed runtime-boundary comparison; it does not claim full OpenXLA, XLA
passes, TPU execution, or custom PJRT runtime integration.

Optional frontend/comparison probes:

```bash
.venv/bin/python tools/run_torch_mlir_tiny_transformer_probe.py
.venv/bin/python tools/run_iree_stablehlo_subset_comparison.py
```

The Torch-MLIR probe exports a tiny RMSNorm/Linear/ReLU block when `torch_mlir`
is installed; otherwise it writes a skip report. The IREE probe compiles the
same StableHLO textual subset through IREE VM/HAL for architecture comparison.

### Apple Silicon MLIR-to-Metal RMSNorm

The Apple Silicon path executes a real Metal RMSNorm kernel and closes the
measured compiler/runtime loop:

```text
llm.rmsnorm
  -> hir.fused_rmsnorm
  -> measured CPU/Metal shape-bucket profile
  -> compiler-selected fused_rmsnorm_metal or cpu_rmsnorm
  -> runtime reads execution plan
  -> real Metal dispatch and numeric correctness report
```

Run the full path:

```bash
PLUGIN=$PWD/build-mlir/HIRMatMulBiasReluFusionPass.dylib \
tools/run_metal_rmsnorm_end_to_end.sh
```

See `docs/APPLE_SILICON_MLIR_METAL_PATH.md` for the measured crossover,
generated artifacts, and validation workflow.

### LLM Compiler Artifact Generation

This project emits Apple-demo-ready LLM compiler/runtime planning artifacts:

```text
LLM MLIR graph / request workload
    ->
compiler analysis extracts prefill/decode and KV-cache roles
    ->
runtime planner emits execution, memory, scheduling, and KV-cache layout artifacts
    ->
validation checks artifact correctness and planning consistency
    ->
dashboard visualizes compiler/runtime planning behavior
```

Generate the artifacts:

```bash
python3 src/ml_graph_compiler_runtime/generate_llm_artifacts.py \
  --config configs/tiny_gpt_llm_config.json \
  --out artifacts/apple_demo
```

Generated outputs:

- `artifacts/apple_demo/llm_graph_ir.json`
- `artifacts/apple_demo/serving_execution_plan.json`
- `artifacts/apple_demo/kv_cache_plan.json`
- `artifacts/apple_demo/memory_plan.json`
- `artifacts/apple_demo/scheduling_plan.json`
- `artifacts/apple_demo/validation_manifest.json`

The Apple-side demo should consume these JSON files directly so changes to
model dimensions, workload shape, KV-cache block sizing, memory budget, or
scheduler settings are reflected in the dashboard after regeneration.

### MLIR LLM Frontend Bridge

Generate Apple-demo-facing LLM compiler/runtime artifacts from a tiny MLIR-style
LLM graph:

```bash
python3 tools/emit_llm_artifacts_from_mlir.py \
  --mlir mlir/tiny_gpt_serving.mlir \
  --config configs/tiny_gpt_llm_config.json \
  --out artifacts/apple_demo \
  --analysis-out trace/mlir_llm_serving_analysis.json
```

### LLM Compiler Analysis Pass

A lightweight Python analysis pass extracts LLM compiler/runtime metadata from
the tiny MLIR graph:

```bash
python3 tools/analyze_llm_serving_mlir.py \
  --mlir mlir/tiny_gpt_serving.mlir \
  --out trace/llm_serving_compiler_analysis.json
```

### Emit Artifacts From Analysis

After the MLIR analysis pass runs, lower the analysis result into
Apple-demo-facing compiler/runtime artifacts:

```bash
python3 tools/emit_llm_artifacts_from_analysis.py \
  --analysis trace/llm_serving_compiler_analysis.json \
  --config configs/tiny_gpt_llm_config.json \
  --out artifacts/apple_demo
```

### Validate LLM Compiler Artifacts

Validate the generated compiler/runtime artifacts before handing them to the
Apple demo:

```bash
python3 tools/validate_llm_serving_artifacts.py \
  --artifacts artifacts/apple_demo \
  --out trace/llm_artifact_validation_report.json
```

### Run The Full LLM Compiler Artifact Pipeline

Run the full MLIR-to-artifacts pipeline:

```bash
tools/run_llm_serving_artifact_pipeline.sh
```

## Documentation

For deeper project notes, start with:

- `docs/architecture.md`
- `docs/data_flow.md`
- `docs/design_decisions.md`
- `docs/technical_debt.md`
- `docs/future_work.md`
- `docs/EXECUTION_PLAN_SCHEMA.md`

These files document implemented compiler/runtime components, generated
artifacts, assumptions, known weak spots, and realistic next steps.
