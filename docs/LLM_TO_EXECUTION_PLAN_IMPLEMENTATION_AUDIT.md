# LLM To ExecutionPlan Implementation Audit

This audit traces the implemented Qwen/LLM serving path from LLM dialect input to `ExecutionPlan` export. It is based on repository code, pass registration, scripts, tests, and generated artifacts. It does not propose a new compiler architecture.

## Scope And Ground Rules

Audited path:

```text
Qwen / LLM input
  -> LLM dialect / serving MLIR
  -> analysis and planning passes
  -> ExecutionPlanBuilder
  -> ExecutionPlanExporter
```

Out of scope:

- CV implementation changes.
- Runtime execution.
- Backend code generation.
- New planning layers.
- Qwen behavior changes.

## 1. Current LLM Entry Paths

| Entry point | Source | Input format | Output format | Topology source | Emits `llm.*` directly | Serving/target/quant attrs | Invoked by |
|---|---|---|---|---|---|---|---|
| `qwen-to-serving-mlir` | `mlir_passes/tools/qwen-to-serving-mlir/main.cpp` | Model spec JSON, e.g. `configs/models/qwen_0_5b_spec.json` | Serving MLIR, e.g. `mlir/qwen_0_5b_serving.mlir`, `mlir/qwen_0_5b_serving_awq.mlir` | Template/scaffold from declared model config, not a full imported graph | Yes: canonical high-level ops such as `llm.embed`, `llm.rmsnorm`, `llm.qkv_projection`, `llm.attention_prefill`, `llm.attention_decode`, `llm.mlp` | Module `llm.*` identity attrs and frontend truth attrs; current source also emits `serving.quantizable` on projections/MLP. The checked plain fixture appears stale relative to current source because it lacks those markers, while the AWQ fixture includes them. | `tools/run_qwen_compiler_pipeline.sh`, `tools/run_qwen_awq_compiler_pipeline.sh`, CMake serving tests |
| `qwen-onnx-to-serving-mlir` | `mlir_passes/tools/qwen-onnx-to-serving-mlir/main.cpp` | Graph-facts JSON, e.g. `configs/models/qwen_0_5b_onnx_graph_facts.json`; the tool does not parse ONNX protobuf directly | Raw pre-normalization LLM MLIR | Per-layer expanded SSA graph derived from graph facts; real topology vocabulary, but not weightful ONNX import | Yes: raw ops such as `llm.q_proj`, `llm.k_proj`, `llm.v_proj`, `llm.attention_scores`, `llm.softmax`, `llm.attention_output`, `llm.kv_cache_write`, `llm.kv_cache_read`, `llm.o_proj`, `llm.mlp` | `serving.layer_index`, `serving.layer_role`, module model attrs, frontend truth attrs | `RunQwenOnnxToServingMlirTest.cmake`, `RunQwenOnnxServingPlanExportTest.cmake`, graph-facts tests |
| `mlir/raw_qwen_frontend_input.mlir` | Handwritten MLIR fixture | Raw LLM dialect fixture | Normalized LLM serving MLIR after `llm-frontend-normalization` | Handwritten pseudo-Qwen attention topology | Yes: raw Q/K/V, attention, softmax, KV-cache ops | Layer and role attrs sufficient for normalization tests | `RunLLMFrontendNormalizationTest.cmake`, `tools/run_mlir_pass_tests.sh`, demo artifact generation |
| `mlir/qwen_0_5b_serving.mlir` | Generated fixture | Already serving MLIR | Input to `compile-for-target` | Template/scaffold | Already contains `llm.*` | Module identity attrs; the checked file lacks some current-source quantizable markers | Qwen pipeline scripts and compile tests |
| `mlir/qwen_0_5b_serving_awq.mlir` | Generated fixture | Already serving MLIR | Input to `compile-for-target` | Template/scaffold | Already contains `llm.*` | Includes `serving.quantizable = true` on quantizable ops | AWQ pipeline scripts and tests |
| `compile-for-target` | `mlir_passes/tools/compile-for-target/main.cpp` | Serving MLIR plus target profile JSON | ExecutionPlan JSON; optional annotated MLIR | Does not build model topology; consumes existing MLIR | No new LLM ops; consumes them | Attaches target constraints, optional global quantization attrs, then runs serving planning passes | Qwen compile scripts/tests |

### Entry Path Conclusions

- There are two frontend styles: a model-spec template generator and a graph-facts-to-MLIR generator.
- The model-spec path is not a full graph importer. It creates a compact serving scaffold.
- The graph-facts path builds a fuller per-layer SSA topology, but still from JSON facts rather than directly from ONNX.
- `compile-for-target` is the real bridge into `ExecutionPlan`. It does not use a separate planning-facts artifact.

## 2. Exact Implemented Pass Pipeline

### Optional Frontend Normalization

`LLMFrontendNormalizationPass` is required for raw Qwen/frontend fixtures and graph-facts output when they contain decomposed attention ops.

| Order | Pass | Source file | Input assumptions | Reads | Writes | Transform or annotate | Later consumer |
|---:|---|---|---|---|---|---|---|
| pre | `LLMFrontendNormalizationPass` | `mlir_passes/lib/serving/LLMFrontendNormalizationPass.cpp` | Raw `llm.q_proj`, `llm.k_proj`, `llm.v_proj`, attention scores, softmax, attention output, KV-cache read/write grouped by `serving.layer_index` | Raw op names, layer attrs, prompt/output token attrs, model head attrs | Canonical attention ops, `kv_cache.role`, `serving.phase`, prompt/output token attrs, frontend truth attrs | Transforms IR | `compile-for-target` serving passes |

This pass is LLM-specific and does not run inside `compile-for-target`.

### Standalone Registered Pipeline

`mlir_passes/lib/serving/ServingPipeline.cpp` registers a standalone `serving-optimization-pipeline` with 16 passes:

1. `ServingPhaseAnalysisPass`
2. `KVLayoutPlanningPass`
3. `ReplayEligibilityPass`
4. `ExecutionProviderPlanningPass`
5. `RepresentationPlanningPass`
6. `LayoutPlanningPass`
7. `BoundaryPlanningPass`
8. `WeightClassificationPlanningPass`
9. `QuantizationStrategyPlanningPass`
10. `KernelAvailabilityPlanningPass`
11. `LoweringDecisionPlanningPass`
12. `QuantizedBoundaryRefinementPass`
13. `AlternativeLoweringPlanningPass`
14. `CandidateGenerationPass`
15. `ServingCostModelPass` via `createCandidateEvaluationPass()`
16. `PlanSelectionPass`

### Actual `compile-for-target` Pipeline

`compile-for-target` does not call the registered 16-pass pipeline by name. It constructs its own pass manager. The implemented order is:

1. `ServingPhaseAnalysisPass`
2. `KVLayoutPlanningPass`
3. `ReplayEligibilityPass`
4. `ExecutionProviderPlanningPass`
5. `RepresentationPlanningPass`
6. `LayoutPlanningPass`
7. `BoundaryPlanningPass`
8. `WeightClassificationPlanningPass`
9. `QuantizationStrategyPlanningPass`
10. `KernelAvailabilityPlanningPass`
11. `LoweringDecisionPlanningPass`
12. `QuantizedBoundaryRefinementPass`
13. `TilePlanningPass`
14. `KernelSelectionPass`
15. `QuantizationCoDesignPass`
16. `AlternativeLoweringPlanningPass`
17. `CandidateGenerationPass`
18. `ServingCostModelPass` via `createCandidateEvaluationPass()`
19. `PlanSelectionPass`
20. `BoundaryMaterializationPass`

The comments in `compile-for-target` and `ExecutionPlan.h` still refer to older 15-pass or 16-pass wording. The implemented driver currently runs 20 nested function passes after attaching target constraints.

### Pass Audit

| Pass | Source file | Input assumptions | Attrs/ops read | Attrs/ops written | Transform? | Specificity | Output actually consumed? | ExecutionPlan effect |
|---|---|---|---|---|---|---|---|---|
| `ServingPhaseAnalysisPass` | `mlir_passes/lib/ServingPhaseAnalysisPass.cpp` | Serving funcs with attention ops and model attrs | `llm.num_layers`, `llm.hidden_size`, target prefill/decode/bandwidth attrs, `serving.prompt_tokens`, `serving.output_tokens`, dtype attrs | `serving.policy`, total latency estimates, decision margin, `serving.confidence`, cost/truth attrs, effective quant dtype attrs | Annotates | LLM serving-specific | Builder reads `serving.policy`; later passes largely independent | `global_decisions.serving`; function phase evidence |
| `KVLayoutPlanningPass` | `mlir_passes/lib/serving/KVLayoutPlanningPass.cpp` | LLM attention/KV-cache model | model dimensions, target memory budget, attention token attrs | `kv.layout`, `kv.layout_reason`, `kv.byte_estimate_mb`, `kv.dtype_bytes`, `kv.truth_boundary` | Annotates | LLM/KV-specific | Execution provider and builder read it | `global_decisions.memory` |
| `ReplayEligibilityPass` | `mlir_passes/lib/serving/ReplayEligibilityPass.cpp` | Serving funcs with shape types and attention phase | Function arg/result shape, attention op kind, target static-shape support | `replay.eligible`, `replay.cuda_graph_bucket`, replay truth attrs | Annotates | Serving-specific | Execution provider and builder read it | Replay fields in serving decision |
| `ExecutionProviderPlanningPass` | `mlir_passes/lib/serving/ExecutionProviderPlanningPass.cpp` | Target constraints attached to module | target allowed/preferred backends, supported precision, paged KV compatibility, `kv.layout`, `replay.eligible` | `execution_provider.primary`, fallback chain, decision source, required precision/layout/replay, truth | Annotates | Serving/target-specific | Representation and builder read it | Function `BackendDecision` |
| `RepresentationPlanningPass` | `mlir_passes/lib/serving/RepresentationPlanningPass.cpp` | Target backend capabilities exist | execution provider, global quant dtype, backend supported dtypes/layouts | `representation.effective_dtype`, dtype source, preferred activation/weight layout, source backend, truth/conflict attrs | Annotates | Generic-ish serving planning | Layout, boundary, quant, kernel passes read it | Indirect; some per-op decisions use derived attrs |
| `LayoutPlanningPass` | `mlir_passes/lib/serving/LayoutPlanningPass.cpp` | Ops in function have tensor/value shapes and selected representation | representation layout/backend, backend layout-agnostic ops, optional initial layout | `layout.required_input_layout`, `layout.effective_layout`, `layout.layout_source`, `layout.transform_required`, truth | Annotates | Generic-ish | Boundary and builder read transform attrs | Optional `LayoutDecision` if transform required |
| `BoundaryPlanningPass` | `mlir_passes/lib/serving/BoundaryPlanningPass.cpp` | Representation/layout/backend capability attrs | dtype/backend/layout attrs, backend support for cast/dequant/layout transform | `boundary.cast_required`, `boundary.dequant_required`, `boundary.layout_transform_required`, `boundary.materialization_required`, `boundary.reason`, truth | Annotates | Generic-ish | Lowering, candidates, materialization, builder read it | Boundary requirements and materialization/deferred evidence |
| `WeightClassificationPlanningPass` | `mlir_passes/lib/serving/WeightClassificationPlanningPass.cpp` | Quantizable ops may have weight operands or constant markers | `serving.quantizable`, `representation.weights_are_constant`, `weight.is_constant`, constants, function args | `weight.classification`, `weight.constant_required`, `weight.constant_satisfied`, reason, truth | Annotates | Generic-ish with LLM op assumptions | Quant and kernel passes read it | Indirect quant/kernel decisions |
| `QuantizationStrategyPlanningPass` | `mlir_passes/lib/serving/QuantizationStrategyPlanningPass.cpp` | Backend capability and weight facts available | backend quant modes/dtypes/granularity, representation dtype, weight classification, `serving.quantizable`, op name fallbacks | `quant.strategy`, weight/activation/accumulation dtype, granularity, accuracy risk, reason, truth | Annotates | Serving/quant-specific | Kernel, boundary refinement, builder read it | Global/per-op quant decisions |
| `KernelAvailabilityPlanningPass` | `mlir_passes/lib/serving/KernelAvailabilityPlanningPass.cpp` | Target kernel libraries lowered to module attrs | target kernel libraries, backend/dtype/layout, quant strategy, weight class, dynamic shape facts | `kernel.backend`, `kernel.exists`, `kernel.library`, `kernel.name`, `kernel.lowering_status`, fallback/rewrite attrs, reason, truth | Annotates | Generic kernel capability check | Lowering, alternative, candidates, builder read it | Kernel decision availability evidence |
| `LoweringDecisionPlanningPass` | `mlir_passes/lib/serving/LoweringDecisionPlanningPass.cpp` | Kernel and boundary attrs exist | `kernel.lowering_status`, boundary requirements | `lowering.decision`, `lowering.reason`, target backend/kernel/library, boundary requirements, truth | Annotates | Generic-ish | Builder and candidate passes read it | Kernel/fallback decisions |
| `QuantizedBoundaryRefinementPass` | `mlir_passes/lib/serving/QuantizedBoundaryRefinementPass.cpp` | Quant and lowering attrs exist | `quant.*`, `lowering.decision`, target backend quant modes | weight dequant boundary attrs and mismatch/fallback reasons | Annotates | Quant-specific | Boundary materialization and builder read deferred boundary attrs | Deferred boundary evidence |
| `TilePlanningPass` | `mlir_passes/lib/serving/TilePlanningPass.cpp` | Static cost profile may declare local memory | static cost memory hierarchy, op shape facts, dtype/quant facts | `tile.plan.*` | Annotates | Generic matmul-like planning | Cost model and builder read it | Optional per-op `TilePlan` |
| `KernelSelectionPass` | `mlir_passes/lib/serving/KernelSelectionPass.cpp` | Runtime kernel descriptors may exist | `target.runtime_kernels`, backend/dtype/layout/quant/tile/shape attrs | `kernel_selection.*` selected/deferred/rejected status | Annotates | Generic kernel descriptor selection | Builder reads it | Optional per-op `KernelSelection` |
| `QuantizationCoDesignPass` | `mlir_passes/lib/serving/QuantizationCoDesignPass.cpp` | Module may opt in via `quant.codesign.policy` | backend/quant/kernel/cost facts | `quant_codesign.*` | Annotates; inert without policy | Quant-specific | Builder reads it | Optional co-design evidence |
| `AlternativeLoweringPlanningPass` | `mlir_passes/lib/serving/AlternativeLoweringPlanningPass.cpp` | Kernel/lowering/boundary attrs exist | kernel status, backend capabilities, library, boundary support, dtype/layout | `alternative.*` | Annotates | Generic fallback planning | Candidate generation reads it | Indirect candidate fields |
| `CandidateGenerationPass` | `mlir_passes/lib/serving/CandidateGenerationPass.cpp` | Backend capabilities and lowerability facts exist | capabilities, representation, weight/dynamic/kernel/lowering/alternative attrs | function `candidates`, op `compiler.candidates`, rejected candidates, counts, truth | Annotates | Generic-ish | Cost model reads it | Indirect selected plan evidence |
| `ServingCostModelPass` / `candidate-evaluation` | `mlir_passes/lib/serving/ServingCostModelPass.cpp` | Candidates exist; shape facts may be static | `compiler.candidates`, shape facts, dtype/quant, static cost profile | `compiler.evaluated_candidates`, `compiler.shape_profile.*` | Annotates | Generic-ish analytical model | Plan selection reads it | Optional shape/cost fields |
| `PlanSelectionPass` | `mlir_passes/lib/serving/PlanSelectionPass.cpp` | Evaluated candidates exist | `compiler.evaluated_candidates` | `selected_plan.*`, `selected_plan.cost.*`, `selected_plan.shape_cost.*` | Annotates | Generic-ish | Builder reads selected cost/shape fields; it does not fully replace `kernel.*` in kernel decision collection | Optional selected-plan cost/shape fields |
| `BoundaryMaterializationPass` | `mlir_passes/lib/serving/BoundaryMaterializationPass.cpp` | Boundary attrs and selected plan attrs exist | boundary flags, selected candidate kind | Inserts `hir.cast` for supported float casts; writes materialized/deferred boundary attrs | Transforms IR | Generic-ish boundary materialization | Builder skips materialized boundary ops and records materialized/deferred evidence | Boundary ops in per-op bundle |

## 3. Attr/Dataflow Diagram

Implemented dataflow is attribute-centric:

```text
Target profile JSON
  -> TargetDeviceProfile in compile-for-target
  -> TargetConstraints module attrs
  -> CapabilityBundle in memory

LLM serving MLIR
  -> serving/kv/replay attrs
  -> execution_provider attrs
  -> representation/layout/boundary attrs
  -> weight/quant/kernel/lowering attrs
  -> tile/kernel_selection/codesign attrs
  -> candidate/evaluated/selected_plan attrs
  -> optional hir.cast materialization
  -> ExecutionPlanBuilder reads annotated MLIR + CapabilityBundle
  -> ExecutionPlanExporter writes JSON
```

There is no separate typed planning-facts object between MLIR and `ExecutionPlan`. The MLIR module is the planning facts carrier.

## 4. ExecutionPlan Field Provenance

| ExecutionPlan field | Source MLIR attribute/op or source object | Producing pass/tool | Consumed by runtime? | Value kind | Scope | LLM-specific or reusable |
|---|---|---|---|---|---|---|
| `schema`, `schema_version` | Constants in `ExecutionPlan` | `ExecutionPlan.h` defaults | Artifact contract | Hardcoded | Plan-wide | Reusable |
| `plan_id` | `profileId + "_serving_plan"` | `compile-for-target` | Artifact identity | Configured | Plan-wide | Serving-specific naming |
| `provenance.compiler_tool` | Constant `"compile-for-target"` | `ExecutionPlanBuilder` | Provenance | Hardcoded | Plan-wide | Reusable |
| `provenance.model_spec_ref` | Empty string/object today | Not populated by driver | No | Not implemented | Plan-wide | Reusable gap |
| `provenance.capability_bundle.hardware_profile_ref` | `CapabilityBundle.hardware.hardware_id` | Target profile lowering | Provenance | Configured | Plan-wide | Reusable |
| `provenance.capability_bundle.backend_profile_refs` | backend capability names | Target profile lowering | Provenance | Configured | Plan-wide | Reusable |
| `provenance.capability_bundle.kernel_profile_refs` | kernel library names | Target profile lowering | Provenance | Configured | Plan-wide | Reusable |
| `model_identity.model_id` | Module `llm.model` | Frontend generator | Runtime/provenance consumer possible | Configured | Model-wide | LLM-specific |
| `model_identity.num_layers` | Module `llm.num_layers` | Frontend generator | Runtime/provenance consumer possible | Configured | Model-wide | LLM-specific |
| `model_identity.hidden_size` | Module `llm.hidden_size` | Frontend generator | Runtime/provenance consumer possible | Configured | Model-wide | LLM-specific |
| `model_identity.num_attention_heads` | Module `llm.num_attention_heads` | Frontend generator | Runtime/provenance consumer possible | Configured | Model-wide | LLM-specific |
| `model_identity.num_kv_heads` | Module `llm.num_key_value_heads` | Frontend generator | Runtime/provenance consumer possible | Configured | Model-wide | LLM-specific |
| `model_identity.model_family`, `attention_mechanism`, `positional_encoding` | Struct fields exist | Not populated in current builder | No | Not implemented | Model-wide | LLM-specific |
| `global_decisions.serving.policy` | First function with `serving.policy` | `ServingPhaseAnalysisPass` | Planning artifact | Analytical formula | Function/model-wide | LLM serving-specific |
| `global_decisions.serving` cost fields | `serving.colocated_total_ms`, `serving.pd_split_total_ms`, margins | `ServingPhaseAnalysisPass` | Planning artifact | Analytical/configured, not measured | Model-wide | LLM serving-specific |
| `global_decisions.serving.replay` | `replay.eligible`, `replay.cuda_graph_bucket` | `ReplayEligibilityPass` | Planning artifact | Static rule | Function/model-wide | Serving-specific |
| `global_decisions.memory.kv_cache_layout` | First function with `kv.layout` | `KVLayoutPlanningPass` | Runtime planning input possible | Static rule | Model-wide | LLM-specific |
| `global_decisions.memory.estimated_kv_peak_mb` | `kv.byte_estimate_mb` | `KVLayoutPlanningPass` | Planning artifact | Analytical estimate | Model-wide | LLM-specific |
| `global_decisions.memory.memory_budget_fraction` | `CapabilityBundle.deployment.memory_budget_fraction` | Target profile lowering | Planning artifact | Configured | Deployment-wide | Reusable |
| `global_decisions.memory.kv_block_size_tokens` | Struct field | Not populated; remains default | No | Not implemented/default | Model-wide | LLM-specific |
| `global_decisions.quantization` | Module `quantization.*` | Driver forced quantization path; possibly future quant pass | Runtime planning input possible | Configured | Model-wide | Quant reusable |
| `global_decisions.calibration` | Struct field | Not populated | No | Not implemented | Model-wide | Reusable |
| `function_plans.function_name` | `func.func` symbol name | Input MLIR | Runtime/provenance | Existing IR | Function-wide | Reusable |
| `function_plans.serving_phase` | Attention op kind or `serving.phase` | Frontend or normalization | Runtime/provenance | Metadata | Function-wide | LLM serving-specific |
| `function_plans.backend` | `execution_provider.primary`, fallback chain and related attrs | `ExecutionProviderPlanningPass` | Runtime planning input possible | Configured/static rule | Function-wide | Reusable |
| `per_op_decisions[].op_name` | Synthetic builder ID | `ExecutionPlanBuilder` | Artifact reference | Hardcoded/synthetic | Op-wide | Reusable |
| `per_op_decisions[].op_type` | MLIR op name | Input/passes | Artifact reference | Existing IR | Op-wide | Reusable |
| `per_op_decisions[].quantization` | `quant.strategy`, dtype/granularity/risk/reason/truth attrs | `QuantizationStrategyPlanningPass` | Runtime planning input possible | Static rule/configured | Op-wide | Reusable |
| `per_op_decisions[].layout` | `layout.*` attrs, emitted only if transform required | `LayoutPlanningPass` | Runtime planning input possible | Static rule | Op-wide | Reusable |
| `per_op_decisions[].kernel` | `lowering.*`, `kernel.*`, optional selected-plan cost evidence | `KernelAvailabilityPlanningPass`, `LoweringDecisionPlanningPass`, `PlanSelectionPass` | Runtime planning input possible | Capability/static rule | Op-wide | Reusable |
| `per_op_decisions[].fallback` | `lowering.decision == fallback_backend` and fallback attrs | `LoweringDecisionPlanningPass` | Runtime planning input possible | Static rule | Op-wide | Reusable |
| `materialized_boundary_ops` | `boundary.materialized_ops` | `BoundaryMaterializationPass` | Runtime/codegen evidence | Actual IR materialization metadata | Op-wide | Reusable |
| `deferred_boundary_ops` | `boundary.materialization.deferred` | `BoundaryMaterializationPass` / boundary refinement | Runtime/codegen evidence | Deferred metadata | Op-wide | Reusable |
| `shape_cost` | `selected_plan.shape_cost.*` | `ServingCostModelPass`, `PlanSelectionPass` | Planning evidence | Analytical, not measured | Op-wide | Reusable |
| `tile_plan` | `tile.plan.*` | `TilePlanningPass` | Planning evidence | Analytical/static | Op-wide | Reusable |
| `kernel_selection` | `kernel_selection.*` | `KernelSelectionPass` | Runtime planning input possible | Descriptor match/configured | Op-wide | Reusable |
| `quantization_codesign` | `quant_codesign.*` | `QuantizationCoDesignPass` | Planning evidence | Static evidence | Op-wide | Reusable |

`ExecutionPlanBuilder` is a collector over annotated MLIR plus `CapabilityBundle`; it does not run a separate planner.

## 5. Does A Separate Planning-Facts Layer Exist?

| Item | Status | Evidence |
|---|---|---|
| Typed planning-facts object between LLM MLIR and `ExecutionPlan` | Not implemented | Passes write MLIR attrs; `ExecutionPlanBuilder::build` reads MLIR attrs directly. |
| Separate candidate-plan object | Partially implemented | Candidate/evaluated/selected plans are encoded as MLIR attrs (`compiler.candidates`, `compiler.evaluated_candidates`, `selected_plan.*`), not as a separate object or artifact. |
| Target-profile object | Implemented | `compile-for-target` parses `TargetDeviceProfile`, lowers it into `TargetConstraints` attrs and a `CapabilityBundle`. |
| Backend selection | Partially implemented | `ExecutionProviderPlanningPass` selects a function-level primary backend and fallback chain from target profile/KV/replay constraints. This is not a multi-backend cost search. |
| Kernel selection | Partially implemented | `KernelAvailabilityPlanningPass` checks declared kernel libraries. `KernelSelectionPass` matches runtime kernel descriptors when present. Many profiles have no concrete runtime kernels, so selection can be deferred. |
| Memory slot assignment | Not implemented | No tensor lifetime or slot reuse is performed in the LLM path. |
| Cost comparison | Partially implemented | Serving phase compares colocated vs prefill/decode split by formula; candidate evaluation scores candidates by static penalties and optional shape-aware estimates. No measured latency comparison. |
| Unresolved-fact structure | Not implemented as a general layer | Individual attrs record reasons, rejected candidates, deferred kernel selection, or deferred boundaries. There is no central unresolved facts object. |
| Region-level summaries | Not implemented | LLM path is model/function/op oriented, not region-summary oriented. |

## 6. One-Op End-To-End Trace: `llm.qkv_projection`

Representative path: model-spec generated Qwen serving MLIR with AWQ-capable ops.

| Stage | Implemented behavior | Exact kind of decision |
|---|---|---|
| Op creation | `qwen-to-serving-mlir` emits `llm.qkv_projection` in prefill/decode scaffolds. Current source marks projections/MLP `serving.quantizable = true`; `qwen_0_5b_serving_awq.mlir` reflects this. | Template-generated metadata, not discovered from imported weights. |
| Normalization | Model-spec path already uses canonical `llm.qkv_projection`, so `LLMFrontendNormalizationPass` is not needed. Graph-facts/raw paths normalize decomposed attention around q/k/v projections but do not necessarily combine q/k/v into this op. | No decision for this path. |
| Shape/type facts | Tensor types come from MLIR. Shape-cost utilities can derive shape facts when ranks/dimensions are static. Dynamic shapes remain dynamic and reduce cost-model precision. | Static type metadata. |
| Target capability | `compile-for-target` attaches target attrs and builds `CapabilityBundle` from the selected profile. | Configured target facts. |
| Quantization | `WeightClassificationPlanningPass` classifies weight availability. If weights are function args/runtime activations rather than constants, `QuantizationStrategyPlanningPass` may choose `fp16_fallback` even under a global AWQ profile. | Static rule over attrs and weight classification. |
| Layout | `LayoutPlanningPass` uses representation/backend layout prefs and op traits to annotate effective layout and transform need. | Static rule. |
| Kernel availability | `KernelAvailabilityPlanningPass` matches short op names and dtype/layout/quant constraints against declared `kernelLibraries`. If absent, the op is marked unsupported/deferred/fallback as applicable. | Capability lookup, not measured. |
| Lowering decision | `LoweringDecisionPlanningPass` converts kernel status plus boundary requirements into `lowering.decision` and related attrs. | Static rule. |
| Backend candidate/selection | Function backend is selected by `ExecutionProviderPlanningPass`; op candidates are generated later as attrs. | Function-level configured/static backend choice; op-level candidate metadata. |
| Kernel selection | `KernelSelectionPass` selects only if `target.runtime_kernels` has a concrete descriptor matching the op. Otherwise it records deferred/rejected status. | Descriptor match, often deferred. |
| ExecutionPlan serialization | `ExecutionPlanBuilder` emits a per-op decision only if at least one decision bundle is present. Quant/kernel/fallback/tile/selection/cost fields are read from attrs. | Collector serialization. |

The `llm.qkv_projection` trace shows a real implemented path to `ExecutionPlan`, but many fields are configured facts or static metadata rather than measured or searched decisions.

## 7. Target Profile And Hardware Facts Audit

Target profiles live under `configs/target_profiles/`. Current profiles include Apple, NVIDIA CUDA/GTX/datacenter, AMD, Intel, ARM, edge NPU, IREE-style accelerator, and test profiles.

### Target Profile Schema In Practice

`compile-for-target` parses:

- Hardware/profile identity: `profileId`.
- High-level compute units: `configuredComputeUnits`.
- Memory budget: `metalMaxWorkingSetMB`.
- Static shape support.
- Supported precisions.
- Paged-KV-compatible backends.
- Serving cost constants: prefill/decode milliseconds per token and prefill/decode bandwidth.
- `backendCapabilities`.
- `kernelLibraries`.
- `runtimeKernels`.
- Optional `forcedQuantization`.
- Optional `quantizationCoDesignPolicy`.
- Optional `staticCostProfile`, including peak FLOP rates, memory bandwidth, local memory, cache line, async copy, and DMA declarations.

The driver comments explicitly state some fields are parsed for provenance only and not forwarded to compiler decisions, including chip name, total RAM, thermal state, low power mode, model identifier, and CPU count.

### Used Versus Ignored

| Field family | Used by LLM planning? | Notes |
|---|---|---|
| `configuredComputeUnits` | Yes | Lowered to preferred/allowed backends. |
| Memory budget | Yes | Used by KV layout/memory decision. |
| Static shape support | Yes | Used by replay eligibility and candidate validity. |
| Supported precisions | Yes | Used in execution provider/representation/quant planning. |
| Paged KV compatible backends | Yes | Used by KV/backend planning. |
| Serving timing constants | Yes | Used by serving phase analysis formula. |
| Backend capabilities | Yes | Used by representation/layout/boundary/quant/candidate planning. |
| Kernel libraries | Yes | Used by kernel availability planning. |
| Runtime kernels | Yes, when present | Used by `KernelSelectionPass`; many profiles declare none. |
| Forced quantization | Yes, when present | Driver writes global quantization module attrs directly. |
| Static cost profile | Yes, when present | Used by tile planning and shape-aware cost estimation. |
| Thermal/device provenance fields | Mostly ignored | Parsed for provenance only; not forwarded to planning. |

Hardware numbers are configured declarations, not measurements. Several truth boundaries explicitly describe declared or public-spec data rather than calibrated runtime measurements.

Backend support is represented through `backendCapabilities`; kernel availability through `kernelLibraries`; concrete dispatchable kernels through `runtimeKernels`. Memory hierarchy is partially represented in `staticCostProfile`. Transfer cost is represented by serving prefill/decode bandwidth and backend transfer cost fields. Layout and tiling constraints are represented through backend capability alignment/layout fields and static local-memory fields.

The LLM pipeline uses one selected target profile directly. It does not run a combinatorial multi-profile search.

## 8. Memory Planning Audit

In the current LLM/Serving path, "memory planning" primarily means KV-cache policy and byte estimation.

| Memory concern | Status | Evidence |
|---|---|---|
| Tensor lifetime analysis | Not implemented | No pass computes value live ranges for LLM serving. |
| Slot allocation/reuse | Not implemented | No buffer slot assignment appears before `ExecutionPlan`. |
| KV-cache layout | Implemented | `KVLayoutPlanningPass` emits `kv.layout` and byte estimate. |
| Model-weight memory | Metadata only/partial | Weight classification exists, but no full model-weight memory planner. |
| Temporary activation memory | Not implemented | Shape cost may estimate traffic, but no activation memory plan. |
| Peak memory estimate | Partial | KV peak estimate exists; no full tensor peak memory. |
| Backend-specific memory spaces | Metadata only/partial | Static cost profile can declare local memory; no allocation into spaces. |
| Transfer buffers | Not implemented | Transfer costs can be represented; explicit buffers are not planned. |

Memory-related `ExecutionPlan` fields are limited to `MemoryDecision`:

- `kv_cache_layout`
- `estimated_kv_peak_mb`
- `memory_budget_fraction`
- `kv_block_size_tokens`, currently default/unpopulated in observed path
- decision provenance/truth fields through the decision struct

This is not real memory slot allocation.

## 9. Backend And Kernel Decision Audit

| Concern | Implemented logic | Source files | Attrs emitted | ExecutionPlan fields | Value source |
|---|---|---|---|---|---|
| Backend candidate generation | Partial | `CandidateGenerationPass.cpp` | `compiler.candidates`, rejected candidates | Indirect selected/cost evidence | Capability/static rules |
| Backend selection | Implemented at function level | `ExecutionProviderPlanningPass.cpp` | `execution_provider.primary`, fallback chain | `FunctionPlan.backend` | Target profile and static constraints |
| Per-op backend search | Partial | Candidate generation/evaluation/selection passes | candidate/evaluated/selected attrs | Partial via kernel/fallback/cost fields | Static candidate scoring |
| Kernel availability | Implemented | `KernelAvailabilityPlanningPass.cpp` | `kernel.exists`, `kernel.name`, `kernel.library`, `kernel.lowering_status` | `KernelDecision` | Declared kernel libraries |
| Concrete kernel selection | Partial | `KernelSelectionPass.cpp` | `kernel_selection.*` | `KernelSelection` | Runtime kernel descriptors when present |
| Fallback generation | Implemented as metadata | Lowering, alternative, candidate passes | `fallback_backend`, `alternative.*`, selected/fallback attrs | `FallbackDecision` | Capability/static rules |
| Cost comparison | Partial | `ServingPhaseAnalysisPass.cpp`, `ServingCostModelPass.cpp`, `PlanSelectionPass.cpp` | serving cost attrs, evaluated candidates, selected cost attrs | Serving decision, optional shape cost | Analytical/static estimates |
| Decision explanation | Implemented as attrs | Many passes | `*.reason`, `*.truth_boundary`, rejected/deferred attrs | Some fields serialized | Static explanations |

Backend/kernel planning is real in the sense that code executes and emits attrs consumed by the builder. It is not a measured optimizer, and it is not a full multi-backend runtime planning system.

## 10. LLM Path Versus Proposed CV Phase 23/24 Concepts

| Concern | Existing LLM implementation | Proposed CV Phase 23/24 concept | Actually necessary for parity? | Can be simplified? |
|---|---|---|---|---|
| Semantic annotation | LLM semantics are mostly represented as `llm.*` ops and module attrs from frontends. | CV semantic attrs over upstream MLIR. | Yes, because real YOLO-Seg uses upstream ops and needs output/region meaning. | Keep boundary/output attrs focused. |
| Planning facts | No separate planning-facts object. MLIR attrs are the facts carrier. | Separate JSON planning-facts analysis/report. | No. Useful for diagnostics, not required for LLM parity. | Use MLIR planning attrs plus optional report. |
| Region summaries | Not used. LLM path is function/op based. | CV region facts with costs/memory. | No for parity. | Begin with output/region IDs only if builder needs them. |
| Target profile | One profile parsed and lowered into attrs plus `CapabilityBundle`. | Potential new CV planning facts/profile relationship. | Reuse existing target profile style if needed. | Avoid a new schema unless a runtime consumer requires it. |
| Candidate plans | Stored as MLIR attrs, not a typed artifact. | Possible separate candidate/facts artifact. | No. | Use attrs if candidates are needed. |
| Backend selection | Function-level static rule; op-level candidates are optional metadata. | Region/domain candidate derivation. | Minimal configured/static selection is enough for parity. | Avoid multi-backend search initially. |
| Kernel selection | Partial and often deferred unless runtime kernels are declared. | Candidate execution domains and future kernel policy. | No concrete kernel selection required for initial parity if LLM parity allows deferred fields. | Record deferred/unsupported explicitly. |
| Memory planning | KV-cache metadata only; no slot allocation. | Full tensor bytes, lifetime, peak temp, region memory. | No for parity. | Keep tensor/memory report optional; do not block plan export. |
| Cost model | Static analytical formulas and candidate penalties, not measured. | Region FLOPs/bytes and planning facts. | Not required for first parity unless plan schema needs evidence. | Basic static estimates can wait. |
| ExecutionPlan builder | Reads attrs directly from MLIR and `CapabilityBundle`. | Could consume CV planning facts JSON. | Builder should be able to read attrs directly for parity. | Avoid mandatory intermediate JSON. |
| Exporter | Serializes typed `ExecutionPlan`. | Same final artifact desired. | Yes. | Reuse exporter/schema where possible. |

The proposed CV planning-facts layer is richer than the implemented LLM path. It may be valuable as diagnostics, but it is not required to reach parity with the existing Qwen/LLM route to `ExecutionPlan`.

## 11. Minimum CV Parity Recommendation

This is the smallest CV path comparable in architectural depth to the implemented LLM path:

```text
real YOLO-Seg upstream MLIR
  -> CV semantic attrs on func/boundary/root ops
  -> minimal CV/target planning attrs
  -> CV-capable ExecutionPlanBuilder collection
  -> ExecutionPlan JSON
```

### Required For Parity

- Keep numerical computation in upstream MLIR.
- Preserve CV semantic attrs for graph outputs and recognized semantic regions.
- Add or reuse minimal target/profile attrs only for fields the plan builder will serialize.
- Add builder support that reads MLIR attrs directly, following the LLM pattern.
- Export a plan with explicit truth boundaries.
- Represent unsupported/deferred backend or kernel decisions honestly rather than inventing decisions.

### Not Required For Parity

- A separate planning-facts JSON as an input to the builder.
- A separate candidate-plan JSON.
- Full region-level cost accounting.
- Full tensor lifetime and peak-memory slot allocation.
- Multi-backend combinatorial planning.
- Detailed transfer graph.
- Runtime kernel selection if no concrete runtime kernels are declared.
- Measured cost calibration.

### Optional Future Sophistication

- Region-level cost and memory estimates.
- Candidate execution domains per semantic region.
- Kernel descriptor matching for CV ops.
- Backend transfer and layout planning.
- Full tensor lifetime and memory slot planning.
- Measured cost calibration and numerical validation.

## 12. Summary Findings

- Exact `compile-for-target` pass count before plan build: 20 nested function passes.
- The standalone registered serving optimization pipeline has 16 passes.
- `LLMFrontendNormalizationPass` is an optional pre-driver normalization stage for raw/frontend inputs, not part of `compile-for-target`.
- A separate planning-facts layer does not exist for LLM.
- Candidate plans exist only as MLIR attributes, not as a separate object/artifact.
- Backend selection is implemented, but it is a function-level static/configured decision rather than a measured global optimizer.
- Kernel availability is implemented; concrete kernel selection is partial and profile-dependent.
- Cost comparison is partial and analytical/static, not measured.
- Memory planning is KV-cache metadata and byte estimation, not tensor slot allocation.
- `ExecutionPlanBuilder` obtains inputs by reading annotated MLIR attrs and an in-memory `CapabilityBundle`.
- Several CV planning layers recently discussed are not necessary for parity with the current LLM implementation.

## Recommended Next Coding Phase

For CV parity, the next coding phase should implement the smallest builder-facing path from existing upstream YOLO-Seg MLIR annotations into `ExecutionPlan`:

1. Define the minimal CV plan fields or existing decision mappings that can be serialized without changing runtime behavior.
2. Add a CV-specific or generalized builder collection path that reads CV attrs from upstream MLIR directly.
3. Emit an `ExecutionPlan` with honest deferred backend/kernel/memory truth boundaries.
4. Keep rich planning-facts JSON as diagnostics only, not as a required intermediate compiler layer.

## Phase 24 Follow-Up

Phase 24 implemented this minimum parity path. Real YOLO-Seg execution plan
generation now uses:

```text
upstream YOLO-Seg MLIR
  -> cv-semantic-annotation
  -> cv-execution-plan-attrs
  -> existing generic kernel/lowering/selection planning attrs
  -> ExecutionPlanBuilder
  -> ExecutionPlanExporter
```

The Phase 23 `cv_planning_facts.py` report remains diagnostic only. It is not
a required compiler IR boundary for `ExecutionPlan` generation.
