# Real YOLO-Seg ExecutionPlan Path

Phase 24 adds the minimum CV-to-`ExecutionPlan` parity path modeled on the implemented LLM path.

The compiler path is:

```text
YOLO-Seg ONNX
  -> ImportedGraphIR
  -> GenericGraphIR
  -> upstream MLIR
  -> CV semantic attrs
  -> CV planning/decision attrs
  -> generic kernel/lowering/selection planning attrs
  -> ExecutionPlanBuilder
  -> ExecutionPlan JSON
```

The path does not use legacy `cv.*` numerical operations and does not depend on the Phase 23 planning-facts JSON.

## Comparison With LLM Path

The LLM audit showed that the real Qwen path uses MLIR attributes as the planning carrier. `ExecutionPlanBuilder` reads those attrs plus `CapabilityBundle`; there is no separate typed planning-facts object between MLIR and the plan.

The YOLO-Seg path follows that same pattern:

- target profile JSON is parsed by `compile-for-target`;
- target profile data is lowered into `target.*` attrs and `CapabilityBundle`;
- `cv-execution-plan-attrs` writes builder-facing attrs on upstream MLIR;
- existing generic kernel/lowering/selection passes refine those attrs when a profile declares capabilities;
- `ExecutionPlanBuilder` collects attrs directly from MLIR.

## Pass Order

Real YOLO-Seg execution plan generation uses:

1. `scripts/run_yoloseg_generic_mlir_emission.sh`
2. `cv-semantic-annotation`
3. `compile-for-target` with an explicit target profile

Inside `compile-for-target`, the relevant order is:

1. target profile lowering to `target.*` attrs and `CapabilityBundle`
2. existing LLM serving passes, which skip CV functions without attention ops
3. `WeightClassificationPlanningPass`
4. `QuantizationStrategyPlanningPass`
5. `CVExecutionPlanAttrsPass`
6. `KernelAvailabilityPlanningPass`
7. `LoweringDecisionPlanningPass`
8. `QuantizedBoundaryRefinementPass`
9. `TilePlanningPass`
10. `KernelSelectionPass`
11. `QuantizationCoDesignPass`
12. `AlternativeLoweringPlanningPass`
13. `CandidateGenerationPass`
14. `ServingCostModelPass`
15. `PlanSelectionPass`
16. `BoundaryMaterializationPass`
17. `ExecutionPlanBuilder`
18. `ExecutionPlanExporter`

`CVExecutionPlanAttrsPass` is skip-safe: it only runs on functions with `cv.semantic_annotation.status = "completed"`.

## Attributes Produced

Function-level attrs:

- `serving.policy = "cv_full_graph"`
- `serving.truth_boundary`
- `execution_provider.primary`
- `execution_provider.fallback_chain`
- `execution_provider.decision_source = "cv-target-profile-static-policy"`
- `execution_provider.required_precision = "f32"`
- `execution_provider.required_kv_layout = "not_applicable"`
- `execution_provider.requires_replay = false`
- `representation.effective_dtype = "f32"`
- `representation.preferred_activation_layout = "nchw"`
- `representation.preferred_weight_layout = "fchw"`
- `representation.source_backend`
- `cv.execution_plan.status`
- `cv.execution_plan.pass_order`
- `cv.execution_plan.truth_boundary`
- `cv.memory.estimated_input_bytes`
- `cv.memory.estimated_output_bytes`
- `cv.memory.estimated_temporary_bytes`
- `cv.memory.estimated_total_tensor_bytes`
- `cv.memory.status`
- `cv.memory.truth_boundary`

Per-op attrs on upstream tensor-producing ops:

- `layout.effective_layout`
- `layout.required_input_layout`
- `layout.transform_required = false`
- `quant.strategy = "none"`
- `quant.weight_dtype`
- `quant.activation_dtype`
- `quant.accumulation_dtype`
- `quant.granularity = "not_applicable"`
- `quant.decision_reason`
- `kernel.backend`
- `kernel.exists = false`
- `kernel.lowering_status`
- `kernel.fallback_backend`
- `kernel.truth_boundary`

Existing generic passes may replace seeded `kernel.*` attrs when target `kernelLibraries` contain a compatible real kernel entry.

## ExecutionPlan Fields

The canonical `ExecutionPlan` now has an optional `cv_extension`. It is emitted only when CV execution-plan attrs are present.

The extension includes:

- model family;
- function name;
- target profile ID;
- graph input tensor contracts;
- graph output tensor contracts;
- semantic regions;
- static byte estimates;
- postprocess boundary;
- truth boundary.

The existing plan fields remain in use:

- `model_identity`
- `provenance.capability_bundle`
- `global_decisions.serving`
- `function_plans[].backend`
- `function_plans[].per_op_decisions`

Qwen plans do not receive `cv_extension` unless CV attrs are present.

## Target Profile Use

The script requires a target profile. By default:

```text
configs/target_profiles/apple_a17pro_mobile.json
```

Override with:

```bash
YOLOSEG_TARGET_PROFILE=configs/target_profiles/<profile>.json scripts/run_yoloseg_execution_plan.sh
```

The selected backend is derived from the same `target.preferred_backend` / `target.allowed_backends` attrs used by the LLM path. For the default Apple A17 Pro profile, the generated plan selects `coreml` with `metal` and `cpu` fallbacks.

## Backend And Kernel Truth Boundary

The Phase 24 plan does not claim production kernel availability.

Seeded CV kernel attrs start as:

- `kernel.exists = false`
- `kernel.lowering_status = "fallback_required"` when a fallback backend exists;
- `kernel.lowering_status = "unsupported"` otherwise.

`KernelAvailabilityPlanningPass` may refine these only when the target profile declares a compatible kernel library entry. `KernelSelectionPass` still requires concrete runtime kernel descriptors and records rejection/deferral when none match.

## Memory Scope

Memory planning is static metadata only:

- input tensor bytes;
- output tensor bytes;
- sum of top-level temporary tensor result bytes;
- total static tensor byte estimate.

There is no tensor lifetime slot assignment, allocation reuse, backend memory-space allocation, runtime buffer planning, or measured peak memory.

## Real Generated Plan Summary

Current artifact:

```text
artifacts/yoloseg_generic_frontend/yoloseg.execution_plan.json
```

Observed summary with the default Apple profile:

- function plans: 1
- function: `main_graph`
- selected backend: `coreml`
- fallback backends: `metal`, `cpu`
- output roles:
  - detection: `[1,116,8400] f32`
  - segmentation prototype: `[1,32,160,160] f32`
- semantic regions:
  - `cv.region.detection_head`
  - `cv.region.segmentation_prototype`
  - `cv.region.mask_coefficient_branch`
  - `cv.region.feature_pyramid`

Truth boundary:

```text
real_yoloseg_execution_plan_compiler_decisions_from_static_capability_and_analysis_no_runtime_execution_no_measured_performance_no_full_memory_slot_allocation
```

## Limitations

- No runtime execution.
- No numerical equivalence validation.
- No backend code generation.
- No production CV runtime kernels are claimed unless a profile explicitly declares them.
- No optimized multi-backend search.
- No memory slot allocation.
- Per-op decisions are intentionally verbose because every tensor-producing upstream op carries explicit static quant/kernel/layout evidence.

