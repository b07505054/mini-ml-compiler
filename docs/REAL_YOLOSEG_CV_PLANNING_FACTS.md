# Real YOLO-Seg CV Planning Facts

Phase 23 adds a planner-facing analysis layer over the Phase 22
CV-annotated upstream MLIR. It consumes semantic attributes and tensor
topology, then emits structured facts for future planning. It does not select
a backend, choose kernels, assign memory slots, generate an `ExecutionPlan`, or
revive legacy numerical `cv.*` operations.

## Inputs

- `artifacts/yoloseg_generic_frontend/yoloseg.cv_annotated.mlir`
- optional `artifacts/yoloseg_generic_frontend/yoloseg.shape_generic_graph_ir.json`

The annotated MLIR is the source of truth for topology and semantic attrs.
The shape IR is used only for graph-boundary and initializer provenance.

## Pipeline

```text
yoloseg.cv_annotated.mlir
  -> tools/cv_planning_facts.py
  -> yoloseg.cv_planning_facts.json
```

Script:

```bash
scripts/run_yoloseg_cv_planning_facts.sh
```

Artifact:

- `artifacts/yoloseg_generic_frontend/yoloseg.cv_planning_facts.json`

Phase 24 note: this artifact remains diagnostic. The real YOLO-Seg
`ExecutionPlan` path does not consume `yoloseg.cv_planning_facts.json`; it
collects equivalent builder-facing information from MLIR attributes, tensor
types, and `CapabilityBundle`, matching the existing LLM path.

Truth boundary:

```text
cv_planning_facts_only_no_backend_selection_no_kernel_selection_no_memory_slot_assignment_no_execution_plan_generation_no_measured_performance
```

## Planning-Facts Schema

The utility uses typed Python dataclasses as the planner-facing representation:

- `CVModelPlanningFacts`
- `CVRegionPlanningFact`
- `CVOutputPlanningFact`
- `CVTensorPlanningFact`

`CVModelPlanningFacts` includes model family, function name, graph
input/output counts, regions, outputs, tensor facts, operation histogram,
memory summary, cost summary, unresolved facts, and provenance.

`CVRegionPlanningFact` includes region id, semantic role, recognition
confidence, operation ids, input/output tensor ids and shapes, dominant dtype,
feature scales, cost estimates, memory estimates, candidate execution domains,
quantization eligibility, fusion eligibility, and planning notes.

`CVOutputPlanningFact` records output role, tensor shape/dtype/layout,
producer region, postprocess boundary, and ownership expectation.

`CVTensorPlanningFact` records stable MLIR-local tensor id, shape, dtype,
layout, byte size, producer, consumers, graph input/output/initializer flags,
temporary flag, semantic role, lifetime, ownership, and provenance.

Stable IDs are derived from MLIR value identity and topological operation
position. Source names are not semantic identity.

## Tensor Facts

For static `f32` tensors:

```text
byte_size = product(shape) * 4
```

The current layout inference is conservative:

- rank-4 tensors: `NCHW`
- rank-3 tensors: `NCX`
- rank-1 tensors: `C`
- scalar tensors: `scalar`
- other ranked tensors: `ranked_unknown_layout`

Dynamic shapes or unsupported dtypes are recorded in `unresolved_facts` rather
than silently assigned invented sizes.

## Cost Estimation

All costs are static analytical estimates, not measured latency.

FLOP convention:

```text
1 MAC = 2 FLOPs
```

Implemented estimates:

- `linalg.conv_2d_nchw_fchw`:
  `2 * N * F * OH * OW * C * KH * KW`
- `linalg.generic` elementwise:
  output element count multiplied by body arithmetic op count
- softmax-like/exp generic:
  approximate reduction/elementwise count when identifiable from body ops
- `linalg.pooling_nchw_max`:
  output element count times comparison count
- `tensor.collapse_shape` / `tensor.expand_shape`:
  view-like, zero moved bytes
- `tensor.extract_slice`:
  view-like in the current analysis, with slice size recorded as moved-byte
  evidence
- `tensor.insert_slice`, `tensor.generate`, `tensor.pad`, `linalg.fill`:
  materializing/data-movement estimates from input and output tensor bytes

The cost summary records model-wide FLOPs, read bytes, write bytes, and FLOPs
by semantic region.

## Lifetime And Memory Model

The lifetime model is a first static topological analysis:

```text
lifetime_start = producer topological position
lifetime_end   = last consumer topological position
```

Graph inputs are externally owned. Initializers are model-state owned when
shape IR initializer metadata is available. Graph outputs extend to function
exit. Temporaries are compiler-owned SSA tensor values produced inside the
function and not returned.

The report includes:

- total tensor bytes
- total initializer bytes
- total temporary bytes
- peak live temporary bytes
- top temporary tensors by size
- top region memory footprints

No slot allocation or reuse plan is produced in Phase 23.

## Candidate Execution Domains

The analysis emits candidates only:

- `accelerator_candidate`
- `cpu_candidate`
- `host_postprocess_candidate`
- `transfer_or_view_operation`
- `unsupported_for_current_target_profile`

It does not map regions to Metal, CUDA, CoreML, CPU, or another concrete
backend. Candidate reasons are recorded for every region.

## Quantization Eligibility

Eligibility is planner-facing only:

- static f32 convolution regions: `eligible`
- view/data-movement regions: `ineligible`
- model output regions: `unknown` until an explicit output precision policy
  exists
- unsupported or unknown dtypes: `unknown`

No quantization decision or rewrite is performed.

## Fusion Eligibility

The analysis records conservative candidate patterns:

- `conv_plus_bias_or_activation_candidate`
- `elementwise_chain_candidate`
- `reshape_slice_view_chain_candidate`
- `resize_concat_boundary_candidate`

No fusion is selected and no graph rewrite is performed.

## Relationship To HIR And ExecutionPlan

Phase 23 does not modify HIR, the `ExecutionPlan` schema, or runtime behavior.
The report is a future planning input that can be consumed before any concrete
backend or kernel decision. It is intentionally separate from the legacy
`CVExecutionPlanBuilder`, which currently reads fixture-driven `cv.*`
operation attrs.

Phase 24 adds a separate canonical `ExecutionPlan` path. It does not make this
planning-facts report mandatory and does not route the real graph through
legacy `cv.*` numerical operations.

## Legacy Infrastructure Reuse

Reviewed components:

- `CVMemoryPlanningPass`: not reused; depends on legacy `cv.*` op names and
  `cv.bytes_estimate` attrs. The generic lifetime idea is reproduced over
  upstream tensor SSA values without slot assignment.
- `CVExecutionDomainPlanningPass`: not reused; classifies hardcoded legacy
  `cv.*` names.
- `CVExecutionPlanBuilder`: not reused; ExecutionPlan schema changes are out
  of scope.
- `ShapeCostModel`: conceptually reused for static shape/dtype accounting, but
  its formulas are LLM/matmul-oriented and not directly reused for YOLO-Seg CV
  ops.

## Real YOLO-Seg Summary

Current report:

- tensor facts: 1004
- total initializer bytes: 13,785,524
- peak live temporary bytes: 31,948,800
- unresolved facts: 0

Estimated FLOPs by region:

| Region | Estimated FLOPs |
| --- | ---: |
| `cv.region.detection_head` | 4,620,000 |
| `cv.region.feature_pyramid` | 0 |
| `cv.region.mask_coefficient_branch` | 0 |
| `cv.region.segmentation_prototype` | 2,009,497,600 |

Model-wide estimated FLOPs:

```text
11,932,092,000
```

Candidate domains:

| Region | Candidates |
| --- | --- |
| detection head | accelerator candidate, CPU candidate |
| segmentation prototype | accelerator candidate, CPU candidate |
| mask coefficient branch | transfer/view operation, CPU candidate |
| feature pyramid | transfer/view operation, CPU candidate |

Quantization/fusion:

- detection head: quantization eligible, fusion candidates present
- segmentation prototype: output precision policy unknown, fusion candidates
  present
- feature pyramid: quantization ineligible, resize/concat fusion boundary
  candidate
- mask coefficient branch: quantization ineligible, no Phase 23 fusion pattern

## Next Phase

Phase 24 should introduce a planner-facing consumer for this JSON schema or an
equivalent in-compiler typed representation. It should still avoid backend
selection unless a target profile is explicitly provided, and it should keep
memory slot assignment separate from this facts-only analysis.
