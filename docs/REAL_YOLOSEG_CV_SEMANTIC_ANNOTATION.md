# Real YOLO-Seg CV Semantic Annotation

Phase 22 keeps the real YOLO-Seg numerical graph in upstream MLIR and adds CV
semantics as structured attributes and a report. It does not introduce new
`cv.*` operations and does not route the real graph through the legacy CV
dialect.

## Boundary

Input:

- `artifacts/yoloseg_generic_frontend/yoloseg.generic.mlir`
- optional shape-annotated GenericGraphIR for report context

Output:

- `artifacts/yoloseg_generic_frontend/yoloseg.cv_annotated.mlir`
- `artifacts/yoloseg_generic_frontend/yoloseg.cv_semantic_report.json`

Truth boundary:

```text
cv_semantic_annotation_only_no_backend_selection_no_memory_plan_no_kernel_selection_no_execution_plan_generation
```

The pass performs semantic annotation only. It is not backend selection, memory
planning, kernel selection, numerical validation, runtime execution, or
`ExecutionPlan` generation.

## Why Upstream MLIR Remains The Numerical IR

The full real YOLO-Seg graph already verifies using only:

- `func`
- `tensor`
- `linalg`
- `arith`
- `math`

The old CV dialect operations duplicate ordinary computation such as
convolution, activation, upsample, and concat. Reintroducing those operations
would only rename upstream MLIR and would violate the architecture rule that
ordinary numerical computation stays in existing upstream dialects.

## Legacy CV Operations

The existing CV dialect and `cv_raw_yoloseg.mlir` remain as legacy
fixture-driven infrastructure. They are not deleted in Phase 22, but the real
YOLO-Seg path does not use `cv.conv2d`, `cv.upsample`, `cv.concat`, or other
duplicate numerical CV operations.

## Attribute Schema

Attributes are attached to `func.func` and selected upstream operations.

| Attribute | Meaning | Initial consumer purpose |
| --- | --- | --- |
| `cv.model_family` | Model-family summary when output contracts and topology support it | planning/reporting boundary |
| `cv.semantic_role` | Semantic role for a boundary/root op or compact region op | future planning and diagnostics |
| `cv.region_id` | Stable region identifier shared by related annotated ops | future region planning |
| `cv.output_role` | Graph output role such as detection or segmentation prototype | output contract planning |
| `cv.feature_scale` | Spatial scale summary for feature tensors | future placement/fusion analysis |
| `cv.postprocess_boundary` | Marks model output boundary before external postprocess | plan boundary |
| `cv.recognition_confidence` | `high`, `medium`, or `low` | confidence-aware planning |
| `cv.recognition_evidence` | Structured evidence strings from topology, shapes, and output reachability | diagnostics and auditability |
| `cv.semantic_annotation.*` | Function-level summary, counts, unresolved questions, truth boundary | artifact reporting |

Source names are not required. If source names are ever consulted, they must be
diagnostic-only evidence, not the recognition rule.

## Recognition Rules

### Output Contracts

Detection output is recognized from:

- return operand type `tensor<1x116x8400xf32>`
- upstream producer topology consistent with assembled detection output

Prototype output is recognized from:

- return operand type `tensor<1x32x160x160xf32>`
- upstream producer topology consistent with linalg/tensor prototype branch

The dimensions are part of the tensor contract for this real artifact. They are
not treated as a model-name string match.

### Detection Head

The pass walks backward from the detection output producer through upstream
reshape, slice, concat, softmax, and detection-assembly dataflow. It uses:

- anchor dimension `8400`
- head tensor contracts
- producer and consumer relationships
- operation families in upstream MLIR

It stops before broadly swallowing the entire backbone or neck.

### Segmentation Prototype

The pass walks backward from the prototype output producer through the mostly
exclusive prototype branch. It recognizes rank-4 prototype tensor contracts
around `80x80` and `160x160` and annotates a compact branch region.

### Mask Coefficients

The pass recognizes the mask coefficient contribution when a
`tensor<1x32x8400xf32>` branch is inserted into the assembled
`tensor<1x116x8400xf32>` detection output. The final output root remains the
detection output; the contributing source side is marked as
`mask_coefficient_branch` when topology is sufficient.

### Feature Pyramid Evidence

The pass recognizes medium-confidence feature pyramid evidence from:

- rank-4 2x resize implemented as `tensor.generate`
- downstream `tensor.insert_slice` fusion/concat data movement
- detection output contract present in the same function
- spatial scales such as `40x40` and `80x80`

Unrelated resize/concat patterns without detection-output context are not
promoted to feature pyramid semantics.

### Backbone

Phase 22 does not label a backbone region. The current evidence separates
heads and feature-fusion evidence, but not a reusable structural backbone
boundary without risking a topological-position-only guess.

## Confidence Model

High confidence is used when output contract, producer topology, and backward
dataflow agree. Medium confidence is used for feature pyramid evidence because
the detected resize/concat pattern is meaningful but not yet consumed by a
planning pass. Low confidence is not emitted in the real YOLO-Seg artifact; an
ambiguous semantic area remains unresolved instead of being guessed.

## Region Representation

Phase 22 uses a compact operation-attribute representation:

- function attributes hold model-level summary and counts
- output/root operations carry `cv.output_role` and boundary attrs
- selected upstream operations in compact backward regions share
  `cv.region_id` and `cv.semantic_role`

The pass intentionally does not annotate all 268 nodes. Later planning
consumers can expand region ownership if they need whole-graph partitioning.

## Real YOLO-Seg Recognition Result

Script:

```bash
scripts/run_yoloseg_cv_semantic_annotation.sh
```

Result:

| Region | Role(s) | Annotated ops | Confidence |
| --- | --- | ---: | --- |
| `cv.region.detection_head` | `detection_head`, `detection_output` | 21 report-visible ops; function summary records 22 collected ops | high |
| `cv.region.segmentation_prototype` | `segmentation_prototype` | 10 | high |
| `cv.region.mask_coefficient_branch` | `mask_coefficient_branch` | 1 | high |
| `cv.region.feature_pyramid` | `feature_pyramid` | 4 | medium |

Output roles:

- `detection`
- `segmentation_prototype`

Model-level annotation:

- `cv.model_family = "yoloseg"`
- `cv.semantic_annotation.source_name_dependency = "none"`
- `cv.semantic_annotation.unresolved = []`

The detection region has one function-summary count that is not visible as a
line-level region op in the report because one collected internal operation is
superseded by the more specific mask-coefficient annotation. The annotated MLIR
itself is the source of truth for attributes; the JSON report is a diagnostic
summary.

## Unresolved Semantic Areas

- Backbone boundary: not annotated until a reusable structural separation is
  established.
- Fine-grained detection subroles such as box regression, class logits, and
  distribution focal loss are not separated yet.
- Feature pyramid is currently evidence-level, not a full region ownership
  model.
- No planning consumer has been attached to these attrs yet.

## Next Planning Consumers

Phase 23 adds the first planner-facing facts consumer for these attributes:

- consume `cv.output_role` for model output planning
- consume `cv.region_id` for region grouping
- keep memory planning generic over tensor/memref types
- avoid dependency on legacy `cv.*` numerical operations

Artifact:

- `artifacts/yoloseg_generic_frontend/yoloseg.cv_planning_facts.json`

See `docs/REAL_YOLOSEG_CV_PLANNING_FACTS.md`.
