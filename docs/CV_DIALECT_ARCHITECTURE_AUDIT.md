# CV Dialect Architecture Audit

This audit evaluates the existing `cv` MLIR dialect before any further
YOLO-Seg integration work. It is audit-only: no dialect code, passes,
ExecutionPlan code, Qwen behavior, or lowering pipelines were changed.

## Rules From CLAUDE.md

Relevant project rules, extracted from `CLAUDE.md`:

- Prefer existing upstream MLIR dialects over introducing custom dialects.
- First check whether an operation can be represented with existing dialects
  such as `func`, `tensor`, `arith`, `math`, `linalg`, `memref`, `scf`,
  `affine`, `vector`, `tosa`, or `stablehlo`.
- Introduce a custom dialect only when existing dialects cannot express the
  semantics cleanly, when meaningful compiler/domain semantics would otherwise
  be lost, or when the custom dialect enables analyses or optimizations that
  cannot reasonably be implemented on existing dialects.
- Never introduce a custom dialect solely because an operation has a different
  name from ONNX or because a new op is simpler.
- Every custom op proposal must explain why upstream dialects are insufficient,
  what compiler semantics the op carries, what analysis/optimization it enables,
  and why direct lowering to upstream dialects is worse.
- Architecture planning order: existing dialect, metadata/attributes,
  compiler pass, then custom dialect/operation only if earlier choices are
  insufficient.
- Existing infrastructure order: upstream dialect, existing project component,
  pass, metadata/attributes, new IR abstraction, new dialect.
- Avoid duplicate abstractions introduced only for organization.
- Compiler outputs are static planning artifacts; runtime execution, measured
  performance, and dynamic scheduling claims are out of scope unless explicitly
  measured elsewhere.

These rules are already clear enough for this decision. No `CLAUDE.md`
clarification is required.

## Current Architecture Placement

The current `cv` dialect lives under `mlir_passes/include/CV/IR` and
`mlir_passes/lib/CV/IR`. It is registered into the main MLIR pass plugin
`HIRMatMulBiasReluFusionPass` and into the standalone
`emit-cv-execution-plan` tool.

Current CV artifact path:

```text
handwritten pseudo-CV MLIR
  -> cv-frontend-normalization
  -> cv-shape-inference
  -> cv-memory-planning
  -> cv-execution-domain-planning
  -> CVExecutionPlanBuilder
  -> CVExecutionPlanExporter
  -> cv_execution_plan.json
```

The current real YOLO-Seg path does not use this dialect:

```text
YOLO-Seg ONNX
  -> ImportedGraphIR
  -> GenericGraphIR
  -> existing-dialect MLIR
  -> mlir-opt verification
  -> one-shot bufferization
```

The real graph emits only `func`, `tensor`, `linalg`, `arith`, and `math`
before bufferization.

## Implementation Inventory

Inspected files:

- `mlir_passes/include/CV/IR/CVDialect.td`
- `mlir_passes/include/CV/IR/CVOps.td`
- `mlir_passes/include/CV/IR/CVDialect.h`
- `mlir_passes/include/CV/IR/CVOps.h`
- `mlir_passes/lib/CV/IR/CVDialect.cpp`
- `mlir_passes/lib/CV/IR/CVOps.cpp`
- `mlir_passes/lib/serving/CVFrontendNormalizationPass.cpp`
- `mlir_passes/lib/serving/CVShapeInferencePass.cpp`
- `mlir_passes/lib/serving/CVMemoryPlanningPass.cpp`
- `mlir_passes/lib/serving/CVExecutionDomainPlanningPass.cpp`
- `mlir_passes/include/serving/CVExecutionPlan.h`
- `mlir_passes/lib/serving/CVExecutionPlanBuilder.cpp`
- `mlir_passes/lib/serving/CVExecutionPlanExporter.cpp`
- `mlir_passes/tools/emit-cv-execution-plan/main.cpp`
- `mlir/cv_raw_yoloseg.mlir`
- `mlir_passes/test/cv/cv_dialect_ops.mlir`
- `mlir_passes/test/serving/cv_frontend_normalization.mlir`
- `mlir_passes/test/serving/cv_shape_inference.mlir`
- `mlir_passes/test/serving/cv_memory_planning.mlir`
- `mlir_passes/test/serving/cv_execution_domain_planning.mlir`
- `mlir_passes/test/serving/CVExecutionPlanBuilderTest.cpp`
- `mlir_passes/test/serving/CVExecutionPlanExporterTest.cpp`
- `mlir_passes/CMakeLists.txt`
- `mlir_passes/README.md`
- `include/frontend/cv_graph_builder.h`
- `src/frontend/cv_graph_builder.cpp`
- `apps/run_cv_graph_demo.cpp`
- `tests/frontend/test_cv_graph_builder.cpp`

Generated op declarations/definitions come from TableGen. The dialect has no
custom builders, custom parsers/printers, traits beyond `Pure`, interfaces,
canonicalization patterns, or conversion/lowering passes. Declarative assembly
formats are used.

## Current CV Dialect Inventory

| Operation | Operands | Results | Attributes | Verifier | Current producer | Current consumers/passes | Runtime or plan effect | Ordinary numerical computation? | Upstream representation | CV-specific semantic information preserved | Dependent analysis/optimization | Exists only because of handwritten fixture/tests? | Used by real YOLO-Seg MLIR? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cv.conv2d` | one `AnyTensor` input | one `AnyTensor` output | arbitrary attrs in fixtures such as `cv.source_op`, `cv.in_channels`, `cv.out_channels`, stride attrs | none | `cv_raw_yoloseg.mlir`, CV tests | counted by frontend normalization; shape/memory/domain passes key off `cv.*` or op name | classified accelerated; exported as a CV step | yes | `linalg.conv_2d_nchw_fchw` plus `tensor.pad`, `linalg.fill`, bias `linalg.generic` | only the name "conv2d"; attrs are not verified | op-name domain classification only | yes | no |
| `cv.batch_norm` | one `AnyTensor` input | one `AnyTensor` output | arbitrary fixture attrs | none | fixture/tests | counted; shape/memory/domain passes | classified accelerated; exported as a CV step | yes | linalg/arith elementwise affine normalization, or folded into conv when valid | only the name "batch_norm" | op-name domain classification only | yes | no |
| `cv.silu` | one `AnyTensor` input | one `AnyTensor` output | arbitrary fixture attrs | none | fixture/tests | counted; shape/memory/domain passes | classified accelerated; exported as a CV step | yes | `linalg.generic` with sigmoid/mul via `math.exp` and `arith` | only the name "silu" | op-name domain classification only | yes | no |
| `cv.upsample` | one `AnyTensor` input | one `AnyTensor` output | arbitrary attrs such as scale/mode | none | fixture/tests | counted; shape/memory/domain passes | classified accelerated; exported as a CV step | yes | current real path uses `tensor.generate`, `tensor.extract`, `arith.divui` for nearest 2x; other forms could use upstream tensor/linalg patterns | only the name "upsample"; attrs are not semantically checked | op-name domain classification only | yes | no |
| `cv.concat` | variadic `AnyTensor` inputs | one `AnyTensor` output | arbitrary attrs such as axis | requires at least one input | fixture/tests | counted; shape/memory/domain passes | classified accelerated; exported as a CV step | yes | `tensor.empty` plus `tensor.insert_slice` | only the name "concat"; axis not verified by op | op-name domain classification only | yes | no |
| `cv.detect_head` | variadic `AnyTensor` inputs | one `AnyTensor` output | arbitrary attrs such as classes/anchors | requires at least one input | fixture/tests | frontend gate anchor; counted; shape/memory/domain passes | classified host; exported as a CV step | partly semantic boundary, but current op has no contract | real graph output region is upstream ops ending in bbox/class/mask concat; semantic role can be attrs on region/output | intended detection-head boundary, but no verified tensor contract | op-name host classification and frontend gate | yes | no |
| `cv.prototype_head` | one `AnyTensor` input | one `AnyTensor` output | arbitrary attrs such as mask count | none | fixture/tests | frontend gate anchor; counted; shape/memory/domain passes | classified host; exported as a CV step | partly semantic boundary, but current op has no contract | real graph prototype branch is convtranspose/conv/silu expressed upstream; semantic role can be attrs on branch/output | intended prototype-mask branch boundary, but no verified tensor contract | op-name host classification and frontend gate | yes | no |
| `cv.custom_op` | one `AnyTensor` input | one `AnyTensor` output | arbitrary fixture attrs | none | domain-planning tests only | execution-domain pass | classified fallback; exported if present | unknown | none; test sentinel only | none | fallback branch test | yes | no |

## Per-Operation Semantic-Value Test

Classification uses the requested categories exactly.

| Operation | Lost if immediately lowered to upstream MLIR? | Concrete pass decision uses it? | Enables fusion/placement/layout/memory/quant/kernel/runtime planning? | Simpler representation possible? | Merely renamed upstream op? | Stable across CV models? | Classification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cv.conv2d` | no; upstream conv preserves computation better | only op-name counting/domain | placement only by hardcoded op name | attribute or analysis over `linalg.conv_2d_*` | yes | yes as computation, but not as custom op | `REPLACE_WITH_UPSTREAM_OP_PLUS_ATTRIBUTES` |
| `cv.batch_norm` | no; no parameters or epsilon semantics are modeled | only op-name counting/domain | placement only by hardcoded op name | upstream elementwise/folded conv plus attrs if needed | yes | yes as computation, but not as custom op | `REPLACE_WITH_UPSTREAM_OP_PLUS_ATTRIBUTES` |
| `cv.silu` | no | only op-name counting/domain | placement only by hardcoded op name | upstream `linalg.generic`/`math` plus attrs if needed | yes | yes as computation, but not as custom op | `REPLACE_WITH_UPSTREAM_OP_PLUS_ATTRIBUTES` |
| `cv.upsample` | no for selected nearest form; upstream captures exact semantics | only op-name counting/domain | placement only by hardcoded op name | upstream tensor/linalg plus attrs | yes | yes as computation, but not as custom op | `REPLACE_WITH_UPSTREAM_OP_PLUS_ATTRIBUTES` |
| `cv.concat` | no; upstream slice insertion captures exact layout | only op-name counting/domain | placement only by hardcoded op name | upstream `tensor.insert_slice` pattern plus attrs | yes | yes as computation, but not as custom op | `REPLACE_WITH_UPSTREAM_OP_PLUS_ATTRIBUTES` |
| `cv.detect_head` | potentially semantic role would be lost, but current op does not prove or verify it | frontend gate and host classification use name | host classification only | region/output attrs over recognized upstream topology are sufficient for current planning | no, but current op is only a marker | plausible across detectors, current contract is not stable enough | `REPLACE_WITH_UPSTREAM_OP_PLUS_ATTRIBUTES` |
| `cv.prototype_head` | potentially semantic role would be lost, but current op does not prove or verify it | frontend gate and host classification use name | host classification only | region/output attrs over recognized upstream topology are sufficient for current planning | no, but current op is only a marker | plausible across segmentation models, current contract is not stable enough | `REPLACE_WITH_UPSTREAM_OP_PLUS_ATTRIBUTES` |
| `cv.custom_op` | no | fallback branch test only | no production planning value | test-only unknown op fixture | yes as a placeholder | no | `DEPRECATE_FIXTURE_ONLY_OP` |

Strict result:

- True semantic operations under current implementation: 0
- Intended semantic marker operations that need evidence/contracts:
  `cv.detect_head`, `cv.prototype_head`
- Duplicate ordinary computation markers: `cv.conv2d`, `cv.batch_norm`,
  `cv.silu`, `cv.upsample`, `cv.concat`
- Fixture-only sentinel: `cv.custom_op`

## Existing CV Pass Audit

| Component | Current behavior | Operates on meaningful semantics? | Hardcoded `cv.*` names? | Duplicates generic metadata? | Fixture dependency | Could operate on upstream MLIR? | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `CVFrontendNormalizationPass` | requires at least one `cv.conv2d`, `cv.detect_head`, and `cv.prototype_head`; stamps `raw_pseudo_cv_mlir` and count attrs | no; detects marker names | yes | yes, counts are trivial | yes, truth boundary says handwritten pseudo-CV | should become a structural analysis over upstream graph topology and output contracts | rewrite |
| `CVShapeInferencePass` | computes `num_elements` and `bytes_estimate` from static result type; assumes fp16 | no; any static tensor op would work | only gates on `cv.` prefix | yes, shapes already exist in GenericGraphIR and MLIR tensor types; dtype assumption is wrong for real f32 YOLO-Seg | yes | yes, as a generic static tensor/memref size analysis | rewrite or fold into generic shape/memory analysis |
| `CVMemoryPlanningPass` | linear first-fit buffer assignment using `cv.bytes_estimate`, single-block order, only `cv.*` single-result ops | no; generic SSA lifetime analysis | gates on `cv.` prefix | partly duplicates post-bufferization allocation/memory facts | tests use hand-authored bytes or CVShapeInference attrs | yes, as generic op/result lifetime analysis over upstream or bufferized IR | rewrite as generic analysis if still needed |
| `CVExecutionDomainPlanningPass` | classifies op names: conv/bn/silu/upsample/concat accelerated, detect/prototype host, unknown fallback | weak; op-name policy only | yes | no, but policy ignores actual backend/kernel availability and shapes | yes | yes, if semantic role attrs or generic op classes are present | rewrite to consume semantic attrs and/or generic op classes |
| `CVExecutionPlanBuilder` | exports any function with completed CV execution-domain plan; collects `cv.*` op attrs | no; collector only | yes | tied to CV attrs | yes | yes, after schema changes to consume role/domain/memory attrs independent of custom ops | preserve concept, rewrite input contract |
| `CVExecutionPlanExporter` | serializes typed CV plan JSON | no; serialization only | indirectly through builder data | no | fixture artifact | yes, if builder output schema is retained | can survive with renamed/generalized input data |
| `emit-cv-execution-plan` | parses CV MLIR, runs four CV passes, exports JSON | no | yes | yes | yes | should be replaced by a real upstream-MLIR analysis/export tool only if needed | retire or replace |
| CV dialect parser/printer smoke tests | verifies TableGen op parsing/printing | no | yes | no | yes | not needed if custom ops are removed | replace with attribute-analysis tests |
| CVGraphBuilder C++ demo | builds toy `Graph` IR `Conv2D -> BatchNorm -> ReLU -> MaxPool -> Flatten -> Linear` | unrelated to MLIR CV dialect | no | separate toy graph path | no direct dialect link | not relevant to real YOLO-Seg MLIR | leave untouched unless demo cleanup is requested |

No CV lowering/conversion pass exists. The CV pipeline is annotation/export
only.

## Comparison With Real YOLO-Seg MLIR

Inspected real artifacts:

- `artifacts/yoloseg_generic_frontend/yoloseg.generic.mlir`
- `artifacts/yoloseg_generic_frontend/yoloseg.generic.verified.mlir`

Real emitted operation counts by source op type:

| Source op type | Count |
| --- | ---: |
| Conv | 76 |
| ConvTranspose | 1 |
| MaxPool | 3 |
| Resize | 2 |
| Concat | 18 |
| Split | 8 |
| Reshape | 11 |
| Slice | 2 |
| Transpose | 1 |
| Softmax | 1 |
| Sigmoid | 67 |
| Mul | 67 |
| Add | 8 |
| Sub | 2 |
| Div | 1 |

The real graph uses:

- `linalg.conv_2d_nchw_fchw` for Conv
- `linalg.generic` plus `math.exp`/`arith` for SiLU/sigmoid/elementwise
- `tensor.generate`/`tensor.extract` for selected resize
- specialized `linalg.generic` for selected ConvTranspose
- `tensor.insert_slice` for concat
- `tensor.extract_slice` for split/slice
- `tensor.collapse_shape`/`tensor.expand_shape` for reshape
- stable softmax using `linalg.generic`, `math.exp`, and `arith`

Correspondence to current `cv.*` ops:

- `cv.conv2d`, `cv.silu`, `cv.upsample`, and `cv.concat` correspond to
  ordinary numerical/data-movement operations already represented upstream.
- `cv.batch_norm` does not correspond to a remaining real YOLO-Seg op in the
  emitted graph; any batchnorm-like behavior has been folded or absent.
- `cv.detect_head` and `cv.prototype_head` correspond only to possible semantic
  regions, not to single real operations.
- `cv.custom_op` has no real counterpart.

Meaningful regions visible in the real graph without source names:

- Output `tensor<1x116x8400xf32>` is assembled from:
  - `tensor<1x4x8400xf32>` decoded/scaled box coordinates,
  - `tensor<1x80x8400xf32>` class probabilities,
  - `tensor<1x32x8400xf32>` mask coefficients.
- Output `tensor<1x32x160x160xf32>` is produced by a prototype branch with
  upsampling/transposed convolution and final activation.
- Multi-scale detection tensors are visible through three spatial extents:
  80x80, 40x40, and 20x20, collapsed to 6400, 1600, and 400 anchors and
  concatenated to 8400.
- The DFL-like bbox decode region is visible as reshape/transpose/softmax/1x1
  conv/slice/add/sub/div/mul over the 8400 anchor dimension.

These are real topology and tensor-contract facts. They do not require custom
numerical operations. They can be represented as attributes or analysis results
on existing upstream operations, function outputs, or recognized regions:

- `cv.semantic_role = "detection_output" | "prototype_output" |
  "mask_coefficients" | "class_scores" | "bbox_decode" | "feature_scale"`
- `cv.region_id`
- `cv.output_role`
- `cv.feature_scale`
- `cv.model_family`, if justified by structural proof
- `cv.postprocess_boundary`

The current CV dialect cannot represent the real graph correctly because it
compresses large real regions into unverified marker ops and omits the real
operator topology that now exists in upstream MLIR.

## Violations Of Project Rules

Strictly against the `CLAUDE.md` rules:

1. `cv.conv2d`, `cv.silu`, `cv.upsample`, and `cv.concat` rename computations
   that upstream dialects already represent in the real YOLO-Seg emitter.
2. `cv.batch_norm` is a marker with no operands for scale/bias/mean/variance
   and no epsilon semantics; it is neither a correct numerical op nor a
   useful semantic boundary.
3. `cv.detect_head` and `cv.prototype_head` are plausible semantic names but
   lack contracts, interfaces, verifiers, or topology evidence. Today they
   exist because the fixture says they exist.
4. CV passes use op names as policy instead of using traits, interfaces,
   semantic attributes, or structural analyses.
5. `CVShapeInferencePass` duplicates static shape metadata and assumes fp16,
   while the real YOLO-Seg MLIR is f32.
6. The dialect provides no concrete optimization that cannot be implemented
   over upstream MLIR plus attributes.
7. The truth boundary in the docs and fixture explicitly says
   `raw_pseudo_cv_mlir_not_full_onnx_importer`, so this path is not evidence
   that the dialect is valid for the real YOLO-Seg graph.

## Candidate Architectures

### Option A - Keep The Existing CV Dialect

Use only if current operations carry reusable CV semantics.

Assessment:

- Not valid for the current repository evidence.
- Five ops are ordinary computation/data movement.
- Two semantic-looking ops are unverified markers.
- One op is a fallback test sentinel.
- The real YOLO-Seg path never enters the dialect.

Operations that would stay under this option:

- All eight current ops.

How real YOLO-Seg would enter it:

- A new lowering/recognition pass would have to collapse verified upstream
  regions into `cv.*` marker ops. That would discard useful detailed topology
  and duplicate upstream semantics.

Reusable passes:

- Only the JSON exporter concept is reusable. The existing op-name passes are
  not strong enough.

Verdict: reject.

### Option B - Reduce The CV Dialect

Keep only high-value semantic boundary operations, for example detection head,
segmentation prototype branch, mask coefficient output, or postprocess boundary.

Assessment:

- Better than Option A, but still premature.
- Current `cv.detect_head` and `cv.prototype_head` are not strong enough to
  keep unchanged because they do not verify tensor contracts or expose
  interfaces.

Existing operations that could conceptually survive after redesign:

- `cv.detect_head`: only if redefined as a verified semantic region/boundary,
  not as a generic variadic marker.
- `cv.prototype_head`: only if redefined as a verified semantic region/boundary,
  not as a unary marker.

Existing operations that should not survive:

- `cv.conv2d`
- `cv.batch_norm`
- `cv.silu`
- `cv.upsample`
- `cv.concat`
- `cv.custom_op`

Verdict: plausible future if attributes prove insufficient, but not the best
next step.

### Option C - Replace CV Operations With Attributes And Analyses

Keep the real graph in upstream dialects and attach semantic metadata:

- `cv.semantic_role`
- `cv.region_id`
- `cv.output_role`
- `cv.feature_scale`
- `cv.model_family`
- `cv.postprocess_boundary`

Assessment:

- Best match for current evidence.
- Preserves exact numerical topology in upstream dialects.
- Avoids renaming ordinary computation.
- Gives planning passes semantic hooks without inventing custom ops.
- Allows CV memory/domain/export planning to become analyses over real MLIR.

Sufficiency for planning:

- Execution-domain planning can classify roles or regions:
  tensor-heavy upstream regions as accelerated candidates, decoded output or
  postprocess boundaries as host candidates.
- Memory planning can operate over tensor/memref values and static shape/dtype
  information without `cv.*` ops.
- ExecutionPlan generation can consume semantic attrs and region summaries.
- Future backend policy can use existing linalg/tensor/memref ops plus
  role metadata.

Verdict: recommended.

### Option D - Rebuild The CV Dialect

Use only if current dialect is fixture-driven or semantically incorrect and
attributes are insufficient.

Assessment:

- The current dialect is fixture-driven.
- A rebuilt dialect could be valid only if it models semantic boundaries that
  cannot be represented as attributes, for example a region op with explicit
  nested upstream IR and verified contracts.

Current code that would be deprecated:

- all current `cv.*` numerical marker ops
- op-name frontend/domain passes
- `cv_raw_yoloseg.mlir` as anything other than a legacy fixture

Minimal replacement, if needed later:

- a small semantic dialect with region-bearing boundary ops such as
  `cv.detection_region` or `cv.segmentation_output_region`
- explicit verifier contracts for output roles, anchors, class count, mask
  coefficient count, and prototype tensor shape
- interfaces exposing planning role, output role, and postprocess boundary

Why attributes may be insufficient later:

- If planning needs to move, clone, outline, or lower whole semantic regions as
  units, a region op could provide stronger ownership than attributes.

Verdict: not selected now; keep as fallback after attribute analysis is tried.

## Final Recommendation

Primary recommendation: `REPLACE_WITH_ATTRIBUTES`

Confidence: high

Evidence:

- Real YOLO-Seg MLIR is fully represented by upstream dialects and verifies.
- Full graph bufferization also verifies without `cv.*`.
- Current CV numerical ops duplicate upstream operations.
- Current CV passes make decisions from op names, not semantic interfaces or
  verified contracts.
- `cv_raw_yoloseg.mlir` openly declares that it is handwritten pseudo-CV, not
  a real ONNX-imported graph.
- No current `cv.*` operation is produced by the real YOLO-Seg path.

Risks:

- Attribute-only semantic recognition must be written carefully from topology,
  shapes, and tensor contracts, not from source names.
- Region-level metadata can become fragile if represented only as loose string
  attrs. Use structured attrs and tests.
- If later transformations need to move semantic regions as units, a rebuilt
  semantic region dialect may become justified.

Migration cost:

- Moderate. The current CV dialect code is small, but tests and the
  `emit-cv-execution-plan` artifact path are tied to `cv.*` names.
- Existing real YOLO-Seg emitter and bufferization pipeline do not need
  conversion away from `cv.*` because they never used it.

Compatibility impact:

- Qwen GraphFacts and Qwen serving behavior should remain untouched.
- Existing CV pseudo-dialect tests would need replacement when code changes
  happen, but this audit makes no code changes.
- Existing `artifacts/apple_demo/cv_execution_plan.json` and integration
  bundle CV plan artifacts should be treated as legacy pseudo-CV artifacts.

Tests that would need replacement in a future coding phase:

- `mlir_passes/test/cv/cv_dialect_ops.mlir`
- `mlir_passes/test/serving/cv_frontend_normalization.mlir`
- `mlir_passes/test/serving/cv_shape_inference.mlir`
- `mlir_passes/test/serving/cv_memory_planning.mlir`
- `mlir_passes/test/serving/cv_execution_domain_planning.mlir`
- `CVExecutionPlanBuilderTest`
- `CVExecutionPlanExporterTest`
- `CVExecutionPlanArtifactToolTest`

Whether current CV ExecutionPlan code can survive:

- `CVExecutionPlanExporter` can mostly survive conceptually as serialization.
- `CVExecutionPlanBuilder` can survive only after its input contract changes
  from collecting `cv.*` ops to collecting semantic role/region attrs from
  upstream MLIR.
- The four CV passes require rewrite.

Status of `cv_raw_yoloseg.mlir`:

- Keep only as a legacy fixture while old tests exist.
- Do not use it as evidence for real YOLO-Seg integration.
- Retire it once replacement attribute-analysis tests over real or reduced
  upstream MLIR snippets exist.

## Migration Plan

1. Freeze the existing CV dialect as legacy pseudo-CV. Do not route real
   YOLO-Seg through it.
2. Add a new analysis over upstream YOLO-Seg MLIR that identifies semantic
   regions using topology, tensor contracts, and graph outputs:
   detection output, bbox decode, class scores, mask coefficients, prototype
   branch, feature-scale regions, and postprocess boundary.
3. Attach structured attributes such as `cv.semantic_role`, `cv.region_id`,
   `cv.output_role`, `cv.feature_scale`, and
   `cv.semantic.truth_boundary`.
4. Rewrite execution-domain planning to consume these attrs and generic
   upstream op classes rather than `cv.*` names.
5. Rewrite memory planning as generic tensor/memref lifetime and size analysis,
   using actual element type sizes instead of hardcoded fp16.
6. Update the CV plan builder/exporter to consume semantic attrs and region
   summaries.
7. Replace pseudo-CV tests with tests over upstream MLIR snippets and the real
   YOLO-Seg artifact when present.
8. Only consider rebuilding a small semantic CV dialect if attributes cannot
   support region ownership, outlining, or backend contract needs.

## Exact Next Implementation Phase

Recommended next coding phase:

```text
Phase 22: upstream-MLIR CV semantic annotation analysis for real YOLO-Seg.
```

Scope:

- Input: `artifacts/yoloseg_generic_frontend/yoloseg.generic.mlir` or a small
  upstream-MLIR fixture with equivalent topology.
- Do not introduce new `cv.*` operations.
- Do not delete the old CV dialect yet.
- Add an analysis pass that recognizes semantic roles from topology, shapes,
  producer/consumer relationships, and graph outputs.
- Attach structured attrs to existing upstream operations/function results.
- Add tests proving recognition without source-node-name dependence.
- Produce a report showing detected roles and truth boundary.

Out of scope:

- runtime execution
- ExecutionPlan integration changes
- SCF/LLVM lowering
- Qwen behavior changes
- deletion or migration of the legacy CV dialect

## Phase 22 Follow-Up

Phase 22 implements the audit recommendation without changing the legacy CV
dialect:

- real YOLO-Seg remains in upstream `func`/`tensor`/`linalg`/`arith`/`math`
  MLIR
- `cv-semantic-annotation` runs on `func.func` and upstream operations
- semantics are attached as attributes such as `cv.output_role`,
  `cv.semantic_role`, `cv.region_id`, `cv.recognition_confidence`, and
  `cv.recognition_evidence`
- the old `cv.*` numerical operations remain legacy fixture infrastructure and
  are not used by the real graph
- the generated report is
  `artifacts/yoloseg_generic_frontend/yoloseg.cv_semantic_report.json`

Recognized real-graph roles:

| Region | Role | Count | Confidence |
| --- | --- | ---: | --- |
| `cv.region.detection_head` | detection output/head | 21 report-visible ops; 22 collected ops in function summary | high |
| `cv.region.segmentation_prototype` | segmentation prototype | 10 | high |
| `cv.region.mask_coefficient_branch` | mask coefficient branch | 1 | high |
| `cv.region.feature_pyramid` | feature pyramid evidence | 4 | medium |

Source names were not required:

```text
cv.semantic_annotation.source_name_dependency = "none"
```

The audit recommendation remains `REPLACE_WITH_ATTRIBUTES`, now with an
initial implementation and real YOLO-Seg regression coverage.

## Phase 23 Follow-Up

Phase 23 adds a facts-only planning layer on top of the Phase 22 attributes:

- input: `yoloseg.cv_annotated.mlir`
- tool: `tools/cv_planning_facts.py`
- script: `scripts/run_yoloseg_cv_planning_facts.sh`
- output: `artifacts/yoloseg_generic_frontend/yoloseg.cv_planning_facts.json`

The analysis builds typed region, output, tensor, cost, lifetime,
candidate-domain, quantization-eligibility, and fusion-eligibility facts. It
still does not reuse legacy numerical `cv.*` operations and does not modify
`ExecutionPlan`.

Legacy reuse decision:

- `CVMemoryPlanningPass`: not reused because it depends on pseudo-CV ops and
  `cv.bytes_estimate`; only the generic linear-lifetime concept is carried
  forward over upstream tensor SSA values.
- `CVExecutionDomainPlanningPass`: not reused because it classifies hardcoded
  pseudo-CV op names.
- `CVExecutionPlanBuilder`: not reused because ExecutionPlan schema changes
  are out of scope.
- generic shape/dtype cost accounting concepts are reused, but formulas are
  implemented for upstream YOLO-Seg `tensor`/`linalg` operations.

The audit recommendation remains `REPLACE_WITH_ATTRIBUTES`; Phase 23 is the
first planner-facing consumer of that replacement path.
