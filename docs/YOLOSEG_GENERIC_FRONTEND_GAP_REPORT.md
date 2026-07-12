# YOLO-Seg Generic Frontend Gap Report

## Summary

`models/yolo-seg.onnx` was available locally and the generic frontend pipeline
completed successfully.

Phase 12A reduced YOLO-Seg `nn.unknown` coverage gaps to zero by adding
model-agnostic mappings and shape rules for:

- `ConvTranspose`
- `Div`
- `MaxPool`
- `Slice`
- `Split`
- `Sub`

Phase 12B added bounded, model-agnostic constant/initializer extraction for
shape-bearing operands. Phase 12C fixed generic ONNX Reshape semantics for
copied `0` dimensions and inferred `-1` dimensions. Current readiness is
`ready_for_generic_lowering`: generic op and static shape coverage are
complete for this model.

Expected model path:

```text
models/yolo-seg.onnx
```

Command:

```bash
scripts/run_yoloseg_generic_frontend.sh
```

Equivalent direct command:

```bash
.venv/bin/python tools/run_generic_onnx_frontend.py models/yolo-seg.onnx artifacts/yoloseg_generic_frontend --prefix yoloseg
```

Setup details: `docs/YOLOSEG_MODEL_SETUP.md`.

Project ONNX inventory: `docs/ONNX_MODEL_INVENTORY.md`.

This report is diagnostics-only. No YOLO/CV recognition, Transformer/LLM
recognition, MLIR lowering, ExecutionPlan generation, or Qwen GraphFacts
changes were added.

## Model Discovery

Searches performed:

- `*yolo*seg*.onnx`
- `*yoloseg*.onnx`
- `*yolo*.onnx`
- all `.onnx` files under the repo root to depth 4

Project ONNX files found:

- `models/bert_tiny.onnx`
- `models/tiny_mlp.onnx`
- `models/matmul_add_relu.onnx`
- `models/yolo-seg.onnx` locally present for this run

No `models/yolo-seg.onnx.data` sidecar was present; this run used the single
ONNX protobuf file.

See `docs/ONNX_MODEL_INVENTORY.md` for the current project ONNX inventory and
the dependency ONNX files that are intentionally ignored.

## Frontend Execution

Status: completed.

Artifacts emitted under `artifacts/yoloseg_generic_frontend/`:

- `yoloseg.imported_graph_ir.json`
- `yoloseg.generic_graph_ir.json`
- `yoloseg.canonical_generic_graph_ir.json`
- `yoloseg.shape_generic_graph_ir.json`
- `yoloseg.diagnostics_report.json`
- `yoloseg.frontend_report.json`

## Diagnostics

Source: `artifacts/yoloseg_generic_frontend/yoloseg.diagnostics_report.json`.

- `frontend_readiness_status`: `ready_for_generic_lowering`
- `node_count`: 268
- `unknown_op_count`: 0
- `unknown_source_op_types`: []
- `shape_inference_status_histogram`:
  - `inferred`: 268

Generic op histogram:

- `nn.conv2d`: 76
- `nn.sigmoid`: 67
- `nn.mul`: 67
- `nn.concat`: 18
- `nn.reshape`: 11
- `nn.split`: 8
- `nn.add`: 8
- `nn.maxpool2d`: 3
- `nn.resize`: 2
- `nn.slice`: 2
- `nn.sub`: 2
- `nn.conv_transpose2d`: 1
- `nn.div`: 1
- `nn.softmax`: 1
- `nn.transpose`: 1

Remaining shape errors:

- None.

Remaining unknown shape nodes:

- None.

Remaining partial shape nodes:

- None.

Missing generic op support:

- None for the current YOLO-Seg graph.

Missing shape inference support:

- None for the current static model.

## Resolved Final Concat Error

The failing node was:

- node id/name: `267`, `/model.22/Concat_4`
- source op: `Concat`
- canonical attrs: `axis=1`
- inputs:
  - `/model.22/Mul_2_output_0`, produced by node 266
    `/model.22/Mul_2` (`Mul`): `[1,4,8400]`
  - `/model.22/Sigmoid_output_0`, produced by node 253
    `/model.22/Sigmoid` (`Sigmoid`): previously `[1,80,-3]`
  - `/model.22/Concat_2_output_0`, produced by node 251
    `/model.22/Concat_2` (`Concat`, `axis=2`): previously `[1,32,-3]`

The concat axis and ranks were correct. Upstream `nn.reshape` nodes had
canonical targets such as `[1,80,-1]` and `[1,32,-1]`, but shape inference
incorrectly emitted `-1` as a static output dimension. Concatenating three
branches on axis 2 then summed those invalid dimensions to `-3`.

The generic fix resolves a single `-1` from the input element count, applies
ONNX `allowzero=0` copy semantics for `0`, validates invalid target shapes,
and leaves the inferred dimension unknown when static resolution is not
possible. The corrected producer shapes have trailing dimension `8400`; the
final concat output is `[1,116,8400]`.

## Existing-MLIR Lowering Contract

Phase 13 checks the shape-annotated graph without emitting MLIR:

```bash
.venv/bin/python tools/check_generic_lowering_contract.py \
  --in artifacts/yoloseg_generic_frontend/yoloseg.shape_generic_graph_ir.json \
  --out artifacts/yoloseg_generic_frontend/yoloseg.lowering_contract.json
```

Current result:

- `contract_status`: `ready_for_existing_mlir_lowering`
- unsupported strategies: none for the current static model
- missing required canonical attrs: none
- missing shapes: none
- missing dtypes: none

Phase 14A selected narrow existing-dialect strategies for the model's exact
semantics. Broader transposed-convolution and resize variants remain blocked.
See `docs/CONV_TRANSPOSE_RESIZE_LOWERING_DECISION.md`.

## GenericGraphIR-to-MLIR Emitter Coverage

Phase 20 emitter coverage over
`artifacts/yoloseg_generic_frontend/yoloseg.shape_generic_graph_ir.json`:

- emitter-supported nodes before Phase 20: 267 / 268
- emitter-supported nodes after Phase 20: 268 / 268
- newly supported by Phase 20:
  - `nn.conv_transpose2d`: 1

Supported emitter nodes by op:

- `nn.conv2d`: 76
- `nn.mul`: 67
- `nn.sigmoid`: 67
- `nn.concat`: 18
- `nn.reshape`: 11
- `nn.add`: 8
- `nn.split`: 8
- `nn.maxpool2d`: 3
- `nn.resize`: 2
- `nn.slice`: 2
- `nn.sub`: 2
- `nn.conv_transpose2d`: 1
- `nn.div`: 1
- `nn.softmax`: 1
- `nn.transpose`: 1

Remaining emitter-unsupported nodes by op:

- None for the current static YOLO-Seg graph.

No YOLO-Seg semantic forms remain unsupported by the Phase 20 emitter
predicates.

### Phase 20 ConvTranspose Inspection

The one YOLO-Seg `nn.conv_transpose2d` node is:

- node id/name: `150`, `/model.22/proto/upsample/ConvTranspose`
- source ONNX name: `/model.22/proto/upsample/ConvTranspose`
- input shape: `[1,64,80,80]`
- weight shape/layout: `[64,64,2,2]`, ONNX ConvTranspose
  `[input_channels, output_channels, kernel_h, kernel_w]`
- bias shape: `[64]`
- output shape: `[1,64,160,160]`
- groups: `1`
- kernel_shape: `[2,2]`
- strides: `[2,2]`
- dilations: `[1,1]`
- pads: `[0,0,0,0]`
- output_padding: `[0,0]`
- dtype: `float` / MLIR `f32`

The emitter lowers this exact non-overlapping subset with bias initialization
and a specialized `linalg.generic` using `oh floordiv 2`, `ow floordiv 2`,
`oh mod 2`, and `ow mod 2` index expressions. Broader grouped, padded,
overlapping, dynamic, or non-`f32` ConvTranspose forms remain rejected.

### Full MLIR Emission

`scripts/run_yoloseg_generic_mlir_emission.sh` runs:

```text
models/yolo-seg.onnx
  -> generic frontend
  -> lowering-contract check
  -> generic_graph_ir_to_mlir.py
  -> yoloseg.generic.mlir
  -> mlir-opt verification
```

Current report:

- total nodes: 268
- emitted nodes: 268
- unsupported nodes: 0
- MLIR verification status: `verified_with_mlir_opt`
- emitted dialects: `func`, `tensor`, `linalg`, `arith`, `math`
- truth boundary:
  `full_graph_mlir_emission_verified_no_backend_codegen_no_runtime_execution_no_execution_plan_generation`

Artifacts:

- `artifacts/yoloseg_generic_frontend/yoloseg.generic.mlir`
- `artifacts/yoloseg_generic_frontend/yoloseg.generic.verified.mlir`
- `artifacts/yoloseg_generic_frontend/yoloseg.generic_mlir_emission_report.json`

### Phase 19 Conv2d Inspection

The 76 YOLO-Seg `nn.conv2d` nodes group into 51 distinct static semantic
forms by input shape, weight shape, bias presence, output shape, groups,
kernel, stride, pads, dilation, dtype, and source names.

All 76 are standard convolution forms:

- layout: input/output NCHW, weight FCHW/OIHW
- dtype: `float` / MLIR `f32`
- groups: `1` for all nodes
- dilations: `[1,1]` for all nodes
- bias: 75 nodes have rank-1 bias; 1 node has no bias
- kernels: 47 nodes use `[3,3]`; 29 nodes use `[1,1]`
- strides: 69 nodes use `[1,1]`; 7 nodes use `[2,2]`
- pads: 47 nodes use `[1,1,1,1]`; 29 nodes use `[0,0,0,0]`

Representative grouped forms:

| Count | Input | Weight | Bias | Output | Groups | Kernel | Strides | Pads | Dilations | Source ONNX names |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 1 | `[1,3,640,640]` | `[16,3,3,3]` | `[16]` | `[1,16,320,320]` | 1 | `[3,3]` | `[2,2]` | `[1,1,1,1]` | `[1,1]` | `/model.0/conv/Conv` |
| 7 | `[1,32,80,80]` | `[32,32,3,3]` | `[32]` | `[1,32,80,80]` | 1 | `[3,3]` | `[1,1]` | `[1,1,1,1]` | `[1,1]` | `/model.4/m.0/cv1/conv/Conv`, `/model.4/m.0/cv2/conv/Conv`, `/model.4/m.1/cv1/conv/Conv`, `/model.4/m.1/cv2/conv/Conv`, `/model.15/m.0/cv1/conv/Conv`, `/model.15/m.0/cv2/conv/Conv`, `/model.22/cv4.0/cv4.0.1/conv/Conv` |
| 9 | `[1,64,40,40]` | `[64,64,3,3]` | `[64]` | `[1,64,40,40]` | 1 | `[3,3]` | `[1,1]` | `[1,1,1,1]` | `[1,1]` | `/model.6/m.0/cv1/conv/Conv`, `/model.6/m.0/cv2/conv/Conv`, `/model.6/m.1/cv1/conv/Conv`, `/model.6/m.1/cv2/conv/Conv`, `/model.12/m.0/cv1/conv/Conv`, `/model.12/m.0/cv2/conv/Conv`, `/model.18/m.0/cv1/conv/Conv`, `/model.18/m.0/cv2/conv/Conv`, `/model.22/cv2.1/cv2.1.1/conv/Conv` |
| 3 | `[1,384,20,20]` | `[256,384,1,1]` | `[256]` | `[1,256,20,20]` | 1 | `[1,1]` | `[1,1]` | `[0,0,0,0]` | `[1,1]` | `/model.8/cv2/conv/Conv`, `/model.21/cv1/conv/Conv`, `/model.21/cv2/conv/Conv` |
| 1 | `[1,16,4,8400]` | `[1,16,1,1]` | none | `[1,1,4,8400]` | 1 | `[1,1]` | `[1,1]` | `[0,0,0,0]` | `[1,1]` | `/model.22/dfl/conv/Conv` |

The emitter now supports all 76 nodes with `tensor.pad` for non-zero ONNX
spatial padding, zero `linalg.fill`, `linalg.conv_2d_nchw_fchw`, and optional
rank-1 bias broadcast with `linalg.generic`.

### Phase 18 Reshape Inspection

Before Phase 18, all 11 YOLO-Seg reshape nodes were rejected by the same
emitter predicate: `nn.reshape requires one input and one output`. They use
the ONNX two-input form where the second input is a static shape tensor.

| Node | Source ONNX name | Input shape | Output shape | Canonical target_shape | allowzero | Phase 18 form |
|---:|---|---|---|---|---:|---|
| 167 | `/model.22/Reshape` | `[1,64,80,80]` | `[1,64,6400]` | `[1,64,-1]` | 0 | `tensor.collapse_shape [[0], [1], [2, 3]]` |
| 168 | `/model.22/Reshape_3` | `[1,80,80,80]` | `[1,80,6400]` | `[1,80,-1]` | 0 | `tensor.collapse_shape [[0], [1], [2, 3]]` |
| 169 | `/model.22/Reshape_6` | `[1,32,80,80]` | `[1,32,6400]` | `[1,32,-1]` | 0 | `tensor.collapse_shape [[0], [1], [2, 3]]` |
| 212 | `/model.22/Reshape_1` | `[1,64,40,40]` | `[1,64,1600]` | `[1,64,-1]` | 0 | `tensor.collapse_shape [[0], [1], [2, 3]]` |
| 213 | `/model.22/Reshape_4` | `[1,80,40,40]` | `[1,80,1600]` | `[1,80,-1]` | 0 | `tensor.collapse_shape [[0], [1], [2, 3]]` |
| 214 | `/model.22/Reshape_7` | `[1,32,40,40]` | `[1,32,1600]` | `[1,32,-1]` | 0 | `tensor.collapse_shape [[0], [1], [2, 3]]` |
| 246 | `/model.22/Reshape_2` | `[1,64,20,20]` | `[1,64,400]` | `[1,64,-1]` | 0 | `tensor.collapse_shape [[0], [1], [2, 3]]` |
| 247 | `/model.22/Reshape_5` | `[1,80,20,20]` | `[1,80,400]` | `[1,80,-1]` | 0 | `tensor.collapse_shape [[0], [1], [2, 3]]` |
| 248 | `/model.22/Reshape_8` | `[1,32,20,20]` | `[1,32,400]` | `[1,32,-1]` | 0 | `tensor.collapse_shape [[0], [1], [2, 3]]` |
| 252 | `/model.22/dfl/Reshape` | `[1,64,8400]` | `[1,4,16,8400]` | `[1,4,16,8400]` | 0 | `tensor.expand_shape [[0], [1, 2], [3]]` |
| 257 | `/model.22/dfl/Reshape_1` | `[1,1,4,8400]` | `[1,4,8400]` | `[1,4,8400]` | 0 | `tensor.collapse_shape [[0], [1, 2], [3]]` |

### Phase 18 Pooling And Softmax Forms

The three maxpool nodes are `/model.9/m/MaxPool`,
`/model.9/m_1/MaxPool`, and `/model.9/m_2/MaxPool`. Each is static `f32`
NCHW with input/output `[1,128,20,20]`, `kernel_shape=[5,5]`,
`pads=[2,2,2,2]`, `strides=[1,1]`, `dilations=[1,1]`, and `ceil_mode=0`.
The emitter pads with negative infinity, initializes the output with negative
infinity, and emits `linalg.pooling_nchw_max`.

The softmax node is `/model.22/dfl/Softmax`, static `f32` shape
`[1,16,4,8400]`, `axis=1`. The emitter lowers it to max reduction,
subtract-max, `math.exp`, sum reduction, and `arith.divf`.

## Phase 21 Post-Emission Bufferization Boundary

The full YOLO-Seg existing-dialect MLIR now also validates through the first
post-emission structural lowering boundary:

```text
yoloseg.generic.mlir
  -> one-shot-bufferize{bufferize-function-boundaries}
  -> buffer-deallocation-pipeline
  -> yoloseg.bufferized.mlir
  -> mlir-opt verification
```

Script:

```bash
scripts/lower_yoloseg_mlir_to_bufferized.sh
```

Artifacts:

- `artifacts/yoloseg_generic_frontend/yoloseg.bufferized.mlir`
- `artifacts/yoloseg_generic_frontend/yoloseg.bufferization_report.json`

Result:

- full graph bufferization: succeeded
- remaining `tensor.*` ops: 0
- remaining `linalg.*` ops: 723
- introduced `memref.*` ops: 989
- `memref.alloc`: 378
- `memref.dealloc`: 376
- `memref.copy`: 102
- function ABI after bufferization: 158 memref arguments and two memref
  returns, `(memref<1x116x8400xf32>, memref<1x32x160x160xf32>)`

All tensor data-movement forms used by the emitter bufferize: `tensor.empty`,
`tensor.pad`, `tensor.generate`, `tensor.extract`, `tensor.extract_slice`,
`tensor.insert_slice`, `tensor.collapse_shape`, and `tensor.expand_shape`.

The truth boundary is:

```text
full_graph_bufferization_verified_no_machine_codegen_no_runtime_execution_no_numerical_equivalence_validation_no_execution_plan_generation
```

Successful bufferization is structural validation only. It is not numerical
correctness, backend code generation, runtime execution, or ExecutionPlan
integration. See `docs/YOLOSEG_MLIR_NEXT_LOWERING_BOUNDARY.md`.

## Phase 22 CV Semantic Annotation

The real full-graph YOLO-Seg MLIR now has an attribute-only CV semantic
analysis pass over upstream MLIR:

```text
yoloseg.generic.mlir
  -> cv-semantic-annotation
  -> yoloseg.cv_annotated.mlir
  -> mlir-opt verification
  -> yoloseg.cv_semantic_report.json
```

Script:

```bash
scripts/run_yoloseg_cv_semantic_annotation.sh
```

Artifacts:

- `artifacts/yoloseg_generic_frontend/yoloseg.cv_annotated.mlir`
- `artifacts/yoloseg_generic_frontend/yoloseg.cv_semantic_report.json`

The pass does not use the legacy CV dialect and does not emit custom `cv.*`
operations. It attaches structured attributes to selected upstream operations
and the function.

Recognized roles:

- detection output: `tensor<1x116x8400xf32>`, high confidence
- segmentation prototype output: `tensor<1x32x160x160xf32>`, high confidence
- detection head region: 21 report-visible annotated ops; function summary
  records 22 collected ops
- segmentation prototype region: 10 annotated ops
- mask coefficient branch: 1 annotated op
- feature pyramid evidence: 4 annotated ops, medium confidence

Source names were not required:

```text
cv.semantic_annotation.source_name_dependency = "none"
```

The truth boundary is:

```text
cv_semantic_annotation_only_no_backend_selection_no_memory_plan_no_kernel_selection_no_execution_plan_generation
```

See `docs/REAL_YOLOSEG_CV_SEMANTIC_ANNOTATION.md`.

## Phase 23 CV Planning Facts

The annotated upstream MLIR now feeds a facts-only planning analysis:

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

Current real YOLO-Seg summary:

- tensor facts: 1004
- total initializer bytes: 13,785,524
- peak live temporary bytes: 31,948,800
- unresolved facts: 0
- estimated total FLOPs: 11,932,092,000

Estimated FLOPs by semantic region:

- detection head: 4,620,000
- segmentation prototype: 2,009,497,600
- feature pyramid: 0
- mask coefficient branch: 0

The truth boundary is:

```text
cv_planning_facts_only_no_backend_selection_no_kernel_selection_no_memory_slot_assignment_no_execution_plan_generation_no_measured_performance
```

See `docs/REAL_YOLOSEG_CV_PLANNING_FACTS.md`.

## Phase 24 ExecutionPlan

The real upstream YOLO-Seg MLIR now reaches the canonical `ExecutionPlan`
export path without a separate planning-facts IR boundary:

```text
yoloseg.cv_annotated.mlir
  -> cv-execution-plan-attrs
  -> existing generic kernel/lowering/selection attrs
  -> ExecutionPlanBuilder
  -> yoloseg.execution_plan.json
```

Script:

```bash
scripts/run_yoloseg_execution_plan.sh
```

Artifacts:

- `artifacts/yoloseg_generic_frontend/yoloseg.execution_plan.json`
- `artifacts/yoloseg_generic_frontend/yoloseg.execution_plan_annotated.mlir`

Current real YOLO-Seg summary with the default Apple A17 Pro profile:

- function plans: 1
- selected backend: `coreml`
- fallback backends: `metal`, `cpu`
- output roles:
  - detection: `[1,116,8400] f32`
  - segmentation prototype: `[1,32,160,160] f32`
- semantic regions: detection head, segmentation prototype, mask coefficient
  branch, feature pyramid

Truth boundary:

```text
real_yoloseg_execution_plan_compiler_decisions_from_static_capability_and_analysis_no_runtime_execution_no_measured_performance_no_full_memory_slot_allocation
```

See `docs/REAL_YOLOSEG_EXECUTION_PLAN.md`.

## Gap Analysis Procedure

Once `models/yolo-seg.onnx` is added, run:

```bash
scripts/run_yoloseg_generic_frontend.sh
```

Then inspect:

- `artifacts/yoloseg_generic_frontend/yoloseg.frontend_report.json`
- `artifacts/yoloseg_generic_frontend/yoloseg.diagnostics_report.json`
- `artifacts/yoloseg_generic_frontend/yoloseg.shape_generic_graph_ir.json`

Use `yoloseg.diagnostics_report.json` as the primary source for:

- generic op coverage gaps via `unknown_source_op_types`
- shape/type gaps via `shape_inference_status_histogram`
- failing nodes via `shape_error_nodes`
- incomplete nodes via `shape_unknown_nodes` and
  `shape_partially_inferred_nodes`
- model size pressure via `top_initializers_by_raw_data_bytes`
- overall readiness via `frontend_readiness_status`

## Phase 26 Source Provenance And Execution ABI

The emitter now carries a source-provenance contract
(`generic_emitter_source_attrs_v1`): every emitted top-level op receives
`source.graph_node_id`, `source.imported_node_id`, `source.op_type`,
`source.generic_op`, `source.onnx_name`, `source.dispatch_group`, and
`source.op_role` as real MLIR attributes (the older `// source_node_id=...`
comments remain for readability but are no longer the provenance carrier —
attributes survive `mlir-opt` round-trips and downstream passes). Function
arguments carry `source.name` / `source.arg_role`
(`model_input`/`weight`/`bias`/`initializer`) / `source.arg_index`; the
module carries `source.model_artifact`. This closes the Phase 25 finding
that provenance died at the first MLIR parse, and it supplies the execution
ABI ingredients (item 2 below) that the ExecutionPlan `tensor_bindings` now
expose. See `docs/YOLOSEG_DISPATCH_UNIT_MATERIALIZATION.md`.

## Recommended Next Implementation Phases

1. Define Phase 22 around the next boundary after verified memref/linalg:
   either stop at memref/linalg as the compiler artifact or lower to SCF loops
   for a CPU baseline.
2. Specify the execution ABI before runtime work: input buffers, initializer
   buffers, returned result buffers, ownership, and deallocation.
   (Addressed at the plan level by Phase 26 `tensor_bindings`; buffer-level
   deallocation contracts remain future runtime work.)
3. Add numerical/runtime validation only after a backend execution boundary is
   explicitly selected.
4. Keep grouped/depthwise convolution support separate unless a verified
   upstream named-op mapping is selected for a real model form.
