# YOLO-Seg Dispatch-Unit Materialization (Phase 26)

Phase 26 converts the YOLO-Seg ExecutionPlan's per-MLIR-op decision dump into a
provenance-preserving **dispatch-unit plan** with a complete tensor ABI,
corrected memory metrics, and a runtime-facing CV postprocess contract.

Truth boundary of everything below:
`real_yoloseg_dispatch_units_materialized_no_runtime_execution_no_registered_cv_kernels_no_measured_performance_no_memory_slot_assignment`

No backend execution was implemented. No kernel execution is claimed. The LLM
(Qwen) plan path is untouched and byte-identical.

## 1. Phase 25 Findings This Phase Resolves

The Phase 25 audit (`docs/YOLOSEG_EXECUTION_PLAN_RUNTIME_READINESS_AUDIT.md`)
found, for the 268-node GenericGraphIR YOLO-Seg graph:

- 929 per-op decisions over lowered MLIR: 306 compute ops, 133 view/pad ops,
  407 `tensor.empty`/`linalg.fill` placeholders, 83 scalar constants — all
  serialized as top-level runtime decisions;
- provenance existed only as text comments, destroyed at the first MLIR parse;
- `temporary_bytes` (924,245,084) was cumulative SSA write volume, ~29× the
  Phase 23 peak-live estimate (31,948,800);
- all 158 function inputs were labeled `graph_input` — image
  indistinguishable from 157 weights; no model-file reference;
- `cv_extension` carried no postprocess contract;
- the runtime loader produced 930/930 `UNSUPPORTED` stages from the plan.

Verdict: `NEEDS_DISPATCH_UNIT_MATERIALIZATION` — implemented here.

## 2. Why 929 MLIR Ops Are Not 929 Runtime Operations

The emitter lowers one source node into several upstream-dialect ops (one
`nn.conv2d` → scalar constant + `tensor.pad` + `tensor.empty` + `linalg.fill`
+ `linalg.conv_2d_nchw_fchw` + `tensor.empty` + bias `linalg.generic`). Those
helpers are implementation detail of one logical operation: they must never be
independent runtime dispatches, and a runtime consuming them 1:1 gets 53%
allocation placeholders and constants. The runtime granule is the source
operation (or a materialized fusion of several), with helpers folded inside.

## 3. Provenance Design (`generic_emitter_source_attrs_v1`)

`tools/generic_graph_ir_to_mlir.py` now attaches **discardable MLIR
attributes** — not comments, not op order — to every top-level op it emits:

| Attr | Meaning |
|---|---|
| `source.graph_node_id` (i64) | GenericGraphIR node id |
| `source.imported_node_id` (i64) | ImportedGraphIR node id |
| `source.op_type` | ONNX op type (`"Conv"`) |
| `source.generic_op` | GenericGraphIR op (`"nn.conv2d"`) |
| `source.onnx_name` | ONNX node name (`"/model.0/conv/Conv"`) |
| `source.dispatch_group` | stable grouping id (`"dg_42"`) |
| `source.op_role` | classification (see §5) |

Function arguments carry `source.name`, `source.arg_role`
(`model_input` / `weight` / `bias` / `initializer`), and `source.arg_index`.
The module carries `source.model_artifact` (e.g. `models/yolo-seg.onnx`) and
`source.provenance_contract`. Attributes survive `mlir-opt` round-trips and
the full annotation/planning pipeline (verified: all 929 ops and 158 args
retain them in `yoloseg.execution_plan_annotated.mlir`). Semantic-region
identity (`cv.region_id`) is added later by `cv-semantic-annotation` and read
per unit. Helper ops emitted while lowering one node carry the same
`source.dispatch_group`, exactly as required.

## 4. DispatchUnit Schema

`ExecutionPlan.function_plans[].dispatch_units[]` (CV full-graph functions
only; typed struct in `mlir_passes/include/serving/ExecutionPlan.h`):

```
dispatch_unit_id            "du_42"
source_graph_node_ids       [42]           # >1 only if a fusion is materialized
source_imported_node_ids    [42]
source_onnx_node_names      ["/model.4/cv1/conv/Conv"]
source_op_type              "Conv"
operation_family            "nn.conv2d"
semantic_region_id          "cv.region.detection_head" | absent
mlir_operation_refs         ["op_12:linalg.conv_2d_nchw_fchw", ...]  # diagnostics
input_tensor_ids            ["arg_0" | "du_41:o0", ...]
output_tensor_ids           ["du_42:o0", ...]   # multi-output for splits
initializer_tensor_ids      ["arg_1", "arg_2"]
backend_intent              {backend, intent_basis}
execution_domain            "unassigned"
kernel_status               see §8
selected_kernel_id          "" unless runtime_registered
fallback_backends           ["metal"]
dtype / layout              from the root op result
estimated_compute_flops     Σ member shape_cost flops (0 = no estimate)
estimated_read_bytes        Σ unit input tensor bytes
estimated_write_bytes       Σ unit output tensor bytes
workspace_bytes             0 (no kernel workspace contracts exist)
decision_provenance         source attrs contract + producing passes
executable                  false unless a runtime kernel is registered
non_executable_reason       "no_runtime_adapter_or_registered_kernel"
```

Generated YOLO-Seg result: **268 dispatch units for 268 GenericGraphIR
nodes** (no fusion is materialized today, so units are 1:1), including:
one Conv unit per conv (4–7 MLIR ops folded), one unit per Sigmoid, one unit
per Concat (empty + insert_slice chain), one **multi-output** unit per Split
(2 outputs each), and one unit for the 12-op Softmax lowering. The
inter-unit tensor graph is closed: every unit input resolves to a function
argument or another unit's output.

## 5. Helper-Op Classification

Every top-level MLIR op receives exactly one `source.op_role`; the builder
reconciles totals in `op_classification` and the dispatch-unit report:

| Classification | YOLO-Seg count | Dispatch? |
|---|---|---|
| `dispatch_root` (produces a node output) | 276 | root of its unit (268 units; splits have 2 roots) |
| `dispatch_internal_compute` | 113 | folded into unit |
| `tensor_contract_operation` (`tensor.pad`) | 50 | folded into unit input contract |
| `allocation_helper` (`tensor.empty`, `linalg.fill`) | 407 | never |
| `scalar_helper` (`arith.constant`) | 83 | never |
| `view_operation` | 0 (root views count as roots) | — |
| `non_dispatch_metadata` (materialized `hir.cast`) | 0 | never |
| `unresolved` | **0** | — |

Total 929 = all classified = all assigned to units.
`per_op_decisions` is now **empty for CV full-graph functions** — internal
MLIR ops no longer appear as top-level runtime decisions (they remain
recoverable from `mlir_operation_refs` and the annotated MLIR dump). The LLM
path keeps its per-op decision list unchanged.

## 6. Tensor ABI (`tensor_bindings`)

Top-level `tensor_bindings` (160 entries for YOLO-Seg):

| Role | Count | Notes |
|---|---|---|
| `model_input` | 1 | `arg_0`, original name `images`, ownership `caller`, mutable |
| `weight` | 77 | conv/conv-transpose weights, ownership `model_state`, immutable |
| `bias` | 76 | conv/upsample biases, ownership `model_state`, immutable |
| `initializer` | 4 | anchor-grid constants baked as args |
| `model_output` | 2 | `result_0` (detection), `result_1` (prototype), ownership `caller` |

Each binding carries `original_name` (ONNX initializer name),
`argument_index`, shape/dtype/layout/byte_size, ownership, mutability, and
`model_artifact_reference: "models/yolo-seg.onnx"` (also in
`provenance.model_spec_ref`). Weights are **referenced, not embedded** — the
13.8 MB of weight data stays in the model file.

## 7. Corrected Memory Metrics (`cv_extension.memory_summary`)

| Field | YOLO-Seg value | Definition |
|---|---|---|
| `model_input_bytes` | 4,915,200 | image argument only |
| `initializer_bytes` | 13,785,284 | weight/bias/constant arguments |
| `model_output_bytes` | 7,174,400 | return operand types |
| `total_intermediate_tensor_bytes` | 917,070,684 | cumulative SSA results excluding outputs |
| `total_intermediate_write_bytes` | 924,245,084 | cumulative SSA results including outputs (legacy `estimated_temporary_bytes` value) |
| `peak_live_temporary_bytes` | **31,948,800** | static lifetime scan (compiler-side reimplementation of the Phase 23 algorithm in `CVExecutionPlanAttrsPass`; matches the Phase 23 artifact exactly) |
| `workspace_bytes` | 0 | no kernel workspace contracts |
| `planned_slot_bytes` | null | no slot allocator exists; none claimed |

The legacy `memory_estimates.estimated_temporary_bytes` is preserved for
schema compatibility and now explicitly labeled
`cumulative_ssa_result_write_volume_not_peak_live_deprecated`.

## 8. Backend / Kernel Truth Model

Per unit, two separate vocabularies:

- **backend_intent.intent_basis**: `configured_preference` |
  `capability_validated` | `analytically_selected` | `measured_selected` |
  `unavailable`. YOLO-Seg today: `configured_preference("coreml")` on every
  unit — the selection comes from the target profile's
  `configuredComputeUnits` policy and was never capability-matched.
- **kernel_status**: `runtime_registered` | `library_available` |
  `lowering_only` | `deferred` | `fallback_only` | `unavailable`, aggregated
  weakest-wins over members that carry a lowering decision or a concrete
  kernel selection (helper constants do not participate). YOLO-Seg today:
  `fallback_only` on all 268 units (fallback backend `metal`).

`executable` is true **only** for `runtime_registered`. Current YOLO-Seg:
**0 executable, 268 non-executable**, reason
`no_runtime_adapter_or_registered_kernel` — exactly the honest expected
output; no Core ML dispatch is claimed.

## 9. CV Postprocess Contract (`cv_extension.postprocess_contract`)

Traced from static `tensor.insert_slice` offsets and `cv.region_id` attrs
(provenance:
`traced_static_insert_slice_offsets_and_cv_region_attrs_no_nms_or_mask_ops_in_compiled_graph`):

- detection `result_0` `[1,116,8400]`; prototype `result_1` `[1,32,160,160]`;
- channel groups: box_regression `[0,4)` (detection_head region),
  class_scores `[4,84)` (sigmoid source), mask_coefficients `[84,116)`
  (mask_coefficient_branch region) — mask range **proven**, confidence high;
- `nms_required: "true"` and `mask_decode_required: true` — the compiled
  graph contains no NMS or mask-decode ops (op set is linalg/tensor/arith
  only), so both are runtime obligations; **no thresholds or algorithms are
  invented**;
- `implementation_status: "runtime_required"` — nothing implements the
  postprocess anywhere today.

## 10. Runtime Parser Requirements (read-only preview; NOT implemented)

`heterogeneous-inference-runtime` (`deployment/execution_plan/`) was
inspected read-only. Verified: the new plan **loads through the existing
loader unchanged** (schema 2.0.0, no measured fields), and instead of 930
noise stages the stage builder now produces 1 function stage; `dispatch_units`
ride along in `FunctionPlan.raw`. Required future runtime additions:

1. `cv_extension` parsing (currently absent from the runtime schema, silently
   dropped) — including `memory_summary` and `postprocess_contract`;
2. `dispatch_units` parsing as the CV stage source (typed, not `raw`);
3. `tensor_bindings` parsing for input/weight/output binding;
4. backend-intent vocabulary: accept `configured_preference` etc., and align
   backend identifiers (`coreml`, `metal`) with `ExecutionUnitRouter`'s
   vocabulary (today only `coreml_ane`/`cuda_*`/`cpu` are routable);
5. kernel-status vocabulary (`fallback_only`, `runtime_registered`, …);
6. postprocess contract consumption (NMS + mask decode obligations).

No runtime execution was implemented in this phase.

## 11. Artifacts and Validation

- `artifacts/yoloseg_generic_frontend/yoloseg.execution_plan.json` —
  regenerated with dispatch units, bindings, memory summary, postprocess
  contract.
- `artifacts/yoloseg_generic_frontend/yoloseg.dispatch_unit_report.json` —
  new reconciliation report (op counts, classification, binding roles,
  executable counts, memory reconciliation) via
  `compile-for-target --dispatch-unit-report`.
- Tests: `mlir_passes/test/serving/DispatchUnitBuilderTest.cpp` (CTest;
  grouping, classification totality, ABI, memory, truth model) and
  `tests/test_yoloseg_dispatch_units.py` (full-graph regression incl.
  peak-live equality with Phase 23 and Qwen schema non-regression);
  `tests/test_generic_graph_ir_to_mlir.py` extended CHECK patterns for the
  provenance attr dicts.
- Qwen compatibility: plans generated before/after Phase 26 code are
  **byte-identical** (verified by direct diff of regenerated artifacts; the
  committed `artifacts/qwen/execution_plan.json` had pre-existing drift
  relative to HEAD code that is unrelated to this phase).

## 12. Current Executable vs Non-Executable Status

All 268 dispatch units are **non-executable**
(`no_runtime_adapter_or_registered_kernel`): the kernel registry contains no
CV kernels, no Core ML/Metal adapter exists, and this phase deliberately
implements none. The plan is now a complete, provenance-preserving,
runtime-consumable *contract* — execution is Phase 27+ work.
