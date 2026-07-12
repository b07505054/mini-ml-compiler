# YOLO-Seg ExecutionPlan Runtime-Readiness Audit (Phase 25)

> **Phase 26 status update:** the verdict below
> (`NEEDS_DISPATCH_UNIT_MATERIALIZATION`) has been implemented — see
> `docs/YOLOSEG_DISPATCH_UNIT_MATERIALIZATION.md`. The regenerated plan now
> carries 268 provenance-preserving `dispatch_units` (per-op decisions
> suppressed for the CV function), a typed `tensor_bindings` ABI
> (1 model_input / 77 weight / 76 bias / 4 initializer / 2 model_output),
> a corrected `memory_summary` whose `peak_live_temporary_bytes`
> (31,948,800) matches the Phase 23 lifetime analysis, a serialized CV
> postprocess contract, and `provenance.model_spec_ref =
> "models/yolo-seg.onnx"`. §2, §3, §6, §7, and §8 below describe the
> **pre-Phase-26** plan and are preserved as the audit record; §4/§5's
> backend/kernel findings still hold (0 executable units,
> `configured_preference` only). Runtime-side parsing gaps in §1/§9 remain
> open.

Audit of `artifacts/yoloseg_generic_frontend/yoloseg.execution_plan.json` (schema
`execution_plan` 2.0.0, plan_id `apple-a17pro-mobile_serving_plan`) for consumption
by a future runtime adapter. No code was modified for this audit.

Evidence sources: the generated plan JSON, `yoloseg.generic.mlir`,
`yoloseg.cv_annotated.mlir`, `yoloseg.execution_plan_annotated.mlir`,
`yoloseg.cv_planning_facts.json` (Phase 23), the A17 Pro target profile,
`ExecutionPlanBuilder.cpp` / `ExecutionPlanExporter.cpp` / `CVExecutionPlanAttrsPass.cpp` /
`LoweringDecisionPlanningPass.cpp` / `compile-for-target/main.cpp`, and the
runtime-side loader/stage builder in
`heterogeneous-inference-runtime/deployment/execution_plan/`.

**Verdict: `NEEDS_DISPATCH_UNIT_MATERIALIZATION`** (details in §9).

---

## 1. Plan Abstraction Level

The plan is a **per-MLIR-operation decision log over the fully lowered upstream
module**, not a dispatchable execution contract. Its per-op layer records what each
planning pass concluded about each top-level op in `main_graph`'s entry block —
including allocation placeholders (`tensor.empty`), init fills, and scalar
constants. Its function/CV layers are honest static policy and I/O summaries.

That abstraction is correct for *explaining compiler decisions* (the repo's stated
compiler deliverable) but is below the level a runtime consumes: the runtime-side
`stage_builder.py` today maps each `per_op_decision` 1:1 to an `ExecutionStage`,
which for this plan would produce 929 "stages" of which zero are backed by a kernel
and ~490 are not executable operations at all.

## 2. Operation-Granularity Problem: Why 929 Decisions vs 268 Nodes

`main_graph` contains 930 top-level ops; the builder walks
`entry.without_terminator()` (929 ops) and `KernelSelectionPass` stamps
`kernel_selection.status` on **every** op, so every op forms a decision bundle
(`ExecutionPlanBuilder.cpp:392-620`). The 268 GenericGraphIR nodes were expanded by
the MLIR emitter — e.g. one `nn.conv2d` becomes `arith.constant` (pad zero) +
`tensor.pad` + `tensor.empty` + `linalg.fill` + `linalg.conv_2d_nchw_fchw` +
`tensor.empty` + bias `linalg.generic`; one `nn.concat` becomes a chain of
`tensor.insert_slice`.

Classification of the 929 decisions:

| Class | Ops | Count | Runtime dispatch? |
|---|---|---|---|
| Executable compute | `linalg.generic` 226, `linalg.conv_2d_nchw_fchw` 76, `linalg.pooling_nchw_max` 3, `linalg.transpose` 1 | **306** | Yes — these are the real dispatch candidates |
| View / materialization / data movement | `tensor.insert_slice` 52, `tensor.pad` 50, `tensor.extract_slice` 18, `tensor.collapse_shape` 10, `tensor.generate` 2, `tensor.expand_shape` 1 | **133** | Only as copies/pack ops folded into a dispatch unit's tensor contract |
| Allocation / init placeholders | `tensor.empty` 326, `linalg.fill` 81 | **407** | Never — memory-planning artifacts; `fill` belongs in the consumer kernel's prologue |
| Scalar constants | `arith.constant` 83 (all scalar f32; bundles carry only a rejected `kernel_selection`) | **83** | Never |
| Arithmetic scalar/body ops | `arith.*`/`math.*` **inside** `linalg` bodies | **0 decisions** | Correctly not enumerated — the builder only walks top-level block ops |
| Semantic region boundary | not a decision class — `cv.region_id` is an attr on 36 of the 929 ops (21+4+1+10) | — | Regions must be groupings over units, not decisions |

306 + 133 + 407 + 83 = 929. ✓

**What runtime should consume: a mixture, and none of the current options
directly.** The right dispatch granularity is the **268 source-level operations
(or fused groups of them), materialized as explicit dispatch units** that each
anchor one compute op (or a fused conv+bias+activation group), absorb their
`empty`/`fill`/`pad`/`constant` helpers as intra-unit detail, and expose view ops
only through input/output tensor contracts. Semantic regions are the correct
*grouping* layer above units (postprocess boundary, region-level
domain/quantization choices), not the dispatch layer itself: today's four regions
cover only 36 ops near the outputs — the backbone belongs to no region, so regions
alone cannot partition the graph.

Decisions that must **not** independently become runtime dispatches: all 407
`tensor.empty`/`linalg.fill`, all 83 `arith.constant`, and (as standalone units)
most of the 133 view ops.

## 3. Provenance and Grouping

Per-op decisions carry exactly two identity fields: `op_name` (`"op_N"`, a
**positional index** into the entry block, assigned at collection time) and
`op_type`. Mapping to anything else:

| Target | Recoverable from the plan? |
|---|---|
| Emitted MLIR operation | Only implicitly: re-walk `yoloseg.execution_plan_annotated.mlir` in the same order, skipping ops with `materialized.by`. Fragile — any pass that inserts/erases ops renumbers everything. |
| GenericGraphIR node ID | **No.** `// source_node_id=N source_op_type=... source_name=...` exists only as *text comments* in `yoloseg.generic.mlir` (268 occurrences). Comments are not IR: they are already gone in `yoloseg.generic.verified.mlir`, `cv_annotated.mlir`, and the plan-annotated module (0 occurrences). Not encoded as `loc(...)` or attrs. |
| ONNX source node | **No** — same loss; ONNX names (`/model.0/conv/Conv`) never survive the first parse. |
| Semantic region | **No** — `cv.region_id` exists on 36 MLIR ops but the builder does not copy it into the per-op bundle; plan regions carry only counts. |
| Runtime dispatch unit | **No such concept exists** in the plan. |
| Produced tensor | **No** — bundles carry no result tensor IDs or shapes (except conv `shape_cost` byte aggregates). |

Missing fields (per-op): `source_node_id`, `source_op_type`, `source_name`,
`region_id`, `result_tensor_ids` (+ shapes/dtypes), `dispatch_unit_id`.
Prerequisite: the emitter must carry provenance as **MLIR attributes or locations**,
not comments, so it survives parsing and pass pipelines.

## 4. Backend Decision Truth

- **What caused `selected_backend: "coreml"`**: the profile field
  `configuredComputeUnits: "CPU+GPU+ANE"`, via a hardcoded mapping in
  `compile-for-target/main.cpp:244-246` (`lowerToTargetConstraints`) →
  `preferred_backend = coreml`, `allowed = {coreml, metal, cpu}`. It is a
  **function-level declared policy**, source
  `cv-target-profile-static-policy`, truth boundary
  `decision_collected_from_v1_mlir_attrs_evidence_not_tracked`.
- **Whole-graph Core ML support**: never established. The profile's Core ML
  `supportedOps` (`matmul, conv2d, relu, softmax, reshape, gather` — generic
  vocabulary) were never matched against the emitted `linalg.*`/`tensor.*` op
  names; no pass claims Core ML can take the emitted graph as one unit, and no
  mlprogram/mlmodel conversion contract exists.
- **Per-op Core ML kernel contracts**: none. Zero per-op decisions select coreml.
- **The plan contradicts itself as a dispatch story**:
  `CVExecutionPlanAttrsPass.cpp:300-311` *seeds* every tensor-producing op with
  `kernel.exists = false`, `lowering_status = fallback_required`,
  `fallback_backend = metal` (first entry of the fallback chain), and
  `LoweringDecisionPlanningPass` then mechanically converts that seed into
  `lowering_path = fallback_backend`, reason `fallback_to_metal`, on **all 846**
  ops that have a kernel decision. So the function header says "coreml" while
  every planned op says "metal" — the fallback is a seeded placeholder, not a
  capability comparison.
- **Metal/CPU fallback executability**: metadata only. No Metal or CPU kernels
  exist for any op in this graph (see §5), and no fallback lowering was
  materialized.

Conclusion: backend selection is **not actionable** for dispatch. It is truthful
as labeled policy, but a runtime adapter must treat it as a hint, not a contract.

## 5. Kernel Decision Truth

| Category | Count | Notes |
|---|---|---|
| Selected real runtime kernels (`kernel_selection` accepted) | **0** | All 929 are `rejected_no_kernel_for_op`. The registry (`kernel_selection_contract_v1`) declares exactly one kernel — `metal_rmsnorm_f32_v1` — and no rmsnorm op exists in this graph. |
| Analytical lowering mappings (KernelDecision) | 846 | All `lowering_path = fallback_backend`, `kernel_exists = false`, empty `selected_kernel`/`kernel_library`; static cost model v1 attached (`total_cost = 22` uniform). Placeholders, not mappings to implementations. |
| Deferred kernels | 0 in `kernel_selection`; 76 tile plans `deferred_missing_memory_hierarchy` (profile declares no `localMemoryBytes`) | |
| Fallback-only decisions | 846 (`fallback_backend = metal`) | |
| Kernel IDs ↔ runtime registry | Vacuously consistent: no kernel ID was selected, so nothing to mismatch — and nothing to dispatch. |

The remaining 83 bundles (`arith.constant`) carry only the rejected
`kernel_selection` (they have no tensor result, so the CV attrs pass never seeded
`kernel.*`/`quant.*`/`layout.*` on them).

## 6. Memory Metric Definitions

Computed in `CVExecutionPlanAttrsPass.cpp:239-267`, copied into the plan by the
builder/exporter:

| Field | Exact definition | Value |
|---|---|---|
| `estimated_input_bytes` | Σ static byte size of all **158 function argument types** (image + weights + baked constants) | 18,700,484 |
| `estimated_output_bytes` | Σ byte size of the two `func.return` operand types (3,897,600 + 3,276,800) | 7,174,400 |
| `estimated_temporary_bytes` | Σ byte size of **every result type of every non-terminator body op** — i.e. **cumulative SSA-value allocation/write volume** | 924,245,084 |
| `estimated_total_tensor_bytes` | input + output + temporary | 950,119,968 |

`estimated_temporary_bytes` is therefore **neither peak live memory nor a
deduplicated sum of temporaries**:

- It counts a `tensor.empty` result *and* the result of the `linalg` op that fills
  it — the same logical buffer twice (or three times with `linalg.fill`).
- It counts the two graph outputs (they are results of body ops), so
  `estimated_total_tensor_bytes` double-counts the 7,174,400 output bytes.
- It equals Phase 23's `estimated_total_write_bytes` (924,245,084) exactly —
  confirming it is write-volume accounting.
- Phase 23's lifetime analysis (`cv_planning_facts.json.memory_summary`) computed
  `peak_live_temporary_bytes = 31,948,800` (~32 MB). The plan's figure is **~29×**
  that. A runtime sizing an arena from `estimated_temporary_bytes` would
  over-provision by ~892 MB on a mobile target.

Recommended explicit fields (schema addition; no slot allocation implied, none was
implemented):

- `total_temporary_tensor_bytes` — deduplicated sum of distinct temporary tensors
  (Phase 23: 917,070,684 under its accounting; definition must be pinned).
- `peak_live_temporary_bytes` — Phase 23's lifetime-scan metric (31,948,800),
  labeled `static_lifetime_analysis_no_slot_allocation`.
- `initializer_bytes` — weights/constants (≈13.79 MB; Phase 23 reports 13,785,524,
  arg-type summation gives 13,785,284 — the definition must state which).
- `workspace_bytes` — explicitly 0 / unknown today (no kernel workspace contracts).
- `planned_slot_bytes` — absent until a slot allocator exists; reserve the name.
- Keep or rename the current field to `cumulative_ssa_value_bytes` so no consumer
  mistakes it for a memory requirement.

## 7. Input / Weight / Output ABI

`cv_extension.inputs` lists 158 entries: `tensor_id` (`arg_0..arg_157`), `shape`,
`dtype`, `layout` — **all with `role: "graph_input"`**.

| ABI need | Status |
|---|---|
| Identify the image input | **Not expressible** — arg_0 is the image only by convention; no role distinguishes it. |
| Identify the 157 initializers/weights | **Missing.** Weights are **passed as function inputs** at the MLIR level and appear in the plan as ordinary graph inputs. No `is_initializer`/`weight` role, no byte totals, no classification. |
| Original tensor names | **Missing** — `model.0.conv.weight`, `images`, etc. exist in `generic.mlir` arg names and in GenericGraphIR/planning facts, but the plan has only positional `arg_N`. |
| External model file reference | **Missing** — `provenance.model_spec_ref = ""`; `models/yolo-seg.onnx` is not referenced. A runtime cannot locate weight data from the plan. |
| Output binding | Partial: 2 outputs with roles `detection` / `segmentation_prototype`, shapes, dtypes, layouts. `result_0/1` are not linkable to internal tensors; no ownership field (Phase 23 had `caller_visible_output`). |
| Layout vocabulary | Heuristic from rank (`rank4→nchw`, `rank2/3→row_major`, else `ranked_tensor`): rank-1 bias tensors get the non-layout `ranked_tensor`. |
| Ownership | Absent everywhere. |

Net: weights are **effectively absent from the runtime contract**. A runtime
adapter cannot bind the model input, cannot load weights, and cannot tell the two
apart, without out-of-band knowledge (the GenericGraphIR JSON or the ONNX file).

## 8. CV Extension Completeness

Serialized and usable: output roles (`detection`, `segmentation_prototype`),
shapes/dtypes, one `postprocess_boundary: "model_output_boundary"` string, four
region summaries (id, role, confidence, op count, feature scales),
`target_profile_id`, `model_family`.

Exists upstream but **not serialized into the plan**:

- **Region membership**: `operation_ids`, `input_tensor_ids`, `output_tensor_ids`,
  `input/output_shapes`, per-region flops/read/write/temporary-byte estimates,
  `candidate_execution_domains`, quantization/fusion eligibility — all present in
  `yoloseg.cv_planning_facts.json` regions; the plan keeps only counts
  (`ExecutionPlanBuilder::collectCVPlanExtension` accumulates count/role/
  confidence/scales and drops the rest; `cv.region_id` attrs on the 36 MLIR ops
  are likewise not exported per op).
- **Per-output postprocess contract**: Phase 23 has `postprocess_required: true`,
  `producer_region`, `ownership_expectation` per output; the plan has none of it.
- **NMS requirement**: stated **nowhere** — not in the plan, the planning facts,
  or the semantic report. Nothing tells a runtime that the `[1,116,8400]`
  detection tensor requires score-threshold + NMS.
- **Mask decoding requirement and channel semantics**: nowhere is it recorded
  that 116 = 4 (box) + 80 (class) + 32 (mask coefficients), nor that masks =
  sigmoid(coefficients × prototypes) cropped to boxes. The
  `mask_coefficient_branch` region name hints at it; no consumable contract.
- **Region dependencies**: no edges between regions (e.g. detection_head and
  mask_coefficient_branch both feed postprocess) are serialized anywhere.

A runtime can locate the two output tensors, and nothing more of the CV story.

## 9. Runtime-Readiness Verdict

**`NEEDS_DISPATCH_UNIT_MATERIALIZATION`**

Why not the neighboring verdicts:

- Not `READY_FOR_RUNTIME_ADAPTER`: zero dispatchable kernels, no dispatch units,
  no weight ABI, contradictory backend story at the per-op layer, misleading
  memory field.
- Not merely `NEEDS_PLAN_SCHEMA_FIXES`: adding fields to the existing 929
  positional bundles would still leave the runtime to re-derive executable units
  from an op stream where 53% of entries are allocation placeholders and
  constants. The unit of execution itself must be materialized by the compiler.
- Not `NOT_RUNTIME_READY`: the pipeline is real and truthfully labeled; the
  function-level plan, I/O shapes/dtypes, and region summaries are correct as far
  as they go. The gap is a missing materialization layer plus schema additions,
  not a rotten foundation.

### Minimum fixes before runtime implementation

1. **Provenance as IR, not comments** — emit `source_node_id` /
   `source_op_type` / `source_name` as MLIR attributes (or locations) in the
   GenericGraphIR→MLIR emitter so they survive parsing and reach the builder.
2. **Dispatch-unit materialization** — group each source node's compute anchor
   with its `empty`/`fill`/`pad`/`constant` helpers; export a `dispatch_units`
   list (unit id, source provenance, region id, input/output tensor contracts,
   kernel/backend decision per unit). Per-op decisions may remain as the audit
   layer beneath units.
3. **ABI completion** — input roles (`image` vs `initializer`), original tensor
   names, external model artifact reference in `provenance.model_spec_ref`,
   output ownership; add `initializer_bytes`.
4. **Memory field honesty** — add `peak_live_temporary_bytes` (Phase 23
   algorithm) and `total_temporary_tensor_bytes`; rename or clearly re-scope the
   current cumulative figure.
5. **CV contract serialization** — region membership/dependencies, per-output
   `postprocess_required`, detection channel semantics (4+80+32), explicit
   NMS-required and mask-decode-required flags at the postprocess boundary.
6. **Backend decision coherence** — either match Core ML capability against the
   emitted ops (and report honest per-unit support) or label the function-level
   selection as `policy_only_not_capability_matched` so the coreml/metal
   contradiction cannot be misread.

All labels above must keep the existing truth-boundary discipline: everything in
this plan remains static/declared; nothing here is measured runtime behavior.
