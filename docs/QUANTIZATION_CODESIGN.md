# Quantization Co-Design (quantization_codesign_contract_v1)

## Four concepts, kept separate

This framework never collapses these into one string:

| Concept | Meaning | What this repository implements TODAY |
|---|---|---|
| **A. Algorithm** | How quantized parameters are produced (PTQ, SmoothQuant, AWQ, GPTQ) | **None.** No quantization algorithm is implemented here. The forced-AWQ profile *declares* an externally produced artifact (`quantization.algorithm = "awq"`, artifact ref) — a declaration, not an implementation. |
| **B. Numeric representation** | The stored/compute format (weight-only INT8, group-wise INT4, activation INT8) | **Modeled statically:** weight-only INT8 (and INT4 where a profile declares it) in planning attrs and byte estimates. No group-wise metadata, no activation quantization parameters, no scale/zero-point values exist anywhere except hand-authored HIR demo tests. |
| **C. Backend / kernel execution support** | An actually dispatchable implementation | **None for quantization.** The only dispatchable runtime kernel is `metal_rmsnorm_f32_v1` (fp32, unquantized). Backend profiles *declare* quant modes (library capability); that never counts as dispatchable. |
| **D. Accuracy evidence** | None / fixture / calibrated / measured | **None.** No calibration data and no measured accuracy exist in this repository. The only reachable statuses are `no_accuracy_evidence` and `algorithm_declared_not_calibrated`. |

## The pass

`QuantizationCoDesignPass` (`quant-codesign-pipeline`; run by
`compile-for-target` after kernel selection) evaluates matmul-like
constant-weight ops and emits per-op `quant_codesign.*` attrs, exported as
the per-op `quantization_codesign` object.

**Inert by default.** It runs only when `quant.codesign.policy` is set (via
the optional profile field `quantizationCoDesignPolicy` or a module attr).
No existing profile sets it, so all existing artifacts are byte-identical.
`quant.strategy` and every other existing planning attr are untouched.

### Policies (recorded in every decision)

| Policy | Selection rule |
|---|---|
| `planning_only` | Evidence only; never selects. |
| `systems_cost_only` | Selects iff estimated systems benefit > 0. |
| `require_dispatchable_kernel` | Additionally requires a concrete `target.runtime_kernels` descriptor that consumes quantized weights. |
| `require_accuracy_evidence` | Defers without calibrated/measured evidence — **always defers today**. |

### Statuses

`selected`, `planning_only_not_selected`, `rejected_weight_not_constant`,
`rejected_backend_not_legal`, `rejected_no_systems_benefit`,
`backend_supported_but_not_dispatchable`, `deferred_no_runtime_kernel`,
`deferred_missing_accuracy_evidence`, `deferred_missing_cost_estimates`,
`deferred_missing_weight_classification` — plus `rejection_reasons` naming
why. Missing information defers; contradicting information rejects.

### Unknowns are omitted, never defaulted

Granularity, group size, axis, and symmetric/asymmetric are **absent** from
the contract because no pass produces them (the legacy
`QuantizationStrategyPlanningPass` default of `per_channel` is exactly the
kind of silent default this contract avoids). `scale_source` /
`zero_point_source` are `not_available_no_calibration`. The hardcoded
`accuracy_risk` strings of the legacy pass are superseded by the
`accuracy_evidence.status` field, which states evidence, not vibes.

## Cost model terms — source, unit, inclusion boundary

All terms are **static compiler estimates**; units are bytes and integer
nanoseconds (roofline form, from `estimateShapeCost` in
`ShapeCostModel.h`). Declared profile peaks only; nothing is measured.

| Term | Source | Unit | Included in |
|---|---|---|---|
| `weight_bytes_before` | `weight_elems × activation dtype bits / 8` (shape facts) | bytes | before-total memory term |
| `weight_bytes_after` | `weight_elems × quantized dtype bits / 8` | bytes | after-total memory term |
| `boundary_bytes` | **Materialized float dequant intermediate**: `2 × float weight bytes` (read quantized, write float) — same formula as `ShapeCostModel`'s dequant boundary | bytes | after-total only, only when **no** dispatchable weight-only kernel exists |
| `total_cost_before_nanos` | `max(flops/peak, bytes_before/bandwidth)` | ns | comparison baseline |
| `total_cost_after_nanos` | `max(flops/peak, bytes_after/bandwidth) + boundary_bytes/bandwidth` | ns | comparison target |
| `systems_benefit_nanos` | `before − after` | ns | selection gate (cost policies) |
| `excluded_terms` | — | — | **Not modeled**: `inline_dequant_unpack_conversion_cost`. A dispatchable weight-only kernel eliminates the *materialized* float intermediate; it does **not** imply zero scale-handling, metadata, unpacking, or inline-conversion cost. That cost is explicitly listed as excluded rather than silently treated as free. |

**Double-counting prevention:** `quant_codesign.est.*` is evidence only. It
is never read by `CandidateEvaluation` (`ServingCostModelPass`) or
`PlanSelection`. This is enforced by the ranking-invariance test in
`tools/run_mlir_pass_tests.sh` (`run_quant_codesign_ranking_invariant`),
which diffs every `evaluation.*` / `selected_plan.*` signal with the pass
enabled vs disabled and requires byte-identical results.

## The key co-design finding this model surfaces

Under a *materialized* dequant boundary (no kernel that consumes quantized
weights), weight-only quantization moves **more** total bytes than fp16
(0.5× weights + 2× dequant traffic) and honestly loses
(`rejected_no_systems_benefit`). It wins only when a concrete runtime
kernel consumes quantized weights directly — which is precisely why kernel
availability, not representation choice, is the gating co-design decision.

## What remains deferred, and why

- **Honest `hir.dequantize` materialization** — split out as its own change
  (PR6B): requires a real quantized-typed operand in the IR plus explicit
  scale/zero-point metadata; the serving path has neither (weights are not
  IR operands; no calibration exists).
- **Calibration / accuracy evaluation** — no data in the repository.
- **Group-wise/per-channel parameter metadata** — no producer exists.
- **Quantized runtime execution** — no dispatchable quantized kernel exists.

Truth boundary of everything above:
`quantization_codesign_static_planning_no_calibration_no_measured_accuracy`.
