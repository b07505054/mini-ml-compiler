# AArch64 Matmul Schedule Candidate Evidence and Cost Model

Turns Stage 12's static backend evidence and Stage 13's Raspberry Pi
measurements into a structured, provenance-tagged, reusable candidate
evidence and scoring model. Corrects the Stage 12 classification policy
that treated any spill as an automatic rejection -- Stage 13 measured two
spilling candidates that were, in fact, faster than their matched
baselines.

## Scope (do not overclaim)

**"Tiled AArch64 matmul schedule candidates varying K-unroll and tile
configuration."** Loop order is fixed to the tiled M/N/K nest already
implemented by `tile_schedule_matmul_bias_relu.template.mlir`. This is
**not** a general loop-interchange framework -- `loop_order_id` is a fixed
constant (`tiled_mnk_row_major_v1`) in every candidate key this stage
produces, and no code here selects between loop orders.

## 1. Candidate/cost-model integration gap (audit finding)

The tiled-scheduled candidates from Stages 10-13 exist **only as scripts
and JSON artifacts** -- this entire backend-codegen series is Python
tooling driving unmodified LLVM/MLIR via subprocess. There is no in-tree
MLIR pass that represents "AArch64 matmul schedule candidate" as a
compiler-side type.

The project's one existing compiler-integrated cost-model/candidate-
selection system (`mlir_passes/include/serving/ServingCostModel.h`,
`mlir_passes/lib/serving/PlanSelectionPass.cpp`, `decision::DecisionCost`)
is a **separate subsystem** for LLM-serving quantization/layout/backend-
fallback decisions in a 15-pass serving pipeline -- unrelated to this
matmul-kernel tile/schedule family, and out of scope to modify per this
task's explicit instruction not to touch unrelated serving code.

**The missing link**: no function or pass turns Stage 3-13's tile/schedule
candidate artifacts into a structured, rankable, provenance-tagged
evidence object. This module (`tools/aarch64_schedule_candidate_model.py`)
creates that link at the Python/tooling level -- deliberately *mirroring*
the existing C++ system's cost-model convention (a decomposed cost struct
whose named components sum to a total, carrying an explicit
`cost_model_id` and `truth_boundary` string, matching `DecisionCost`'s
design) rather than inventing an unrelated schema, but implemented
independently in Python since a genuine C++/MLIR-pass integration is out
of scope for this stage (task section 12 explicitly permits an
analysis/reporting-mode-first approach).

## 2. Candidate representation

`CandidateKey` (frozen dataclass, `tools/aarch64_schedule_candidate_model.py`):
`target_arch`, `target_cpu`, `target_features`, `dtype`, `shape_m/n/k`,
`tile_m/n/k`, `vector_width` (4 -- f32 lanes per 128-bit NEON register,
`v*.4s`, fixed throughout this series), `schedule_unroll_k`,
`loop_order_id` (fixed), `microkernel_id`. Identity is derived purely from
these fields via `canonical_id()` -- a display `label` lives *outside* the
key on `CandidateEvidenceRecord` and never participates in equality or
hashing (verified: `test_labels_do_not_affect_identity`).

A real bug was found and fixed while wiring Stage 12 and Stage 13 data
into this model: **the two stages independently chose different label
strings for the same candidate** (Stage 12's `primary_unroll2` is Stage
13's `primary_uk2`). An earlier version of the loader merged by label and
silently produced two half-populated records for one real candidate --
exactly the identity failure this design exists to prevent. Fixed by
merging via `CandidateKey` equality, with an alias map returned so
callers holding Stage-13-native labels can still resolve records.

## 3-4. Evidence categories and provenance

Every evidence field is an `EvidenceValue` (`value`, `provenance`, `note`,
`estimated`), never a raw number. `Provenance` records `source_level` (one
of `mlir`, `llvm_ir`, `mir`, `assembly`, `llvm_mca`,
`raspberry_pi_measured`, `manual_config`), `source_artifact`,
`tool_version`, `target`, `revision`, `candidate_id`, `timestamp`.
`EvidenceValue.__post_init__`-equivalent validation (`Provenance`'s
constructor) raises on an invalid `source_level` string, so a measured
latency literally cannot be mis-recorded as `llvm_mca` evidence --
enforced, not just documented (`test_measured_latency_cannot_be_recorded_as_llvm_mca_evidence`).

Three structurally separate evidence classes, never merged into one map:
`StaticIRBEvidence` (MLIR-level), `LLVMBackendEvidence` (MIR/assembly-
level), `MeasuredHardwareEvidence` (Raspberry Pi). Missing evidence is
`None` with a `note`, never a fabricated zero
(`test_missing_evidence_is_none_not_fabricated_zero`).

## 5. Attribution findings

`build_attribution()` produces a benefit/cost-separated report for a
matched pair. For the primary candidate (uk1 -> uk2, see
`attribution_summary.json`):

**Benefits** (K-loop dynamic trip count 4 -> 2, a 50% reduction -- labeled
`estimated: true`, derived as `K_trip_pre_unroll / schedule_unroll_k`, not
an instrumented count; branch and induction-update reduction estimated at
the same 50% ratio; larger scheduling region: 128 -> 256 static FMLA per
body).

**Costs** (static body growth +100%; code size +24.5% (2608 -> 3248
bytes); spills/reloads/stack-frame/physical-registers: all **zero
change**).

**Measured result** (the only non-estimated, non-static number in the
report): **+9.29% real latency improvement**.

For the full-K-unroll diagnostic (uk1 -> uk4): same benefit mechanism (75%
trip-count reduction, 4 dynamic iterations -> 1) but real costs this time
(spills 0 -> 11, reloads 0 -> 12, code size +62%), and the measured result
is still **+20.0% improvement** -- the benefit outweighed the cost at this
problem size, which is exactly the finding this stage exists to represent
truthfully rather than hide behind a hard rejection.

## 6. Corrected spill interpretation

Replaced the Stage 12 "`spill_count > 0` -> Class D -> reject" policy with
two **independent** classification dimensions (never mutually exclusive):

- **Backend safety**: `backend_safe` | `backend_costly` (spills/reloads
  present)
- **Hardware confirmation**: `hardware_confirmed_profitable` |
  `hardware_confirmed_neutral` | `hardware_confirmed_regression` |
  `hardware_unknown` (no compatible measurement) | `incorrect`

Real output (`classification_summary.json`):

| Candidate | Backend safety | Hardware confirmation |
|---|---|---|
| primary_unroll2 | Backend-safe | hardware-confirmed profitable |
| **primary_full_unroll** | **Backend-costly** | **hardware-confirmed profitable** |
| **alt_k_tile_unroll2** | **Backend-costly** | **hardware-confirmed profitable** |
| cube64_unroll2 | Backend-safe | hardware-confirmed profitable |
| cube16_uk2 / rect_uk2 / large_uk2 | Backend-safe | hardware-confirmed profitable |

**"Backend-costly, hardware-confirmed profitable"** is now a real,
representable, non-contradictory label -- exactly the Stage 13 finding.

## 7. Cost equation and default weights

```
total_cost = static_compute_cost + loop_control_cost
           + register_pressure_penalty + spill_penalty + reload_penalty
           + code_size_penalty + stack_frame_penalty + target_specific_penalty
           (- calibration_bonus, calibrated mode: replaces the static sum
              when compatible measured evidence exists)
```

Lower is preferred. Default weights (`CostWeights`, all visible in every
`CostBreakdown`, none hidden):

| Component | Default weight | Rationale |
|---|---|---|
| `compute_cost_scale` | 1e-6 * shape_flops | per-shape constant; does not differentiate same-shape candidates, included for transparency |
| `loop_control_cost_per_iteration` | 1.0 per estimated dynamic K-loop iteration | the one term that genuinely decreases with unrolling -- the mechanism behind every Stage 13 win |
| `register_pressure_weight` | **0.0 (deliberate)** | Stage 12 found the only available pre-RA heuristic (`approx_peak_live_vector_registers`) over-counts with no loop-back-edge modeling on this project's MIR; reported for visibility, not scored |
| `spill_weight` | 2.0 per spill store | soft penalty, never a veto |
| `reload_weight` | 1.5 per reload load | soft penalty |
| `code_size_weight_per_kb` | 0.05 per KB | soft penalty |
| `stack_frame_weight_per_64b` | 0.1 per 64 bytes | soft penalty |
| `target_specific_penalty` | 0.0 | no Cortex-A76-specific penalty identified in this slice |
| `calibration_weight` | 1000.0 | calibrated mode: `measured_median_ms * this` dominates the static sum when compatible evidence exists |

Two legitimate, mode-independent hard rejections remain (distinct in kind
from the removed "any spill" veto): `UNSUPPORTED` (no compiled object
exists at all) and `INCORRECT` (measured on real hardware and found
numerically wrong) -- neither is a backend-cost signal, both mean "not a
valid candidate."

## 8. Measured-evidence integration policy

`check_compatibility()` requires target_arch/cpu/features/dtype/
microkernel_id/vector_width/loop_order_id to match exactly (fails closed,
confidence 0.0, on any mismatch) plus a matching `BENCHMARK_METHODOLOGY_VERSION`
(`stage13_pi5_harness_v1` -- a stale version fails closed regardless of
everything else matching). Four compatibility levels: `exact_match`
(confidence 1.0), `cross_shape_same_schedule` (0.7 -- same tile/unroll,
different shape; Stage 13 observed this schedule win on 5 independent
shapes), `shape_bucket` (0.4 -- same tile, different unroll/shape),
`incompatible` (0.0). Missing/incompatible evidence lowers confidence, it
never silently reads as zero cost or zero confidence being ignored.

## 9. Shape-aware findings

From the real 5-shape Stage 13 result set (`shape_aware_findings.json`):

| Statement | Classification |
|---|---|
| uk1->uk2 profitable on every shape tested (3 shapes beyond the primary/cube64 pairs already covered by attribution: 16x16x16, 32x64x32, 128x128x128) | **directly measured** |
| Effect generalizes to matmul shapes broadly (any M/N/K, any tile) | **unsupported** -- only 1 tile/unroll combination ever tested across shapes |
| Speedup magnitude varies (10.8%-31.5%) but sign is consistent | **repeated cross-shape observation** |
| Speedup correlates with K-loop trip count / code-size growth in a fittable way | **unsupported** -- 3-5 data points is a curve through noise, not a validated model; task brief explicitly requires discrete evidence buckets over an unjustified formula at this sample size |
| Spilling candidates can be profitable | **directly measured** -- 2 of 2 tested Group B diagnostics |
| Spilling candidates are *usually* profitable | **plausible hypothesis only** -- both spilling candidates measured came from the same 32x32x32 shape and the same underlying mechanism; not sufficient to generalize to problem sizes where the spilled working set no longer fits L1 |

## 10-11. Ranking experiment and prediction-vs-measured comparison

Full detail: `ranking_comparison.md`, `static_ranking.json`,
`calibrated_ranking.json`, `ranking_prediction_comparison.json`. Headline:

| Group | Static (either mode) top pick | Measured top pick | Static pairwise agreement | Calibrated pairwise agreement |
|---|---|---|---|---|
| primary (uk1/uk2/uk4) | primary_unroll2 | **primary_full_unroll** | **33%** (WRONG top pick) | 100% |
| alt_k_tile (uk1/uk2) | alt_k_tile_unroll1 | **alt_k_tile_unroll2** | **0%** (WRONG top pick) | 100% |
| cube64 (uk1/uk2) | cube64_unroll2 | cube64_unroll2 | 100% (correct) | 100% |

**Honest result, not forced to agree**: the static model correctly ranks
the zero-spill group (cube64) but gets the top pick WRONG in both groups
that include a spilling candidate -- it has no way to know that Stage 13
measured those spilling candidates as faster. This is exactly the valid,
non-overfit Stage 14 outcome the task brief anticipated: *"static model
correctly identifies safe candidates, but cannot fully rank profitable
spilling candidates, measured calibration is required for accurate
selection."* No weights were tuned to force agreement; the default weights
above are the same simple, documented values used throughout.

## 12-13. Compiler integration boundary and experimental options

**Analysis/reporting mode only** in this stage -- no automatic compiler
materialization is added or enabled. `tools/run_aarch64_schedule_cost_model.py`
implements:

- `--schedule-candidate-mode=static|calibrated` (wired to
  `RANKING_MODE_STATIC_SOFT_PENALTY` / `RANKING_MODE_CALIBRATED_PI`
  respectively; `RANKING_MODE_STATIC_HARD_REJECT` is intentionally **not**
  exposed as a selectable production mode -- it exists only inside the
  ranking-comparison experiment to demonstrate what the corrected policy
  replaced)
- `--schedule-profile=<path>` (measured-evidence pool source; defaults to
  this repo's own Stage 13 artifact -- calibration is tied to a declared
  profile, never implicit)
- `--emit-schedule-cost-breakdown=<path>` (every score component visible)

Every option is wired to real behavior; none are no-op flags. The
production-mode report states candidates considered, candidates rejected
(none for legality reasons in this candidate family; the mechanism exists
and is tested), static evidence, measured evidence used (or the fallback
reason when unavailable), the full cost breakdown, selected candidate, and
confidence.

## Truth boundary

- This model is calibrated only for the tested Raspberry Pi 5 Cortex-A76
  path (`raspberry_pi_5_cortex_a76`, methodology version
  `stage13_pi5_harness_v1`). Using it to inform decisions about different
  hardware requires new measurement, not extrapolation.
- Spill count is a cost signal, never a universal rejection condition.
- Hardware measurements override inaccurate static severity predictions
  only for compatible profiles (`check_compatibility()` fails closed).
- General loop-interchange selection is not implemented -- `loop_order_id`
  is fixed throughout.
- Stage 12's original static classification is preserved, not deleted or
  rewritten -- see "Documentation correction" below.

## Documentation correction (Stage 12 wording)

Stage 12's artifacts correctly reported the static finding as of Stage
12: full-K-unroll and the alternate K-tile showed real spills and were
classified "Class D -- regression risk". That was an honest reading of
the evidence available at the time and is **not** rewritten here. What
changed is Stage 13 added real hardware measurement, which **contradicted
the severity implied by "regression risk"** for both candidates. The
correct, non-revisionist language going forward:

> Stage 12 static prediction: regression risk due to spills.
> Stage 13 measured result: prediction contradicted by hardware latency --
> both candidates were faster than their matched baselines.
> Current classification: **backend-costly, but hardware-confirmed
> profitable** (at the tested Raspberry Pi 5 / 32x32x32 problem size).

## Files in this directory

- `README.md` -- this file
- `commands.txt` -- every command used
- `candidate_schema.json` -- the full CandidateKey/evidence/cost schema, machine-readable
- `evidence.json` -- every candidate's full evidence record (all 3 categories + provenance)
- `static_ranking.json` -- both static modes (legacy hard-reject comparison + corrected soft-penalty default), per matched group
- `calibrated_ranking.json` -- calibrated-mode ranking per matched group
- `ranking_prediction_comparison.json` -- predicted vs. measured order, pairwise agreement, per group per mode
- `attribution_summary.json` -- benefit/cost/measured-result attribution for 7 matched pairs
- `classification_summary.json` -- backend-safety x hardware-confirmation for every candidate
- `shape_aware_findings.json` -- what can honestly be generalized from 5 shapes
- `ranking_comparison.md` -- human-readable version of the ranking experiment
- `production_cost_breakdown_static.json` / `production_cost_breakdown_calibrated.json` -- `--emit-schedule-cost-breakdown` output for the full 14-candidate production-mode run in each mode

This directory does not duplicate Stage 12's 6.8MB of MIR/register
evidence or Stage 13's 3.2MB of Pi execution evidence -- every number here
traces back to those artifacts via `provenance.source_artifact`.
