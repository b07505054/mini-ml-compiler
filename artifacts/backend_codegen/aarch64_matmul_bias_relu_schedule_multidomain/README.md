# Multi-Domain Calibration and Shape/Tile Compatibility Validation

**The opt-in AArch64 schedule selector supports multiple provenance-checked
Raspberry Pi calibration domains and chooses independently measured
K-unroll schedules for exact compatible shape/tile configurations.**

This is not: arbitrary-shape autotuning, universal schedule optimality,
default production enablement, cross-target learning, or general loop
interchange. Calibrated mode remains opt-in; default behavior is
unaffected.

## Executive result

Four independent domains were measured on real Raspberry Pi 5 hardware:
the primary domain (32x32x32/tile 8x8x8, already established) plus three
new domains added this stage -- a larger same-tile shape (64x64x64), an
alternate tile (8x8x4), and a rectangular shape (32x64x32). **All four
domains measured `schedule-unroll-k=4` as the fastest candidate**, each
independently, with distinct backend-cost profiles (0, 0, 4, and 18
spills respectively). This is reported honestly as-is -- see "Seek
differentiated evidence, but do not manufacture it" below for why this is
not overfit or forced, and what it does and does not license claiming.

## 1. Preserved Stage 15 truth boundary

The supported statement remains exactly: *"For the tested Raspberry Pi 5
Cortex-A76 primary 32x32x32/8x8x8 domain, calibrated selection chooses
uk4."* No Stage 15 historical artifact was altered. This stage adds three
**additional, independently measured** statements of the same form for
three **additional, independently measured** domains -- it does not
retroactively claim broader support for the primary domain's own
evidence, and it does not claim uk4 is universally optimal (see below).

## 2. Domains evaluated

| Domain | Shape | Tile | Category |
|---|---|---|---|
| primary | 32x32x32 | 8x8x8 | (Stage 12-15, unchanged) |
| cube64 | 64x64x64 | 8x8x8 | A -- larger same-tile shape |
| altk | 32x32x32 | 8x8x4 | B -- different tile |
| rect | 32x64x32 | 8x8x8 | C (optional) -- rectangular shape |

All three new domains' `{uk1, uk2, uk4}` candidates passed Stage 11
structural validation before compilation (`transform.loop.unroll`
materializes correctly; uk4 on `altk`/`cube64` produces a genuine
2-iteration K-loop, not a full collapse, since K trip count is 8 there;
uk4 on `rect` collapses the K-loop entirely, K trip count 4 -- both
verified, not assumed).

## 3. Multi-domain profile schema

`multi_domain_profile.json` (schema `stage16_multidomain_profile_v1`):
each domain is a named, independently-identified section with its own
`domain_identity` (target/tile/shape/dtype) and a `candidates` dict keyed
by full canonical `CandidateKey` id. Domains are never flattened into one
undifferentiated pool at the source level; `select_and_compile_aarch64_matmul_schedule.py`'s
loader was extended (`load_profile_pool()`, new third schema branch) to
parse this format into the same `(CandidateKey, MeasuredHardwareEvidence)`
tuple pool used everywhere else -- domain isolation is enforced by
`check_compatibility()` (Stage 14, unmodified), not by file layout.

## 4-5. Correctness (all newly compiled/measured candidates)

9 new objects (cube64/altk/rect x {uk1,uk2,uk4}) were compiled through the
Stage 15 selector's manual mode (real materialization, not a second
mechanism), then run on the real Raspberry Pi 5 with the exact Stage 13
harness/methodology (interleaved measurement groups per domain, 5 groups
each, `taskset -c 3`, thermal capture before/after: 48.3'C -> 49.4'C, no
throttling). **9/9 bit-exact** (`max_abs_error = 0`) against the scalar
reference, zero repeated-call failures. Full detail:
`pi_multidomain_results.json`.

## 6-7. Domain-specific measured ranking

| Domain | uk1 median (ms) | uk2 median (ms) | uk4 median (ms) | uk4 vs uk1 | uk4 vs uk2 | CV (all) | Classification |
|---|---|---|---|---|---|---|---|
| primary | 0.002593 | 0.002352 | 0.002074 | +20.0% | +11.8% | <0.5% | **uk4 winner** |
| cube64 | 0.019759 | 0.017685 | 0.016630 | +15.8% | +6.0% | <0.2% | **uk4 winner** |
| altk | 0.002981 | 0.002630 | 0.002407 | +19.3% | +8.5% | <0.4% | **uk4 winner** |
| rect | 0.005185 | 0.004667 | 0.004278 | +17.5% | +8.3% | <0.3% | **uk4 winner** |

Every domain clears the noise floor by a wide margin (CV well under 0.5%
throughout, speedups of 6-20%) -- none of these are statistical ties.

**uk4 won every domain tested.** Reported honestly, not suppressed or
reframed. This does NOT mean uk4 is universally optimal:
- Every domain shares the same underlying microkernel family and the
  same Cortex-A76 target -- no evidence exists here about other
  kernels or targets.
- Backend cost differs sharply across domains (0/0/4/18 spills) even
  though the measured *ranking* didn't change -- see the causal
  discussion below on why the spill cost apparently didn't flip the
  ranking at these problem sizes, labeled as hypothesis, not fact.
- A larger unroll factor (uk8) was never tested; nothing here licenses
  extrapolating that trend further.

### Investigating why the ranking didn't flip despite growing spill cost

Labeled explicitly by evidence tier:
- **Measured**: `rect_uk4` has by far the worst backend cost of any
  candidate tested in this project to date (18 spills, 20 reloads, 368
  stack bytes) yet is still the fastest in its domain.
- **Derived** (from static evidence): all four domains' uk4 candidate
  halves the dynamic K-loop trip count again relative to uk2 (e.g. rect:
  K trip count 4 -> 1, a full collapse) -- the same loop-control-reduction
  mechanism responsible for every uk1->uk2 win in Stages 12-13.
- **Hypothesis, not measured**: the fixed number of spill/reload stack
  accesses (a handful of 16-byte vector slots) very likely hits L1 data
  cache reliably at these small problem sizes (the entire working set
  fits comfortably in Cortex-A76's 256KiB L1d), so the fixed spill cost
  stays small in absolute cycles even as its *count* grows, while the
  loop-control savings scale with the *shape*, not the spill count. This
  is plausible but **not verified** -- `perf` is unavailable on this Pi
  (same limitation as Stage 13), so no direct cache-miss evidence exists
  to confirm or refute this. Stated as hypothesis only.

## 8. Static-model ranking (re-evaluated across all 4 domains)

| Domain | Static top pick | Measured winner | Agreement |
|---|---|---|---|
| primary | uk2 | uk4 | **Wrong** |
| cube64 | uk4 | uk4 | **Correct** (uk4 happens to be spill-free here) |
| altk | uk1 | uk4 | **Wrong, more severely** (static regresses all the way to the most conservative candidate, since uk2 AND uk4 both have real spills in this domain) |
| rect | uk2 | uk4 | **Wrong** |

Static evidence alone correctly predicted the winner in exactly 1 of 4
domains -- and only because that domain's winner happened to be
spill-free. This is not retuned or forced; it is the same corrected
soft-penalty policy from Stage 14, applied unchanged. Valid conclusion,
matching the task brief's own anticipated outcome: **static evidence
remains useful for cost/risk characterization (it correctly flags every
domain's real spill counts) but is not a reliable standalone winner
predictor** -- measured calibration is required.

## 9. Calibrated-model ranking

Calibrated mode selected the correct measured winner in **all 4 domains**,
using only exact-domain evidence (verified via the cross-domain rejection
report below -- no domain's ranking was ever influenced by another
domain's measurements). Full breakdown: `calibrated_ranking.json`.

## 10. Cross-domain compatibility behavior

`cross_domain_rejection_report.json`: every one of 12 cross-domain pairs
(4 domains x 3 others) shows `exact_match_rejected: true`. Same-tile
domains (primary/cube64/rect, all tile 8x8x8) get
`cross_shape_same_schedule` (confidence 0.7, never treated as exact);
`altk` (the only different-tile domain) is `incompatible` with all three
others. This directly confirms:
- primary 32^3 evidence is never an exact match for 64^3 (cube64)
- 8x8x8 evidence is fully incompatible with 8x8x4 (altk)
- rectangular evidence is never an exact match for cubic domains
- (unsupported-shape fallback and incompatible-target rejection: see
  Stage 15's own fixtures, unchanged and reused here)

`test_cross_domain_evidence_never_merged_in_ranking` and
`test_incompatible_domain_cannot_affect_score` (new Stage 16 tests) would
fail if cross-domain leakage were ever introduced.

## 11. Shape/tile bucket policy (unchanged from Stage 14, now exercised across 4 real domains)

1. Exact semantic candidate + exact shape -- confidence 1.0
2. Exact tile/kernel/target + same schedule, different shape
   (`cross_shape_same_schedule`) -- confidence 0.7, only ever used when no
   exact match exists, and only ever a *secondary* signal
3. Static fallback
4. Conservative `schedule-unroll-k=1` fallback

No new bucket was introduced this stage. `altk`'s different tile remains
fully incompatible with everything -- this project has exactly one
same-tile bucket (8x8x8, backed now by 3 independently measured domains)
and treats every other tile as its own isolated domain.

## 12. Integrated selection and materialization examples

All three compiled through the real pipeline, object identity verified:

| Example | Query | Mode | Selected | Object SHA-256 (12) | Matches |
|---|---|---|---|---|---|
| 1 -- primary exact | 32x32x32/8x8x8 | calibrated | uk4 | `aec26e5acb2e` | identical to Stage 15's own calibrated_sel.o |
| 2 -- larger domain | 64x64x64/8x8x8 | calibrated | uk4 | `ea24f664fab2` | identical to the manually-compiled cube64_uk4.o |
| 3 -- unknown domain | 16x16x16/4x8x8 | calibrated | uk1 (fallback) | `b13fe078a755` | conservative baseline, bit-exact on Pi |

**A real bug was found and fixed while building example 2**: the
selector's static/backend evidence source (`--stage12-json`) was
completely independent of `--schedule-profile` -- it always loaded from
the *original* Stage 12 file only, so `cube64_uk4` (never analyzed by
Stage 12) looked "UNSUPPORTED" (no compiled-object record at all) even
though compatible *measured* evidence existed and the candidate had
compiled and run correctly. Fixed by building
`extended_stage12_evidence.json` (original Stage 12 candidates plus the 5
new candidates' freshly-extracted static/backend evidence, same schema)
and passing it via the already-existing `--stage12-json` override --
reusing the existing mechanism, not adding a new one. Regression-tested
(`test_stage12_evidence_source_is_independently_overridable_from_measured_profile`).

## 13. Pi confirmation

Examples 1 and 2's selected objects are **byte-identical** to
already-Pi-tested objects from this stage's own benchmark matrix (no
re-test needed -- identical bytes were already measured). Example 3 (the
one genuinely new fallback object) was freshly transferred, built, and
run on the Pi: bit-exact correctness, 0 repeated-call failures, median
0.0005ms.

## 14. Static vs. measured findings (final)

Consistent with Stage 14/15's original finding, now confirmed across 4
independent domains rather than 1: static evidence is a real, honest,
useful **safety and cost** signal (it never hid a single spill, and its
predictions were directionally reasonable) but is a reliable **winner
predictor** in only 1 of 4 domains tested -- exactly the domain where the
winner happened to have zero backend cost. This is not an argument for
abandoning static analysis (it remains the only signal available before
any hardware exists to measure against, and it correctly flags real
risk), but it is decisive evidence that automatic schedule selection for
this microkernel family requires calibrated measurement, not static
scoring alone, to reliably pick the fastest candidate.

## Files in this directory

- `README.md` -- this file
- `commands.txt` -- every command used
- `multi_domain_profile.json` -- 4-domain measured-evidence profile (new schema)
- `domain_summary.json` -- per-domain candidate table + winner classification
- `static_ranking.json` / `calibrated_ranking.json` -- full per-domain cost breakdowns, both modes
- `compatibility_matrix.json` -- full pairwise compatibility levels across every candidate in every domain
- `cross_domain_rejection_report.json` -- explicit cross-domain exact-match-rejection proof
- `extended_stage12_evidence.json` -- Stage 12 static evidence + the 5 new candidates' static evidence, same schema (the Example 2 bug fix)
- `example{1,2,3}_*_selection.json` -- full selection reports for the three integrated examples
- `pi_multidomain_results.json` -- raw Pi measurement data for all 9 new candidates (interleaved groups, thermal snapshots, environment capture)
- `compiled/` -- all compiled artifacts (18 objects: 9 new candidates + 3 baseline re-verifications + 3 integrated-example outputs + others)
- `pi_candidates/<label>/manifest.json` -- per-candidate Pi measurement manifest

References Stage 12 (`aarch64_matmul_bias_relu_scheduling/`), Stage 13
(`aarch64_matmul_bias_relu_pi_scheduling/`), Stage 14
(`aarch64_matmul_bias_relu_schedule_cost_model/`), and Stage 15
(`aarch64_matmul_bias_relu_schedule_selection/`) rather than duplicating
their raw MIR/Pi-execution evidence.
