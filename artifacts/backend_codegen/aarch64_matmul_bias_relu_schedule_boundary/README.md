# Schedule-Unroll Boundary Search and Counterexample-Oriented Validation

**uk4 remained the measured winner across all tested Cortex-A76 stress
domains, including small, high-K, alternate-tile, rectangular, and
spilling configurations. This expands the measured support region but
does not establish universal optimality.**

No counterexample was found. This is reported as a valid result, not a
failure -- see "Outcome C" below for exactly what this does and does not
license claiming.

## Preserved historical truth

Unchanged from prior stages, restated per this stage's explicit
instruction not to rewrite history:
- Stage 12 predicted spilling candidates as regression risks (a correct
  reading of the static evidence available at the time).
- Stage 13 measured real hardware and contradicted the runtime-severity
  implication of that prediction for two candidates.
- Stage 14 separated backend cost from hardware profitability as two
  independent classification dimensions.
- Stage 15 connected calibrated selection to real MLIR/LLVM materialization.
- Stage 16 found uk4 fastest in 4 independent domains.
- **Stage 17 (this stage) searched deliberately for a domain where uk4
  loses, found none in 2 new stress domains, and is reporting that
  absence honestly rather than treating 6/6 as proof of universality.**

## 1-3. Stress domains evaluated and rationale

| Domain | Shape | Tile | Category | Why it could plausibly break the uk4 pattern |
|---|---|---|---|---|
| **smallA** (new) | 16x16x16 | 8x8x4 | A -- very small workload | Smallest problem with uk4 legally available (K trip=4, full collapse). Benefits: fewer K-loop branches/induction updates. Possible costs: fixed spill/reload overhead (real spills found: 2 @ uk2, 10 @ uk4), larger prologue/epilogue *fraction* of total work, code footprint (+26% at uk4) relative to genuinely tiny useful compute (16x16x16 = 4096 total FLOPs). |
| **highK** (new) | 32x32x128 | 8x8x8 | B -- larger K-loop trip count | K trip count 16 (double any domain tested before), uk4 = genuine partial unroll (4 remaining dynamic iterations, not a collapse). Benefits: same repeated loop-control reduction mechanism, larger scheduling region. Possible costs: longer live ranges across a bigger unrolled body, repeated spill traffic if spills occur. |
| **rect** (reused, Stage 16) | 32x64x32 | 8x8x8 | C -- highest register pressure | No larger validated tile currently exists beyond the 8x8x8-family (`{(8,8,8),(8,8,4),(4,8,8)}`) -- documented limitation, no new tiling implementation was added for this experiment. rect_uk4 (18 spills, 20 reloads, 368 stack bytes) remains the strongest pressure case in this project to date, reused here rather than duplicated. |
| **primary + rect** (reused) | 32x32x32 / 32x64x32 | 8x8x8 | D -- code-size/footprint stress | Largest static code-size growth observed (primary +62%, rect +61% at uk4). Treated as a stress **signal**, not a measured instruction-cache effect -- `perf` remains unavailable on this Pi (same limitation as Stages 13/16), so no hardware-counter evidence of real icache pressure exists. A dedicated multi-kernel rotation stress harness was NOT built this stage (see "Untested dimensions" below) -- an explicit, documented scope limitation, not an oversight. |

Candidate matrix: uk1/uk2/uk4 generated for both new domains (all 6
legal per the K-trip-divisibility rule); no candidate was rejected for
smallA/highK. Both new domains passed Stage 11 structural validation
before compilation.

## 4. Timing-quality safeguards (new this stage)

Empirically measured on this exact Pi 5 / `taskset -c 3` this session
(`tools/timer_overhead_probe.cpp`, 100k/50k samples): **clock-read
overhead (back-to-back `steady_clock::now()`) = 37ns median**;
malloc(4160)+free overhead (the generated kernel's own real sret-return
allocation, not harness overhead) = 37ns median. Clock resolution
verified genuinely 1ns via `clock_getres(CLOCK_MONOTONIC)` -- the
suspiciously-perfect zero-variance across `smallA`'s 5 measurement groups
is real determinism (a tiny, branch-free, always-cache-resident kernel on
an idle system), not clock quantization.

Quality thresholds applied to every new candidate (`timing_quality.json`):
`reliable` if median >= 370ns (10x clock overhead), `borderline` if
185-370ns, `unreliable` below that. Result: **5 of 6 new candidates
reliable; `smallA_uk4` (351ns) borderline** (10.5% overhead fraction).
Because clock overhead is approximately constant across uk1/uk2/uk4 for
the same domain (identical malloc call site/size), the *relative ranking*
within `smallA` remains defensible even though its absolute latency
numbers are not precise to more than ~1 significant figure -- stated
explicitly, not glossed over.

**Retroactive finding**: applying this same threshold to Stage 13's
smallest-ever measurement (`small_control_uk1`, 92ns) gives an overhead
fraction of 40% -- solidly "unreliable" by this stage's standard. That
measurement never fed an unroll-factor *comparison* (only uk1 was legal
for that shape/tile), so no prior ranking conclusion is invalidated, but
it is flagged here as a real, previously-uncaught measurement-rigor gap.

## 5-6. Structural and backend results

Both domains passed Stage 11 structural validation for all 3 factors.
Backend evidence (`backend_evidence.json` via `winner_summary.json`):

| Candidate | Spills | Reloads | Stack bytes | Object bytes | FMLA count |
|---|---|---|---|---|---|
| smallA_uk1 | 0 | 0 | 128 | 2336 | 64 |
| smallA_uk2 | 2 | 2 | 176 | 2640 | 128 |
| smallA_uk4 | **10** | **11** | 256 | 2952 | 256 |
| highK_uk1 | 0 | 0 | 96 | 2608 | 128 |
| highK_uk2 | 0 | 0 | 96 | 3248 | 256 |
| highK_uk4 | 0 | 0 | 96 | 4528 | 512 |

`smallA` has heavier spilling at uk4 (10/11) than any domain except
`rect` (18/20) despite being the *smallest* shape tested -- a genuinely
new, differentiated finding: smaller M/N trip counts appear to correlate
with *worse* spilling at full-K-collapse unroll, not better (fewer outer
iterations mean less amortization, and a K trip count of exactly 4 forces
uk4 into a full collapse where all 4 accumulator-chain steps are visible
to the register allocator simultaneously with no loop-carry boundary --
labeled **derived**, not directly measured, since no deeper MIR
live-range trace was run to confirm this specific mechanism).
`highK` is completely spill-free at all 3 factors -- also new: a K trip
count of 16 gives the allocator enough loop-carried reuse structure that
even the largest static body (512 FMLA) never spills.

## 7. Correctness

12/12 measurements (6 new candidates x deterministic+random-seed trials,
plus repeated-call stress) bit-exact (`max_abs_error = 0`), zero
repeated-call failures, thermal stable (48.3'C -> 48.8'C, no throttling).

## 8. Per-domain measured ranking

| Domain | uk1 median | uk2 median | uk4 median | uk4 vs uk1 | uk4 vs uk2 | CV | Classification |
|---|---|---|---|---|---|---|---|
| smallA | 500ns | 463ns | 351ns (borderline) | +29.8% | +24.2% | 0% (see timing-quality note) | **uk4 winner** |
| highK | 9.186ms | 8.222ms | 7.778ms | +15.3% | +5.4% | <0.4% | **uk4 winner** |

Both clear the variance-aware noise floor comfortably even accounting for
`smallA`'s timing-quality caveat (relative-ranking argument above).

## 9. Counterexample outcome: **C -- uk4 wins every domain tested**

Expanded tested region (6 domains total across Stages 16-17): 32x32x32,
64x64x64, 32x32x32/tile-8x8x4, 32x64x32, 16x16x16/tile-8x8x4,
32x32x128 -- covering shape scale from 4,096 to 262,144 total FLOPs, K
trip counts from 4 to 16, spill counts from 0 to 18, and both full and
partial K-loop collapse. **uk4 has not lost in any of the 6 domains
measured.** Universality remains unproven and is not claimed. uk4 is
**not** enabled globally; calibrated selection remains fully opt-in.

## 10. Boundary analysis

Discrete observations (6 data points -- no regression fit attempted,
matching the task's explicit anti-overfitting instruction):
- Every domain's uk4 candidate reduces the dynamic K-loop trip count by
  50-75% relative to uk1 -- the one static signal present in every win.
- Spill count at uk4 ranges from 0 (cube64, highK) to 18 (rect), with no
  observed shape/tile combination where spilling was severe enough to
  flip the measured ranking.
- The smallest tested problem (smallA, 4,096 FLOPs) still favored uk4
  despite having the second-worst spill count (10/11) of any candidate
  measured to date -- fixed prologue/epilogue and spill-setup overhead
  did NOT dominate at this scale, contrary to the a priori hypothesis
  that motivated testing it.
- The highest tested K-trip-count domain (highK, trip 16) was completely
  spill-free and showed the cleanest static-model agreement.

## 11. Static-model evaluation (6 domains)

Static soft-penalty ranking correctly predicted the measured winner in
**exactly 2 of 6 domains (33%): `cube64` and `highK` -- both, and only,
the domains where the measured winner (uk4) happens to be spill-free.**
Every domain where the winner has real spills (`primary`, `altk`, `rect`,
`smallA`) was mispredicted by static ranking. This is a clean, perfect
correlation, not merely "directionally useful" -- but weights were **not**
retuned to fit it. **A real bug was found and fixed while computing this
result**: an evidence-merge condition initially left `rect_uk1`/`rect_uk2`
with `object_bytes=None` (Stage 12 never analyzed `rect` at all), which
`compute_cost`'s UNSUPPORTED hard-rejection correctly flagged -- silently
disqualifying them from static ranking and making `rect_uk4` "win" by
default rather than by genuine evidence. Fixed by always fully rebuilding
both evidence categories when real static data is available on disk,
regardless of whether a blank placeholder record already existed.
Regression-tested. Conclusion, unchanged in substance from Stage 14-16:
**no currently collected static metric reliably predicts the measured
winner; exact-domain calibration remains necessary.**

## 12. Policy comparison

| Policy | Agreement (6 domains) | Notes |
|---|---|---|
| 1 -- exact calibration only | 6/6 | Uses real measured evidence per domain; falls back to static/conservative when absent (never triggered here, all 6 domains had exact evidence) |
| 2 -- bounded heuristic (cortex-a76 + tile in {8x8x8,8x8x4} + uk4 legal) | 6/6 | **Offline evaluation only -- not wired into the compiler driver** |
| 3 -- universal uk4 (deliberately unsafe) | 6/6 | **Evaluation only, never exposed as a production option** -- would silently select uk4 for any untested domain/target, exactly the failure mode this whole stage series exists to prevent |

Policies 2 and 3 achieve the same agreement as Policy 1 *only because* no
counterexample was found in the domains tested -- this is not evidence
that Policy 3 is safe to expose; it is evidence that Policy 1 (real
calibration) is the only policy that would have caught a counterexample
had one existed.

## 13. Updated calibration profile

`updated_multidomain_profile.json` extends Stage 16's 4-domain profile
with `smallA` and `highK`, each carrying full domain identity, semantic
CandidateKeys, correctness status, measured distributions, timing-quality
classification, and backend evidence -- kept as separate named domains,
never flattened into the existing bucket.

## 14. Integrated selector examples (all verified via object hash)

| Example | Query | Mode | Selected | Verification |
|---|---|---|---|---|
| 1 -- exact domain, uk4 selected | highK (32x32x128/8x8x8) | calibrated | uk4 | object hash identical to independently-compiled `highK_uk4.o` |
| 2 -- new boundary domain | smallA (16x16x16/8x8x4) | calibrated | uk4 (despite 10 spills) | object hash identical to independently-compiled `smallA_uk4.o` |
| 3 -- unsupported domain fallback | 16x16x16/tile 4x8x8 (never tested) | calibrated | uk1 (conservative fallback) | object hash identical to Stage 16's already-Pi-verified `example3_unknown.o` |

No 4th (counterexample) example -- none exists to demonstrate.

## Untested dimensions (explicit, per task section 18)

- No tile larger than 8x8x8 in any dimension has ever been structurally
  validated -- Category C register-pressure testing is bounded by this.
- No dedicated instruction-cache rotation stress harness was built;
  code-size is treated only as a static signal.
- No domain has been tested with an unroll factor other than {1, 2, 4}.
- No non-Cortex-A76 target, no non-f32 dtype, no non-8x8x8-family
  microkernel has ever been measured.
- Only 2 measurement sessions (5 interleaved groups each) back each new
  domain's ranking -- not independently reproduced across separate days
  or reboots.

## Files in this directory

See `commands.txt` for full reproduction. Key artifacts: `domain_design.json`
(rationale), `winner_summary.json` (per-domain table), `timing_quality.json`,
`static_model_evaluation.json`, `policy_comparison.json`,
`updated_multidomain_profile.json`, `cross_domain_rejection_report.json`
(30 pairs across 6 domains, all exact-match-rejected), `integrated_selection_examples/`.
References Stage 12-16 raw artifacts rather than duplicating them.
