# Raspberry Pi Correctness and Controlled Runtime Validation (Machine-Scheduling Slice)

Target: Raspberry Pi 5 Model B Rev 1.1, real hardware (`allen@100.110.37.6`)
Purpose: validate whether the Stage 12 static/backend classifications
(Class A "scheduling win likely" for `schedule-unroll-k=2` at the primary
tile, Class D "regression risk" for the two spilling diagnostics)
correspond to real, measured, executable behavior.

## Result in one line

**Every Stage 12 Class A prediction was confirmed on real hardware across
5 independently-tested shapes** (8.6%-24.0% real median-latency
improvement, all clearing a variance-aware noise floor), **and both
Stage 12 Class D (spilling) diagnostics were CONTRADICTED** -- both
spilling candidates measured FASTER than their matched, non-spilling
baselines. Static spill evidence predicted a runtime cost that did not
materialize at these problem sizes. This is reported as-is, not forced to
match Stage 12's expectation (see "Spill-prediction validation" below).

## Truth boundary

Same boundary as every prior slice: the project owns candidate selection,
harness construction, measurement methodology, and analysis. **The Pi's
CPU, OS scheduler, and the compiled kernel itself own every measured
nanosecond** -- no project code fabricates or adjusts a timing number.
Every latency figure here came from `std::chrono::steady_clock` wrapping a
real call into a real compiled AArch64 object executing on a real
Cortex-A76 core.

## Why a new harness template (not the existing 5-way tiled_harness)

`aarch64_matmul_bias_relu_tiled_harness.cpp` links multiple *variants* of
the same shape into one binary, which works because each variant's HIR
fixture produces a distinct exported symbol. The tiled-scheduled variant
at different `--schedule-unroll-k` values reuses the *same* HIR fixture and
therefore exports the *identical* symbol regardless of unroll factor -- two
such objects cannot be linked into one binary (duplicate-symbol error).
`aarch64_matmul_bias_relu_schedule_harness.cpp.template` is generated
per-shape (`mlir_passes/tools/generate_schedule_harness.sh`, mirroring the
existing `generate_scheduled_transform.sh` convention) and compiled once
per (shape, tile, schedule-unroll-k) candidate, one object per binary --
candidates are compared by running separate processes, the same
"process isolation" principle this project has used since the very first
Pi-integration slice, extended one level further.

Reuses, verbatim: the scalar reference implementation, deterministic LCG
input fill, `MemRef2D` ABI, and guarded-buffer/bit-exact-sentinel
correctness methodology from the existing `aarch64_matmul_bias_relu_tiled_harness.cpp`
and `aarch64_matmul_bias_relu_repeated_call_test.cpp`. Tolerance policy
unchanged: `max_abs_error < 1e-3`.

## Pi environment

| | |
|---|---|
| Model | Raspberry Pi 5 Model B Rev 1.1 |
| CPU | 4x Cortex-A76 |
| OS | Debian GNU/Linux 13 (trixie) |
| Kernel | 6.18.34+rpt-rpi-2712 |
| Governor | `performance` (already set; not changed by this script) |
| CPU frequency | 2400 MHz (max) throughout |
| Temperature | 48.8'C before -> 49.9'C after (stable, no meaningful drift) |
| Throttling | `throttled=0x0` before and after (never throttled) |
| Core affinity | `taskset -c 3` for every benchmark invocation (one consistent core) |
| `perf` | **NOT installed.** Not installed by this script -- installing new system packages on shared hardware without explicit authorization is out of scope. Hardware-counter evidence (cycles/IPC/branch-misses/cache-misses) is therefore unavailable this slice; stated explicitly rather than omitted silently. |
| g++ | 14.2.0 (Debian) |
| Target | `aarch64-linux-gnu`, `-mcpu=cortex-a76`, `-O2` |

Full raw command output: `environment_raw.txt`. Structured: `environment.json`.

## Candidate matrix (14 candidates: 11 Group A + 3 Group B)

| Candidate | Shape | Tile | uk | Object bytes | Correctness | Groups | Median-of-medians (ms) | CV |
|---|---|---|---|---|---|---|---|---|
| small_control_uk1 | 8x8x8 | 4x8x8 | 1 | 1696 | PASS | 1 | 0.000092 | 0.0 |
| cube16_uk1 | 16x16x16 | 8x8x8 | 1 | 2624 | PASS | 1 | 0.000463 | 0.0 |
| cube16_uk2 | 16x16x16 | 8x8x8 | 2 | 2968 | PASS | 1 | 0.000352 | 0.0 |
| **primary_uk1** | 32x32x32 | 8x8x8 | 1 | 2608 | PASS | 5 | 0.002593 | 0.0033 |
| **primary_uk2** | 32x32x32 | 8x8x8 | 2 | 3248 | PASS | 5 | 0.002352 | 0.0042 |
| cube64_uk1 | 64x64x64 | 8x8x8 | 1 | 2704 | PASS | 5 | 0.019760 | 0.0021 |
| cube64_uk2 | 64x64x64 | 8x8x8 | 2 | 3344 | PASS | 5 | 0.017685 | 0.0007 |
| rect_uk1 | 32x64x32 | 8x8x8 | 1 | 2704 | PASS | 1 | 0.005111 | 0.0 |
| rect_uk2 | 32x64x32 | 8x8x8 | 2 | 3344 | PASS | 1 | 0.004611 | 0.0 |
| large_uk1 | 128x128x128 | 8x8x8 | 1 | 2720 | PASS | 1 | 0.159074 | 0.0 |
| large_uk2 | 128x128x128 | 8x8x8 | 2 | 3360 | PASS | 1 | 0.142981 | 0.0 |
| diag_full_unroll_uk4 (Group B) | 32x32x32 | 8x8x8 | 4 | 4232 | PASS | 5 | 0.002074 | 0.0 |
| diag_alt_ktile_uk1 (Group B) | 32x32x32 | 8x8x4 | 1 | 2336 | PASS | 1 | 0.002963 | 0.0 |
| diag_alt_ktile_uk2 (Group B) | 32x32x32 | 8x8x4 | 2 | 2640 | PASS | 1 | 0.002611 | 0.0 |

**14/14 candidates correct.** Zero repeated-call failures, zero guard-buffer
corruption, zero descriptor-sanity failures, across 8x8x8 through
128x128x128, tile-4x8x8/8x8x8/8x8x4, and schedule-unroll-k 1/2/4.

`small_control_uk1` exercises the trip-count-one canonicalization case (N
and K loops both collapse, per Stage 11); `cube16_uk2` independently
exercises full K-loop collapse at a *different* shape than Stage 12's
dedicated `diag_full_unroll_uk4` diagnostic (16x16x16's K trip count is
2, so unroll-k=2 collapses it -- a second, unplanned confirmation that the
collapsed-loop path is correct).

## Numerical correctness detail

Every candidate ran two correctness trials (deterministic seed `0x1234`,
random seed `0x5eed`) plus 200-500 repeated same-process invocations with
guarded input buffers (256-byte sentinel prefix/suffix on every buffer,
bit-exact sentinel comparison after every call). All trials: `max_abs_error
= 0` (bit-exact vs. the scalar reference, not merely within tolerance) for
every candidate tested. Zero guard corruption, zero repeated-call
failures, zero output-descriptor sanity failures anywhere in the matrix.

## Baseline vs. uk2 output-order comparison (primary candidate)

`primary_uk1` and `primary_uk2` were run with the identical deterministic
input seed and their raw output buffers dumped to disk.

```
baseline (uk1)  sha256: 6859fa73f405f76caeecdb0245ef8e02f812f10d99f0289c3bb89363602c6406
scheduled (uk2) sha256: 6859fa73f405f76caeecdb0245ef8e02f812f10d99f0289c3bb89363602c6406
bitwise identical: true
```

**Bitwise identical.** Combined with Stage 12's finding of zero fast-math
flags anywhere in the LLVM IR, this is direct runtime confirmation --  not
just a static/MIR-level inference -- that both variants execute the exact
same floating-point reduction order. This directly answers task section 5:
the runtime result is consistent with Stage 12's serial-accumulator-chain
finding, checked, not assumed.

## Primary performance result (Class A, confirmed)

All five Group A comparisons (matched: identical shape, tile, target CPU,
optimization level, iteration/warmup policy, and git revision -- verified
by an automated guard, `assert_matched_comparison`, that raises
`MismatchedComparisonError` on any mismatch) show a real, reproducible,
noise-floor-clearing improvement:

| Comparison | Speedup | % improvement | Runtime class |
|---|---|---|---|
| cube16 (uk1 vs uk2, full collapse) | 1.32x | 23.97% | **A** |
| **primary (uk1 vs uk2)** | 1.09x | 8.60%-9.29%* | **A** |
| cube64 (uk1 vs uk2) | 1.12x | 10.50% | **A** |
| rect 32x64x32 (uk1 vs uk2) | 1.11x | 9.78% | **A** |
| large 128x128x128 (uk1 vs uk2) | 1.11x | 10.12% | **A** |

*two independent runs of the full matrix produced 8.60% and 9.29% for the
primary comparison -- both comfortably clear the noise floor; reported
range rather than a single run's number for honesty about run-to-run
variance.

Every comparison's noise floor is `max(3%, 2x the larger side's
coefficient of variation)` -- a flat 3% floor alone could not be gamed by
a single noisy measurement, since a high-CV candidate widens its own gate
(unit-tested: `test_high_variance_widens_noise_floor_to_avoid_spurious_win`).
All five results clear even this adaptive floor by 2x-8x.

## Runtime variability and measurement quality

- `primary` and `cube64` used 5 *interleaved* measurement groups
  (baseline-group-0, scheduled-group-0, baseline-group-1, scheduled-group-1,
  ...) specifically to avoid systematic bias from thermal drift or OS
  jitter trending across a run -- not two back-to-back blocks.
- Cross-group coefficient of variation was low throughout (0.07%-0.42%
  for the full-rigor candidates), indicating a genuinely stable
  measurement, not noise the classifier happened to clear by luck.
- Single-group candidates (cube16, rect, large, and the Group B
  diagnostics) show `cv=0.0` because `min_ms` happened to equal `median_ms`
  within the harness's floating-point formatting precision in a single
  group -- consistent with the very short (sub-microsecond to
  sub-millisecond) kernel latencies and a `performance`-governor, idle,
  unthrottled Pi.

## Hardware-counter results

**Not collected.** `perf` is not installed on this Pi and this script does
not install new system packages on shared hardware without explicit
authorization (see Pi environment table). Runtime attribution (task
section 8) is therefore limited to: measured wall-clock latency (above),
static code size / object bytes (candidate matrix above), and the
already-established Stage 12 static evidence (dynamic K-loop iteration
count reduction, spill/reload counts, stack-frame size) -- cross-referenced
against measured latency in the spill-prediction table below, not
conflated with a direct cache/IPC measurement.

## Spill-prediction validation (Group B diagnostics)

| Candidate | Stage 12 spills/reloads | Stage 12 stack bytes | Matched baseline median (ms) | Diagnostic median (ms) | Relative latency | Correctness | Outcome |
|---|---|---|---|---|---|---|---|
| diag_full_unroll_uk4 | 11 / 12 | 224 | 0.002593 (primary_uk1) | 0.002074 | **+20.02% faster** | PASS | **Contradicted** |
| diag_alt_ktile_uk2 | 2 / 2 | 176 | 0.002963 (diag_alt_ktile_uk1) | 0.002611 | **+11.88% faster** | PASS | **Contradicted** |

**Both Stage 12 Class D diagnostics were faster than their matched,
spill-free baselines, not slower.** This is reported honestly as
"contradicted" per the task's own defined outcome taxonomy -- the result
was not forced to match Stage 12's static expectation. A plausible (not
verified, since `perf` is unavailable) explanation: the extra spill/reload
traffic in both cases is a small, fixed number of stack accesses (11-12
and 2-2) that almost certainly hit L1 data cache given the tiny working
set (a handful of 16-byte vector spill slots within a function whose
entire live state fits comfortably in L1), while the *dominant* effect at
these problem sizes is still the reduced dynamic K-loop trip count from
unrolling -- the same mechanism responsible for every Group A win. **The
practical implication: static spill evidence alone was not predictive of
a runtime regression for this workload class on this CPU** -- a genuine,
non-obvious finding this validation step exists to surface, not suppress.

## Static (Stage 12) vs. measured (Stage 13) correlation

| Stage 12 static classification | Stage 13 measured outcome | Agreement |
|---|---|---|
| Class A (primary, 8x8x8 tile, uk1->uk2) | Confirmed runtime win, +8.6-9.3% | **Agrees** |
| Class A (cube64, confirmation shape) | Confirmed runtime win, +10.5% | **Agrees** |
| Class D (full K-unroll, uk4, 11 spills) | Measured **faster**, +20.0% | **Contradicts spill-severity prediction** |
| Class D (alt K-tile, uk2, 2 spills) | Measured **faster**, +11.9% | **Contradicts spill-severity prediction** |

Stage 12's zero-spill/no-pressure-increase evidence for the primary and
cube64 candidates was a correct predictor of a real runtime win. Stage
12's spill evidence, however, did NOT correctly predict a runtime
regression for either Group B diagnostic at this problem size -- the
reduced-loop-overhead benefit of extra unrolling outweighed the spill cost
in both cases tested. **Conclusion for Stage 14: static spill/register
evidence remains a valid SAFETY gate (reject anything with genuinely
unbounded or excessive spilling) but is not, by itself, a reliable
performance predictor at this problem-size class -- real hardware
measurement remains necessary before any performance claim, exactly as
this stage set out to establish.**

## Performance-gate classification

Primary `schedule-unroll-k=2` candidate (tile 8x8x8):
- All correctness tests pass (14/14 candidates, this candidate included): **PASS**
- No memory corruption or repeated-call failures: **PASS**
- No new spills relative to Stage 12 evidence (both primary_uk1 and
  primary_uk2 report 0 spills in Stage 12): **PASS**
- Median latency improved (not merely "not worse"): **PASS**
- Stable across repeated groups (CV 0.33%-0.42% across 5 interleaved
  groups): **PASS**

**Classification: A -- Confirmed runtime win.** The primary uk2 candidate
advances to Stage 14.

## Bugs found and fixed while building this validation

1. Initial candidate loop compiled+built+ran each candidate to completion
   before starting the next, so a "5 measurement groups" pair actually ran
   as two back-to-back blocks (uk1 x5, then uk2 x5) rather than truly
   interleaved -- defeating the point of interleaving (bias from any
   trend across the run would land almost entirely on whichever candidate
   ran second). Restructured into a build phase followed by a dedicated
   interleaved run phase driven by the comparison-pair list.
2. The stale-object identity check was originally inline inside the
   Pi-transfer function (real SSH/SCP calls), making it untestable in
   isolation. Factored into a pure `verify_object_identity()` function
   with its own `StaleArtifactError`, unit-tested for match/mismatch/
   missing-checksum cases.
3. The diagnostic spill-validation table did not originally run the same
   `assert_matched_comparison` guard the main comparison table uses --
   added for consistency (verified after the fact that both diagnostic
   pairs were, in fact, correctly matched; the guard is now enforced going
   forward rather than relying on manual verification).

## Files in this directory

- `README.md` -- this file
- `summary.json` -- concise machine-readable summary (candidate matrix,
  comparisons, spill-prediction table, output-order comparison)
- `pi_validation_results.json` -- full raw evidence: every harness JSON
  output, every measurement group, full Pi environment capture, Stage 12
  cross-references
- `environment.json` / `environment_raw.txt` -- Pi environment capture
- `thermal_snapshots.jsonl` -- temperature/throttle state before and after
  the full run
- `commands.txt` -- every command used to produce this directory
- `compiled/` -- per-candidate LLVM-dialect MLIR, LLVM IR, assembly,
  object files (dev-host compilation output)
- `candidates/<label>/manifest.json` -- per-candidate self-contained
  record (identity, build info, measurement groups, Stage 12
  cross-reference)

This directory does NOT duplicate Stage 12's 6.8MB of MIR/register/
scheduling evidence -- see `artifacts/backend_codegen/aarch64_matmul_bias_relu_scheduling/`
for that (referenced here via each candidate's `stage12_key` /
`stage12_evidence` field).
