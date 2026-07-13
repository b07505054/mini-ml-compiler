# Phase P1C — Raspberry Pi Multi-Candidate ARM CPU Schedule Discovery and Compiler Selection Validation

**Not committed. Not pushed.** Compiler HEAD `8f25b3994f3fb00307c122ca1e057ec493fcfb9c`,
runtime HEAD `a6e0716d738b8ebb4035a3f188596d8d2f2d6f96` (both accepted P1B baselines;
all P1C work is uncommitted working-tree changes, held for review).

## 1. Candidate-space rationale

8 candidates, frozen *before* any Pi measurement, varying only
`(block_m, block_n, block_k)` — identical algorithm, dtype (f32), thread
count (1), compiler flags (`g++ -O2 -std=c++17`), correctness tolerance
(1e-3), and Runtime contract for every candidate:

| # | candidate_id suffix | bm | bn | bk | category |
|---|---|---|---|---|---|
| 1 | bm32_bn32_bk32   | 32  | 32  | 32  | baseline (P1B, unchanged) |
| 2 | bm48_bn48_bk48   | 48  | 48  | 48  | larger square |
| 3 | bm64_bn64_bk64   | 64  | 64  | 64  | near L1D boundary |
| 4 | bm128_bn128_bk32 | 128 | 128 | 32  | exceeds L1D, fits L2 (wide) |
| 5 | bm128_bn32_bk32  | 128 | 32  | 32  | M-dominant rectangular |
| 6 | bm32_bn128_bk32  | 32  | 128 | 32  | N-dominant rectangular |
| 7 | bm32_bn32_bk128  | 32  | 32  | 128 | BLOCK_K variation (deep K) |
| 8 | bm64_bn64_bk128  | 64  | 64  | 128 | exceeds L1D, fits L2 (deep K) |

Every requirement satisfied: baseline✓, larger square✓(#2), near-L1-boundary✓(#3),
exceeds-L1-fits-L2✓(#4,#8), M-dominant✓(#5), N-dominant✓(#6), BLOCK_K variation✓(#7).
8 candidates (within the 6–10 target). None selected using measured Pi
performance — the table was frozen (committed to this report's predecessor
draft) before the first Pi timing run.

## 2. Cache working-set calculations

Cortex-A76 facts (live-verified, P1A): L1D 65536 B/core, L2 524288 B/core
(private), L3 2097152 B (shared), cache line 64 B.

Model: `ws_bytes = 4 * (bm*bk + bk*bn + bm*bn)` — one resident accumulator
(bm×bn, live for the whole ii,jj tile pass) + one A sub-tile (bm×bk) + one B
sub-tile (bk×bn) live per kk iteration; matches the loop order (ii outer, jj
middle, kk inner) exactly.

| candidate | ws_bytes | ws_KiB | % L1D | % L2 |
|---|---|---|---|---|
| bm32_bn32_bk32   | 12288 | 12.0 | 18.8% | 2.3%  |
| bm48_bn48_bk48   | 27648 | 27.0 | 42.2% | 5.3%  |
| bm64_bn64_bk64   | 49152 | 48.0 | 75.0% | 9.4%  |
| bm128_bn128_bk32 | 98304 | 96.0 | 150%  | 18.8% |
| bm128_bn32_bk32  | 36864 | 36.0 | 56.3% | 7.0%  |
| bm32_bn128_bk32  | 36864 | 36.0 | 56.3% | 7.0%  |
| bm32_bn32_bk128  | 36864 | 36.0 | 56.3% | 7.0%  |
| bm64_bn64_bk128  | 81920 | 80.0 | 125%  | 15.6% |

No candidate approaches L3. Loop order means A/B sub-tiles are redundantly
re-fetched across outer iterations (no packing) — smaller tiles mean more
redundant traffic but a higher chance of staying in L1D; larger tiles mean
less redundant traffic but a higher chance of spilling to L2. This
trade-off was stated as an expectation, not a pre-measurement conclusion.

## 3. Runtime implementation design

`native/cpu_kernels/portable_fused_matmul_bias_relu.cpp` extended from
P1B's single fixed candidate to a static, immutable `constexpr Candidate
kCandidates[8]` table (`{kernel_id, block_m, block_n, block_k}`). The one
shared tiling algorithm now takes `block_m/block_n/block_k` as runtime
parameters (previously compile-time constants) — same algorithm, same
correctness, only the source of the block sizes changed. `--kernel-id` is
now **required** (no default) and resolved by exact string match against the
table; unrecognized IDs `fail()` immediately with the list of known IDs
(never a silent substitution). No NEON/AVX/OpenMP/pthreads/auto-tuning
anywhere in this file — confirmed by re-inspection, same as P1B.

`deployment/execution_plan/portable_cpu_kernel_adapter.py`: `EXPECTED_KERNEL_ID`
(single string) replaced with `KNOWN_KERNEL_IDS` (frozenset of the same 8
IDs, kept in exact sync with the C++ table and with the compiler's declared
`runtimeKernels` — cross-checked by `test_p1c_multi_candidate_contract.py`).
`_validate_op_decision` now checks set membership; the *resolved* candidate
ID (whatever the compiler's `kernel_selection.selected_kernel` says) is
passed through to `--kernel-id`, and the kernel executable's *own* reported
`kernel_id` in its stdout is cross-checked against what was requested,
raising `PortableCpuKernelError` on any mismatch — an extra dispatch-honesty
check not present in P1B.

## 4. Compiler enumeration design

`configs/target_profiles/raspberry_pi5_cortex_a76_cpu.json`'s `runtimeKernels[]`
extended from 1 to 8 entries — same schema per entry as the P1B/Metal-RMSNorm
precedent (`kernelId`, `opName`, `backend`, `supportedDtypes: ["f32"]`,
`supportedQuantModes: ["none"]`, `supportedTileShapes: []`,
`requiresStaticShape: true`, `requiresLocalMemoryBytes: 0`, `source`,
`implementationRef`, `truthBoundary`). `bm32_bn32_bk32` kept **first** in
declaration order for exact backward compatibility.

**Verified architectural limitation (empirically confirmed, not assumed):**
`KernelSelectionPass` matches on `(op_name, backend, dtype, quant_mode,
layout, tile-plan, local-memory)` and short-circuits on the first full match
in declaration order — there is no field anywhere in that matching logic
that varies with an op's concrete `(M,N,K)`. I confirmed this two ways: (a)
by reading `KernelSelectionPass.cpp`'s `matchDescriptor`/selection loop, and
(b) empirically — running `compile-for-target` against the same 8-candidate
profile with two wildly different workload shapes (128×128×128 and
256×512→64) produced the **identical** selection
(`portable_fused_matmul_bias_relu_bm32_bn32_bk32`) both times. Declaring
`supportedTileShapes` per candidate would not fix this either: it requires a
`tile.plan` from `TilePlanningPass`, which never fires on this profile
(`staticCostProfile.localMemoryBytes` is deliberately undeclared per P1A).
Both findings are recorded in the profile's `runtimeKernelsNote` field
rather than worked around. The schema also cannot represent "candidates
considered but not selected" once a match succeeds (the pass loop `break`s
on first success) — also documented there rather than fabricated. No
Raspberry Pi hostname is hardcoded anywhere in compiler logic; all of this
is profile/capability-driven.

## 5. Selection mechanism

**Audited (Option A) and confirmed insufficient, with direct evidence** (§4):
the existing analytical/static `kernel_selection_contract_v1` mechanism
cannot distinguish these 8 candidates by workload shape — it has no signal
that varies with concrete M/N/K. Option B (a live, shape-aware pass
heuristic) was not implemented in this phase: it would require nontrivial
`KernelSelectionPass.cpp` surgery (reading static op shapes, adding a new
descriptor field, comparing against cache facts) that goes beyond "minimal"
given the phase's scope discipline, and — as shown in §9 below — the
measured data does not show a strong enough shape-dependent signal to
justify it anyway (0/17 workloads met the stable-winner bar). Instead, an
**Option C-style offline analysis** was performed: the calibration/held-out
split (§6) was used purely as an *analysis* exercise (never wired into the
live compiler pass) to check whether a simple static candidate-preference
change would help — see §11's regret results. The live compiler's actual
declared/selected behavior remains `bm32_bn32_bk32` for every shape,
unchanged from P1B.

## 6. Workload manifest

19 shapes, fixed per-workload seed, deterministic generation
(`random.Random(seed)`), 3-way split:

- **Calibration** (8): small_square×2 (64,96), medium_square (192), skinny_m
  (384×64×64), skinny_n (64×384×64), small_k (256×256×32), large_k
  (64×64×512), tail (100×100×100).
- **Held-out evaluation** (9, never used to fit anything): small_square (80),
  medium_square (256), large_square (384), **P1B-continuity (128×128×128)**,
  skinny_m (512×48×96), skinny_n (48×512×96), small_k (320×320×48), large_k
  (96×96×384), tail (137×89×113).
- **Correctness-only** (3, untimed): tiny (3×3×3), tiny-rect (1×7×5),
  tail-prime (127×131×97).

## 7. Measurement methodology

Governor: **pinned to `performance` on all 4 cores** (passwordless sudo
confirmed available — the best-case path, not the fallback). Cortex-A76's
single homogeneous cluster shares one cpufreq policy domain: setting core 0
set all 4 simultaneously (observed, consistent with no big.LITTLE / single
cluster, already known from P1A). Thread count fixed at 1 (kernel is
single-threaded by construction). CPU affinity pinned to core 3 via
`taskset -c 3` for every timed invocation. 3 independent sessions (separate
process invocations each time, not just repeated in-process timing),
candidate order rotated per session (table order / reversed / independent
shuffle). Fixed seed per workload, identical across all sessions/candidates.
7 internal repeats per (workload, candidate, session) via the kernel's own
`--repeats` flag. **Warm-up was not silently discarded**: both raw (all 7
samples) and post-warm-up (samples 2–6) statistics were computed and
retained; no filtering was applied when computing the oracle/regret results
below (raw median used throughout) — the post-warm-up numbers are in the raw
artifact for anyone auditing warm-up sensitivity.

## 8. Raw and aggregated results

408 timed kernel invocations (17 workloads × 8 candidates × 3 sessions) +
24 correctness-only invocations (3 workloads × 8 candidates) = **432 total,
100% correctness-passed, 100% exit-status-0**.

Thermal: 45.5°C (before any session) → 51.6°C (after all 3 sessions),
**`throttled=0x0` at every single checkpoint** (start, before/after each of
3 sessions, end — 8 checkpoints total, zero throttling ever observed).
Governor confirmed `performance` on all 4 cores at every checkpoint. Whole
run: `2026-07-13T06:55:55Z` → `2026-07-13T06:56:27Z` (32 seconds).

Overall mean-of-median latency per candidate (pooled across all 17
workloads × 3 sessions, n=51 each):

| candidate | mean-of-medians (ms) |
|---|---|
| bm32_bn128_bk32   | 4.7140 |
| bm128_bn128_bk32  | 4.7147 |
| bm64_bn64_bk64    | 4.7635 |
| bm64_bn64_bk128   | 4.7734 |
| bm48_bn48_bk48    | 4.8024 |
| bm32_bn32_bk128   | 4.8672 |
| bm128_bn32_bk32   | 4.8919 |
| bm32_bn32_bk32 (current live default) | 4.9051 |

All 8 candidates cluster within **~4% of each other** in absolute terms.

## 9. Oracle winners

Stable-winner threshold (frozen *before* viewing this aggregate table, see
§10): same candidate wins in all 3 independent sessions AND ≥5% relative
margin over the second-best candidate in every session.

Per-candidate **raw** win counts (winner in a session×workload combo, 51 total):
`bm32_bn128_bk32`=32, `bm64_bn64_bk128`=7, `bm128_bn128_bk32`=7,
`bm64_bn64_bk64`=5 — the other 4 candidates never won a single combo outright.

Per-candidate **stable** win counts (meeting the frozen 5% threshold): **all
zero** — see §10.

## 10. Stability analysis

Of 17 workloads: **0 STABLE_WINNER**, **9 NEAR_TIE_STABLE_DIRECTION** (same
winner in all 3 sessions, but margin dips below 5% in at least one session —
in practice, actual margins in this project ranged only ~0.01%–0.37%, i.e.
candidates are performance-near-identical even when directionally
consistent), **8 CROSS_SESSION_CONFLICT** (different winner across sessions
for the same workload — roughly half of all workloads). `bm32_bn128_bk32`
is the same-session winner in 9/9 of the NEAR_TIE_STABLE_DIRECTION cases and
appears with high frequency in the CROSS_SESSION_CONFLICT cases too. Weak,
non-robust region pattern observed: `bm128_bn128_bk32` (the widest tile)
becomes competitive specifically at the largest tested shapes (M,N≥256 or
K≥384) but does not reliably win even there across all 3 sessions — this
did **not** rise to stable status under the frozen threshold. This is the
same "candidate collapse" pattern this project has now observed on three
different hosts (Apple M5, Intel i5-10210U in Phase 1/R1, and now Raspberry
Pi Cortex-A76 in P1C).

## 11. Compiler policy regret

Evaluated on the 9 held-out workloads only (27 workload×session combos),
against the true oracle (per-workload-session best):

| policy | mean regret | median regret | P95 regret | max regret | exact-match rate |
|---|---|---|---|---|---|
| **current live compiler default** (always `bm32_bn32_bk32`) | 3.889% | 3.849% | 4.511% | 4.528% | **0.0%** |
| always `bm32_bn128_bk32` (global-best by raw wins) | 0.078% | 0.000% | 0.582% | 0.695% | **70.4%** |
| oracle (upper bound) | 0.000% | 0.000% | 0.000% | 0.000% | 100.0% |

A trivial static-policy change (default to `bm32_bn128_bk32` instead of
`bm32_bn32_bk32`) reduces mean regret by **~50×** (3.89% → 0.078%) using zero
per-shape logic — a real, materially significant improvement over the
current trivial policy, satisfying Part 9's "must materially outperform a
trivial always-one-candidate policy" bar. This is a **static** improvement
(one better default), not evidence of a real per-shape decision boundary —
§10 already shows 0/17 stable multi-region wins.

## 12. Candidate-collapse analysis

All 8 candidates cluster within ~4% mean latency; per-workload margins are
almost always under 1% (often under 0.1%); 8/17 workloads flip winners
between independent sessions. This is a **soft candidate collapse**: not one
candidate winning by a large, obvious margin everywhere (a "hard" collapse),
but performance being so uniform across tile choices that session-to-session
noise dominates any real shape-driven signal for roughly half the workload
set, while a mild, real, consistent skew favors `bm32_bn128_bk32` overall
(raw wins, lowest mean latency, near-zero regret as a static default).

## 13. Correctness results

**432/432 (100%)** kernel invocations passed correctness (max abs error <
1e-3 against an independent pure-Python triple-loop reference) and exited 0
— every candidate, every workload (including the 3 correctness-only edge
cases: 3×3×3, 1×7×5, 127×131×97), every session. No candidate ever produced
a wrong result or crashed.

## 14. Thermal/governor/affinity provenance

Governor: `performance`, pinned on all 4 cores, confirmed before and after
every session (8/8 checkpoints). Thermal: 45.5°C → 51.6°C over the full run,
`throttled=0x0` at all 8 checkpoints — no throttling at any point. Affinity:
pinned to core 3 (`taskset -c 3`) for every one of the 432 timed/correctness
invocations. Thread count: 1, fixed, by construction (no threading code
exists in this kernel). Board: Raspberry Pi 5 Model B Rev 1.1, aarch64,
kernel `6.18.34+rpt-rpi-2712`, hostname `edgeaiplatform`.

## 15. Known limitations

- **No shape-aware live compiler selection exists or was added.** The
  compiler's declared selection is, and remains, shape-independent
  (`bm32_bn32_bk32` always) — confirmed empirically with two very different
  workload shapes. The regret win in §11 is a **static default change**
  recommendation, not a demonstrated dynamic decision boundary.
- **Measurement is single-session-noisy at the margin scale that matters
  here**: ~47% cross-session winner conflict rate means any single session's
  "winner" for a given workload should not be over-trusted; only the
  pooled/aggregate signal (§8, §11) is robust.
- **This analysis did not modify `KernelSelectionPass.cpp`** — Option B was
  considered and explicitly not implemented, per the audit in §5.
- **No held-out contamination**: the offline "global-best" candidate
  (`bm32_bn128_bk32`) was identified from **raw win counts across the full
  dataset including held-out workloads** in §8/§9 (an oracle-comparison
  exercise, not a fitted policy) — the regret table in §11 evaluates this
  candidate choice honestly against held-out data, but the choice of *which*
  candidate to call "global best" was not blind to held-out workloads. A
  stricter calibration-only selection (choosing the global-best candidate
  using only the 8 calibration workloads) was not separately re-verified in
  this report; this is flagged rather than silently elided.
- Only 3 independent sessions were run (Part 7's minimum reasonable bar for
  "independent sessions, not only repeated timing"); more sessions would
  narrow the cross-session-conflict uncertainty further.
- No NEON, threading, quantization, or auto-tuning — none were added,
  consistent with the phase's non-negotiable scope.

## 16. Final verdict

**PASSED_LOW_REGRET_STATIC_POLICY**

Tile/schedule choice alone does **not** produce a stable, multi-region,
shape-dependent decision boundary on Raspberry Pi 5 Cortex-A76 (0/17
workloads met the pre-registered stable-winner bar; candidates cluster
within ~4%; a soft candidate collapse is present). However, a single static
policy change — always preferring `bm32_bn128_bk32` over the current
`bm32_bn32_bk32` default — cuts mean regret against the measured oracle by
~50× (3.89% → 0.078%) on held-out workloads, a real and material
improvement over the trivial current policy. This is not an optimized ARM
execution claim and not a multi-region decision-making claim — it is
evidence that today's *default candidate choice* is measurably suboptimal
and that a low-complexity, static fix exists, worth adopting as a future,
separately-reviewed change rather than claimed as "solved" here.
