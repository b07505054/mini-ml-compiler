# Phase P1D — Raspberry Pi ARM CPU Thread-Decomposition Foundation and Multi-Thread Decision Study

**Not committed. Not pushed.** Base commits: compiler `1ebc233b90f0fac152f1aaf666933c9bc7be16b3`,
runtime `eabfaf0f46b43e7399ecc440289328860c5fcd1b` (both accepted P1C.1 baselines;
all P1D work below is uncommitted working-tree changes, held for review).

Truth-boundary key used throughout: **[CODE]** = verified by reading/building
the actual source; **[LIVE]** = verified live on real Pi hardware this
session; **[BENCH]** = benchmark/measurement-derived evidence; **[INFER]** =
reasoned conclusion from the above; **[UNKNOWN]** = not established.

## 1. Baseline reproduction

**[CODE]/[LIVE]** Compiler HEAD and runtime HEAD matched the accepted hashes
exactly; both repos showed `up to date with origin`, clean working trees, no
untracked binary present. Rebuilt `compile-for-target` and re-ran it against
the accepted profile — selected kernel confirmed
`portable_fused_matmul_bias_relu_bm32_bn128_bk32` (P1C.1 default, unchanged).
Re-ran one real Raspberry Pi correctness execution (M=N=K=128, 1 thread):
correctness passed (max abs error 2.47e-05), `thread_count` self-reported as
1, median latency 1.571 ms. Full provenance recorded live: compiler commit
`1ebc233b...`, runtime commit `eabfaf0f...`, target profile ID
`raspberry-pi5-cortex-a76-cpu`, hostname `edgeaiplatform`, board
`Raspberry Pi 5 Model B Rev 1.1`, architecture `aarch64`, physical core count
`4` (`nproc`), governor `performance` on all 4 cores, temperature 48.8°C,
`throttled=0x0`. Baseline reproduced exactly; work proceeded.

## 2. Schema audit

**[CODE]** `DispatchUnit` (Phase 26, `ExecutionPlan.h`) is an unrelated
CV-full-graph routing structure with zero thread fields — not used by the
`fused_matmul_bias_relu` op path at all. The real per-op decision structure
our op path uses is `PerOpDecisionBundle`, which already carries
`kernel_selection` (`KernelSelection`: `status`, `selected_kernel_id`,
`source`, `rejection_reasons`, `contract_version`, `truth_boundary`) —
confirmed via full read of `Decision.h`/`ExecutionPlan.h`: **no thread field
existed anywhere in the schema before this phase.**

Implementation identity (kernel/tile) lives in `KernelSelection`. Decided
(and implemented) that thread information belongs in a **new, separate
schedule object** (`ThreadSchedule`), not bolted onto `KernelSelection`
(different, orthogonal decision: which kernel vs. how many threads) and not
onto `DispatchUnit` (unrelated structure). Smallest coherent extension: one
new struct + one new `std::optional<ThreadSchedule> thread_schedule` field
on `PerOpDecisionBundle`, resolved by **extending the existing
`KernelSelectionPass`** (no new pass/registration/pipeline-wiring needed) —
mirroring exactly how `tile_plan`/`kernel_selection`/`shape_cost` are
already added as sibling optional fields.

Backward-compatible absence: `std::optional`, same convention as every
other decision in the struct — absent means the key is omitted from
exported JSON entirely (verified live: running against `apple_a17pro_mobile.json`,
which declares no `threadSchedules`, produces **zero** `thread_schedule` keys
anywhere in the output). Schema version stays `2.0.0` — purely additive.

Exact compiler files changed: `include/decision/Decision.h` (new
`ThreadSchedule` struct), `include/serving/ExecutionPlan.h` (new field),
`include/serving/TargetConstraints.h` (new `RuntimeThreadScheduleOption`
struct + `supported_thread_schedules` field on `RuntimeKernelDescriptor`),
`lib/serving/TargetConstraints.cpp` (MLIR-attr lowering of the new nested
array), `tools/compile-for-target/main.cpp` (JSON parsing of profile's new
`threadSchedules` array), `lib/serving/KernelSelectionPass.cpp` (resolution
logic + `physical_compute_units` consumption), `lib/serving/ExecutionPlanBuilder.cpp`
(read attrs into struct), `lib/serving/ExecutionPlanExporter.cpp` (JSON export).

## 3. Thread contract

`thread_schedule_contract_v1`: `thread_count` ∈ {1, 2, 4}, `partition_axis`
∈ {none, m, n} (`two_dimensional` not implemented — not needed, see §5/§12),
`partition_strategy` ∈ {serial, contiguous_chunks} (`static_2d` not
implemented). Rules enforced identically at **three** layers (compiler
declaration validity is structural; Python adapter; native kernel — defense
in depth): `thread_count=1` ⟺ `partition_axis=none` ∧ `partition_strategy=serial`;
`thread_count>1` ⟹ explicit `partition_axis∈{m,n}` ∧ `partition_strategy=contiguous_chunks`.
Invalid combinations are hard failures at the adapter (before dispatch) and
independently at the kernel (its own CLI validation) — never silently
reinterpreted. The native kernel self-reports its actual dispatched
`kernel_id`/`thread_count`/`partition_axis`/`partition_strategy` in its JSON
stdout; the adapter cross-checks this against what it requested and raises
`PortableCpuKernelError` on any mismatch. Absent `thread_schedule` (every
P1B/P1C plan) resolves to the documented default (1 thread, serial) —
**tested** (`test_3_old_plan_without_thread_schedule_defaults_to_one_thread_serial`).

## 4. Runtime implementation

**[CODE]** `native/cpu_kernels/portable_fused_matmul_bias_relu.cpp` extended
with `std::thread` (dependency-free, no OpenMP — the audit found no reason
a smaller dependency-free implementation couldn't work, so OpenMP was never
needed). One generalized function,
`run_fused_tiled_matmul_bias_relu_range(..., m_start, m_end, n_start, n_end, ...)`,
replaces the old whole-range function; passing the full `[0,M)×[0,N)` range
reproduces P1B/P1C behavior byte-for-byte. `dispatch()` spawns
`thread_count` workers via `std::thread`, each computing a disjoint,
ceiling-division contiguous chunk of rows (`partition_axis=m`) or columns
(`partition_axis=n`) — every output element computed by exactly one thread,
exactly once. No shared output ownership, no atomics in the hot loop, no
work stealing, no hidden pool, no nested parallelism — `std::thread::join()`
is the only synchronization point, after all compute finishes. One
implementation family (one file, one generalized function), not one file
per candidate.

**[LIVE]** Correctness coverage, all passing: divisible dims (128×128×64),
non-divisible dims (137×89×113), thread count > small dimension (M=3,N=100
with 4 threads; M=100,N=3 with 4 threads), tiny workloads (1×1×1 serial and
4-thread), zero/invalid dims rejected, unsupported thread_count (3) rejected,
unsupported strategy combos (serial+multithread, contiguous_chunks+1-thread,
axis=none+multithread) all rejected with clear messages.

## 5. Candidate definitions (frozen before measurement)

| ID | threads | axis | strategy | output partition | ownership | launch overhead | sync | false-sharing risk | expected suitability |
|---|---|---|---|---|---|---|---|---|---|
| A | 1 | none | serial | whole M×N | single-threaded, trivial | none | none | none | tiny workloads where any thread overhead dominates |
| B | 2 | m | contiguous_chunks | 2 disjoint row bands | each thread owns ⌈M/2⌉-row band | 1 `std::thread` spawn+join | join only | row-major layout, disjoint bands separated by full rows — negligible | medium/large, especially tall M |
| C | 4 | m | contiguous_chunks | 4 disjoint row bands | each thread owns ⌈M/4⌉-row band | 3 spawns+joins | join only | same as B, smaller bands | medium/large; best available parallel candidate given 4 physical cores |
| D | 2 | n | contiguous_chunks | 2 disjoint column bands | each thread owns ⌈N/2⌉-col band | 1 spawn+join | join only | adjacent-column writes near a band boundary could share a cache line at the seam — bounded, not in a hot-written loop element | wide-N workloads |
| E | 4 | n | contiguous_chunks | 4 disjoint column bands | each thread owns ⌈N/4⌉-col band | 3 spawns+joins | join only | same as D, more seams | wide-N workloads |

Tile identity fixed at `bm32_bn128_bk32` for every candidate (accepted
P1C.1 default) — the ONE new dimension this phase varies is thread
decomposition. 2D partitioning was **not** added: not needed to answer the
research question (see §11/§12 — a clean row-vs-column, size-driven
boundary was found without it), and adding it would have meant a second,
unfrozen implementation dimension mixed into this phase's scope.

## 6. Workload manifest

23 shapes, fixed per-workload seed: 8 calibration, 10 held-out (never used
to fit anything), 5 correctness-only. Categories: tiny, small/medium/large
square, skinny-M/wide-N, wide-M/skinny-N, small K, large K, non-divisible M,
non-divisible N, M<thread_count, N<thread_count, tail-prime. `eval_p1c_continuity_128`
(128×128×128) ties directly back to every prior phase's baseline shape.

## 7. Measurement methodology

Governor pinned to `performance` on all 4 cores (confirmed before/after
every session — Cortex-A76's single cluster shares one cpufreq policy
domain, same finding as P1C). **CPU affinity matched to thread count**
(new for P1D, more rigorous than P1C): 1-thread candidates pinned to core 3
only; 2-thread candidates to cores 2–3; 4-thread candidates to cores 0–3 —
avoiding idle-core waste or oversubscription, exact core set recorded per
measurement. 3 independent sessions (separate process invocations), rotated
candidate order per session (table / reversed / shuffled). Fixed seed per
workload, 7 repeats/invocation, raw + post-warmup statistics both retained
(no silent discarding). Same binary, same compiler flags, same
`std::thread`/contiguous-chunks strategy across every multi-thread
candidate.

## 8. Correctness results

**270/270 (100%)** timed measurements + **25/25 (100%)** correctness-only
edge cases passed (max abs error < 1e-3 against an independent pure-Python
reference), across all 5 candidates × 23 workloads × 3 sessions. Zero
crashes, zero non-zero exit codes.

## 9. Raw and aggregated latency

Full raw data: `results/p1d_raspberry_pi_thread_decomposition/p1d_raw_measurements.json`
(343 KB, complete provenance per measurement). Thermal: 48.8°C → 54.9°C over
the full run (~22 s of continuous multi-core compute across all 3
sessions), `throttled=0x0` at every one of 8 checkpoints — no throttling
ever. Governor `performance` confirmed at every checkpoint.

## 10. Speedup and parallel efficiency

Average parallel efficiency (speedup over serial ÷ thread_count), pooled
across all 18 timed workloads × 3 sessions (n=54 per candidate):

| candidate | avg parallel efficiency |
|---|---|
| B (2-thread, split M) | 0.834 |
| D (2-thread, split N) | 0.828 |
| C (4-thread, split M) | 0.764 |
| E (4-thread, split N) | 0.734 |

76–83% efficiency at 2–4 threads on 4 physical cores is a real, substantial,
non-trivial parallel speedup (not a rounding artifact) — candidate C reaches
up to **3.8–4.0×** speedup over serial on medium/large square workloads.
**No negative scaling was hidden**: efficiency is reported as measured,
including the catastrophic case where threading actively hurts (see §11).

## 11. Stable winners

Pre-registered threshold (same 5% relative-margin convention as P1C, reused
for cross-phase consistency, not chosen after viewing this phase's data):
same winner in all 3 sessions AND ≥5% margin over runner-up in every
session. Result: **6/18 STABLE_WINNER**, **8/18 NEAR_TIE_STABLE_DIRECTION**
(same winner every session, margin dips under 5% in at least one), **4/18
CROSS_SESSION_CONFLICT**. Candidate C (`splitM_4thread`) is the raw winner
in 42/54 session-workload combos and the stable winner in 4/6
STABLE_WINNER workloads; candidate A (serial) is the stable winner in the
other 2 (both tiny workloads, with **85–97% margins** — an enormous,
unambiguous signal that thread-launch overhead dominates at tiny scale).

## 12. Decision-region analysis

A real, clean **two-region** boundary emerged, driven by **workload size**
(M×N×K), not by shape/axis: tiny workloads (16³=4096 and 8³=512 elements)
stably and overwhelmingly prefer serial; every calibration/held-out workload
at or above 64×64×64 (262,144 elements) stably or near-stably prefers
`C_splitM_4thread`. Split-M dominates split-N almost everywhere, including
several skinny-M/wide-N shapes where split-N might intuitively seem
better — split-N only wins outright in 3/18 workloads (`cal_skinny_m_...`,
`eval_skinny_m_...`, `eval_nondivisible_n_151`), and even there the margin
over split-M is small (0.1–3.6%) and not session-stable. This is a
genuine multi-region decision (size-driven), but **not** a strong
axis-driven (shape-driven) one — split-M is close to a dominant default once
past the tiny-workload boundary.

## 13. Compiler policy

**[CODE]/[LIVE]** audited and confirmed: `KernelSelectionPass`'s thread
resolution has no signal that varies with an op's concrete shape — it
always selects the first declared schedule whose `thread_count` fits the
profile's `physical_compute_units` cap, which for the current declaration
order (A first) means it **always selects A (1-thread, serial)**,
regardless of workload size. This is Part 7's honest **Option C**: a simple
static one-thread baseline, with the multi-thread study performed by
invoking Runtime directly with each declared schedule (same methodology as
Phase 1/R1 and Phase P1C).

## 14. Oracle regret (held-out only, 10 workloads, 30 workload-session combos)

| policy | mean regret | median regret | exact-match | avg speedup vs. serial | worst-case speedup vs. serial |
|---|---|---|---|---|---|
| **always A (current live compiler policy)** | 230.99% | 254.38% | 10.0% | 1.000× | 1.000× |
| always B (2-thread, split M) | 392.00% | 89.18% | 0.0% | 1.711× | 0.029× (34× **slowdown**) |
| always C (4-thread, split M) | 411.65% | 0.00% | 80.0% | 3.208× | 0.023× (43× **slowdown**) |
| always D (2-thread, split N) | 360.20% | 92.55% | 0.0% | 1.701× | 0.031× (32× **slowdown**) |
| always E (4-thread, split N) | 423.05% | 2.01% | 10.0% | 3.130× | 0.022× (45× **slowdown**) |
| **shape-aware size threshold** (Part 7 Option A, calibration-only fit) | **0.14%** | 0.00% | 90.0% | 3.305× | **1.000×** (no slowdown) |
| oracle (upper bound) | 0.00% | 0.00% | 100.0% | 3.310× | 1.000× |

The shape-aware policy is a single deterministic threshold on M×N×K
(geometric midpoint, in log-space, between the largest calibration workload
where serial won and the smallest where 4-thread won — **32,768**,
derived using calibration data only, never touching held-out labels,
re-verified explicitly). It is not a learned model. Every "always-N-thread"
static policy risks a 30–45× catastrophic slowdown on tiny held-out
workloads; the shape-aware threshold avoids this entirely while capturing
essentially all of the oracle's speedup (3.305× vs. 3.310× average). This
materially outperforms every trivial static policy and introduces **no**
unacceptable worst-case slowdown — the bar Part 11 sets for justifying a
non-trivial policy is met. **This policy was evaluated only; it was not
implemented in the live compiler pass this phase** (mirroring the
P1C→P1C.1 precedent — recommended as a follow-up phase, see §19).

## 15. Cross-repo contract proof

19 new tests (`tests/test_p1d_thread_schedule_contract.py`), all passing,
against real freshly-generated `ExecutionPlan` JSON (not grep, not static
fixtures): compiler serialization, runtime parsing, old-plan
backward-compatible default, invalid thread-count rejection,
thread_count-vs-`physicalComputeUnits` compliance (verified against the
real profile: declared 4, never exceeded), invalid axis/strategy rejection,
exact-schedule dispatch + self-report agreement for all 5 candidates,
mutated-thread_count and mutated-partition_axis self-report mismatches both
correctly fail, all 5 candidates numerically correct on a
non-tile-aligned/non-thread-aligned shape, and the P1C kernel-ID/thread-count
contract remains valid. Full pre-existing suites re-run: runtime 683
passed (same 2 pre-existing failures + 13 pre-existing CUDA errors,
confirmed unchanged); compiler CTest 19/20 (same pre-existing unrelated
segfault).

## 16. Hardware/profile implications

**[LIVE]** `ml-platform-capabilities` (read-only inspected): zero fields
anywhere for physical compute units, thread topology, SMT, or worker count
(confirmed by full-repository grep); no Raspberry Pi hardware profile
exists there at all. The compiler's own target-profile JSON
(`hardwareExecutionProfile.physicalComputeUnits`) remains the sole practical
source of truth for this fact — a **pre-existing architectural gap**,
documented again here, not migrated or redesigned in this phase (out of
scope, per instruction). The P1D measured scaling evidence
(speedup/efficiency/regret) was kept entirely in
`results/p1d_raspberry_pi_thread_decomposition/` (benchmark evidence), never
duplicated into any static hardware capability profile.

## 17. Known limitations

- The shape-aware threshold policy (§14) was evaluated, not implemented in
  the live compiler pass — a real, evidence-backed recommendation for a
  future phase, not a claim of present dynamic behavior.
- Only 2 calibration data points anchor the threshold's geometric midpoint;
  a wider calibration sweep would narrow the boundary's true location.
- 4/18 workloads showed cross-session conflicts even at the pooled level —
  mostly in the split-M-vs-split-N competitive zone, not in the
  serial-vs-threaded boundary (which was extremely stable).
- 2D partitioning, `static_2d` strategy, and thread counts beyond 4 were not
  implemented — not needed to answer the research question, and the phase's
  scope discipline explicitly limited expansion without justification.
- OpenMP was never needed or added; `std::thread` alone sufficed for this
  scope.
- This is a functional/decision-boundary study on ONE Raspberry Pi board at
  ONE point in time — not a general claim about all ARM Cortex-A76 systems.

## 18. Final verdict

**PASSED_MULTI_REGION_THREAD_DECISION**

A real, stable, size-driven two-region decision boundary exists between
serial execution (tiny workloads, 85–97% margin) and 4-thread split-M
execution (everything else, 76% average parallel efficiency, up to ~4×
speedup) on Raspberry Pi 5 Cortex-A76. This is a materially stronger,
cleaner result than Phase P1C's tile-schedule study (which found only a
soft candidate collapse). A simple, honest, calibration-only shape-aware
threshold captures this boundary with 0.14% mean regret and zero
catastrophic worst-case slowdown, decisively outperforming every trivial
static thread policy — while the currently-live compiler policy (always
serial) leaves substantial real speedup on the table (230% mean regret) on
held-out workloads.

## 19. Recommended next phase

A **P1D.1**-style focused review (mirroring the P1C→P1C.1 precedent):
audit the shape-aware size-threshold policy for leakage/fairness exactly as
P1C.1 did, and — if it survives — make the smallest possible compiler-only
change to select thread schedule by a declared shape threshold rather than
always-first-declared-fits. This should remain a static, deterministic,
non-learned decision (per this phase's own scope discipline) and should NOT
be bundled with thread-count expansion beyond 4, 2D partitioning, NEON, or
any other explicitly out-of-scope item.
