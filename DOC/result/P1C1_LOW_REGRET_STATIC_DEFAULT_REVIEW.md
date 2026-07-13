# Phase P1C.1 — Raspberry Pi Low-Regret Static Default Review

Base commits reviewed: compiler `213dc9b814344a96066af11ab491e52da72899c7`
(P1C), runtime `3784e1d9330f656a93eb306c309180c0103109c9` (P1C).

## Leakage / fairness review (against committed P1C evidence)

1. **Calibration workloads were not mixed with held-out evaluation.**
   Confirmed: `results/p1c_raspberry_pi_multi_candidate/p1c_regret_analysis.json`
   restricts its regret computation to the 9 `held_out`-split workloads only
   (`held_out_workload_count: 9`); the 8 `calibration`-split workloads never
   enter that computation.

2. **The proposed candidate was selected using only calibration data** — this
   was **re-verified in this review**, not merely re-asserted: the P1C
   report had flagged, as a known limitation, that the original "global
   best" identification (§8/§9 of the P1C report) ranked candidates using
   the full dataset (including held-out). Recomputing the ranking using
   **only the 8 calibration workloads** (24 session-workload samples per
   candidate) from the committed `p1c_raw_measurements.json` gives:

   | candidate | calibration-only mean-of-medians (ms) | calibration-only raw wins |
   |---|---|---|
   | **bm32_bn128_bk32** | **1.5644** | **13/24** |
   | bm128_bn128_bk32 | 1.5656 | 2/24 |
   | bm64_bn64_bk64 | 1.5723 | 5/24 |
   | bm64_bn64_bk128 | 1.5732 | 4/24 |
   | (remaining 4 candidates) | 1.59–1.62 | 0/24 |

   `bm32_bn128_bk32` is the winner by both metrics using calibration data
   alone — identical to the full-dataset conclusion. The leak did not change
   the outcome, but it is worth recording precisely that it was checked
   rather than assumed away.

3. **Held-out regret (from the committed P1C regret analysis, evaluated only
   on the 9 held-out workloads, 27 workload-session combos):**

   | metric | value |
   |---|---|
   | mean regret | 0.078% |
   | median regret | 0.000% |
   | P95 regret | 0.582% |
   | max regret | 0.695% |
   | exact-match rate | 70.4% |

4. **Comparison:**

   | policy | mean regret | exact-match rate |
   |---|---|---|
   | current default (bm32_bn32_bk32) | 3.889% | 0.0% |
   | **proposed default (bm32_bn128_bk32)** | **0.078%** | **70.4%** |
   | globally best candidate (all workloads) | 0.078% (same candidate) | 70.4% (same) |
   | per-workload oracle | 0.000% (by definition) | 100.0% |

   The calibration-only choice and the all-workloads choice are the **same
   candidate**, so these two rows are identical — consistent with point 2.

5. **The policy does not depend on held-out labels** — confirmed by point 2's
   re-verification: the calibration-only ranking alone determines
   `bm32_bn128_bk32` as the choice; held-out data was used only to *measure*
   regret afterward, never to *select* the candidate.

6. **Zero workloads met the stable-winner threshold.** Confirmed from the
   committed `p1c_oracle_analysis.json`: `stable_winner_count: 0` out of 17
   workloads (same winner in all 3 independent sessions with ≥5% margin, the
   threshold frozen before the P1C results were viewed).

7. **This is an aggregate low-regret static policy, not a shape-aware
   decision boundary and not a universal optimum.** The compiler's selection
   mechanism (`KernelSelectionPass`) has no signal that varies with an op's
   concrete shape (re-confirmed in P1C, unchanged in P1C.1) — the same
   candidate is selected for every workload regardless of M/N/K. This
   change only picks a better *default*; it does not add, and must not be
   read as adding, any per-shape decision-making capability.

**Verdict: the proposed default survives the leakage/fairness review.**

## Compiler-only change

`configs/target_profiles/raspberry_pi5_cortex_a76_cpu.json`: the
`runtimeKernels[]` array's **first** entry changed from
`portable_fused_matmul_bias_relu_bm32_bn32_bk32` to
`portable_fused_matmul_bias_relu_bm32_bn128_bk32` (a straight swap of
position — the `bm32_bn128_bk32` object was moved to the front, the
`bm32_bn32_bk32` object moved to second position; no other candidate's
identity, fields, or relative order among the remaining 6 was touched, and
no candidate was deleted). `runtimeKernelsNote` extended (not rewritten) to
record the P1C.1 change and its evidence basis directly in the profile
itself.

**Why ordering is the selection mechanism (re-confirmed, not re-derived):**
`KernelSelectionPass` matches registry entries against an op's `(op_name,
backend, dtype, quant_mode, layout, tile-plan, local-memory)` and returns the
**first full match in declaration order**, short-circuiting immediately —
there is no field in that match anywhere that depends on the op's concrete
shape. This was empirically re-confirmed for this exact change: the same
128×128×128 MLIR fixture now resolves to `bm32_bn128_bk32` (previously
`bm32_bn32_bk32`), with nothing else about the pipeline, MLIR input, or pass
list changed — proving the swap in declaration order, and only that, is what
moved the selection.

Focused test updated: `mlir_passes/test/serving/RunRaspberryPiFusedMatMulBiasReluKernelSelectionTest.cmake`
now asserts `"selected_kernel": "portable_fused_matmul_bias_relu_bm32_bn128_bk32"`
(previously asserted the old default) — same test, same purpose (prove the
declared default is actually selected end to end, and that the two upstream
weight/bias ops remain honestly `rejected_no_kernel_for_op`), updated
expectation only.

## Real ExecutionPlan regeneration and Raspberry Pi execution

Freshly generated via `compile-for-target` against the updated profile +
the unchanged `mlir/p1b_fused_matmul_bias_relu_cpu.mlir` fixture — confirmed
`kernel_selection.selected_kernel = portable_fused_matmul_bias_relu_bm32_bn128_bk32`,
`status = selected`. Deployed (fresh plan + the 3 unchanged pure-stdlib
adapter files + the already-built, unchanged native kernel binary — no
Runtime code changed, so no rebuild was needed) to the Raspberry Pi
(`100.110.37.6`) and executed there:

| check | result |
|---|---|
| Selected kernel ID | `portable_fused_matmul_bias_relu_bm32_bn128_bk32` (confirmed exact match) |
| P1B-compatible 128×128×128 correctness | max abs error `5.47e-05`, **passed** |
| P1B-compatible 128×128×128 latency | 10 raw samples, median **1.568 ms** |
| Held-out shape 256×256×256 (`eval_medium_square_256`) correctness | max abs error `1.48e-04`, **passed** |
| Held-out shape 256×256×256 latency | 10 raw samples, median **12.491 ms** |
| Invalid-plan rejection (fabricated kernel ID) | **rejected**, `PortableCpuKernelError`, no silent substitution |
| Target profile ID | `raspberry-pi5-cortex-a76-cpu` |
| Compiler commit | `213dc9b814344a96066af11ab491e52da72899c7` |
| Runtime commit | `3784e1d9330f656a93eb306c309180c0103109c9` |
| CPU governor | `performance` on all 4 cores, before and after |
| Thermal | 49.9°C → 51.6°C, `throttled=0x0` both times |
| Affinity | not pinned for this verification run (correctness/contract check, not a timing study — differs from P1C's dedicated measurement protocol, which did pin affinity) |
| Runtime dispatched `bm32_bn128_bk32` exactly | **confirmed** (`dispatched_bm32_bn128_bk32_exactly: true`) |

Raw evidence: `p1c1_verification_evidence.json` (432-byte-scale JSON,
generated on the Pi this run — see below for commit disposition).

## Runtime-side consequence (reported, not silently committed)

Two **runtime-repo test files** required updates so existing tests continue
to pass against the new compiler default (they assert "what does the
compiler currently select," which changed): `tests/test_p1b_cross_repo_contract.py`
and `tests/test_p1c_multi_candidate_contract.py`. These are test-oracle
updates only (new expected-value strings, one test restructured to
explicitly request `bm32_bn32_bk32` rather than relying on the compiler's
default, to keep testing genuine backward-compatible dispatch rather than
tracking a moving target) — **no adapter, kernel, or Runtime dispatch logic
was changed**. Per instruction ("do not modify Runtime in P1C.1 unless an
actual contract defect is found; report rather than silently broadening
scope"), **these two files are left modified but uncommitted** in the
runtime repository's working tree; no runtime commit was created in P1C.1.
All 23 runtime tests (16 P1B + 7 P1C) pass with these updates in place; the
full runtime suite (671 collected) shows the same pre-existing 2
failures/13 errors as every prior phase, confirmed via `git status`/direct
run, not stash-diffed again since no other runtime file changed.

## Compiler verification (full checklist)

- `git diff --check`: clean.
- Full CTest suite: **19/20 pass**, same pre-existing `ServingStaticCostModelV1Test`
  segfault (unrelated, confirmed in every prior phase).
- `RaspberryPiFusedMatMulBiasReluKernelSelectionTest`: **passes** with the
  updated expectation.
- Profile JSON: valid, exactly 8 unique `kernelId`s, no duplicates, no
  deletions.
