# LLVM Scheduling Evidence Comparison: tiled-scheduled vs. tiled-vectorized-equivalent baseline

Target: Raspberry Pi 5 / Cortex-A76 (no Pi execution in this slice --
static/backend evidence only)
Analysis type: Real LLVM MIR at 5 pipeline boundaries, real register
allocation output, real LLVM IR, and `llvm-mca` Cortex-A76 static
estimates, for the `--variant tiled-scheduled` compile path added in
Stage 10.

## What this directory answers

Stage 11 established that the `tiled-scheduled` variant's MLIR structure
is correct (K-loop unroll materializes as expected, tail-free, serial
accumulator chain preserved at the MLIR level). Stage 12 asks the next
question: **what does LLVM actually do with that transformed input**, and
does it change anything that matters (register allocation, spills, code
size, scheduling)?

## Truth boundary

Same boundary as every prior slice in this series. The project owns: which
tile/unroll parameters to generate, MIR/IR *extraction* at named pass
boundaries, and *analysis* of the resulting text. **LLVM owns everything
else** -- instruction selection, the pre-RA machine scheduler, live-range
coalescing, greedy register allocation, spill-code insertion, and
prologue/epilogue generation are all stock, unmodified LLVM 21.1.8. No
project code makes a scheduling or allocation decision anywhere in this
slice.

## Controlled comparison design

Baseline and scheduled candidates both go through **the same**
`--variant tiled-scheduled` code path (Stage 10), differing only in
`--schedule-unroll-k`. This was possible because Stage 11 proved
`schedule-unroll-k=1` is a byte-for-byte `.text` no-op vs. the separate
`tiled-vectorized` variant -- so using `tiled-scheduled` uniformly on both
sides of every comparison, rather than switching between two different
`--variant` flags, removes any doubt that something other than the
schedule/unroll choice differs between baseline and scheduled.

## LLVM pass boundaries used (LLVM 21.1.8, `-mtriple=aarch64-linux-gnu -mcpu=cortex-a76`)

| Boundary | Flag | Note |
|---|---|---|
| Instruction-selection output | `--stop-after=finalize-isel` | virtual registers present, AArch64 opcodes present |
| Machine-scheduler input | `--stop-before=machine-scheduler` | after two-address/live-interval/coalescing, immediately before the scheduler runs |
| Machine-scheduler output (= pre-RA) | `--stop-after=machine-scheduler` | the LAST boundary before the allocator; still 100% virtual registers |
| Register-allocator output (post-RA) | `--stop-after=virtregrewriter` | **not** `--stop-after=greedy` alone -- verified (prior slice) that `-greedy` alone leaves virtual registers unrewritten; physical registers only appear after `-virtregrewriter` |
| Final emission input | `--stop-after=prologepilog` | stack frame finalized, callee-saved save/restore visible |
| Post-RA scheduler | not enabled/inspected | task brief explicitly excludes custom/post-RA scheduling work in this slice |

## Candidate matrix

| Label | Shape | Tile M/N/K | unroll-k | Loop structure | Object bytes | Spills | Reloads | Accum. chains | Max chain len |
|---|---|---|---|---|---|---|---|---|---|
| primary_unroll1 | 32x32x32 | 8x8x8 | 1 | M,N,K all present | 2608 | 0 | 0 | 16 | 8 |
| primary_unroll2 | 32x32x32 | 8x8x8 | 2 | M,N,K all present | 3248 | 0 | 0 | 16 | 16 |
| primary_full_unroll (edge case) | 32x32x32 | 8x8x8 | 4 | K collapsed (full unroll) | 4232 | **11** | **12** | 16 | 32 |
| alt_k_tile_unroll1 | 32x32x32 | 8x8x4 | 1 | M,N,K all present | 2336 | 0 | 0 | 16 | 4 |
| alt_k_tile_unroll2 | 32x32x32 | 8x8x4 | 2 | M,N,K all present | 2640 | **2** | **2** | 16 | 8 |
| cube64_unroll1 | 64x64x64 | 8x8x8 | 1 | M,N,K all present | 2704 | 0 | 0 | 16 | 8 |
| cube64_unroll2 | 64x64x64 | 8x8x8 | 2 | M,N,K all present | 3344 | 0 | 0 | 16 | 16 |
| small_control_collapsed (edge case) | 8x8x8 | 4x8x8 | 1 | N and K both collapsed | 1696 | 0 | 0 | 8 | 8 |

All 8 candidates: `post_ra` physical vector registers referenced = 28 for
the zero-spill candidates, 32 (the full architectural budget) for the two
that spill -- consistent with genuine register-pressure exhaustion, not a
parser artifact (`spill_slot_bytes` = 16 x spill count in both cases,
matching one 128-bit vector spill slot per spill).

## Evidence classification (task section 12: A/B/C/D)

| Comparison | Classification | Basis |
|---|---|---|
| primary (unroll 1 vs 2) | **A** -- scheduling win likely | 0 new spills, physical vector registers unchanged (28->28), scheduling metrics not worse, fewer dynamic K-loop iterations |
| primary_full_unroll_edge_case (unroll 1 vs 4) | **D** -- regression risk | 11 new spill stores + 12 new reloads; RA exhausts the full 32-register budget |
| alt_k_tile (unroll 1 vs 2, tile 8x8x4) | **D** -- regression risk | 2 new spill stores + 2 new reloads at this smaller K-tile |
| cube64 (unroll 1 vs 2) | **A** -- scheduling win likely | identical pattern to primary; confirms the result isn't an artifact of one specific shape |

**This is the main finding of Stage 12**: `schedule-unroll-k=2` at the
primary tile shape (8x8x8) is safe (zero spills, confirmed on two
different M/N/K shapes), but unrolling further is NOT free -- both the
full-K-unroll variant (factor 4) and a smaller K-tile (8x8x4) at factor 2
introduce real allocator spills. The register budget, not the MLIR
transform's correctness, is the limiting factor. `small_control_collapsed`
(8x8x8/tile-4x8x8) has no valid unroll-k>1 (K trip count is already 1 at
unroll-k=1, both N and K loops collapse via ordinary trip-count-1
canonicalization -- reported here as a structural edge case, not mixed
into the primary comparison, per instruction.)

## Floating-point reduction-order finding (task section 6)

MLIR's `vector-to-llvm` lowering emits `@llvm.fmuladd.v8f32` calls
directly for every `vector.contract` -- fusion here is unconditional
(part of the intrinsic's own definition), **not gated by fast-math flags**.
Checked explicitly across all 8 candidates: zero fast-math-flag tokens
(`fast`/`reassoc`/`contract`/`afn`/`nnan`/`ninf`/`nsz`/`arcp`) found on any
`@llvm.fmuladd` call. Accumulator chains were reconstructed at the LLVM IR
level (independently of the MIR-level `analyze_aarch64_machine_schedule.py`
chain metric) by following each call's accumulator operand back to an
earlier call's destination in the same function:

- Every candidate: `all_fmuladd_calls_accounted_for = true` (every call is
  part of exactly one reconstructed chain -- no orphaned or double-counted
  calls).
- Chain count matches the number of independent output-tile positions
  (e.g. 8 for the 8x8x8-tile candidates, one per row); chain length scales
  with `schedule_unroll_k` as expected (8 -> 16 -> 32 for the primary tile
  at unroll 1/2/4).
- **No evidence of reassociation into parallel partial sums anywhere in
  the matrix.** LLVM preserves the MLIR transform's serial reduction
  order end to end.

A real bug was found and fixed while building this check: the first
`fmuladd` in a fully-unrolled reduction chain sometimes accumulates
directly into the literal keyword `zeroinitializer` (no `%` prefix)
instead of a named `%cst` SSA value -- a regex requiring a leading `%`
silently dropped these calls (undercounted 256 real calls as 248 for the
full-unroll candidate, caught by cross-checking against a plain
`grep -c llvm.fmuladd`). Fixed and regression-tested
(`tests/test_aarch64_schedule_comparison.py::test_bareword_zeroinitializer_accumulator_operand_is_counted`).

Separately, assembly-level FMLA counts are consistently **2x** the LLVM
IR-level `fmuladd` call count across every candidate (e.g. 128 IR calls ->
256 asm `fmla` instructions for `primary_unroll2`) -- LLVM's own backend
performs an additional static unroll of the small-trip-count K-loop beyond
MLIR's transform, independent of this project's `schedule-unroll-k`
parameter. This is stock LLVM behavior (not project-owned) and is reported
here as observed evidence, not further modified or relied upon.

## llvm-mca (STATIC machine-model estimate -- primary comparison only)

Modeled CPU: `cortex-a76` (Cortex-A76 scheduling model, the closest
available static model to the real Raspberry Pi 5 target; llvm-mca does
not model the exact Pi 5 SoC's actual observed behavior, only the
Cortex-A76 pipeline description shipped with LLVM 21). Region: the
innermost ("This Inner Loop Header") K-loop, located via its LLVM
block-comment annotation and scoped with `# LLVM-MCA-BEGIN`/`# LLVM-MCA-END`
markers. 100 iterations, `-mtriple=aarch64-linux-gnu -mcpu=cortex-a76`.

| | primary_unroll1 | primary_unroll2 |
|---|---|---|
| Instructions modeled | 17500 | 33500 |
| Total cycles | 12845 | 25645 |
| Total uOps | 23900 | 39900 |
| Dispatch width | 3 | 3 |
| uOps/cycle | 1.86 | 1.56 |
| IPC | 1.36 | 1.31 |
| Block RThroughput (per static loop body) | 128.0 | 256.0 |

Block RThroughput roughly doubles with the static body size (as expected
-- twice as many static instructions per loop body), while the scheduled
variant's K-loop executes **half as many dynamic iterations**. Aggregate
modeled compute (RThroughput x dynamic trip count) is essentially
unchanged; the loop-control-overhead reduction from fewer dynamic
iterations is the mechanism any real speedup would come from, not a
per-instruction throughput improvement. **This is a static estimate, not
measured Raspberry Pi latency** -- no Pi execution happens anywhere in
this script or slice.

## Bugs found and fixed while building this tooling

1. `find_innermost_loop_region`: the regex `This (?:Inner )?Loop Header`
   matched the OUTER loop's "This Loop Header: Depth=1" comment first
   (since "Inner" is optional), locking onto the wrong region for every
   multi-level loop nest. Fixed to select the loop-header comment with the
   highest `Depth=N`, not the first match. Regression-tested.
2. `check_fp_reduction_order`'s operand regex required a leading `%`,
   silently dropping `zeroinitializer`-literal accumulator operands (see
   above). Fixed and regression-tested.
3. `classify_pair`'s original pressure-cost check compared raw
   `virtual_vector_registers_before_ra` SSA-def counts, which grow
   ~linearly with static unroll factor regardless of actual live-range
   overlap (242 -> 402 for the primary candidate at unroll 1 -> 2, despite
   BOTH allocating to the identical 28 physical registers with zero
   spills). Replaced with the authoritative post-RA physical
   register-count delta and real spill/reload evidence; the MIR-derived
   `approx_peak_live_vector_registers` heuristic is now reported as
   context only, never as a classification gate (its own docstring notes
   it has no loop-back-edge modeling and over-counts on this project's
   loop-bodied MIR).
4. `classify_pair`'s overlap-improvement check used `>=` (non-strict),
   which let two IDENTICAL `schedule-unroll-k=1` candidates be misreported
   as a scheduling "win". Fixed by additionally requiring
   `schedule_unroll_k` to have strictly increased (the real source of any
   expected benefit -- fewer dynamic loop iterations). Regression-tested
   (`test_identical_metrics_at_unroll_1_is_neutral_not_a_or_d`).
5. Added an explicit `MismatchedComparisonError` guard in `classify_pair`:
   comparing two candidates with different shape or tile now raises
   immediately rather than silently scoring the comparison, per the task
   brief's explicit warning against attributing tile/shape differences to
   the schedule transform.

## Files in this directory

- `schedule_comparison_results.json` -- full machine-readable record for
  all 8 candidates: metadata (git commit/dirty state, target
  triple/cpu/features, llc/mlir-opt versions, full compiler commands),
  pass boundaries, register-allocation metrics (reusing
  `analyze_aarch64_candidate_mir.py`'s spill/callee-saved classification
  unmodified), pre/post-scheduler MIR metrics (reusing
  `analyze_aarch64_machine_schedule.py` unmodified), FP-reduction-order
  findings, assembly-level counts, object sizes, and the A/B/C/D
  classification for each baseline/scheduled pair.
- `schedule_comparison_summary.md` -- short auto-generated classification
  summary (same data as above, condensed).
- `commands.txt` -- every command used to produce this directory's
  contents, in order.
- `candidates/<label>/` -- per-candidate raw evidence: LLVM-dialect MLIR,
  LLVM IR, 5 MIR stages, final assembly/object, register/schedule metrics
  JSON, and (primary candidates only) the extracted hot-loop-region
  assembly used for `llvm-mca`.

## What Stage 12 does NOT claim

No Raspberry Pi execution happened in this slice. Every number here is
static: LLVM's own compile-time output (MIR, register allocation, object
size) or `llvm-mca`'s modeled Cortex-A76 estimate. The "A" (scheduling win
likely) classification for the primary and cube64 comparisons is backend
evidence that `schedule-unroll-k=2` is *safe* (no spills, no code-size
blowup, schedule not worse) and *structurally plausible* to help (fewer
dynamic loop iterations) -- it is not a measured speedup. Stage 13/14 (Pi
correctness and benchmarking) is required before any performance claim.

## Update: Stages 13-15 (measured evidence, cost model, opt-in selection)

Stage 12's "Class D -- regression risk" verdict for the full-K-unroll and
alternate-K-tile spilling candidates was an honest reading of the static
evidence available at the time -- **it is not deleted or rewritten here**.
Stage 13 (`artifacts/backend_codegen/aarch64_matmul_bias_relu_pi_scheduling/`)
measured both of those candidates on real Raspberry Pi 5 hardware and
found the static severity prediction **contradicted**: both were faster
than their matched, non-spilling baselines. The corrected, non-revisionist
framing (Stage 14/15): *"backend-costly, but hardware-confirmed
profitable."* Spill count remains a real backend-cost signal -- it is not,
by itself, a valid hard-rejection rule.

Stage 14 (`artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_cost_model/`)
turned this into a provenance-tagged candidate evidence and cost model.
Stage 15 (`artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_selection/`)
connects that model to the real compilation path behind an **opt-in**
`--schedule-candidate-mode=manual|static|calibrated` interface on a new
driver script (`tools/select_and_compile_aarch64_matmul_schedule.py`),
proving the selected candidate actually changes the compiled object (not
just a report) via a hard identity guard. Default compiler behavior is
unchanged; calibrated selection is never enabled automatically.

## Update: Stage 16 (multi-domain calibration)

Stage 15's calibrated result was anchored to a single domain (32x32x32,
tile 8x8x8). Stage 16
(`artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_multidomain/`)
extends real Pi measurement to three more independent domains: a larger
same-tile shape (64x64x64), a different tile (8x8x4), and a rectangular
shape (32x64x32). **`schedule-unroll-k=4` measured fastest in all four
domains tested to date**, each with a distinct backend-cost profile (0,
0, 4, and 18 spills respectively) -- reported as-is, not as evidence of
universal optimality (only 4 domains, one microkernel family, one target
have ever been measured). Static evidence alone predicted the correct
winner in only 1 of these 4 domains (the one where the winner happened to
be spill-free), reinforcing Stage 14/15's finding that calibrated
measurement, not static scoring, is required for reliable winner
selection with this microkernel family. Cross-domain compatibility
checking was verified explicitly: no domain's measured evidence has ever
been used to rank another domain's candidates (same-tile domains get a
reduced-confidence "cross-shape" bucket, never an exact match;
different-tile domains are fully incompatible).

## Update: Stage 17 (counterexample search -- none found)

Stage 17 (`artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_boundary/`)
deliberately searched for a domain where `schedule-unroll-k=4` loses,
selecting two new stress domains specifically to break the Stage 16
pattern: the smallest problem size where uk4 is legal (16x16x16, tile
8x8x4 -- 4,096 total FLOPs, real spilling: 10 stores/11 reloads) and the
largest K-loop trip count tested to date (32x32x128, trip 16, a genuine
partial unroll rather than a full collapse). **uk4 won both new domains**,
extending the measured support region to 6 total domains with no
counterexample found. This is reported as a valid "no counterexample"
result, not proof of universality -- uk4 remains opt-in-calibrated-mode
only, never a default, and major dimensions (larger tiles, other targets,
other dtypes, other kernel families) remain untested. A real
timing-quality concern was investigated and documented: empirically
measured clock-read overhead on this Pi is ~37ns, making the smallest new
candidate's absolute latency (351ns) a "borderline" measurement by a
10x-overhead reliability threshold -- flagged explicitly rather than
silently trusted, and applied retroactively to reveal that Stage 13's
smallest-ever measurement (92ns) would also fail this stricter standard
(though it never fed a ranking comparison, so no prior conclusion is
invalidated). A real evidence-merge bug was also found and fixed while
re-evaluating the static model across all 6 domains: static evidence
correctly predicts the measured winner in exactly the domains where that
winner is spill-free (2 of 6) and nowhere else -- a clean, unforced
correlation, not a retuned fit.
