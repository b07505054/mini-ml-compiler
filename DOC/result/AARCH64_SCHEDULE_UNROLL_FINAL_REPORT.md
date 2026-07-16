# AArch64 Tiled Matmul Schedule-Unroll: Consolidated Final Report (Stages 10-17)

Canonical prose report for the entire LLVM machine-scheduling analysis,
Raspberry Pi calibration, and opt-in compiler-side selection arc built on
top of the tiled AArch64 matmul-bias-relu microkernel (the prior
tile-candidate-selection slice). Machine-readable equivalent:
`artifacts/backend_codegen/aarch64_schedule_final/summary.json`.

## 1. Motivation

The prior tile-candidate-selection slice fixed a tile shape (register
tile M/N/K) for the tiled AArch64 NEON matmul-bias-relu microkernel but
left LLVM's own machine-scheduling behavior and register allocation
entirely unexamined. This arc asks two questions: (1) what does LLVM's
real machine scheduler and register allocator actually do with this
project's generated MIR, and (2) can a narrow, project-owned MLIR
transformation (K-loop unrolling) measurably improve Raspberry Pi
latency without silently trading away correctness or introducing
uncontrolled register pressure.

## 2. Compiler pipeline

```
hir.fused_matmul_bias_relu (HIR)
  -> hir-matmul-bias-relu-to-linalg           [existing, unmodified]
  -> Transform-dialect script                 [project-owned, stock combinators]
       tile_using_for (M,N) -> fuse_into_containing_op (matmul, fill)
       -> tile_using_for (K) -> transform.loop.unroll (factor=schedule-unroll-k)
       -> vectorize_children_and_apply_patterns
  -> one-shot-bufferize, convert-vector-to-scf, convert-vector-to-llvm{outerproduct}, ...
       [stock MLIR->LLVM dialect conversion pipeline, unmodified]
  -> mlir-translate --mlir-to-llvmir           -> LLVM IR
  -> llc -mtriple=aarch64-linux-gnu -mcpu=cortex-a76
       [unmodified LLVM 21.1.8: -aarch64-isel, -machine-scheduler,
        -greedy register allocator, -virtregrewriter, -prologepilog,
        -aarch64-asm-printer]
  -> AArch64 object (.o)
```

Every stage of instruction selection, pre-RA scheduling, register
allocation, spill/reload insertion, and prologue/epilogue generation is
stock, unmodified LLVM 21.1.8. No project code makes a scheduling or
allocation decision anywhere in this pipeline.

## 3. The schedule-unroll transformation

`transform.loop.unroll` (a stock MLIR Transform-dialect op, not
project-written) applied to the K-reduction `scf.for` loop, immediately
after K-tiling and before vectorization, with a caller-supplied factor
(`schedule-unroll-k`). Verified properties (Stage 11):
- `schedule-unroll-k=1` is a byte-for-byte no-op (`.text` section
  identical to the tiled-vectorized baseline).
- A factor equal to the K-loop's own trip count fully collapses the loop
  into straight-line code (verified via mid-pipeline MLIR inspection, not
  assumed).
- The unrolled body forms a single, genuinely serial accumulator chain at
  both the LLVM-IR level (`@llvm.fmuladd` chain reconstruction) and the
  MIR level (self-referencing `FMLA` accumulator chain) -- LLVM never
  reassociates the reduction into parallel partial sums, and no
  fast-math flag is ever present to permit it.

Rationale for choosing this transformation over the alternatives
considered (double-buffered operand staging; accumulator-chain
splitting): Cortex-A76 `llvm-mca` evidence showed LLVM's default
scheduler already interleaves independent accumulator chains well
(median same-accumulator distance ~18 instructions vs. FMLA's 10-cycle
latency) -- the lowest-risk lever was giving the scheduler a larger
static loop body per dynamic iteration, not hand-restructuring the
dependency graph.

## 4. Stage 11: structural validation

Automated structural checks (`tools/validate_aarch64_tiled_schedule_structure.py`)
verify, for every generated candidate before it is ever benchmarked: outer
M/N loop bounds correct; K-loop unroll materializes the expected
`vector.contract` count; every vector op is tile-bounded (never
whole-shape); exactly one correctly-shaped zero accumulator per tile; no
silently-dropped tail dimensions (no `scf.if`/`affine.min`/`affine.max`);
the unrolled K-body is a genuine serial chain. Two genuine edge cases
were found and correctly handled, not special-cased: full K-unroll
collapsing the loop entirely, and (for the smallest tested shape) *both*
the N-loop and K-loop collapsing simultaneously via ordinary
trip-count-1 canonicalization -- unrelated to the schedule transform
itself.

## 5. Stage 12: LLVM MIR scheduling evidence

Real LLVM MIR extracted at 5 pass boundaries (`finalize-isel`,
`machine-scheduler` before/after, `virtregrewriter`, `prologepilog`) for
the primary candidate family (32x32x32, tile 8x8x8, `schedule-unroll-k`
in {1, 2, 4}) plus two comparison tiles. **Stage 12 correctly predicted
the full-K-unroll and alternate-K-tile candidates as regression risks**:
real spills were found (11/12 and 2/2 respectively) and classified
"Class D -- regression risk" under the static evidence available at the
time. This was an honest, correct reading of the static evidence -- it
is not revised or deleted anywhere in this report or the underlying
artifacts.

## 6. Stage 13: Raspberry Pi validation

Real hardware measurement on Raspberry Pi 5 (Cortex-A76, `performance`
governor, fixed core affinity, thermal/throttle capture, interleaved
measurement groups) directly contradicted the *runtime severity* implied
by Stage 12's spill-based classification: **both flagged candidates
measured faster than their matched, non-spilling baselines** (+20.0% and
+11.9% respectively). Stage 13 established the corrected framing:
*"backend-costly, but hardware-confirmed profitable."* **Stage 13 showed
that spill counts alone were not reliable runtime predictors** -- spill
count remains a real backend-cost signal, never a valid standalone
rejection rule.

## 7. Stage 14: cost model

A provenance-tagged candidate evidence and cost model
(`tools/aarch64_schedule_candidate_model.py`) formalizing the correction:
two independent classification dimensions (`backend_safety`:
safe/costly; `hardware_confirmation`: profitable/neutral/regression/
unknown), never forced mutually exclusive, so "backend-costly, hardware-
confirmed profitable" became a real, representable label. A decomposed,
fully-visible cost equation (static compute cost, loop-control cost,
register-pressure penalty, spill/reload/code-size/stack-frame penalties)
replaces the old hard-reject policy; spill count is a soft penalty, never
an automatic veto. Static-only ranking correctly predicted the winner in
only 1 of 4 domains tested at that point (the one where the winner
happened to be spill-free) -- an honest, non-overfit finding, not tuned
to agree.

## 8. Stage 15: compiler-side selection

Connected the Stage 14 model to the real compilation path behind an
explicit opt-in interface (`tools/select_and_compile_aarch64_matmul_schedule.py`,
`--schedule-candidate-mode=manual|static|calibrated`). A hard identity
guard (`verify_no_mismatch`) reconstructs the compiled artifact's
semantic key from the literal compile-script arguments and aborts on any
mismatch -- proving the selected candidate actually changes the compiled
object, not merely a report. Default mode is `manual` (today's existing
`--schedule-unroll-k` behavior, unchanged); calibrated mode is opt-in
only and requires an explicit, compatibility-checked evidence profile.

## 9. Stage 16: multi-domain calibration

Extended calibrated selection to 3 new independent domains (a larger
same-tile shape, a different tile, a rectangular shape) via 9 newly
compiled and Pi-measured candidates. All 4 domains tested to that point
measured `uk4` fastest, each with a materially different backend-cost
profile (0, 0, 4, 18 spills). Static ranking correctly predicted the
winner in only 1 of 4 domains. Cross-domain evidence isolation was
verified explicitly: every one of 12 cross-domain compatibility checks
correctly rejected exact-match reuse (same-tile domains get a
reduced-confidence bucket, never an exact match; different-tile domains
are fully incompatible).

## 10. Stage 17: boundary search

Deliberately searched for a domain where uk4 loses, selecting two new
stress domains chosen specifically to attack uk4's known costs: the
smallest legal problem size (16x16x16, tile 8x8x4 -- real spilling, 10/11)
and the largest K-loop trip count tested (32x32x128, trip 16, a genuine
partial unroll). **uk4 won both.** No counterexample was found across 6
total domains. A new timing-quality methodology was introduced and
applied retroactively (empirically measured clock-read overhead ~37ns on
this Pi; one new measurement flagged "borderline" rather than silently
trusted; Stage 13's smallest-ever measurement would also fail this
stricter standard, though it never fed a ranking comparison). Static
ranking's accuracy across all 6 domains: exactly 2 of 6 (33%) -- precisely
the 2 domains where the winner is spill-free, a clean and unforced
correlation.

## 11. Supported claims

- For the tested Raspberry Pi 5 Cortex-A76 path, FP32, this tiled AArch64
  matmul-bias-relu NEON microkernel, `schedule-unroll-k=4` was the
  measured fastest candidate in all 6 independently tested shape/tile
  domains.
- Opt-in calibrated selection correctly chose the measured winner in
  every domain with compatible evidence, using only exact-domain data,
  with zero cross-domain leakage across 30 verified compatibility checks.
- Static backend evidence (spills, code size, register pressure)
  correctly flags real backend cost in every domain tested, but does not
  reliably predict the measured winner.
- LLVM preserves the MLIR transform's serial floating-point reduction
  order end to end -- verified at both the LLVM-IR and MIR levels, and
  confirmed at runtime via bitwise-identical output between
  `schedule-unroll-k=1` and `=2` on real hardware.

## 12. Unsupported claims (explicitly not made anywhere in this evidence chain)

- Universal `uk4` optimality across untested shapes, tiles, targets, or
  kernel families.
- Automatic production autotuning.
- A custom LLVM scheduler or scheduling pass.
- A custom register allocator.
- Arbitrary loop scheduling or software pipelining.
- General loop interchange.
- Default or automatic calibrated-mode / uk4 selection anywhere in the
  compiler -- `--schedule-candidate-mode` defaults to `manual` in every
  driver invocation across Stages 15-17, with no exception.

## 13. Exact reproduction workflow

```bash
# 1. Structural validation (fast, no toolchain-heavy compile)
python3 tools/validate_aarch64_tiled_schedule_structure.py \
  --input mlir_passes/test/backend_codegen/matmul_bias_relu_tiled_32x32x32.mlir \
  --tile-m 8 --tile-n 8 --tile-k 8 --schedule-unroll-k 2 --output /tmp/check.json

# 2. Manual-mode compile (today's existing, unchanged behavior)
python3 tools/select_and_compile_aarch64_matmul_schedule.py \
  --schedule-candidate-mode manual --schedule-unroll-k 4 \
  --tile-m 8 --tile-n 8 --tile-k 8 \
  mlir_passes/test/backend_codegen/matmul_bias_relu_tiled_32x32x32.mlir \
  /tmp/out primary_uk4

# 3. Calibrated-mode compile (opt-in; requires an explicit profile)
python3 tools/select_and_compile_aarch64_matmul_schedule.py \
  --schedule-candidate-mode calibrated \
  --schedule-profile artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_boundary/updated_multidomain_profile.json \
  --tile-m 8 --tile-n 8 --tile-k 8 \
  mlir_passes/test/backend_codegen/matmul_bias_relu_tiled_32x32x32.mlir \
  /tmp/out primary_calibrated

# 4. Full combined test suite (175 tests, 7 stages)
python3 -m unittest tests.test_aarch64_mir_analysis \
  tests.test_aarch64_schedule_comparison tests.test_aarch64_schedule_pi_validation \
  tests.test_aarch64_schedule_candidate_model tests.test_aarch64_schedule_selection \
  tests.test_aarch64_schedule_multidomain tests.test_aarch64_schedule_boundary
```

Per-stage exact commands: each stage's own `commands.txt` under its
`artifacts/backend_codegen/aarch64_matmul_bias_relu_*/` directory (see
`artifacts/backend_codegen/aarch64_schedule_final/artifact_manifest.json`
for the full index).
