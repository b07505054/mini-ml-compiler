# FINAL_RELEASE_NOTES.md — AArch64 Schedule-Unroll Evidence + Opt-In Selection Slice

## Completed Work (Stages 10-20)

A narrow, real LLVM/MLIR AArch64 backend-compiler machine-scheduling slice
for the fused MatMul-Bias-ReLU HIR kernel, spanning: a stock Transform-dialect
K-loop unroll transformation; structural, LLVM-MIR, and real Raspberry Pi 5
hardware validation across 6 independent shape/tile domains; a
provenance-tagged static + measured evidence/cost model; an opt-in
compiler-driver candidate-selection and materialization layer; a
counterexample-oriented boundary search; and full repository finalization
(artifact curation, documentation consolidation, truth-boundary audit,
regression, and release documentation).

## Main Compiler Achievement

Applying the stock MLIR Transform-dialect op `transform.loop.unroll` to the
K-reduction loop of a tiled AArch64 NEON matmul-bias-relu microkernel, at a
deliberately chosen point in the existing lowering pipeline, and building a
real evidence chain — from unmodified LLVM 21.1.8 MIR at 5 pass boundaries
through to real Raspberry Pi 5 Cortex-A76 wall-clock measurement — that
answers, for 6 independently tested shape/tile domains, which unroll factor
is actually fastest and why static compile-time signals alone cannot
reliably predict it.

## Pipeline (unchanged, LLVM-owned throughout)

`hir.fused_matmul_bias_relu` → `hir-matmul-bias-relu-to-linalg` →
project-owned Transform-dialect script (tile + fuse + K-unroll + vectorize,
stock combinators only) → stock MLIR-to-LLVM-dialect lowering →
`mlir-translate` → LLVM IR → `llc` (unmodified LLVM 21.1.8 instruction
selection, pre-RA machine scheduler, greedy register allocator) → AArch64
object. The project owns exactly two things: (1) where to apply the stock
`transform.loop.unroll` op, and (2) an opt-in Python compiler-driver
selector (`tools/select_and_compile_aarch64_matmul_schedule.py`) that picks
an unroll factor before invoking this unchanged pipeline.

## Evidence Chain

- **Structural** (Stage 11): tile configurations `{8x8x8, 8x8x4, 4x8x8}` and
  unroll factors `{1, 2, 4}` verified byte-for-byte at the MLIR level.
- **LLVM MIR** (Stage 12): real backend evidence at 5 pass boundaries —
  finalize-isel, machine-scheduler before/after, virtregrewriter,
  prologepilog — for the `primary` domain (32x32x32, tile 8x8x8).
- **Raspberry Pi 5** (Stage 13): real hardware correctness + benchmark
  validation, bit-exact across all measured candidates.
- **Cost model** (Stage 14): provenance-tagged static + measured evidence
  categories, corrected soft-penalty ranking policy (spill is a cost, never
  an automatic veto).
- **Selection** (Stage 15): opt-in `manual|static|calibrated` compiler-driver
  modes, hard identity guard, deterministic 4-tier fallback.
- **Multi-domain calibration** (Stage 16) and **boundary search** (Stage 17):
  4 additional independently designed domains, including domains
  specifically chosen to stress uk4's known costs (spills, small-problem
  fixed overhead, high-K live ranges) — no counterexample was found.
- **Finalization** (Stages 18-20): 3-tier artifact manifest + checksums,
  consolidated report, truth-boundary audit, repository audit, full
  regression, release documentation.

## Measured Domains and Winners

| Domain | Shape | Tile | Winner | Spills/Reloads | Introduced |
|---|---|---|---|---|---|
| primary | 32x32x32 | 8x8x8 | uk4 | 11/12 | Stage 12 |
| cube64 | 64x64x64 | 8x8x8 | uk4 | 0/0 | Stage 16 |
| altk | 32x32x32 | 8x8x4 | uk4 | 4/4 | Stage 16 |
| rect | 32x64x32 | 8x8x8 | uk4 | 18/20 | Stage 16 |
| smallA | 16x16x16 | 8x8x4 | uk4 | 10/11 | Stage 17 |
| highK | 32x32x128 | 8x8x8 | uk4 | 0/0 | Stage 17 |

All 6 tested domains measured `schedule-unroll-k=4` as fastest, on real
Raspberry Pi 5 / Cortex-A76 / FP32 hardware. This is reported as-is and is
explicitly **not** claimed as universal (see Truth Boundary below).

## Static-Model Findings

Static backend evidence alone (spill count, code-size growth, llvm-mca
estimates) correctly predicted the measured winner in **exactly 2 of 6
domains** — `cube64` and `highK`, precisely the two domains where the
measured winner happens to be spill-free. This is a clean, unforced
correlation (spill-free ⇒ static-predictable), not a retuned fit, and it
means static evidence is not a reliable standalone winner predictor across
the tested domain set, even though it never hid a real spill.

## Calibrated-Model Findings

Opt-in calibrated mode, driven by real Raspberry Pi measurement, selected
the correct measured winner in **6 of 6 domains** with compatible evidence.
All 30 cross-domain compatibility pairs (6 domains × 5 others) were
correctly rejected for exact-match reuse — same-tile domains fall back to a
reduced-confidence cross-shape bucket (0.7), never a false exact match;
different-tile domains are fully incompatible. Zero cross-domain evidence
leakage events.

## Truth Boundary

**Supported claim**: for the tested Raspberry Pi 5 Cortex-A76 path, FP32,
this tiled NEON matmul-bias-relu microkernel, `schedule-unroll-k=4` was the
measured fastest candidate in all 6 independently tested shape/tile domains.

**Not claimed**: universal uk4 optimality across untested shapes, tiles,
targets, or kernel families; automatic production autotuning; a custom LLVM
scheduler or scheduling pass; a custom register allocator; arbitrary loop
scheduling or software pipelining; general loop interchange; default or
automatic calibrated-mode/uk4 selection anywhere in the compiler.

**Default compiler behavior is unchanged.** `--schedule-candidate-mode`
defaults to `manual` everywhere; calibrated mode and any uk4 preference are
opt-in only, in every driver invocation across Stages 15-17.

## Known Limitations

- No tile larger than 8x8x8 in any dimension has ever been structurally validated.
- No target other than Raspberry Pi 5 / Cortex-A76 has ever been measured.
- No dtype other than f32 has ever been measured.
- No kernel family other than this tiled fused matmul-bias-relu microkernel has ever been measured.
- No hardware performance-counter evidence (cycles, cache misses) exists anywhere in this chain — only wall-clock timing with an empirically validated ~37ns overhead floor, and code-size/llvm-mca static estimates.
- Each of the 6 domains is backed by a single measurement session; no domain has been independently reproduced across separate days/reboots.
- The register-pressure heuristic is a linear scan with no loop-back-edge modeling — documented, and never used as a default scoring signal.
- Selection happens in a Python compiler driver, not a native C++ MLIR pass.
- `tools/run_boundary_pi.py` and `tools/run_multidomain_pi.py` hardcode the validation Pi's address with no CLI override (see `FINAL_AUDIT.md` §5) — a reproducibility inconvenience, not a correctness issue.

## Future Work

- Extend structural validation to additional tile shapes beyond `{8x8x8, 8x8x4, 4x8x8}`.
- Measure on additional AArch64 targets to test whether the uk4 result generalizes beyond Cortex-A76.
- Collect real hardware performance-counter evidence (e.g. via `perf`) if/when available on validation hardware.
- Add a CLI override for the Pi host in `run_boundary_pi.py` / `run_multidomain_pi.py`.
- If a production autotuning integration is ever pursued, it should build on the existing opt-in `calibrated` mode and its deterministic fallback policy rather than replacing it.
