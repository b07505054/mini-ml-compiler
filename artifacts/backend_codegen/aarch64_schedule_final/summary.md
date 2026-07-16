# AArch64 Tiled Matmul Schedule-Unroll: Final Summary (Stages 10-17)

Canonical, reviewer-facing entry point for the entire schedule-unroll
evidence chain. See `artifact_manifest.json` for the full 3-tier file
index and `checksums.txt` for integrity verification of every underlying
artifact. This directory does not duplicate raw MIR/LLVM IR/assembly --
see the tier-2/tier-3 references in the manifest.

## Pipeline

```
hir.fused_matmul_bias_relu (HIR)
  -> hir-matmul-bias-relu-to-linalg
  -> project-owned Transform-dialect script
       (tile + fuse + K-loop-unroll + vectorize; stock combinators only)
  -> stock MLIR-to-LLVM-dialect lowering
  -> mlir-translate -> LLVM IR
  -> llc (unmodified LLVM 21.1.8: instruction selection, machine
          scheduler, greedy register allocator)
  -> AArch64 object
```

Driver: `mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh --variant tiled-scheduled`.
Opt-in selector (Stage 15): `tools/select_and_compile_aarch64_matmul_schedule.py`
sits *above* the compile script and invokes it unmodified -- no second
materialization mechanism exists.

## The transformation

Stock `transform.loop.unroll` applied to the K-reduction `scf.for` loop,
factor = `schedule-unroll-k`, right after K-tiling and before
vectorization. LLVM owns everything downstream (instruction selection,
scheduling, register allocation, spill insertion) -- entirely unmodified.

## Supported configuration space

- **Tiles** (Stage 11 structurally validated): `8x8x8`, `8x8x4`, `4x8x8`. No larger tile has ever been validated.
- **Unroll factors**: `1`, `2`, `4` -- factor must evenly divide the K-loop's pre-unroll trip count.

## The six measured domains

| Domain | Shape | Tile | Measured winner | Spills @ winner | Reloads @ winner | Introduced |
|---|---|---|---|---|---|---|
| primary | 32x32x32 | 8x8x8 | **uk4** | 11 | 12 | Stage 12 |
| cube64 | 64x64x64 | 8x8x8 | **uk4** | 0 | 0 | Stage 16 |
| altk | 32x32x32 | 8x8x4 | **uk4** | 4 | 4 | Stage 16 |
| rect | 32x64x32 | 8x8x8 | **uk4** | 18 | 20 | Stage 16 |
| smallA | 16x16x16 | 8x8x4 | **uk4** | 10 | 11 | Stage 17 |
| highK | 32x32x128 | 8x8x8 | **uk4** | 0 | 0 | Stage 17 |

**6 of 6 domains measured `schedule-unroll-k=4` as fastest**, each
independently, on real Raspberry Pi 5 Cortex-A76 hardware. Stage 17
deliberately searched for a counterexample (smallest legal problem size,
largest K-trip-count tested) and found none. This does **not** establish
universal optimality -- see Truth Boundary below.

## Correctness

Every candidate measured across Stages 13/16/17 (30+ total): **bit-exact**
(`max_abs_error = 0`, not merely within the repository's `1e-3`
tolerance), zero repeated-call failures, zero guard-buffer corruption.

## Backend evidence (static)

Unmodified LLVM 21.1.8 MIR at 5 pass boundaries, analyzed by project-owned
tooling. Spill counts observed across all 6 domains' winners: 0-18. Code
size growth at uk4: 0-62%. **Static ranking alone correctly predicted the
measured winner in exactly 2 of 6 domains (33%) -- precisely the two
domains where the winner happens to be spill-free.** Conclusion,
unchanged since Stage 14: static evidence characterizes cost/risk but is
not a reliable standalone winner predictor.

## Calibrated evidence (measured)

Opt-in `--schedule-candidate-mode=calibrated` selected the real measured
winner in **6 of 6 domains** with compatible evidence, using only
exact-domain data. **Zero cross-domain evidence leakage** across 30
verified cross-domain compatibility checks.

## Truth boundary

**Supported claim**: *For the tested Raspberry Pi 5 Cortex-A76 path,
FP32, this tiled AArch64 matmul-bias-relu NEON microkernel,
`schedule-unroll-k=4` was the measured fastest candidate in all 6
independently tested shape/tile domains.*

**Not claimed, anywhere in this evidence chain**: universal uk4
optimality; automatic production autotuning; a custom LLVM scheduler or
register allocator; arbitrary loop scheduling or software pipelining;
general loop interchange; default/automatic calibrated-mode or uk4
selection.

**Default compiler behavior is unchanged** -- `--schedule-candidate-mode`
defaults to `manual` everywhere; calibrated mode is opt-in only, in every
driver invocation across Stages 15-17.

## Limitations

- No tile larger than 8x8x8 in any dimension has ever been validated.
- No target other than Raspberry Pi 5 Cortex-A76 has ever been measured.
- No dtype other than f32; no kernel family other than this tiled fused microkernel.
- `perf` is unavailable on this Pi -- no hardware-counter (cache-miss/cycle) evidence exists anywhere in this chain.
- Each domain is backed by one measurement session (5 interleaved groups); no domain reproduced across separate days/reboots.
- Selection happens in a Python compiler driver, not a native C++ MLIR pass.

## Files

- `summary.json` / `summary.md` -- this content, machine- and human-readable
- `commands.txt` -- how to regenerate every number in this summary
- `artifact_manifest.json` -- full 3-tier index of all 449 files across Stages 12-17 (13 Tier-1, 160 Tier-2, 276 Tier-3; 17.1MB total)
- `checksums.txt` -- SHA-256 for every file in the manifest
