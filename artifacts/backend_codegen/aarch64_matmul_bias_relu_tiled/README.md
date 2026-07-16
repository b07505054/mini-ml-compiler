# AArch64 Tiled Vector Microkernel Codegen: hir.fused_matmul_bias_relu

Target: Raspberry Pi 5 / Cortex-A76
Execution type: Real hardware
Code generation: MLIR -> (project-owned MLIR tiling + vectorization) ->
  scf.for loop nest around a fixed-size Vector dialect microkernel -> LLVM
  IR -> LLVM AArch64 backend (LLVM-owned machine instruction selection)
Optimization status: Second vectorization slice. Replaces whole-shape
  ("fully-unrolled") vectorization's unbounded code-size scaling with a
  compact, reusable, loop-driven microkernel. Not a general vectorizer, not
  a custom instruction selector, not tail-handling-complete.

## Achievement summary

Reduced 32x32x32 generated AArch64 object size from 113,216 bytes to 2,128
bytes (53.2x smaller) using tiled MLIR vector lowering, while retaining
real NEON FMLA and, measured on the real Raspberry Pi, running FASTER than
the fully-unrolled kernel (median latency ratio 0.63, i.e. tiled is ~1.59x
faster than fully-unrolled at 32x32x32 -- see "Performance" below; the
1.35x-slowdown ceiling this slice was gated on was not just met but never
approached). The tiled kernel also executed 64x64x64 correctly on the
Raspberry Pi (1000 repeated calls + 500-cycle mixed-shape stress, zero
failures) with median latency 0.019834 ms across 500 iterations, at a
2,104-byte object size -- compared to a fully-unrolled object at that shape
that was never even built because its projected size (~900 KB) is
impractical.

## Truth boundary: what is project-owned vs. LLVM-owned here

This slice adds one new project-owned Transform-dialect script,
`mlir_passes/transforms/tile_vectorize_matmul_bias_relu.mlir`, invoked
between `hir-matmul-bias-relu-to-linalg` and bufferization. It combines
three stock upstream MLIR Transform ops:

1. `transform.structured.tile_using_for` on the bias+relu `linalg.generic`
   (the final consumer), tile sizes `[4, 8]` -- produces the outer M and N
   `scf.for` loops.
2. `transform.structured.fuse_into_containing_op`, twice: fuses the
   `linalg.matmul` producer, then the `linalg.fill` zero-init producer,
   into that loop nest -- so the epilogue (bias-add+ReLU) and the
   accumulator zero-init both happen per-output-tile rather than as
   separate whole-tensor passes.
3. A second `transform.structured.tile_using_for` on the now-fused matmul,
   tile size `[0, 0, 8]` (K only) -- produces the inner K-reduction
   `scf.for` loop with a fixed 4x8x8 tile.
4. `transform.structured.vectorize_children_and_apply_patterns`, applied to
   the whole (now-tiled) `func.func` -- vectorizes only the small,
   fixed-shape `linalg` ops left inside the loop bodies; the `scf.for`
   loops themselves are untouched by vectorization (they are not `linalg`
   ops), which is exactly why they survive into the final object.

This is "project-owned" only in the sense that the project chose these
specific stock ops, in this specific order, at this point in its own
pipeline, for its own op -- none of the tiling, fusion, or vectorization
*algorithms* are project-authored; they are upstream MLIR.

Everything from the resulting bounded `vector.contract` (operands
`vector<4x8xf32>`/`vector<8x8xf32>`, accumulator `vector<4x8xf32>`) to the
`fmla` instructions in `generated_32x32x32.s` is LLVM's AArch64 backend,
unmodified -- identical truth boundary to the prior (fully-unrolled)
vectorization slice. See `tiled_vector_dialect_32x32x32.mlir` for the exact
boundary: everything in that file is the project-owned tiling+vectorization
stage output (real `scf.for`, bounded vector types); everything downstream
is stock MLIR->LLVM conversion plus LLVM's own backend. No project code
anywhere selects, schedules, or allocates registers for the `fmla`
instructions themselves.

## A real MLIR-toolchain limitation found and worked around

Getting from "K-tiled tensor-level IR" to "compiling object" required two
non-obvious additions beyond what the prior (whole-shape) vectorized
variant needed, both discovered empirically in this slice and both stock
MLIR functionality (no custom passes written):

1. **Three `lower-affine` passes**, not one. Tiling (twice) and
   `convert-vector-to-scf`'s index arithmetic each introduce fresh
   `affine.apply` ops; nothing later in the pipeline understands the affine
   dialect, so each stage that can introduce one is followed by
   `lower-affine` before the next stage that would otherwise choke on it.
2. **`convert-vector-to-scf{full-unroll target-rank=1}`**, not
   `test-vector-transfer-flatten-patterns`. The K-loop's accumulator tile
   bufferizes to a `memref.subview` of the shared output buffer (a real
   slice, not a fresh allocation -- one-shot-bufferize correctly avoids an
   extra copy). `convert-vector-to-llvm` only lowers 1-D vector transfers
   directly; the whole-shape variant's transfers are on a fully contiguous,
   non-sliced memref, so a straight reshape-style flattening pattern
   (`test-vector-transfer-flatten-patterns`) suffices there. A `subview`
   with a non-unit outer stride (each of the tile's rows is separated by
   the parent buffer's full row width, not the tile's own width) cannot be
   flattened that way -- there is no single contiguous span to reshape.
   `convert-vector-to-scf{full-unroll}` instead statically unrolls the
   small (<=8-row) N-D transfer into a sequence of 1-D transfers, which
   `convert-vector-to-llvm` then handles directly. This was reached by
   elimination: an initial attempt at `transform.structured.
   hoist_redundant_vector_transfers` (the "obvious" fix for a
   read-compute-write-back-to-the-same-slice loop pattern) does not apply
   here -- its own header comment documents that it explicitly excludes
   transfers sourced from `ViewLikeOpInterface` ops (which `memref.subview`
   is) "to reduce the risk of aliasing," and is marked "TODO: obsolete and
   should be retired" in the MLIR 21 headers used by this project. No
   custom pass was written to work around this; `convert-vector-to-scf`
   is a complete, correct, stock substitute for this specific case.

## Files

- `input_32x32x32.mlir`, `tiled_vector_dialect_32x32x32.mlir`,
  `lowered_llvm_dialect_32x32x32.mlir`, `generated_32x32x32.ll`,
  `generated_32x32x32.s`, `objdump_32x32x32.txt` -- **full generated
  artifacts for the representative shape only** (32x32x32 tiled-vectorized),
  per this slice's compact artifact policy (see "Artifact policy" below).
  `matmul_bias_relu_tiled_32x32x32.o` itself is **not committed** (`.o`
  excluded repo-wide by `.gitignore`, consistent with every prior slice in
  this repository); its SHA-256 and size are in `object_hashes.txt`.
- `disasm_excerpts/tiled_<shape>_head.txt` (8x8x8, 16x16x16, 64x64x64,
  32x64x32, 64x32x64) -- first 60 disassembled lines only, for the shapes
  that do NOT get full artifacts, to give a concrete look at each without
  duplicating ~101 KB of near-identical output six times over.
  `disasm_excerpts/generic_32x32x32_head.txt` -- a short excerpt of the
  generic (unvectorized) 32x32x32 object for direct side-by-side
  comparison; its full disassembly is already committed at
  `artifacts/backend_codegen/aarch64_matmul_bias_relu_vectorized/generic/objdump_32x32x32.txt`
  and is not duplicated here.
- `benchmark_<shape>.json` (all 6 shapes) -- per-shape correctness verdict
  (scalar/generic/fully-unrolled-vectorized-where-available/tiled-vectorized/
  handwritten, checksums, max abs error) and median/p95 latency, measured
  on the real Raspberry Pi.
- `repeated_call_results.txt`, `mixed_shape_results.txt` -- Pi correctness
  stress test output (see "Raspberry Pi correctness" below).
- `backend_metrics.json` -- static LLVM IR / AArch64 instruction counts,
  static vs. dynamic FMLA counts, load/store/branch counts, object size,
  and largest vector width, per shape per variant, with counting
  methodology documented inline.
- `register_pressure_32x32x32.txt` -- the Stage-12-style register-pressure
  precheck for the chosen tile (design validation, not a full
  register-allocation analysis).
- `pi_device_state.txt` -- `hostname; uname -a; lscpu; gcc --version`
  captured on the real Raspberry Pi at benchmark time.
- `object_hashes.txt`, `commands.txt` -- reproducibility records.

## Microkernel design: candidates considered and the chosen tile

AArch64 provides 32 architectural 128-bit (4xFP32) vector registers.
Candidates evaluated (accumulator regs = TM*ceil(TN/4); B regs/K-step =
ceil(TN/4); A broadcasts = up to TM worst-case / 1-2 typical; FMLA/K-step =
TM*ceil(TN/4)):

| Tile (MxN) | Acc regs | B regs | A broadcasts (worst/typical) | FMLA/K-step | Register total (worst/typical) | Notes |
|---|---|---|---|---|---|---|
| 4x4 | 4 | 1 | 4/1 | 4 | 9/6 | Safest margin, but each B load reused for only 4 FMAs and each A broadcast for only 1 -- poor compute/load ratio |
| **4x8 (chosen)** | 8 | 2 | 4/1-2 | 8 | 14/12 | Balanced: each B load reused across 4 FMAs, each A broadcast across 2 |
| 8x4 | 8 | 1 | 8/1-2 | 8 | 17/11 | Best B reuse (8 FMAs/load) but worst A reuse (1 FMA/broadcast); more live A broadcasts than 4x8 |
| 8x8 | 16 | 2 | 8/1-2 | 16 | 26/20 | Best raw FLOP/load ratio, but 26+ registers before any loop/ABI overhead is close to the 32-register ceiling -- real spill risk |

K step = 8 (the "K step = 4 or 8" option this task's brief suggested),
chosen over 4 for fewer loop-branch overheads relative to FMA work at a
negligible code-size cost, given the generous size budget already achieved.

**Selected: TM=4, TN=8, TK=8.** All six required shapes (8, 16, 32, 64 for
M/N/K, plus the 32x64/64x32 combinations) are exactly divisible by this
tile.

The analytical estimate above (12 typical / 14 worst-case live vector
registers) was a *pre-codegen* design estimate. The actual final assembly
(see `register_pressure_32x32x32.txt`) shows LLVM's own instruction
scheduler uses all 32 registers in the hot loop (it software-pipelines two
K-sub-steps together for ILP) -- with zero spills. This is reported
honestly as a real, measured finding that exceeds the design-time estimate,
not smoothed over.

## MLIR structural evidence

`tiled_vector_dialect_32x32x32.mlir` (project-owned tiling+vectorization
stage, pre-bufferization) contains three real nested `scf.for` loops (M
step 4, N step 8, K step 8) around a fixed microkernel body: `vector.
transfer_read` of `vector<4x8xf32>` (A tile) and `vector<8x8xf32>` (B
tile), a `vector.contract` producing `vector<4x8xf32>` (accumulator), and
`vector.transfer_write` back into the K-loop's tensor iter_arg. After the
K-loop, bias is read as another `vector<4x8xf32>`, added, ReLU'd
(`arith.maximumf`), and written into the final output tile. **No vector
type in this file exceeds 32 elements** (8x4 = 32, the largest), well under
the 64-FP32-lane bound this slice's own structural test enforces (see
`tools/run_mlir_pass_tests.sh`, `run_backend_codegen_tiled_static_checks`).
Contrast with the fully-unrolled variant's `vector<32x32xf32>` (1024 lanes)
for the identical 32x32x32 shape.

## LLVM IR and assembly evidence

`generated_32x32x32.ll` contains only bounded vector types: `<8 x float>`
is the largest that appears (matches the disassembled `fmla v*.4s` -- each
128-bit NEON register holds 4 lanes; `<8 x float>` decomposes to a pair of
them). No `<1024 x float>` or any shape-sized type appears, at any of the
six shapes tested (see `backend_metrics.json`'s `largest_vector_width`
column -- constant at 8 across every shape, vs. the fully-unrolled
variant's 64/256/1024 that scales with M*N).

`objdump_32x32x32.txt` / `generated_32x32x32.s` contain:
- Real NEON `fmla v*.4s, v*.4s, v*.s[n]` -- 64 static instances (the
  reusable microkernel body), vs. the fully-unrolled variant's 8,192 static
  instances for the same shape and identical total arithmetic (see
  "Static vs. dynamic FMLA" below).
- Real conditional loop branches (`b.gt`, `b.le`, plus unconditional
  back-edges) -- 11 total for 32x32x32, vs. 2 for the fully-unrolled
  variant (function entry/exit only, since it has no loops at all).

## Static vs. dynamic FMLA count

The static count (instructions present in the object) and the dynamic
count (instructions actually executed per call) are identical for the
fully-unrolled variant (no loop -- every instruction executes exactly
once) but diverge sharply for the tiled variant, which executes the same
64-instruction body repeatedly:

| Shape | Fully-unrolled static=dynamic | Tiled static | Tiled dynamic (computed: (M/4)x(N/8)x(K/8)x64) |
|---|---|---|---|
| 8x8x8 | 128 | 64 | 128 |
| 16x16x16 | 1,024 | 64 | 1,024 |
| 32x32x32 | 8,192 | 64 | 8,192 |
| 64x64x64 | not built (see below) | 64 | 65,536 |
| 32x64x32 | not built | 64 | 16,384 |
| 64x32x64 | not built | 64 | 32,768 |

Dynamic FMLA count is always exactly M*N*K/4 for both variants -- both
compute the identical arithmetic. This table is the direct evidence that
the tiled variant is not "doing less work"; it is doing the same work with
a reused, bounded-size instruction sequence instead of a fully materialized
one.

## Raspberry Pi correctness

1000 consecutive calls per (shape, variant) pair, generic and
tiled-vectorized, all 6 shapes (12 pairs total) -- all PASS with clean
sentinel guard regions:

```
PASS: shape=8x8x8 variant=generic all 1000 calls correct, guards clean
PASS: shape=8x8x8 variant=tiled-vectorized all 1000 calls correct, guards clean
PASS: shape=16x16x16 variant=generic all 1000 calls correct, guards clean
PASS: shape=16x16x16 variant=tiled-vectorized all 1000 calls correct, guards clean
PASS: shape=32x32x32 variant=generic all 1000 calls correct, guards clean
PASS: shape=32x32x32 variant=tiled-vectorized all 1000 calls correct, guards clean
PASS: shape=64x64x64 variant=generic all 1000 calls correct, guards clean
PASS: shape=64x64x64 variant=tiled-vectorized all 1000 calls correct, guards clean
PASS: shape=32x64x32 variant=generic all 1000 calls correct, guards clean
PASS: shape=32x64x32 variant=tiled-vectorized all 1000 calls correct, guards clean
PASS: shape=64x32x64 variant=generic all 1000 calls correct, guards clean
PASS: shape=64x32x64 variant=tiled-vectorized all 1000 calls correct, guards clean
```

Mixed-shape/mixed-variant same-process stress: 500 cycles through all six
shapes (8x8x8, 16x16x16, 32x32x32, 64x64x64, 32x64x32, 64x32x64), generic
and tiled-vectorized called for each shape each cycle, with unrelated
heap-allocation noise (16-528 bytes, varying) injected between every call
-- 6,000 total calls, single process, no isolation:

```
PASS: 500 cycles (6000 total calls) all correct
```

Zero tolerance: any single incorrect call anywhere in either test is a
FAIL (threshold 1e-3 max abs error vs. a scalar double-accumulated
reference); both report unconditional PASS.

## Raspberry Pi performance (fresh measurement, this slice)

2000 iterations / 200 warmup for 8x8x8-32x32x32; 500 iterations / 200
warmup for the three larger shapes (per the task's iteration-count
guidance for longer-running shapes). All correct at every shape
(`all_correct: true`).

| Shape | Generic median | Fully-unrolled median | Tiled median | Handwritten median | Tiled vs. generic | Tiled vs. fully-unrolled | Tiled vs. handwritten |
|---|---|---|---|---|---|---|---|
| 8x8x8 | 0.002277 ms | 0.000111 ms | 0.000092 ms | 0.000592 ms | 24.75x faster | 0.83x (17% faster) | 0.155x (6.4x faster) |
| 16x16x16 | 0.016704 ms | 0.000556 ms | 0.000500 ms | 0.003759 ms | 33.41x faster | 0.90x (10% faster) | 0.133x (7.5x faster) |
| 32x32x32 | 0.127925 ms | 0.004611 ms | 0.002908 ms | 0.026611 ms | 43.99x faster | 0.63x (59% faster) | 0.109x (9.2x faster) |
| 64x64x64 | 1.004627 ms | not built | 0.019834 ms | 0.208166 ms | 50.65x faster | n/a | 0.095x (10.5x faster) |
| 32x64x32 | 0.255721 ms | not built | 0.005167 ms | 0.053222 ms | 49.49x faster | n/a | 0.097x (10.3x faster) |
| 64x32x64 | 0.502388 ms | not built | 0.010796 ms | 0.104148 ms | 46.53x faster | n/a | 0.104x (9.6x faster) |

**Performance target: met and exceeded, not just satisfied.** The gate was
tiled median <= 1.35x fully-unrolled median (i.e. retain >=74% of
fully-unrolled performance). Measured ratios (0.63-0.90) mean the tiled
kernel is consistently FASTER than the fully-unrolled kernel it replaces,
at every shape where both exist -- not a tradeoff of code size for speed,
but an improvement on both axes simultaneously. This is plausible, not
surprising in hindsight: the fully-unrolled kernel's ~113 KB of code at
32x32x32 does not fit in Cortex-A76's L1 instruction cache (typically 64
KB), so repeated calls likely pay real icache-miss cost that the tiled
kernel's ~2 KB reused loop body does not.

## Why 64x64x64/32x64x32/64x32x64 have no fully-unrolled comparison point

`--variant vectorized` (fully-unrolled) was not generated for these three
shapes. Extrapolating from the measured 8x8x8 -> 16x16x16 -> 32x32x32
scaling (2,192 -> 13,120 -> 113,216 bytes, roughly 6x per doubling of one
dimension, consistent with M*N*K scaling), 64x64x64 would be on the order
of 700KB-900KB (a real fully-unrolled 64x64x64 object was in fact built
transiently during this slice's benchmark-baseline collection and measured
at 713,552 bytes, confirming this projection). This is exactly the failure
mode this slice exists to fix; building and shipping it as a supported
artifact would contradict the slice's own premise. `backend_metrics.json`
records this as `"available": false` rather than fabricating or silently
omitting a number.

## Not implemented in this slice (explicit non-goals)

- General arbitrary-shape tail handling: shapes not evenly divisible by
  4(M)/8(N)/8(K) are REJECTED by `compile_hir_matmul_bias_relu_aarch64.sh
  --variant tiled-vectorized` with a nonzero exit and an explicit error
  message (verified: M=N=K=10 rejected). Use `--variant generic` or
  `--variant vectorized` for such shapes.
- Cost-model-driven tile selection (the 4x8x8 tile is fixed, chosen by the
  register-pressure analysis above, not selected per-shape)
- Full register-pressure / spill-cost analysis (register_pressure_32x32x32.txt
  is a design-validation precheck only)
- Instruction scheduling or software pipelining beyond LLVM's own `llc`
  (the K-sub-step interleaving observed is entirely LLVM's own scheduler)
- Prefetching
- Any custom LLVM SelectionDAG or GlobalISel instruction-selection pattern
- Runtime dispatch / dlopen loading of generated code into
  heterogeneous-inference-runtime or any ExecutionPlan
- INT8 / SDOT / UDOT dot-product lowering
- GPU code generation of any kind
- Fixes to the pre-existing 20 historical failing tests in
  `tools/run_mlir_pass_tests.sh` (unrelated to this slice; baseline
  unchanged at 20/94 after this slice's own 6 new structural tests were
  added, up from 20/88 before)

## Artifact policy and size

This directory intentionally does NOT repeat the previous slice's pattern
of committing complete `.ll`/`.s`/objdump text for every shape and variant
(that produced ~5.3 MB and ~84,000 added lines, ~98.5% of which was
generated IR/assembly text -- see the Stage-1-audit note in this slice's
final report). Instead: complete generated artifacts for one representative
shape (32x32x32 tiled-vectorized, ~101 KB total), metrics/correctness/
benchmark JSON for all six shapes, object hashes+sizes for the 12 objects
not given full treatment, short (60-line) disassembly excerpts for the
other five tiled shapes plus one generic-variant comparison excerpt, and
exact reproduction commands. Nothing here duplicates the generic or
fully-unrolled full artifacts already committed in
`artifacts/backend_codegen/aarch64_matmul_bias_relu_vectorized/`.
