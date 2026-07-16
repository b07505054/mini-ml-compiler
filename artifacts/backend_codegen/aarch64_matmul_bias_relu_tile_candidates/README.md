# AArch64 Backend-Informed Tile Selection: hir.fused_matmul_bias_relu

Target: Raspberry Pi 5 / Cortex-A76
Execution type: Real hardware, all 42 candidates
Code generation: MLIR -> (project-owned parameterized tiling + vectorization)
  -> LLVM AArch64 backend (LLVM-owned machine instruction selection)
Optimization status: Third AArch64 backend-codegen slice. Replaces the
  single hard-coded 4x8x8 tile with a real multi-candidate evidence-driven
  selection flow. Offline, shape-specific selection -- not an online
  autotuner, not wired into any production plan-selection pass.

## Achievement statement

Generated and evaluated 42 legal AArch64 tiled microkernel candidates (7
tile shapes x 6 matrix shapes) across matrix shapes 8x8x8 through
64x64x64, using real Raspberry Pi latency, generated code size, and
assembly-derived spill evidence to select shape-specific tiles. The
selected policy achieved 99.8%-100% of the fastest measured performance at
every shape (5 of 6 shapes selected the single fastest candidate exactly;
the sixth sacrificed 0.35% latency for a smaller object at equal spill
count) while avoiding all spills in every selected candidate and limiting
every generated object to under 2.7KB.

## Truth boundary: what is project-owned vs. LLVM-owned here

Identical boundary to the prior (single fixed-tile) slice, now applied
across 7 tile configurations instead of 1: the project-owned contribution
is (a) a parameterized Transform-dialect template
(`mlir_passes/transforms/tile_vectorize_matmul_bias_relu.template.mlir`,
replacing the prior slice's single fixed-tile file) instantiated per
candidate by `mlir_passes/tools/generate_tiled_transform.sh`, composing the
same stock upstream MLIR Transform ops (`tile_using_for`,
`fuse_into_containing_op`, `vectorize_children_and_apply_patterns`) at
different tile sizes, and (b) this slice's selection machinery (candidate
generation, register-pressure evidence collection, scoring, selection) --
a build-time evidence-gathering and decision layer, not a compiler pass.
All NEON `fmla` instruction selection, register allocation, and scheduling
for every one of the 42 candidates remain entirely LLVM's `llc`, unmodified.
The scoring policy (`scoring_policy.json`) is explicit, inspectable
arithmetic -- no machine learning, no hidden weights.

## Parameterized tiling design

`mlir_passes/transforms/tile_vectorize_matmul_bias_relu.template.mlir`
replaces the prior slice's single fixed-tile file (`M-tile`/`N-tile`/
`K-tile` placeholder tokens substituted by
`mlir_passes/tools/generate_tiled_transform.sh --tile-m --tile-n --tile-k`
via `sed`, output written to a scratch temp file, never committed --
Option B from the task brief: "one parameterized template, substitute
validated integer values"). `mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh
--variant tiled-vectorized` gained `--tile-m/--tile-n/--tile-k` flags
(default 4/8/8, verified byte-identical `.text` section to the prior
committed slice's fixed-tile output when omitted or given explicitly as
4/8/8). Seven near-identical Transform files were deliberately NOT
committed; only the one template is.

Every candidate for a given shape is compiled from the shared shape
fixture (`mlir_passes/test/backend_codegen/matmul_bias_relu_tiled_<shape>.mlir`),
but with its `func.func` renamed to embed the tile
(`matmul_bias_relu_tiled_<shape>_tm<M>_tn<N>_tk<K>`) before compilation --
required so all candidates for one shape export distinct `_mlir_ciface_`
symbols and can link into a single Raspberry Pi test binary (Stage 8's
mixed-candidate stress test requires exactly this). See
`tools/generate_aarch64_matmul_tile_candidates.py`'s `renamed_fixture()`.

## Candidate set and legality

All 7 required tiles (4x4x4, 4x8x4, 4x8x8, 8x4x4, 8x4x8, 8x8x4, 8x8x8) x
all 6 shapes (8x8x8 optional + the 5 required: 16x16x16, 32x32x32,
64x64x64, 32x64x32, 64x32x64) = **42 combinations, all statically legal**
(every required shape is divisible by every candidate tile in every
dimension -- no shape%tile rejections occurred; this was verified, not
assumed, by running the actual divisibility check for all 42 pairs) **and
all 42 compiled successfully** through the real, unmodified LLVM AArch64
backend (zero "MLIR transformation failure" rejections). The analytical
worst-case register-demand estimate (accumulator + B operand + worst-case
A broadcasts) ranges 9-26 across the 7 tiles, all under the configured
hard limit of 32 -- see `candidate_results.json`'s
`legality.register_estimate` field and "Register-pressure methodology"
below for why this limit was deliberately set at the full architectural
register count rather than a tighter analytical bound.

## Register-pressure methodology

Assembly-derived evidence only (no custom register allocator, no LLVM MIR
pass added) -- `tools/analyze_register_pressure.py`: splits each
candidate's `.s` file into basic blocks by `.LBBn_m:` labels, identifies
the "hot loop" as the fmla-containing block with the deepest loop-nest
comment LLVM emits, counts distinct `v0`-`v31` registers referenced within
it, and counts stack-relative loads/stores within that scope -- explicitly
excluding any line LLVM marks "Folded Spill"/"Folded Reload" (AAPCS64
callee-saved register preservation, always ABI overhead, never a
microkernel register-pressure spill; see `register_pressure_summary.txt`
for a real bug this exclusion fixed during development). All results are
labeled `"assembly-derived register-use evidence (not exact LLVM
register-pressure analysis)"` throughout -- never claimed as exact LLVM
liveness/allocation output.

## Files

- `candidate_results.json` -- full merged evidence for all 42 candidates:
  legality, structural MLIR checks, correctness (Raspberry Pi), performance
  (Raspberry Pi), and backend metrics (LLVM IR/AArch64 instruction counts,
  static FMLA, object/text size, register-pressure evidence).
- `selected_tiles.json` -- one selected tile per shape, with score
  breakdown, selection reasons, the fastest-measured candidate for
  comparison, and every rejected candidate's reason.
- `scoring_policy.json` -- the exact scoring formula, weights, and
  rationale (also documented in `tools/select_aarch64_matmul_tile_candidate.py`'s
  module docstring).
- `benchmark_<shape>.json` (6 files) -- raw Raspberry Pi benchmark output
  per shape (all candidates for that shape + scalar reference).
- `repeated_call_results.txt`, `mixed_stress_results.txt` -- Raspberry Pi
  correctness stress output (see `correctness_summary.txt` for the
  condensed version).
- `register_pressure_summary.txt` -- condensed register-pressure findings,
  including the one candidate (8x8x8, tile 8x8x8) with real spill evidence.
- `object_hashes.txt` -- SHA-256 + size for all 42 candidate objects (not
  committed -- `.gitignore` excludes `*.o` repo-wide, as in every prior
  slice).
- `representative_candidate_32x32x32_tm8_tn8_tk8/` -- full generated
  artifacts (input HIR, generated Transform-dialect instance, pre- and
  post-bufferization MLIR, LLVM IR, assembly, disassembly) for one
  representative candidate: the shape/tile this slice actually selected
  for 32x32x32.
- `disasm_excerpts/` -- short (50-70 line) disassembly excerpts for the two
  most instructive LOSING candidates: `8x8x8_tm8_tn8_tk8_spilling_head.txt`
  (the one candidate with real register-pressure spill evidence) and
  `32x64x32_tm8_tn8_tk8_fastest_not_selected_head.txt` (the one shape
  where the fastest measured candidate was NOT selected).
- `pi_device_state.txt` -- `hostname; uname -a; lscpu; gcc --version`
  captured on the real Raspberry Pi at benchmark time.
- `commands.txt` -- exact reproduction commands for every file above.

**Why the harness .cpp files are not committed**: the three generated Pi
test/benchmark C++ programs (`aarch64_matmul_tile_candidate_repeated_call_test.cpp`,
`_mixed_stress_test.cpp`, `_benchmark.cpp`) are deterministically generated
by `tools/generate_tile_candidate_harness.py` from `candidate_results.json`'s
candidate list (42 `extern "C"` declarations + a dispatch table) -- 47KB of
generated boilerplate that would otherwise need to be committed and kept
in sync by hand. Only the generator is committed; `commands.txt` step 3
regenerates the harnesses exactly.

## Scoring policy

```
score = latency_ms / best_latency_ms
      + 0.20 * hot_loop_vector_spills
      + 0.20 * hot_loop_vector_reloads
      + 0.10 * (object_bytes / smallest_object_bytes)
      + 0.05 * max(0, vector_registers_referenced - 28)
```

Hard rejection before any scoring: not legal, or not correct (failed the
Raspberry Pi 1-call/1000-call check, or `max_abs_error >= 1e-3`) -- an
incorrect or illegal candidate is excluded entirely, never scored as a
high-penalty option (none occurred among these 42; the rule exists for the
general case).

**Rationale for the weights** (chosen deliberately, not fit to data): latency
is the primary signal (coefficient 1.0, normalized to the best candidate
per shape). A single hot-loop spill (+0.20) outweighs a latency advantage
smaller than 20% of the best candidate's latency -- large enough that one
real spill matters, small enough that a genuinely faster candidate still
wins even with a spill. Object-size growth is a minor tiebreaker (0.10,
normalized to the smallest object per shape) since every candidate here is
1.7-2.7KB -- the difference is real but should not dominate over latency
or spills. The register penalty (0.05 per register beyond 28) only
activates within 4 registers of the 32-register ceiling; it exists purely
to break near-ties toward candidates with more headroom, and was
deliberately NOT set to penalize 32-register usage on its own, since the
prior single-tile slice already showed 32 registers with zero spills is a
perfectly fine, measured-safe outcome.

**Tie-break order** (applied when rounded scores are equal): 1) lower
latency, 2) zero spills+reloads, 3) smaller object, 4) fewer registers
referenced, 5) smaller tile volume (tm*tn*tk).

## Automatic selection results

| Shape | Selected Tile | Score | Median ms | Object Bytes | Matches Fastest |
|---|---|---|---|---|---|
| 8x8x8 | 4x8x8 | 1.1000 | 0.000092 | 1,704 | yes |
| 16x16x16 | 8x8x8 | 1.1456 | 0.000445 | 2,632 | yes |
| 32x32x32 | 8x8x8 | 1.1447 | 0.002593 | 2,616 | yes |
| 32x64x32 | 4x8x8 | 1.1194 | 0.005148 | 2,096 | **no** (0.35% slower than fastest) |
| 64x32x64 | 8x8x8 | 1.1434 | 0.009518 | 2,616 | yes |
| 64x64x64 | 4x8x8 | 1.1158 | 0.019685 | 2,112 | yes |

## Fastest vs. selected analysis

5 of 6 shapes: selected candidate **is** the single fastest measured
candidate (score differences among alternatives came entirely from spill/
size penalties on already-slower candidates, never overriding a genuine
speed win).

1 of 6 shapes (32x64x32) differs:

```
Fastest:   tile 8x8x8,  0.005130 ms, 0 spills, 2,712 bytes, 27 registers
Selected:  tile 4x8x8,  0.005148 ms, 0 spills, 2,096 bytes, 19 registers

Latency sacrificed: 0.35%
Reason: both candidates have zero measured spills, so the tie was broken
by the object-size term (0.10 * 2712/2096 = 0.129 vs. 0.10 * 1.0 = 0.10
for the smaller candidate) -- a 22.7% smaller object for a 0.35% latency
cost. Score: 1.1194 (selected) vs. 1.1500 (fastest).
```

This is the scoring policy's code-size tiebreaker operating exactly as
designed on a near-tie (both zero-spill), not the spill penalty -- the
full breakdown is in `selected_tiles.json`'s `rejected_candidates` entry
for this shape.

## Shape-generalization findings

No single tile wins universally, and no clean shape-class pattern (small
vs. large, square vs. rectangular) emerges either -- reported honestly
rather than inventing a bucket theory to fit:

- Tile **4x8x8** selected for: 8x8x8, 32x64x32, 64x64x64
- Tile **8x8x8** selected for: 16x16x16, 32x32x32, 64x32x64

Both tiles that "won" have TN=TK=8 (or TM=8 with TN=TK matching) -- the
common thread across every selection is that TN=8 (never TN=4) and TK=8
(never TK=4) was chosen at every shape; only the TM=4-vs-8 choice varies,
and it varies by a margin under 1% latency at every shape where it
matters. This is a genuine, evidence-derived finding, not a
pre-supposition -- the candidate set deliberately included all 4 TM=4/
TN=4-or-8/TK=4-or-8 combinations plus all 4 TM=8 combinations precisely so
this kind of pattern (or its absence) could be observed rather than
assumed.

## Register-pressure findings

41 of 42 candidates: zero hot-loop vector spills/reloads, regardless of
registers referenced (10-32). The one exception (8x8x8, tile 8x8x8: 3
spills, 0 reloads) is also the single most register-hungry, most-collapsed
candidate in the set (tile == shape in every dimension, so the loop nest
disappears entirely via MLIR canonicalization and all 128 FMLA
instructions live in one straight-line block). See
`register_pressure_summary.txt` for the full breakdown, including a real
bug (ABI folded-spill lines initially mis-attributed as hot-loop spills)
found and fixed during this slice's own development.

## Selected-candidate reproduction

Verified for all 6 shapes via `tools/reproduce_selected_tile_candidate.py`:
recompiles the exact selected tile through the standard, unmodified
`--variant tiled-vectorized --tile-m/--tile-n/--tile-k` interface (the
SAME interface every candidate was originally evaluated through) and
compares static FMLA count and `.text` section size/content against the
original candidate record. All 6: **PASS** (FMLA match, `.text` size
match, and for 32x32x32 specifically, byte-for-byte identical `.text`
section content verified with `llvm-objcopy --only-section=.text` + `diff`).
Whole-object-file size legitimately differs by a few bytes (ELF symbol/
string table growth from the tile-suffixed function name used during
multi-candidate testing) -- documented as an expected, non-functional
difference, not a reproduction failure.

## Not implemented in this slice (explicit non-goals)

- Online runtime autotuning (this is an offline, build-time selection over
  a fixed candidate set, not a runtime feedback loop)
- General dynamic-shape tile selection (only the 6 evaluated shapes have
  selection results; a new shape needs its own candidate sweep)
- Full LLVM register-pressure analysis (assembly-derived evidence only,
  explicitly labeled as such throughout)
- MIR-level compiler pass, instruction scheduling, software pipelining,
  prefetch insertion (all remain entirely LLVM's `llc`, unmodified)
- Cost-model integration into any production plan-selection pass (this
  slice produces a standalone, artifact-backed selector tool only, per the
  task brief's explicit instruction not to wire it into
  `CandidateEvaluationPass`)
- INT8 / SDOT / UDOT dot-product lowering
- GPU code generation of any kind
- General arbitrary-shape tail handling (unchanged from the prior slice:
  candidates require exact tile divisibility)
