# LLVM MIR Analysis for AArch64 Tile Candidates

Target: Raspberry Pi 5 / Cortex-A76
Execution type: Real hardware for every object tested
Analysis type: Real LLVM Machine IR at four pipeline boundaries
  (post-instruction-selection, pre-register-allocation,
  post-register-allocation, post-prologue/epilogue), extracted via LLVM's
  own `llc --stop-after=<pass>` -- no custom passes, no LLVM patches, no
  custom register allocator.

## Achievement statement

Captured and analyzed LLVM pre-RA and post-RA MIR for 8 AArch64 tile
candidates (plus 3 of them re-analyzed under `-regalloc=fast` for an
11-configuration total), quantifying up to 385 pre-RA virtual registers
and correcting one assembly-derived spill-count claim (previously "3
spills, 0 reloads" for 8x8x8/tile-8x8x8 -- MIR shows the true figure is 1
real allocator spill + 1 real reload) using LLVM's own explicit
`type: spill-slot` + empty-`callee-saved-register` stack-object
annotation. This correction did NOT change any prior tile-selection
outcome. MIR evidence also explained why 32x32x32 selected tile 8x8x8 over
three zero-spill alternatives: not register-pressure avoidance (all four
32x32x32 candidates analyzed have zero real spills, including the winner,
which has the HIGHEST pre-RA register count of the four), but dynamic loop
iteration count -- the 8x8x8 tile requires only 64 loop-body executions to
cover the 32x32x32 output vs. 128 for the two second-place tiles and 512
for the slowest, directly explaining the measured latency ordering.

## Truth boundary: what is project-owned vs. LLVM-owned here

Identical boundary to every prior slice in this backend-codegen series,
now extended one level deeper: the project owns MIR *extraction*
(choosing which `--stop-after` pass boundaries to dump and why),
*analysis* (parsing the resulting MIR text into structured metrics), and
*measurement* (correlating those metrics with real Raspberry Pi latency).
**LLVM owns the entire register allocator** -- instruction selection,
pre-RA scheduling, live-interval computation, coalescing, greedy (or fast)
allocation, virtual-register rewriting, spill-code insertion, and
prologue/epilogue generation are all stock, unmodified LLVM 21.1.8. No
project code makes an allocation decision, inserts a spill, or selects a
register anywhere in this slice.

## LLVM Machine-Pass Discovery

Pass names were discovered empirically via
`llc -mtriple=aarch64-linux-gnu -mcpu=cortex-a76 -O2 -debug-pass=Structure`
on this exact LLVM 21.1.8 build -- NOT assumed from another LLVM version.
Four boundaries were selected and validated:

| Stage | `--stop-after=` | Verified property |
|---|---|---|
| post_isel | `finalize-isel` | Virtual registers present, AArch64 machine opcodes present, right after instruction selection |
| pre_ra | `machine-scheduler` | The LAST pass before the allocator runs. Verified: 100% virtual registers still present (e.g. 385 for 32x32x32/tile-8x8x8), 0 physical vector-register mentions |
| post_ra | `virtregrewriter` | Verified: 0 virtual registers, real physical registers (`$q0`-`$q31`, `$x0`-`$x30`) throughout. **Critically, `--stop-after=greedy` alone was tested and found NOT to satisfy this** -- see below |
| post_prologue_epilogue | `prologepilog` | Stack frame finalized; callee-saved save/restore visible |

**A real toolchain finding, not assumed:** `--stop-after=greedy` was tested
first, on the hypothesis that it would be the natural "post-RA" boundary.
It is NOT: the printed MIR after `-greedy` alone still shows 100% virtual
registers (185 distinct vregs for an initial test candidate, 0 physical
vector-register mentions) -- indistinguishable from pre-RA MIR by this
measure. The `-greedy` pass populates an internal `VirtRegMap` but does
not rewrite `MachineOperand`s; that rewriting is a SEPARATE pass,
`-virtregrewriter`, which runs immediately after. This is documented here
because the task's own instructions explicitly require validating this
("Do not call an artifact post-RA unless physical registers have been
assigned and allocator effects are visible") and the naive choice would
have failed that check silently.

## Existing Candidate Reproduction

32x32x32/tile-8x8x8 recompiled via the unmodified pipeline: object SHA-256
`6690f743...` -- byte-identical to the committed selection artifact's
recorded hash. Static FMLA 128, matches. All 8 mandatory candidates'
freshly-compiled objects were verified byte-size-identical to the
committed `tile_candidates/object_hashes.txt` entries, and two were
verified full SHA-256-identical (32x32x32/tile-8x8x8 and
8x8x8/tile-8x8x8) -- justifying reuse of the already-committed Raspberry
Pi correctness/benchmark data for all 8 greedy-allocator candidates rather
than re-measuring.

## MIR Extraction Pipeline

`tools/extract_aarch64_candidate_mir.py` -- takes LLVM IR + cpu + shape +
tile + output dir (+ optional `--regalloc greedy|fast`), runs `llc` four
times (one per `--stop-after` boundary above) plus a full
`-filetype=asm`/`-filetype=obj` run, writing all outputs to the given
directory. See its module docstring for the full pipeline-boundary
rationale (reproduced above) and `commands.txt` for exact invocations.

## Pre-RA MIR Evidence

Verified for every one of the 8 candidates: `registers:` YAML section
lists every virtual register with an explicit `class:` (e.g. `fpr128` for
NEON/vector, `gpr64`/`gpr64common`/`gpr64all`/`gpr64sp`/`gpr32` for
integer); the `body:` section contains AArch64 post-isel/post-schedule
opcodes (`FMLAv4i32_indexed`, `LDRQui`, etc.) operating on `%N:class`
virtual register operands; zero `$q*`/`$x*` physical register mentions
appear anywhere in the body. See `representative_32x32x32_tm8_tn8_tk8_pre_ra.mir`
(complete, 385 virtual registers, 242 of them `fpr128`).

## Post-RA MIR Evidence

Verified for every one of the 8 candidates: zero virtual registers remain;
every operand is a physical register (`$q0`-`$q31`, `$x0`-`$x30`, etc.);
where a real allocator spill exists (exactly one candidate in this set --
see below), the `stack:` YAML list contains an entry with `type:
spill-slot` and an EMPTY `callee-saved-register` field, and the body
contains a matching `STRQui ... %stack.N` / `LDRQui %stack.N` pair. See
`representative_32x32x32_tm8_tn8_tk8_post_ra.mir` (complete, 0 vregs, 28
distinct physical `$qN` registers referenced, `stack: []` -- zero real
spills) and `spill_example_8x8x8_tm8_tn8_tk8_post_ra.mir` (complete, the
one real spill+reload pair, at MIR body lines matching the manually
cross-checked assembly).

Final assembly cross-check: real NEON `fmla` present in every candidate's
`.s` output (static counts unchanged from the corresponding pre-MIR-slice
committed values -- register allocation does not change instruction
selection), and object hashes verified byte-identical to the previously
committed, Pi-tested objects (Existing Candidate Reproduction, above).

## MIR Analysis Tool

`tools/analyze_aarch64_candidate_mir.py` -- parses MIR text (handling
LLVM's multi-line YAML flow-mapping entries, see "A parser bug found and
fixed" below), splits multi-machine-function files (kernel +
`_mlir_ciface_` wrapper) and analyzes the kernel function by default,
extracts virtual-register/register-class/frame-object/spill/reload/copy
metrics per stage, and emits both structured JSON and a human-readable
report. Unit-tested against small synthetic MIR fixtures (see
`tests/test_aarch64_mir_analysis.py`).

### Two real parser bugs found and fixed during this slice's own development

1. **Multi-line YAML entries.** LLVM wraps each `stack:`/`fixedStack:`
   flow-mapping entry (`- { id: 0, type: spill-slot, ... }`) across two or
   more physical lines once it gets long -- `callee-saved-register`
   routinely lands on the continuation line, never the same line as `id:`.
   An initial line-by-line field extraction therefore missed EVERY
   callee-saved annotation, which almost caused a serious over-count: with
   this bug, callee-saved slots and real allocator spills were
   indistinguishable (both showed as bare `type: spill-slot` with no
   visible `callee-saved-register`), which would have misreported ~12
   ordinary ABI-preservation slots per candidate as "allocator spills."
   Fixed by joining each `- { ... }` entry across its continuation lines
   (brace-balance tracking) before field extraction.
2. **Opcode-position assumption.** An initial reload-detection regex
   anchored the opcode to the start of the line (`^\s*LDR...`). This
   correctly found spill STORES (`STRQui ...` has no destination operand,
   so the mnemonic IS line-initial) but silently missed every spill
   RELOAD (`$q0 = LDRQui ...` -- the mnemonic follows a destination
   assignment, never line-initial). Fixed by searching for the opcode
   after an optional `= ` prefix, not anchored to line start. Caught by
   manually cross-checking the raw MIR for the 8x8x8/tile-8x8x8 candidate
   against the tool's own output before trusting it across the full set.

Both bugs are recorded here, with their real consequences, rather than
silently fixed and forgotten -- per this project's standing documentation
practice for tooling-level findings.

## Candidate Set

8 mandatory `(shape, tile)` combinations, all analyzed at pre_isel/pre_ra/
post_ra/post_prologue_epilogue with `-regalloc=greedy`; 3 of them
additionally analyzed with `-regalloc=fast`:

- 8x8x8: tile 4x8x8 (selected), tile 8x8x8 (spill contrast)
- 32x32x32: tile 4x8x8, tile 8x8x8 (selected), tile 8x8x4, tile 4x4x4
- 64x64x64: tile 4x8x8 (selected), tile 8x8x8

## Virtual-Register and Register-Class Metrics

Full per-candidate data in `mir_analysis_results.json`. Register classes
observed (real LLVM 21 AArch64 names, not renamed/generalized in the raw
data): `fpr128` (128-bit NEON/vector -- the register class that matters
for this workload), `gpr64`, `gpr64common`, `gpr64all`, `gpr64sp`,
`gpr32`. Pre-RA virtual register totals range 173 (8x8x8/tile-4x8x8) to
387 (64x64x64/tile-8x8x8).

## Register-Pressure Methodology

LLVM 21.1.8 on this toolchain is an Optimized/Release build --
`-debug-only=regalloc` and similar debug-logging flags produce no output
(debug logging is compiled out of Release builds). No LLVM-reported exact
peak register-pressure metric was available. This is recorded as a
limitation, not worked around by fabricating one. Instead, a **clearly
labeled MIR-derived approximation** is computed: a linear scan over the
pre-RA body's def/kill order (using the `killed` flags already present in
the MIR text), tracking how many distinct `fpr128`-class virtual registers
are simultaneously live. This is explicitly NOT a dataflow analysis (no
CFG join-point or loop back-edge handling) and is reported under the key
`approx_peak_live_vector_registers` -- never referred to as exact LLVM
register pressure, either in the tool's output or in this document.

## Spill, Reload, Copy, and Stack-Slot Results

**7 of 8 candidates: zero real allocator spills/reloads.** The one
exception, 8x8x8/tile-8x8x8, has exactly 1 spill store + 1 reload load
(1 spill slot, 16 bytes = one 128-bit NEON register) -- see the dedicated
investigation below. Classification method: a stack object counts as a
real allocator spill only if `type: spill-slot` AND `callee-saved-register`
is empty; this correctly excludes the 7-13 callee-saved ABI slots every
candidate's `post_prologue_epilogue` stage shows (LLVM tags those with the
SAME `type: spill-slot` label, distinguished only by a non-empty
`callee-saved-register` field -- see "parser bugs" above). Copies removed
by RA (coalescing): 0-3 per candidate, small as expected for this
already-tightly-scheduled microkernel code.

## Greedy vs Fast Register Allocator

`-regalloc=fast` is accepted by LLVM 21.1.8 for this target/pipeline with
no diagnostics. Full results in `register_allocator_comparison.json`; summary:

| Shape | Tile | Allocator | Spills | Reloads | Stack Frame | Median ms | vs. greedy |
|---|---|---|---|---|---|---|---|
| 8x8x8 | 8x8x8 | greedy | 1 | 1 | 112 | 0.000092 | -- |
| 8x8x8 | 8x8x8 | fast | 3 | 3 | 96 | 0.000092 | +0.0% |
| 32x32x32 | 8x8x8 | greedy | 0 | 0 | 96 | 0.002574 | -- |
| 32x32x32 | 8x8x8 | fast | 76 | 29 | 992 | 0.003834 | **+48.96%** |
| 32x32x32 | 4x8x8 | greedy | 0 | 0 | 128 | 0.002926 | -- |
| 32x32x32 | 4x8x8 | fast | 44 | 29 | 528 | 0.004167 | **+42.41%** |

Greedy never underperforms fast in this experiment; at 8x8x8 the choice is
immaterial to measured latency, and at 32x32x32 greedy is 42-49% faster.
No fast-allocator object is committed as a production variant -- this is
an analysis experiment per the task's explicit instruction.

## Raspberry Pi Correctness

All 6 (candidate x allocator) pairs used for the greedy-vs-fast
experiment: 1-call + 1000-call correctness, zero failures. 200-cycle
mixed allocator/candidate stress (alternating greedy and fast objects for
the same 3 candidates, 1200 total calls with allocator noise between
calls): zero failures, zero guard/descriptor corruption. The remaining 8
mandatory MIR-analysis candidates (greedy-only) reuse already-committed,
already-Pi-validated correctness data since their objects are
byte-identical (see Existing Candidate Reproduction).

## Raspberry Pi Performance Correlation

See `performance_correlation.json` for the full table and correlation
findings. Headline: spills/reloads only correlate with measured latency
once the surrounding loop has enough dynamic iterations to make the spill
traffic's cost visible (compare the two investigations below); virtual
register count and approximate peak pressure do NOT predict which tile
wins at a shape (the winning 32x32x32 tile has the HIGHEST register count
of the four analyzed, with zero spills).

## 8x8x8 Spill-Contrast Investigation

Original claim (prior slice, assembly-derived): 8x8x8/tile-8x8x8 had "3
assembly-derived spills, 0 reloads." **MIR evidence corrects this to 1
real spill store + 1 real reload load**, tied to one 16-byte spill slot
(`type: spill-slot`, empty `callee-saved-register`) whose store and load
are both inside the function's single collapsed compute block (this
shape/tile combination has `scf_for_count=0` per the tiled-vectorization
slice's own structural metrics -- tile equals shape in every dimension,
so the loop nest canonicalizes away entirely, leaving one straight-line
128-FMLA block). Answering the task's specific questions:

1. **Were the 3 reported spills genuine allocator spills?** No -- MIR
   shows exactly 1 genuine spill+reload pair. The original assembly-level
   heuristic (counting `str q*, [sp` instruction lines) over-counted,
   likely by conflating this spill with adjacent non-spill stack traffic
   in the same instruction window (the assembly-derived tool never
   distinguished frame-index provenance the way MIR analysis does).
2. **Which virtual register spilled?** An `fpr128`-class (NEON/vector)
   virtual register, confirmed by the `STRQui killed renamable $q0,
   %stack.0` / `LDRQui %stack.0` instruction pair (both operate on a
   128-bit `q` register).
3. **Register class?** `fpr128` (vector), not a GPR.
4. **Which stage introduced the stack slot?** By `--stop-after=virtregrewriter`
   (post-RA), the slot and both instructions are already present; by
   `--stop-after=machine-scheduler` (pre-RA) there is no stack slot at
   all -- the spill is introduced strictly within the register-allocation
   pass sequence (`-greedy` decides it, `-virtregrewriter` materializes it),
   consistent with standard LLVM behavior.
5. **Reloaded inside the hot compute region?** Yes -- both the store and
   the load are surrounded on all sides by `FMLAv4i32_indexed` instructions
   in the same single basic block (verified by direct inspection of the
   MIR body, not inferred).
6. **Why did measured latency remain effectively equal (0.000092ms for
   both tiles)?** One spill+reload pair is 2 extra memory instructions
   inserted once into a function whose entire body executes exactly once
   per call (no loop -- `scf_for_count=0`). At this shape, both the
   overall instruction count (~250) and the wall-clock latency (~92ns) are
   already dominated by fixed costs (function-call overhead, ABI argument
   marshalling, malloc/free for the output buffer) rather than by the
   compute body itself -- 2 extra instructions do not move the needle
   against a ~92,000-picosecond budget with a real timer resolution floor.
7. **Was the shape too small for spill cost to dominate?** Yes -- directly
   supported by the fact that the SAME tile (8x8x8) at the SAME single-block
   structure but under `-regalloc=fast` (which triples the spill count to
   3+3) STILL shows an identical 0.000092ms median (see Greedy vs Fast
   table above) -- the spill count can change 3x with zero latency effect
   at this shape.
8. **Did reduced loop overhead compensate for spill traffic?** Not
   applicable in the usual sense (there is no loop to reduce -- trip count
   is 1), but this IS the mechanism: a single-execution straight-line
   block has no repeated per-iteration spill cost to accumulate, unlike
   the 32x32x32/tile-8x8x8 fast-allocator case (Greedy vs Fast table),
   where a real loop repeats the spill traffic every iteration and the
   cost becomes clearly visible (+48.96%).

**Correction applied**: the artifact and documentation (`ARCHITECTURE_STATUS.md`,
`KNOWN_GAPS.md`, and this directory) now state the true figure (1 spill, 1
reload) rather than repeating the prior assembly-derived "3 spills, 0
reloads" claim. See Selection-Policy Impact below for whether this changes
the prior tile selection (it does not).

## 32x32x32 Winner Investigation

Four candidates compared (all `-regalloc=greedy`, all zero real spills):

| Tile | Median ms | Static FMLA | Pre-RA VRegs | Approx Peak | Loop-body executions* |
|---|---|---|---|---|---|
| 4x4x4 | 0.004130 | 16 | 174 | 29 | 512 |
| 4x8x8 | 0.002926 | 64 | 262 | 65 | 128 |
| 8x8x4 | 0.002982 | 64 | 305 | 97 | 128 |
| **8x8x8** | **0.002593** | 128 | 385 | 113 | **64** |

*Loop-body executions = (32/TM)·(32/TN)·(32/TK), i.e. how many times the
tiled microkernel body runs to cover the full 32x32x32 output; each
execution's total FMA count × execution count = 8,192 for all four (same
total arithmetic, verified consistent with M·N·K/4).

**Why 8x8x8 won, checked against every candidate factor the task lists:**

- **Larger accumulator tile / fewer loop iterations**: CONFIRMED as the
  primary factor. 8x8x8 requires only 64 microkernel-body executions vs.
  128 for the two second-place tiles (whose latencies, 0.002926/0.002982,
  are indeed close to each other, both ~2x slower loop-trip-count than the
  winner) and 512 for the slowest (4x4x4, ~1.6x slower latency than the
  winner, matching its 8x higher trip count qualitatively though not
  linearly -- loop overhead is a fraction of total cost, not all of it).
- **Register pressure / spill avoidance**: NOT a differentiator -- all
  four have zero real spills; the winner has the HIGHEST register count
  and pressure of the four, yet still doesn't spill. LLVM's greedy
  allocator handled up to 385 virtual registers / 113 approximate peak
  live vector registers into 32 physical registers without difficulty for
  every 32x32x32 candidate analyzed.
- **Branch reduction**: NOT a differentiator -- static branch count is 11
  for all four tiles (same 3-level loop-nest structure; the tile size
  changes trip counts, which is a dynamic property MIR/static-branch-count
  does not capture, not the static branch count).
- **Instruction count**: works AGAINST the naive hypothesis -- the winner
  (8x8x8) has the MOST static instructions (358) of the four, not the
  fewest. A larger, less-frequently-executed body wins over a smaller,
  more-frequently-executed one.
- **Greater ILP**: plausible contributing factor (larger tile body gives
  the scheduler and superscalar pipeline more independent FMLA chains per
  loop iteration) but not independently isolated in this slice -- flagged
  as a secondary hypothesis consistent with, but not proven beyond, the
  loop-iteration-count evidence above.

**Conclusion**: the 32x32x32 tile-8x8x8 win is explained by amortizing
per-iteration loop overhead (address computation, branch, loop-control)
over a larger unit of work (64 vs. 128-512 executions), NOT by avoiding
register pressure (irrelevant here -- nobody in this set spills) and NOT
by reducing static code size (the winner is the largest object of the
four, 2,616 bytes vs. 1,808-2,344 for the others).

## Assembly-Derived Metric Corrections

One correction made this slice: 8x8x8/tile-8x8x8's spill count, corrected
from the prior slice's assembly-derived "3 spills, 0 reloads" to the
MIR-verified "1 spill, 1 reload" (see investigation above). All other
assembly-derived spill counts checked in this slice's 8-candidate subset
(0 spills for the other 7) were CONFIRMED by MIR, not contradicted.

## Selection-Policy Impact

Recomputing the existing scoring formula (`score = latency/best_latency +
0.20·spills + 0.20·reloads + 0.10·(bytes/smallest_bytes) +
register_penalty`) for 8x8x8 with the corrected spill/reload counts:

- tile 4x8x8 (unchanged): score = 1.0 + 0 + 0 + 0.10·(1704/1704) + 0 = **1.10**
- tile 8x8x8, OLD (3 spills, 0 reloads): score = 1.0 + 0.60 + 0 + 0.121 + 0.20 = 1.921
- tile 8x8x8, CORRECTED (1 spill, 1 reload): score = 1.0 + 0.20 + 0.20 + 0.121 + 0.20 = **1.721**

**Selection does NOT change**: tile 4x8x8 remains selected for 8x8x8 by a
wide margin either way (1.10 vs. 1.721-1.921). The correction makes the
margin narrower but does not flip the outcome. For 32x32x32 and 64x64x64,
every analyzed tile shows zero real spills (matching the prior
assembly-derived zero-spill findings for those candidates), so no score
changes and no selection changes there either.

This slice's deep-analysis scope covers 3 of the 6 shapes (8x8x8,
32x32x32, 64x64x64, per the task's own "Required deep-analysis candidates"
list). 16x16x16, 32x64x32, and 64x32x64 were NOT re-verified with real MIR
in this slice -- their existing assembly-derived spill counts (all zero,
per the tile-candidates artifact) are unconfirmed by MIR and should be
treated with the same caveat the original artifact already carries,
pending a future slice extending this analysis to the remaining shapes.

## Not implemented in this slice (explicit non-goals)

- Custom register allocator (LLVM's greedy/fast allocators are used
  entirely unmodified)
- Production LLVM MachineFunction pass (this is offline analysis tooling,
  not a compiler pass)
- Exact general-purpose register-pressure model (the
  `approx_peak_live_vector_registers` metric is an explicitly labeled,
  simple linear-scan approximation, not a dataflow analysis)
- Instruction scheduling transformation, software pipelining, prefetch
  insertion (all remain LLVM's own, unmodified)
- Online autotuning, dynamic-shape tile selection (unchanged from the
  prior slice)
- INT8/SDOT/UDOT lowering, GPU code generation (unchanged, out of scope)
- Full MIR re-analysis of the 34 candidates outside this slice's 8-candidate
  deep-analysis subset (by design -- see Scope in the task brief)
