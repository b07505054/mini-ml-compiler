# Architecture Status

Last verified: 2026-07-15.

This table is a maturity assessment, not a roadmap promise and not a percentage score.

| Area | Status | Evidence | Limits |
|---|---|---|---|
| Portable Pi CPU execution | production/canonical + measured | P1B/P1C/P1D/P1D.1 reports, Runtime P1D evidence | fixed FP32 fused MatMul + Bias + ReLU scope |
| ImplementationCandidate architecture | canonical for active paths | A1-A5 tests/reports, E3 XNNPACK provider | not universal across every old decision system |
| Provider/feasibility/policy separation | production for portable path, evaluation-canonical for E3 | `PortableCPUProvider`, `XNNPACKCandidateProvider` | Triton/AWQ not production-integrated |
| E3 same-XNNPACK comparison | measured narrow comparison | runtime `results/executorch_e3` | static X1 winner, not complex learned policy |
| Triton provider | shadow | A6 report/tests | unresolved IR bridge, no Runtime dispatch |
| AWQ/vLLM | executable parallel path + measured serving | AWQ artifact, vLLM materialization, runtime traces | no accuracy/perplexity calibration, contradictory per-op/global plan details |
| Capability DB | partial declared source | `ml-platform-capabilities` profiles | not sole source of truth; compiler-local profiles are richer for Pi paths |
| Implementation IR | partial | HIR and limited boundary materialization | memory spaces, DMA, synchronization, NPU command regions incomplete |
| Runtime boundary | strong for canonical paths | strict adapter validation and E3 contract validation | older simulations/evaluation paths must stay scoped |
| AArch64 native codegen: generic baseline (fused MatMul-Bias-ReLU only) | narrow real implementation + measured on real Raspberry Pi 5 hardware | `artifacts/backend_codegen/aarch64_matmul_bias_relu/` (superseded object hashes; see vectorized artifact for current), `artifacts/backend_codegen/aarch64_matmul_bias_relu_vectorized/generic/` (current); `mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh --variant generic` | one op (`hir.fused_matmul_bias_relu`), three fixed shapes (8x8x8/16x16x16/32x32x32), no project-owned target-specific instruction selection/scheduling/register allocation, generated code slower than an `-O2` scalar C++ reference on the same device; not wired into the runtime's `OpRegistry` dispatch or the compiler's candidate/cost-model selection |
| AArch64 native codegen: MLIR vectorization slice (fused MatMul-Bias-ReLU only) | narrow real implementation + measured on real Raspberry Pi 5 hardware, real NEON `fmla` confirmed | `artifacts/backend_codegen/aarch64_matmul_bias_relu_vectorized/` (both variants' generated `.mlir`/`.ll`/`.s`/`.o`, objdump, vector-dialect intermediate, repeated-call and mixed-shape correctness logs, benchmark/metrics JSON, Pi device state); `mlir_passes/transforms/vectorize_matmul_bias_relu.mlir`; `tools/run_backend_codegen_vectorized_pi_integration.sh` | project-owned contribution is invoking upstream MLIR's `vectorize_children_and_apply_patterns` Transform op at this point in the project's own pipeline, not a project-authored instruction selector; all NEON `fmla` selection, register allocation, and scheduling remain LLVM's `llc`, unmodified; one op, three fixed static shapes, fully unrolled (object size/instruction count scale with M*N*K -- SUPERSEDED for larger shapes by the tiled slice below, kept for the three original shapes as a comparison baseline); not wired into the runtime's `OpRegistry` dispatch or the compiler's candidate/cost-model selection |
| AArch64 native codegen: tiled vector microkernel slice (fused MatMul-Bias-ReLU only) | narrow real implementation + measured on real Raspberry Pi 5 hardware, real NEON `fmla` confirmed, bounded code size confirmed | `artifacts/backend_codegen/aarch64_matmul_bias_relu_tiled/` (full artifacts for the 32x32x32 representative shape, metrics/correctness/benchmark JSON for all 6 shapes, disassembly excerpts, register-pressure precheck) | project-owned contribution is composing stock upstream MLIR Transform ops (tile_using_for + fuse_into_containing_op + vectorize_children_and_apply_patterns) into a fixed 4x8x8 tile, not a project-authored tiling algorithm or instruction selector; the fixed 4x8x8 choice is now backed by comparative evidence (see the tile-candidate row below) rather than being the only tile evaluated; requires exact tile divisibility (no tail handling); not wired into the runtime's `OpRegistry` dispatch or the compiler's candidate/cost-model selection |
| AArch64 native codegen: tile-candidate selection slice (fused MatMul-Bias-ReLU only) | narrow real implementation + measured on real Raspberry Pi 5 hardware across 42 candidates, artifact-backed offline selection | `artifacts/backend_codegen/aarch64_matmul_bias_relu_tile_candidates/` (candidate_results.json, selected_tiles.json, scoring_policy.json, per-shape benchmarks, register-pressure summary, one full representative candidate, disassembly excerpts for two instructive losing candidates); `mlir_passes/transforms/tile_vectorize_matmul_bias_relu.template.mlir`; `tools/generate_aarch64_matmul_tile_candidates.py`, `tools/analyze_register_pressure.py`, `tools/select_aarch64_matmul_tile_candidate.py`, `tools/reproduce_selected_tile_candidate.py` | replaces the single fixed 4x8x8 tile with 42 measured candidates (7 tiles x 6 shapes, all legal, all correct); selection is OFFLINE and shape-specific -- a new shape requires its own candidate sweep, no online/runtime autotuning, no dynamic-shape generalization, not wired into any production plan-selection pass; register-pressure evidence is assembly-derived, not exact LLVM liveness analysis; no single tile is universally best (2 tiles split the 6 shapes 3/3, both sharing TN=8/TK=8) |

See `PROJECT_MATURITY.md` for the four-pillar assessment.

## AArch64 native codegen detail (2026-07-15)

This is the project's first path that produces and executes real machine
code rather than selecting among precompiled handwritten kernels. Scope and
truth boundaries, precisely:

**Newly implemented:** one narrow HIR -> Linalg -> LLVM dialect -> LLVM IR ->
AArch64 object -> Raspberry Pi execution path for `hir.fused_matmul_bias_relu`,
covering three fixed static shapes. The HIR -> LLVM dialect stage reuses the
existing, already-FileCheck-verified `hir-matmul-bias-relu-to-linalg` pass
and stock upstream MLIR conversion passes (no new project-owned dialect
lowering was added). New work: `mlir-translate` to textual LLVM IR, `llc`
to AArch64 assembly and object code, a `buffer-deallocation-pipeline`
addition to the pass pipeline (a real memory leak in the intermediate
matmul buffer was found and fixed during hardware validation -- see the
artifact README), linking on the Raspberry Pi itself with its installed
`g++` (no LLVM toolchain was installed on the Pi), and measured correctness
and latency on real hardware across three shapes, all numerically correct
(max absolute error 1.8e-07 or better) against a scalar C++ reference.

**Still not implemented** (unchanged by this slice; do not infer otherwise
from the presence of a working AArch64 object):
- Project-owned target-specific instruction selection (no NEON/SDOT/FMLA
  intrinsic lowering; the generated code is the generic path LLVM produces
  from unmodified Linalg-derived loops)
- Instruction scheduling / software pipelining
- Register allocation ownership (delegated entirely to LLVM's `llc`, and not
  yet measured -- no register-pressure or spill report exists for this path)
- MIR-level passes
- Generated-code runtime loading (the runtime's `OpRegistry` still only
  dispatches statically-linked handwritten kernels; this AArch64 object is
  not wired into it)
- General operator coverage (only `hir.fused_matmul_bias_relu`, not
  `hir.fused_rmsnorm` or any quantized op, has been taken through this path)
- GPU code generation of any kind

**Correction (2026-07-15, later in the same day):** a real repeated-call
correctness bug was subsequently found in the shared lowering pass this
slice uses (`hir-matmul-bias-relu-to-linalg`, in
`mlir_passes/lib/MatMulBiasReluFusionPass.cpp`) and is fixed as of the
vectorized slice below. The matmul accumulator was fed from an
un-zero-initialized `tensor.empty()`; on heap-address reuse across repeated
calls, this silently accumulated stale data from a prior call's own output
buffer, producing wrong (growing) results after the first call in some
allocator states. This affected the generic path documented above from the
start, not just the vectorized path -- Stage 1's original benchmark loop
here timed repeated calls but never re-verified their correctness after the
first, so it went undetected until the vectorization slice's
repeated-call/mixed-shape stress testing surfaced it. The object hashes
under `artifacts/backend_codegen/aarch64_matmul_bias_relu/object_hashes.txt`
reflect the pre-fix build and are superseded by
`artifacts/backend_codegen/aarch64_matmul_bias_relu_vectorized/generic/`.
See that artifact's README for the full root-cause writeup.

## AArch64 vectorized codegen detail (2026-07-15)

Builds on the slice above by adding one project-owned pipeline stage:
`mlir_passes/transforms/vectorize_matmul_bias_relu.mlir`, a Transform
dialect script invoked between `hir-matmul-bias-relu-to-linalg` and
bufferization. See
`artifacts/backend_codegen/aarch64_matmul_bias_relu_vectorized/README.md`
for the full truth-boundary explanation, root-cause writeup for the
accumulator bug above, correctness evidence, and performance numbers; only
the summary is repeated here.

**Newly implemented:** the project-owned MLIR vectorization stage rewrites
`linalg.matmul`/`linalg.generic` (as already produced by the existing
`hir-matmul-bias-relu-to-linalg` pass) into `vector.contract` /
`vector.transfer_read` / `vector.transfer_write`, using stock upstream MLIR
Transform ops -- the project's contribution is choosing to invoke this
transform at this point in its own pipeline for its own op, not authoring
new matching/rewrite logic. `convert-vector-to-llvm{vector-contract-
lowering=outerproduct}` then lowers this to LLVM IR in a shape that causes
LLVM's own (unmodified) AArch64 instruction selector to choose real NEON
`fmla` (fused multiply-accumulate) instructions -- confirmed by disassembly
(128/1024/8192 `fmla` instructions for the 8x8x8/16x16x16/32x32x32 shapes
respectively). Measured on the real Raspberry Pi, the vectorized kernel is
correct (`all_correct: true`, all shapes) and ~20x-30x faster (median
latency) than the generic (unvectorized) AArch64 path, and for 16x16x16 and
32x32x32 is also faster than the pre-existing handwritten
`fused_matmul_add_relu` CPU kernel. Repeated-call (1000 calls/shape/variant)
and mixed-shape/mixed-variant (500 cycles, 6000 calls, with allocator noise)
correctness was verified with zero tolerance for failure on real hardware.

**Still not implemented** (unchanged by this slice; do not infer otherwise
from the presence of real NEON `fmla` instructions):
- Any project-owned/custom LLVM SelectionDAG or GlobalISel instruction-
  selection pattern (the `fmla` selection is entirely LLVM's own, from
  stock `vector.contract` lowering)
- INT8 / SDOT / UDOT dot-product lowering
- Instruction scheduling / software pipelining beyond LLVM's own `llc`
- Register allocation ownership (still entirely LLVM's `llc`)
- MIR-level passes
- Tiling: these fixed static shapes are fully unrolled by the Transform op
  used, so object size and instruction count scale with M*N*K (e.g.
  32x32x32's vectorized object is ~64x the instruction count of 8x8x8's).
  **Addressed for larger/practical shapes by the tiled slice below** (this
  variant is retained only as a comparison baseline for the three original
  shapes, not because the scaling problem is unsolved).
- Generated-code runtime loading (same gap as the generic path above)
- General operator coverage (same gap as the generic path above)

## AArch64 tiled vector microkernel codegen detail (2026-07-16)

Replaces whole-shape vectorization's unbounded code-size scaling (above)
with a fixed 4(M)x8(N)x8(K) register-tile microkernel, reused via real
`scf.for` loops rather than fully unrolled. Full truth-boundary
explanation, register-pressure analysis, and evidence in
`artifacts/backend_codegen/aarch64_matmul_bias_relu_tiled/README.md`; only
the summary is repeated here.

**Newly implemented:** a project-owned Transform-dialect script
(`mlir_passes/transforms/tile_vectorize_matmul_bias_relu.mlir`) that tiles
the bias+relu consumer, fuses the matmul and zero-init producers into that
tile, tiles the fused matmul's K (reduction) dimension separately, then
vectorizes only the small resulting ops -- composing four stock upstream
MLIR Transform ops, none project-authored as algorithms. Reduces the
32x32x32 generated object from 113,216 bytes (fully-unrolled) to 2,128
bytes (53.2x smaller), and enables 64x64x64 (2,104 bytes) and two
non-square shapes (32x64x32, 64x32x64) that were never practical to
fully-unroll. Measured on the real Raspberry Pi, the tiled kernel is
FASTER than the fully-unrolled kernel it replaces at every shape where both
exist (median latency ratio 0.63-0.90), likely because the fully-unrolled
kernel's larger code does not fit Cortex-A76's L1 instruction cache.
Repeated-call (1000 calls/shape/variant, 12 pairs) and mixed-shape/
mixed-variant (500 cycles, 6,000 calls, all 6 shapes, with allocator noise)
correctness verified with zero tolerance for failure on real hardware.

**Still not implemented** (unchanged by this slice):
- Any project-owned/custom LLVM SelectionDAG or GlobalISel instruction-
  selection pattern (the `fmla` selection is entirely LLVM's own)
- INT8 / SDOT / UDOT dot-product lowering
- Instruction scheduling / software pipelining beyond LLVM's own `llc`
  (LLVM's scheduler is observed to interleave two K-sub-steps for ILP,
  using all 32 vector registers with zero spills -- this is LLVM's choice,
  not a project-owned scheduling pass)
- Register allocation ownership (still entirely LLVM's `llc`); no full
  register-pressure/spill-cost analysis was performed, only a design-time
  precheck (see the artifact's `register_pressure_32x32x32.txt`)
- MIR-level passes
- General arbitrary-shape tail handling: shapes not divisible by 4/8/8 are
  rejected outright by `compile_hir_matmul_bias_relu_aarch64.sh
  --variant tiled-vectorized`, not silently miscompiled
- Cost-model-driven tile selection at build time for this one fixed 4x8x8
  default. **Addressed as a standalone, offline, evidence-backed capability**
  by the tile-candidate selection slice below (still not integrated into
  any production compiler pass).
- Prefetching
- Generated-code runtime loading (same gap as the other AArch64 slices)
- General operator coverage (only `hir.fused_matmul_bias_relu`)
- GPU code generation of any kind

## AArch64 tile-candidate selection detail (2026-07-16)

Replaces the single hard-coded 4x8x8 tile (above) with a real
multi-candidate, evidence-driven selection flow: 7 tile configurations x 6
matrix shapes = 42 candidates, all compiled through the real, unmodified
LLVM AArch64 backend and executed on the real Raspberry Pi. Full
methodology, scoring policy, and results in
`artifacts/backend_codegen/aarch64_matmul_bias_relu_tile_candidates/README.md`;
only the summary is repeated here.

**Newly implemented:** a parameterized Transform-dialect template
(`mlir_passes/transforms/tile_vectorize_matmul_bias_relu.template.mlir`,
replacing the prior slice's single fixed-tile file) plus a build-time
candidate-generation, register-pressure-analysis, and scoring/selection
pipeline (`tools/generate_aarch64_matmul_tile_candidates.py`,
`tools/analyze_register_pressure.py`,
`tools/select_aarch64_matmul_tile_candidate.py`). All 42 candidates are
statically legal (shape%tile divisibility) and all 42 compiled and
executed correctly on the Raspberry Pi (1-call + 1000-call repeated-call
tests, plus a 200-cycle mixed-candidate same-process stress test with
allocator noise -- zero failures). The scoring policy is explicit,
inspectable arithmetic (latency + spill/reload penalties + a code-size
tiebreaker + a near-ceiling register penalty) -- no machine learning, no
hidden weights. Selection matches the single fastest measured candidate at
5 of 6 shapes exactly; at the sixth (32x64x32) it trades 0.35% latency for
a 22.7% smaller object at equal (zero) spill count. No single tile wins
universally: 2 tiles (4x8x8, 8x8x8) split the 6 shapes 3/3, with both
sharing TN=8/TK=8 and differing only in TM.

**Still not implemented:**
- Online runtime autotuning (offline, build-time selection over a fixed,
  pre-enumerated candidate set)
- General dynamic-shape tile selection (only the 6 evaluated shapes have a
  selection result; a new shape needs its own candidate sweep)
- Full LLVM register-pressure analysis (assembly-derived evidence only,
  labeled as such throughout `candidate_results.json`)
- MIR-level compiler pass, instruction scheduling, software pipelining,
  prefetch insertion (all remain entirely LLVM's `llc`)
- Cost-model integration into any production plan-selection pass (this
  slice is a standalone, artifact-backed selector tool, not wired into
  `CandidateEvaluationPass` or any runtime dispatch)
- INT8 / SDOT / UDOT dot-product lowering
- GPU code generation of any kind
