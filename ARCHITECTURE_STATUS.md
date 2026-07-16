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
| AArch64 native codegen: MLIR vectorization slice (fused MatMul-Bias-ReLU only) | narrow real implementation + measured on real Raspberry Pi 5 hardware, real NEON `fmla` confirmed | `artifacts/backend_codegen/aarch64_matmul_bias_relu_vectorized/` (both variants' generated `.mlir`/`.ll`/`.s`/`.o`, objdump, vector-dialect intermediate, repeated-call and mixed-shape correctness logs, benchmark/metrics JSON, Pi device state); `mlir_passes/transforms/vectorize_matmul_bias_relu.mlir`; `tools/run_backend_codegen_vectorized_pi_integration.sh` | project-owned contribution is invoking upstream MLIR's `vectorize_children_and_apply_patterns` Transform op at this point in the project's own pipeline, not a project-authored instruction selector; all NEON `fmla` selection, register allocation, and scheduling remain LLVM's `llc`, unmodified; one op, three fixed static shapes, fully unrolled (object size/instruction count scale with M*N*K, not tiled); not wired into the runtime's `OpRegistry` dispatch or the compiler's candidate/cost-model selection |

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
  32x32x32's vectorized object is ~64x the instruction count of 8x8x8's);
  this specific approach would not scale to large or dynamic shapes without
  adding explicit tiling first
- Generated-code runtime loading (same gap as the generic path above)
- General operator coverage (same gap as the generic path above)
- GPU code generation of any kind
