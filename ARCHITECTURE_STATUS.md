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
| AArch64 native codegen (fused MatMul-Bias-ReLU only) | narrow real implementation + measured on real Raspberry Pi 5 hardware | `artifacts/backend_codegen/aarch64_matmul_bias_relu/` (generated `.mlir`/`.ll`/`.s`/`.o`, objdump, correctness/benchmark/metrics JSON, Pi device state); `mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh`; `tools/run_backend_codegen_pi_integration.sh` | one op (`hir.fused_matmul_bias_relu`), three fixed shapes (8x8x8/16x16x16/32x32x32), no project-owned target-specific instruction selection/scheduling/register allocation, generated code currently 3.5x-4.3x slower than an `-O2` scalar C++ reference on the same device; not wired into the runtime's `OpRegistry` dispatch or the compiler's candidate/cost-model selection |

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
