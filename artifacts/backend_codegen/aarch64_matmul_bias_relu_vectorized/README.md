# AArch64 Vectorized Codegen: hir.fused_matmul_bias_relu

Target: Raspberry Pi 5 / Cortex-A76
Execution type: Real hardware
Code generation: MLIR -> (project-owned MLIR vectorization) -> Vector dialect
  -> LLVM IR -> LLVM AArch64 backend (LLVM-owned machine instruction selection)
Optimization status: First target-aware vectorization slice. Not a general
  vectorizer, not a custom instruction selector.

## Truth boundary: what is project-owned vs. LLVM-owned here

This slice adds exactly one new project-owned artifact to the pipeline:
`mlir_passes/transforms/vectorize_matmul_bias_relu.mlir`, a Transform dialect
script invoked between `hir-matmul-bias-relu-to-linalg` and bufferization.
It calls stock upstream MLIR Transform ops
(`transform.structured.vectorize_children_and_apply_patterns`) to turn the
`linalg.matmul`/`linalg.generic` produced by the existing HIR lowering into
`vector.contract` / `vector.transfer_read` / `vector.transfer_write` --
this is "project-owned" only in the sense that the project chose to invoke
this transform, at this point in its own pipeline, for its own op; the
vectorization patterns themselves are upstream MLIR, not project-authored
matching/rewrite logic.

Everything from there to the `fmla` instructions in the .s files --
selecting `vector.contract` -> NEON `fmla` (rather than separate `fmul`/
`fadd`, or SDOT/UDOT, or scalar code), register allocation of the 4 vector
registers, instruction scheduling, ABI lowering -- is LLVM's AArch64
backend, unmodified. Nothing in this repository implements a custom
SelectionDAG/GlobalISel pattern, a custom scheduler, or a custom register
allocator. `convert-vector-to-llvm{vector-contract-lowering=outerproduct}`
selects the *lowering strategy MLIR emits into LLVM IR* (an outer-product
sequence of `llvm.intr.fmuladd`-shaped vector ops); it is LLVM's own
instruction selector that then maps those to the real `fmla v*.4s, v*.4s,
v*.s[n]` opcode. See `vectorized/vector_dialect_<shape>.mlir` for the exact
boundary: everything in that file is the project-owned MLIR vectorization
stage output; everything downstream of it is stock MLIR->LLVM conversion
plus LLVM's own backend.

## What this is

```
hir.fused_matmul_bias_relu (typed HIR op)
  -> hir-matmul-bias-relu-to-linalg          (project-owned pass, existing,
                                               now with the accumulator fix
                                               below)
  -> transform-preload-library +             (project-owned invocation of
     transform-interpreter                    upstream MLIR Transform ops --
                                               NEW in this slice)
  -> vector.contract / vector.transfer_*     (Vector dialect -- see
                                               vectorized/vector_dialect_<shape>.mlir)
  -> one-shot-bufferize + buffer-deallocation-pipeline + stock MLIR->LLVM
     dialect conversion, including convert-vector-to-llvm{vector-contract-
     lowering=outerproduct}                   (stock upstream MLIR passes)
  -> mlir-translate --mlir-to-llvmir         -> LLVM IR text (vector types)
  -> llc -mtriple=aarch64-linux-gnu -mcpu=cortex-a76
       -> AArch64 assembly and object code, containing real `fmla` NEON
          instructions                        (LLVM-owned instruction
                                               selection, unmodified)
  -> g++ link on the Raspberry Pi itself (object built on the x86_64 dev
     host, transferred via scp, linked with gcc/g++ already installed on
     the Pi)
  -> executed on the real Raspberry Pi, correctness- and latency-checked
     against a scalar C++ reference, the Stage 1 generic (unvectorized)
     AArch64 kernel, and the pre-existing handwritten
     `fused_matmul_add_relu` kernel (src/kernels/cpu_kernels.cpp)
```

Both `generic/` (Stage 1's unvectorized path, rebuilt here with the
accumulator fix -- see below) and `vectorized/` (this slice's new path) are
included side by side for every shape, so the vector-specific contribution
can be isolated from the shared-pass fix.

## A real bug found and fixed during this slice

The task that began this slice was to find and fix a suspected out-of-bounds
write in the vectorized kernel. The actual root cause turned out to be
different, and pipeline-wide rather than vectorization-specific: the shared
lowering pass (`mlir_passes/lib/MatMulBiasReluFusionPass.cpp`,
`HIRMatMulBiasReluToLinalgPattern`) fed a bare `tensor.empty()` directly
into `linalg.matmul`'s accumulator ("outs") operand, with no
zero-initialization. `tensor.empty()`'s contents are documented as
undefined by MLIR; in practice, on a fresh never-reused heap allocation the
backing buffer is often (accidentally) zero, so a single isolated call looks
correct. On repeated invocation, once the allocator reuses a heap address
previously written by the SAME kernel's own prior output buffer, the stale
non-zero contents get accumulated into, producing wrong, growing results
(`(call_count+1) x correct` for the affected elements). This is a read of
stale data through an unintialized value, not an out-of-bounds write; the
severe glibc heap-corruption symptom reported in the prior session's
diagnostic notes was a downstream symptom of this same bug under a specific
non-freeing test variant, not a separate defect.

This bug was present in BOTH the vectorized path being developed in this
slice AND the already-committed Stage 1 generic kernel -- it lives in the
pass both variants share, upstream of where they diverge. It was not caught
in Stage 1 because that slice's benchmark loop timed repeated calls but
never re-verified their correctness after the first. It is fixed here for
both variants simultaneously, since the fix is in the shared pass:

```cpp
auto zero = arith::ConstantOp::create(
    rewriter, loc, rewriter.getFloatAttr(elementType, 0.0));
auto matmulEmpty = tensor::EmptyOp::create(
    rewriter, loc, matmulOutputType.getShape(), elementType);
auto matmulInit = linalg::FillOp::create(
    rewriter, loc, ValueRange{zero.getResult()},
    ValueRange{matmulEmpty.getResult()});
auto matmul = linalg::MatmulOp::create(
    rewriter, loc, matmulOutputType,
    ValueRange{lhs, rhs},
    ValueRange{matmulInit.getResult(0)});
```

This mirrors the pre-existing, correct pattern already used for the
RMSNorm row accumulator earlier in the same file. It also corrects the
Stage 1 artifact's "second, smaller effect... worked around rather than
fully root-caused" note (see
`artifacts/backend_codegen/aarch64_matmul_bias_relu/README.md`): that
cross-shape corruption was this same bug, now fully root-caused and fixed
at the source rather than only worked around via process isolation.

Consequence for `generic/`: the fixed generic objects here are 150 AArch64
instructions / 1768 bytes at every shape, vs. 134 instructions / 1704 bytes
in the (now-superseded, pre-fix) Stage 1 artifact -- the +16
instructions/+64 bytes is the honest cost of the `linalg.fill` this fix
adds.

## Repeated-call and mixed-shape correctness (the gate for this slice)

Verified on the real Raspberry Pi, both variants, all three shapes:

- `repeated_call_results.txt`: 1000 consecutive calls per (shape, variant)
  pair, 6 pairs, all PASS with clean sentinel guard regions (no OOB read/
  write into caller-owned buffer padding).
- `mixed_shape_results.txt`: 500 cycles (6000 total calls) of interleaved
  8x8x8 / 16x16x16 / 32x32x32, generic and vectorized both called per shape
  per cycle, with unrelated heap-allocation noise between every call. PASS.

Zero tolerance: any single incorrect call anywhere in either test is a
FAIL; both report unconditional PASS above the correctness threshold
(1e-3 max abs error vs. a scalar double-precision-accumulated reference).

## Files

- `generic/`, `vectorized/` -- each contains, per shape:
  - `input_<shape>.mlir` -- HIR input (canonical copies at
    `mlir_passes/test/backend_codegen/matmul_bias_relu[_vectorized]_<shape>.mlir`)
  - `lowered_llvm_dialect_<shape>.mlir` -- mlir-opt output (LLVM dialect)
  - `generated_<shape>.ll` -- mlir-translate output (textual LLVM IR)
  - `generated_<shape>.s` -- llc AArch64 assembly
  - `generated_<shape>.o` -- llc AArch64 ELF object. **Not committed** (see
    `object_hashes.txt` for SHA-256 + size; `.gitignore` excludes `*.o`
    repo-wide, no binary object has ever been committed in this repository's
    history)
  - `objdump_<shape>.txt` -- full `llvm-objdump -d` disassembly (this text
    file *is* committed)
  - `vectorized/` only: `vector_dialect_<shape>.mlir` -- the pre-bufferization
    Vector-dialect intermediate (`vector.contract`/`vector.transfer_read`/
    `vector.transfer_write`), i.e. the project-owned MLIR vectorization stage
    output in isolation, before any LLVM-owned lowering
- `benchmark_<shape>.json` -- per-shape correctness verdict (scalar/generic/
  vectorized/handwritten, checksums, max abs error) and median/p95 latency
  for all four, measured on the real Raspberry Pi (`iterations: 2000`,
  `warmup: 200`)
- `repeated_call_results.txt`, `mixed_shape_results.txt` -- see above
- `backend_metrics.json` -- static LLVM IR / AArch64 instruction counts,
  vector-instruction and `fmla` counts, load/store/branch counts, and
  object size, per shape per variant, with counting methodology documented
  inline
- `pi_device_state.txt` -- `hostname; uname -a; lscpu; gcc --version`
  captured on the real Raspberry Pi at benchmark time
- `object_hashes.txt`, `commands.txt` -- reproducibility records (see above)

## Results summary (see benchmark_<shape>.json / backend_metrics.json for full precision)

| Shape | All correct | Generic median | Vectorized median | Speedup vs. generic | Vectorized vs. handwritten |
|---|---|---|---|---|---|
| 8x8x8 | yes | 0.002277 ms | 0.000093-0.000111 ms | ~20-24x | ~1.0x (parity) |
| 16x16x16 | yes | 0.016686 ms | 0.000556 ms | ~30.0x | vectorized ~6.8x faster |
| 32x32x32 | yes | 0.127926 ms | 0.004592-0.004611 ms | ~27.7-27.9x | vectorized ~5.8-6.4x faster |

All four implementations (scalar, generic, vectorized, handwritten) are
numerically correct at every shape (`all_correct: true`, max abs error at
FP32 rounding level, no logic error). The vectorized kernel is real NEON
`fmla`-based code (see Vector and FMLA evidence below), not a simulated or
estimated result -- every number above comes directly from
`benchmark_<shape>.json`, produced by executing on the physical Raspberry Pi
listed in `pi_device_state.txt`.

## Vector and FMLA evidence

`fmla` counts by shape (from `backend_metrics.json`, cross-checked against
`vectorized/objdump_<shape>.txt`): 8x8x8 = 128, 16x16x16 = 1024,
32x32x32 = 8192. These scale with M*N*K/4 as expected for a 4-wide
`float32x4_t` outer-product FMA idiom. `generic/` has zero `fmla` and zero
vector instructions at every shape by construction (no transform is applied
to it).

## Scalability caveat (not a hidden limitation -- stated plainly)

`transform.structured.vectorize_children_and_apply_patterns` fully unrolls
these fixed static shapes rather than emitting a strided loop over vector
tiles. Object size and instruction count scale with M*N*K, not with M*N*K/4:
32x32x32's vectorized object is 113216 bytes / 28026 instructions, a ~64x
increase over the ~1768 byte / 150-instruction (unvectorized) generic
object for the same shape, and the vectorized `.ll`/`.s` text files for
32x32x32 are correspondingly large (see `commands.txt` methodology; sizes
recorded in `backend_metrics.json`). This is expected behavior for this
transform applied to fully static shapes, not a bug -- but it means this
exact approach, unmodified, would not scale to large or dynamic shapes
without adding explicit tiling before vectorization, which is out of scope
for this slice (see "Not implemented" below).

## Not implemented in this slice (explicit non-goals)

- INT8 / SDOT / UDOT dot-product lowering
- Any custom LLVM SelectionDAG or GlobalISel instruction-selection pattern
- Custom instruction scheduling or software pipelining
- Custom register allocation or MIR-level passes
- General vectorization across arbitrary HIR/Linalg operators (this slice
  is scoped to `hir.fused_matmul_bias_relu` only)
- Tiling for shapes beyond the three validated here, or for dynamic shapes
- Runtime dispatch / dlopen loading of generated code into
  heterogeneous-inference-runtime or any ExecutionPlan
- Fixes to the pre-existing 20 historical failing tests in
  `tools/run_mlir_pass_tests.sh` (unrelated to this slice; baseline
  unchanged, see repository test results in the commit message / final
  report for this slice)
