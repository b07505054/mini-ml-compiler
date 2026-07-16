# Known Gaps

Last verified: 2026-07-15.

| Gap | Status | Publication wording |
|---|---|---|
| Universal policy engine | missing | P1D.1 and E3 are evidence-driven loops; many other decisions remain declared-profile, rule-based, shadow, or experimental. |
| Triton production integration | missing | Triton is shadow/provider-shaped with real artifacts, but not production Runtime dispatch. |
| Quantization scope | advanced partial | Static INT8 is materialized and executed for the validated fused Pi operator. Full-model accuracy, graph-wide mixed precision, INT4/AWQ/GPTQ, and NPU quantization remain absent. |
| AWQ plan consistency | known inconsistency | The positive AWQ plan has global AWQ/int4 intent while per-op entries contain fp16 fallback strategy text and int4 dtype fields. Do not claim complete INT4 support. |
| Capability DB canonicality | partial | Capability profiles are intended ownership, but compiler-local profiles remain richer and synchronization is incomplete. |
| Memory hierarchy / DMA / transfer model | missing | No mature memory-space/DMA/synchronization IR or bandwidth/transfer model. |
| NPU execution | missing | NPU profiles/plans are planning-only. |
| E2 correctness | invalid | Preserve invalid verdict. |
| E2.1 compiler-only interpretation | incorrect | Reclassify as implementation-stack comparison. |
| Fake `5.7 ms` ExecuTorch number | placeholder only | Must not appear as measured evidence. |
| General superiority claims | forbidden | No “beats ExecuTorch” or universal project/runtime superiority claim. |
| AArch64 native codegen: target-specific instruction selection | missing | The AArch64 object generated for `hir.fused_matmul_bias_relu` (`artifacts/backend_codegen/aarch64_matmul_bias_relu/`) uses only generic LLVM lowering of unmodified Linalg-derived loops. No NEON, SDOT/UDOT, or FMLA intrinsic is emitted or selected by this project. Do not describe this as vectorized or NEON-accelerated code. |
| AArch64 native codegen: instruction scheduling / register allocation | missing | Scheduling and register allocation for the generated object are entirely LLVM's stock `llc` behavior, unmeasured and uninspected by this project. No register-pressure, spill, or scheduling report exists yet. |
| AArch64 native codegen: runtime integration | missing | The generated AArch64 object is not loaded by the C++ runtime's `OpRegistry` and is not a candidate in the compiler's cost model / plan selection. It runs only via the standalone harness in `mlir_passes/tools/aarch64_matmul_bias_relu_harness.cpp`, executed directly on the Raspberry Pi. |
| AArch64 native codegen: performance | measured, not yet competitive | On the real Raspberry Pi 5, the generated kernel is 3.5x-4.3x slower (median latency) than an `-O2` scalar C++ reference across all three measured shapes (8x8x8, 16x16x16, 32x32x32). This is expected given no unrolling/vectorization is applied yet (see `artifacts/backend_codegen/aarch64_matmul_bias_relu/README.md`); do not present this path as a performance improvement over existing handwritten kernels. |
| AArch64 native codegen: multi-shape single-process benchmarking | known reliability limit | Running all three shapes' correctness+benchmark loops back-to-back in one process intermittently produced incorrect output on one shape, even after fixing a real buffer-deallocation leak in the pipeline; root cause not fully isolated. Worked around by running each shape as its own process (`tools/run_backend_codegen_pi_integration.sh`). Do not run this harness with all three shapes in a single process for anything other than quick manual smoke checks. |
| Pre-existing, unrelated test failures | pre-existing, not introduced by the AArch64 codegen slice | As of 2026-07-15, roughly two dozen `tools/run_mlir_pass_tests.sh` cases fail at HEAD `ab74ca24`. The dominant cause (majority of cases) is a shared `linalg.map` operand/mapper-arity verifier rejection spanning canonicalization, fusion, and quantization test fixtures alike; a smaller cluster fails on `'hir.quantize' op requires integer attribute 'clamp_min'`; a few more are FileCheck attribute-string/output drift in serving and LLM-frontend-normalization fixtures. Verified via `git stash` isolation to already fail identically on the unmodified script before any change in that session. Out of scope for the AArch64 codegen slice; needs its own fix. |
