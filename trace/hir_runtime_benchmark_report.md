# HIR Runtime Benchmark Report

Status: `passed`

| HIR op | Runtime kernel | Backend | Custom ms | Baseline ms | Speedup | Correct |
|---|---|---:|---:|---:|---:|---:|
| hir.fused_matmul_bias_relu | unfused_matmul_add_relu | CPU | 5.354697 | 5.145997 | 0.961025 | True |
| hir.fused_rmsnorm | fused_rmsnorm_cuda | CUDA | 0.02975 | 0.086751 | 2.916 | True |

## Validation

### MatMul-Bias-ReLU

- Compiler emitted: `hir.fused_matmul_bias_relu`
- Runtime dispatch: `unfused_matmul_add_relu` on `CPU`
- Selection reason: `profile_calibrated_fallback`
- Profile source: `/Users/allen/Documents/Codex/project/ml-graph-compiler-runtime/trace/matmul_bias_relu_kernel_profile.json,/Users/allen/Documents/Codex/project/heterogeneous-inference-runtime/results/cuda_transformer/rmsnorm_benchmark.json`
- compiler_emitted_typed_hir_op: `True`
- runtime_dispatch_contract_present: `True`
- runtime_decision_profile_calibrated: `True`
- runtime_decision_matches_benchmark: `True`
- numeric_correctness_passed: `True`
- benchmark_available: `True`

### RMSNorm

- Compiler emitted: `hir.fused_rmsnorm`
- Runtime dispatch: `fused_rmsnorm_cuda` on `CUDA`
- Selection reason: `profile_calibrated_fastest`
- Profile source: `/Users/allen/Documents/Codex/project/ml-graph-compiler-runtime/trace/matmul_bias_relu_kernel_profile.json,/Users/allen/Documents/Codex/project/heterogeneous-inference-runtime/results/cuda_transformer/rmsnorm_benchmark.json`
- compiler_emitted_typed_hir_op: `True`
- runtime_dispatch_contract_present: `True`
- runtime_decision_profile_calibrated: `True`
- runtime_decision_matches_benchmark: `True`
- numeric_correctness_passed: `True`
- benchmark_available: `True`

