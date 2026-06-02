# RMSNorm Compiler-Runtime Case Study

Status: `passed`

## Pipeline

```text
llm.rmsnorm
  -> RMSNormKernelSelectionPass
  -> hir.fused_rmsnorm
  -> profile-calibrated runtime dispatch contract
  -> fused_rmsnorm_cuda vs torch_rmsnorm
```

## Runtime Decision

- HIR op: `hir.fused_rmsnorm`
- Selected kernel: `fused_rmsnorm_cuda`
- Fallback kernel: `torch_rmsnorm`
- Backend: `CUDA`
- Selection reason: `profile_calibrated_fastest`
- Profile source: `/Users/allen/Documents/Codex/project/ml-graph-compiler-runtime/trace/profile_calibrated_cost_table.json,/Users/allen/Documents/Codex/project/ml-graph-compiler-runtime/trace/qmatmul_bias_relu_kernel_profile.json,/Users/allen/Documents/Codex/project/ml-graph-compiler-runtime/trace/matmul_bias_relu_kernel_profile.json,/Users/allen/Documents/Codex/project/heterogeneous-inference-runtime/results/cuda_transformer/rmsnorm_benchmark.json`

## Performance Evidence

- Shape bucket: `16x4096:f32`
- Custom latency: `0.031597 ms`
- PyTorch latency: `0.090415 ms`
- Speedup: `2.8615x`
- Correct: `True`
- Custom effective bandwidth: `33.186 GB/s`
- PyTorch effective bandwidth: `11.597 GB/s`
- Bytes/token: `65536`
- FLOPs/token: `16384`
- Arithmetic intensity: `0.25 FLOPs/byte`

## Roofline Interpretation

- RMSNorm is memory-bound: arithmetic intensity is low and each token streams input, weight, and output data.
- The custom CUDA path reduces framework overhead and uses a shape-specialized reduction/writeback kernel.
- The compiler does not assume the custom kernel wins; it selects `fused_rmsnorm_cuda` because runtime profile evidence says it is faster and correct for the shape bucket.
