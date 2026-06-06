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
- Selection reason: `gpu_pgo_like_lowest_p95_latency`
- Feedback loop: `gpu_pgo_like_kernel_selection`
- Profile source: `trace/profile_calibrated_cost_table.json,trace/qmatmul_bias_relu_kernel_profile.json,trace/matmul_bias_relu_kernel_profile.json,/Users/allen/Documents/Codex/project/heterogeneous-inference-runtime/results/cuda_transformer/rmsnorm_benchmark.json`

## Performance Evidence

- Shape bucket: `16x768:f16`
- Custom latency: `0.030196 ms`
- PyTorch latency: `0.088261 ms`
- Speedup: `2.923x`
- Correct: `True`
- Custom effective bandwidth: `34.726 GB/s`
- PyTorch effective bandwidth: `11.88 GB/s`
- Bytes/token: `65536`
- FLOPs/token: `16384`
- Arithmetic intensity: `0.25 FLOPs/byte`

## Roofline Interpretation

- RMSNorm is memory-bound: arithmetic intensity is low and each token streams input, weight, and output data.
- The custom CUDA path reduces framework overhead and uses a shape-specialized reduction/writeback kernel.
- The compiler does not assume the custom kernel wins; it selects `fused_rmsnorm_cuda` because GPU PGO-like runtime evidence says it is faster and correct for the shape bucket.

## GPU PGO-like Gate

- Input: `compiler-emitted HIR RMSNorm op plus runtime shape/workload distribution`
- Decision: `profile-guided kernel selection among CUDA/Triton/PyTorch candidates by shape bucket`
- Metric: `kernel p95 latency, effective bandwidth, TPOT projection, throughput projection`

## Serving Impact Projection

- Baseline TPOT p95: `3.244` ms/token
- Projected TPOT p95: `3.182144` ms/token
- TPOT delta: `0.061856` ms/token
- Projected tokens/sec gain: `24.029`
