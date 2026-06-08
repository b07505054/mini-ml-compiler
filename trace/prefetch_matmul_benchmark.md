# CPU Software Prefetch MatMul Benchmark

Status: `measured`

- Input: `HIR fused MatMul-Bias-ReLU CPU backend workload`
- Decision: `profile-guided choice between tiled CPU kernel and prefetch tiled CPU kernel`
- Metric: `p50/p95 latency, speedup, correctness, estimated bytes moved`

| Kernel | p95 ms |
|---|---:|
| fused_matmul_add_relu_optimized | 43.112 |
| fused_matmul_add_relu_prefetch | 47.3591 |

- Speedup: `0.895305x`
- Correct: `true`
- Selected prefetch candidate: `false`
