# MatMul Post-Op ReLU Benchmark Report

## Configuration

- Mode: `sweep-candidates`
- Execution plan: `trace/mlir_execution_plan.json`
- Build type: `Release`
- Warmup iterations per repeat: `50`
- Measured iterations per repeat: `300`
- Repeats: `5`
- Shape: `M=128, K=128, N=128`
- Tile size: `32`
- Host: `allen-ZenBook-UX534FTC-UX534FT`
- Machine: `Linux allen-ZenBook-UX534FTC-UX534FT 7.0.0-27-generic #27-Ubuntu SMP PREEMPT_DYNAMIC Thu Jun 18 19:13:49 UTC 2026 x86_64 GNU/Linux`

## Variant Results

| Pattern | Variant | Kernel | Mean ms | p50 ms | p95 ms | Stddev | CV | Correct | Rank |
| ------- | ------- | ------ | ------: | -----: | -----: | -----: | -: | ------: | ---: |
| bias | naive_unfused | cpu_naive_matmul_bias_relu_unfused_f32 | 1.766207 | 1.763127 | 1.777363 | 0.008632 | 0.004887 | true | 3 |
| bias | tiled_unfused | cpu_tiled_matmul_bias_relu_unfused_f32 | 0.488669 | 0.490020 | 0.492078 | 0.003327 | 0.006808 | true | 2 |
| bias | naive_one_pass_fused | cpu_naive_matmul_bias_relu_one_pass_f32 | 1.784996 | 1.784170 | 1.794226 | 0.008308 | 0.004654 | true | 4 |
| bias | tiled_one_pass_fused | cpu_tiled_matmul_bias_relu_one_pass_f32 | 0.413919 | 0.414472 | 0.414646 | 0.000880 | 0.002126 | true | 1 |

## Runtime Trace


## Comparisons

Fair fusion comparison: `tiled_unfused` vs `tiled_one_pass_fused`.

Full-stack comparison: `naive_unfused` vs `tiled_one_pass_fused`.

| Pattern | Comparison | Speedup | Latency reduction |
| ------- | ---------- | ------: | ----------------: |
| bias | tiling_speedup | 3.614318 | 72.332264% |
| bias | fusion_speedup_naive | 0.989474 | -1.063795% |
| bias | fusion_speedup_fair | 1.180593 | 15.296776% |
| bias | full_stack_speedup | 4.267037 | 76.564536% |

## One-Pass Evidence

- `run_naive_one_pass_fused`: `intermediate_tensor_count=0`, `full_output_post_op_passes=0`, `final_output_store_passes=1`.
- `run_tiled_one_pass_fused`: `intermediate_tensor_count=0`, `full_output_post_op_passes=0`, `final_output_store_passes=1`.
- These are static implementation properties, not measured hardware memory traffic.
