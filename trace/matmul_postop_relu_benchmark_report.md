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
| bias | naive_unfused | cpu_naive_matmul_bias_relu_unfused_f32 | 1.748104 | 1.740309 | 1.769905 | 0.016121 | 0.009222 | true | 3 |
| bias | tiled_unfused | cpu_tiled_matmul_bias_relu_unfused_f32 | 0.483640 | 0.483904 | 0.484345 | 0.000930 | 0.001924 | true | 2 |
| bias | naive_one_pass_fused | cpu_naive_matmul_bias_relu_one_pass_f32 | 1.755765 | 1.755203 | 1.761371 | 0.004300 | 0.002449 | true | 4 |
| bias | tiled_one_pass_fused | cpu_tiled_matmul_bias_relu_one_pass_f32 | 0.415001 | 0.415592 | 0.416452 | 0.001637 | 0.003945 | true | 1 |
| elementwise_add | naive_unfused | cpu_naive_matmul_add_relu_unfused_f32 | 1.757719 | 1.755030 | 1.768325 | 0.008109 | 0.004614 | true | 3 |
| elementwise_add | tiled_unfused | cpu_tiled_matmul_add_relu_unfused_f32 | 0.484578 | 0.484312 | 0.485843 | 0.001127 | 0.002326 | true | 2 |
| elementwise_add | naive_one_pass_fused | cpu_naive_matmul_add_relu_one_pass_f32 | 1.784999 | 1.772896 | 1.819834 | 0.025396 | 0.014228 | true | 4 |
| elementwise_add | tiled_one_pass_fused | cpu_tiled_matmul_add_relu_one_pass_f32 | 0.413980 | 0.414010 | 0.415157 | 0.001027 | 0.002480 | true | 1 |

## Runtime Trace


## Comparisons

Fair fusion comparison: `tiled_unfused` vs `tiled_one_pass_fused`.

Full-stack comparison: `naive_unfused` vs `tiled_one_pass_fused`.

| Pattern | Comparison | Speedup | Latency reduction |
| ------- | ---------- | ------: | ----------------: |
| bias | tiling_speedup | 3.614473 | 72.333449% |
| bias | fusion_speedup_naive | 0.995637 | -0.438254% |
| bias | fusion_speedup_fair | 1.165395 | 14.192210% |
| bias | full_stack_speedup | 4.212290 | 76.259944% |
| elementwise_add | tiling_speedup | 3.627322 | 72.431448% |
| elementwise_add | fusion_speedup_naive | 0.984717 | -1.552007% |
| elementwise_add | fusion_speedup_fair | 1.170533 | 14.568828% |
| elementwise_add | full_stack_speedup | 4.245899 | 76.447863% |

## One-Pass Evidence

- `run_naive_one_pass_fused`: `intermediate_tensor_count=0`, `full_output_post_op_passes=0`, `final_output_store_passes=1`.
- `run_tiled_one_pass_fused`: `intermediate_tensor_count=0`, `full_output_post_op_passes=0`, `final_output_store_passes=1`.
- These are static implementation properties, not measured hardware memory traffic.
