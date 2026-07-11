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
| elementwise_add | naive_unfused | cpu_naive_matmul_add_relu_unfused_f32 | 1.751976 | 1.752818 | 1.767335 | 0.012032 | 0.006867 | true | 3 |
| elementwise_add | tiled_unfused | cpu_tiled_matmul_add_relu_unfused_f32 | 0.485528 | 0.485899 | 0.487020 | 0.001416 | 0.002917 | true | 2 |
| elementwise_add | naive_one_pass_fused | cpu_naive_matmul_add_relu_one_pass_f32 | 1.766116 | 1.758911 | 1.807463 | 0.030756 | 0.017415 | true | 4 |
| elementwise_add | tiled_one_pass_fused | cpu_tiled_matmul_add_relu_one_pass_f32 | 0.417932 | 0.416447 | 0.422797 | 0.003583 | 0.008573 | true | 1 |

## Runtime Trace


## Comparisons

Fair fusion comparison: `tiled_unfused` vs `tiled_one_pass_fused`.

Full-stack comparison: `naive_unfused` vs `tiled_one_pass_fused`.

| Pattern | Comparison | Speedup | Latency reduction |
| ------- | ---------- | ------: | ----------------: |
| elementwise_add | tiling_speedup | 3.608394 | 72.286839% |
| elementwise_add | fusion_speedup_naive | 0.991994 | -0.807095% |
| elementwise_add | fusion_speedup_fair | 1.161741 | 13.922265% |
| elementwise_add | full_stack_speedup | 4.192018 | 76.145139% |

## One-Pass Evidence

- `run_naive_one_pass_fused`: `intermediate_tensor_count=0`, `full_output_post_op_passes=0`, `final_output_store_passes=1`.
- `run_tiled_one_pass_fused`: `intermediate_tensor_count=0`, `full_output_post_op_passes=0`, `final_output_store_passes=1`.
- These are static implementation properties, not measured hardware memory traffic.
