# MatMul Post-Op ReLU Benchmark Report

## Configuration

- Mode: `use-plan`
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
| bias | tiled_one_pass_fused | cpu_tiled_matmul_bias_relu_one_pass_f32 | 0.421292 | 0.421094 | 0.428538 | 0.005657 | 0.013428 | true | 0 |

## Runtime Trace

- Pattern `bias` planned `cpu_tiled_matmul_bias_relu_one_pass_f32`, dispatched `cpu_tiled_matmul_bias_relu_one_pass_f32`, dispatch_count=`1`, plan_matched_runtime=`true`.

## Comparisons

Fair fusion comparison: `tiled_unfused` vs `tiled_one_pass_fused`.

Full-stack comparison: `naive_unfused` vs `tiled_one_pass_fused`.

| Pattern | Comparison | Speedup | Latency reduction |
| ------- | ---------- | ------: | ----------------: |
| bias | tiling_speedup | 0.000000 | 0.000000% |
| bias | fusion_speedup_naive | 0.000000 | 0.000000% |
| bias | fusion_speedup_fair | 0.000000 | 0.000000% |
| bias | full_stack_speedup | 0.000000 | 0.000000% |

## One-Pass Evidence

- `run_naive_one_pass_fused`: `intermediate_tensor_count=0`, `full_output_post_op_passes=0`, `final_output_store_passes=1`.
- `run_tiled_one_pass_fused`: `intermediate_tensor_count=0`, `full_output_post_op_passes=0`, `final_output_store_passes=1`.
- These are static implementation properties, not measured hardware memory traffic.
