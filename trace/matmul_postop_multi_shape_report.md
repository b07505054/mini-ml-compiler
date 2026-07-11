# MatMul Post-Op Multi-Shape Evaluation

Static cost values are analytical features, not measured hardware traffic.

## Summary
- Completed workload-pattern measurements: `26`
- Skipped workloads: `0`
- Exact-profiled top-1 accuracy: `1.000000`
- Exact-profiled mean/median/p95 regret: `0.000000` / `0.000000` / `0.000000`
- Held-out fallback mean regret: `0.228066`

## Layer 1: Compiler Correctness

| Workload | Pattern | Profile match | Selected kernel | Oracle kernel | Fallback | Correct | Planned dispatch |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| rep_m1_k768_n3072 | bias | true | cpu_tiled_matmul_bias_relu_unfused_f32 | cpu_tiled_matmul_bias_relu_unfused_f32 |  | true | True |
| rep_m1_k768_n3072 | elementwise_add | true | cpu_tiled_matmul_add_relu_unfused_f32 | cpu_tiled_matmul_add_relu_unfused_f32 |  | true | True |
| stress_m512_n512_k16 | bias | true | cpu_tiled_matmul_bias_relu_one_pass_f32 | cpu_tiled_matmul_bias_relu_one_pass_f32 |  | true | True |
| stress_m512_n512_k16 | elementwise_add | true | cpu_tiled_matmul_add_relu_one_pass_f32 | cpu_tiled_matmul_add_relu_one_pass_f32 |  | true | True |
| stress_m512_n512_k32 | bias | true | cpu_tiled_matmul_bias_relu_one_pass_f32 | cpu_tiled_matmul_bias_relu_one_pass_f32 |  | true | True |
| stress_m512_n512_k32 | elementwise_add | true | cpu_tiled_matmul_add_relu_one_pass_f32 | cpu_tiled_matmul_add_relu_one_pass_f32 |  | true | True |
| stress_m1024_n1024_k16 | bias | true | cpu_tiled_matmul_bias_relu_one_pass_f32 | cpu_tiled_matmul_bias_relu_one_pass_f32 |  | true | True |
| stress_m1024_n1024_k16 | elementwise_add | true | cpu_tiled_matmul_add_relu_one_pass_f32 | cpu_tiled_matmul_add_relu_one_pass_f32 |  | true | True |
| balanced_m64_n64_k64 | bias | true | cpu_tiled_matmul_bias_relu_one_pass_f32 | cpu_tiled_matmul_bias_relu_one_pass_f32 |  | true | True |
| balanced_m64_n64_k64 | elementwise_add | true | cpu_tiled_matmul_add_relu_one_pass_f32 | cpu_tiled_matmul_add_relu_one_pass_f32 |  | true | True |
| balanced_m128_n128_k128 | bias | true | cpu_tiled_matmul_bias_relu_one_pass_f32 | cpu_tiled_matmul_bias_relu_one_pass_f32 |  | true | True |
| balanced_m128_n128_k128 | elementwise_add | true | cpu_tiled_matmul_add_relu_one_pass_f32 | cpu_tiled_matmul_add_relu_one_pass_f32 |  | true | True |
| balanced_m256_n256_k256 | bias | true | cpu_tiled_matmul_bias_relu_one_pass_f32 | cpu_tiled_matmul_bias_relu_one_pass_f32 |  | true | True |
| balanced_m256_n256_k256 | elementwise_add | true | cpu_tiled_matmul_add_relu_one_pass_f32 | cpu_tiled_matmul_add_relu_one_pass_f32 |  | true | True |
| unfriendly_m128_n128_k512 | bias | true | cpu_tiled_matmul_bias_relu_one_pass_f32 | cpu_tiled_matmul_bias_relu_one_pass_f32 |  | true | True |
| unfriendly_m128_n128_k512 | elementwise_add | true | cpu_tiled_matmul_add_relu_one_pass_f32 | cpu_tiled_matmul_add_relu_one_pass_f32 |  | true | True |
| unfriendly_m128_n128_k1024 | bias | true | cpu_tiled_matmul_bias_relu_one_pass_f32 | cpu_tiled_matmul_bias_relu_one_pass_f32 |  | true | True |
| unfriendly_m128_n128_k1024 | elementwise_add | true | cpu_tiled_matmul_add_relu_one_pass_f32 | cpu_tiled_matmul_add_relu_one_pass_f32 |  | true | True |
| unfriendly_m64_n64_k4096 | bias | true | cpu_tiled_matmul_bias_relu_one_pass_f32 | cpu_tiled_matmul_bias_relu_one_pass_f32 |  | true | True |
| unfriendly_m64_n64_k4096 | elementwise_add | true | cpu_tiled_matmul_add_relu_one_pass_f32 | cpu_tiled_matmul_add_relu_one_pass_f32 |  | true | True |
| ksweep_m1024_n1024_k8 | bias | true | cpu_tiled_matmul_bias_relu_one_pass_f32 | cpu_tiled_matmul_bias_relu_one_pass_f32 |  | true | True |
| ksweep_m1024_n1024_k8 | elementwise_add | true | cpu_tiled_matmul_add_relu_one_pass_f32 | cpu_tiled_matmul_add_relu_one_pass_f32 |  | true | True |
| holdout_m192_n192_k192 | bias | false | cpu_tiled_matmul_bias_relu_unfused_f32 | cpu_tiled_matmul_bias_relu_one_pass_f32 | no_exact_shape_match | true | True |
| holdout_m192_n192_k192 | elementwise_add | false | cpu_tiled_matmul_add_relu_unfused_f32 | cpu_tiled_matmul_add_relu_one_pass_f32 | no_exact_shape_match | true | True |
| holdout_m768_n768_k32 | bias | false | cpu_tiled_matmul_bias_relu_unfused_f32 | cpu_tiled_matmul_bias_relu_one_pass_f32 | no_exact_shape_match | true | True |
| holdout_m768_n768_k32 | elementwise_add | false | cpu_tiled_matmul_add_relu_unfused_f32 | cpu_tiled_matmul_add_relu_one_pass_f32 | no_exact_shape_match | true | True |

## Layer 2: Static Optimization Impact

The fused path is modeled as reducing runtime dispatch count from 3 to 1, logical intermediate tensors from 2 to 0, and full-output post-op passes from 2 to 0.

## Layer 3: Measured Runtime Impact

| Group | Count | Geo mean fair fusion | Geo mean plan speedup | Top-1 | Mean regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| representative | 2 | 0.977307 | 1.000000 | 1.000000 | 0.000000 |
| fusion_friendly_memory_stress | 8 | 1.249863 | 1.172173 | 0.750000 | 0.073170 |
| balanced | 7 | 1.158930 | 1.109869 | 0.714286 | 0.046701 |
| fusion_unfriendly_compute_heavy | 6 | 1.134314 | 1.134314 | 1.000000 | 0.000000 |
| k_sweep | 2 | 1.503278 | 1.503278 | 1.000000 | 0.000000 |
| exact_profiled | 21 | 1.182831 | 1.185420 | 1.000000 | 0.000000 |
| unprofiled_fallback | 4 | 1.226354 | 1.000000 | 0.000000 | 0.228066 |
| bias | 13 | 1.203221 | 1.167731 | 0.846154 | 0.035730 |
| elementwise_add | 12 | 1.175201 | 1.138470 | 0.833333 | 0.037315 |

## Correlation Analysis

- `k`: Pearson `-0.265333`, Spearman `-0.841103`
- `output_bytes`: Pearson `0.836826`, Spearman `0.708578`
- `fusion_pressure_score`: Pearson `0.848507`, Spearman `0.841103`
- `matmul_flops`: Pearson `0.217283`, Spearman `0.066391`
- `output_bytes_per_matmul_flop`: Pearson `0.848507`, Spearman `0.841103`
