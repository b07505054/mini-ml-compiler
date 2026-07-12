# Triton Shape-Aware Kernel Selection

This report evaluates shape-aware analytical decisions for unseen Triton V1/V3 workloads. It does not add kernels or train an opaque model.

## Policy Comparison

| Policy | V1 selections | V3 selections | Fallbacks | Mean regret | Median regret | P95 regret | Within 1% | Within 3% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always_v3 | 0 | 12 | 0 | 0.000000 | 0.000000 | 0.000000 | 1.0000 | 1.0000 |
| current_v1_fallback | 12 | 0 | 0 | 0.485363 | 0.137154 | 1.564303 | 0.2500 | 0.4167 |
| nearest_profile | 0 | 12 | 0 | 0.000000 | 0.000000 | 0.000000 | 1.0000 | 1.0000 |
| analytical_winner | 0 | 12 | 0 | 0.000000 | 0.000000 | 0.000000 | 1.0000 | 1.0000 |
| confidence_guided | 9 | 3 | 9 | 0.194073 | 0.008591 | 0.952816 | 0.5000 | 0.6667 |

## Per-Workload Confidence-Guided Decisions

| Workload | M/N/K | V1 pred | V3 pred | Winner | Confidence | Final | Fallback | Oracle | Regret |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| holdout_m1024_n1024_k24 | 1024/1024/24 | 0.319442 | 0.135609 | V3 | high | V3 | None | V3 | 0.000000 |
| holdout_m1024_n1024_k48 | 1024/1024/48 | 0.423569 | 0.229782 | V3 | high | V3 | None | V3 | 0.000000 |
| holdout_m1024_n1024_k96 | 1024/1024/96 | 0.629391 | 0.415928 | V3 | low | V1 | low_model_confidence | V3 | 0.457349 |
| holdout_m1024_n1024_k192 | 1024/1024/192 | 1.045901 | 0.792620 | V3 | low | V1 | low_model_confidence | V3 | 0.236972 |
| holdout_m192_n192_k192 | 192/192/192 | 0.056052 | 0.039046 | V3 | low | V1 | low_model_confidence | V3 | 1.558386 |
| holdout_m768_n768_k32 | 768/768/32 | 0.206128 | 0.097356 | V3 | high | V3 | None | V3 | 0.000000 |
| holdout_m48_k768_n3072 | 48/3072/768 | 0.581459 | 0.498742 | V3 | low | V1 | low_model_confidence | V3 | 0.037336 |
| boundary_m1_n4096_k65536 | 1/4096/65536 | 9.536077 | 8.617391 | V3 | low | V1 | low_model_confidence | V3 | 0.003742 |
| boundary_m1_n2048_k65536 | 1/2048/65536 | 4.832824 | 4.364045 | V3 | low | V1 | low_model_confidence | V3 | 0.000785 |
| boundary_m1_n11008_k8192 | 1/11008/8192 | 3.289882 | 2.967349 | V3 | low | V1 | low_model_confidence | V3 | 0.002112 |
| boundary_m64_n64_k8192 | 64/64/8192 | 0.196304 | 0.170478 | V3 | low | V1 | low_model_confidence | V3 | 0.018747 |
| boundary_m256_n256_k2048 | 256/256/2048 | 0.609935 | 0.535965 | V3 | low | V1 | low_model_confidence | V3 | 0.013441 |

## Collapse Audit

- analytical collapsed to always V3: `True`
- confidence-guided collapsed to always V3: `False`

## Truth Boundary

Predictions are calibrated from permitted training profiles and are not runtime latency guarantees. Low-confidence V1 selections are conservative fallbacks, not predicted V1 performance wins.

## Feature Schema

{
  "shape_features": [
    "m",
    "n",
    "k",
    "dtype",
    "log2_m",
    "log2_n",
    "log2_k",
    "output_elements",
    "a_elements",
    "b_elements",
    "flops",
    "output_bytes",
    "bias_bytes",
    "a_bytes",
    "b_bytes",
    "m_over_n",
    "n_over_m",
    "k_over_m",
    "k_over_n",
    "arithmetic_intensity",
    "output_tile_count",
    "k_tile_count",
    "m_edge_utilization",
    "n_edge_utilization",
    "k_tail_utilization",
    "small_m",
    "small_n",
    "extreme_k",
    "skinny_output",
    "log2_output_elements"
  ],
  "candidate_features": [
    "variant",
    "kernel_id",
    "runtime_operation_count",
    "expected_launch_count",
    "full_size_intermediate_count",
    "estimated_full_size_intermediate_bytes",
    "estimated_global_bytes",
    "block_m",
    "block_n",
    "block_k",
    "num_warps",
    "num_stages",
    "precision_mode",
    "output_tile_count",
    "k_tile_count",
    "tile_utilization",
    "low_parallelism"
  ],
  "forbidden_model_fields": [
    "expected_region",
    "final_classification",
    "fresh_oracle",
    "measured_winner",
    "oracle_best_kernel",
    "oracle_latency",
    "relative_regret"
  ]
}
