# Triton Fused Config Kernel Selection

This report evaluates shape-aware selection among one-pass fused Triton tile configurations only.

## Policy Comparison

| Policy | Config diversity | Top-1 | Macro accuracy | Mean regret | P95 regret | Max regret | Within 3% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static_bm16_bn16_bk32_w4_s3 | 1 | 0.1667 | 0.2500 | 0.8878 | 2.0868 | 2.1293 | 0.3333 |
| static_bm32_bn32_bk32_w4_s3 | 1 | 0.1667 | 0.2500 | 0.3941 | 0.6075 | 0.6203 | 0.1667 |
| static_bm64_bn64_bk32_w4_s3 | 1 | 0.3333 | 0.2500 | 0.4434 | 1.1395 | 1.2886 | 0.5000 |
| static_bm16_bn64_bk32_w4_s3 | 1 | 0.3333 | 0.2500 | 0.2484 | 0.6760 | 0.6779 | 0.5000 |
| nearest_shape | 3 | 0.5000 | 0.3750 | 0.1027 | 0.4361 | 0.5653 | 0.6667 |
| analytical | 1 | 0.1667 | 0.2500 | 0.8878 | 2.0868 | 2.1293 | 0.3333 |
| calibrated_analytical | 2 | 0.6667 | 0.5000 | 0.2151 | 0.9670 | 1.2886 | 0.8333 |
| confidence_aware | 2 | 0.6667 | 0.5000 | 0.2150 | 0.9668 | 1.2886 | 0.8333 |

## Per-Workload Confidence-Aware Decisions

| Workload | M | N | K | Oracle | Selected | Confidence | Regret |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| rep_m1_k768_n3072 | 1 | 3072 | 768 | bm16_bn64_bk32_w4_s3 | bm16_bn64_bk32_w4_s3 | high | 0.0000 |
| rep_m128_k768_n3072 | 128 | 3072 | 768 | bm64_bn64_bk32_w4_s3 | bm64_bn64_bk32_w4_s3 | high | 0.0000 |
| balanced_m64_n64_k64 | 64 | 64 | 64 | bm32_bn32_bk32_w4_s3 | bm16_bn64_bk32_w4_s3 | low | 0.0017 |
| unfriendly_m64_n64_k4096 | 64 | 64 | 4096 | bm16_bn16_bk32_w4_s3 | bm64_bn64_bk32_w4_s3 | high | 1.2886 |
| boundary_m1_n4096_k65536 | 1 | 4096 | 65536 | bm16_bn64_bk32_w4_s3 | bm16_bn64_bk32_w4_s3 | high | 0.0000 |
| boundary_m256_n256_k2048 | 256 | 256 | 2048 | bm64_bn64_bk32_w4_s3 | bm64_bn64_bk32_w4_s3 | high | 0.0000 |

## Collapse Audit

- analytical collapsed: `True`
- calibrated collapsed: `False`
- confidence-aware collapsed: `False`
