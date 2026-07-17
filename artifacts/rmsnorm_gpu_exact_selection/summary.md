# Exact RMSNorm GPU configuration selection

This artifact closes an operator-level, target-specific `weighted_rmsnorm` selection slice for FP32 on the NVIDIA GeForce GTX 1650 Max-Q (compute capability 7.5). It is not full-model or serving integration. The current single-input `hir.fused_rmsnorm` op remains semantically unweighted; the weighted identity is explicit in the measured profile, exact candidate, ExecutionPlan, and Runtime validation layers.

The retained cost table contains 84 measured rows: 48 CUDA configurations (four block sizes), 24 Triton configurations (two warp counts), and 12 PyTorch fallback rows. Each optimized row passed `rtol=1e-4`, `atol=1e-4` correctness before selection.

| Tokens | Hidden | Exact p50 winner | p50 ms | p95 ms | GB/s |
|---:|---:|---|---:|---:|---:|
| 1 | 768 | `cuda_rmsnorm_fp32_bs512_v1` | 0.036032 | 0.042368 | 0.335 |
| 1 | 1024 | `cuda_rmsnorm_fp32_bs256_v1` | 0.035328 | 0.053792 | 0.432 |
| 1 | 4096 | `cuda_rmsnorm_fp32_bs64_v1` | 0.034816 | 0.040416 | 1.846 |
| 1 | 8192 | `cuda_rmsnorm_fp32_bs128_v1` | 0.034816 | 0.054624 | 3.505 |
| 16 | 768 | `cuda_rmsnorm_fp32_bs128_v1` | 0.035616 | 0.041280 | 5.413 |
| 16 | 1024 | `cuda_rmsnorm_fp32_bs128_v1` | 0.035232 | 0.043040 | 7.201 |
| 16 | 4096 | `cuda_rmsnorm_fp32_bs64_v1` | 0.034816 | 0.039040 | 29.612 |
| 16 | 8192 | `cuda_rmsnorm_fp32_bs256_v1` | 0.034752 | 0.039712 | 58.955 |
| 128 | 768 | `cuda_rmsnorm_fp32_bs512_v1` | 0.034784 | 0.043072 | 44.513 |
| 128 | 1024 | `cuda_rmsnorm_fp32_bs256_v1` | 0.034016 | 0.035840 | 61.155 |
| 128 | 4096 | `cuda_rmsnorm_fp32_bs512_v1` | 0.069632 | 0.078048 | 118.820 |
| 128 | 8192 | `triton_rmsnorm_fp32_block8192_warps4_stages_default_v1` | 0.116768 | 0.122592 | 142.606 |

Lookup evaluation on the same complete measured table is 12/12 oracle matches with mean, p95, and maximum regret 0. This is calibration-set agreement, not held-out predictive-model evidence. There were zero invalid measured candidates, zero selected/executed mismatches, and zero runtime reselections in the two retained execution proofs.
