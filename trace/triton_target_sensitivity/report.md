# Triton Target-Sensitive Schedule Decisions

This report is analytical-only. Synthetic profiles are not benchmark devices and do not provide oracle latency claims.

## Profiles

| Profile | Kind | Effective CUs | Source |
| --- | --- | ---: | --- |
| synthetic_gpu_8cu | synthetic_analytical | 8 | target_profile |
| synthetic_gpu_16cu | synthetic_analytical | 16 | target_profile |
| synthetic_gpu_40cu | synthetic_analytical | 40 | target_profile |
| synthetic_gpu_80cu | synthetic_analytical | 80 | target_profile |

## Classification Counts

- `feature-sensitive`: 288
- `ranking-sensitive`: 55
- `selection-sensitive`: 30

## Direct Answers

1. Changing only the target profile changes computed scheduling features: `yes`.
2. Changing only the target profile changes cost terms: `yes`.
3. Changing only the target profile changes candidate ranking: `True`.
4. Changing only the target profile changes final selected config: `True`.
5. Closest decision boundaries are listed below.
6. Boundary terms are derived from the largest changed cost component between the crossing profiles.
7. Result is analytical-only, not benchmark-backed for synthetic profiles.
8. Still NVIDIA/Triton-specific: block sizes, warps, stages, Triton program/SM compatibility aliases.
9. Portability blockers: no CPU/NPU schedule adapter, no occupancy/register/shared-memory residency model.
10. Next hardware field: effective parallel slots from `effectiveComputeUnits * maxConcurrentWorkItemsPerUnit`, after defining semantics.

## Example Ranking-Sensitive Case

- workload: `small_square_m32_n32_k1024`
- shape: `{'m': 32, 'n': 32, 'k': 1024, 'dtype': 'f32'}`
- selected configs remain: `{'synthetic_gpu_8cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn16_bk32_w4_s3'}`
- rankings change: `{'synthetic_gpu_8cu': ['bm16_bn16_bk32_w4_s3', 'bm16_bn64_bk32_w4_s3', 'bm32_bn32_bk32_w4_s3', 'bm64_bn64_bk32_w4_s3'], 'synthetic_gpu_16cu': ['bm16_bn16_bk32_w4_s3', 'bm32_bn32_bk32_w4_s3', 'bm16_bn64_bk32_w4_s3', 'bm64_bn64_bk32_w4_s3'], 'synthetic_gpu_40cu': ['bm16_bn16_bk32_w4_s3', 'bm32_bn32_bk32_w4_s3', 'bm16_bn64_bk32_w4_s3', 'bm64_bn64_bk32_w4_s3'], 'synthetic_gpu_80cu': ['bm16_bn16_bk32_w4_s3', 'bm32_bn32_bk32_w4_s3', 'bm16_bn64_bk32_w4_s3', 'bm64_bn64_bk32_w4_s3']}`

## Example Selection-Sensitive Case

- workload: `small_square_m64_n64_k512`
- shape: `{'m': 64, 'n': 64, 'k': 512, 'dtype': 'f32'}`
- selected configs change: `{'synthetic_gpu_8cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm32_bn32_bk32_w4_s3'}`
- boundary term: `low_parallelism`

## Boundary Cases

| Workload | M | N | K | Classification | Selected configs | Boundary term |
| --- | ---: | ---: | ---: | --- | --- | --- |
| small_square_m32_n32_k1024 | 32 | 32 | 1024 | ranking-sensitive | {'synthetic_gpu_8cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn16_bk32_w4_s3'} | ranking_order_without_top1_change |
| small_square_m48_n48_k4096 | 48 | 48 | 4096 | ranking-sensitive | {'synthetic_gpu_8cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn16_bk32_w4_s3'} | ranking_order_without_top1_change |
| small_square_m48_n48_k12288 | 48 | 48 | 12288 | ranking-sensitive | {'synthetic_gpu_8cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn16_bk32_w4_s3'} | ranking_order_without_top1_change |
| small_square_m64_n64_k512 | 64 | 64 | 512 | selection-sensitive | {'synthetic_gpu_8cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm32_bn32_bk32_w4_s3'} | low_parallelism |
| small_square_m64_n64_k1024 | 64 | 64 | 1024 | selection-sensitive | {'synthetic_gpu_8cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn64_bk32_w4_s3'} | calibration |
| small_square_m96_n96_k512 | 96 | 96 | 512 | ranking-sensitive | {'synthetic_gpu_8cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm32_bn32_bk32_w4_s3'} | ranking_order_without_top1_change |
| small_square_m128_n128_k512 | 128 | 128 | 512 | ranking-sensitive | {'synthetic_gpu_8cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm32_bn32_bk32_w4_s3'} | ranking_order_without_top1_change |
| skinny_m1_n256_k4096 | 1 | 256 | 4096 | ranking-sensitive | {'synthetic_gpu_8cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn16_bk32_w4_s3'} | ranking_order_without_top1_change |
| skinny_m1_n512_k512 | 1 | 512 | 512 | selection-sensitive | {'synthetic_gpu_8cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn16_bk32_w4_s3'} | k_dominant_parallelism |
| skinny_m1_n512_k1024 | 1 | 512 | 1024 | selection-sensitive | {'synthetic_gpu_8cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn16_bk32_w4_s3'} | k_dominant_parallelism |
| skinny_m1_n512_k2048 | 1 | 512 | 2048 | selection-sensitive | {'synthetic_gpu_8cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn16_bk32_w4_s3'} | k_dominant_parallelism |
| skinny_m1_n512_k4096 | 1 | 512 | 4096 | selection-sensitive | {'synthetic_gpu_8cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn16_bk32_w4_s3'} | k_dominant_parallelism |
| skinny_m1_n1024_k512 | 1 | 1024 | 512 | selection-sensitive | {'synthetic_gpu_8cu': 'bm16_bn64_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn64_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn64_bk32_w4_s3'} | k_dominant_parallelism |
| skinny_m1_n1024_k1024 | 1 | 1024 | 1024 | selection-sensitive | {'synthetic_gpu_8cu': 'bm16_bn64_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn64_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn16_bk32_w4_s3'} | k_dominant_parallelism |
| skinny_m1_n1024_k2048 | 1 | 1024 | 2048 | selection-sensitive | {'synthetic_gpu_8cu': 'bm16_bn64_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn16_bk32_w4_s3'} | k_dominant_parallelism |
| skinny_m1_n1024_k4096 | 1 | 1024 | 4096 | selection-sensitive | {'synthetic_gpu_8cu': 'bm16_bn64_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn16_bk32_w4_s3'} | k_dominant_parallelism |
| skinny_m1_n2048_k512 | 1 | 2048 | 512 | ranking-sensitive | {'synthetic_gpu_8cu': 'bm16_bn64_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn64_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn64_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn64_bk32_w4_s3'} | ranking_order_without_top1_change |
| skinny_m1_n2048_k1024 | 1 | 2048 | 1024 | ranking-sensitive | {'synthetic_gpu_8cu': 'bm16_bn64_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn64_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn64_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn64_bk32_w4_s3'} | ranking_order_without_top1_change |
| skinny_m2_n256_k4096 | 2 | 256 | 4096 | ranking-sensitive | {'synthetic_gpu_8cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn16_bk32_w4_s3'} | ranking_order_without_top1_change |
| skinny_m2_n512_k512 | 2 | 512 | 512 | selection-sensitive | {'synthetic_gpu_8cu': 'bm32_bn32_bk32_w4_s3', 'synthetic_gpu_16cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_40cu': 'bm16_bn16_bk32_w4_s3', 'synthetic_gpu_80cu': 'bm16_bn16_bk32_w4_s3'} | k_dominant_parallelism |

## Truth Boundary

- Measured: no new benchmark measurements.
- Modeled: target-profile-driven analytical ranking with frozen repaired calibration.
- Synthetic: 8/16/40/80 CU profiles are analytical probes only.
- Unsupported: no CUDA occupancy, register pressure, CPU/NPU schedule model, or C++ pass ownership.
