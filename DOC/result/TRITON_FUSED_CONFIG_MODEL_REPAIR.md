# Triton Fused Config Model Repair

## Executive Summary

This repair targets the previous catastrophic small-square/high-K miss:

| Workload | Previous selected | Fresh oracle | Previous regret | Repaired selected | Repaired regret |
| --- | --- | --- | ---: | --- | ---: |
| `unfriendly_m64_n64_k4096` | `bm64_bn64_bk32_w4_s3` | `bm16_bn16_bk32_w4_s3` | 1.2886 | `bm16_bn16_bk32_w4_s3` | 0.0000 |

The previous model ranked `64x64` too high because the analytical formula did not sufficiently penalize too few large output programs for K-dominant small-output GEMMs. The repair adds continuous features for K dominance, programs per SM, work per program, output-area-to-K, and tile area relative to output area.

The repaired calibrated selector chooses all four configs on the held-out set and reduces mean/p95/max regret from `0.2150 / 0.9668 / 1.2886` to `0.0008 / 0.0039 / 0.0055`.

## Targeted Training

Additional formal training evidence was measured for `training_region = small_square_high_k`:

| Workload | M | N | K |
| --- | ---: | ---: | ---: |
| `repair_train_sq32_k2048` | 32 | 32 | 2048 |
| `repair_train_sq32_k4096` | 32 | 32 | 4096 |
| `repair_train_sq48_k4096` | 48 | 48 | 4096 |
| `repair_train_sq96_k2048` | 96 | 96 | 2048 |
| `repair_train_sq96_k4096` | 96 | 96 | 4096 |
| `repair_train_sq128_k1024` | 128 | 128 | 1024 |
| `repair_train_sq128_k8192` | 128 | 128 | 8192 |
| `repair_train_sq192_k2048` | 192 | 192 | 2048 |

The original held-out failure `unfriendly_m64_n64_k4096` was not added to training. A separate `repair_holdout_sq96_k8192` workload remains held out.

## Model Repair

New/corrected features:

- `k_dominance_ratio = K / (M*N)`
- `output_area_to_k = (M*N) / K`
- `programs_per_sm = output_program_count / SM_count`
- `k_iterations_per_output_program`
- `work_per_program = BLOCK_M * BLOCK_N * K_tile_count`
- `parallelism_to_compute_ratio`
- `tile_area_relative_to_output_area`
- `output_program_waves`
- `k_dominant_parallelism_unit`

The repaired formula is:

```text
predicted_latency =
    fixed
  + effective_compute
  + memory
  + padding
  + low_parallelism
  + k_dominant_parallelism
  + excessive_programs
  + shape_config_mismatch
```

The repair keeps the model interpretable and uses per-config affine calibration. No workload-specific winner rule is added.

## Fresh Oracle

Fresh oracle measurements were rerun with:

- warmup: `50`
- iterations: `300`
- repeats: `5`
- sessions: `3`
- candidate order: `alternating`
- candidates: all four fused configs

Fresh oracle winner counts:

| Config | Oracle wins | Stable wins |
| --- | ---: | ---: |
| `bm16_bn16_bk32_w4_s3` | 1 | 1 |
| `bm32_bn32_bk32_w4_s3` | 1 | 1 |
| `bm64_bn64_bk32_w4_s3` | 2 | 2 |
| `bm16_bn64_bk32_w4_s3` | 3 | 1 |

## Policy Comparison

| Policy | Diversity | Top-1 | Macro | Mean Regret | P95 | Max | Within 3% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static_bm16_bn16_bk32_w4_s3 | 1 | 0.1429 | 0.2500 | 0.8201 | 2.0666 | 2.1102 | 0.2857 |
| static_bm32_bn32_bk32_w4_s3 | 1 | 0.1429 | 0.2500 | 0.3331 | 0.6074 | 0.6293 | 0.2857 |
| static_bm64_bn64_bk32_w4_s3 | 1 | 0.2857 | 0.2500 | 0.5089 | 1.1866 | 1.3328 | 0.4286 |
| static_bm16_bn64_bk32_w4_s3 | 1 | 0.4286 | 0.2500 | 0.2268 | 0.6740 | 0.6797 | 0.4286 |
| nearest_shape | 4 | 0.5714 | 0.5833 | 0.0959 | 0.4391 | 0.6081 | 0.7143 |
| previous_calibrated | 3 | 0.7143 | 0.6667 | 0.1930 | 0.9384 | 1.3328 | 0.8571 |
| previous_confidence | 3 | 0.8571 | 0.7500 | 0.1904 | 0.9329 | 1.3328 | 0.8571 |
| repaired_analytical | 1 | 0.1429 | 0.2500 | 0.8201 | 2.0666 | 2.1102 | 0.2857 |
| repaired_calibrated | 4 | 0.8571 | 0.9167 | 0.0008 | 0.0039 | 0.0055 | 1.0000 |
| repaired_confidence | 4 | 0.8571 | 0.9167 | 0.0008 | 0.0039 | 0.0055 | 1.0000 |

## Per-Workload Decisions

| Workload | M | N | K | Oracle | Repaired selected | Regret |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `rep_m1_k768_n3072` | 1 | 3072 | 768 | `bm16_bn64_bk32_w4_s3` | `bm16_bn64_bk32_w4_s3` | 0.0000 |
| `rep_m128_k768_n3072` | 128 | 3072 | 768 | `bm64_bn64_bk32_w4_s3` | `bm64_bn64_bk32_w4_s3` | 0.0000 |
| `balanced_m64_n64_k64` | 64 | 64 | 64 | `bm16_bn64_bk32_w4_s3` | `bm32_bn32_bk32_w4_s3` | 0.0055 |
| `unfriendly_m64_n64_k4096` | 64 | 64 | 4096 | `bm16_bn16_bk32_w4_s3` | `bm16_bn16_bk32_w4_s3` | 0.0000 |
| `boundary_m1_n4096_k65536` | 1 | 4096 | 65536 | `bm16_bn64_bk32_w4_s3` | `bm16_bn64_bk32_w4_s3` | 0.0000 |
| `boundary_m256_n256_k2048` | 256 | 256 | 2048 | `bm64_bn64_bk32_w4_s3` | `bm64_bn64_bk32_w4_s3` | 0.0000 |
| `repair_holdout_sq96_k8192` | 96 | 96 | 8192 | `bm32_bn32_bk32_w4_s3` | `bm32_bn32_bk32_w4_s3` | 0.0000 |

## Runtime Validation

ExecutionPlan use-plan validation:

| Metric | Value |
| --- | ---: |
| workload count | 7 |
| planned kernel equals actual | 1.0000 |
| planned config equals actual | 1.0000 |
| correctness pass rate | 1.0000 |
| runtime override count | 0 |

## Readiness

The repaired calibrated selector is ready for a C++ lift review because the fresh oracle contains multiple stable config winners, the compiler selects all four configs, mean and tail regret are no longer dominated by the small-square/high-K miss, and runtime dispatch remains exact.

The raw uncalibrated repaired analytical model still collapses to static `16x16`; only the calibrated repair should be considered for lift.
