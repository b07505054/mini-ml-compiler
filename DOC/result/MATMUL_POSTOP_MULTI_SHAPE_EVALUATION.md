# MatMul Post-Op Multi-Shape Evaluation

## Executive Summary

This evaluation expands MatMul post-op fusion from a single 128^3 shape into a
versioned multi-shape suite with candidate sweeps, profile-guided selection,
ExecutionPlan validation, held-out fallback measurement, and layered reporting.

Fusion is not beneficial for every workload. On the stable formal-core results,
representative workloads measured a `0.9773x` geometric-mean fair fusion
speedup, while fusion-friendly stress workloads measured `1.2499x`. The best
case measured fair fusion speedup was `1.6017x`, and the worst case was
`0.9749x`.

Exact-profiled stable formal-core workloads achieved `100%` top-1 kernel
selection accuracy with mean/median/p95 regret of `0/0/0`. Held-out exact-match
fallback remains a real limitation: fallback mean regret was `22.81%`.

The analytical fusion-pressure feature correlates strongly with measured fusion
benefit in this formal-core run: Pearson `0.8485`, Spearman `0.8411`. The next
step should be a shape-aware analytical or hybrid selection policy, not
unconditional fusion.

## Architecture

```text
manifest
-> candidate sweep
-> profile generation
-> compiler selection
-> ExecutionPlan
-> typed runtime dispatch
-> fresh oracle validation
-> aggregate report
```

The benchmark runner preserves four variants for both Bias and elementwise Add:

```text
V0 naive_unfused
V1 tiled_unfused
V2 naive_one_pass_fused
V3 tiled_one_pass_fused
```

The primary fair fusion comparison is `V1 tiled_unfused` vs
`V3 tiled_one_pass_fused`. The full-stack comparison is `V0 naive_unfused` vs
`V3 tiled_one_pass_fused`.

## Workload Coverage

The versioned manifest is `benchmarks/matmul_postop_workloads.json`.

Categories:

- `representative`
- `fusion_friendly_memory_stress`
- `balanced`
- `fusion_unfriendly_compute_heavy`
- `k_sweep`

Manifest tiers:

- `formal_core`: warmup 50, iterations 300, repeats 5
- `extended`: retained for broader measurement, not mixed into the primary aggregate
- `resource_heavy`: retained with reduced-budget metadata or skip rationale

Formal-core run coverage:

- Completed formal-core workload IDs: `13`
- Workload-pattern measurements: `26`
- Bias measurements: `13`
- Elementwise Add measurements: `13` raw, `12` admitted to the stable aggregate
- Unstable records excluded from primary aggregate: `1`
- Formal-core run skipped workloads: `0`
- Manifest-level skipped workloads preserved with structured reason: `1`

The manifest-level skipped workload is:

```text
rep_m128_k4096_n4096
```

Reason: estimated naive latency is about 1790 ms per iteration at the manifest's
calibrated host estimate, making a four-candidate resource-heavy sweep exceed
the intended laptop wall-clock budget.

## Layer 1: Compiler Correctness

The artifacts record per workload-pattern:

- recognized pattern and post-op semantics
- profile match status
- selected kernel
- fallback reason
- planned kernel
- actual dispatched kernel
- numerical correctness

All formal-core use-plan runs had `planned_kernel == actual_dispatched_kernel`.
Held-out workloads used explicit fallback:

```text
fallback_reason = no_exact_shape_match
```

The exact-profiled path includes the previous 128^3 anchor shape and preserves
selection of:

```text
cpu_tiled_matmul_bias_relu_one_pass_f32
```

## Layer 2: Static Optimization Impact

The static-cost model reports logical and analytical properties only:

- logical intermediate tensors: `2 -> 0`
- runtime dispatch count: `3 -> 1`
- full-output post-op passes: `2 -> 0`
- final output store pass in the fused path
- analytical estimated post-op bytes eliminated
- `fusion_pressure_score`

The `fusion_pressure_score` is an analytical feature:

```text
estimated_postop_bytes_eliminated / matmul_flops
```

It is not measured hardware traffic. These results do not claim actual DRAM
traffic reduction or actual peak RSS reduction.

## Layer 3: Measured Runtime Impact

Stable primary aggregates:

| Group | Count | Geo mean fair fusion speedup | Top-1 accuracy | Mean regret |
| --- | ---: | ---: | ---: | ---: |
| representative | 2 | 0.9773x | 100.00% | 0.00% |
| fusion-friendly stress | 8 | 1.2499x | 75.00% | 7.32% |
| fusion-unfriendly compute-heavy | 6 | 1.1343x | 100.00% | 0.00% |
| exact-profiled | 21 | 1.1828x | 100.00% | 0.00% |
| unprofiled fallback | 4 | 1.2264x | 0.00% | 22.81% |
| bias | 13 | 1.2032x | 84.62% | 3.57% |
| elementwise Add | 12 | 1.1752x | 83.33% | 3.73% |

Best and worst cases:

- Best-case fair fusion speedup: `1.6017x`
- Worst-case fair fusion speedup: `0.9749x`

Exact-profiled selection:

- Top-1 accuracy: `100%`
- Mean regret: `0`
- Median regret: `0`
- P95 regret: `0`

Held-out fallback:

- Mean regret: `0.2281`
- Median regret: `0.2246`
- P95 regret: `0.2980`

Correlation with fair fusion speedup:

| Feature | Pearson | Spearman |
| --- | ---: | ---: |
| K | -0.2653 | -0.8411 |
| output_bytes | 0.8368 | 0.7086 |
| fusion_pressure_score | 0.8485 | 0.8411 |
| matmul_flops | 0.2173 | 0.0664 |
| output_bytes / matmul_flops | 0.8485 | 0.8411 |

These are correlations only and do not establish causality.

## Formal Artifacts

Artifacts:

- `trace/matmul_postop_multi_shape_profile.json`
- `trace/matmul_postop_multi_shape_validation.json`
- `trace/matmul_postop_selection_summary.json`
- `trace/matmul_postop_correlation.csv`
- `trace/matmul_postop_multi_shape_report.md`

Formal benchmark configuration:

- warmup: `50`
- iterations: `300`
- repeats: `5`
- build type: `Release`

Remote provenance:

- host: `allen-ZenBook-UX534FTC-UX534FT`
- remote benchmark address: `100.87.220.5`
- base commit at measurement time: `fc3480229fa4c8f789ab228c4a3b5c13a2450386`
- kernel: `Linux 7.0.0-27-generic x86_64`
- CPU: `Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz`
- compiler: `c++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0`
- CMake: `4.2.3`

The formal-core aggregate artifact was generated from one run with a single
process ID and one UTC start timestamp. Each workload-pattern record stores the
profile artifact SHA-256 and ExecutionPlan SHA-256. Use-plan outputs are marked
as validation artifacts, not profile-selection evidence.

## Supported Claims

```text
Implemented true one-pass MatMul post-op fusion with fair tiled baselines.
```

```text
Achieved 1.25x geometric-mean fair fusion speedup on fusion-friendly stress workloads and up to 1.60x best-case speedup.
```

```text
Achieved 100% top-1 kernel-selection accuracy with zero regret on stable exact-profiled formal-core workloads.
```

```text
Identified a strong correlation between analytical fusion pressure and measured fusion benefit.
```

## Unsupported Claims

Do not claim:

- fusion improves all representative workloads
- model-level speedup
- actual DRAM traffic reduction
- actual peak memory reduction
- held-out generalization is solved
- single-host results generalize to all CPUs

## Next Step

```text
Build and validate a shape-aware analytical/hybrid selection policy to reduce held-out mean regret from the current 22.81% baseline.
```
