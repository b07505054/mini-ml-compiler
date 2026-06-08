# Compiler Runtime Decision Report

Status: `passed`

## Planner Decision

- Chosen plan: `current`
- Chosen latency: `0.557655 ms`
- All-Metal candidate latency: `0.630856 ms`
- All-Metal transfer cost: `0.137995 ms`
- Delta vs all-Metal: `0.073201 ms`
- Reason: CostReport-driven planner keeps memory-heavy CPU ops on CPU because moving pool/flatten to Metal adds transfer cost.

## Kernel And Dispatch Decisions

| Case | HIR op | Kernel | Backend | Selection | Custom ms | Fallback ms | Speedup | Tile | SRAM bytes |
|---|---|---|---|---|---:|---:|---:|---|---:|
| MatMul-Bias-ReLU | hir.fused_matmul_bias_relu | unfused_matmul_add_relu | CPU | profile_calibrated_fallback | 5.354697 | 5.145997 | 0.961025 | 16x64x128 | 49152 |
| RMSNorm | hir.fused_rmsnorm | fused_rmsnorm_cuda | CUDA | gpu_pgo_like_lowest_p95_latency | 0.030196 | 0.088261 | 2.923 |  |  |
| INT8 QMatMul-Bias-ReLU | hir.fused_qmatmul_bias_relu | int8_qmatmul_bias_relu | CPU | profile_calibrated_fastest | 4.94182 | 5.72657 | 1.1588 | 64x64x128 | 24576 |

## CPU Software Prefetch Candidate

- Input: `HIR fused MatMul-Bias-ReLU CPU backend workload`
- Decision: `profile-guided choice between tiled CPU kernel and prefetch tiled CPU kernel`
- Metric: `p50/p95 latency, speedup, correctness, estimated bytes moved`
- Candidate: `fused_matmul_add_relu_prefetch`
- Fallback: `fused_matmul_add_relu_optimized`
- Baseline p95: `43.112046` ms
- Prefetch p95: `47.359127` ms
- Selection ready: `False`
- Selection reason: `fallback_p95_not_improved`

## Checks

- planner_uses_cost_report_v2: `True`
- planner_decision_changed_from_all_metal: `True`
- all_metal_has_transfer_cost: `True`
- matmul_profile_falls_back_when_fused_slower: `True`
- rmsnorm_profile_selects_cuda: `True`
- qmatmul_profile_selects_int8: `True`
- dispatch_descriptors_have_tiles: `True`
- prefetch_candidate_profile_valid: `True`
