# Stage 12 Schedule Comparison Summary

## primary: primary_unroll1 vs primary_unroll2
Classification: **A** -- Scheduling win likely -- better overlap or shorter modeled throughput, no harmful spill increase
- schedule_unroll_k increased (1 -> 2): fewer dynamic K-loop iterations at runtime
- accumulator_chains: baseline=16 scheduled=16; same_accumulator_distance median: baseline=18.0 scheduled=18.0; load_to_use median: baseline=1.0 scheduled=1.0 (not worse)
- no new spills; post_ra physical vector registers unchanged (28 -> 28, both at or under the 32-register budget); pre_ra approx_peak_live_vector_registers (MIR-derived estimate, not gating): baseline=113 scheduled=145

## primary_full_unroll_edge_case: primary_unroll1 vs primary_full_unroll
Classification: **D** -- Regression risk -- spills, worse modeled throughput, broken FMLA generation, or excessive code growth
- spill_stores delta=11, reload_loads delta=12 (scheduled variant introduces new allocator spills)

## alt_k_tile: alt_k_tile_unroll1 vs alt_k_tile_unroll2
Classification: **D** -- Regression risk -- spills, worse modeled throughput, broken FMLA generation, or excessive code growth
- spill_stores delta=2, reload_loads delta=2 (scheduled variant introduces new allocator spills)

## cube64: cube64_unroll1 vs cube64_unroll2
Classification: **A** -- Scheduling win likely -- better overlap or shorter modeled throughput, no harmful spill increase
- schedule_unroll_k increased (1 -> 2): fewer dynamic K-loop iterations at runtime
- accumulator_chains: baseline=16 scheduled=16; same_accumulator_distance median: baseline=19.5 scheduled=20.0; load_to_use median: baseline=0.0 scheduled=0.0 (not worse)
- no new spills; post_ra physical vector registers unchanged (28 -> 28, both at or under the 32-register budget); pre_ra approx_peak_live_vector_registers (MIR-derived estimate, not gating): baseline=113 scheduled=145

