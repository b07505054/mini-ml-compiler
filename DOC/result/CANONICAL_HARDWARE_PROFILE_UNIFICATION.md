# Canonical Hardware Profile Unification

## Summary

Before this change, the C++ compiler target-profile path and the Python Triton
fused-config selector used disconnected hardware facts. The compiler consumed
`configs/target_profiles/*.json` through `TargetConstraints`, while the Triton
selector resolved scheduling capacity from benchmark artifact metadata or a
Python compatibility default of `16`.

This PR makes the target profile the canonical source for the Triton selector's
effective compute-unit count without changing the selector formulas or measured
oracle data.

## Canonical Fields

`configs/target_profiles/nvidia_gtx1650_maxq.json` now declares:

- `hardwareExecutionProfile.physicalComputeUnits = 16`
- `hardwareExecutionProfile.effectiveComputeUnits = 16`
- `hardwareExecutionProfile.maxConcurrentWorkItemsPerUnit = 1`
- `hardwareExecutionProfile.supportsLatencyHiding = true`
- `hardwareExecutionProfile.localMemoryKind = "software_managed_shared_memory"`

Other profiles remain backward compatible: missing fields stay missing and
tooling records explicit fallback provenance.

## Resolution Precedence

The Python selector resolves effective compute units in this order:

1. explicit CLI override, only with intentional override support
2. canonical target profile
3. validated benchmark artifact environment
4. compatibility fallback

The selected source is recorded in `hardware_profile.effective_compute_units_source`.
Profile/artifact mismatches are rejected rather than silently accepted.

## C++ Dataflow

```text
target profile JSON
→ compile-for-target TargetDeviceProfile
→ TargetConstraints.hardware_execution_profile
→ target.hardware.* MLIR attributes
→ TargetConstraints::fromModule round-trip
```

The new MLIR attributes are:

- `target.hardware.physical_compute_units`
- `target.hardware.effective_compute_units`
- `target.hardware.max_concurrent_work_items_per_unit`
- `target.hardware.supports_latency_hiding`
- `target.hardware.local_memory_kind`

## Python Dataflow

```text
--target-profile configs/target_profiles/nvidia_gtx1650_maxq.json
→ tools/target_hardware_profile.py
→ effective_compute_units = 16
→ sm_count compatibility alias = 16
→ existing Triton selector formulas
```

The formulas are intentionally unchanged in this PR.

## Compatibility Aliases

The selector keeps old Triton-specific fields and emits generic aliases:

- `total_output_program_count` → `parallel_work_items`
- `programs_per_sm` → `work_items_per_compute_unit`
- `output_program_waves` → `execution_waves`
- `work_per_program` → `compute_work_per_item`
- `k_dominance_ratio` → `reduction_dominance`
- `masked_output_fraction` → `padding_waste_ratio`
- `padding_amplification` → `work_amplification`

Artifacts use `feature_schema_version = 2`.

## Triton-Specific Fields

These remain explicitly Triton/NVIDIA-specific and are not claimed to be
backend-neutral:

- `block_m`
- `block_n`
- `block_k`
- `num_warps`
- `num_stages`
- `sm_count`
- `programs_per_sm`

## Parity Results

Running the repaired selector with the GTX target profile preserved the prior
selector behavior:

- top-1 accuracy: `0.8571428571`
- macro accuracy: `0.9166666667`
- mean regret: `0.0007865582`
- p95 regret: `0.0038541353`
- max regret: `0.0055059075`
- plan config validation rate: `1.0`

For `M=64,N=64,K=4096`, calibrated ranking remains:

1. `bm16_bn16_bk32_w4_s3`
2. `bm32_bn32_bk32_w4_s3`
3. `bm16_bn64_bk32_w4_s3`
4. `bm64_bn64_bk32_w4_s3`

## Scope Boundary

This is not yet a universal CPU/GPU/NPU hardware abstraction. It only unifies
the canonical source for scheduling-visible compute-unit facts and adds
compatibility aliases. Backend-neutral formulas and adapters remain future work.

## Next PR

Next justified PR:

```text
Backend Feature Adapter and Hardware-Normalized Feature Contract
```

That PR should define backend-normalized formulas rather than mechanically
renaming Triton/NVIDIA-specific features.
