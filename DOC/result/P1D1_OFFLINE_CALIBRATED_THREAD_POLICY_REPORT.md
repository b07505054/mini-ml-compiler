# Phase P1D.1 - IR-Centered Offline-Calibrated Raspberry Pi Thread-Schedule Policy

Last verified: 2026-07-13
Source host: GPU Linux `/home/allen/Desktop/Project/ml-graph-compiler-runtime`
Compiler baseline HEAD: `63eac07544b425810d644f50092804883f1ad1b5`
Runtime baseline HEAD: `7dd4288e0f63fd0e7bb002879dc1209c774df169`
Raspberry Pi target: `edgeaiplatform`, `aarch64`, performance governor, `throttled=0x0`

Verdict: `PASSED_IR_CENTERED_LOW_REGRET_THREAD_POLICY`

## Architectural Layer Classification

P1D.1 is a compiler policy change at the Semantic IR -> legality -> Execution Contract boundary.

- Semantic IR authority: shape, dtype, and fused-region identity are read from IR.
- Legality: kernel selection and declared thread schedules must already be legal.
- Evidence: offline Raspberry Pi P1D calibration ranks legal schedules only.
- Implementation IR truth boundary: this does not generate parallel loop IR. The selected portable CPU kernel is opaque prebuilt Runtime code, so `thread_count`, `partition_axis`, and `partition_strategy` remain in the ExecutionPlan contract.
- Runtime: unchanged exact validation and dispatch.

## Baseline Reproduction

Before implementation, fresh compiler plans for `8x8x8` and `256x256x256` both selected:

```json
{"thread_count": 1, "partition_axis": "none", "partition_strategy": "serial"}
```

A real Pi serial correctness run on `64x64x64` passed with kernel self-report `thread_count=1`, `partition_axis=none`, `partition_strategy=serial`, max abs error `9.27e-06`, median latency `0.200444 ms`, performance governor, and no throttling.

Runtime P1D tests initially skipped on GPU Linux because the native test executable was absent. After compiling the existing Runtime C++ kernel without source changes, the test suite ran. P1D.1 required a test-only Runtime expectation update because the compiler now selects 4-thread split-M for the existing `128^3` test plan.

## Calibration and Leakage Audit

Committed P1D evidence reconstructed from:

- `heterogeneous-inference-runtime/results/p1d_raspberry_pi_thread_decomposition/p1d_workload_manifest.json`
- `heterogeneous-inference-runtime/results/p1d_raspberry_pi_thread_decomposition/p1d_raw_measurements.json`
- `heterogeneous-inference-runtime/results/p1d_raspberry_pi_thread_decomposition/p1d_oracle_analysis.json`
- `heterogeneous-inference-runtime/results/p1d_raspberry_pi_thread_decomposition/p1d_regret_analysis.json`

Evidence hashes:

- Raw measurements SHA-256: `92aba0b1dd80846b469c870b921293c9c63054c84fa43bb9bdcf2afc4b6fa375`.
- Policy artifact evidence reference: `sha256:92aba0b1dd80846b469c870b921293c9c63054c84fa43bb9bdcf2afc4b6fa375`.
- Workload manifest SHA-256: `176b58b26c62a584c4d85954048be04e8eb1d239fc31f7003576e648c314051d`.

Leakage result: pass.

- Calibration workloads: 8.
- Held-out workloads: 10.
- Correctness-only workloads: 5.
- Calibration and held-out IDs are disjoint.
- Correctness-only workloads were not used for fitting.
- Candidate set A-E was frozen before measurement.
- Threshold was selected from calibration workloads only.
- Held-out latency, winners, regret, and labels were not used to choose the metric, threshold, boundary rule, or schedule choice.

## IR-Derived Metric

Chosen metric: `matmul_mnk = M * N * K`.

IR source: existing `computeShapeFacts()` in `mlir_passes/include/serving/OpShapeFacts.h`.

Semantic meaning: static matrix multiplication work dimension for the fused MatMul + Bias + ReLU region. For this fused op, `flops_estimate = 2 * M * N * K`, so `matmul_mnk` is FLOPs/2 and avoids dtype-dependent naming.

Availability: compile-time only for fully static ranked tensor shapes. Dynamic or invalid dimensions do not activate parallel policy.

Overflow: the compiler uses checked multiplication before comparing to the threshold; overflow safely falls back to serial if serial is legal.

## Derived Threshold and Boundary Rule

Final threshold: `262144`.

Boundary rule: `metric >= threshold` selects 4-thread split-M; below threshold selects serial.

Reason: calibration-only evidence has a serial winner at `16^3 = 4096`, and the smallest calibration workload where the selected parallel region is supported by measured evidence is `64^3 = 262144`. A geometric midpoint (`32768`) was rejected during validation because it lies in an unmeasured gap and a real synthetic boundary run showed thread overhead. The final threshold is therefore the smallest measured calibration parallel-winning region, not a held-out-tuned value.

## Policy Artifact

Compiler-local artifact:

`configs/thread_schedule_policies/raspberry_pi5_cortex_a76_p1d1_thread_policy.json`

Profile reference:

`configs/target_profiles/raspberry_pi5_cortex_a76_cpu.json`

The profile stores only artifact path, expected policy ID, and expected evidence hash. It does not embed raw latency tables.

Policy fields include policy ID/version, target profile ID, fused-region identity, dtype, kernel ID, metric, threshold, boundary rule, below/above schedules, evidence reference/hash, calibration scope, generated timestamp, calibration commit provenance, and truth boundary.

Truth boundary: offline Raspberry Pi calibration only; not current-compilation measurement, not generic ARM scheduling, not accuracy evidence, valid only for the declared profile/kernel/dtype/region contract.

## Legality Preconditions

The compiler applies policy only after:

- op is `fused_matmul_bias_relu`
- static `M/N/K` are known and valid
- dtype is `f32`
- selected kernel is `portable_fused_matmul_bias_relu_bm32_bn128_bk32`
- serial schedule is declared
- 4-thread split-M schedule is declared
- `physicalComputeUnits >= 4`
- target profile matches `raspberry-pi5-cortex-a76-cpu`
- policy ID and evidence hash match the profile reference
- kernel descriptor legality already succeeded

If a precondition fails and serial is legal, the compiler selects serial and records the reason. Policy never creates a schedule candidate.

## Compiler Implementation

Changed compiler behavior:

- `compile-for-target` reads a compiler-local offline thread policy artifact referenced by the target profile.
- `TargetConstraints` lowers validated policy metadata into module attrs.
- `KernelSelectionPass` preserves old P1D descriptor-order behavior when no valid policy exists.
- With a valid policy, `KernelSelectionPass` resolves `matmul_mnk` from IR and selects serial or 4-thread split-M among already-declared schedules.
- `ExecutionPlanBuilder` and `ExecutionPlanExporter` preserve policy provenance inside `thread_schedule`.

Runtime implementation is unchanged.

## ExecutionPlan Contract Examples

Tiny region (`8x8x8`, metric `512`):

```json
{
  "status": "selected",
  "thread_count": 1,
  "partition_axis": "none",
  "partition_strategy": "serial",
  "policy_metric": "matmul_mnk",
  "policy_metric_value": 512,
  "policy_threshold": 262144,
  "policy_selection_reason": "metric_below_threshold_select_serial"
}
```

Boundary/parallel region (`64x64x64`, metric `262144`):

```json
{
  "status": "selected",
  "thread_count": 4,
  "partition_axis": "m",
  "partition_strategy": "contiguous_chunks",
  "policy_metric": "matmul_mnk",
  "policy_metric_value": 262144,
  "policy_threshold": 262144,
  "policy_selection_reason": "metric_at_or_above_threshold_select_parallel"
}
```

## Held-Out Evaluation

Evaluation used the ten held-out workloads only after freezing metric and threshold.

| Policy | Exact match | Mean regret | Median regret | P95 regret | Max regret | Avg speedup vs serial | Worst slowdown vs serial |
|---|---:|---:|---:|---:|---:|---:|---:|
| Always serial | 10.0% | 233.004% | 258.048% | 297.012% | 297.281% | 1.000x | 1.000x |
| Always 2-thread split-M | 0.0% | 359.620% | 89.591% | 2687.106% | 3042.585% | 1.717x | 31.426x |
| Always 4-thread split-M | 76.7% | 408.403% | 0.000% | 4045.207% | 4169.654% | 3.230x | 42.697x |
| Always 2-thread split-N | 0.0% | 349.440% | 91.086% | 2642.357% | 2797.840% | 1.709x | 28.978x |
| Always 4-thread split-N | 13.3% | 407.345% | 2.013% | 4030.371% | 4094.174% | 3.147x | 41.942x |
| Implemented threshold policy | 86.7% | 0.067% | 0.000% | 0.489% | 0.769% | 3.328x | 1.000x |

Selection counts over 30 held-out workload/session pairs:

- Serial: 3
- 4-thread split-M: 27

Stable-region match rate: 100% over 8 non-conflict held-out workloads. Two held-out workloads had cross-session conflicts.

## Raspberry Pi End-to-End Validation

Fresh compiler plans were deployed to the real Raspberry Pi target and executed through the unchanged Runtime portable CPU adapter copied under `/tmp`, pointing at the existing deployed native kernel.

Environment:

- Hostname: `edgeaiplatform`
- Architecture: `aarch64`
- Governor: `performance` on all four cores
- Temperature: `48.8'C`
- Throttling: `throttled=0x0`
- UTC: `2026-07-13T17:33:44Z`

| Case | Shape | Selected schedule | Median ms | P95 ms | Serial median ms | Speedup vs serial | Result |
|---|---:|---|---:|---:|---:|---:|---|
| Tiny serial region | 8x8x8 | 1 thread serial | 0.000963 | 0.000982 | 0.000945 | 0.981x | correctness passed |
| Large parallel region | 256x256x256 | 4-thread split-M | 3.213220 | 3.362000 | 12.473500 | 3.882x | correctness passed |
| Boundary region | 64x64x64 | 4-thread split-M | 0.090277 | 0.114852 | 0.200259 | 2.218x | correctness passed |

Runtime rejection and compatibility:

- Mutated `thread_count=8` was rejected: Runtime refused to silently clamp or round.
- Old plan without `thread_schedule` defaulted to 1-thread serial, median `0.004037 ms`.
- Plan with policy provenance was accepted; Runtime ignored provenance fields and validated the exact schedule.

## Regression Tests

Compiler:

- `cmake --build build-mlir --target compile-for-target -j4`: pass.
- `python3 tests/test_p1d1_thread_schedule_policy.py`: 15 tests pass.

Runtime:

- Existing C++ kernel compiled from unchanged source for tests only.
- `.venv/bin/python -m pytest tests/test_p1d_thread_schedule_contract.py -q`: 19 tests pass after test-only expected schedule update.

## Limitations

- This is not a unified `ImplementationCandidate` implementation.
- This is not a general ARM scheduling policy.
- This does not implement split-N live policy.
- This does not add 2-thread live policy.
- This does not change tile/kernel selection.
- This does not generate parallel loop Implementation IR.
- This does not modify Runtime production implementation.
- This does not perform online benchmarking.
- The threshold is valid only for the declared Raspberry Pi profile, fused op, f32 dtype, and fixed portable kernel.

## Final Verdict

`PASSED_IR_CENTERED_LOW_REGRET_THREAD_POLICY`
