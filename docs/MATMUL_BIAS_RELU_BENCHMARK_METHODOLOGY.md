# MatMul Post-Op ReLU Benchmark Methodology

This benchmark evaluates MatMul followed by either bias Add or elementwise Add,
then ReLU. It is scoped to fair benchmark infrastructure and explicit CPU
implementation variants only. It does not change MLIR fusion passes, runtime
dispatch, quantized kernels, CUDA kernels, or Metal kernels.

## Variants

The benchmark reports four variants:

- `naive_unfused`: naive MatMul, separate Add traversal, separate ReLU traversal.
- `tiled_unfused`: tiled MatMul, separate Add traversal, separate ReLU traversal.
- `naive_one_pass_fused`: naive MatMul accumulation with Add/ReLU applied before
  the final output store.
- `tiled_one_pass_fused`: tiled MatMul accumulation with Add/ReLU applied before
  the final output store.

The primary fair fusion comparison is:

```text
tiled_unfused vs tiled_one_pass_fused
```

The full-stack comparison is:

```text
naive_unfused vs tiled_one_pass_fused
```

Do not describe the full-stack comparison as fusion-only speedup because it
combines tiling and post-op fusion.

## Add Semantics

The benchmark supports two distinct post-op forms:

- `--pattern bias`: MatMul `[M,K] x [K,N]`, bias `[N]`, ReLU.
- `--pattern elementwise-add`: MatMul `[M,K] x [K,N]`, addend `[M,N]`, ReLU.

A full `[M,N]` tensor is reported as `elementwise_add`, not as bias.

## Run

Use a Release build for reported numbers:

```bash
cmake -S . -B build-benchmark-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-benchmark-release --target run_mlir_fused_kernel_benchmark -j
```

Formal runs:

```bash
build-benchmark-release/run_mlir_fused_kernel_benchmark \
  --pattern all \
  --variant all \
  --warmup 50 \
  --iterations 300 \
  --repeats 5 \
  --output trace/matmul_postop_relu_kernel_profile.json \
  --report-output trace/matmul_postop_relu_benchmark_report.md
```

Flags:

- `--pattern bias|elementwise-add|all`
- `--variant naive-unfused|tiled-unfused|naive-one-pass-fused|tiled-one-pass-fused|all`
- `--warmup`: discarded warmup iterations before measured iterations.
- `--iterations`: measured iterations per repeat.
- `--repeats`: process-local repeat groups used for aggregate statistics.
- `--tile-size`: shared tiled MatMul tile size for V1 and V3.
- `--output`: JSON artifact path.
- `--report-output`: markdown report path.

## Correctness

`naive_unfused` is the authoritative reference. All other variants are checked
against that output with:

- absolute tolerance
- relative tolerance
- maximum absolute error
- maximum relative error
- NaN detection
- Inf detection

Correctness validation, allocation, initialization, JSON writing, and report
generation are outside timed regions.

## One-Pass Invariants

The fused variants report static implementation metadata:

```json
{
  "post_op_strategy": "one_pass_fused",
  "intermediate_tensor_count": 0,
  "full_output_post_op_passes": 0,
  "final_output_store_passes": 1
}
```

These are source-level implementation properties, not measured hardware memory
traffic or peak memory claims.

## Google Benchmark Alignment

The harness preserves the methodology infrastructure from the previous PR:

- Warmup iterations are discarded before measurement.
- Measured iterations are configurable.
- Repeats are configurable.
- Aggregate statistics include sample count, mean, median, p50, p95, min, max,
  sample standard deviation, and coefficient of variation.
- JSON output is machine-readable for future cost-model ingestion.
- Release mode is required for formal benchmark claims.

## Profile-Guided Kernel Selection

The benchmark JSON doubles as the measured kernel benchmark profile consumed
by the compiler exporter (`tools/mlir_fusion_to_runtime_json.py`, via
`KERNEL_PROFILE=... tools/run_mlir_fusion_pipeline.sh`). The document is
self-identifying:

- `schema: "kernel_benchmark_profile"`, `schema_version: 2`,
  `benchmark: "matmul_postop_relu"`.
- Machine identity, build type/compiler, warmup/iteration/repeat counts, dtype,
  and M/N/K live in the top-level `machine`, `build`, and `configuration`
  blocks and apply to every measurement in the document.
- Each `patterns.<pattern>.variants.<variant>` record carries `kernel_id`,
  latency statistics (mean/p50/p95/stddev/CV), correctness status, and
  implementation properties including the measured tile size.

### Strict matching

A measurement is used as selection evidence only when ALL of these match the
op being planned:

- semantic pattern (`postop_semantics`: `bias_shape_N` vs
  `elementwise_add_shape_MxN` — bias `[N]` and elementwise Add `[M,N]` are
  distinct profile keys)
- backend (`cpu`)
- dtype
- exact M, N, K
- `kernel_id`
- kernel configuration: tiled kernels must be measured at exactly the planned
  tile config; naive kernels do not consume the tile configuration and are
  config-compatible by construction
- `correctness.passed == true`

### Measurement validation

A measurement is rejected (never silently used) when any of these fail:

- latency statistics must be finite and positive (mean, p50, p95)
- `warmup > 0`, `iterations > 0`, `repeats > 0`
- coefficient of variation must be finite, non-negative, and within the
  documented acceptance threshold of **5%** (`MAX_ACCEPTED_CV = 0.05`)
- profile `schema_version` must be supported (currently: 2)
- `kernel_id` must be present
- documents produced by `--mode use-plan` are rejected as selection evidence
  (circular-measurement guard: a plan-driven validation run must never feed
  back into the selection that produced the plan)

When multiple `--kernel-profile` documents provide a measurement for the same
kernel and key, the later document overrides the earlier one.

### Ranking and tie-breaking

Eligible legal candidates are ranked by **mean latency** (`mean_latency_ms`);
p50/p95/CV are retained as supporting evidence in the emitted plan. Exact ties
are broken deterministically, in this order:

1. lower `p95_ms`
2. lower coefficient of variation
3. fewer runtime dispatches (one-pass fused = 1, unfused = 3)
4. fewer intermediate tensors (one-pass fused = 0, unfused = 2)
5. stable lexical `kernel_id` ordering

Ranking never depends on JSON object iteration order.

### Fallback policy

When no valid exact-match evidence exists, the exporter selects the
deterministic safe fallback `cpu_tiled_<pattern>_unfused_f32` and emits
`selection.policy = "safe_fallback"`, `fallback_used = true`, and one of these
structured reasons:

- `profile_not_provided` (no `--kernel-profile`, or all profile files missing)
- `unsupported_profile_schema` (wrong schema version, non-benchmark document,
  or a rejected `use-plan` document)
- `no_matching_pattern`
- `no_exact_shape_match` (shape or dtype mismatch)
- `no_correctness_passing_candidate`
- `invalid_profile_measurement` (bad latency/warmup/iterations/CV, missing
  `kernel_id`, or kernel-config mismatches; detail in `fallback_detail`)
- `all_profiled_candidates_illegal`

A fallback selection never claims `policy = "profile_guided_latency"`.

### ExecutionPlan emission

The typed `operations[]` entry gains a `selection` object (policy, metric,
selected value, profile schema version, match kind, fallback flags, tie-break
order, CV threshold, profile source and generation mode) and a
`kernel_candidates` array (per legal candidate: profile latency/p50/p95/CV,
rank, eligibility, and ineligibility reason). The runtime contract fields
(`op_id`, `op_type`, `backend`, `selected_kernel`, `kernel_config`, `inputs`,
`outputs`) are unchanged.

### Anti-leakage protocol

1. Generate the profile in a dedicated profiling run (`force-variant` or
   `sweep-candidates` mode) and save the artifact.
2. Run the compiler pipeline with `KERNEL_PROFILE` pointing at that artifact;
   the exporter records the profile file's SHA-256 and mtime in the plan's
   `kernel_profile.fingerprints` block.
3. Validate with a separate `--mode use-plan` benchmark invocation, and
   compare against a separate fresh `--mode sweep-candidates` invocation.
   `use-plan` outputs are rejected as selection evidence by the loader.

## Remaining Limitations

- Repeats are in-process repeat groups, not separate OS process launches.
- CPU frequency scaling, thermal state, and host load are recorded but not
  controlled by the benchmark.
- The benchmark records static implementation properties, not hardware counter
  memory traffic.
- Broad shape sweeps are intentionally out of scope for this PR.
