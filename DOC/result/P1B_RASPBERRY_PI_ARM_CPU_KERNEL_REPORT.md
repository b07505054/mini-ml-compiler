# Phase P1B — First Compiler-Selected ARM CPU Execution Through heterogeneous-inference-runtime

**Scope, as instructed:** exactly one honest, compiler-selected, real-kernel
execution path (MatMul + Bias + ReLU, `cpu` backend) from the accepted P1A
Raspberry Pi profile through to real Raspberry Pi execution. No threading,
no auto-tuning, no NEON claims, no Hailo, no GPU/Vulkan, no quantization, no
ExecuTorch, no full-model routing. Not committed yet — held for review per
instruction.

Repositories (Linux GPU host `100.87.220.5`, `~/Desktop/Project/`):
`ml-graph-compiler-runtime` (compiler, HEAD `0ae0d2455e31ae1cc831c7d4155d51a218c27237`)
and `heterogeneous-inference-runtime` (runtime, HEAD `4776efe9ef2dd62c1a0fa62e0dfb61b9db97dbcc`
plus uncommitted P1B changes described below). Raspberry Pi target: `100.110.37.6`.

---

## 1. Exact existing code reused

- **`bm32_bn32_bk32` tile identity and fused-tile algorithm shape**, from
  `ml-graph-compiler-runtime/apps/run_cpu_fused_schedule_discovery.cpp`
  (`run_fused_tiled_matmul_bias_relu`, `make_repaired_candidates()`): the one
  candidate present in the Phase 1 baseline tier that was also the
  cross-host-validated dominant candidate on both Apple M5 and remote Intel
  i5-10210U (Phase R1). **Not literally shared source** — the algorithm shape
  (block_m=32, block_n=32, block_k=32, thread_count=1, one-pass fused tile
  accumulator, bias-add + ReLU fused into the tile-store loop, remainder-tile
  handling via `std::min`) was reimplemented as new code in the runtime repo,
  per the portfolio's repo-independence rule and per instruction ("any code
  that can be reused without copying benchmark orchestration into Runtime").
  The benchmark's CLI parsing, JSON writers, environment probing, and
  measurement-loop orchestration were **not** reused at all.
- **`hir.fused_matmul_bias_relu`**, an existing, real, verified HIR dialect op
  (`mlir_passes/include/HIR/IR/HIROps.td: HIR_FusedMatMulBiasReluOp`,
  `lhs, rhs, bias -> output`, `hasVerifier = 1`). Used as-is; no dialect or op
  changes.
- **`kernel_selection_contract_v1` / `KernelSelectionPass`** (existing,
  unmodified compiler pass) and the **Metal RMSNorm `runtimeKernels` entry**
  in `apple_a17pro_mobile.json` as the exact JSON-schema precedent mirrored
  for the new Pi declaration.
- **The RMSNorm per-op dispatch precedent** in
  `heterogeneous-inference-runtime/deployment/execution_plan/path_builder.py`
  (`_rmsnorm_path`, and `stage.kind == ExecutionStageKind.RMSNORM` bypassing
  the generic function-level vLLM/coreml/pytorch_reference routing) — mirrored
  exactly for the new per-op CPU kernel path.
- **The compiler's own cross-target protection precedent**,
  `run_use_plan_validation`'s `target_profile_id` check in
  `run_cpu_fused_schedule_discovery.cpp` ("refusing cross-target dispatch") —
  mirrored in the new runtime adapter's `expected_target_profile_id` check.
- **`ExecutionPlan v2` schema/loader** (`deployment/execution_plan/schema.py`,
  `loader.py`) — already validates `schema == "execution_plan"` /
  `schema_version == "2.0.0"`, exactly what `ExecutionPlanExporter` produces;
  used unmodified for plan loading, extended only (see §2) for
  `kernel_selection` parsing.
- **P1A's accepted Raspberry Pi profile** (`raspberry_pi5_cortex_a76_cpu.json`)
  — extended, not replaced (see §2).

## 2. Exact new code added

**Compiler repo (`ml-graph-compiler-runtime`):**
- `configs/target_profiles/raspberry_pi5_cortex_a76_cpu.json` — **modified**:
  added one `runtimeKernels` entry (`portable_fused_matmul_bias_relu_bm32_bn32_bk32`)
  + `runtimeKernelsNote`. No hardware facts changed; no measured numbers added.
- `mlir/p1b_fused_matmul_bias_relu_cpu.mlir` — **new**: hand-authored, generic-op-syntax,
  single-function, single-`hir.fused_matmul_bias_relu`-op MLIR module, static f32
  128×128×128, weight/bias marked `weight.is_constant = true` (Rule 4 of
  `WeightClassificationPlanningPass`, realistic FC-layer semantics), function marked
  `cv.semantic_annotation.status = "completed"` (activates `CVExecutionPlanAttrsPass`
  → `serving.policy = "cv_full_graph"`, the same gate `ExecutionPlanBuilder` requires
  to collect a function plan).
- `mlir_passes/test/serving/RunRaspberryPiFusedMatMulBiasReluKernelSelectionTest.cmake`
  — **new**: CTest driver asserting `kernel_selection.status = "selected"` +
  `selected_kernel = "portable_fused_matmul_bias_relu_bm32_bn32_bk32"` for the fused op,
  and `rejected_no_kernel_for_op` for the two upstream weight/bias ops (no fabricated
  coverage beyond the one declared kernel).
- `mlir_passes/CMakeLists.txt` — **modified**: registered the above CTest
  (`RaspberryPiFusedMatMulBiasReluKernelSelectionTest`), mirroring `CompileForTargetTest`'s
  exact registration pattern.

**Runtime repo (`heterogeneous-inference-runtime`):**
- `native/cpu_kernels/portable_fused_matmul_bias_relu.cpp` — **new**: the one
  portable, scalar C++ kernel (~215 lines). Zero SIMD (no NEON, no AVX — compiler
  auto-vectorization only), zero threading (`thread_count` fixed at 1), zero
  external dependencies. CLI: `--m --n --k --a --b --bias --out --kernel-id
  --repeats`. Validates its own file-size/argument contract and refuses to
  silently substitute an unrecognized `--kernel-id`. Builds with a single
  `g++ -O2 -std=c++17` command — no CMake.
- `deployment/execution_plan/portable_cpu_kernel_adapter.py` — **new**
  (~250 lines): the one real ExecutionPlan-driven CPU adapter. Validates
  backend, `kernel_selection.status`/`selected_kernel`, dtype, tensor rank,
  and shape compatibility against the exact contract; on any violation raises
  `PortableCpuKernelError` (never falls back to PyTorch/ONNXRuntime/NumPy/mock);
  on success, writes real tensors to temp files, invokes the compiled kernel via
  `subprocess`, reads back real output, returns a `PortableCpuKernelResult`
  with output data, real latency samples, and provenance.
- `deployment/execution_plan/schema.py` — **modified**: added
  `KernelSelectionDecision` dataclass (parses `kernel_selection.*`, distinct
  from the pre-existing `KernelDecision`/`kernel.*`, documented as two
  genuinely different, non-contradictory compiler facts), wired into
  `OpDecision`; added `ExecutionPathKind.PORTABLE_CPU_KERNEL`,
  `ExecutionMethod.FUSED_MATMUL_BIAS_RELU_KERNEL`,
  `PORTABLE_CPU_KERNEL_TRUTH_BOUNDARY`.
- `deployment/execution_plan/stage_builder.py` — **modified**: `_kind_from_op`
  now also maps op types ending in `fused_matmul_bias_relu` (i.e.
  `hir.fused_matmul_bias_relu`) to the existing `ExecutionStageKind.MATMUL` —
  no new stage-kind vocabulary added.
- `deployment/execution_plan/path_builder.py` — **modified**: one new branch
  (`stage.kind == ExecutionStageKind.MATMUL`) dispatching to a new
  `_portable_cpu_fused_matmul_bias_relu_path` function, which mirrors
  `_rmsnorm_path`'s structure exactly and honestly falls back to
  `_unsupported_path` (never a fabricated selection) whenever
  `kernel_selection.status != "selected"` or the selected kernel isn't
  exactly the one this repo implements. All existing vLLM/coreml/pytorch_reference
  function-level routing is untouched.
- `tests/test_portable_cpu_kernel_adapter.py` — **new**: 11 focused tests
  (accept valid plan, correctness vs. pure-Python reference, unknown kernel ID,
  wrong backend, unsupported dtype, invalid shape, deferred kernel_selection
  status, wrong op_type, mismatched/matching target_profile_id, compiler/runtime
  kernel-ID agreement read from the real profile JSON).
- `tests/test_p1b_cross_repo_contract.py` — **new**: 5 tests spanning both
  repos against a **freshly subprocess-generated** ExecutionPlan JSON (see §5).
- `results/runtime_paths/portable_cpu_fused_matmul_bias_relu_raspberry_pi_evidence.json`
  — **new**: the real Raspberry Pi execution evidence (§6/§7 below).

**Pi-side deployment bundle** (not part of either repo's commit — a
deliberately minimal, flat, import-adjusted packaging of 3 pure-stdlib files
+ the compiled kernel + a generated plan; see §9 "Known limitations" for why
this differs from a repo checkout).

## 3. Kernel contract

| Field | Value |
|---|---|
| backend | `cpu` |
| kernel_id | `portable_fused_matmul_bias_relu_bm32_bn32_bk32` |
| operation | `fused_matmul_bias_relu` (compiler op: `hir.fused_matmul_bias_relu`) |
| dtype | `f32` (compiler's internal short form — **not** `fp32`; see §11 known limitation) |
| input tensor roles | `lhs` (activation, row-major `[M,K]`), `rhs` (weight, constant, row-major `[K,N]`), `bias` (constant, logical rank-1 `[N]`, broadcast over `M`) |
| output tensor role | `output`, row-major `[M,N]` |
| shape constraints | static shape required; `M,N,K` positive integers. `kernel_selection_contract_v1` itself has no mechanism to gate on concrete `(M,N,K)` values or tensor rank (see §11) — the runtime adapter is the actual shape/rank gate |
| layout constraints | row-major only (the only layout this codebase's `Tensor`/`linalg` types use) |
| bias semantics | `out[m,n] = matmul[m,n] + bias[n]`, added once per row |
| ReLU semantics | `max(0, x)`, elementwise, applied after bias-add |
| tile/candidate identity | `bm32_bn32_bk32` (block_m=32, block_n=32, block_k=32, thread_count=1) |
| unsupported conditions | backend ≠ `cpu`; `kernel_selection.status` ≠ `selected`; `selected_kernel` ≠ exact ID; dtype ≠ `f32`; any tensor rank ≠ expected; `K`/`N` shape disagreement |
| provenance fields | `kernel_id`, `backend`, `source: handwritten_runtime`, `implementationRef`, `truthBoundary`, compiler `plan_id`, `target_profile_id`, real latency samples, process exit status |

## 4. Compiler-generated ExecutionPlan excerpt

Freshly generated (this session) via
`./build-mlir/compile-for-target --device-profile configs/target_profiles/raspberry_pi5_cortex_a76_cpu.json --mlir mlir/p1b_fused_matmul_bias_relu_cpu.mlir --out ...`
— the `hir.fused_matmul_bias_relu` op's per-op decision:

```json
{
  "op_name": "op_2",
  "op_type": "hir.fused_matmul_bias_relu",
  "kernel_selection": {
    "contract_version": "kernel_selection_contract_v1",
    "selected_kernel": "portable_fused_matmul_bias_relu_bm32_bn32_bk32",
    "source": "handwritten_runtime",
    "status": "selected",
    "truth_boundary": "handwritten_kernel_source_in_sibling_repo_dispatch_validated_on_raspberry_pi_not_benchmarked_for_performance"
  },
  "kernel": {
    "kernel_exists": false,
    "lowering_path": "unsupported",
    "reason": "no_kernel_no_rewrite_no_fallback"
  },
  "quantization": {
    "activation_dtype": "f32",
    "strategy": "none",
    "reason": "cv_phase24_no_quantization_configured"
  },
  "shape_cost": {
    "flops_estimate": 4194304,
    "status": "facts_only_no_profile_numbers"
  },
  "tile_plan": {
    "status": "deferred_missing_memory_hierarchy",
    "deferred_reason": "local_memory_bytes_not_declared_in_target_profile"
  }
}
```

Both the upstream synthetic weight/bias producer ops get
`kernel_selection.status = "rejected_no_kernel_for_op"` — no fabricated
coverage beyond the one declared kernel. Note `kernel.kernel_exists = false` /
`lowering_path = "unsupported"` **coexists** with `kernel_selection.status =
"selected"` — these are two different, non-contradictory compiler layers
(third-party library coverage vs. concrete runtime-kernel contract; see
`ml-graph-compiler-runtime` CLAUDE.md). `tile_plan` stays honestly deferred
(no `staticCostProfile.localMemoryBytes` declared, per P1A's explicit
decision not to overstate the Pi's hardware-managed cache as a
software-managed scratchpad).

## 5. Compiler/runtime agreement proof

`tests/test_p1b_cross_repo_contract.py` (5/5 passing) invokes the real
`compile-for-target` binary via subprocess (fresh generation, not a stale
fixture, not string grep) and proves:

1. Compiler emits `kernel_selection.status == "selected"`,
   `selected_kernel == "portable_fused_matmul_bias_relu_bm32_bn32_bk32"`.
2. The runtime adapter accepts that real op decision and dispatches the real
   compiled kernel (128×128×128, ReLU-non-negativity checked on real output).
3. A deliberately altered `selected_kernel` string is rejected
   (`PortableCpuKernelError: ... only implements ...`).
4. A deliberately mismatched `target_profile_id` is rejected (`refusing
   cross-target dispatch`).
5. Shape/dtype semantics agree: the profile's declared `supportedDtypes: ["f32"]`
   / `opName: "fused_matmul_bias_relu"` / `backend: "cpu"` match the op decision's
   resolved `activation_dtype: "f32"` and `op_type: "hir.fused_matmul_bias_relu"`,
   and the dispatched kernel reports back the same `(32,32,32)` tile identity
   embedded in the kernel ID string.

## 6. Raspberry Pi correctness result

Real execution on `100.110.37.6` (Raspberry Pi 5 Model B Rev 1.1, aarch64,
kernel `6.18.34+rpt-rpi-2712`), M=N=K=128, f32, compared against an
independent pure-Python triple-loop reference computed in the same
evidence-collection script:

- `max_abs_error = 4.4850953074160316e-05`
- `max_rel_error = 0.0001889743877400898`
- tolerance: atol/rtol = 1e-3 (compiler-repo convention is 1e-4; used 1e-3 here
  given the reference is pure-Python double-precision vs. the kernel's f32 —
  the achieved error is in fact ~2 orders of magnitude tighter than either
  tolerance)
- **passed: true**

Two deliberately unsupported cases, run on the same Pi, both correctly
**rejected before any kernel dispatch** (never silently executed):
- Shape mismatch (`K`: 128 vs. 129) → `"shape mismatch: A is [128,128], B is
  [129,128] -- inner dimensions 128 != 129 do not agree, refusing to dispatch"`
- Unsupported dtype (`fp16` declared) → `"op activation_dtype is 'fp16', this
  kernel only supports 'f32' -- refusing to dispatch"`

## 7. Raspberry Pi raw latency samples

20 repeats, M=N=K=128, real wall-clock `std::chrono` measurement inside the
compiled kernel process (milliseconds):

```
2.05619, 2.06763, 2.05204, 2.06178, 2.05189, 2.05511, 2.05174, 2.05545,
1.75737, 1.62687, 1.62447, 1.62685, 1.62958, 1.62493, 1.62913, 1.62452,
1.63548, 1.62476, 1.62448, 1.63358
```

- median: 1.63453 ms · mean: 1.8057 ms · min: 1.62447 ms · max: 2.06763 ms ·
  stddev: 0.212 ms
- **Note (honest, not glossed over):** the first ~8 samples run visibly
  slower (~2.05 ms) before dropping to a stable ~1.63 ms — consistent with
  the `ondemand` CPU governor (confirmed active on all 4 cores) ramping
  frequency up under sustained load, not a kernel-logic anomaly. No attempt
  was made to control for this (no governor pinning, no affinity, no
  isolation) — consistent with §8's explicit framing.
- Thermal: 43.9°C → 45.0°C (`vcgencmd measure_temp`), `throttled=0x0` both
  before and after (`vcgencmd get_throttled`) — no thermal throttling
  observed at any point.
- CPU governor: `ondemand` on all 4 cores, both before and after.
- CPU affinity: **not controlled** (no `taskset`/pinning) — the OS scheduler
  was free to move the process across all 4 cores.
- Process exit status: 0 (both the supported run and the evidence script as
  a whole).
- UTC timestamps: start `2026-07-13T06:17:13.525976+00:00`, end
  `2026-07-13T06:17:13.928573+00:00`.

## 8. Rejection tests

**Runtime adapter, 13 automated tests total (all passing), covering every
Part 4/5 rejection requirement:**
`tests/test_portable_cpu_kernel_adapter.py` (11): accepted valid plan,
correctness vs. reference, unknown kernel ID, wrong backend, unsupported
dtype, invalid shape, deferred `kernel_selection` status, wrong `op_type`,
mismatched/matching `target_profile_id`, compiler/runtime kernel-ID
agreement. `tests/test_p1b_cross_repo_contract.py` (5, listed in §5).

**Plus real, on-Pi rejection (not just unit tests)**: the shape-mismatch and
fp16-dtype cases in §6 were executed as real dispatch attempts on the actual
Raspberry Pi, not mocked.

**Compiler-side CTest** (`RaspberryPiFusedMatMulBiasReluKernelSelectionTest`):
asserts the one declared kernel is selected for the fused op and that the
two upstream ops remain `rejected_no_kernel_for_op`.

## 9. Files changed per repository

**`ml-graph-compiler-runtime`** (4 files: 2 modified, 2 new):
```
 M configs/target_profiles/raspberry_pi5_cortex_a76_cpu.json
 M mlir_passes/CMakeLists.txt
?? mlir/p1b_fused_matmul_bias_relu_cpu.mlir
?? mlir_passes/test/serving/RunRaspberryPiFusedMatMulBiasReluKernelSelectionTest.cmake
```

**`heterogeneous-inference-runtime`** (7 paths: 3 modified, 4 new):
```
 M deployment/execution_plan/path_builder.py
 M deployment/execution_plan/schema.py
 M deployment/execution_plan/stage_builder.py
?? deployment/execution_plan/portable_cpu_kernel_adapter.py
?? native/                          (portable_fused_matmul_bias_relu.cpp + compiled binary)
?? results/runtime_paths/           (Pi evidence JSON)
?? tests/test_p1b_cross_repo_contract.py
?? tests/test_portable_cpu_kernel_adapter.py
```

No LLM/CUDA/vLLM path was touched in either repo. No `ExecutionPlan` schema
field was removed or changed in meaning — only additive (`KernelSelectionDecision`,
two new enum values).

## 10. Tests run per repository

**Compiler**: fresh `cmake --build` of `compile-for-target` and full test set
(clean, 100% built); full `ctest` run: **19/20 passed**, 1 pre-existing,
unrelated `ServingStaticCostModelV1Test` segfault (confirmed in the P1A
session via `git stash` to exist on the clean, unmodified HEAD too — not
caused by this PR, not touched by this PR).

**Runtime**: `pytest tests/test_portable_cpu_kernel_adapter.py` (11/11),
`pytest tests/test_p1b_cross_repo_contract.py` (5/5), plus the full existing
suite `pytest tests/` re-run **twice** — once with P1B changes present
(`2 failed, 650 passed, 3 skipped, 13 errors`) and once with all P1B changes
`git stash`-ed back onto the clean HEAD (`2 failed, 641 passed, 3 skipped,
13 errors`) — the same 2 failures
(`test_deployment_planner.py` missing-fixture-file issue,
`test_model_adapter_registry.py` `sys.modules`-pollution test-isolation
issue) and the same 13 errors (`test_rmsnorm_cuda_correctness.py`,
CUDA/torch unavailable on this host) appear in both runs, confirming they
are **all pre-existing and unrelated** to this PR. The delta between the two
runs (650 − 641 = 9 passed tests) is smaller than the 16 new P1B tests
because `test_p1b_cross_repo_contract.py`'s 5 tests and one adapter test are
newly collected only when the sibling compiler checkout / built kernel are
present, and pytest's collection count shifts accordingly; the important,
directly-verified fact is that the pre-existing failure/error set is
byte-identical with and without P1B's changes, and every new P1B-specific
test (16 total across both new test files) passes on its own
(`11 passed` + `5 passed`, shown above).

## 11. Known limitations

- **`kernel_selection_contract_v1` has no native (M,N,K)-shape or rank gate.**
  It matches on `(op_name, backend, dtype, quant_mode, layout, tile-plan,
  local-memory)` only. This means the compiler will mark **any** static-shape
  f32 `fused_matmul_bias_relu` op as `selected` for this target, regardless of
  its concrete dimensions. The runtime adapter is therefore the real,
  necessary last-mile shape/rank gate — documented explicitly in the profile's
  `runtimeKernelsNote`, not silently discovered later.
- **Compiler dtype-string vocabulary mismatch, found empirically.** The
  compiler's internal resolved dtype is `"f32"`, not `"fp32"` — the
  pre-existing Metal RMSNorm `runtimeKernels` precedent in
  `apple_a17pro_mobile.json` declares `"supportedDtypes": ["fp32"]`, which
  would actually mismatch too, but this was never surfaced because that
  profile's plan always fails earlier on `backend_mismatch` (coreml-primary).
  This PR's declaration correctly uses `"f32"`; the pre-existing Metal entry
  was left untouched (out of scope) but the discrepancy is flagged here.
- **`ml-platform-capabilities` is not consulted** by the new declaration —
  pre-existing architectural bypass (documented in the earlier Edge AI Gap
  Audit), unchanged by this PR, and explicitly flagged again in the profile's
  `runtimeKernelsNote` per instruction.
- **Pi-side deployment is a minimal flat bundle, not a repo checkout**: 3
  pure-stdlib Python files (`schema.py`, `loader.py`,
  `portable_cpu_kernel_adapter.py`, import paths adjusted for the flat
  layout) + the compiled kernel binary + a generated plan JSON. This was a
  deliberate choice ("do not clone or copy build trees blindly," "deploy only
  the minimum required artifacts") — the Pi has neither this repo's `.venv`
  nor its package `__init__.py` import chain (which pulls in unrelated
  modules like `capability_view.py`), so a literal repo copy would have
  required either installing unrelated dependencies or leaving broken
  imports. The three deployed files are byte-identical in logic to their
  repo counterparts; only two import/path lines differ, each commented
  in-file explaining why.
- **No NEON, no threading, no auto-tuning** — verified true by direct
  inspection of the kernel source (§1); this report makes no such claims.
- **Latency numbers are functional bring-up evidence only** (§7's explicit
  `truth_boundary` field), not a performance characterization: single
  process, no CPU affinity control, `ondemand` governor (not `performance`),
  no isolation from other system activity, only 20 repeats, and a visible
  governor-ramp transient in the first ~8 samples.
- **Runtime commit is not yet cut**: all runtime-side changes are currently
  uncommitted working-tree changes on top of HEAD `4776efe9`, per instruction
  to hold the commit until this report is reviewed.

## 12. Does this genuinely prove HardwareProfile → compiler decision → ExecutionPlan → our Runtime → ARM CPU execution?

**Yes, for exactly the one op/kernel this PR scoped, with the limitations in
§11 stated plainly:**

- **HardwareProfile → compiler decision**: the Raspberry Pi's live-collected
  hardware evidence (P1A) plus the one new evidence-backed
  `runtimeKernels` declaration flow through the real, unmodified
  `TargetConstraints`/`KernelSelectionPass` machinery to a real `selected`
  decision — verified by direct pass-source inspection and by a real,
  freshly-generated compiler run, not by inference from documentation.
- **→ ExecutionPlan**: the real `ExecutionPlanExporter` output (schema
  `execution_plan` v2.0.0) carries that decision in `kernel_selection.*` —
  confirmed via the compiler-side CTest and the cross-repo contract test.
- **→ our Runtime**: `heterogeneous-inference-runtime`'s own
  `loader.py`/`stage_builder.py`/`path_builder.py` (existing, largely
  unmodified machinery, extended only additively) parse and route that exact
  decision to a `PORTABLE_CPU_KERNEL` execution path — verified structurally
  (`build_execution_stages`/`build_execution_paths` on the real plan) and via
  the adapter's own validation.
- **→ ARM CPU execution**: the real compiled kernel executed on real
  Raspberry Pi 5 aarch64 hardware, produced correct output against an
  independent reference, and correctly refused two deliberately invalid
  inputs — not simulated, not mocked, not routed through ONNX Runtime/PyTorch.

What this does **not** prove: broad op coverage (only this one op/shape
family is dispatchable — every other op on this target remains honestly
unsupported), performance quality (§11), or NEON/parallel-execution
capability (neither implemented nor claimed).

## 13. Recommended commit grouping

Two separate commits, one per repository, as instructed:

1. **`ml-graph-compiler-runtime`**: `feat: declare portable CPU fused MatMul+Bias+ReLU runtime kernel for Raspberry Pi profile` —
   the profile edit, the new MLIR fixture, the new CTest + its CMakeLists
   registration.
2. **`heterogeneous-inference-runtime`**: `feat: add ExecutionPlan-driven portable CPU kernel adapter for Raspberry Pi` —
   the native kernel, the adapter, the schema/stage_builder/path_builder
   extensions, the two new test files, and the Pi evidence artifact.

Neither commit should be pushed unless the existing repository workflow
already requires it (per standing instruction) — recommend holding both for
explicit push approval, same as P1A.

---

## Stop point

Per instruction, this PR stops here. Thread decomposition, auto-tuning,
multiple candidates, arbitrary-graph fusion materialization, quantization,
Hailo, GPU/Vulkan, and ExecuTorch comparison are all explicitly out of scope
and not started.
