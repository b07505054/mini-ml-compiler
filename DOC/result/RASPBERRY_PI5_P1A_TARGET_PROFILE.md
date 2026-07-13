DOCUMENT STATUS: HISTORICAL

Superseded by: CURRENT_STATE.md and PROJECT_STATUS_2026.md.

Truth boundary: This report reflects the P1A Raspberry Pi target-profile phase and older compiler/runtime commit heads. It remains evidence/history and must not be read as the current repository state.

# Phase P1A — Raspberry Pi 5 CPU Target Profile

**Scope (as instructed): compiler-only.** No runtime changes, no `ExecutionPlan`
schema changes, no ARM kernels, no Hailo support, no compiler-logic changes.
This PR adds exactly one new target-profile JSON file and verifies the
existing, unmodified compiler pipeline accepts it end-to-end.

Repository: `ml-graph-compiler-runtime`, remote host `100.87.220.5`,
`~/Desktop/Project/ml-graph-compiler-runtime` (branch `master`, HEAD `a7776f5b`
at time of work).

---

## 1. New Raspberry Pi target profile

`configs/target_profiles/raspberry_pi5_cortex_a76_cpu.json` (new file):

```json
{
  "profileId": "raspberry-pi5-cortex-a76-cpu",
  "configuredComputeUnits": "CPU",
  "staticShapeSupport": true,
  "supportedPrecisions": ["fp32", "fp16"],
  "truthBoundary": "raspberry_pi5_live_hardware_evidence_cpu_only_no_verified_gpu_or_npu_kernel_library_declared_not_measured_performance",
  "hardwareExecutionProfile": {
    "physicalComputeUnits": 4
  }
}
```

Sole evidence source: `raspberry_pi_hardware_profile_raw.json`
(`evidence_type: "live_queried"`, collected `2026-07-13T04:30:20Z` from
`ssh allen@100.110.37.6`), re-read fresh from disk in this session — not
recalled from memory.

### Why `"CPU"` and not `"CPU+GPU-ARM"`

The only existing ARM-flavored profile, `arm_compute_cpu_gpu.json`, maps to
`configuredComputeUnits = "CPU+GPU-ARM"` → `preferred_backend = "arm_compute"`
in `main.cpp`'s hardcoded mapping table. That profile's own
`backendCapabilities[].backendName` values are `"cpu"`/`"gpu"` — a name
mismatch with the driver's `"arm_compute"` label that means
`KernelAvailabilityPlanningPass`'s lookup (`target.kernel_libraries.<backend>`)
can never match anything for it, and it is referenced by zero
tests/CMake/tools anywhere in the repo (confirmed dead code). Reusing that
pattern for the Pi would inherit a known-broken mapping. `"CPU"` is the one
compute-unit value confirmed to resolve correctly
(`preferred_backend = "cpu"`, `allowed_backends = ["cpu"]`) with zero code
changes, and it is the only compute path this hardware evidence actually
supports: the VideoCore VII/V3D GPU's Vulkan capability is package-level only
(`mesa-vulkan-drivers`, `libvulkan1` installed) with no device-level execution
verified, and the Hailo-8 NPU is physically present on the PCIe bus
(`pcie` evidence) but `hailo_identify`/`hailo_scan` both returned empty
strings — no functional driver/runtime evidence.

### Per-field provenance

| Field | Value | Evidence source | Confidence | Consuming stage | Why |
|---|---|---|---|---|---|
| `profileId` | `raspberry-pi5-cortex-a76-cpu` | `board.model` ("Raspberry Pi 5 Model B Rev 1.1"), `cpu.lscpu` ("Model name: Cortex-A76") | Directly observed | Everywhere (plan_id, provenance.capability_bundle.hardware_profile_ref) | Only field the parser hard-requires (`parseDeviceProfile` returns a `StringError` if absent) |
| `configuredComputeUnits` | `"CPU"` | `cpu.lscpu` (4× Cortex-A76, single cluster, no SMT); GPU/NPU excluded per rationale above | Directly observed (CPU) + reasoned exclusion (GPU/NPU) | `lowerToTargetConstraints` → `preferred_backend`/`allowed_backends` | Drives backend selection for the whole compile |
| `staticShapeSupport` | `true` | Not a hardware fact — matches the default and every existing profile's value | N/A (convention) | `ReplayEligibilityPass` only, and only for functions containing `llm.attention_prefill`/`llm.attention_decode` | Confirmed via `ReplayEligibilityPass.cpp:56-61` to be completely inert for this PR's CV verification run (YOLO-Seg has no attention ops) |
| `supportedPrecisions` | `["fp32", "fp16"]` | `cpu.features`/`lscpu` Flags: `fp asimd ... fphp asimdhp` (NEON fp32 baseline + fp16 storage/arithmetic) | Directly observed (OS/ISA-reported) | `RepresentationPlanningPass`/`QuantizationStrategyPlanningPass` dtype legality | `asimddp` (INT8 dot-product) is also present in the ISA flags but is **deliberately excluded** — no verified ML kernel library on this box (no Arm Compute Library; only reference BLAS/LAPACK + package-installed onnxruntime/tflite, neither verified at compile time to expose an INT8 execution path) actually dispatches it. Declaring `"int8"` here would assert target-wide legality the evidence does not support. |
| `truthBoundary` | (see JSON) | Descriptive of the profile's own evidence class | N/A | `bundle.hardware.truth_boundary` / `bundle.deployment.truth_boundary` (provenance/export labeling only, not a decision input) | Follows existing convention of long, explicit truth-boundary strings; distinguishes this as *live-queried* evidence, unlike the `public_docs`-sourced GPU/server profiles |
| `hardwareExecutionProfile.physicalComputeUnits` | `4` | `cpu.lscpu` ("CPU(s): 4"), `cpu.logical_cpu_count: 4`, `Thread(s) per core: 1` (confirms no SMT, so physical == logical) | Directly observed | **None today** — confirmed via `grep -rn "hardware_execution_profile\|physical_compute_units" lib/ include/`: this block is round-tripped by `TargetConstraints.cpp` (parsed → attached as `target.hardware.physical_compute_units` module attr → read back into the struct) and exercised only by `TargetConstraintsTest.cpp`. **No serving pass reads it for any decision.** | Declared anyway because it is directly evidenced, harmless, and future passes may consume it; documented here so this is not mistaken for a working parallelism-aware decision today |

### Fields deliberately omitted (not fabricated, not silently forgotten)

| Field / block | Why omitted | Effect of omission (verified, not assumed) |
|---|---|---|
| `staticCostProfile` (peak FLOPs fp32/fp16/int8, memory bandwidth, local memory bytes, cache line, async-copy/DMA) | **Explicitly prohibited by the task instruction** ("Do NOT invent: peak FLOPs, memory bandwidth"). No public spec or measurement exists for this exact SoC's memory subsystem beyond raw cache *sizes* (which are a different, non-fungible capability — see below) | `ServingCostModelPass` falls back to the fixed V1 penalty model instead of roofline time estimates; `TilePlanningPass` stamps every matmul-like op `tile.plan.status = "deferred_missing_memory_hierarchy"` — verified: 76/76 `linalg.conv_2d_nchw_fchw` ops in the YOLO-Seg graph got this status |
| `staticCostProfile.localMemoryBytes` specifically, populated from the real L1/L2/L3 cache sizes we *do* have (64K/512K/2M) | The schema's own semantics (per `synthetic_gpu_8cu.json`, `localMemoryKind: "software_managed_shared_memory"`) model this as software-managed scratchpad/shared memory reachable via explicit DMA/async-copy — a Cortex-A76's hardware-managed cache hierarchy (no scratchpad, no explicit DMA into L1/L2) is a materially different capability. Declaring a cache size under this field would overstate what the hardware actually offers. | Same as above — `tile_planning_v1` stays deferred, honestly |
| `backendCapabilities[]`, `kernelLibraries[]`, `runtimeKernels[]` | No verified ML kernel library exists for this board: no Arm Compute Library, only reference BLAS/LAPACK (`libblas3`, `liblapack3`) plus package-installed `libonnxruntime1.21`/`libtensorflow-lite2.20.0` — real installed software, but **not** verified at compile time to expose any specific (op, dtype, layout) capability matrix. Inventing a plausible `supportedOps`/`supportedDtypes` list would be fabrication. | `KernelSelectionPass`: `deferred_no_kernel_library_declared` on **929/929** ops. `LoweringDecisionPlanningPass`: `kernel.lowering_path = "unsupported"`, `kernel_exists: false`, `reason: "no_kernel_no_rewrite_no_fallback"` on **846/846** ops that reach lowering decisions (929 total minus 83 `arith.constant` ops, which these passes skip) |
| `pagedKVCompatibleBackends`, `prefillMsPerToken`/`decodeMsPerToken`/`pdBandwidthMbPerMs`, `metalMaxWorkingSetMB`, `quantizationCoDesignPolicy`, `forcedQuantization` | LLM-serving-specific or Apple/Metal-specific fields with zero applicability to a CPU CV bring-up profile; no evidence and no need | All-absent defaults apply (empty paged-KV backend list, no formula-calibration override, co-design pass stays inert, no forced quant) |
| `hardwareExecutionProfile.effectiveComputeUnits` / `maxConcurrentWorkItemsPerUnit` / `supportsLatencyHiding` / `localMemoryKind` | Same "zero pass consumer today" finding as `physicalComputeUnits`, **plus** these specific sub-fields carry GPU-style occupancy-modeling semantics (`software_managed_shared_memory`, latency-hiding via oversubscription) that don't map cleanly onto a general-purpose CPU core without asserting an execution model not in evidence | No effect either way (unconsumed); omitted for honesty, not caution about breakage |

---

## 2. Verification log

```
$ cmake --build build-mlir --target compile-for-target -j4
...
[100%] Built target compile-for-target      # exit 0, full rebuild, zero errors

$ ./build-mlir/compile-for-target \
    --device-profile configs/target_profiles/raspberry_pi5_cortex_a76_cpu.json \
    --mlir artifacts/yoloseg_generic_frontend/yoloseg.cv_annotated.mlir \
    --out trace/raspberry_pi5_p1a/execution_plan.json \
    --dump-annotated-mlir trace/raspberry_pi5_p1a/annotated.mlir \
    --dispatch-unit-report trace/raspberry_pi5_p1a/dispatch_unit_report.json

compile-for-target: raspberry-pi5-cortex-a76-cpu → yoloseg
  canonical:      trace/raspberry_pi5_p1a/execution_plan.json
  function plans: 1

  main_graph:
    serving_phase:    unknown
    selected_backend: cpu
    decision_source:  cv-target-profile-static-policy
  global.serving:
    topology:         cv_full_graph
    replay_eligible:  false

EXIT CODE: 0
```

Rerun with stdout/stderr captured separately to check for warnings:
`stderr` byte count = **0**. No warnings, no diagnostics.

`execution_plan.json` re-parsed successfully with `json.load` (valid JSON,
schema fields present: `cv_extension`, `function_plans`, `global_decisions`,
`model_identity`, `plan_id`, `provenance`, `schema`, `schema_version`).

Input graph: 929 ops (`tensor.empty` ×326, `linalg.generic` ×226,
`arith.constant` ×83, `linalg.fill` ×81, `linalg.conv_2d_nchw_fchw` ×76,
`tensor.insert_slice` ×52, `tensor.pad` ×50, `tensor.extract_slice` ×18,
`tensor.collapse_shape` ×10, `linalg.pooling_nchw_max` ×3,
`tensor.generate` ×2, `tensor.expand_shape` ×1, `linalg.transpose` ×1).

## 3. Every compiler stage reached

`HardwareProfile parse → TargetConstraints lowering → CapabilityBundle
lowering → module attribute attach` all completed (confirmed: `profile_id`,
`static_shape_support`, `preferred_backend=cpu`/`allowed_backends=[cpu]`,
`supported_precisions=[fp32,fp16]`, `hardware.physical_compute_units=4`
correctly threaded through — verified via `provenance.capability_bundle.
hardware_profile_ref = "raspberry-pi5-cortex-a76-cpu"` in the exported plan).

All 16 serving passes + `BoundaryMaterializationPass` ran to completion with
`pm.run(...)` returning success (a `.failed()` return would have produced
`"error: serving pass pipeline failed"` and exit 1 — did not occur):

`ServingPhaseAnalysis → KVLayoutPlanning → ReplayEligibility →
ExecutionProviderPlanning → RepresentationPlanning → LayoutPlanning →
BoundaryPlanning → WeightClassificationPlanning →
QuantizationStrategyPlanning → CVExecutionPlanAttrs →
KernelAvailabilityPlanning → LoweringDecisionPlanning →
QuantizedBoundaryRefinement → TilePlanning → KernelSelection →
QuantizationCoDesign → AlternativeLoweringPlanning →
CandidateGeneration → CandidateEvaluation → PlanSelection →
BoundaryMaterialization`

Then `ExecutionPlanBuilder::build` → `ExecutionPlanExporter::exportToFile`
succeeded (canonical `execution_plan.json` written, valid JSON, non-empty).

**Capability Checking / Candidate Enumeration / Cost Model / Decision**,
concretely, per-op:

- **Capability checking**: `KernelAvailabilityPlanningPass` /
  `LoweringDecisionPlanningPass` found no declared backend/kernel capability
  for the `cpu` backend on 846/846 real (non-constant) ops →
  `kernel.lowering_path = "unsupported"`, `kernel_exists: false`.
- **Candidate enumeration / decision**: with no viable kernel candidate,
  `CandidateGenerationPass`/`CandidateEvaluationPass`/`PlanSelectionPass`
  select the single `unsupported` outcome (`evidence.cost.total_cost = 100`,
  `unsupported_penalty = 100`) — a real, non-fabricated static-penalty
  decision, not a crash or silent skip.
- **Cost model**: `shape_cost` is present on the 76 `conv_2d` ops with
  `status: "facts_only_no_profile_numbers"` — real static FLOPs/byte facts
  computed from the actual tensor shapes (e.g. one op:
  `flops_estimate: 2103705600`, `total_memory_bytes_estimate: 12321328`),
  but **no roofline time estimate**, because `staticCostProfile` peak
  numbers were intentionally not declared.
- **Layout**: `selected_layout = "nchw"`, `requires_layout_transform: false`
  on all 846 ops — derived purely from the upstream MLIR tensor contract,
  independent of the target profile.
- **Quantization**: `strategy: "none"` on all 846 ops
  (`reason: "cv_phase24_no_quantization_configured"`) — no quantization
  attempted, correctly, since no quant policy or capability was declared.

## 4. Compiler warnings

**None.** `stderr` was empty across both the instrumented run and the
plain rerun.

## 5. Deferred decisions (intentional, due to absent evidence — not fabricated)

| Decision | Status | Count | Deferred reason |
|---|---|---|---|
| `kernel_selection` (kernel_selection_contract_v1) | `deferred_no_kernel_library_declared` | 929 / 929 ops | No `runtimeKernels` declared in the profile |
| `tile_plan` (tile_planning_v1) | `deferred_missing_memory_hierarchy` | 76 / 76 `conv_2d` ops (the only tile-plannable op kind present) | No `staticCostProfile.localMemoryBytes` declared |
| `shape_cost` roofline time estimate | `facts_only_no_profile_numbers` | 76 / 76 `conv_2d` ops | No `staticCostProfile` peak FLOPs / memory bandwidth declared |

No stage was worked around or given a substitute/fabricated number to make
it "activate" — each one reports its own honest deferral reason, exactly as
designed.

## 6. Unsupported features

`kernel.lowering_path = "unsupported"` on **846 / 846** real (non-constant)
ops — i.e. **100% of this graph's operators are currently marked
unsupported** on this profile. This is the direct, correct consequence of
declaring zero `backendCapabilities`/`kernelLibraries`/`runtimeKernels` for
the `cpu` backend (per the task's explicit "never invent capability"
instruction) — it is not a bug, a pipeline failure, or evidence of a broken
profile. It means: **this PR proves the profile is structurally accepted and
threads correctly through every stage; it does not yet claim any operator is
executable on real Raspberry Pi CPU hardware.** Declaring real, verified CPU
kernel capability (e.g. from a measured ONNX Runtime CPU execution provider
run) is future work, out of this PR's scope.

## 7. Exact files changed

- `configs/target_profiles/raspberry_pi5_cortex_a76_cpu.json` — **new file**, the target profile (7 lines of substantive JSON, shown in §1).
- `DOC/result/RASPBERRY_PI5_P1A_TARGET_PROFILE.md` — **new file**, this report.
- No other source, test, build, or config file was modified. No `ExecutionPlan` schema change. No runtime change. No ARM kernel added. No Hailo support added. No compiler pass logic changed.
- `trace/raspberry_pi5_p1a/` (execution_plan.json, annotated.mlir, dispatch_unit_report.json) — generated verification artifacts, gitignored (matches existing `trace/` convention), not part of the committed diff.

## 8. Tests executed

- **Build**: `cmake --build build-mlir --target compile-for-target -j4` — full rebuild (stale `build-mlir` cache triggered a from-scratch build of all 30 serving-pass translation units), **0 errors, 0 warnings**, binary linked successfully.
- **Manual pipeline invocation** (not a registered CTest): `compile-for-target` run twice against the new profile + the existing, unmodified, proven `artifacts/yoloseg_generic_frontend/yoloseg.cv_annotated.mlir` input — exit 0 both times, valid JSON output, zero stderr.
- **JSON schema sanity**: `python3 -c "import json; json.load(...)"` on both the profile file and the exported plan — both parse cleanly.
- **Not run / not applicable**: `tools/validate_compiler_artifacts.py` targets the separate legacy `cv_execution_plan_v2.json`/`cv_cost_report.json`-family CV artifacts, not the serving `ExecutionPlan` schema this driver produces — inspected and confirmed not applicable, not run against irrelevant input.
- **No new automated test was added** (e.g. no `NvidiaProfileLoweringTest`-style CTest registered for this profile) — deliberately out of scope per the task's explicit instruction to touch only the profile file; this is manual, one-shot verification only, and should be read as such.

## 9. Build status

**SUCCESS.** Clean full rebuild of `compile-for-target`, 100% of targets built, zero compiler errors or warnings.

---

## Stop point

Per instruction, this PR stops here. Runtime work (`heterogeneous-inference-runtime`), declaring real backend/kernel capability for the Pi CPU, and populating `staticCostProfile` from an actual measured benchmark are all explicitly out of scope for this phase and are not started.
