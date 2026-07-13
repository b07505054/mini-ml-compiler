DOCUMENT STATUS: HISTORICAL HOST-SCOPED AUDIT

Superseded for current project state by: CURRENT_STATE.md, ARCHITECTURE_STATUS.md, and ARCHITECTURE_PATHS.md.

Truth boundary: This document reflects a Mac-host YOLO-Seg audit before the current GPU Linux/P1D state. Mac-only statements such as no CUDA/Triton are not project-wide current facts.

# YOLO-Seg ExecutionPlan — Environment Identity and Runtime-Scope Verification Audit

Follow-up to `docs/YOLOSEG_EXECUTION_PLAN_RUNTIME_READINESS_AUDIT.md` (Phase 25).
Purpose: verify that the Phase 25 audit inspected the correct machine, repositories,
runtime implementation, target profile, and artifacts — and reassess its claims
against the **complete two-repo system** (compiler + `heterogeneous-inference-runtime`).

No code was modified. Nothing was committed. The empirical runtime trace in §4 was
run read-only against the existing artifact.

---

## 1. Environment Identity (verified 2026-07-11)

| Item | Value |
|---|---|
| Hostname | `AllendeMacBook-Pro.local` |
| OS | macOS 26.5.1 (build 25F80), Darwin 25.5.0 |
| Architecture | `arm64` — Apple M5 (Metal 4 support) |
| Working directory | `/Users/allen/Documents/Codex/project/systems-portfolio/ml-graph-compiler-runtime` |
| Compiler repo root / branch | same path, branch `master`, HEAD `7008c067` ("benchmark: add matmul postop benchmark harness", 2026-07-11 07:39 -0700) |
| Compiler repo remote | `origin git@github.com:b07505054/ml-graph-compiler-runtime.git` ✓ matches expected |
| Runtime repo root / branch | `…/systems-portfolio/heterogeneous-inference-runtime`, branch `main`, HEAD `4776efe9`, clean tree |
| Runtime repo remote | `origin git@github.com:b07505054/heterogeneous-inference-runtime.git` ✓ matches expected |
| Compiler repo `git status` | Dirty: modified CMakeLists/benchmark/test files, untracked Triton benchmark tools and the Phase 25 audit doc. None of the dirty files touch the serving passes, plan builder/exporter, or YOLO-Seg scripts. |
| Python | `python3` → pyenv shim, Python 3.11.11; repo `.venv/bin/python` also 3.11.11 |
| MLIR tools | `/opt/homebrew/opt/llvm/bin/{mlir-opt, mlir-translate, FileCheck}` (Homebrew LLVM) |
| CUDA | **Not available** (no `nvcc`, no `nvidia-smi`) |
| Triton | **Not importable** in `python3` |
| Metal | Available (Xcode toolchain `metal` found; Apple M5, Metal 4) |
| Core ML | Framework present (macOS system framework) |
| Relevant env vars | No `MLIR/LLVM/CUDA/METAL/TRITON`-prefixed variables set; no secrets encountered or printed |

Conclusion: one physical machine (Apple Silicon Mac), both repos present side by
side in the portfolio workspace, remotes matching the two GitHub URLs in question.

## 2. Artifact Freshness and Origin

`artifacts/yoloseg_generic_frontend/` is **gitignored** (`.gitignore:119`), so git
history cannot date the artifacts; mtimes and embedded provenance paths do.

| Artifact | Absolute path (under compiler repo) | Modified | Size |
|---|---|---|---|
| Execution plan | `artifacts/yoloseg_generic_frontend/yoloseg.execution_plan.json` | 2026-07-10 20:14:59 | 2,990,041 B |
| Generic MLIR | `artifacts/yoloseg_generic_frontend/yoloseg.generic.mlir` | 2026-07-10 20:14:59 | 395,057 B |
| CV-annotated MLIR | `artifacts/yoloseg_generic_frontend/yoloseg.cv_annotated.mlir` | 2026-07-10 20:14:59 | 191,034 B |
| Phase 25 audit doc | `docs/YOLOSEG_EXECUTION_PLAN_RUNTIME_READINESS_AUDIT.md` | 2026-07-11 22:11:36 | 17,273 B |

- All pipeline artifacts share the **same second** (one generation run on
  2026-07-10 evening); `yoloseg.cv_planning_facts.json` is from an earlier Phase 23
  run the same day (14:28); the bufferization artifacts are from a separate
  2026-07-11 07:34 run.
- Generating script: `scripts/run_yoloseg_execution_plan.sh` (which invokes
  `run_yoloseg_generic_mlir_emission.sh`, `run_yoloseg_cv_semantic_annotation.sh`,
  then `build-mlir/compile-for-target`).
- Target profile used: the script default
  `configs/target_profiles/apple_a17pro_mobile.json` — confirmed by the plan's
  `target_profile_id: "apple-a17pro-mobile"` and `plan_id`.
- Origin machine/worktree: every provenance path embedded in the reports
  (`frontend_report.json`, `cv_semantic_report.json`, planning facts) is
  `/Users/allen/Documents/Codex/project/systems-portfolio/ml-graph-compiler-runtime/…`
  — **this worktree on this machine**.
- Exact generating commit is not recoverable (artifacts untracked, no commit hash
  embedded), but the artifacts predate HEAD by ~11 hours and postdate the last
  commit touching the serving passes (`70e40cae`, "feat: add CV compiler planning
  passes"), which is consistent with generation at or near current HEAD.

**Verdict: the Phase 25 audit used current, locally generated artifacts — not
stale copies from another worktree or machine.**

## 3. Actual Runtime Owner

### 3a. What lives in `ml-graph-compiler-runtime`

| Component | Reality |
|---|---|
| Canonical ExecutionPlan schema/builder/exporter (`mlir_passes/…/ExecutionPlan*`) | Implemented — **producer only** |
| `kernel_selection_contract_v1` registry | Profile-declared `runtimeKernels`; exactly one entry (`metal_rmsnorm_f32_v1`) in `apple_a17pro_mobile.json` |
| Demo runtime (`src/runtime/`) | Custom toy `Graph`/`Node` IR with its own `op_registry.cpp` (MatMul/Add/ReLU/Attention/LayerNorm… CPU kernels in `src/kernels/cpu_kernels.cpp`), executors, schedulers. **Consumes its own IR, not the canonical `execution_plan.json`.** |
| `src/runtime/execution_plan_v2.cpp` | A *different*, internal step-list plan IR (`ExecutionPlanV2` with step ids/memory offsets) for the toy runtime — despite the name, unrelated to the canonical schema-2.0.0 JSON |
| Metal | `metal_backend.mm` (logs dispatch info, per CLAUDE.md not full graph execution) + `metal_rmsnorm_executor.mm` — one real, dispatch-validated Metal RMSNorm path consuming the RMSNorm-specific `trace/metal_rmsnorm_execution_plan.json` |
| Core ML | **No Core ML API call anywhere** — "coreml" exists only as profile/attr vocabulary |
| Canonical-plan runtime adapter / materializer | **None** |

### 3b. What lives in `heterogeneous-inference-runtime`

| Component | Reality |
|---|---|
| ExecutionPlan parser | `deployment/execution_plan/loader.py` — validates `schema: execution_plan`, `schema_version: 2.0.0`, rejects measured-field contamination. Implemented. |
| Runtime plan schema | `deployment/execution_plan/schema.py` (`ExecutionPlan`/`FunctionPlan`/`OpDecision`). **No `cv_extension` field — the CV section is silently dropped at parse.** |
| Stage builder | `stage_builder.py` — 1:1 function+per-op → `ExecutionStage`; op-kind vocabulary is `rmsnorm/matmul/attention/microbenchmark` (LLM-centric) |
| Path builder | `path_builder.py` — builds executable paths only for `vllm` (compiler-guided vLLM config) and `custom_cuda` (RMSNorm microbenchmark); everything else → `UNSUPPORTED` |
| Vocabulary router | `execution_unit_router.py` — maps `cuda/cuda_triton/cuda_cublas → vllm`, `cpu → pytorch_reference`, `coreml_ane/arm_compute → coreml`. **Neither `"coreml"` nor `"metal"` (the identifiers in the YOLO-Seg plan) exists in the map.** |
| `ExecutionEngine` / `BackendDispatcher` | `deployment/execution_engine.py`, `backend_dispatcher.py` — function-level decision pipeline producing frozen `RuntimeResult`; backend availability is an **injectable set, not hardware probing**; no kernel execution |
| Model-adapter registry | `deployment/model_adapter/registry.py` — registers exactly one adapter kind: `"mock"` |
| Backends | `backends/*.py`: whole-model wrappers (onnxruntime, pytorch, tensorrt, executorch, cpp) — not plan-driven |
| CV support | `deployment/onnx_cv_backend.py`: ONNX Runtime MobileNet-v2 demo (CPU EP). Not plan-driven, not YOLO, no CoreML EP wired |
| Core ML support | `deployment/coreml_edge_policy.py`: policy selection over *measured CoreML artifacts* — a JSON policy generator, no CoreML execution or kernels |
| Metal support | **None** (no `metal` reference in `deployment/`) |
| CUDA/Triton | Real kernels exist (`cuda_backend/kernels/{vector_add,matmul_naive,matmul_tiled}.cu`, `cuda_transformer_kernels/rmsnorm_kernel.cu`) but require a CUDA host (this machine has none) and are wired only to the RMSNorm microbenchmark path |
| Fallback behavior | `BackendDispatcher` walks primary → fallback chain → emergency CPU, as **decision objects only** |

Note: no classes literally named `RuntimeExecutionPlan` or `CompilerRuntimeAdapter`
exist in either repo; the closest real components are the ones named above.

### 3c. Capability ownership table

| Capability | Compiler repo | Heterogeneous runtime repo | Actual owner | Status |
|---|---|---|---|---|
| Canonical ExecutionPlan production | `ExecutionPlanBuilder/Exporter` | — | compiler | implemented |
| Canonical ExecutionPlan parsing | — (`ExecutionPlanV2` is a different IR) | `execution_plan/loader.py` + schema | runtime | implemented |
| `cv_extension` consumption | producer only | **absent from schema — dropped** | nobody | missing |
| Stage/path derivation from plan | — | `stage_builder.py` / `path_builder.py` | runtime | implemented (LLM vocabulary) |
| Backend dispatch decision | — | `BackendDispatcher` (injectable availability) | runtime | implemented as simulation/decision only |
| Executable plan-driven path | — | vLLM config materializer (LLM); custom-CUDA RMSNorm microbenchmark | runtime | implemented for LLM/RMSNorm only |
| Plan-addressable kernel registry | profile `runtimeKernels` (1 kernel) | — | compiler (declared) | metadata (1 dispatch-validated kernel behind it) |
| CPU kernels | toy-IR `op_registry` (MatMul/ReLU/…) | pytorch/onnxruntime whole-model backends | both, different layers | implemented, **not plan-addressable** |
| Metal kernels | RMSNorm f32 executor (real dispatch) | none | compiler repo (demo) | implemented (1 kernel, RMSNorm only) |
| Core ML execution | none | none (policy JSON only) | nobody | metadata only |
| CUDA/Triton kernels | none (no CUDA host) | rmsnorm/matmul/vector_add `.cu` | runtime repo | implemented, CUDA-host-gated, not usable here |
| CV (YOLO) runtime adapter/materializer | none | none (`yolo` appears nowhere) | nobody | missing |

## 4. Tracing the Generated YOLO-Seg Plan into the Actual Runtime

Executed read-only on this machine against
`artifacts/yoloseg_generic_frontend/yoloseg.execution_plan.json`:

```
load_execution_plan: OK                      (schema 2.0.0 accepted)
plan_id: apple-a17pro-mobile_serving_plan
has cv_extension attr on schema object: False   (section silently dropped)
function_plans: ['main_graph']                  (recognized as a FunctionPlan)
selected_backend: coreml
stages: 930   kinds: {MICROBENCHMARK: 929, OTHER: 1}
paths:  930   kinds: {UNSUPPORTED: 930}
              reason: unknown_execution_unit  (x930)
```

Findings:

- **Parses**: yes — the loader validates and accepts the plan.
- **`cv_extension`**: not recognized; not a field of the runtime schema; dropped.
- **`main_graph`**: recognized as a function plan (name is irrelevant to routing).
- **Backend identifiers**: `"coreml"` is **unknown vocabulary** to
  `ExecutionUnitRouter` (only `coreml_ane` maps, and its `coreml` adapter is
  explicitly `adapter_not_implemented` in `path_builder.py:83`); `"metal"` does
  not exist in the router at all; `"cpu"` maps to `pytorch_reference`, also
  `adapter_not_implemented`.
- **Expected granularity**: the runtime consumes function-level backend decisions
  plus per-op decisions it treats as microbenchmark stages; it has **no concept
  of dispatch units, regions, or source-level ops** — and every lowered
  `linalg/tensor/arith` op-type falls into the catch-all `MICROBENCHMARK` kind.
- **CV kernel/operator materializer**: none (`yolo` appears nowhere in the repo;
  the only CV backend is a MobileNet ONNX Runtime demo, not plan-driven).
- **I/O binding**: nothing in the runtime reads the plan's inputs/outputs at all
  (that data lives only in the dropped `cv_extension`), so image/weight/output
  binding is doubly impossible: not serialized usefully (Phase 25 §7) *and* not
  parsed.
- **Bottom line**: the runtime can *parse* the plan today and produces **930/930
  UNSUPPORTED paths, zero executable**.

## 5. Target-Profile Correctness

- `scripts/run_yoloseg_execution_plan.sh` uses
  `TARGET_PROFILE="${YOLOSEG_TARGET_PROFILE:-…/configs/target_profiles/apple_a17pro_mobile.json}"`.
- Profile: `profileId: apple-a17pro-mobile`, `configuredComputeUnits: "CPU+GPU+ANE"`.
- Backend order: `compile-for-target/main.cpp:244-246` hardcodes
  `"CPU+GPU+ANE"` → preferred `coreml`, allowed `{coreml, metal, cpu}` — hence
  Core ML primary with Metal/CPU fallbacks. This is declared policy, not
  capability matching (unchanged from Phase 25 §4).
- **Intentional Apple target**: yes — the default is documented in
  `docs/REAL_YOLOSEG_EXECUTION_PLAN.md` (lines 137–140), and the development
  machine is an Apple Silicon Mac with no CUDA and no Triton, so no GPU/CUDA plan
  could have been generated or intended here.
- Other profiles exist (`nvidia_gtx1650_maxq.json`, `nvidia_gtx1650_maxq_awq_forced.json`,
  `nvidia_cuda_tensorcore.json`, `amd_instinct_datacenter_gpu.json`, …) but belong
  to the Qwen/LLM and benchmark paths. No capability from any NVIDIA profile was
  attributed to the Apple plan in the Phase 25 audit (verified: all cited fields —
  `runtimeKernels`, `backendCapabilities`, `configuredComputeUnits` — came from
  `apple_a17pro_mobile.json`).
- **The Phase 25 audit did not accidentally evaluate the wrong profile.**

## 6. Reassessment of Phase 25 Claims

| # | Claim | Classification | Evidence |
|---|---|---|---|
| 1 | "Only one real runtime kernel exists" | **Incomplete** (correct for the plan-addressable registry; wrong if read as a two-repo kernel inventory) | Correct scope: `kernel_selection_contract_v1` registry = profile `runtimeKernels` = 1 entry (`metal_rmsnorm_f32_v1`). But the system contains more real kernels: compiler-repo toy-runtime CPU kernels (`src/runtime/op_registry.cpp`, `src/kernels/cpu_kernels.cpp`) and runtime-repo CUDA kernels (`cuda_transformer_kernels/rmsnorm_kernel.cu`, `cuda_backend/kernels/*.cu`). None are plan-addressable and none cover YOLO ops, so the *consequence* for YOLO-Seg is unchanged. |
| 2 | "Zero YOLO-Seg kernel decisions map to real runtime implementations" | **Correct for the complete two-repo system** | All 929 `kernel_selection.status = rejected_no_kernel_for_op`; empirical trace: 930/930 paths `UNSUPPORTED`; no CV/linalg kernel exists in either repo. |
| 3 | "Metal/CPU fallbacks are metadata only" | **Correct for the complete two-repo system** | `"metal"` is absent from `ExecutionUnitRouter`'s vocabulary; `"cpu"` routes to `pytorch_reference` = `adapter_not_implemented`; `BackendDispatcher` emits decision objects with injectable availability, never executes. The compiler repo's real Metal RMSNorm executor covers no YOLO op. |
| 4 | "Core ML selection is not actionable" | **Correct for the complete two-repo system** (and strengthened) | Compiler side: hardcoded `configuredComputeUnits` policy, no capability match (Phase 25 §4). Runtime side: literal `"coreml"` is unknown vocabulary (`unknown_execution_unit`); even the mapped `coreml_ane` adapter is `adapter_not_implemented`; no Core ML API call exists in either repo. |
| 5 | "The runtime cannot consume the plan" (Phase 25: per-op layer not consumable as dispatch units) | **Incomplete — needs refinement** | The runtime **can parse** the plan (empirically verified) and recognizes `main_graph`. It cannot produce any executable path (930/930 unsupported), drops `cv_extension` entirely, and mislabels every CV op as `MICROBENCHMARK`. "Parseable but not executable, CV contract invisible" is the precise statement. |
| 6 | "Dispatch-unit materialization is the next required phase" | **Correct but incomplete as the *only* next phase** | The granularity problem is real and confirmed (929 positional bundles → 929 meaningless stages). But this verification exposes a co-equal blocker the Phase 25 audit under-weighted: the **compiler↔runtime vocabulary/contract gap** (backend identifiers `coreml`/`metal` vs router's `coreml_ane`; `cv_extension` absent from the runtime schema; no CV adapter registered). Materializing dispatch units alone would still yield 100% `UNSUPPORTED` paths. |

### Was the wrong repository scope inspected?

Partially. The Phase 25 audit *did* open the runtime repo's plan loader, schema,
and stage builder (its §1 cites `stage_builder.py`), so it was not blind to the
second repo — and its YOLO-Seg-specific conclusions all survive. But it did not
audit the runtime repo's execution layer (`ExecutionUnitRouter`, `path_builder`
outcomes, adapter registry, `ExecutionEngine`/`BackendDispatcher`, CUDA kernels),
which is why claims 1 and 5 were stated at compiler scope without the runtime-side
evidence, and why the vocabulary-contract blocker went unlisted in the minimum
fixes.

## 7. Summary

- **Host/machine**: `AllendeMacBook-Pro.local`, macOS 26.5.1, Apple M5 (arm64);
  Metal + Core ML present, CUDA and Triton absent.
- **Repos/branches**: `ml-graph-compiler-runtime` @ `master`/`7008c067`
  (dirty in unrelated benchmark/test files) and `heterogeneous-inference-runtime`
  @ `main`/`4776efe9` (clean); remotes match the expected GitHub URLs.
- **Artifacts**: current — generated 2026-07-10/11 on this machine in this
  worktree by `scripts/run_yoloseg_execution_plan.sh`; not stale, not foreign.
- **Repository scope of the prior audit**: partially compiler-scoped; the runtime
  repo's execution layer had not been traced. All prior YOLO-Seg conclusions
  survive the two-repo trace; two claims (kernel inventory, plan consumability)
  are refined above.
- **Target profile**: `apple_a17pro_mobile.json` — the intended, documented Apple
  target; no NVIDIA/CUDA profile was mixed in, and none could have run here.
- **Real runtime/kernel registry locations**: plan-addressable registry =
  target-profile `runtimeKernels` (compiler repo, 1 Metal RMSNorm kernel);
  plan-consuming runtime = `heterogeneous-inference-runtime/deployment/`
  (loader/stage/path/engine — executable only for the LLM vLLM path and a CUDA
  RMSNorm microbenchmark); additional non-plan-addressable kernels exist in both
  repos (toy-IR CPU kernels; CUDA `.cu` kernels needing a CUDA host).
- **Prior conclusions still valid**: claims 2, 3, 4 fully; claim 6's direction;
  claims 1 and 5 valid for YOLO-Seg after scope refinement.
- **Corrected runtime-readiness verdict**: still **`NEEDS_DISPATCH_UNIT_MATERIALIZATION`**
  on the compiler side — now with the empirically verified addendum that the
  runtime side is equally blocking: today the plan parses but yields 930/930
  `UNSUPPORTED` paths and the CV contract (`cv_extension`) is not even parsed.
- **Corrected next coding phase**: a two-sided contract phase, not compiler-only:
  1. compiler — provenance-preserving **dispatch-unit materialization** plus the
     Phase 25 schema fixes (ABI roles, memory fields, CV contract);
  2. runtime — extend the ExecutionPlan schema/loader to parse `cv_extension`
     and dispatch units, align the backend-identifier vocabulary
     (`coreml`/`metal` ↔ router), and register a real (initially CPU/ONNX-Runtime)
     CV adapter so at least one executable path exists end to end.

All statements above are static/code-level findings plus one read-only parse
trace; no runtime performance or execution claims are made.
