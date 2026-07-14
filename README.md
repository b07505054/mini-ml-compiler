# Edge AI Implementation-Decision Compiler

This project is an IR-centered, hardware-aware, evidence-driven implementation-decision compiler for Edge AI backends.

It is not a replacement for every runtime or kernel library. It makes implementation choices explicit, IR-rooted, feasibility-checked, evidence-backed where evidence exists, provenance-tracked, and materialized into an exact runtime contract.

## Why This Exists

Edge AI deployment is usually a chain of implicit choices: backend, kernel, tile, thread schedule, precision, artifact, runtime flags, and fallback behavior. This compiler makes those choices inspectable.

```text
Model / GenericGraphIR
  -> Semantic IR
  -> Program Analysis
  -> Candidate Providers
  -> complete ImplementationCandidates
  -> Feasibility
  -> Evidence-backed Policy
  -> PolicyResult
  -> Implementation IR or Execution Contract
  -> Runtime validates
  -> Runtime executes exactly
  -> Real measurement
  -> Offline calibration
```

Core constitution: **Compiler chooses. Runtime validates. Runtime executes.**

Runtime must not search backends, kernels, tiles, thread schedules, precision, benchmark online, silently fallback, or silently rewrite compiler decisions.

## Production / Verified Now

- Raspberry Pi 5 portable CPU path for FP32 `Y = ReLU(A @ B + bias)`.
- `PortableCPUProvider`, typed feasibility, policy, `PolicyResult`, and selected-candidate materialization for the fixed portable kernel path.
- Current portable kernel: `portable_fused_matmul_bias_relu_bm32_bn128_bk32`.
- Current portable policy: `M*N*K < 262144` selects serial; `M*N*K >= 262144` selects 4-thread split-M.
- XNNPACK evaluation path with live Compiler invocation, X1/X4 candidates, typed feasibility, compiler-generated comparison contract, and same-runner/same-PTE execution.
- Triton shadow provider over real measured/predicted artifacts, with unresolved IR bridge, no production ExecutionPlan effect.
- AWQ artifact and vLLM materialization path, with measured serving traces but no accuracy/perplexity calibration.

## Key Measured Results

| Result | Scope | Value | Evidence |
|---|---|---:|---|
| P1D.1 exact match | Pi portable CPU held-out rows | 86.67% | runtime `results/p1d_raspberry_pi_thread_decomposition/p1d_raw_measurements.json` |
| P1D.1 mean regret | same | 0.067392% | recomputed from raw rows at threshold 262144 |
| P1D.1 P95 / max regret | same | 0.489076% / 0.768578% | same |
| P1D.1 avg speedup over serial | same | 3.327722x | same |
| E2.1 correctness | historical implementation-stack comparison | 324/324 records, 0 failures | runtime `results/executorch_e2_1/e21_analysis.json` |
| E2.1 stack comparison | project portable scalar/native stack vs ExecuTorch/XNNPACK | project geomean speedup 0.380026x, about 2.631x slower | same |
| E3 candidate-space verdict | same XNNPACK stack | `XNNPACK_ONE_STATIC_WINNER`, static X1 | runtime `results/executorch_e3/discovery/e3_analysis.json` |
| E3 formal comparison | live Compiler XNNPACK contract vs ExecuTorch default | 2 wins, 8 ties, 0 losses; geomean default/project ratio 1.031686x | runtime `results/executorch_e3/formal/e3_formal_analysis.json` |

These are narrow, target-scoped results. They are not cross-model, cross-device, energy, NPU, or universal superiority claims.

## ExecuTorch Comparison Classification

E2 is historical and invalid: the frozen correctness predicate used independent absolute and relative gates, and the relative gate rejected near-zero outputs despite small absolute errors.

E2.1 repaired correctness but compared different implementation stacks. The project side bypassed the live Compiler, hardcoded the threshold and portable kernel ID in Python, and ran the project scalar/native C++ kernel. ExecuTorch used `.pte`, XNNPACK, and a warm pthreadpool. E2.1 is therefore an `IMPLEMENTATION_STACK_COMPARISON`, not a compiler-only comparison.

E3 repaired the chain: Semantic IR -> live Compiler -> XNNPACK candidates -> feasibility -> policy -> compiler-generated contract -> common ExecuTorch/XNNPACK runner. E3 compared live-Compiler-selected XNNPACK X1 against ExecuTorch default using the same Pi, same `.pte`, same runner, same ExecuTorch commit, same XNNPACK commit, same input bytes, same timing boundary, same process-lifetime class, and same oracle.

E3 found a static XNNPACK winner for this narrow target/workload scope. The Compiler contribution was candidate exposure, feasibility/provenance validation, calibration, low-regret static selection, contract generation, and exact same-stack execution.

## Four Pillars

| Pillar | Current maturity | Summary |
|---|---|---|
| Hardware Abstraction | `ADVANCED_PARTIAL` | Pi 5 profile, compute units, thread capability, kernel/runtime descriptors, and XNNPACK software/artifact requirements exist; memory hierarchy, DMA, NPU, bandwidth, and transfer models remain incomplete. |
| Decision-making Compiler | `STRONGEST_PILLAR / ADVANCED_PARTIAL` | Semantic IR, complete candidates, providers, feasibility, policy, `PolicyResult`, materialization, P1D.1 and E3 calibrated loops. Not all decisions share one universal policy engine. |
| Quantization Co-design | `EARLY_PARTIAL` | Real AWQ artifact and vLLM materialization exist. Missing accuracy/perplexity calibration, unified FP16/INT8/INT4 candidates, canonical feasibility/policy, and NPU quantization. |
| Hardware-Compiler Co-design | `EARLY_PARTIAL` | Pi measurements influence portable policy; Pi/XNNPACK measurements influence X1. No hardware parameter sweep, SRAM/bandwidth/DMA/NPU design feedback yet. |

## Repository Map

- `ml-graph-compiler-runtime`: canonical compiler architecture, IR, candidates, feasibility, policy, ExecutionPlan/contract generation, phase reports.
- `heterogeneous-inference-runtime`: runtime validation/dispatch, evaluation harnesses, raw evidence, ExecuTorch E1/E2/E2.1/E3 results.
- `ml-platform-capabilities`: intended home for declared capability facts; partial today and not the sole source of truth.
- `Inference-Validation-Platform`: divergent local validation/control-plane project; not the canonical source repository host.
- Raspberry Pi: real execution/evidence target only.

## Current Limitations

- Triton is shadow-only and not production-integrated.
- Quantization co-design is not complete; no accuracy/perplexity calibration exists.
- Many decisions remain declared-profile, rule-based, shadow, or experimental rather than measured-profile-driven.
- Capability profiles are not yet fully synchronized across repositories.
- Implementation IR is incomplete for memory spaces, DMA, synchronization, heterogeneous partitioning, and NPU command regions.
- No general superiority claim over ExecuTorch, XNNPACK, vLLM, TVM, ONNX Runtime, TensorRT, or any device family is made.

## Reproduction Pointers

- P1D.1 policy artifact: `configs/thread_schedule_policies/raspberry_pi5_cortex_a76_p1d1_thread_policy.json`.
- P1D.1 raw evidence: sibling runtime `results/p1d_raspberry_pi_thread_decomposition/p1d_raw_measurements.json`.
- E3 compiler contract generator: `tools/e3_xnnpack_contract.py`.
- E3 runtime harness/evidence: sibling runtime `evaluation/executorch_e3/` and `results/executorch_e3/`.
- Canonical docs: `ARCHITECTURE_CONSTITUTION.md`, `CURRENT_STATE.md`, `PUBLICATION_STATUS.md`, `PROJECT_MATURITY.md`, `WHY_THIS_PROJECT.md`.
