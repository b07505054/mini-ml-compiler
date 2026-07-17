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
- Static symmetric INT8 fused Linear/MatMul + Bias + ReLU path: compiler-owned calibration and packed-weight artifacts, explicit Q/DQ integer IR, Cortex-A76 dot-product kernel lowering, canonical ExecutionPlan execution, and Raspberry Pi validation.
- Complete candidate search for the validated fused operator spans portable FP32, portable packed INT8, and ExecuTorch/XNNPACK FP32/INT8 implementations. Backend, runtime, delegate, quantization scheme, layout, fixed thread count, target capabilities, artifacts, and measured evidence are part of candidate identity.
- AArch64 backend-codegen evidence path for the same fused operator: project-owned Transform-dialect tiling/vectorization/unroll choices, unmodified LLVM 21.1.8 lowering/MIR/RA/scheduling, Raspberry Pi 5 validation, and explicit truth-boundary artifacts.
- Exact measured RMSNorm GPU candidate selection path: CUDA/Triton benchmark profiles normalize candidate identity, shape, target GPU, launch config, artifact hash, and measured p50 evidence into a canonical runtime plan with runtime redecision forbidden.
  This slice carries `weighted_rmsnorm` explicitly at the profile/candidate/plan boundary. The current single-input `hir.fused_rmsnorm` semantic op remains unweighted and is not presented as equivalent; a unified weighted typed HIR op is not implemented here.
- Measured vLLM `max_num_seqs` policy selector for one target/model/workload scope; the compiler emits the exact selected serving knob and forbids runtime redecision.
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
| Slice 3G complete-candidate agreement | fused Linear + Bias + ReLU, four Pi shapes | 4/4 selections agree with the constrained measured oracle; normalized regret 0.0 | external Slice 3G `selection_summary.json` |
| AArch64 schedule-unroll validation | Pi 5 Cortex-A76, FP32 tiled fused MatMul + Bias + ReLU | `schedule-unroll-k=4` measured fastest in 6/6 tested shape/tile domains; all candidates bit-exact | `artifacts/backend_codegen/aarch64_schedule_final/summary.md` |
| RMSNorm exact GPU selection | GTX 1650 Max-Q, weighted RMSNorm, exact shape/target/profile match | selected candidate is keyed by candidate id, backend, launch config, artifact hash, target GPU, p50 evidence; fallback used only when no valid exact measured candidate exists | `tools/mlir_fusion_to_runtime_json.py`, `tests/test_rmsnorm_exact_gpu_selection.py` |
| vLLM max-num-seqs selector | GTX 1650 Max-Q, Qwen 0.5B, vLLM 0.24.0, one knob | 45 measured baseline sessions plus 9 proof sessions; every proof executed the compiler-selected value with runtime redecision count 0 | runtime `artifacts/vllm_max_num_seqs_evaluation/summary.md` |

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
| Quantization Co-design | `ADVANCED_PARTIAL` | The fused Pi operator has compiler-owned static INT8 calibration, packed weights, Q/DQ integer IR, integer lowering, complete-candidate selection, and runtime validation. Full-model quantization, graph-wide mixed precision, INT4/AWQ/GPTQ integration, and NPU quantization remain incomplete. |
| CPU Attention / KV | `OPERATOR_LEVEL_EXECUTABLE` | A strict FP32 causal MHA pattern lowers to distinct prefill/decode contracts plus compiler-selected contiguous or paged KV implementations. Runtime owns KV lifetime; this is not full-model serving. |
| Hardware-Compiler Co-design | `EARLY_PARTIAL` | Pi measurements influence portable policy, XNNPACK X1, and AArch64 tile/unroll candidate selection. No SRAM/bandwidth/DMA/NPU design feedback yet. |

## Repository Map

- `ml-graph-compiler-runtime`: canonical compiler architecture, IR, candidates, feasibility, policy, ExecutionPlan/contract generation, phase reports.
- `heterogeneous-inference-runtime`: runtime validation/dispatch, evaluation harnesses, raw evidence, ExecuTorch E1/E2/E2.1/E3 results.
- `ml-platform-capabilities`: intended home for declared capability facts; partial today and not the sole source of truth.
- `Inference-Validation-Platform`: divergent local validation/control-plane project; not the canonical source repository host.
- Raspberry Pi: real execution/evidence target only.

## Current Limitations

- Triton is shadow-only and not production-integrated.
- Quantization co-design is complete only for the validated fused operator; there is no full-model accuracy/perplexity evaluation or graph-wide quantization policy.
- CPU attention is limited to static FP32 causal MHA with equal Q/KV heads and real runtime-owned contiguous or single-request paged KV. Full-model execution, general Transformer import, production serving, continuous batching, multi-request scheduling, shared-prefix/COW/eviction/swapping, FlashAttention, RoPE/QKV fusion, GQA/MQA, KV quantization, GPU/CUDA attention, distributed KV, explicit SIMD/NEON/AVX2, and predictive cost modelling remain unsupported.

### CPU attention implementation status

| Candidate or symbol | Status |
|---|---|
| `cpu_contiguous_kv_fp32_reordered_v1` — `token_major_contiguous_v_accumulation` | **PRODUCTION** |
| `cpu_paged_kv_fp32_page_major_v1` — `page_major_cached_page_base` | **PRODUCTION** |
| `cpu_contiguous_kv_fp32_v1` — `dimension_major_strided_v_accumulation` | **HISTORICAL_EXECUTABLE_BASELINE** |
| `cpu_paged_kv_fp32_v1` (artifact identity `cpu_paged_kv_fp32_token_major_v1`) — `token_major_block_translation` | **HISTORICAL_EXECUTABLE_BASELINE** |
| Runtime alias `hir_cpu_attention_decode_contiguous_kv_reordered_control_fp32` | **BENCHMARK_COMPATIBILITY_ONLY**; never compiler-selected |

Historical baselines remain executable for comparison, correctness regression,
performance ablation, selection validation, and artifact reproducibility.
Compiler owns candidate generation, legality, implementation identity,
layout/ABI contracts, exact measured-profile selection, and ExecutionPlan
export. Runtime owns allocation/lifetime, live KV/page state, fail-closed
validation, and exact selected-entry-point execution. The proven contract is
selected candidate equals executed candidate, with zero kernel/layout
reselection and zero temporary full-history materialization.
- Many decisions remain declared-profile, rule-based, shadow, or experimental rather than measured-profile-driven.
- Capability profiles are not yet fully synchronized across repositories.
- Implementation IR is incomplete for memory spaces, DMA, synchronization, heterogeneous partitioning, and NPU command regions.
- No general superiority claim over ExecuTorch, XNNPACK, vLLM, TVM, ONNX Runtime, TensorRT, or any device family is made.

## Reproduction Pointers

- P1D.1 policy artifact: `configs/thread_schedule_policies/raspberry_pi5_cortex_a76_p1d1_thread_policy.json`.
- P1D.1 raw evidence: sibling runtime `results/p1d_raspberry_pi_thread_decomposition/p1d_raw_measurements.json`.
- E3 compiler contract generator: `tools/e3_xnnpack_contract.py`.
- E3 runtime harness/evidence: sibling runtime `evaluation/executorch_e3/` and `results/executorch_e3/`.
- AArch64 backend-codegen schedule summary: `artifacts/backend_codegen/aarch64_schedule_final/summary.md`.
- RMSNorm exact GPU selection bridge: `tools/build_profile_cost_table.py`, `tools/mlir_fusion_to_runtime_json.py`, and `tests/test_rmsnorm_exact_gpu_selection.py`.
- vLLM `max_num_seqs` selector: `tools/select_vllm_max_num_seqs.py`; measured runtime evidence lives in sibling runtime `artifacts/vllm_max_num_seqs_evaluation/`.
- Canonical docs: `ARCHITECTURE_CONSTITUTION.md`, `CURRENT_STATE.md`, `PUBLICATION_STATUS.md`, `PROJECT_MATURITY.md`, `WHY_THIS_PROJECT.md`.
