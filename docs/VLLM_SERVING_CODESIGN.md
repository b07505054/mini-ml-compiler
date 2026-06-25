# vLLM Serving Co-Design: Compiler Planning Artifacts

## Goal

This document describes a compiler-runtime co-design line that generates
serving planning artifacts aligned with vLLM's runtime architecture. The
artifacts express decisions a compiler can make upstream of a serving
framework — CUDA Graph bucket selection, KV cache layout, prefill/decode phase
split, and runtime replanning rules — without modifying vLLM source code.

The output is four JSON planning artifacts written to
`artifacts/vllm_serving_plan/`.

---

## Artifacts

### A. `serving_analysis.json`

Classifies a synthetic request trace into serving-relevant shapes.

| Field | Description |
|---|---|
| `prefill_decode_split` | Which ops are KV producers (prefill) and consumers (decode) |
| `shape_classification` | Prefill is dynamic-shape; decode is static-shape (token step) |
| `request_classes` | short_prompt / long_prompt / decode_heavy, with CUDA Graph eligibility |
| `batch_shape_buckets` | Candidate batch sizes for downstream bucket planning |

**Truth boundary:** `simulated` — derived from config and synthetic trace, not live traffic.

---

### B. `cuda_graph_bucket_plan.json`

Enumerates compiler-derived candidate buckets for vLLM CUDA Graph decode replay.

| Field | Description |
|---|---|
| `decode_buckets` | One entry per (batch_size × context_len) pair |
| `query_len` | Always 1 — decode steps process one token per sequence |
| `replay_safe` | True when batch_size ≤ 8 and context ≤ 2048 |
| `fallback_reason` | Set when replay_safe is false |
| `vllm_dispatcher_note` | Explains the relationship to vLLM V1 CUDA Graph dispatcher |

**Truth boundary:** `metadata_only_does_not_capture_cuda_graph` — no CUDA Graph
is captured or compiled here. This is planning metadata that could inform a
vLLM dispatcher configuration.

---

### C. `kv_cache_layout_plan.json`

Estimates KV cache footprint per request and flags PD disaggregation candidates.

| Field | Description |
|---|---|
| `per_request_entries` | One entry per request with KV byte estimates and recommendation |
| `pd_split_candidate` | True when prompt_tokens ≥ 256 (long prefill warrants PD split) |
| `recommendation` | `colocated` or `disaggregated` |
| `estimated_kv_transfer_mb` | Non-zero only for disaggregated candidates |
| `aggregate` | Totals and counts across all requests |
| `vllm_kv_transfer_note` | Explains the relationship to `vllm/distributed/kv_transfer` |

KV bytes per token are computed from the model config:
`num_layers × 2 × hidden_size × dtype_bytes(kv_dtype)`.

**Truth boundary:** `planning_artifact_not_actual_vllm_kv_transfer` — no KV
transfer occurs. The artifact identifies which requests would benefit from
disaggregated prefill in a vLLM deployment.

---

### D. `runtime_replan_report.json`

Applies a rule set to synthetic observed serving metrics and reports which
replanning actions should be triggered.

| Field | Description |
|---|---|
| `observed_metrics` | TTFT, TPOT, replay_hit_rate, kv_transfer_latency, queue_wait |
| `observed_metrics_note` | Explicit label: synthetic placeholders, not benchmark results |
| `replanning_rules` | Four rules with conditions and actions |
| `triggered_rules` | Which rules fired against the synthetic metrics |
| `actions` | The corresponding replanning actions |

Rules:
- `merge_buckets` — fires when replay_hit_rate > 0.90
- `split_buckets` — fires when TTFT_ms > 100.0
- `reduce_pd_candidates` — fires when kv_transfer_latency_ms > 10.0
- `prefer_colocated_fallback` — fires when TPOT_ms > 20.0

**Truth boundary:** `rule_based_replanning_simulation_not_online_vllm_control` —
this is a static rule evaluation against synthetic metrics. It does not
instrument or control a live vLLM instance.

---

## vLLM Alignment

| Artifact | vLLM Extension Point |
|---|---|
| `cuda_graph_bucket_plan.json` | V1 CUDA Graph dispatcher — selects execution mode based on batch shape |
| `kv_cache_layout_plan.json` | Disaggregated prefill — separate prefill/decode instances with KV transfer under `vllm/distributed/kv_transfer` |
| `serving_analysis.json` | Prefill/decode phase split — foundational to vLLM's continuous batching model |
| `runtime_replan_report.json` | Serving policy adaptation — would feed a control loop above the vLLM scheduler |

None of these artifacts modify vLLM source code.

---

## How to Run

```bash
python3 tools/generate_vllm_serving_plan.py
```

Uses defaults:
- Config: `configs/tiny_gpt_llm_config.json`
- Request trace: `configs/vllm_serving_request_trace.json`
- Output: `artifacts/vllm_serving_plan/`

With explicit paths:

```bash
python3 tools/generate_vllm_serving_plan.py \
  --config configs/tiny_gpt_llm_config.json \
  --request-trace configs/vllm_serving_request_trace.json \
  --out artifacts/vllm_serving_plan
```

Run tests:

```bash
python3 -m unittest tests.test_vllm_serving_plan -v
```

---

## What This Is Not

- **Not a vLLM fork or patch.** No vLLM source is modified.
- **Not a live CUDA Graph capture.** `cuda_graph_bucket_plan.json` is metadata only.
- **Not an actual KV transfer implementation.** `kv_cache_layout_plan.json` identifies
  candidates; it does not transfer KV tensors between prefill and decode instances.
- **Not a measured benchmark.** All numeric values in `runtime_replan_report.json`
  are synthetic placeholders. The `observed_metrics_note` field says so explicitly.
- **Not connected to heterogeneous-inference-runtime or mini-llm-serving-runtime-demo.**
  This is a standalone compiler artifact generation tool inside `ml-graph-compiler-runtime`.
