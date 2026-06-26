# vLLM Serving Co-Design: Compiler Planning Artifacts

## Goal

This document describes a compiler-runtime co-design line that generates
serving planning artifacts aligned with vLLM's runtime architecture. The
artifacts express decisions a compiler can make upstream of a serving
framework — CUDA Graph bucket selection, KV cache layout, prefill/decode phase
split, and runtime replanning rules — without modifying vLLM source code.

The output is six JSON planning artifacts written to
`artifacts/vllm_serving_plan/`: four analysis/planning artifacts, one
compiler serving decision artifact that joins them into per-request decisions,
and one heuristic cost report that estimates colocated vs pd_split serving cost
per request.

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

### E. `compiler_serving_plan.json`

Consumes the four planning artifacts above and emits a per-request compiler
serving decision. The planner supports two decision modes, selected at runtime:

**`heuristic_rules` (default):** Applies a fixed priority rule list to choose
`execution_mode`. No dependency on the cost report.

**`cost_report_driven` (recommended):** Consumes `compiler_serving_cost_report.json`
and uses its `recommended_execution_mode` as the source of truth for execution
mode. The planner still computes `cuda_graph_bucket`, `replay_safe`, `kv_layout`,
and `input_pd_split_candidate` independently. CUDA Graph replay-safety does not
override the cost-driven execution mode — `replay_safe=false` is recorded but
does not change the mode.

Top-level fields:

| Field | Description |
|---|---|
| `decision_mode` | `heuristic_rules` or `cost_report_driven` |
| `cost_report_path` | Path to the cost report consumed (cost-driven mode only) |
| `cost_model_version` | Version of the cost model used (cost-driven mode only) |
| `cost_model_truth_boundary` | Truth boundary inherited from the cost report |
| `thresholds` | KV transfer and KV layout threshold constants |
| `per_request_decisions` | One decision entry per request in the trace |
| `summary` | Aggregate counts (colocated / pd_split / fallback / replay_safe) |

Per-request fields (all modes):

| Field | Description |
|---|---|
| `execution_mode` | `colocated`, `pd_split`, or `fallback` |
| `decision_source` | `heuristic_rules` or `compiler_serving_cost_report` |
| `input_pd_split_candidate` | Value from kv_cache_layout_plan (prompt-length based) |
| `final_pd_split_decision` | Whether execution_mode resolved to pd_split |
| `cuda_graph_bucket` | Selected replay-safe bucket for the decode step, or null |
| `replay_safe` | True when a matching replay-safe bucket exists |
| `kv_layout` | `paged` (≥ 15 MB) or `contiguous` |
| `estimated_kv_transfer_mb` | Non-zero for disaggregated candidates |
| `truth_boundary` | Per-request truth boundary label |

Additional per-request fields in **heuristic mode**:

| Field | Description |
|---|---|
| `decision_reasons` | List of rule names that drove the execution mode choice |

Additional per-request fields in **cost-driven mode**:

| Field | Description |
|---|---|
| `confidence` | `low` / `medium` / `high` — from cost report |
| `cost_summary.colocated_total_ms` | Estimated colocated serving latency |
| `cost_summary.pd_split_total_ms` | Estimated pd_split serving latency |
| `cost_summary.decision_margin_ms` | `abs(col − pd)` in ms |
| `cost_summary.decision_margin_pct` | `margin_ms / min(col, pd)` |
| `cost_explanation` | List of labels explaining the cost recommendation |

**Heuristic rule priority** (applies only in `heuristic_rules` mode):

1. `kv_transfer_mb > 25.0` → force `colocated` even if pd_split_candidate is true
2. `pd_split_candidate == true` (from kv_cache_layout_plan) → `pd_split`
3. `output_tokens >= 256` (decode-heavy) → `pd_split`
4. `prompt_tokens < 128` (short prompt) → `colocated`
5. else → `fallback`

**Divergence between modes:** On the 8-request fixture, cost-driven mode
disagrees with heuristic mode on 4 requests:

| Request | Heuristic | Cost-driven | Reason |
|---|---|---|---|
| r3 | colocated | **pd_split** | Queue savings (10.24→6.10 ms) outweigh KV transfer (1.13 ms); low confidence |
| r4 | pd_split | **colocated** | PD coordination overhead (2.0 ms) exceeds queue savings (0.58 ms); low confidence |
| r5 | pd_split | **colocated** | Same pattern; low confidence |
| r6 | pd_split | **colocated** | Same pattern; low confidence |

The `input_pd_split_candidate` / `final_pd_split_decision` split makes overrides
visible in both modes.

**Truth boundary:** `compiler_serving_plan_simulated_not_online_vllm_control` —
decisions are deterministic evaluations over planning artifacts. No CUDA Graph is
captured, no KV transfer occurs, and no vLLM instance is controlled. In
cost-driven mode, the execution decision is only as reliable as the cost model;
`cost_model_truth_boundary` in the artifact makes this traceable.

---

### F. `compiler_serving_cost_report.json`

Estimates the serving cost of two execution modes — `colocated` and `pd_split`
— for every request in the trace, and recommends the lower-cost mode. This is
a compiler-side heuristic cost model, not a measured vLLM benchmark.

| Field | Description |
|---|---|
| `per_request_costs` | One entry per request with two cost breakdowns and a recommendation |
| `colocated_cost` | Five cost components for the colocated execution path |
| `pd_split_cost` | Five cost components for the disaggregated prefill/decode path |
| `recommended_execution_mode` | `colocated` or `pd_split` — whichever has lower `total_ms` |
| `confidence` | `low` / `medium` / `high` based on the relative cost margin |
| `cost_explanation` | List of rule labels that explain the recommendation |
| `cost_sources` | Per-component source labels (e.g. `formula_synthetic`, `bandwidth_formula`) |
| `assumptions` | All constant values used by the model, named and labeled |
| `summary` | Aggregate win counts (colocated_wins / pd_split_wins) |

Cost components and their sources:

| Component | Formula | Source label |
|---|---|---|
| `compute_ms` | `PREFILL_MS_PER_TOKEN × prompt + DECODE_MS_PER_TOKEN × output` | `formula_synthetic` |
| `kv_transfer_ms` | `kv_transfer_mb / PD_BANDWIDTH_MB_PER_MS` (pd_split only) | `bandwidth_formula` / `not_applicable` |
| `queue_ms` | `QUEUE_MS_PER_PROMPT_TOKEN × prompt` (colocated); scaled + coordination overhead (pd_split) | `heuristic_queue_model` |
| `replay_gain_ms` | `REPLAY_GAIN_MS` if replay_safe, else 0 | `heuristic_cuda_graph_replay_gain` |
| `memory_penalty_ms` | `max(0, kv_mb − threshold) × MEMORY_PENALTY_FACTOR` | `heuristic_memory_pressure` |

`total_ms = compute_ms + kv_transfer_ms + queue_ms − replay_gain_ms + memory_penalty_ms`

Confidence is computed from the relative cost margin:
`diff_pct = abs(pd_total − col_total) / min(pd_total, col_total)`
- `< 0.05` → `low`
- `0.05 – 0.15` → `medium`
- `≥ 0.15` → `high`

**Divergence from heuristic planner:** On the 8-request fixture, the cost model
disagrees with `compiler_serving_plan.json` on 4 requests. Notably, r3
(prompt=512, kv_transfer=27 MB) is forced to `colocated` by the heuristic
planner's transfer-cost threshold rule, but the cost model recommends `pd_split`
because the queue savings for a large-prompt request (10.24 ms → 6.10 ms)
outweigh the KV transfer cost (1.13 ms). This divergence is intentional: the
cost report is the foundation for a future cost-based rewrite of
`compiler_serving_planner.py`.

**Truth boundary:** `heuristic_cost_model_not_measured_vllm_benchmark` — all
cost values are derived from token-count formulas and assumed constants, not
from GPU profiling or live vLLM runs. See the `assumptions` block in the
artifact for all constant values.

---

## vLLM Alignment

| Artifact | vLLM Extension Point |
|---|---|
| `cuda_graph_bucket_plan.json` | V1 CUDA Graph dispatcher — selects execution mode based on batch shape |
| `kv_cache_layout_plan.json` | Disaggregated prefill — separate prefill/decode instances with KV transfer under `vllm/distributed/kv_transfer` |
| `serving_analysis.json` | Prefill/decode phase split — foundational to vLLM's continuous batching model |
| `runtime_replan_report.json` | Serving policy adaptation — would feed a control loop above the vLLM scheduler |
| `compiler_serving_plan.json` | Decision synthesis — joins all four planning artifacts into per-request execution decisions |
| `compiler_serving_cost_report.json` | Cost-based serving mode selection — heuristic model for colocated vs pd_split cost per request |

None of these artifacts modify vLLM source code.

---

## How to Run

### Recommended workflow (cost-driven mode)

**Step 1 — Generate planning artifacts (A–D):**

```bash
python3 tools/generate_vllm_serving_plan.py
```

**Step 2 — Run the cost model (F):**

```bash
python3 tools/compiler_serving_cost_model.py
```

**Step 3 — Run the planner with the cost report (E, cost-driven mode):**

```bash
python3 tools/compiler_serving_planner.py \
  --cost-report artifacts/vllm_serving_plan/compiler_serving_cost_report.json
```

All three steps with explicit paths:

```bash
python3 tools/generate_vllm_serving_plan.py \
  --config configs/tiny_gpt_llm_config.json \
  --request-trace configs/vllm_serving_request_trace.json \
  --out artifacts/vllm_serving_plan

python3 tools/compiler_serving_cost_model.py \
  --plan-dir artifacts/vllm_serving_plan \
  --request-trace configs/vllm_serving_request_trace.json \
  --out artifacts/vllm_serving_plan/compiler_serving_cost_report.json

python3 tools/compiler_serving_planner.py \
  --plan-dir artifacts/vllm_serving_plan \
  --request-trace configs/vllm_serving_request_trace.json \
  --cost-report artifacts/vllm_serving_plan/compiler_serving_cost_report.json \
  --out artifacts/vllm_serving_plan/compiler_serving_plan.json
```

### Heuristic-only mode (no cost report)

If the cost report has not been generated or `--cost-report` is omitted, the
planner falls back to heuristic rules automatically:

```bash
python3 tools/generate_vllm_serving_plan.py
python3 tools/compiler_serving_planner.py
```

### Run tests

```bash
python3 -m unittest tests.test_vllm_serving_plan -v
python3 -m unittest tests.test_compiler_serving_planner -v
python3 -m unittest tests.test_compiler_serving_cost_model -v
```

---

## What This Is Not

- **Not a vLLM fork or patch.** No vLLM source is modified.
- **Not a live CUDA Graph capture.** `cuda_graph_bucket_plan.json` is metadata only.
- **Not an actual KV transfer implementation.** `kv_cache_layout_plan.json` identifies
  candidates; it does not transfer KV tensors between prefill and decode instances.
- **Not a measured benchmark.** All numeric values in `runtime_replan_report.json`
  are synthetic placeholders. The `observed_metrics_note` field says so explicitly.
- **Not a live decision controller.** `compiler_serving_plan.json` is a static
  evaluation over planning artifacts. It does not instrument or communicate with a
  running vLLM process in either mode.
- **Not a calibrated cost model.** Cost-driven mode produces decisions that are only
  as reliable as the cost model's heuristic formulas. Until the cost model is
  calibrated against measured GPU latency and real KV transfer bandwidth,
  `cost_model_truth_boundary` in the artifact reads
  `heuristic_cost_model_not_measured_vllm_benchmark`. Low confidence scores
  (margin < 5%) indicate the two modes are close and either decision is defensible.
- **Not a measured serving benchmark.** `compiler_serving_cost_report.json` uses
  token-count formulas and assumed bandwidth constants. No GPU profiling, vLLM
  benchmarking, or actual KV transfer measurement backs any value in it.
- **Not connected to heterogeneous-inference-runtime or mini-llm-serving-runtime-demo.**
  This is a standalone compiler artifact generation tool inside `ml-graph-compiler-runtime`.
