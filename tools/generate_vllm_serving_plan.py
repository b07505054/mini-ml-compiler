#!/usr/bin/env python3
"""Generate vLLM-facing serving planning artifacts for compiler-runtime co-design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DTYPE_BYTES: dict[str, int] = {
    "fp16": 2,
    "bf16": 2,
    "float16": 2,
    "fp32": 4,
    "float32": 4,
    "int8": 1,
    "fp8": 1,
}

CONTEXT_BUCKETS: list[int] = [128, 256, 512, 1024, 2048, 4096]

REPLAN_RULES: list[dict[str, str]] = [
    {
        "rule_id": "merge_buckets",
        "condition": "replay_hit_rate > 0.90",
        "action": "consolidate underused small decode buckets to reduce CUDA Graph capture overhead",
    },
    {
        "rule_id": "split_buckets",
        "condition": "TTFT_ms > 100.0",
        "action": "split large batch buckets into smaller decode groups to reduce head-of-line blocking",
    },
    {
        "rule_id": "reduce_pd_candidates",
        "condition": "kv_transfer_latency_ms > 10.0",
        "action": "reduce PD disaggregation candidates; prefer colocated for latency-sensitive requests",
    },
    {
        "rule_id": "prefer_colocated_fallback",
        "condition": "TPOT_ms > 20.0",
        "action": "switch disaggregated candidates to colocated mode to reduce per-token latency overhead",
    },
]

SYNTHETIC_OBSERVED_METRICS: dict[str, float] = {
    "TTFT_ms": 120.0,
    "TPOT_ms": 8.0,
    "replay_hit_rate": 0.85,
    "kv_transfer_latency_ms": 3.5,
    "queue_wait_ms": 15.0,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _kv_bytes_per_token(model_cfg: dict) -> int:
    m = model_cfg["model"]
    kv_dtype = model_cfg["kv_cache"]["kv_dtype"]
    return m["num_layers"] * 2 * m["hidden_size"] * DTYPE_BYTES[kv_dtype]


def _classify_request(req: dict) -> str:
    if req["output_tokens"] > 256:
        return "decode_heavy"
    if req["prompt_tokens"] > 64:
        return "long_prompt"
    return "short_prompt"


def build_serving_analysis(model_cfg: dict, request_trace: dict) -> dict:
    requests = request_trace["requests"]
    by_class: dict[str, list[str]] = {
        "short_prompt": [],
        "long_prompt": [],
        "decode_heavy": [],
    }
    for req in requests:
        by_class[_classify_request(req)].append(req["request_id"])

    return {
        "schema_version": "0.1",
        "artifact_type": "serving_analysis",
        "truth_boundary": "simulated",
        "model": model_cfg["model"]["name"],
        "prefill_decode_split": {
            "prefill_phase": "attention_prefill",
            "prefill_kv_role": "producer",
            "decode_phase": "attention_decode",
            "decode_kv_role": "consumer",
            "pd_separation_viable": True,
        },
        "prompt_output_assumptions": {
            "short_prompt_max_tokens": 64,
            "long_prompt_min_tokens": 65,
            "decode_heavy_min_output_tokens": 257,
        },
        "shape_classification": {
            "prefill": "dynamic",
            "decode": "static",
        },
        "request_classes": [
            {
                "class": "short_prompt",
                "request_ids": by_class["short_prompt"],
                "cuda_graph_eligible": True,
                "reason": "small static decode step fits smallest CUDA Graph bucket",
            },
            {
                "class": "long_prompt",
                "request_ids": by_class["long_prompt"],
                "cuda_graph_eligible": False,
                "reason": "dynamic prompt length may exceed bucket boundary; prefill uses eager execution",
            },
            {
                "class": "decode_heavy",
                "request_ids": by_class["decode_heavy"],
                "cuda_graph_eligible": True,
                "reason": "long decode sequences benefit from CUDA Graph replay at decode step",
            },
        ],
        "batch_shape_buckets": [1, 2, 4, 8, 16],
    }


def build_cuda_graph_bucket_plan(serving_analysis: dict, model_cfg: dict) -> dict:
    batch_sizes: list[int] = serving_analysis["batch_shape_buckets"]
    max_context: int = model_cfg["kv_cache"]["max_context_tokens"]

    buckets = []
    bucket_id = 0
    for bs in batch_sizes:
        for ctx in CONTEXT_BUCKETS:
            if ctx > max_context:
                continue
            replay_safe = bs <= 8 and ctx <= 2048
            if not replay_safe:
                fallback_reason = (
                    "batch_size_exceeds_safe_capture_range"
                    if bs > 8
                    else "context_exceeds_safe_capture_range"
                )
            else:
                fallback_reason = None
            buckets.append({
                "bucket_id": bucket_id,
                "batch_size": bs,
                "query_len": 1,
                "context_bucket": ctx,
                "replay_safe": replay_safe,
                "fallback_reason": fallback_reason,
            })
            bucket_id += 1

    return {
        "schema_version": "0.1",
        "artifact_type": "cuda_graph_bucket_plan",
        "truth_boundary": "metadata_only_does_not_capture_cuda_graph",
        "model": model_cfg["model"]["name"],
        "vllm_dispatcher_note": (
            "vLLM V1 selects CUDA Graph mode based on batch shape at runtime. "
            "This plan describes compiler-derived candidate buckets for decode replay. "
            "No CUDA Graph is captured here; this artifact is planning metadata only."
        ),
        "decode_buckets": buckets,
    }


def build_kv_cache_layout_plan(request_trace: dict, model_cfg: dict) -> dict:
    kv_per_token = _kv_bytes_per_token(model_cfg)
    pd_threshold = 256
    entries = []
    total_kv_bytes = 0
    pd_split_count = 0
    colocated_count = 0
    disaggregated_count = 0

    for req in request_trace["requests"]:
        total_tokens = req["prompt_tokens"] + req["output_tokens"]
        kv_bytes = kv_per_token * total_tokens
        kv_mb = round(kv_bytes / (1024 * 1024), 4)
        pd_split = req["prompt_tokens"] >= pd_threshold
        recommendation = "disaggregated" if pd_split else "colocated"

        total_kv_bytes += kv_bytes
        if pd_split:
            pd_split_count += 1
            disaggregated_count += 1
        else:
            colocated_count += 1

        entries.append({
            "request_id": req["request_id"],
            "prompt_tokens": req["prompt_tokens"],
            "output_tokens": req["output_tokens"],
            "total_tokens": total_tokens,
            "estimated_kv_bytes": kv_bytes,
            "estimated_kv_mb": kv_mb,
            "kv_producer_phase": "prefill",
            "kv_consumer_phase": "decode",
            "pd_split_candidate": pd_split,
            "estimated_kv_transfer_mb": kv_mb if pd_split else 0.0,
            "recommendation": recommendation,
        })

    return {
        "schema_version": "0.1",
        "artifact_type": "kv_cache_layout_plan",
        "truth_boundary": "planning_artifact_not_actual_vllm_kv_transfer",
        "model": model_cfg["model"]["name"],
        "vllm_kv_transfer_note": (
            "vLLM disaggregated prefill uses separate prefill/decode instances with KV transfer "
            "connectors under vllm/distributed/kv_transfer. This artifact identifies candidate "
            "requests for PD disaggregation based on prompt length. "
            "No actual KV transfer occurs here."
        ),
        "pd_split_threshold_prompt_tokens": pd_threshold,
        "per_request_entries": entries,
        "aggregate": {
            "total_estimated_kv_mb": round(total_kv_bytes / (1024 * 1024), 4),
            "pd_split_candidate_count": pd_split_count,
            "colocated_count": colocated_count,
            "disaggregated_count": disaggregated_count,
        },
    }


def _evaluate_rules(
    metrics: dict[str, float],
) -> tuple[list[str], list[str]]:
    checks: dict[str, bool] = {
        "merge_buckets": metrics["replay_hit_rate"] > 0.90,
        "split_buckets": metrics["TTFT_ms"] > 100.0,
        "reduce_pd_candidates": metrics["kv_transfer_latency_ms"] > 10.0,
        "prefer_colocated_fallback": metrics["TPOT_ms"] > 20.0,
    }
    triggered: list[str] = []
    actions: list[str] = []
    for rule in REPLAN_RULES:
        rid = rule["rule_id"]
        if checks.get(rid, False):
            triggered.append(rid)
            actions.append(rule["action"])
    return triggered, actions


def build_runtime_replan_report(serving_analysis: dict) -> dict:
    metrics = dict(SYNTHETIC_OBSERVED_METRICS)
    triggered, actions = _evaluate_rules(metrics)

    return {
        "schema_version": "0.1",
        "artifact_type": "runtime_replan_report",
        "truth_boundary": "rule_based_replanning_simulation_not_online_vllm_control",
        "model": serving_analysis["model"],
        "observed_metrics_note": (
            "Synthetic placeholder values for schema demonstration. "
            "Not measured benchmark results. "
            "Replace with live vLLM metrics when instrumented."
        ),
        "observed_metrics": metrics,
        "replanning_rules": REPLAN_RULES,
        "triggered_rules": triggered,
        "actions": actions,
    }


def generate_all(model_cfg: dict, request_trace: dict, out_dir: Path) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis = build_serving_analysis(model_cfg, request_trace)
    cuda_plan = build_cuda_graph_bucket_plan(analysis, model_cfg)
    kv_plan = build_kv_cache_layout_plan(request_trace, model_cfg)
    replan = build_runtime_replan_report(analysis)

    artifacts: dict[str, Any] = {
        "serving_analysis.json": analysis,
        "cuda_graph_bucket_plan.json": cuda_plan,
        "kv_cache_layout_plan.json": kv_plan,
        "runtime_replan_report.json": replan,
    }

    for filename, data in artifacts.items():
        write_json(out_dir / filename, data)

    return list(artifacts.keys())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate vLLM-facing serving planning artifacts."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "tiny_gpt_llm_config.json"),
        help="Model config JSON (default: configs/tiny_gpt_llm_config.json)",
    )
    parser.add_argument(
        "--request-trace",
        default=str(REPO_ROOT / "configs" / "vllm_serving_request_trace.json"),
        help="Synthetic request trace JSON (default: configs/vllm_serving_request_trace.json)",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "artifacts" / "vllm_serving_plan"),
        help="Output directory (default: artifacts/vllm_serving_plan/)",
    )
    args = parser.parse_args()

    model_cfg = load_json(Path(args.config))
    request_trace = load_json(Path(args.request_trace))
    out_dir = Path(args.out)

    written = generate_all(model_cfg, request_trace, out_dir)
    for filename in written:
        print(filename)


if __name__ == "__main__":
    main()
