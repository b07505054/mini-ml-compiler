#!/usr/bin/env python3
"""Compiler serving decision engine: joins vLLM planning artifacts into per-request decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

KV_TRANSFER_COLOCATED_OVERRIDE_MB: float = 25.0
KV_LAYOUT_PAGED_THRESHOLD_MB: float = 15.0

THRESHOLDS: dict[str, float] = {
    "kv_transfer_colocated_override_mb": KV_TRANSFER_COLOCATED_OVERRIDE_MB,
    "kv_layout_paged_threshold_mb": KV_LAYOUT_PAGED_THRESHOLD_MB,
}

TRUTH_BOUNDARY = "compiler_serving_plan_simulated_not_online_vllm_control"
DECISION_SOURCE_HEURISTIC = "heuristic_rules"
DECISION_SOURCE_COST_REPORT = "compiler_serving_cost_report"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def load_cost_entries(
    cost_report_path: Path | None,
) -> tuple[dict[str, dict] | None, dict]:
    """Load cost report and index by request_id.

    Returns (cost_entries_by_id, meta) on success, (None, {}) on any error.
    Warnings are printed to stderr so the caller can always fall back to heuristics.
    """
    if cost_report_path is None:
        return None, {}
    if not cost_report_path.exists():
        print(
            f"[warn] cost report not found: {cost_report_path}; falling back to heuristic rules",
            file=sys.stderr,
        )
        return None, {}
    try:
        data = load_json(cost_report_path)
    except Exception as exc:
        print(
            f"[warn] cost report parse error ({exc}): {cost_report_path}; "
            "falling back to heuristic rules",
            file=sys.stderr,
        )
        return None, {}
    if not isinstance(data, dict) or "per_request_costs" not in data:
        print(
            f"[warn] cost report malformed (missing per_request_costs): {cost_report_path}; "
            "falling back to heuristic rules",
            file=sys.stderr,
        )
        return None, {}
    cost_entries_by_id: dict[str, dict] = {
        e["request_id"]: e for e in data["per_request_costs"]
    }
    meta: dict = {
        "cost_report_path": str(cost_report_path),
        "cost_model_version": data.get("cost_model_version", "unknown"),
        "cost_model_truth_boundary": data.get("truth_boundary", "unknown"),
    }
    return cost_entries_by_id, meta


def _cost_summary(cost_entry: dict) -> dict:
    col: float = cost_entry["colocated_cost"]["total_ms"]
    pd: float = cost_entry["pd_split_cost"]["total_ms"]
    margin_ms = round(abs(col - pd), 4)
    margin_pct = round(margin_ms / min(col, pd), 4)
    return {
        "colocated_total_ms": col,
        "pd_split_total_ms": pd,
        "decision_margin_ms": margin_ms,
        "decision_margin_pct": margin_pct,
    }


def _decide_execution_mode(req: dict, kv_entry: dict) -> tuple[str, list[str]]:
    """Apply priority-ordered rules. Returns (execution_mode, decision_reasons)."""
    kv_transfer_mb: float = kv_entry["estimated_kv_transfer_mb"]
    pd_candidate: bool = kv_entry["pd_split_candidate"]
    prompt: int = req["prompt_tokens"]
    output: int = req["output_tokens"]

    # Rule 1 (highest): KV transfer cost too high — force colocated regardless of other rules
    if kv_transfer_mb > KV_TRANSFER_COLOCATED_OVERRIDE_MB:
        return "colocated", ["kv_transfer_cost_exceeds_threshold_colocated_fallback"]

    # Rule 2: KV layout plan flagged this request as a PD split candidate
    if pd_candidate:
        return "pd_split", ["pd_split_candidate_from_kv_layout_plan"]

    # Rule 3: decode-heavy — long output sequence benefits from disaggregated decode
    if output >= 256:
        return "pd_split", ["decode_heavy_pd_split"]

    # Rule 4: short prompt — no disaggregation benefit
    if prompt < 128:
        return "colocated", ["short_prompt_colocated"]

    return "fallback", ["no_matching_rule"]


def _select_bucket(
    total_tokens: int, decode_buckets: list[dict]
) -> tuple[dict | None, str | None]:
    """Find smallest replay_safe bucket (batch_size=1) where context_bucket >= total_tokens."""
    candidates = [
        b for b in decode_buckets
        if b["replay_safe"] and b["batch_size"] == 1 and b["context_bucket"] >= total_tokens
    ]
    if not candidates:
        return None, "no_replay_safe_bucket_for_context_size"
    return min(candidates, key=lambda b: b["context_bucket"]), None


def _decide_kv_layout(kv_mb: float) -> str:
    return "paged" if kv_mb >= KV_LAYOUT_PAGED_THRESHOLD_MB else "contiguous"


def plan_request(
    req: dict,
    kv_entry: dict,
    decode_buckets: list[dict],
    cost_entry: dict | None = None,
) -> dict:
    total_tokens = req["prompt_tokens"] + req["output_tokens"]
    bucket, fallback_reason = _select_bucket(total_tokens, decode_buckets)

    cuda_graph_bucket: dict | None = None
    if bucket is not None:
        cuda_graph_bucket = {
            "bucket_id": bucket["bucket_id"],
            "batch_size": bucket["batch_size"],
            "context_bucket": bucket["context_bucket"],
            "replay_safe": True,
        }

    if cost_entry is not None:
        execution_mode = cost_entry["recommended_execution_mode"]
        decision_source = DECISION_SOURCE_COST_REPORT
        extra: dict = {
            "confidence": cost_entry["confidence"],
            "cost_summary": _cost_summary(cost_entry),
            "cost_explanation": cost_entry["cost_explanation"],
        }
    else:
        execution_mode, reasons = _decide_execution_mode(req, kv_entry)
        decision_source = DECISION_SOURCE_HEURISTIC
        extra = {"decision_reasons": reasons}

    return {
        "request_id": req["request_id"],
        "prompt_tokens": req["prompt_tokens"],
        "output_tokens": req["output_tokens"],
        "execution_mode": execution_mode,
        "decision_source": decision_source,
        "input_pd_split_candidate": kv_entry["pd_split_candidate"],
        "final_pd_split_decision": execution_mode == "pd_split",
        "cuda_graph_bucket": cuda_graph_bucket,
        "replay_safe": bucket is not None,
        "replay_fallback_reason": fallback_reason,
        "kv_layout": _decide_kv_layout(kv_entry["estimated_kv_mb"]),
        "estimated_kv_transfer_mb": kv_entry["estimated_kv_transfer_mb"],
        "truth_boundary": TRUTH_BOUNDARY,
        **extra,
    }


def plan_all(
    model_name: str,
    requests: list[dict],
    kv_entries_by_id: dict[str, dict],
    decode_buckets: list[dict],
    cost_entries_by_id: dict[str, dict] | None = None,
    cost_report_meta: dict | None = None,
) -> dict:
    decisions = [
        plan_request(
            req,
            kv_entries_by_id[req["request_id"]],
            decode_buckets,
            cost_entries_by_id.get(req["request_id"]) if cost_entries_by_id is not None else None,
        )
        for req in requests
    ]
    result: dict = {
        "schema_version": "0.1",
        "artifact_type": "compiler_serving_plan",
        "truth_boundary": TRUTH_BOUNDARY,
        "model": model_name,
        "thresholds": THRESHOLDS,
        "decision_mode": "cost_report_driven" if cost_entries_by_id is not None else "heuristic_rules",
        "per_request_decisions": decisions,
        "summary": {
            "total_requests": len(decisions),
            "colocated_count": sum(1 for d in decisions if d["execution_mode"] == "colocated"),
            "pd_split_count": sum(1 for d in decisions if d["execution_mode"] == "pd_split"),
            "fallback_count": sum(1 for d in decisions if d["execution_mode"] == "fallback"),
            "replay_safe_count": sum(1 for d in decisions if d["replay_safe"]),
        },
    }
    if cost_report_meta:
        result.update(cost_report_meta)
    return result


def load_plan_inputs(
    plan_dir: Path, trace_path: Path
) -> tuple[str, list[dict], dict[str, dict], list[dict]]:
    """Load and pre-process artifacts from plan_dir and trace_path for plan_all()."""
    kv_plan = load_json(plan_dir / "kv_cache_layout_plan.json")
    cuda_plan = load_json(plan_dir / "cuda_graph_bucket_plan.json")
    analysis = load_json(plan_dir / "serving_analysis.json")
    request_trace = load_json(trace_path)

    model_name: str = analysis["model"]
    kv_entries_by_id: dict[str, dict] = {
        e["request_id"]: e for e in kv_plan["per_request_entries"]
    }
    decode_buckets: list[dict] = cuda_plan["decode_buckets"]
    requests: list[dict] = request_trace["requests"]

    return model_name, requests, kv_entries_by_id, decode_buckets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit per-request compiler serving decisions from vLLM planning artifacts."
    )
    parser.add_argument(
        "--plan-dir",
        default=str(REPO_ROOT / "artifacts" / "vllm_serving_plan"),
        help="Directory containing serving_analysis.json, cuda_graph_bucket_plan.json, kv_cache_layout_plan.json",
    )
    parser.add_argument(
        "--request-trace",
        default=str(REPO_ROOT / "configs" / "vllm_serving_request_trace.json"),
        help="Synthetic request trace JSON",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "artifacts" / "vllm_serving_plan" / "compiler_serving_plan.json"),
        help="Output path for compiler_serving_plan.json",
    )
    parser.add_argument(
        "--cost-report",
        default=None,
        help=(
            "Path to compiler_serving_cost_report.json. "
            "If provided, uses cost model recommendations for execution_mode. "
            "Falls back to heuristic rules if the file is missing or malformed."
        ),
    )
    args = parser.parse_args()

    model_name, requests, kv_entries_by_id, decode_buckets = load_plan_inputs(
        Path(args.plan_dir), Path(args.request_trace)
    )

    cost_report_path = Path(args.cost_report) if args.cost_report else None
    cost_entries_by_id, meta = load_cost_entries(cost_report_path)

    result = plan_all(
        model_name,
        requests,
        kv_entries_by_id,
        decode_buckets,
        cost_entries_by_id,
        meta if cost_entries_by_id is not None else None,
    )

    out_path = Path(args.out)
    write_json(out_path, result)
    print(out_path)


if __name__ == "__main__":
    main()
