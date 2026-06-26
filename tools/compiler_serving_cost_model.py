#!/usr/bin/env python3
"""Compiler-side distributed serving cost model for vLLM serving plan artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Cost model constants (all heuristic — see truth_boundary)
# ---------------------------------------------------------------------------
PREFILL_MS_PER_TOKEN: float = 0.08
DECODE_MS_PER_TOKEN: float = 0.12
PD_BANDWIDTH_MB_PER_MS: float = 24.0       # 24 GB/s expressed as MB/ms
QUEUE_MS_PER_PROMPT_TOKEN: float = 0.02
PD_SPLIT_QUEUE_FACTOR: float = 0.4         # separate queues reduce HOL blocking by 60%
PD_COORDINATION_MS: float = 2.0            # fixed overhead for cross-instance coordination
REPLAY_GAIN_MS: float = 1.5
MEMORY_PENALTY_THRESHOLD_MB: float = 15.0
MEMORY_PENALTY_FACTOR: float = 0.05        # ms per MB above threshold

TRUTH_BOUNDARY = "heuristic_cost_model_not_measured_vllm_benchmark"

ASSUMPTIONS: dict[str, Any] = {
    "prefill_ms_per_token": PREFILL_MS_PER_TOKEN,
    "decode_ms_per_token": DECODE_MS_PER_TOKEN,
    "pd_bandwidth_mb_per_ms": PD_BANDWIDTH_MB_PER_MS,
    "queue_ms_per_prompt_token": QUEUE_MS_PER_PROMPT_TOKEN,
    "pd_split_queue_factor": PD_SPLIT_QUEUE_FACTOR,
    "pd_coordination_ms": PD_COORDINATION_MS,
    "replay_gain_ms": REPLAY_GAIN_MS,
    "memory_penalty_threshold_mb": MEMORY_PENALTY_THRESHOLD_MB,
    "memory_penalty_factor_ms_per_mb": MEMORY_PENALTY_FACTOR,
    "note": (
        "All values are heuristic placeholders. "
        "compute_ms uses a token-count formula by default, or nearest-neighbor GPU proxy lookup "
        "when --gpu-batch-scaling-artifact is provided. "
        "kv_transfer_ms uses an assumed 24 GB/s bandwidth. "
        "queue_ms and other components are synthetic estimates. "
        "See truth_boundary and compute_cost_source."
    ),
}


# ---------------------------------------------------------------------------
# Cost source builder
# ---------------------------------------------------------------------------

def _make_cost_sources(compute_source: str, is_pd_split: bool) -> dict[str, str]:
    return {
        "compute_ms": compute_source,
        "kv_transfer_ms": "bandwidth_formula" if is_pd_split else "not_applicable",
        "queue_ms": "heuristic_queue_model",
        "replay_gain_ms": "heuristic_cuda_graph_replay_gain",
        "memory_penalty_ms": "heuristic_memory_pressure",
    }


@dataclass
class ServingCostBreakdown:
    compute_ms: float
    kv_transfer_ms: float
    queue_ms: float
    replay_gain_ms: float
    memory_penalty_ms: float
    total_ms: float
    cost_sources: dict[str, str]
    compute_lookup: dict  # describes how compute_ms was derived

    def to_dict(self) -> dict[str, Any]:
        return {
            "compute_ms": round(self.compute_ms, 4),
            "kv_transfer_ms": round(self.kv_transfer_ms, 4),
            "queue_ms": round(self.queue_ms, 4),
            "replay_gain_ms": round(self.replay_gain_ms, 4),
            "memory_penalty_ms": round(self.memory_penalty_ms, 4),
            "total_ms": round(self.total_ms, 4),
            "cost_sources": self.cost_sources,
            "compute_lookup": self.compute_lookup,
        }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# GPU batch-scaling artifact loader
# ---------------------------------------------------------------------------

def load_gpu_batch_scaling(path: Path | None) -> tuple[dict | None, dict]:
    """Load GPU batch-scaling artifact and build metadata dict.

    Returns (artifact, meta) on success, (None, {}) on any error.
    Warns to stderr so the caller can always fall back to formula_synthetic.
    """
    if path is None:
        return None, {}
    if not path.exists():
        print(
            f"[warn] gpu_batch_scaling artifact not found: {path}; using formula_synthetic",
            file=sys.stderr,
        )
        return None, {}
    try:
        data = load_json(path)
    except Exception as exc:
        print(
            f"[warn] gpu_batch_scaling parse error ({exc}): {path}; using formula_synthetic",
            file=sys.stderr,
        )
        return None, {}
    results = data.get("results", {})
    if "decode" not in results or "prefill" not in results:
        print(
            f"[warn] gpu_batch_scaling missing results.decode or results.prefill: {path}; "
            "using formula_synthetic",
            file=sys.stderr,
        )
        return None, {}
    hidden_size = data.get("config", {}).get("hidden_size", "unknown")
    meta: dict = {
        "gpu_batch_scaling_artifact": str(path),
        "compute_cost_source": "gpu_measured_proxy",
        "compute_calibration_truth_boundary": (
            "GPU latency measurements were collected from a larger synthetic transformer workload "
            "and are used as a proxy calibration signal, not a direct latency prediction for tiny-gpt."
        ),
        "artifact_hardware": data.get("hardware", {}).get("gpu_name", "unknown"),
        "artifact_truth_boundary": data.get("truth_boundary", "unknown"),
        "lookup_method": "nearest_neighbor",
        "proxy_model_note": f"artifact hidden_size={hidden_size}; tiny-gpt hidden_size=768",
    }
    return data, meta


# ---------------------------------------------------------------------------
# Nearest-neighbor lookup helpers
# ---------------------------------------------------------------------------

def _nearest_prefill_row(
    artifact: dict, prompt_tokens: int, batch_size: int = 1
) -> dict | None:
    rows = [r for r in artifact["results"]["prefill"] if r["batch_size"] == batch_size]
    if not rows:
        return None
    return min(
        rows,
        key=lambda r: (abs(r["prefill_tokens"] - prompt_tokens), r["prefill_tokens"]),
    )


def _nearest_decode_row(
    artifact: dict, context_tokens: int, batch_size: int = 1
) -> dict | None:
    rows = [r for r in artifact["results"]["decode"] if r["batch_size"] == batch_size]
    if not rows:
        return None
    return min(
        rows,
        key=lambda r: (abs(r["context_tokens"] - context_tokens), r["context_tokens"]),
    )


def _nearest_prefill_ms(
    artifact: dict, prompt_tokens: int, batch_size: int = 1
) -> float | None:
    row = _nearest_prefill_row(artifact, prompt_tokens, batch_size)
    return row["latency_ms"]["mean_ms"] if row is not None else None


def _nearest_decode_step_ms(
    artifact: dict, context_tokens: int, batch_size: int = 1
) -> float | None:
    row = _nearest_decode_row(artifact, context_tokens, batch_size)
    return row["latency_ms"]["mean_ms"] if row is not None else None


def _resolve_compute_ms(
    prompt: int, output: int, artifact: dict | None = None
) -> tuple[float, str, dict]:
    """Return (compute_ms, compute_source, compute_lookup).

    GPU proxy mode uses nearest-neighbor lookup from the artifact.
    Falls back to formula_synthetic when artifact is None or lookup fails.
    """
    if artifact is not None:
        avg_ctx = int(prompt + output / 2)
        prefill_row = _nearest_prefill_row(artifact, prompt)
        decode_row = _nearest_decode_row(artifact, avg_ctx)
        if prefill_row is not None and decode_row is not None:
            compute = round(
                prefill_row["latency_ms"]["mean_ms"]
                + decode_row["latency_ms"]["mean_ms"] * output,
                4,
            )
            lookup: dict = {
                "mode": "gpu_proxy",
                "lookup_method": "nearest_neighbor",
                "prefill_lookup_tokens": prefill_row["prefill_tokens"],
                "decode_lookup_context_tokens": decode_row["context_tokens"],
                "avg_decode_context_tokens": avg_ctx,
            }
            return compute, "gpu_measured_proxy", lookup
    formula = round(PREFILL_MS_PER_TOKEN * prompt + DECODE_MS_PER_TOKEN * output, 4)
    return formula, "formula_synthetic", {"mode": "formula", "lookup_method": None}


# ---------------------------------------------------------------------------
# Per-mode estimators
# ---------------------------------------------------------------------------

def estimate_colocated(
    req: dict, kv_entry: dict, replay_safe: bool, artifact: dict | None = None
) -> ServingCostBreakdown:
    prompt: int = req["prompt_tokens"]
    output: int = req["output_tokens"]
    kv_mb: float = kv_entry["estimated_kv_mb"]

    compute, compute_source, compute_lookup = _resolve_compute_ms(prompt, output, artifact)
    kv_transfer = 0.0
    queue = QUEUE_MS_PER_PROMPT_TOKEN * prompt
    replay = REPLAY_GAIN_MS if replay_safe else 0.0
    memory = max(0.0, kv_mb - MEMORY_PENALTY_THRESHOLD_MB) * MEMORY_PENALTY_FACTOR
    total = compute + kv_transfer + queue - replay + memory

    return ServingCostBreakdown(
        compute_ms=compute,
        kv_transfer_ms=kv_transfer,
        queue_ms=queue,
        replay_gain_ms=replay,
        memory_penalty_ms=memory,
        total_ms=total,
        cost_sources=_make_cost_sources(compute_source, is_pd_split=False),
        compute_lookup=compute_lookup,
    )


def estimate_pd_split(
    req: dict, kv_entry: dict, replay_safe: bool, artifact: dict | None = None
) -> ServingCostBreakdown:
    prompt: int = req["prompt_tokens"]
    output: int = req["output_tokens"]
    kv_mb: float = kv_entry["estimated_kv_mb"]
    kv_transfer_mb: float = kv_entry["estimated_kv_transfer_mb"]

    compute, compute_source, compute_lookup = _resolve_compute_ms(prompt, output, artifact)
    kv_transfer = kv_transfer_mb / PD_BANDWIDTH_MB_PER_MS
    queue = QUEUE_MS_PER_PROMPT_TOKEN * prompt * PD_SPLIT_QUEUE_FACTOR + PD_COORDINATION_MS
    replay = REPLAY_GAIN_MS if replay_safe else 0.0
    memory = max(0.0, kv_mb - MEMORY_PENALTY_THRESHOLD_MB) * MEMORY_PENALTY_FACTOR
    total = compute + kv_transfer + queue - replay + memory

    return ServingCostBreakdown(
        compute_ms=compute,
        kv_transfer_ms=kv_transfer,
        queue_ms=queue,
        replay_gain_ms=replay,
        memory_penalty_ms=memory,
        total_ms=total,
        cost_sources=_make_cost_sources(compute_source, is_pd_split=True),
        compute_lookup=compute_lookup,
    )


# ---------------------------------------------------------------------------
# Confidence and explanation
# ---------------------------------------------------------------------------

def _confidence(col_total: float, pd_total: float) -> str:
    diff_pct = abs(pd_total - col_total) / min(pd_total, col_total)
    if diff_pct < 0.05:
        return "low"
    if diff_pct < 0.15:
        return "medium"
    return "high"


def _confidence_reasons(
    col_total: float, pd_total: float, compute_source: str
) -> list[str]:
    diff_pct = abs(pd_total - col_total) / min(pd_total, col_total)
    if diff_pct < 0.05:
        reasons = ["decision_margin_below_5_percent"]
        if compute_source == "gpu_measured_proxy":
            reasons.append("proxy_model_mismatch_can_dominate_small_policy_margins")
        return reasons
    if diff_pct < 0.15:
        return ["decision_margin_between_5_and_15_percent"]
    return ["decision_margin_at_least_15_percent"]


def _build_explanation(
    col: ServingCostBreakdown, pd: ServingCostBreakdown, recommended: str
) -> list[str]:
    explanation: list[str] = []
    if recommended == "pd_split":
        explanation.append("pd_split_lower_total_cost")
        if pd.queue_ms < col.queue_ms:
            explanation.append("pd_reduces_queue_delay_for_large_prompt")
        if pd.kv_transfer_ms > 0:
            explanation.append("kv_transfer_cost_included_in_pd_estimate")
    else:
        explanation.append("colocated_lower_total_cost")
        if pd.queue_ms > col.queue_ms:
            explanation.append("pd_coordination_overhead_exceeds_queue_savings")
    return explanation


# ---------------------------------------------------------------------------
# Per-request and batch estimation
# ---------------------------------------------------------------------------

def estimate_request_cost(
    req: dict, kv_entry: dict, replay_safe: bool, artifact: dict | None = None
) -> dict:
    col = estimate_colocated(req, kv_entry, replay_safe, artifact)
    pd = estimate_pd_split(req, kv_entry, replay_safe, artifact)
    recommended = "colocated" if col.total_ms <= pd.total_ms else "pd_split"
    confidence = _confidence(col.total_ms, pd.total_ms)
    compute_source = col.cost_sources["compute_ms"]
    confidence_reasons = _confidence_reasons(col.total_ms, pd.total_ms, compute_source)
    explanation = _build_explanation(col, pd, recommended)
    return {
        "request_id": req["request_id"],
        "colocated_cost": col.to_dict(),
        "pd_split_cost": pd.to_dict(),
        "recommended_execution_mode": recommended,
        "confidence": confidence,
        "confidence_reasons": confidence_reasons,
        "cost_explanation": explanation,
    }


def estimate_all(
    model_name: str,
    requests: list[dict],
    kv_entries_by_id: dict[str, dict],
    replay_safe_by_id: dict[str, bool],
    artifact: dict | None = None,
    artifact_meta: dict | None = None,
) -> dict:
    per_request = [
        estimate_request_cost(
            req,
            kv_entries_by_id[req["request_id"]],
            replay_safe_by_id.get(req["request_id"], True),
            artifact,
        )
        for req in requests
    ]
    colocated_wins = sum(
        1 for r in per_request if r["recommended_execution_mode"] == "colocated"
    )
    result: dict = {
        "schema_version": "0.1",
        "artifact_type": "compiler_serving_cost_report",
        "truth_boundary": TRUTH_BOUNDARY,
        "cost_model_version": "0.1",
        "model": model_name,
        "assumptions": ASSUMPTIONS,
        "per_request_costs": per_request,
        "summary": {
            "total_requests": len(per_request),
            "colocated_wins": colocated_wins,
            "pd_split_wins": len(per_request) - colocated_wins,
        },
    }
    if artifact_meta:
        result.update(artifact_meta)
    return result


# ---------------------------------------------------------------------------
# Artifact loader
# ---------------------------------------------------------------------------

def load_cost_inputs(
    plan_dir: Path, trace_path: Path
) -> tuple[str, list[dict], dict[str, dict], dict[str, bool]]:
    """Load planning artifacts and derive replay_safe per request from bucket plan."""
    kv_plan = load_json(plan_dir / "kv_cache_layout_plan.json")
    cuda_plan = load_json(plan_dir / "cuda_graph_bucket_plan.json")
    analysis = load_json(plan_dir / "serving_analysis.json")
    trace = load_json(trace_path)

    model_name: str = analysis["model"]
    kv_entries_by_id: dict[str, dict] = {
        e["request_id"]: e for e in kv_plan["per_request_entries"]
    }
    decode_buckets: list[dict] = cuda_plan["decode_buckets"]
    requests: list[dict] = trace["requests"]

    replay_safe_by_id: dict[str, bool] = {}
    for req in requests:
        total = req["prompt_tokens"] + req["output_tokens"]
        candidates = [
            b for b in decode_buckets
            if b["replay_safe"] and b["batch_size"] == 1 and b["context_bucket"] >= total
        ]
        replay_safe_by_id[req["request_id"]] = len(candidates) > 0

    return model_name, requests, kv_entries_by_id, replay_safe_by_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate colocated vs pd_split serving cost per request."
    )
    parser.add_argument(
        "--plan-dir",
        default=str(REPO_ROOT / "artifacts" / "vllm_serving_plan"),
        help="Directory containing kv_cache_layout_plan.json, cuda_graph_bucket_plan.json, serving_analysis.json",
    )
    parser.add_argument(
        "--request-trace",
        default=str(REPO_ROOT / "configs" / "vllm_serving_request_trace.json"),
        help="Synthetic request trace JSON",
    )
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "artifacts" / "vllm_serving_plan" / "compiler_serving_cost_report.json"
        ),
        help="Output path for compiler_serving_cost_report.json",
    )
    parser.add_argument(
        "--gpu-batch-scaling-artifact",
        default=None,
        help=(
            "Path to gpu_decode_batch_scaling JSON artifact "
            "(e.g., from heterogeneous-inference-runtime). "
            "When provided, compute_ms uses nearest-neighbor lookup as proxy calibration. "
            "Falls back to formula_synthetic if the file is missing or malformed. "
            "Note: the artifact may be from a different model architecture; "
            "see compute_calibration_truth_boundary in the output."
        ),
    )
    args = parser.parse_args()

    model_name, requests, kv_entries_by_id, replay_safe_by_id = load_cost_inputs(
        Path(args.plan_dir), Path(args.request_trace)
    )

    artifact_path = (
        Path(args.gpu_batch_scaling_artifact)
        if args.gpu_batch_scaling_artifact
        else None
    )
    artifact, artifact_meta = load_gpu_batch_scaling(artifact_path)

    result = estimate_all(
        model_name,
        requests,
        kv_entries_by_id,
        replay_safe_by_id,
        artifact,
        artifact_meta,
    )

    out_path = Path(args.out)
    write_json(out_path, result)
    print(out_path)


if __name__ == "__main__":
    main()
