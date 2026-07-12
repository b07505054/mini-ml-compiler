#!/usr/bin/env python3
"""Competitive Triton fused MatMul + Bias + ReLU candidate discovery.

This runner is intentionally separate from the V1/V3 fair-fusion benchmark.
It evaluates only one-pass fused kernels with different fixed Triton
configurations, so the measured winner reflects scheduling/configuration
trade-offs rather than fusion-vs-unfusion attribution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_triton_matmul_bias_relu_benchmark as base  # noqa: E402
from matmul_postop_workloads import (  # noqa: E402
    canonical_workloads,
    decision_boundary_workloads,
    load_manifest,
)


SCHEMA = "triton_matmul_bias_relu_fused_candidate_sweep"
SCHEMA_VERSION = 1
BACKEND = "triton_cuda"
KERNEL_FAMILY_ID = "triton_matmul_bias_relu_one_pass_f32"
PATTERN = "bias"
DTYPE = "f32"
DEFAULT_TIE_THRESHOLD = 0.01
DEFAULT_NEAR_BEST_THRESHOLD = 0.03
DEFAULT_STABLE_CV_LIMIT = 0.05


@dataclass(frozen=True)
class FusedCandidateConfig:
    config_id: str
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    precision_mode: str = "ieee"
    hypothesis: str = ""

    def validate(self) -> None:
        if self.block_m <= 0 or self.block_n <= 0 or self.block_k <= 0:
            raise ValueError("block dimensions must be positive")
        if self.num_warps not in {1, 2, 4, 8}:
            raise ValueError(f"unsupported num_warps for {self.config_id}")
        if self.num_stages <= 0:
            raise ValueError("num_stages must be positive")
        if self.precision_mode != "ieee":
            raise ValueError("candidate discovery is fp32 ieee only")
        if self.block_m * self.block_n > 4096:
            raise ValueError(f"unsupported tile area for {self.config_id}")

    def upper(self) -> dict[str, Any]:
        return {
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "BLOCK_K": self.block_k,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "precision_mode": self.precision_mode,
            "input_dtype": "torch.float32",
            "accumulator_dtype": "fp32",
            "tf32_enabled": False,
            "config_source": "fused_candidate_discovery",
        }

    def typed(self) -> dict[str, Any]:
        return {
            "kernel_family_id": KERNEL_FAMILY_ID,
            "config_id": self.config_id,
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "precision_mode": self.precision_mode,
            "input_dtype": DTYPE,
            "accumulator_dtype": "fp32",
            "runtime_operations": 1,
            "expected_launches": 1,
            "full_size_intermediates": 0,
            "fusion": "one_pass_epilogue",
            "hypothesis": self.hypothesis,
        }


DEFAULT_CANDIDATES = [
    FusedCandidateConfig("bm16_bn16_bk32_w4_s3", 16, 16, 32, 4, 3, hypothesis="small tile baseline for skinny or edge-heavy shapes"),
    FusedCandidateConfig("bm32_bn32_bk32_w4_s3", 32, 32, 32, 4, 3, hypothesis="balanced square tile for regular shapes"),
    FusedCandidateConfig("bm64_bn64_bk32_w4_s3", 64, 64, 32, 4, 3, hypothesis="larger tile for regular output reuse"),
    FusedCandidateConfig("bm16_bn64_bk32_w4_s3", 16, 64, 32, 4, 3, hypothesis="wide tile for small-M wide-N shapes"),
    FusedCandidateConfig("bm64_bn16_bk32_w4_s3", 64, 16, 32, 4, 3, hypothesis="tall tile for tall or small-N shapes"),
    FusedCandidateConfig("bm32_bn64_bk32_w4_s3", 32, 64, 32, 4, 3, hypothesis="moderate wide tile for LLM-like N-heavy shapes"),
]


def candidate_by_id(config_id: str) -> FusedCandidateConfig:
    for cfg in DEFAULT_CANDIDATES:
        if cfg.config_id == config_id:
            return cfg
    raise ValueError(f"unknown candidate config: {config_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/matmul_postop_workloads.json")
    parser.add_argument("--mode", choices=("feasibility", "coarse-sweep", "formal-sweep"), default="coarse-sweep")
    parser.add_argument("--workload-id", action="append")
    parser.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--include-decision-boundary", action="store_true")
    parser.add_argument("--decision-boundary-only", action="store_true")
    parser.add_argument("--candidate-config", action="append", dest="candidate_configs")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-order", choices=("listed", "alternating", "randomized"), default="alternating")
    parser.add_argument("--memory-safety-fraction", type=float, default=base.MEMORY_SAFETY_FRACTION)
    parser.add_argument("--tie-threshold", type=float, default=DEFAULT_TIE_THRESHOLD)
    parser.add_argument("--near-best-threshold", type=float, default=DEFAULT_NEAR_BEST_THRESHOLD)
    parser.add_argument("--stable-cv-limit", type=float, default=DEFAULT_STABLE_CV_LIMIT)
    parser.add_argument("--dominated-threshold", type=float, default=0.05)
    parser.add_argument("--smoke-test-mode", action="store_true")
    parser.add_argument("--output", default="trace/matmul_postop_triton_fused_candidate_sweep.json")
    parser.add_argument("--report-output", default="trace/matmul_postop_triton_fused_candidate_report.md")
    parser.add_argument("--doc-output", default="DOC/result/TRITON_FUSED_KERNEL_CANDIDATE_DIVERSITY.md")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup <= 0 and not args.smoke_test_mode:
        raise ValueError("warmup must be nonzero unless --smoke-test-mode is set")
    formal = args.warmup >= 50 and args.iterations >= 300 and args.repeats >= 5
    if not formal and not args.smoke_test_mode:
        raise ValueError("formal runs require warmup>=50, iterations>=300, repeats>=5")
    if args.mode == "formal-sweep" and args.sessions < 3 and not args.smoke_test_mode:
        raise ValueError("formal fused-candidate sweep requires --sessions >= 3")
    if args.sessions <= 0:
        raise ValueError("--sessions must be positive")
    if args.tie_threshold <= 0.0 or args.near_best_threshold <= 0.0:
        raise ValueError("thresholds must be positive")
    if args.dominated_threshold <= 0.0:
        raise ValueError("--dominated-threshold must be positive")


def selected_candidates(args: argparse.Namespace) -> list[FusedCandidateConfig]:
    configs = [candidate_by_id(cid) for cid in args.candidate_configs] if args.candidate_configs else list(DEFAULT_CANDIDATES)
    if not (4 <= len(configs) <= 8) and not args.smoke_test_mode:
        raise ValueError("candidate discovery requires 4 to 8 initial configs")
    seen: set[str] = set()
    for cfg in configs:
        cfg.validate()
        if cfg.config_id in seen:
            raise ValueError(f"duplicate candidate config: {cfg.config_id}")
        seen.add(cfg.config_id)
    return configs


def select_workloads(args: argparse.Namespace) -> list[Any]:
    workloads = load_manifest(args.manifest)
    if args.decision_boundary_only:
        workloads = decision_boundary_workloads(workloads)
    elif args.include_decision_boundary:
        workloads = canonical_workloads(workloads) + decision_boundary_workloads(workloads)
    else:
        workloads = canonical_workloads(workloads)
    ids = set(args.workload_id or [])
    selected = workloads if args.all_eligible else [w for w in workloads if w.workload_id in ids]
    missing = ids - {w.workload_id for w in selected}
    if missing:
        raise ValueError(f"unknown workload id(s): {sorted(missing)}")
    if not selected:
        raise ValueError("no workloads selected")
    return selected


def candidate_order(configs: list[FusedCandidateConfig], args: argparse.Namespace, session: int, workload_index: int) -> list[FusedCandidateConfig]:
    ordered = list(configs)
    if args.candidate_order == "listed":
        return ordered
    if args.candidate_order == "alternating":
        shift = (session + workload_index) % len(ordered)
        return ordered[shift:] + ordered[:shift]
    rng = random.Random(args.seed + session * 1009 + workload_index * 9176)
    rng.shuffle(ordered)
    return ordered


def build_one_pass_kernel(triton: Any, tl: Any):
    @triton.jit
    def _matmul_bias_relu_one_pass_kernel(a_ptr, b_ptr, bias_ptr, out_ptr,
                                          M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                          BLOCK_K: tl.constexpr, INPUT_PRECISION: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            a = tl.load(a_ptr + offs_m[:, None] * K + k_idxs[None, :],
                        mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K), other=0.0)
            b = tl.load(b_ptr + k_idxs[:, None] * N + offs_n[None, :],
                        mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b, input_precision=INPUT_PRECISION)
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
        acc = tl.maximum(acc, 0.0)
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    return _matmul_bias_relu_one_pass_kernel


def make_fused_runner(torch: Any, kernel: Any, cfg: FusedCandidateConfig, m: int, n: int, k: int,
                      a: Any, b: Any, bias: Any) -> Callable[[], Any]:
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    grid = (math.ceil(m / cfg.block_m), math.ceil(n / cfg.block_n))

    def run() -> Any:
        kernel[grid](
            a, b, bias, out,
            M=m, N=n, K=k,
            BLOCK_M=cfg.block_m,
            BLOCK_N=cfg.block_n,
            BLOCK_K=cfg.block_k,
            INPUT_PRECISION=cfg.precision_mode,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
        return out

    return run


def benchmark_candidate(torch: Any, runner: Callable[[], Any], reference: Any, args: argparse.Namespace) -> dict[str, Any]:
    try:
        actual = runner()
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - GPU dependent
        return {
            "status": "compile_or_launch_failed",
            "failure_reason": repr(exc),
            "correctness": {"passed": False},
            "timing": None,
        }
    correctness = base.correctness(torch, actual, reference)
    if not correctness["passed"]:
        return {
            "status": "failed_correctness",
            "failure_reason": "correctness_failed",
            "correctness": correctness,
            "timing": None,
        }
    timing = base.measure_gpu(torch, runner, args.warmup, args.iterations, args.repeats)
    return {
        "status": "completed",
        "failure_reason": None,
        "correctness": correctness,
        "timing": timing,
    }


def benchmark_workload_session(torch: Any, triton: Any, tl: Any, workload: Any, configs: list[FusedCandidateConfig],
                               ordered_configs: list[FusedCandidateConfig], args: argparse.Namespace,
                               free_bytes: int, total_bytes: int, session_seed: int) -> dict[str, Any]:
    required = base.estimate_required_bytes(workload.m, workload.n, workload.k)
    row: dict[str, Any] = {
        "workload_id": workload.workload_id,
        "category": workload.category,
        "tier": workload.tier,
        "group": getattr(workload, "group", None),
        "subgroup": getattr(workload, "subgroup", None),
        "benchmark_purpose": getattr(workload, "benchmark_purpose", "fusion_coverage"),
        "shape": {"m": workload.m, "n": workload.n, "k": workload.k, "dtype": DTYPE},
        "pattern": PATTERN,
        "bias_shape": [workload.n],
        "estimated_required_bytes": required,
        "available_gpu_memory_bytes": free_bytes,
        "total_gpu_memory_bytes": total_bytes,
        "candidate_execution_order": [cfg.config_id for cfg in ordered_configs],
        "candidate_purpose": "kernel_selection_fused_config_only",
    }
    if workload.status != "active":
        return {**row, "status": "skipped", "skip_reason": workload.skip_reason or workload.status, "candidates": {}}
    if workload.dtype != DTYPE or PATTERN not in workload.patterns:
        return {**row, "status": "skipped", "skip_reason": "unsupported_dtype_or_pattern", "candidates": {}}
    if required["total"] > int(free_bytes * args.memory_safety_fraction):
        return {**row, "status": "skipped", "skip_reason": "estimated_required_bytes_exceeds_memory_safety_fraction", "candidates": {}}

    allocate = base.make_generators(torch, session_seed, "cuda")
    a, b, bias = allocate(workload.m, workload.n, workload.k)
    if bias.shape != (workload.n,):
        raise ValueError("bias shape must be [N]")
    kernel = build_one_pass_kernel(triton, tl)
    runners = {
        cfg.config_id: make_fused_runner(torch, kernel, cfg, workload.m, workload.n, workload.k, a, b, bias)
        for cfg in configs
    }
    with torch.no_grad():
        reference = torch.relu(torch.matmul(a, b) + bias)
        torch.cuda.synchronize()
    results: dict[str, Any] = {}
    for cfg in ordered_configs:
        result = benchmark_candidate(torch, runners[cfg.config_id], reference, args)
        result.update(cfg.typed())
        results[cfg.config_id] = result
    completed = {cid: r for cid, r in results.items() if r.get("status") == "completed"}
    winner = None
    second = None
    margin = None
    near_best: list[str] = []
    if completed:
        ranked = sorted(
            completed.items(),
            key=lambda item: (item[1]["timing"]["statistics"]["median_ms"], item[0]),
        )
        winner = ranked[0][0]
        second = ranked[1][0] if len(ranked) > 1 else None
        best_ms = ranked[0][1]["timing"]["statistics"]["median_ms"]
        second_ms = ranked[1][1]["timing"]["statistics"]["median_ms"] if len(ranked) > 1 else None
        margin = ((second_ms / best_ms) - 1.0) if second_ms and best_ms > 0 else None
        near_best = [
            cid for cid, rec in ranked
            if rec["timing"]["statistics"]["median_ms"] / best_ms - 1.0 <= args.near_best_threshold
        ]
    return {
        **row,
        "status": "completed" if len(completed) == len(configs) else "incomplete",
        "skip_reason": None,
        "candidates": results,
        "oracle_config": winner,
        "second_config": second,
        "winner_margin": margin,
        "within_3_percent_configs": near_best,
    }


def winner_from_relative(best_ms: float, other_ms: float, tie_threshold: float) -> bool:
    return other_ms / best_ms - 1.0 > tie_threshold


def classify_formal(records: list[dict[str, Any]], configs: list[FusedCandidateConfig], args: argparse.Namespace) -> dict[str, Any]:
    completed = [r for r in records if r.get("status") == "completed"]
    if len(completed) != len(records) or (len(completed) < 3 and not args.smoke_test_mode):
        return {"classification": "unstable", "stability_reason": "requires_all_sessions_completed"}
    session_winners = [r["oracle_config"] for r in completed]
    winner_counts = {cfg.config_id: session_winners.count(cfg.config_id) for cfg in configs}
    medians_by_candidate: dict[str, float] = {}
    max_cv = 0.0
    for cfg in configs:
        vals = [r["candidates"][cfg.config_id]["timing"]["statistics"]["median_ms"] for r in completed]
        cvs = [r["candidates"][cfg.config_id]["timing"]["statistics"]["coefficient_of_variation"] for r in completed]
        medians_by_candidate[cfg.config_id] = statistics.median(vals)
        max_cv = max(max_cv, *(cv for cv in cvs if cv is not None))
    ranked = sorted(medians_by_candidate.items(), key=lambda item: (item[1], item[0]))
    best_id, best_ms = ranked[0]
    second_id, second_ms = ranked[1] if len(ranked) > 1 else (None, None)
    margin = second_ms / best_ms - 1.0 if second_ms else None
    near_best = [cid for cid, ms in ranked if ms / best_ms - 1.0 <= args.near_best_threshold]
    if margin is not None and margin <= args.tie_threshold:
        classification = "statistical_tie"
        reason = "cross_session_top_configs_within_tie_threshold"
    elif all(w == best_id for w in session_winners) and max_cv <= args.stable_cv_limit:
        classification = "stable_candidate_win"
        reason = "same_session_winner_and_cv_within_limit"
    else:
        classification = "unstable"
        reason = "session_winner_changes_or_cv_exceeds_limit"
    return {
        "classification": classification,
        "stability_reason": reason,
        "session_level_winners": session_winners,
        "session_level_winner_counts": winner_counts,
        "cross_session_median_ms_by_config": medians_by_candidate,
        "oracle_config": best_id,
        "second_config": second_id,
        "winner_margin": margin,
        "within_1_percent_configs": [cid for cid, ms in ranked if ms / best_ms - 1.0 <= args.tie_threshold],
        "within_3_percent_configs": near_best,
        "max_candidate_cv": max_cv,
    }


def build_sessions(torch: Any, triton: Any, tl: Any, workloads: list[Any], configs: list[FusedCandidateConfig],
                   args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sessions = []
    by_workload: dict[str, list[dict[str, Any]]] = {w.workload_id: [] for w in workloads}
    for session_index in range(args.sessions):
        session_seed = args.seed + session_index * 1009
        session = {
            "session_index": session_index,
            "seed": session_seed,
            "candidate_order_policy": args.candidate_order,
            "utc_start": datetime.now(timezone.utc).isoformat(),
            "gpu_state_before": base.gpu_state_snapshot(torch),
            "workload_results": [],
        }
        for workload_index, workload in enumerate(workloads):
            torch.cuda.empty_cache()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            order = candidate_order(configs, args, session_index, workload_index)
            record = benchmark_workload_session(
                torch, triton, tl, workload, configs, order, args, int(free_bytes), int(total_bytes), session_seed
            )
            record["session_index"] = session_index
            record["session_seed"] = session_seed
            by_workload[workload.workload_id].append(record)
            session["workload_results"].append({
                "workload_id": workload.workload_id,
                "status": record.get("status"),
                "oracle_config": record.get("oracle_config"),
                "candidate_execution_order": record.get("candidate_execution_order"),
            })
            torch.cuda.empty_cache()
        session["gpu_state_after"] = base.gpu_state_snapshot(torch)
        session["utc_end"] = datetime.now(timezone.utc).isoformat()
        sessions.append(session)

    workloads_payload = []
    for workload in workloads:
        records = by_workload[workload.workload_id]
        first = records[0]
        payload = {
            "workload_id": workload.workload_id,
            "category": workload.category,
            "tier": workload.tier,
            "group": getattr(workload, "group", None),
            "subgroup": getattr(workload, "subgroup", None),
            "benchmark_purpose": getattr(workload, "benchmark_purpose", "fusion_coverage"),
            "shape": first["shape"],
            "pattern": PATTERN,
            "bias_shape": [workload.n],
            "estimated_required_bytes": first["estimated_required_bytes"],
            "sessions": records,
        }
        if args.mode == "formal-sweep":
            payload.update(classify_formal(records, configs, args))
        else:
            payload.update({
                "classification": "coarse_observation",
                "oracle_config": first.get("oracle_config"),
                "second_config": first.get("second_config"),
                "winner_margin": first.get("winner_margin"),
                "within_3_percent_configs": first.get("within_3_percent_configs", []),
            })
        workloads_payload.append(payload)
    return sessions, workloads_payload


def prune_candidates(workloads: list[dict[str, Any]], configs: list[FusedCandidateConfig], args: argparse.Namespace) -> dict[str, Any]:
    config_ids = [cfg.config_id for cfg in configs]
    win_counts = {cid: 0 for cid in config_ids}
    stable_win_counts = {cid: 0 for cid in config_ids}
    near_counts = {cid: 0 for cid in config_ids}
    normalized_latencies = {cid: [] for cid in config_ids}
    worst_regret = {cid: 0.0 for cid in config_ids}
    for row in workloads:
        by_cfg = row.get("cross_session_median_ms_by_config")
        if not by_cfg:
            sessions = [r for r in row.get("sessions", []) if r.get("status") == "completed"]
            if not sessions:
                continue
            by_cfg = {
                cid: sessions[0]["candidates"][cid]["timing"]["statistics"]["median_ms"]
                for cid in config_ids
                if sessions[0]["candidates"].get(cid, {}).get("status") == "completed"
            }
        if not by_cfg:
            continue
        best = min(by_cfg.values())
        winner = min(by_cfg.items(), key=lambda item: (item[1], item[0]))[0]
        win_counts[winner] += 1
        if row.get("classification") == "stable_candidate_win":
            stable_win_counts[winner] += 1
        for cid, ms in by_cfg.items():
            norm = ms / best
            normalized_latencies[cid].append(norm)
            worst_regret[cid] = max(worst_regret[cid], norm - 1.0)
            if norm - 1.0 <= args.near_best_threshold:
                near_counts[cid] += 1
    candidate_summary = {}
    retained = []
    dominated = []
    for cid in config_ids:
        values = normalized_latencies[cid]
        mean_norm = statistics.fmean(values) if values else None
        strictly_dominated = win_counts[cid] == 0 and near_counts[cid] == 0
        over_5_slower = values and all(v - 1.0 > args.dominated_threshold for v in values)
        status = "dominated" if strictly_dominated or over_5_slower else "retained"
        if status == "retained":
            retained.append(cid)
        else:
            dominated.append(cid)
        candidate_summary[cid] = {
            "wins": win_counts[cid],
            "stable_wins": stable_win_counts[cid],
            "ties_or_near_best": near_counts[cid],
            "mean_normalized_latency": mean_norm,
            "worst_regret": worst_regret[cid],
            "pruning_status": status,
            "pruning_reason": "never_best_or_near_best" if strictly_dominated else ("always_more_than_threshold_slower" if over_5_slower else None),
        }
    return {
        "candidate_summary": candidate_summary,
        "retained_candidate_set": retained,
        "recommended_primary_candidate_set": [
            cid for cid in retained if win_counts[cid] > 0
        ][:4],
        "dominated_candidates": dominated,
        "winner_histogram": win_counts,
        "stable_winner_histogram": stable_win_counts,
        "near_best_histogram": near_counts,
        "genuine_selection_diversity": sum(1 for count in win_counts.values() if count > 0) >= 2,
    }


def build_payload(args: argparse.Namespace, env: dict[str, Any], configs: list[FusedCandidateConfig],
                  sessions: list[dict[str, Any]], workloads: list[dict[str, Any]], started: str) -> dict[str, Any]:
    pruning = prune_candidates(workloads, configs, args)
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "backend": BACKEND,
        "kernel_family_id": KERNEL_FAMILY_ID,
        "candidate_purpose": "kernel_selection_fused_config_only",
        "fusion_attribution_candidate_set": ["V1", "V3"],
        "selection_candidate_set": [cfg.typed() for cfg in configs],
        "environment": env,
        "benchmark_config": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "sessions": args.sessions,
            "seed": args.seed,
            "candidate_order": args.candidate_order,
            "tie_threshold": args.tie_threshold,
            "near_best_threshold": args.near_best_threshold,
            "stable_cv_limit": args.stable_cv_limit,
            "memory_safety_fraction": args.memory_safety_fraction,
        },
        "workload_manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "sessions": sessions,
        "workloads": workloads,
        "aggregate": pruning,
        "truth_boundary": "measured_fused_config_candidate_diversity_not_runtime_autotuning",
        "utc_start": started,
        "utc_end": datetime.now(timezone.utc).isoformat(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# Triton Fused MatMul-Bias-ReLU Candidate Diversity",
        "",
        "This report evaluates one-pass fused Triton configurations only. V1/V3 fusion attribution remains separate.",
        "",
        "## Candidate Configurations",
        "",
        "| Candidate | BLOCK_M | BLOCK_N | BLOCK_K | Warps | Stages | Hypothesis |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for cfg in payload["selection_candidate_set"]:
        lines.append(
            f"| {cfg['config_id']} | {cfg['block_m']} | {cfg['block_n']} | {cfg['block_k']} | "
            f"{cfg['num_warps']} | {cfg['num_stages']} | {cfg['hypothesis']} |"
        )
    lines += [
        "",
        "## Candidate Summary",
        "",
        "| Candidate | Stable wins | Ties/near-best | Mean normalized latency | Worst regret | Status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for cid, row in agg["candidate_summary"].items():
        mean_norm = row["mean_normalized_latency"]
        mean_text = f"{mean_norm:.4f}" if mean_norm is not None else "n/a"
        lines.append(
            f"| {cid} | {row['stable_wins']} | {row['ties_or_near_best']} | "
            f"{mean_text} | {row['worst_regret']:.4f} | {row['pruning_status']} |"
        )
    lines += [
        "",
        "## Winner Histogram",
        "",
    ]
    for cid, count in agg["winner_histogram"].items():
        lines.append(f"- `{cid}`: `{count}`")
    lines += [
        "",
        "## Per-Workload Results",
        "",
        "| Workload | M | N | K | Oracle config | Second config | Margin | Stable? |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for row in payload["workloads"]:
        shape = row["shape"]
        margin = row.get("winner_margin")
        lines.append(
            f"| {row['workload_id']} | {shape['m']} | {shape['n']} | {shape['k']} | "
            f"{row.get('oracle_config')} | {row.get('second_config')} | "
            f"{margin if margin is not None else 0.0:.4f} | {row.get('classification')} |"
        )
    lines += [
        "",
        "## Recommended Primary Candidate Set",
        "",
        ", ".join(f"`{cid}`" for cid in agg.get("recommended_primary_candidate_set", [])) or "None",
        "",
        "## Retained Candidate Set",
        "",
        ", ".join(f"`{cid}`" for cid in agg["retained_candidate_set"]) or "None",
        "",
        "## Dominated Candidates",
        "",
        ", ".join(f"`{cid}`" for cid in agg["dominated_candidates"]) or "None",
        "",
        "## Interpretation",
        "",
        f"Genuine selection diversity: `{agg['genuine_selection_diversity']}`.",
        "If one configuration still dominates, a specialized M=1 or split-K candidate should be proposed separately with evidence.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def unavailable_payload(reason: str, args: argparse.Namespace, started: str, env: dict[str, Any],
                        configs: list[FusedCandidateConfig]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "backend": BACKEND,
        "profile_status": "unavailable",
        "unavailable_reason": reason,
        "candidate_purpose": "kernel_selection_fused_config_only",
        "fusion_attribution_candidate_set": ["V1", "V3"],
        "selection_candidate_set": [cfg.typed() for cfg in configs],
        "environment": env,
        "workloads": [],
        "aggregate": {},
        "utc_start": started,
        "utc_end": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    configs = selected_candidates(args)
    workloads = select_workloads(args)
    started = datetime.now(timezone.utc).isoformat()
    torch, triton, tl, import_error = base.import_torch_triton()
    env = base.environment_metadata(torch, triton)
    if import_error:
        payload = unavailable_payload(import_error, args, started, env, configs)
        write_json(Path(args.output), payload)
        write_report(Path(args.report_output), payload) if "aggregate" in payload and payload["aggregate"] else None
        print(json.dumps(payload, indent=2))
        return 0
    if not torch.cuda.is_available():
        payload = unavailable_payload("CUDA is not available", args, started, env, configs)
        write_json(Path(args.output), payload)
        print(json.dumps(payload, indent=2))
        return 0
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    sessions, rows = build_sessions(torch, triton, tl, workloads, configs, args)
    payload = build_payload(args, env, configs, sessions, rows, started)
    write_json(Path(args.output), payload)
    write_report(Path(args.report_output), payload)
    write_report(Path(args.doc_output), payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
