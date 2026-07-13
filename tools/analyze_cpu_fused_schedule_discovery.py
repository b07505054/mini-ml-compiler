#!/usr/bin/env python3
"""Phase 1 analysis layer: consumes real measurements from
run_cpu_fused_schedule_discovery and derives oracle winners, winner regions,
static-policy regret, and the discovery summary/report.

This script does not run any benchmark and does not invent any latency
value. Every number here is a deterministic function of
benchmark_measurements.json, which itself is real measured data.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matmul_postop_workloads import percentile  # noqa: E402

CLASSIFICATIONS = (
    "stable_winner",
    "near_tie",
    "noisy_inconclusive",
    "candidate_failure",
    "correctness_failure",
)


def noise_threshold_pct(best_cv: float, second_cv: float) -> float:
    """Margin must clear this to be called a stable winner.

    Formula: max(2.0%, 3x the larger of the top-two candidates' CV). This is
    a documented, fixed rule — not tuned per workload — so a 0.1% difference
    under 1% run-to-run noise is never reported as a decision boundary.
    """
    return max(2.0, 3.0 * max(best_cv, second_cv) * 100.0)


def classify_workload(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if any(not c["correctness"]["passed"] for c in candidates):
        failing = [c["candidate_id"] for c in candidates if not c["correctness"]["passed"]]
        return {
            "classification": "correctness_failure",
            "failing_candidates": failing,
            "oracle_winner": None,
        }

    ranked = sorted(candidates, key=lambda c: c["stats"]["mean_ms"])
    best, second = ranked[0], ranked[1]
    margin_pct = (
        (second["stats"]["mean_ms"] - best["stats"]["mean_ms"]) / best["stats"]["mean_ms"] * 100.0
        if best["stats"]["mean_ms"] > 0 else 0.0
    )
    threshold = noise_threshold_pct(
        best["stats"]["coefficient_of_variation"], second["stats"]["coefficient_of_variation"]
    )
    if margin_pct >= threshold:
        classification = "stable_winner"
    elif margin_pct >= threshold / 2.0:
        classification = "near_tie"
    else:
        classification = "noisy_inconclusive"

    return {
        "classification": classification,
        "oracle_winner": best["candidate_id"],
        "oracle_winner_mean_ms": best["stats"]["mean_ms"],
        "second_best_candidate": second["candidate_id"],
        "second_best_mean_ms": second["stats"]["mean_ms"],
        "margin_pct": margin_pct,
        "noise_threshold_pct": threshold,
        "best_cv": best["stats"]["coefficient_of_variation"],
        "second_cv": second["stats"]["coefficient_of_variation"],
        "ranked_candidate_ids": [c["candidate_id"] for c in ranked],
    }


def build_oracle_winners(measurements: dict[str, Any]) -> dict[str, Any]:
    records = []
    for wl in measurements["workloads"]:
        result = classify_workload(wl["candidates"])
        record = {
            "workload_id": wl["workload_id"],
            "family": wl["family"],
            "m": wl["m"], "n": wl["n"], "k": wl["k"],
            "flops": wl["flops"],
            **result,
        }
        records.append(record)
    return {
        "schema": "cpu_fused_schedule_oracle_winners",
        "schema_version": 1,
        "noise_rule": "stable_winner requires margin_pct >= max(2.0, 3 * max(best_cv, second_cv) * 100)",
        "records": records,
    }


def build_winner_regions(oracle_winners: dict[str, Any], candidate_ids: list[str]) -> dict[str, Any]:
    by_family: dict[str, Any] = {}
    for record in oracle_winners["records"]:
        family = record["family"]
        entry = by_family.setdefault(family, {
            "workload_count": 0,
            "stable_winner_counts": {cid: 0 for cid in candidate_ids},
            "any_winner_counts": {cid: 0 for cid in candidate_ids},
            "classification_counts": {c: 0 for c in CLASSIFICATIONS},
        })
        entry["workload_count"] += 1
        entry["classification_counts"][record["classification"]] += 1
        if record.get("oracle_winner"):
            entry["any_winner_counts"][record["oracle_winner"]] += 1
            if record["classification"] == "stable_winner":
                entry["stable_winner_counts"][record["oracle_winner"]] += 1

    overall_stable = {cid: 0 for cid in candidate_ids}
    overall_any = {cid: 0 for cid in candidate_ids}
    for record in oracle_winners["records"]:
        if record.get("oracle_winner"):
            overall_any[record["oracle_winner"]] += 1
            if record["classification"] == "stable_winner":
                overall_stable[record["oracle_winner"]] += 1

    candidates_with_stable_win = [cid for cid, count in overall_stable.items() if count > 0]
    total_stable = sum(overall_stable.values())
    dominant_candidate = max(overall_stable, key=overall_stable.get) if total_stable > 0 else None
    dominant_share_pct = (
        100.0 * overall_stable[dominant_candidate] / total_stable
        if dominant_candidate and total_stable > 0 else 0.0
    )

    return {
        "schema": "cpu_fused_schedule_winner_regions",
        "schema_version": 1,
        "by_family": by_family,
        "overall_stable_winner_counts": overall_stable,
        "overall_any_winner_counts": overall_any,
        "candidates_with_at_least_one_stable_win": candidates_with_stable_win,
        "distinct_stable_winner_count": len(candidates_with_stable_win),
        "dominant_candidate": dominant_candidate,
        "dominant_candidate_stable_share_pct": dominant_share_pct,
    }


def build_static_policy_comparison(measurements: dict[str, Any], candidate_ids: list[str]) -> dict[str, Any]:
    per_policy: dict[str, Any] = {}
    usable_workloads = [
        wl for wl in measurements["workloads"]
        if all(c["correctness"]["passed"] for c in wl["candidates"])
    ]
    excluded = len(measurements["workloads"]) - len(usable_workloads)

    for policy_id in candidate_ids:
        regrets = []
        for wl in usable_workloads:
            by_id = {c["candidate_id"]: c for c in wl["candidates"]}
            oracle_mean = min(c["stats"]["mean_ms"] for c in wl["candidates"])
            policy_mean = by_id[policy_id]["stats"]["mean_ms"]
            regret = (policy_mean - oracle_mean) / oracle_mean if oracle_mean > 0 else 0.0
            regrets.append(regret)
        per_policy[policy_id] = {
            "workload_count": len(regrets),
            "mean_regret": statistics.fmean(regrets) if regrets else 0.0,
            "median_regret": statistics.median(regrets) if regrets else 0.0,
            "p95_regret": percentile(regrets, 95) if regrets else 0.0,
            "max_regret": max(regrets) if regrets else 0.0,
        }

    return {
        "schema": "cpu_fused_schedule_static_policy_comparison",
        "schema_version": 1,
        "note": "Regret of ALWAYS selecting one fixed candidate for every workload, "
                "vs the oracle (measured best candidate) for that workload. "
                "Not a learned or calibrated selector — Phase 1 does not build one.",
        "excluded_correctness_failure_workloads": excluded,
        "policies": per_policy,
    }


def build_summary(
    measurements: dict[str, Any],
    oracle_winners: dict[str, Any],
    winner_regions: dict[str, Any],
    static_policy: dict[str, Any],
    environment: dict[str, Any],
    candidate_contract: dict[str, Any],
    plan_dispatch_validation: dict[str, Any],
) -> dict[str, Any]:
    classification_counts = {c: 0 for c in CLASSIFICATIONS}
    for record in oracle_winners["records"]:
        classification_counts[record["classification"]] += 1

    distinct_stable = winner_regions["distinct_stable_winner_count"]
    dominant_share = winner_regions["dominant_candidate_stable_share_pct"]

    if distinct_stable >= 2 and dominant_share < 95.0:
        verdict = "SUCCESS"
    elif dominant_share >= 95.0 and classification_counts["stable_winner"] > 0:
        verdict = "FAILED_FOUNDATION"
    elif classification_counts["stable_winner"] == 0:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "SUCCESS"

    any_correctness_failure = classification_counts["correctness_failure"] > 0

    return {
        "schema": "cpu_fused_schedule_discovery_summary",
        "schema_version": 1,
        "total_workloads": len(oracle_winners["records"]),
        "classification_counts": classification_counts,
        "distinct_candidates_with_stable_win": distinct_stable,
        "candidates_with_at_least_one_stable_win": winner_regions["candidates_with_at_least_one_stable_win"],
        "dominant_candidate": winner_regions["dominant_candidate"],
        "dominant_candidate_stable_share_pct": dominant_share,
        "any_correctness_failure": any_correctness_failure,
        "plan_dispatch_total_override_count": plan_dispatch_validation.get("total_override_count"),
        "fusion_attribution_baseline": measurements.get("fusion_attribution_baseline"),
        "environment_summary": {
            "cpu_model": environment["cpu_model"]["value"],
            "os": environment["os"]["value"],
            "arch": environment["arch"]["value"],
            "compiler": environment["compiler"]["value"],
            "benchmark_thread_count": environment["benchmark_thread_count"]["value"],
        },
        "candidate_count": len(candidate_contract["candidates"]),
        "phase1_verdict": verdict,
        "serial_execution_caveat": (
            "This validates serial CPU tiling behavior, not multicore hardware-capacity "
            "scheduling. benchmark_thread_count is fixed at 1 for all candidates."
        ),
    }


def write_report(
    path: Path,
    measurements: dict[str, Any],
    oracle_winners: dict[str, Any],
    winner_regions: dict[str, Any],
    static_policy: dict[str, Any],
    summary: dict[str, Any],
    environment: dict[str, Any],
    plan_dispatch_validation: dict[str, Any],
) -> None:
    lines = [
        "# CPU Fused Schedule Candidate Discovery — Phase 1 Report",
        "",
        "Real CPU measurements from `run_cpu_fused_schedule_discovery`. No cache, SIMD, "
        "or latency value in this report is fabricated; every figure traces back to "
        "`benchmark_measurements.json`.",
        "",
        "## Environment",
        f"- CPU: `{environment['cpu_model']['value']}` (source: {environment['cpu_model']['source']})",
        f"- OS: `{environment['os']['value']}`, arch `{environment['arch']['value']}`",
        f"- Compiler: `{environment['compiler']['value']}`",
        f"- Build type: `{environment['build_type']['value']}`",
        f"- Physical cores: `{environment['physical_cores_total']['value']}` "
        f"(P: {environment['performance_core_count']['value']}, "
        f"E: {environment['efficiency_core_count']['value']})",
        f"- Benchmark thread count: `{environment['benchmark_thread_count']['value']}` "
        f"({summary['serial_execution_caveat']})",
        "",
        "## Phase 1 Verdict",
        f"**{summary['phase1_verdict']}**",
        "",
        "## Classification Summary",
        "| Classification | Count |",
        "| --- | ---: |",
    ]
    for c, count in summary["classification_counts"].items():
        lines.append(f"| {c} | {count} |")
    lines += [
        "",
        f"- Distinct candidates with >=1 stable win: `{summary['distinct_candidates_with_stable_win']}` "
        f"({', '.join(summary['candidates_with_at_least_one_stable_win']) or 'none'})",
        f"- Dominant candidate: `{summary['dominant_candidate']}` "
        f"({summary['dominant_candidate_stable_share_pct']:.1f}% of stable wins)",
        "",
        "## Winner Regions By Family",
        "| Family | Workloads | Stable-win counts by candidate |",
        "| --- | ---: | --- |",
    ]
    for family, entry in winner_regions["by_family"].items():
        counts = ", ".join(f"{cid}={c}" for cid, c in entry["stable_winner_counts"].items() if c > 0) or "none stable"
        lines.append(f"| {family} | {entry['workload_count']} | {counts} |")
    lines += [
        "",
        "## Static Policy Regret (always pick one fixed candidate vs oracle)",
        "| Policy | N | Mean regret | Median regret | P95 regret | Max regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy_id, stats in static_policy["policies"].items():
        lines.append(
            f"| {policy_id} | {stats['workload_count']} | {stats['mean_regret']*100:.3f}% | "
            f"{stats['median_regret']*100:.3f}% | {stats['p95_regret']*100:.3f}% | "
            f"{stats['max_regret']*100:.3f}% |"
        )
    fusion = measurements.get("fusion_attribution_baseline", {})
    lines += [
        "",
        "## Fusion Attribution Baseline (SEPARATE from schedule oracle)",
        f"- Shape: `{fusion.get('shape')}`, tile config: `{fusion.get('tile_config')}`",
        f"- Unfused (3 launches, 2 full intermediates): mean `{fusion.get('unfused', {}).get('mean_ms'):.5f} ms`, "
        f"correctness passed: `{fusion.get('unfused', {}).get('correctness_passed')}`",
        f"- Fused (1 launch, 0 full intermediates): mean `{fusion.get('fused', {}).get('mean_ms'):.5f} ms`, "
        f"correctness passed: `{fusion.get('fused', {}).get('correctness_passed')}`",
        f"- Fusion speedup: `{fusion.get('fusion_speedup'):.3f}x` "
        f"(`{fusion.get('latency_reduction_percent'):.1f}%` latency reduction)",
        "",
        "## Plan/Dispatch Validation",
        f"- Total override count: `{plan_dispatch_validation.get('total_override_count')}` (expected 0)",
        "| Candidate | Plan matched runtime |",
        "| --- | --- |",
    ]
    for entry in plan_dispatch_validation.get("validations", []):
        lines.append(f"| {entry['candidate_id']} | {entry['plan_matched_runtime']} |")
    lines += [
        "",
        "## Per-Workload Detail",
        "| Workload | Family | Winner | Margin % | Noise threshold % | Classification |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for record in oracle_winners["records"]:
        if record["classification"] == "correctness_failure":
            lines.append(f"| {record['workload_id']} | {record['family']} | - | - | - | correctness_failure |")
            continue
        lines.append(
            f"| {record['workload_id']} | {record['family']} | {record['oracle_winner']} | "
            f"{record['margin_pct']:.2f} | {record['noise_threshold_pct']:.2f} | {record['classification']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-dir", default="trace/cpu_fused_schedule_discovery")
    args = parser.parse_args()
    directory = Path(args.discovery_dir)

    measurements = json.loads((directory / "benchmark_measurements.json").read_text())
    environment = json.loads((directory / "environment.json").read_text())
    candidate_contract = json.loads((directory / "candidate_contract.json").read_text())
    plan_dispatch_validation = json.loads((directory / "plan_dispatch_validation.json").read_text())
    candidate_ids = [c["candidate_id"] for c in candidate_contract["candidates"]]

    oracle_winners = build_oracle_winners(measurements)
    winner_regions = build_winner_regions(oracle_winners, candidate_ids)
    static_policy = build_static_policy_comparison(measurements, candidate_ids)
    summary = build_summary(
        measurements, oracle_winners, winner_regions, static_policy,
        environment, candidate_contract, plan_dispatch_validation,
    )

    (directory / "oracle_winners.json").write_text(json.dumps(oracle_winners, indent=2), encoding="utf-8")
    (directory / "winner_regions.json").write_text(json.dumps(winner_regions, indent=2), encoding="utf-8")
    (directory / "static_policy_comparison.json").write_text(json.dumps(static_policy, indent=2), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(
        directory / "report.md", measurements, oracle_winners, winner_regions,
        static_policy, summary, environment, plan_dispatch_validation,
    )

    print(f"Phase 1 verdict: {summary['phase1_verdict']}")
    print(f"Distinct stable-winner candidates: {summary['distinct_candidates_with_stable_win']}")
    print(f"Dominant candidate: {summary['dominant_candidate']} "
          f"({summary['dominant_candidate_stable_share_pct']:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
