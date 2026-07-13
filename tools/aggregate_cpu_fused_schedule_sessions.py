#!/usr/bin/env python3
"""Phase R1: aggregate >=3 independent CPU schedule discovery sessions.

Two outputs:
  1. A pooled benchmark_measurements.json (samples_ms concatenated across
     all sessions per workload/candidate, stats recomputed over the pooled
     set) written to the parent trace directory — this becomes the input
     to analyze_cpu_fused_schedule_discovery.py for the "official" R1
     oracle_winners/winner_regions/static_policy_comparison/summary/report.
  2. session_summary.json — each session's INDEPENDENT verdict computed
     separately, plus a cross-session agreement check: does the same
     candidate win the same workload in every session that calls it a
     stable winner? This is the actual "repeatable across sessions" check
     the spec requires, distinct from noise reduction via pooling.

No new latency values are invented — every number here is a deterministic
function of the sessions' own real measurements.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_cpu_fused_schedule_discovery as analysis  # noqa: E402


def summarize(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"mean_ms": 0.0, "median_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0,
                "stddev_ms": 0.0, "coefficient_of_variation": 0.0}
    mean = statistics.fmean(samples)
    stddev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "mean_ms": mean,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stddev_ms": stddev,
        "coefficient_of_variation": (stddev / mean) if mean > 0 else 0.0,
    }


def pool_measurements(sessions: list[dict[str, Any]], session_dirs: list[str]) -> dict[str, Any]:
    base = sessions[0]
    pooled_workloads = []
    for wl_idx, wl in enumerate(base["workloads"]):
        pooled_candidates = []
        for cand_idx, cand in enumerate(wl["candidates"]):
            pooled_samples: list[float] = []
            all_passed = True
            max_abs = 0.0
            max_rel = 0.0
            for session in sessions:
                s_cand = session["workloads"][wl_idx]["candidates"][cand_idx]
                assert s_cand["candidate_id"] == cand["candidate_id"], "candidate order mismatch across sessions"
                pooled_samples.extend(s_cand["samples_ms"])
                all_passed = all_passed and s_cand["correctness"]["passed"]
                max_abs = max(max_abs, s_cand["correctness"]["max_abs_error"])
                max_rel = max(max_rel, s_cand["correctness"]["max_rel_error"])
            pooled_candidates.append({
                "candidate_id": cand["candidate_id"],
                "correctness": {
                    "passed": all_passed, "max_abs_error": max_abs, "max_rel_error": max_rel,
                    "contains_nan": False, "contains_inf": False,
                },
                "stats": summarize(pooled_samples),
                "samples_ms": pooled_samples,
                "pooled_from_sessions": len(sessions),
            })
        pooled_workloads.append({
            "workload_id": wl["workload_id"], "family": wl["family"],
            "m": wl["m"], "n": wl["n"], "k": wl["k"], "flops": wl["flops"],
            "budget": wl["budget"], "candidates": pooled_candidates,
        })

    return {
        "schema": "cpu_fused_schedule_benchmark_measurements",
        "schema_version": 1,
        "provenance": base["provenance"],
        "pooled_from_session_dirs": session_dirs,
        "pooling_note": "samples_ms per (workload, candidate) is the concatenation of all "
                        "session repeats; stats recomputed over the pooled set. This increases "
                        "the repeat count for noise estimation but does NOT by itself prove "
                        "cross-session repeatability -- see session_summary.json for that.",
        "timing_methodology": base["timing_methodology"],
        "input_distribution": base["input_distribution"],
        "correctness_tolerance": base["correctness_tolerance"],
        "candidate_order_policy": base["candidate_order_policy"],
        "fusion_attribution_baseline": base["fusion_attribution_baseline"],
        "workloads": pooled_workloads,
    }


def build_session_summary(session_dirs: list[Path], candidate_ids: list[str]) -> dict[str, Any]:
    per_session = []
    per_session_oracle: list[dict[str, Any]] = []
    for d in session_dirs:
        measurements = json.loads((d / "benchmark_measurements.json").read_text())
        oracle_winners = analysis.build_oracle_winners(measurements)
        regions = analysis.build_winner_regions(oracle_winners, candidate_ids)
        per_session_oracle.append({r["workload_id"]: r for r in oracle_winners["records"]})
        per_session.append({
            "session_dir": str(d),
            "provenance": measurements.get("provenance"),
            "distinct_stable_winner_count": regions["distinct_stable_winner_count"],
            "candidates_with_at_least_one_stable_win": regions["candidates_with_at_least_one_stable_win"],
            "dominant_candidate": regions["dominant_candidate"],
            "dominant_candidate_stable_share_pct": regions["dominant_candidate_stable_share_pct"],
            "overall_stable_winner_counts": regions["overall_stable_winner_counts"],
        })

    # Cross-session agreement: for each workload, among sessions where the
    # classification was stable_winner, do they all name the same winner?
    agreement_records = []
    all_workload_ids = list(per_session_oracle[0].keys())
    for wid in all_workload_ids:
        stable_winners_by_session = [
            oracle[wid]["oracle_winner"] for oracle in per_session_oracle
            if oracle[wid]["classification"] == "stable_winner"
        ]
        if not stable_winners_by_session:
            status = "no_session_called_it_stable"
        elif len(set(stable_winners_by_session)) == 1:
            status = "agree" if len(stable_winners_by_session) == len(session_dirs) else "partial_stable_agree"
        else:
            status = "CONFLICT_different_winners_across_sessions"
        agreement_records.append({
            "workload_id": wid,
            "sessions_calling_it_stable": len(stable_winners_by_session),
            "distinct_winners_among_stable_sessions": sorted(set(stable_winners_by_session)),
            "status": status,
        })

    conflicts = [r for r in agreement_records if r["status"] == "CONFLICT_different_winners_across_sessions"]

    return {
        "schema": "cpu_fused_schedule_session_summary",
        "schema_version": 1,
        "session_count": len(session_dirs),
        "per_session_independent_verdicts": per_session,
        "cross_session_agreement": agreement_records,
        "cross_session_conflict_count": len(conflicts),
        "cross_session_conflicts": conflicts,
        "note": "A workload is trustworthy evidence of a real decision boundary only if it is "
                "stable_winner in multiple sessions AND all such sessions name the same winner. "
                "cross_session_conflict_count > 0 would mean apparent stability did not "
                "replicate and must not be reported as a real boundary.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", action="append", required=True,
                        help="Repeatable: path to one session's discovery output directory")
    parser.add_argument("--output-dir", required=True,
                        help="Parent directory to write pooled benchmark_measurements.json "
                             "and session_summary.json")
    args = parser.parse_args()

    if len(args.session_dir) < 3:
        print(f"FAIL: R1 requires at least 3 independent sessions, got {len(args.session_dir)}",
              file=sys.stderr)
        return 1

    session_dirs = [Path(d) for d in args.session_dir]
    sessions = [json.loads((d / "benchmark_measurements.json").read_text()) for d in session_dirs]

    contract = json.loads((session_dirs[0] / "candidate_contract.json").read_text())
    candidate_ids = [c["candidate_id"] for c in contract["candidates"]]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pooled = pool_measurements(sessions, [str(d) for d in session_dirs])
    (output_dir / "benchmark_measurements.json").write_text(json.dumps(pooled, indent=2), encoding="utf-8")

    for name in ("environment.json", "candidate_contract.json", "workload_manifest.json"):
        (output_dir / name).write_text((session_dirs[0] / name).read_text(), encoding="utf-8")

    session_summary = build_session_summary(session_dirs, candidate_ids)
    (output_dir / "session_summary.json").write_text(json.dumps(session_summary, indent=2), encoding="utf-8")

    print(f"Pooled {len(session_dirs)} sessions into {output_dir}")
    print(f"Cross-session conflicts: {session_summary['cross_session_conflict_count']}")
    for s in session_summary["per_session_independent_verdicts"]:
        print(f"  {s['session_dir']}: dominant={s['dominant_candidate']} "
              f"distinct_stable={s['distinct_stable_winner_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
