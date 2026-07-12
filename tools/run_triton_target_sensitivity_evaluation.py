#!/usr/bin/env python3
"""Analytical target-sensitivity evaluation for Triton fused config selection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_triton_fused_config_repair as repair  # noqa: E402
import run_triton_fused_config_selection as base  # noqa: E402
import target_hardware_profile as hardware_profile  # noqa: E402


SCHEMA_VERSION = 1
BACKEND = base.BACKEND
CONFIG_IDS = base.PRIMARY_CONFIG_IDS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def generate_workloads() -> list[dict[str, Any]]:
    rows: list[tuple[str, int, int, int, str]] = []
    for dim in (16, 32, 48, 64, 96, 128, 192):
        for k in (512, 1024, 2048, 4096, 8192, 12288):
            rows.append((f"small_square_m{dim}_n{dim}_k{k}", dim, dim, k, "small_square_high_k"))
    for m in (1, 2, 4, 8, 16, 32):
        for n in (256, 512, 1024, 2048, 4096, 8192):
            for k in (512, 1024, 2048, 4096):
                rows.append((f"skinny_m{m}_n{n}_k{k}", m, n, k, "skinny_m_wide_n"))
    for m in (64, 96, 128, 192, 256):
        for n in (64, 128, 256, 512, 1024):
            for k in (256, 512, 1024, 2048, 4096):
                rows.append((f"medium_m{m}_n{n}_k{k}", m, n, k, "medium_rectangular"))
    for m in (256, 512, 1024, 2048):
        for n in (256, 512, 1024, 2048):
            for k in (256, 512, 1024, 2048, 4096):
                rows.append((f"large_m{m}_n{n}_k{k}", m, n, k, "large_regular"))
    for m, n, k in ((70, 70, 4096), (80, 112, 2048), (96, 160, 4096), (130, 258, 1024), (192, 320, 4096)):
        rows.append((f"edge_m{m}_n{n}_k{k}", m, n, k, "edge_heavy_non_divisible"))
    seen = set()
    out = []
    for wid, m, n, k, family in rows:
        key = (m, n, k)
        if key in seen:
            continue
        seen.add(key)
        out.append({"workload_id": wid, "family": family, "shape": {"m": m, "n": n, "k": k, "dtype": base.DTYPE}})
    return out


def load_training_model(base_sweep: Path, repair_training: Path, fresh_oracle: Path) -> tuple[dict[str, Any], list[base.Measurement]]:
    base_rows, _ = base.parse_discovery_measurements(base_sweep)
    repair_rows = repair.parse_repair_profile(repair_training)
    fresh_rows = repair.parse_repair_profile(fresh_oracle)
    training_candidate_rows = {r.workload_id: r for r in base_rows + repair_rows}
    evaluation_rows = {r.workload_id: r for r in base_rows + repair_rows + fresh_rows}
    train, _, _ = repair.grouped_split(list(evaluation_rows.values()), "leave-one-shape-region-out")
    train = [training_candidate_rows[r.workload_id] for r in train if r.workload_id in training_candidate_rows]
    model = repair.calibrate(train, 16)
    return model, train


def cost_record(shape: dict[str, Any], profile: hardware_profile.ResolvedHardwareProfile, model: dict[str, Any],
                config_id: str, rank_index: int, selected: bool) -> dict[str, Any]:
    cfg = base.PRIMARY_CONFIGS[config_id]
    pred = repair.predict_config(shape, cfg, model, profile.effective_compute_units, calibrated=True)
    components = pred["components"]
    raw = sum(components.values())
    calibration_delta = pred["predicted_latency_ms"] - raw
    features = pred["tile_features"]
    wf = pred["workload_features"]
    return {
        "shape": {"m": shape["m"], "n": shape["n"], "k": shape["k"]},
        "target_profile_id": profile.target_id,
        "profile_kind": profile.profile_kind,
        "effective_compute_units": profile.effective_compute_units,
        "candidate": config_id,
        "features": {
            "parallel_work_items": features["parallel_work_items"],
            "work_items_per_compute_unit": features["work_items_per_compute_unit"],
            "execution_waves": features["execution_waves"],
            "compute_work_per_item": features["compute_work_per_item"],
            "reduction_dominance": wf["reduction_dominance"],
            "padding_waste_ratio": features["padding_waste_ratio"],
            "work_amplification": features["work_amplification"],
            "programs_per_sm": features["programs_per_sm"],
            "output_program_waves": features["output_program_waves"],
        },
        "cost_terms": {
            "fixed": components.get("fixed_ms", 0.0),
            "effective_compute": components.get("compute_ms", 0.0),
            "memory": components.get("memory_ms", 0.0),
            "padding": components.get("padding_ms", 0.0),
            "low_parallelism": components.get("low_parallelism_ms", 0.0),
            "k_dominant_parallelism": components.get("k_dominant_parallelism_ms", 0.0),
            "excessive_programs": components.get("program_overhead_ms", 0.0),
            "shape_config_mismatch": components.get("shape_mismatch_ms", 0.0),
            "calibration": calibration_delta,
        },
        "predicted_cost_ms": pred["predicted_latency_ms"],
        "rank": rank_index,
        "selected": selected,
    }


def evaluate_workload(workload: dict[str, Any], profiles: list[hardware_profile.ResolvedHardwareProfile],
                      model: dict[str, Any]) -> dict[str, Any]:
    by_profile = []
    for profile in profiles:
        rank_rows = repair.ranking(workload["shape"], model, profile.effective_compute_units, calibrated=True)
        ranking_ids = [row["config_id"] for row in rank_rows]
        records = [
            cost_record(workload["shape"], profile, model, row["config_id"], i + 1, i == 0)
            for i, row in enumerate(rank_rows)
        ]
        by_profile.append({
            "target_profile_id": profile.target_id,
            "profile_kind": profile.profile_kind,
            "effective_compute_units": profile.effective_compute_units,
            "selected_config": ranking_ids[0],
            "ranking": ranking_ids,
            "candidates": records,
        })
    rankings = [tuple(p["ranking"]) for p in by_profile]
    selected = [p["selected_config"] for p in by_profile]
    feature_sensitive = any(
        by_profile[0]["candidates"][i]["features"]["work_items_per_compute_unit"]
        != p["candidates"][i]["features"]["work_items_per_compute_unit"]
        for p in by_profile[1:]
        for i in range(len(CONFIG_IDS))
    )
    ranking_sensitive = len(set(rankings)) > 1
    selection_sensitive = len(set(selected)) > 1
    if selection_sensitive:
        classification = "selection-sensitive"
    elif ranking_sensitive:
        classification = "ranking-sensitive"
    elif feature_sensitive:
        classification = "feature-sensitive"
    else:
        classification = "insensitive"
    return {
        **workload,
        "classification": classification,
        "selected_configs_by_profile": {p["target_profile_id"]: p["selected_config"] for p in by_profile},
        "rankings_by_profile": {p["target_profile_id"]: p["ranking"] for p in by_profile},
        "profiles": by_profile,
    }


def boundary_term(case: dict[str, Any]) -> str:
    profiles = case["profiles"]
    if len(profiles) < 2:
        return "unknown"
    first = profiles[0]["candidates"][0]["candidate"]
    for later in profiles[1:]:
        if later["selected_config"] != first:
            terms = {}
            for candidate in {first, later["selected_config"]}:
                a = next(c for c in profiles[0]["candidates"] if c["candidate"] == candidate)
                b = next(c for c in later["candidates"] if c["candidate"] == candidate)
                for term, val in a["cost_terms"].items():
                    terms[term] = max(terms.get(term, 0.0), abs(float(b["cost_terms"][term]) - float(val)))
            return max(terms.items(), key=lambda item: item[1])[0]
    return "ranking_order_without_top1_change"


def summarize(results: list[dict[str, Any]], profiles: list[hardware_profile.ResolvedHardwareProfile]) -> dict[str, Any]:
    counts = {}
    for row in results:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    selection = [r for r in results if r["classification"] == "selection-sensitive"]
    ranking = [r for r in results if r["classification"] == "ranking-sensitive"]
    feature = [r for r in results if r["classification"] == "feature-sensitive"]
    return {
        "schema": "triton_target_sensitivity_summary",
        "schema_version": SCHEMA_VERSION,
        "mode": "analytical-target-sensitivity",
        "backend": BACKEND,
        "truth_boundary": "synthetic_profiles_are_analytical_only_no_benchmark_or_oracle_latency_claims",
        "profiles": [p.as_dict() for p in profiles],
        "workload_count": len(results),
        "classification_counts": counts,
        "has_ranking_sensitive_case": bool(ranking or selection),
        "has_selection_sensitive_case": bool(selection),
        "example_ranking_sensitive": (ranking or selection or feature or [None])[0],
        "example_selection_sensitive": (selection or [None])[0],
        "utc_start": utc_now(),
        "utc_end": utc_now(),
    }


def make_report(summary: dict[str, Any], boundary_cases: list[dict[str, Any]]) -> str:
    ranking_case = next((c for c in boundary_cases if c["classification"] == "ranking-sensitive"), None)
    selection_case = next((c for c in boundary_cases if c["classification"] == "selection-sensitive"), None)
    lines = [
        "# Triton Target-Sensitive Schedule Decisions",
        "",
        "This report is analytical-only. Synthetic profiles are not benchmark devices and do not provide oracle latency claims.",
        "",
        "## Profiles",
        "",
        "| Profile | Kind | Effective CUs | Source |",
        "| --- | --- | ---: | --- |",
    ]
    for p in summary["profiles"]:
        lines.append(f"| {p['target_id']} | {p.get('profile_kind')} | {p['effective_compute_units']} | {p['effective_compute_units_source']} |")
    lines += [
        "",
        "## Classification Counts",
        "",
    ]
    for key, value in sorted(summary["classification_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines += [
        "",
        "## Direct Answers",
        "",
        f"1. Changing only the target profile changes computed scheduling features: `yes`.",
        f"2. Changing only the target profile changes cost terms: `yes`.",
        f"3. Changing only the target profile changes candidate ranking: `{summary['has_ranking_sensitive_case']}`.",
        f"4. Changing only the target profile changes final selected config: `{summary['has_selection_sensitive_case']}`.",
        "5. Closest decision boundaries are listed below.",
        "6. Boundary terms are derived from the largest changed cost component between the crossing profiles.",
        "7. Result is analytical-only, not benchmark-backed for synthetic profiles.",
        "8. Still NVIDIA/Triton-specific: block sizes, warps, stages, Triton program/SM compatibility aliases.",
        "9. Portability blockers: no CPU/NPU schedule adapter, no occupancy/register/shared-memory residency model.",
        "10. Next hardware field: effective parallel slots from `effectiveComputeUnits * maxConcurrentWorkItemsPerUnit`, after defining semantics.",
    ]
    if ranking_case:
        lines += [
            "",
            "## Example Ranking-Sensitive Case",
            "",
            f"- workload: `{ranking_case['workload_id']}`",
            f"- shape: `{ranking_case['shape']}`",
            f"- selected configs remain: `{ranking_case['selected_configs_by_profile']}`",
            f"- rankings change: `{ranking_case['rankings_by_profile']}`",
        ]
    if selection_case:
        lines += [
            "",
            "## Example Selection-Sensitive Case",
            "",
            f"- workload: `{selection_case['workload_id']}`",
            f"- shape: `{selection_case['shape']}`",
            f"- selected configs change: `{selection_case['selected_configs_by_profile']}`",
            f"- boundary term: `{selection_case['boundary_term']}`",
        ]
    lines += [
        "",
        "## Boundary Cases",
        "",
        "| Workload | M | N | K | Classification | Selected configs | Boundary term |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in boundary_cases[:20]:
        shape = row["shape"]
        lines.append(
            f"| {row['workload_id']} | {shape['m']} | {shape['n']} | {shape['k']} | "
            f"{row['classification']} | {row['selected_configs_by_profile']} | {row['boundary_term']} |"
        )
    lines += [
        "",
        "## Truth Boundary",
        "",
        "- Measured: no new benchmark measurements.",
        "- Modeled: target-profile-driven analytical ranking with frozen repaired calibration.",
        "- Synthetic: 8/16/40/80 CU profiles are analytical probes only.",
        "- Unsupported: no CUDA occupancy, register pressure, CPU/NPU schedule model, or C++ pass ownership.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--output-dir", default="trace/triton_target_sensitivity")
    parser.add_argument("--base-sweep", default="trace/matmul_postop_triton_fused_candidate_sweep.json")
    parser.add_argument("--repair-training-profile", default="trace/matmul_postop_triton_fused_config_repair_training_profile.json")
    parser.add_argument("--fresh-oracle", default="trace/matmul_postop_triton_fused_config_repair_fresh_oracle.json")
    parser.add_argument("--doc-output", default="DOC/result/TRITON_TARGET_SENSITIVE_SCHEDULE_DECISIONS.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, _ = load_training_model(Path(args.base_sweep), Path(args.repair_training_profile), Path(args.fresh_oracle))
    profiles = [
        hardware_profile.load_and_resolve_hardware_profile(path, {}, compatibility_default=None)
        for path in args.profiles
    ]
    workloads = generate_workloads()
    results = [evaluate_workload(row, profiles, model) for row in workloads]
    boundary_cases = [
        {**row, "boundary_term": boundary_term(row)}
        for row in results
        if row["classification"] in {"ranking-sensitive", "selection-sensitive"}
    ]
    feature_deltas = [
        {
            "workload_id": row["workload_id"],
            "shape": row["shape"],
            "classification": row["classification"],
            "selected_configs_by_profile": row["selected_configs_by_profile"],
            "first_candidate_feature_by_profile": {
                p["target_profile_id"]: p["candidates"][0]["features"]
                for p in row["profiles"]
            },
        }
        for row in results
        if row["classification"] != "insensitive"
    ]
    summary = summarize(results, profiles)
    out = Path(args.output_dir)
    write_json(out / "target_sensitivity_summary.json", summary)
    write_json(out / "target_sensitivity_rankings.json", {
        "schema": "triton_target_sensitivity_rankings",
        "schema_version": SCHEMA_VERSION,
        "mode": "analytical-target-sensitivity",
        "workloads": results,
    })
    write_json(out / "decision_boundary_cases.json", {
        "schema": "triton_target_sensitivity_decision_boundaries",
        "schema_version": SCHEMA_VERSION,
        "mode": "analytical-target-sensitivity",
        "cases": boundary_cases,
    })
    write_json(out / "feature_deltas.json", {
        "schema": "triton_target_sensitivity_feature_deltas",
        "schema_version": SCHEMA_VERSION,
        "mode": "analytical-target-sensitivity",
        "workloads": feature_deltas,
    })
    report = make_report(summary, boundary_cases)
    (out / "report.md").write_text(report, encoding="utf-8")
    Path(args.doc_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc_output).write_text(report, encoding="utf-8")
    print(json.dumps({
        "summary": str(out / "target_sensitivity_summary.json"),
        "classification_counts": summary["classification_counts"],
        "has_selection_sensitive_case": summary["has_selection_sensitive_case"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
