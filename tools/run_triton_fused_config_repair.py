#!/usr/bin/env python3
"""Small-square/high-K repair for Triton fused config selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_triton_fused_config_selection as base  # noqa: E402


SCHEMA_VERSION = 1
PRIMARY_CONFIG_IDS = base.PRIMARY_CONFIG_IDS
PRIMARY_CONFIGS = base.PRIMARY_CONFIGS
KERNEL_FAMILY_ID = base.KERNEL_FAMILY_ID
BACKEND = base.BACKEND
FORBIDDEN_MODEL_FIELDS = set(base.FORBIDDEN_MODEL_FIELDS) | {
    "training_region_label",
    "oracle_winner",
    "measured_rank",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_forbidden(payload: dict[str, Any]) -> None:
    present = sorted(FORBIDDEN_MODEL_FIELDS & set(payload))
    if present:
        raise ValueError(f"forbidden evaluation fields in repair model input: {present}")


def workload_features(shape: dict[str, Any]) -> dict[str, Any]:
    reject_forbidden(shape)
    f = base.workload_features(shape)
    output_area = f["mn"]
    k = f["k"]
    f.update(
        {
            "k_dominance_ratio": k / max(output_area, 1),
            "output_area_to_k": output_area / max(k, 1),
            "small_square_high_k": f["square_like"] and output_area <= 16384 and k >= 1024,
            "log2_k_dominance": base.safe_log2(k / max(output_area, 1)),
        }
    )
    f["feature_schema_version"] = base.FEATURE_SCHEMA_VERSION
    f["reduction_dominance"] = f["k_dominance_ratio"]
    return f


def tile_features(shape_features: dict[str, Any], cfg: base.TritonFusedConfig, sm_count: int = 16) -> dict[str, Any]:
    tf = base.tile_features(shape_features, cfg, sm_count)
    k_iters = tf["k_tile_count"]
    programs = tf["total_output_program_count"]
    work_per_program = cfg.block_m * cfg.block_n * k_iters
    programs_per_sm = programs / max(sm_count, 1)
    waves = math.ceil(programs / max(sm_count, 1))
    tf.update(
        {
            "programs_per_sm": programs_per_sm,
            "k_iterations_per_output_program": k_iters,
            "work_per_program": work_per_program,
            "reuse_per_output_program": cfg.block_m * cfg.block_n,
            "parallelism_to_compute_ratio": programs_per_sm / max(k_iters, 1),
            "tile_area_relative_to_output_area": (cfg.block_m * cfg.block_n) / max(shape_features["mn"], 1),
            "program_granularity": cfg.block_m * cfg.block_n,
            "output_program_waves": waves,
            "k_dominant_small_output": shape_features["k_dominance_ratio"] >= 0.25 and shape_features["mn"] <= 16384,
        }
    )
    tf.update(
        {
            "feature_schema_version": base.FEATURE_SCHEMA_VERSION,
            "parallel_work_items": programs,
            "work_items_per_compute_unit": programs_per_sm,
            "execution_waves": waves,
            "compute_work_per_item": work_per_program,
            "reduction_dominance": shape_features["k_dominance_ratio"],
            "padding_waste_ratio": tf["masked_output_fraction"],
            "work_amplification": tf["padding_amplification"],
        }
    )
    return tf


def component_units(shape: dict[str, Any], cfg: base.TritonFusedConfig, sm_count: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    wf = workload_features(shape)
    tf = tile_features(wf, cfg, sm_count)
    useful_flops_g = wf["flops"] / 1e9
    effective_flops_g = tf["effective_tiled_flops"] / 1e9
    memory_gb = (wf["a_bytes"] + wf["b_bytes"] + wf["bias_bytes"] + wf["output_bytes"]) / 1e9
    padding_extra = max(0.0, effective_flops_g - useful_flops_g)
    programs_per_sm = tf["programs_per_sm"]
    work = tf["work_per_program"]
    low_parallel = max(0.0, 1.0 - programs_per_sm / 2.0)
    # Repair term: large tiles with too few output programs and high K get a
    # continuous penalty. This captures the 64x64x4096 failure without using
    # workload IDs or winner labels.
    k_dominant_parallelism = (
        max(0.0, 1.0 - programs_per_sm / 2.0)
        * min(4.0, wf["k_dominance_ratio"])
        * (work / max(16 * 16 * tf["k_tile_count"], 1))
    )
    excessive_programs = max(0.0, programs_per_sm / 128.0 - 1.0) * (1.0 / max(wf["k_dominance_ratio"], 1e-6))
    shape_mismatch = 0.0
    if wf["skinny_m"] and cfg.block_m > 16:
        shape_mismatch += (cfg.block_m / 16) - 1.0
    if wf["small_square_high_k"] and cfg.block_m > 16:
        shape_mismatch += (cfg.block_m / 16) - 1.0
    if wf["square_like"] and cfg.block_m != cfg.block_n:
        shape_mismatch += 0.25
    if wf["large_m"] and wf["large_n"] and cfg.block_m < 32:
        shape_mismatch += 0.5
    units = {
        "fixed_unit": 1.0,
        "effective_compute_unit": effective_flops_g,
        "memory_unit": memory_gb,
        "padding_unit": padding_extra,
        "low_parallelism_unit": low_parallel,
        "k_dominant_parallelism_unit": k_dominant_parallelism,
        "program_overhead_unit": excessive_programs,
        "shape_mismatch_unit": shape_mismatch,
    }
    return wf, tf, units


def unit_sum(coeff: dict[str, float], units: dict[str, float]) -> float:
    return (
        coeff["fixed_ms"] * units["fixed_unit"]
        + coeff["compute_ms_per_gflop"] * units["effective_compute_unit"]
        + coeff["memory_ms_per_gbyte"] * units["memory_unit"]
        + coeff["padding_ms_per_gflop"] * units["padding_unit"]
        + coeff["low_parallelism_ms"] * units["low_parallelism_unit"]
        + coeff["k_dominant_parallelism_ms"] * units["k_dominant_parallelism_unit"]
        + coeff["program_overhead_ms"] * units["program_overhead_unit"]
        + coeff["shape_mismatch_ms"] * units["shape_mismatch_unit"]
    )


def median_ratio(actuals: list[float], preds: list[float], default: float) -> float:
    vals = [a / p for a, p in zip(actuals, preds) if a > 0 and p > 0 and math.isfinite(a / p)]
    return statistics.median(vals) if vals else default


def calibrate(rows: list[base.Measurement], sm_count: int) -> dict[str, Any]:
    all_lat, all_compute, all_memory, all_padding = [], [], [], []
    for row in rows:
        for cid in PRIMARY_CONFIG_IDS:
            _, _, units = component_units(row.shape, PRIMARY_CONFIGS[cid], sm_count)
            all_lat.append(row.latencies[cid])
            all_compute.append(units["effective_compute_unit"])
            all_memory.append(units["memory_unit"])
            all_padding.append(units["padding_unit"])
    med = statistics.median(all_lat) if all_lat else 0.01
    coeff = {
        "fixed_ms": med * 0.025,
        "compute_ms_per_gflop": median_ratio(all_lat, all_compute, 0.1) * 0.55,
        "memory_ms_per_gbyte": median_ratio(all_lat, all_memory, 0.1) * 0.08,
        "padding_ms_per_gflop": median_ratio(all_lat, all_padding, 0.0) * 0.05 if any(all_padding) else 0.0,
        "low_parallelism_ms": med * 0.05,
        "k_dominant_parallelism_ms": med * 0.18,
        "program_overhead_ms": med * 0.025,
        "shape_mismatch_ms": med * 0.08,
    }
    corrections: dict[str, dict[str, float]] = {}
    residuals = []
    for cid in PRIMARY_CONFIG_IDS:
        actuals, raws = [], []
        for row in rows:
            _, _, units = component_units(row.shape, PRIMARY_CONFIGS[cid], sm_count)
            actuals.append(row.latencies[cid])
            raws.append(unit_sum(coeff, units))
        scale = max(0.0, median_ratio(actuals, raws, 1.0))
        corrections[cid] = {"affine_scale": scale, "affine_bias_ms": 0.0}
        for actual, raw in zip(actuals, raws):
            pred = raw * scale
            residuals.append(abs(pred - actual) / max(actual, 1e-9))
    mean_norm = {}
    for cid in PRIMARY_CONFIG_IDS:
        vals = []
        for row in rows:
            best = min(row.latencies.values())
            vals.append(row.latencies[cid] / best)
        mean_norm[cid] = statistics.fmean(vals) if vals else float("inf")
    fallback = min(mean_norm.items(), key=lambda item: (item[1], item[0]))[0]
    return {
        "formula": "fixed + effective_compute + memory + padding + low_parallelism + k_dominant_parallelism + excessive_programs + shape_config_mismatch",
        "coefficients": coeff,
        "per_config_affine_calibration": corrections,
        "cross_validation_error_proxy": {
            "mean_relative_abs_error": statistics.fmean(residuals) if residuals else 0.0,
            "median_relative_abs_error": statistics.median(residuals) if residuals else 0.0,
            "p95_relative_abs_error": base.percentile(residuals, 95),
        },
        "confidence_thresholds": {
            "minimum_confident_margin": max(0.03, statistics.median(residuals) if residuals else 0.03),
            "tie_relative_difference": 0.01,
        },
        "fallback_config_id": fallback,
        "fallback_config_training_mean_normalized_latency": mean_norm[fallback],
    }


def predict_config(shape: dict[str, Any], cfg: base.TritonFusedConfig, model: dict[str, Any], sm_count: int, calibrated: bool) -> dict[str, Any]:
    wf, tf, units = component_units(shape, cfg, sm_count)
    coeff = model["coefficients"]
    components = {
        "fixed_ms": coeff["fixed_ms"] * units["fixed_unit"],
        "compute_ms": coeff["compute_ms_per_gflop"] * units["effective_compute_unit"],
        "memory_ms": coeff["memory_ms_per_gbyte"] * units["memory_unit"],
        "padding_ms": coeff["padding_ms_per_gflop"] * units["padding_unit"],
        "low_parallelism_ms": coeff["low_parallelism_ms"] * units["low_parallelism_unit"],
        "k_dominant_parallelism_ms": coeff["k_dominant_parallelism_ms"] * units["k_dominant_parallelism_unit"],
        "program_overhead_ms": coeff["program_overhead_ms"] * units["program_overhead_unit"],
        "shape_mismatch_ms": coeff["shape_mismatch_ms"] * units["shape_mismatch_unit"],
    }
    raw = sum(components.values())
    corr = model["per_config_affine_calibration"][cfg.config_id]
    pred = raw * corr["affine_scale"] + corr["affine_bias_ms"] if calibrated else raw
    return {
        "semantic_kernel_id": KERNEL_FAMILY_ID,
        "config_id": cfg.config_id,
        "predicted_latency_ms": pred,
        "components": components,
        "workload_features": wf,
        "tile_features": tf,
        "affine_calibration": corr,
    }


def ranking(shape: dict[str, Any], model: dict[str, Any], sm_count: int, calibrated: bool) -> list[dict[str, Any]]:
    rows = [predict_config(shape, PRIMARY_CONFIGS[cid], model, sm_count, calibrated) for cid in PRIMARY_CONFIG_IDS]
    return sorted(rows, key=lambda r: (r["predicted_latency_ms"], r["config_id"]))


DISTANCE_FEATURES = base.DISTANCE_FEATURES + ("log2_k_dominance",)


def feature_stats(rows: list[base.Measurement]) -> dict[str, Any]:
    feats = [workload_features(r.shape) for r in rows]
    stats = {}
    for key in DISTANCE_FEATURES:
        vals = [float(f[key]) for f in feats]
        stats[key] = {"mean": statistics.fmean(vals) if vals else 0.0, "std": statistics.stdev(vals) if len(vals) > 1 else 1.0}
    return stats


def vector(features: dict[str, Any], stats: dict[str, Any]) -> list[float]:
    return [(float(features[k]) - stats[k]["mean"]) / (stats[k]["std"] or 1.0) for k in DISTANCE_FEATURES]


def ood(shape: dict[str, Any], train: list[base.Measurement], stats: dict[str, Any]) -> dict[str, Any]:
    vf = vector(workload_features(shape), stats)
    best = (float("inf"), None)
    train_vecs = [(r, vector(workload_features(r.shape), stats)) for r in train]
    for row, tv in train_vecs:
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vf, tv)) / len(vf))
        if dist < best[0]:
            best = (dist, row)
    return {"distance_to_training_distribution": best[0], "nearest_training_shape": best[1].workload_id if best[1] else None}


def choose_policy(policy: str, shape: dict[str, Any], model: dict[str, Any], train: list[base.Measurement], stats: dict[str, Any], sm_count: int) -> dict[str, Any]:
    if policy.startswith("static_"):
        cid = policy.removeprefix("static_")
        return {"selected_config_id": cid, "selection_source": f"static_{cid}", "candidate_ranking": []}
    if policy == "nearest_shape":
        od = ood(shape, train, stats)
        nearest = next(r for r in train if r.workload_id == od["nearest_training_shape"])
        selected = min(nearest.latencies.items(), key=lambda item: (item[1], item[0]))[0]
        return {"selected_config_id": selected, "selection_source": "nearest_measured_shape", **od}
    calibrated = policy in {"repaired_calibrated", "repaired_confidence"}
    rank = ranking(shape, model, sm_count, calibrated)
    best, second = rank[0], rank[1]
    margin = second["predicted_latency_ms"] / max(best["predicted_latency_ms"], 1e-9) - 1.0
    od = ood(shape, train, stats)
    if policy == "repaired_confidence":
        threshold = model["confidence_thresholds"]["minimum_confident_margin"]
        if margin < threshold:
            selected = model["fallback_config_id"]
            return {
                "selected_config_id": selected,
                "selection_source": "fused_config_confidence_fallback",
                "fallback_reason": "low_model_confidence",
                "candidate_ranking": rank,
                "confidence": {"level": "low", "predicted_margin": margin, "ood_distance": od["distance_to_training_distribution"]},
                **od,
            }
    return {
        "selected_config_id": best["config_id"],
        "selection_source": "repaired_calibrated_cost_model" if calibrated else "repaired_analytical_cost_model",
        "fallback_reason": None,
        "candidate_ranking": rank,
        "confidence": {"level": "high", "predicted_margin": margin, "ood_distance": od["distance_to_training_distribution"]},
        **od,
    }


def grouped_split(rows: list[base.Measurement], split: str) -> tuple[list[base.Measurement], list[base.Measurement], dict[str, Any]]:
    if split == "leave-one-small-square-family-out":
        held_ids = {"unfriendly_m64_n64_k4096", "repair_holdout_sq96_k8192"}
    elif split == "leave-one-K-band-out":
        held_ids = {r.workload_id for r in rows if r.shape["k"] >= 4096 and "repair_train" not in r.workload_id}
    elif split == "leave-one-shape-region-out":
        held_ids = {
            "rep_m1_k768_n3072",
            "rep_m128_k768_n3072",
            "balanced_m64_n64_k64",
            "unfriendly_m64_n64_k4096",
            "boundary_m1_n4096_k65536",
            "boundary_m256_n256_k2048",
            "repair_holdout_sq96_k8192",
        }
    else:
        raise ValueError(split)
    train = [r for r in rows if r.workload_id not in held_ids]
    held = [r for r in rows if r.workload_id in held_ids]
    return train, held, {"strategy": split, "training_workload_ids": [r.workload_id for r in train], "held_out_workload_ids": [r.workload_id for r in held]}


POLICIES = (
    "static_bm16_bn16_bk32_w4_s3",
    "static_bm32_bn32_bk32_w4_s3",
    "static_bm64_bn64_bk32_w4_s3",
    "static_bm16_bn64_bk32_w4_s3",
    "nearest_shape",
    "previous_calibrated",
    "previous_confidence",
    "repaired_analytical",
    "repaired_calibrated",
    "repaired_confidence",
)


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return base.evaluate(rows)


def build_plan(row: base.Measurement, decision: dict[str, Any], model_sha: str) -> dict[str, Any]:
    cfg = PRIMARY_CONFIGS[decision["selected_config_id"]]
    return base.build_plan(row, {**decision, "candidate_ranking": decision.get("candidate_ranking", [])}, model_sha)


def parse_repair_profile(path: Path) -> list[base.Measurement]:
    if not path.exists():
        return []
    rows, _ = base.parse_discovery_measurements(path)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-sweep", default="trace/matmul_postop_triton_fused_candidate_sweep.json")
    p.add_argument("--repair-training-profile", default="trace/matmul_postop_triton_fused_config_repair_training_profile.json")
    p.add_argument("--fresh-oracle", default="trace/matmul_postop_triton_fused_config_repair_fresh_oracle.json")
    p.add_argument("--split", default="leave-one-shape-region-out", choices=("leave-one-shape-region-out", "leave-one-K-band-out", "leave-one-small-square-family-out"))
    p.add_argument("--cost-model-output", default="trace/matmul_postop_triton_fused_config_repair_cost_model.json")
    p.add_argument("--plans-output", default="trace/matmul_postop_triton_fused_config_repair_plans.json")
    p.add_argument("--plan-validation-output", default="trace/matmul_postop_triton_fused_config_repair_plan_validation.json")
    p.add_argument("--summary-output", default="trace/matmul_postop_triton_fused_config_repair_summary.json")
    p.add_argument("--report-output", default="trace/matmul_postop_triton_fused_config_repair_report.md")
    p.add_argument("--doc-output", default="DOC/result/TRITON_FUSED_CONFIG_MODEL_REPAIR.md")
    p.add_argument("--target-profile", default=None)
    p.add_argument("--effective-compute-units", type=int, default=None)
    p.add_argument("--allow-hardware-override", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = utc_now()
    base_rows, base_payload = base.parse_discovery_measurements(Path(args.base_sweep))
    repair_rows = parse_repair_profile(Path(args.repair_training_profile))
    fresh_rows = parse_repair_profile(Path(args.fresh_oracle))
    training_candidate_rows = {r.workload_id: r for r in base_rows + repair_rows}
    evaluation_candidate_rows = {r.workload_id: r for r in base_rows + repair_rows + fresh_rows}
    train, held, split = grouped_split(list(evaluation_candidate_rows.values()), args.split)
    # Fresh oracle rows can define/evaluate held-out workloads, but fitting is
    # restricted to rows present in the designated training profiles.
    train = [training_candidate_rows[r.workload_id] for r in train if r.workload_id in training_candidate_rows]
    split["training_workload_ids"] = [r.workload_id for r in train]
    fresh_map = {r.workload_id: r for r in fresh_rows}
    held_eval = [fresh_map.get(r.workload_id, r) for r in held]
    env = base_payload.get("environment", {})
    resolved_hardware = base.hardware_profile.load_and_resolve_hardware_profile(
        args.target_profile,
        env,
        cli_override=args.effective_compute_units,
        compatibility_default=16,
        allow_cli_override=args.allow_hardware_override,
    )
    sm_count = resolved_hardware.effective_compute_units
    model = calibrate(train, sm_count)
    stats = feature_stats(train)
    prev_model = read_json(Path("trace/matmul_postop_triton_fused_config_cost_model.json"))["analytical_model"]
    prev_stats = base.feature_stats(train)
    cost_payload = {
        "schema": "triton_matmul_bias_relu_fused_config_repair_cost_model",
        "schema_version": 1,
        "backend": BACKEND,
        "target_gpu": {"gpu_model": env.get("gpu_model"), "compute_capability": env.get("compute_capability")},
        "hardware_profile": resolved_hardware.as_dict(),
        "candidate_registry": [PRIMARY_CONFIGS[cid].typed() for cid in PRIMARY_CONFIG_IDS],
        "feature_schema": {
            "feature_schema_version": base.FEATURE_SCHEMA_VERSION,
            "new_features": [
                "k_dominance_ratio",
                "reduction_dominance",
                "output_area_to_k",
                "programs_per_sm",
                "work_items_per_compute_unit",
                "work_per_program",
                "compute_work_per_item",
                "parallelism_to_compute_ratio",
                "tile_area_relative_to_output_area",
                "output_program_waves",
                "execution_waves",
            ],
            "compatibility_aliases": {
                "total_output_program_count": "parallel_work_items",
                "programs_per_sm": "work_items_per_compute_unit",
                "output_program_waves": "execution_waves",
                "work_per_program": "compute_work_per_item",
                "k_dominance_ratio": "reduction_dominance",
                "masked_output_fraction": "padding_waste_ratio",
                "padding_amplification": "work_amplification",
            },
            "forbidden_model_fields": sorted(FORBIDDEN_MODEL_FIELDS),
        },
        "analytical_model": model,
        "grouped_split": split,
        "training_profile_sha256": sha256_file(Path(args.repair_training_profile)) if Path(args.repair_training_profile).exists() else None,
        "base_sweep_sha256": sha256_file(Path(args.base_sweep)),
        "fresh_oracle_sha256": sha256_file(Path(args.fresh_oracle)) if Path(args.fresh_oracle).exists() else None,
        "truth_boundary": "repair_model_uses_training_profiles_only_fresh_oracle_excluded_from_fitting",
        "utc_start": started,
        "utc_end": utc_now(),
    }
    write_json(Path(args.cost_model_output), cost_payload)
    model_sha = sha256_file(Path(args.cost_model_output))
    policy_rows = {p: [] for p in POLICIES}
    workloads_summary = []
    plans = []
    for row in held_eval:
        oracle = min(row.latencies.items(), key=lambda item: (item[1], item[0]))[0]
        oracle_latency = row.latencies[oracle]
        decisions: dict[str, Any] = {}
        for policy in POLICIES:
            if policy == "previous_calibrated":
                d = base.choose_policy("calibrated_analytical", row.shape, prev_model, train, prev_stats, sm_count)
            elif policy == "previous_confidence":
                d = base.choose_policy("confidence_aware", row.shape, prev_model, train, prev_stats, sm_count)
            else:
                d = choose_policy(policy, row.shape, model, train, stats, sm_count)
            selected = d["selected_config_id"]
            regret = row.latencies[selected] / oracle_latency - 1.0
            result = {
                **d,
                "selected_latency_ms": row.latencies[selected],
                "oracle_latency_ms": oracle_latency,
                "oracle_config": oracle,
                "top1_correct": selected == oracle,
                "regret": regret,
                "within_1_percent": regret <= 0.01,
                "within_3_percent": regret <= 0.03,
            }
            decisions[policy] = result
            policy_rows[policy].append(result)
        plans.append(build_plan(row, decisions["repaired_calibrated"], model_sha))
        workloads_summary.append({"workload_id": row.workload_id, "shape": row.shape, "oracle_config": oracle, "latencies": row.latencies, "policy_results": decisions})
    validation = base.validate_plans(plans)
    policy_aggregates = {name: evaluate(rows) for name, rows in policy_rows.items()}
    summary = {
        "schema": "triton_matmul_bias_relu_fused_config_repair_summary",
        "schema_version": 1,
        "mode": "aggregate-report",
        "backend": BACKEND,
        "cost_model_artifact_sha256": model_sha,
        "plan_validation_aggregate": validation["aggregate"],
        "grouped_split": split,
        "policy_aggregates": policy_aggregates,
        "workloads": workloads_summary,
        "diagnosis_workload": "unfriendly_m64_n64_k4096",
        "collapse_audit": {name: policy_aggregates[name]["config_diversity"] == 1 for name in ("repaired_analytical", "repaired_calibrated", "repaired_confidence")},
        "utc_start": started,
        "utc_end": utc_now(),
    }
    write_json(Path(args.plans_output), {"schema": "triton_matmul_bias_relu_fused_config_repair_plans", "schema_version": 1, "mode": "compiler-selection", "plans": plans, "cost_model_artifact_sha256": model_sha})
    write_json(Path(args.plan_validation_output), validation)
    write_json(Path(args.summary_output), summary)
    report = make_report(summary)
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(report, encoding="utf-8")
    Path(args.doc_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc_output).write_text(report, encoding="utf-8")
    print(json.dumps({"policy_aggregates": policy_aggregates, "validation": validation["aggregate"]}, indent=2))
    return 0


def make_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Triton Fused Config Model Repair",
        "",
        "## Policy Comparison",
        "",
        "| Policy | Diversity | Top-1 | Macro | Mean Regret | P95 | Max | Within 3% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in summary["policy_aggregates"].items():
        lines.append(f"| {name} | {row['config_diversity']} | {row['top1_accuracy']:.4f} | {row['macro_accuracy']:.4f} | {row['mean_regret']:.4f} | {row['p95_regret']:.4f} | {row['max_regret']:.4f} | {row['within_3_percent']:.4f} |")
    lines += ["", "## Per-Workload Decisions", "", "| Workload | Oracle | Repaired selected | Regret |", "| --- | --- | --- | ---: |"]
    for w in summary["workloads"]:
        d = w["policy_results"]["repaired_calibrated"]
        lines.append(f"| {w['workload_id']} | {w['oracle_config']} | {d['selected_config_id']} | {d['regret']:.4f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
