#!/usr/bin/env python3
"""Shape-aware confidence-guided Triton V1/V3 kernel decisions.

PR C scope only: label-safe feature extraction, an interpretable analytical
latency model, confidence-aware fallback, ExecutionPlan metadata, and policy
comparison. No new kernels, autotuning, or opaque learned model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlir_fusion_to_runtime_json as bridge  # noqa: E402
from matmul_postop_workloads import canonical_workloads, decision_boundary_workloads, load_manifest  # noqa: E402
from run_triton_matmul_bias_relu_exact_selection import (  # noqa: E402
    latency_for_kernel,
    oracle_winner,
    profile_target_environment,
    selected_config_from_profile,
)


BACKEND = "triton_cuda"
DTYPE = "f32"
PATTERN = "bias"
SCHEMA_VERSION = 1
V1_KERNEL = "triton_tiled_matmul_bias_relu_unfused_f32"
V3_KERNEL = "triton_tiled_matmul_bias_relu_one_pass_f32"
VARIANT_TO_KERNEL = {"V1": V1_KERNEL, "V3": V3_KERNEL}
KERNEL_TO_VARIANT = {v: k for k, v in VARIANT_TO_KERNEL.items()}
FORBIDDEN_MODEL_FIELDS = {
    "expected_region",
    "measured_winner",
    "final_classification",
    "oracle_latency",
    "oracle_best_kernel",
    "fresh_oracle",
    "relative_regret",
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


def safe_log2(value: float) -> float:
    return math.log2(max(float(value), 1.0))


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def geometric_mean(values: list[float]) -> float:
    positives = [v for v in values if v > 0]
    return math.exp(sum(math.log(v) for v in positives) / len(positives)) if positives else 0.0


def shape_features(shape: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    reject_forbidden_model_input(shape)
    m, n, k = int(shape["m"]), int(shape["n"]), int(shape["k"])
    dtype_bytes = 4
    output_elements = m * n
    a_elements = m * k
    b_elements = k * n
    flops = 2 * m * n * k
    output_bytes = output_elements * dtype_bytes
    a_bytes = a_elements * dtype_bytes
    b_bytes = b_elements * dtype_bytes
    bias_bytes = n * dtype_bytes
    block_m = int(config["block_m"])
    block_n = int(config["block_n"])
    block_k = int(config["block_k"])
    output_tile_count = ceil_div(m, block_m) * ceil_div(n, block_n)
    k_tile_count = ceil_div(k, block_k)
    total_bytes_once = a_bytes + b_bytes + bias_bytes + output_bytes
    return {
        "m": m,
        "n": n,
        "k": k,
        "dtype": shape.get("dtype", DTYPE),
        "log2_m": safe_log2(m),
        "log2_n": safe_log2(n),
        "log2_k": safe_log2(k),
        "output_elements": output_elements,
        "a_elements": a_elements,
        "b_elements": b_elements,
        "flops": flops,
        "output_bytes": output_bytes,
        "bias_bytes": bias_bytes,
        "a_bytes": a_bytes,
        "b_bytes": b_bytes,
        "m_over_n": m / n,
        "n_over_m": n / m,
        "k_over_m": k / m,
        "k_over_n": k / n,
        "arithmetic_intensity": flops / max(total_bytes_once, 1),
        "output_tile_count": output_tile_count,
        "k_tile_count": k_tile_count,
        "m_edge_utilization": m / (ceil_div(m, block_m) * block_m),
        "n_edge_utilization": n / (ceil_div(n, block_n) * block_n),
        "k_tail_utilization": k / (ceil_div(k, block_k) * block_k),
        "small_m": m <= 16,
        "small_n": n <= 128,
        "extreme_k": k >= 8192,
        "skinny_output": m <= 16 or n <= 128,
        "log2_output_elements": safe_log2(output_elements),
    }


def candidate_features(variant: str, features: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if variant not in VARIANT_TO_KERNEL:
        raise ValueError(f"unknown variant: {variant}")
    output_bytes = features["output_bytes"]
    common_bytes = features["a_bytes"] + features["b_bytes"] + features["bias_bytes"]
    if variant == "V1":
        launches = 3
        intermediates = 2
        global_bytes = common_bytes + (6 * output_bytes)
    else:
        launches = 1
        intermediates = 0
        global_bytes = common_bytes + output_bytes
    utilization = (
        features["m_edge_utilization"]
        * features["n_edge_utilization"]
        * features["k_tail_utilization"]
    )
    return {
        "variant": variant,
        "kernel_id": VARIANT_TO_KERNEL[variant],
        "runtime_operation_count": launches,
        "expected_launch_count": launches,
        "full_size_intermediate_count": intermediates,
        "estimated_full_size_intermediate_bytes": intermediates * output_bytes,
        "estimated_global_bytes": global_bytes,
        "block_m": config["block_m"],
        "block_n": config["block_n"],
        "block_k": config["block_k"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        "precision_mode": config["precision_mode"],
        "output_tile_count": features["output_tile_count"],
        "k_tile_count": features["k_tile_count"],
        "tile_utilization": utilization,
        "low_parallelism": features["output_tile_count"] < 64,
    }


def reject_forbidden_model_input(payload: dict[str, Any]) -> None:
    present = sorted(FORBIDDEN_MODEL_FIELDS & set(payload))
    if present:
        raise ValueError(f"forbidden evaluation fields in model input: {present}")


@dataclass
class Measurement:
    workload_id: str
    shape: dict[str, Any]
    config: dict[str, Any]
    category: str
    held_out: bool
    source: str
    classification: str | None
    latencies: dict[str, float]


def fixed_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = ((payload.get("benchmark_config") or {}).get("fixed_config") or {})
    if not cfg and payload.get("fixed_config"):
        cfg = payload["fixed_config"]
    return {
        "block_m": cfg.get("BLOCK_M", cfg.get("block_m")),
        "block_n": cfg.get("BLOCK_N", cfg.get("block_n")),
        "block_k": cfg.get("BLOCK_K", cfg.get("block_k")),
        "num_warps": cfg.get("num_warps"),
        "num_stages": cfg.get("num_stages"),
        "precision_mode": cfg.get("precision_mode", "ieee"),
    }


def parse_profile_measurements(path: Path) -> list[Measurement]:
    payload = read_json(path)
    config = fixed_config_from_payload(payload)
    rows: list[Measurement] = []
    if payload.get("schema") == "triton_matmul_bias_relu_fixed_config_profile":
        for row in payload.get("workloads", []):
            if row.get("status") != "completed":
                continue
            latencies = {}
            for variant in ("V1", "V3"):
                timing = ((row["variants"][variant].get("timing") or {}).get("statistics") or {})
                if row["variants"][variant].get("correctness", {}).get("passed") and timing.get("median_ms"):
                    latencies[variant] = float(timing["median_ms"])
            if set(latencies) == {"V1", "V3"}:
                rows.append(
                    Measurement(
                        workload_id=row["workload_id"],
                        shape=dict(row["shape"]),
                        config=selected_config_from_profile(row),
                        category=row.get("category", ""),
                        held_out=bool(row.get("held_out", False)),
                        source=str(path),
                        classification=None,
                        latencies=latencies,
                    )
                )
    elif payload.get("schema") == "triton_matmul_bias_relu_decision_boundary_profile":
        for row in payload.get("workloads", []):
            if row.get("status") != "completed":
                continue
            if row.get("cross_session_v1_median_ms") and row.get("cross_session_v3_median_ms"):
                rows.append(
                    Measurement(
                        workload_id=row["workload_id"],
                        shape=dict(row["shape"]),
                        config=selected_config_from_profile(row),
                        category=row.get("category", "decision_boundary"),
                        held_out=False,
                        source=str(path),
                        classification=row.get("final_classification"),
                        latencies={
                            "V1": float(row["cross_session_v1_median_ms"]),
                            "V3": float(row["cross_session_v3_median_ms"]),
                        },
                    )
                )
    else:
        raise ValueError(f"unsupported training profile schema: {payload.get('schema')}")
    return rows


def split_measurements(measurements: list[Measurement]) -> tuple[list[Measurement], list[Measurement], dict[str, Any]]:
    boundary_holdout = {
        "boundary_m1_n4096_k65536",
        "boundary_m1_n2048_k65536",
        "boundary_m1_n11008_k8192",
        "boundary_m64_n64_k8192",
        "boundary_m256_n256_k2048",
    }
    train, heldout = [], []
    for row in measurements:
        if row.category == "decision_boundary":
            (heldout if row.workload_id in boundary_holdout else train).append(row)
        elif row.held_out:
            heldout.append(row)
        else:
            train.append(row)
    return train, heldout, {
        "strategy": "canonical_held_out_plus_boundary_grouped_holdout",
        "primary_future_strategy": "leave_one_k_band_out",
        "secondary_future_strategy": "leave_one_n_family_out",
        "boundary_heldout_ids": sorted(boundary_holdout),
    }


def raw_component_units(variant: str, features: dict[str, Any], cand: dict[str, Any]) -> dict[str, float]:
    util_penalty = max(0.0, (1.0 / max(cand["tile_utilization"], 1e-6)) - 1.0)
    parallel_penalty = max(0.0, (64.0 - cand["output_tile_count"]) / 64.0)
    return {
        "fixed_unit": 1.0,
        "launch_unit": float(cand["expected_launch_count"]),
        "compute_unit": features["flops"] / 1e9,
        "memory_unit": cand["estimated_global_bytes"] / 1e9,
        "tile_penalty_unit": util_penalty,
        "parallelism_penalty_unit": parallel_penalty,
    }


def median_ratio(numerators: list[float], denominators: list[float], default: float) -> float:
    vals = [n / d for n, d in zip(numerators, denominators) if d > 0 and n > 0 and math.isfinite(n / d)]
    return statistics.median(vals) if vals else default


def calibrate_model(training: list[Measurement]) -> dict[str, Any]:
    # Deterministic interpretable calibration. The base component weights come
    # from median training latency divided by each component unit and are scaled
    # down so components sum without double counting. Per-candidate affine
    # correction then absorbs remaining systematic error.
    all_lat = []
    all_launch, all_compute, all_memory = [], [], []
    by_variant_units: dict[str, list[tuple[float, dict[str, float]]]] = {"V1": [], "V3": []}
    for row in training:
        features = shape_features(row.shape, row.config)
        for variant in ("V1", "V3"):
            cand = candidate_features(variant, features, row.config)
            units = raw_component_units(variant, features, cand)
            latency = row.latencies[variant]
            all_lat.append(latency)
            all_launch.append(units["launch_unit"])
            all_compute.append(units["compute_unit"])
            all_memory.append(units["memory_unit"])
            by_variant_units[variant].append((latency, units))
    coeff = {
        "fixed_ms": max(0.0, statistics.median(all_lat) * 0.02) if all_lat else 0.0,
        "launch_ms_per_launch": max(0.0, median_ratio(all_lat, all_launch, 0.01) * 0.02),
        "compute_ms_per_gflop": max(0.0, median_ratio(all_lat, all_compute, 0.1) * 0.55),
        "memory_ms_per_gbyte": max(0.0, median_ratio(all_lat, all_memory, 0.1) * 0.20),
        "tile_penalty_ms": max(0.0, statistics.median(all_lat) * 0.02) if all_lat else 0.0,
        "parallelism_penalty_ms": max(0.0, statistics.median(all_lat) * 0.03) if all_lat else 0.0,
    }
    corrections = {}
    residuals = []
    for variant, rows in by_variant_units.items():
        raw_preds = [component_sum(coeff, units) for _, units in rows]
        actuals = [lat for lat, _ in rows]
        scale = median_ratio(actuals, raw_preds, 1.0)
        corrections[variant] = {"affine_scale": max(0.0, scale), "affine_bias_ms": 0.0}
        for actual, raw in zip(actuals, raw_preds):
            pred = raw * corrections[variant]["affine_scale"]
            residuals.append(abs(pred - actual) / max(actual, 1e-9))
    cv_error = {
        "mean_relative_abs_error": statistics.fmean(residuals) if residuals else 0.0,
        "median_relative_abs_error": statistics.median(residuals) if residuals else 0.0,
        "p95_relative_abs_error": percentile(residuals, 95),
    }
    margin_threshold = max(0.01, cv_error["p95_relative_abs_error"])
    return {
        "formula": "fixed + launch + compute + memory + tile_tail_penalty + low_parallelism_penalty",
        "coefficients": coeff,
        "candidate_affine_corrections": corrections,
        "coefficient_constraints": "non_negative_component_coefficients_and_non_negative_affine_scales",
        "cross_validation_error_proxy": cv_error,
        "confidence_thresholds": {
            "minimum_confident_margin": margin_threshold,
            "tie_relative_difference": 0.01,
        },
    }


def component_sum(coeff: dict[str, float], units: dict[str, float]) -> float:
    return (
        coeff["fixed_ms"] * units["fixed_unit"]
        + coeff["launch_ms_per_launch"] * units["launch_unit"]
        + coeff["compute_ms_per_gflop"] * units["compute_unit"]
        + coeff["memory_ms_per_gbyte"] * units["memory_unit"]
        + coeff["tile_penalty_ms"] * units["tile_penalty_unit"]
        + coeff["parallelism_penalty_ms"] * units["parallelism_penalty_unit"]
    )


def predict_candidate(variant: str, features: dict[str, Any], config: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    cand = candidate_features(variant, features, config)
    units = raw_component_units(variant, features, cand)
    coeff = model["coefficients"]
    components = {
        "fixed_ms": coeff["fixed_ms"] * units["fixed_unit"],
        "launch_ms": coeff["launch_ms_per_launch"] * units["launch_unit"],
        "compute_ms": coeff["compute_ms_per_gflop"] * units["compute_unit"],
        "memory_ms": coeff["memory_ms_per_gbyte"] * units["memory_unit"],
        "tile_penalty_ms": coeff["tile_penalty_ms"] * units["tile_penalty_unit"],
        "parallelism_penalty_ms": coeff["parallelism_penalty_ms"] * units["parallelism_penalty_unit"],
    }
    raw = sum(components.values())
    corr = model["candidate_affine_corrections"][variant]
    predicted = raw * corr["affine_scale"] + corr["affine_bias_ms"]
    return {
        "variant": variant,
        "kernel_id": VARIANT_TO_KERNEL[variant],
        "predicted_latency_ms": predicted,
        "analytical_components": components,
        "candidate_features": cand,
    }


NORMALIZED_DISTANCE_FEATURES = (
    "log2_m",
    "log2_n",
    "log2_k",
    "log2_output_elements",
    "arithmetic_intensity",
    "output_tile_count",
    "k_tile_count",
)


def feature_vector(features: dict[str, Any], stats: dict[str, Any]) -> list[float]:
    values = []
    for key in NORMALIZED_DISTANCE_FEATURES:
        value = float(features[key])
        mean = stats[key]["mean"]
        std = stats[key]["std"] or 1.0
        values.append((value - mean) / std)
    return values


def training_feature_stats(training: list[Measurement]) -> dict[str, Any]:
    rows = [shape_features(row.shape, row.config) for row in training]
    stats = {}
    for key in NORMALIZED_DISTANCE_FEATURES:
        vals = [float(r[key]) for r in rows]
        mean = statistics.fmean(vals) if vals else 0.0
        std = statistics.stdev(vals) if len(vals) > 1 else 1.0
        stats[key] = {"mean": mean, "std": std}
    vectors = [feature_vector(r, stats) for r in rows]
    return {"feature_stats": stats, "training_vectors": vectors}


def ood_analysis(features: dict[str, Any], training: list[Measurement], stats_bundle: dict[str, Any]) -> dict[str, Any]:
    stats = stats_bundle["feature_stats"]
    vector = feature_vector(features, stats)
    best_dist = None
    best_row = None
    for row in training:
        train_vec = feature_vector(shape_features(row.shape, row.config), stats)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vector, train_vec)) / len(vector))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_row = row
    training_dists = []
    for i, row in enumerate(training):
        f = shape_features(row.shape, row.config)
        v = feature_vector(f, stats)
        others = []
        for j, other in enumerate(training):
            if i == j:
                continue
            ov = feature_vector(shape_features(other.shape, other.config), stats)
            others.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(v, ov)) / len(v)))
        if others:
            training_dists.append(min(others))
    p90 = percentile(training_dists, 90) or 1.0
    kind = "interpolation" if best_dist is not None and best_dist <= p90 else "near_boundary_extrapolation"
    if best_dist is not None and best_dist > 2.0 * p90:
        kind = "far_extrapolation"
    return {
        "distance_to_training_distribution": best_dist or 0.0,
        "nearest_training_shape": best_row.workload_id if best_row else None,
        "interpolation_kind": kind,
        "training_distance_p90": p90,
    }


def confidence_decision(shape: dict[str, Any], config: dict[str, Any], model: dict[str, Any],
                        training: list[Measurement], stats_bundle: dict[str, Any]) -> dict[str, Any]:
    features = shape_features(shape, config)
    predictions = {v: predict_candidate(v, features, config, model) for v in ("V1", "V3")}
    v1 = predictions["V1"]["predicted_latency_ms"]
    v3 = predictions["V3"]["predicted_latency_ms"]
    analytical_winner = "V3" if v3 < v1 else "V1"
    predicted_margin = abs(v1 - v3) / max(min(v1, v3), 1e-9)
    ood = ood_analysis(features, training, stats_bundle)
    threshold = model["confidence_thresholds"]["minimum_confident_margin"]
    predicted_tie = predicted_margin <= model["confidence_thresholds"]["tie_relative_difference"]
    far = ood["interpolation_kind"] == "far_extrapolation"
    if predicted_tie:
        level = "low"
        score = max(0.0, 1.0 - predicted_margin / max(threshold, 1e-9))
        final = "V1"
        source = "confidence_aware_fallback"
        reason = "predicted_statistical_tie"
    elif far or predicted_margin < threshold:
        level = "low"
        score = predicted_margin / max(threshold, 1e-9)
        final = "V1"
        source = "confidence_aware_fallback"
        reason = "low_model_confidence"
    else:
        level = "high" if ood["interpolation_kind"] == "interpolation" else "medium"
        score = min(1.0, predicted_margin / max(threshold, 1e-9))
        final = analytical_winner
        source = "analytical_cost_model"
        reason = None
    return {
        "backend": BACKEND,
        "profile_match": "unseen",
        "selection_source": source,
        "selected_variant": final,
        "selected_kernel": VARIANT_TO_KERNEL[final],
        "selected_config": dict(config),
        "selected_latency_ms": predictions[final]["predicted_latency_ms"],
        "predicted_candidates": [predictions["V1"], predictions["V3"]],
        "analytical_winner": analytical_winner,
        "predicted_margin": predicted_margin,
        "predicted_tie": predicted_tie,
        "confidence_level": level,
        "confidence_score": score,
        "fallback_reason": reason,
        **ood,
        "truth_boundary": (
            "conservative_choice_due_to_low_model_confidence"
            if source == "confidence_aware_fallback"
            else "predicted_from_training_profiles_not_directly_measured_for_this_shape"
        ),
    }


def nearest_profile_decision(shape: dict[str, Any], config: dict[str, Any], training: list[Measurement],
                             stats_bundle: dict[str, Any]) -> dict[str, Any]:
    features = shape_features(shape, config)
    ood = ood_analysis(features, training, stats_bundle)
    nearest = next(row for row in training if row.workload_id == ood["nearest_training_shape"])
    winner = "V3" if nearest.latencies["V3"] <= nearest.latencies["V1"] else "V1"
    return {
        "selected_variant": winner,
        "selected_kernel": VARIANT_TO_KERNEL[winner],
        "selection_source": "nearest_profile_candidate",
        **ood,
    }


def policy_decisions(shape: dict[str, Any], config: dict[str, Any], model: dict[str, Any],
                     training: list[Measurement], stats_bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    analytical = confidence_decision(shape, config, model, training, stats_bundle)
    raw_winner = analytical["analytical_winner"]
    return {
        "always_v3": {"selected_variant": "V3", "selected_kernel": V3_KERNEL, "selection_source": "always_v3_baseline"},
        "current_v1_fallback": {"selected_variant": "V1", "selected_kernel": V1_KERNEL, "selection_source": "current_conservative_v1_fallback"},
        "nearest_profile": nearest_profile_decision(shape, config, training, stats_bundle),
        "analytical_winner": {
            **analytical,
            "selection_source": "analytical_cost_model_no_confidence_fallback",
            "selected_variant": raw_winner,
            "selected_kernel": VARIANT_TO_KERNEL[raw_winner],
            "fallback_reason": None,
        },
        "confidence_guided": analytical,
    }


def plan_for_decision(row: Measurement, decision: dict[str, Any], model_sha: str, target: dict[str, Any]) -> dict[str, Any]:
    op = {
        "op_id": "matmul_bias_relu_0",
        "op_type": "MatMulBiasRelu",
        "backend": BACKEND,
        "selected_kernel": decision["selected_kernel"],
        "variant": decision["selected_variant"],
        "kernel_config": decision["selected_config"],
        "shape": {**row.shape},
        "inputs": ["A", "B", "bias"],
        "outputs": ["Y"],
        "selection_source": decision["selection_source"],
        "profile_match": decision["profile_match"],
        "fallback_reason": decision.get("fallback_reason"),
        "predicted_candidates": decision.get("predicted_candidates"),
        "predicted_margin": decision.get("predicted_margin"),
        "confidence_level": decision.get("confidence_level"),
        "confidence_score": decision.get("confidence_score"),
        "distance_to_training_distribution": decision.get("distance_to_training_distribution"),
        "interpolation_kind": decision.get("interpolation_kind"),
        "cost_model_artifact_sha256": model_sha,
        "truth_boundary": decision.get("truth_boundary"),
    }
    return {
        "schema": "runtime_execution_plan",
        "schema_version": 2,
        "mode": "compiler-selection",
        "graph_id": f"{row.workload_id}_bias_relu",
        "workload_id": row.workload_id,
        "backend": BACKEND,
        "target_gpu_identity": {
            "gpu_model": target.get("gpu_model"),
            "compute_capability": target.get("compute_capability"),
        },
        "cost_model_artifact_sha256": model_sha,
        "operations": [op],
    }


def oracle_map(paths: list[Path]) -> dict[str, Measurement]:
    rows: dict[str, Measurement] = {}
    for path in paths:
        if path.exists():
            for row in parse_profile_measurements(path):
                rows[row.workload_id] = row
    return rows


def evaluate_policy(policy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    regrets = [r["regret"] for r in policy_rows if r.get("regret") is not None]
    return {
        "workload_count": len(policy_rows),
        "v1_selections": sum(1 for r in policy_rows if r["selected_variant"] == "V1"),
        "v3_selections": sum(1 for r in policy_rows if r["selected_variant"] == "V3"),
        "fallbacks": sum(1 for r in policy_rows if r.get("fallback_reason")),
        "predicted_ties": sum(1 for r in policy_rows if r.get("predicted_tie")),
        "mean_regret": statistics.fmean(regrets) if regrets else 0.0,
        "median_regret": statistics.median(regrets) if regrets else 0.0,
        "p95_regret": percentile(regrets, 95),
        "within_1_percent": sum(1 for r in policy_rows if (r.get("regret") or 0.0) <= 0.01) / max(len(policy_rows), 1),
        "within_3_percent": sum(1 for r in policy_rows if (r.get("regret") or 0.0) <= 0.03) / max(len(policy_rows), 1),
    }


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Triton Shape-Aware Kernel Selection",
        "",
        "This report evaluates shape-aware analytical decisions for unseen Triton V1/V3 workloads. It does not add kernels or train an opaque model.",
        "",
        "## Policy Comparison",
        "",
        "| Policy | V1 selections | V3 selections | Fallbacks | Mean regret | Median regret | P95 regret | Within 1% | Within 3% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, agg in summary["policy_aggregates"].items():
        lines.append(
            f"| {policy} | {agg['v1_selections']} | {agg['v3_selections']} | {agg['fallbacks']} | "
            f"{agg['mean_regret']:.6f} | {agg['median_regret']:.6f} | {agg['p95_regret']:.6f} | "
            f"{agg['within_1_percent']:.4f} | {agg['within_3_percent']:.4f} |"
        )
    lines += [
        "",
        "## Per-Workload Confidence-Guided Decisions",
        "",
        "| Workload | M/N/K | V1 pred | V3 pred | Winner | Confidence | Final | Fallback | Oracle | Regret |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    for row in summary["workloads"]:
        d = row["decisions"]["confidence_guided"]
        preds = {p["variant"]: p for p in d["predicted_candidates"]}
        lines.append(
            f"| {row['workload_id']} | {row['shape']['m']}/{row['shape']['n']}/{row['shape']['k']} | "
            f"{preds['V1']['predicted_latency_ms']:.6f} | {preds['V3']['predicted_latency_ms']:.6f} | "
            f"{d['analytical_winner']} | {d['confidence_level']} | {d['selected_variant']} | "
            f"{d.get('fallback_reason')} | {row.get('oracle_variant')} | {row['policy_results']['confidence_guided']['regret']:.6f} |"
        )
    lines += [
        "",
        "## Collapse Audit",
        "",
        f"- analytical collapsed to always V3: `{summary['collapse_audit']['analytical_collapsed_to_always_v3']}`",
        f"- confidence-guided collapsed to always V3: `{summary['collapse_audit']['confidence_guided_collapsed_to_always_v3']}`",
        "",
        "## Truth Boundary",
        "",
        "Predictions are calibrated from permitted training profiles and are not runtime latency guarantees. Low-confidence V1 selections are conservative fallbacks, not predicted V1 performance wins.",
    ]
    return "\n".join(lines) + "\n"


def build_doc(summary: dict[str, Any], model: dict[str, Any]) -> str:
    return build_report(summary) + "\n## Feature Schema\n\n" + json.dumps(model["feature_schema"], indent=2) + "\n"


def run_plan_validation(runner: Path, plans_payload: dict[str, Any], output: Path,
                        work_dir: Path, warmup: int, iterations: int, repeats: int) -> dict[str, Any]:
    run_dir = output.parent / "matmul_postop_triton_analytical_plan_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for plan in plans_payload["plans"]:
        plan_path = run_dir / f"{plan['workload_id']}.plan.json"
        out_path = run_dir / f"{plan['workload_id']}.use_plan.json"
        write_json(plan_path, plan)
        subprocess.run(
            [
                sys.executable,
                str(runner),
                "--mode",
                "use-plan",
                "--execution-plan",
                str(plan_path),
                "--warmup",
                str(warmup),
                "--iterations",
                str(iterations),
                "--repeats",
                str(repeats),
                "--output",
                str(out_path),
                "--report-output",
                str(run_dir / f"{plan['workload_id']}.md"),
            ],
            cwd=work_dir,
            check=True,
        )
        records.append(read_json(out_path)["workloads"][0])
    payload = {
        "schema": "triton_matmul_bias_relu_analytical_plan_validation",
        "schema_version": SCHEMA_VERSION,
        "mode": "use-plan",
        "backend": BACKEND,
        "workloads": records,
        "aggregate": {
            "workload_count": len(records),
            "planned_kernel_equals_actual_rate": sum(1 for r in records if r["planned_equals_actual"]) / max(len(records), 1),
            "planned_config_equals_actual_rate": sum(1 for r in records if r["config_equals_actual"]) / max(len(records), 1),
            "correctness_pass_rate": sum(1 for r in records if r["correctness"]["passed"]) / max(len(records), 1),
        },
        "utc_start": utc_now(),
        "utc_end": utc_now(),
    }
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="benchmarks/matmul_postop_workloads.json")
    p.add_argument("--profile", action="append", default=[])
    p.add_argument("--oracle", action="append", default=[])
    p.add_argument("--runner", default="tools/run_triton_matmul_bias_relu_benchmark.py")
    p.add_argument("--cost-model-output", default="trace/matmul_postop_triton_analytical_cost_model.json")
    p.add_argument("--plans-output", default="trace/matmul_postop_triton_analytical_selection_plans.json")
    p.add_argument("--plan-validation-output", default="trace/matmul_postop_triton_analytical_plan_validation.json")
    p.add_argument("--fresh-oracle-output", default="trace/matmul_postop_triton_analytical_fresh_oracle.json")
    p.add_argument("--summary-output", default="trace/matmul_postop_triton_analytical_selection_summary.json")
    p.add_argument("--report-output", default="trace/matmul_postop_triton_analytical_selection_report.md")
    p.add_argument("--doc-output", default="DOC/result/TRITON_SHAPE_AWARE_KERNEL_SELECTION.md")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--skip-use-plan", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 50 or args.iterations < 300 or args.repeats < 5:
        raise ValueError("formal analytical selection requires warmup>=50 iterations>=300 repeats>=5")
    profiles = [Path(p) for p in args.profile] or [
        Path("trace/matmul_postop_triton_fixed_config_profile.json"),
        Path("trace/matmul_postop_triton_decision_boundary_profile.json"),
    ]
    measurements = [row for path in profiles if path.exists() for row in parse_profile_measurements(path)]
    train, heldout, split = split_measurements(measurements)
    if not train or not heldout:
        raise ValueError("analytical selection requires non-empty training and held-out measurements")
    model = calibrate_model(train)
    stats_bundle = training_feature_stats(train)
    first_payload = read_json(profiles[0])
    target = profile_target_environment(first_payload)
    model_payload = {
        "schema": "triton_matmul_bias_relu_analytical_cost_model",
        "schema_version": SCHEMA_VERSION,
        "backend": BACKEND,
        "target_gpu_identity": target,
        "fixed_config": heldout[0].config,
        "feature_schema": {
            "shape_features": list(shape_features(heldout[0].shape, heldout[0].config).keys()),
            "candidate_features": list(candidate_features("V1", shape_features(heldout[0].shape, heldout[0].config), heldout[0].config).keys()),
            "forbidden_model_fields": sorted(FORBIDDEN_MODEL_FIELDS),
        },
        "training_workload_ids": [r.workload_id for r in train],
        "held_out_workload_ids": [r.workload_id for r in heldout],
        "grouped_split_definition": split,
        "analytical_model": model,
        "training_profile_sha256": {str(p): sha256_file(p) for p in profiles if p.exists()},
        "truth_boundary": "interpretable_analytical_model_calibrated_from_training_profiles_not_runtime_latency_guarantee",
        "utc_start": utc_now(),
        "utc_end": utc_now(),
    }
    cost_model_path = Path(args.cost_model_output)
    write_json(cost_model_path, model_payload)
    model_sha = sha256_file(cost_model_path)

    decisions = []
    plans = []
    oracle_sources = [Path(p) for p in args.oracle] or profiles
    oracles = oracle_map(oracle_sources)
    policy_rows: dict[str, list[dict[str, Any]]] = {k: [] for k in ("always_v3", "current_v1_fallback", "nearest_profile", "analytical_winner", "confidence_guided")}
    for row in heldout:
        decisions_by_policy = policy_decisions(row.shape, row.config, model, train, stats_bundle)
        oracle = oracles.get(row.workload_id, row)
        oracle_kernel, oracle_latency = oracle_winner({"variants": {
            "V1": {"kernel_id": V1_KERNEL, "correctness": {"passed": True}, "timing": {"statistics": {"median_ms": oracle.latencies["V1"]}}},
            "V3": {"kernel_id": V3_KERNEL, "correctness": {"passed": True}, "timing": {"statistics": {"median_ms": oracle.latencies["V3"]}}},
        }})
        oracle_variant = KERNEL_TO_VARIANT[oracle_kernel] if oracle_kernel else None
        policy_result_map = {}
        for policy, decision in decisions_by_policy.items():
            selected_variant = decision["selected_variant"]
            selected_latency = oracle.latencies[selected_variant]
            regret = selected_latency / oracle_latency - 1.0 if oracle_latency else None
            result = {
                "workload_id": row.workload_id,
                "selected_variant": selected_variant,
                "selected_kernel": VARIANT_TO_KERNEL[selected_variant],
                "oracle_variant": oracle_variant,
                "regret": regret,
                "within_1_percent": regret is not None and regret <= 0.01,
                "within_3_percent": regret is not None and regret <= 0.03,
                "fallback_reason": decision.get("fallback_reason"),
                "predicted_tie": decision.get("predicted_tie", False),
            }
            policy_rows[policy].append(result)
            policy_result_map[policy] = result
        confidence = decisions_by_policy["confidence_guided"]
        decisions.append(
            {
                "workload_id": row.workload_id,
                "shape": row.shape,
                "category": row.category,
                "classification": row.classification,
                "exact_or_unseen": "unseen",
                "decisions": decisions_by_policy,
                "oracle_variant": oracle_variant,
                "oracle_latencies": oracle.latencies,
                "policy_results": policy_result_map,
            }
        )
        plans.append(plan_for_decision(row, confidence, model_sha, target))

    plans_payload = {
        "schema": "triton_matmul_bias_relu_analytical_selection_plans",
        "schema_version": SCHEMA_VERSION,
        "mode": "compiler-selection",
        "backend": BACKEND,
        "cost_model_artifact_sha256": model_sha,
        "plans": plans,
        "selections": decisions,
        "utc_start": utc_now(),
        "utc_end": utc_now(),
    }
    write_json(Path(args.plans_output), plans_payload)
    validation = None
    if not args.skip_use_plan:
        validation = run_plan_validation(Path(args.runner), plans_payload, Path(args.plan_validation_output), Path.cwd(), args.warmup, args.iterations, args.repeats)
    fresh_oracle_payload = {
        "schema": "triton_matmul_bias_relu_analytical_fresh_oracle",
        "schema_version": SCHEMA_VERSION,
        "mode": "fresh-oracle-summary",
        "backend": BACKEND,
        "oracle_sources": [str(p) for p in oracle_sources],
        "truth_boundary": "oracle_latencies_loaded_from_independent_measurement_artifacts_not_used_for_training",
        "workloads": [
            {"workload_id": r.workload_id, "latencies": oracles.get(r.workload_id, r).latencies}
            for r in heldout
        ],
        "utc_start": utc_now(),
        "utc_end": utc_now(),
    }
    write_json(Path(args.fresh_oracle_output), fresh_oracle_payload)
    policy_aggregates = {name: evaluate_policy(rows) for name, rows in policy_rows.items()}
    summary = {
        "schema": "triton_matmul_bias_relu_analytical_selection_summary",
        "schema_version": SCHEMA_VERSION,
        "mode": "aggregate-report",
        "backend": BACKEND,
        "cost_model_artifact_sha256": model_sha,
        "plan_validation_aggregate": (validation or {}).get("aggregate"),
        "policy_aggregates": policy_aggregates,
        "workloads": decisions,
        "decision_diversity": {
            "confidence_guided_v1_decisions": policy_aggregates["confidence_guided"]["v1_selections"],
            "confidence_guided_v3_decisions": policy_aggregates["confidence_guided"]["v3_selections"],
            "predicted_v1_performance_wins": sum(1 for row in decisions if row["decisions"]["confidence_guided"]["analytical_winner"] == "V1" and row["decisions"]["confidence_guided"]["selection_source"] == "analytical_cost_model"),
            "conservative_v1_fallbacks": policy_aggregates["confidence_guided"]["fallbacks"],
            "predicted_ties": policy_aggregates["confidence_guided"]["predicted_ties"],
        },
        "collapse_audit": {
            "analytical_collapsed_to_always_v3": policy_aggregates["analytical_winner"]["v1_selections"] == 0,
            "confidence_guided_collapsed_to_always_v3": policy_aggregates["confidence_guided"]["v1_selections"] == 0,
        },
        "utc_start": utc_now(),
        "utc_end": utc_now(),
    }
    write_json(Path(args.summary_output), summary)
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(build_report(summary), encoding="utf-8")
    Path(args.doc_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc_output).write_text(build_doc(summary, model_payload), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
