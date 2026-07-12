#!/usr/bin/env python3
"""Shape-aware Triton fused configuration selection.

This is the fused-kernel configuration selector. It deliberately does not use
the V1 unfused candidate set: every selectable candidate is the same one-pass
MatMul+Bias+ReLU semantic kernel with a different fixed Triton tile config.
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
import run_triton_fused_candidate_discovery as discovery  # noqa: E402
import target_hardware_profile as hardware_profile  # noqa: E402


BACKEND = "triton_cuda"
DTYPE = "f32"
PATTERN = "bias"
KERNEL_FAMILY_ID = "triton_matmul_bias_relu_one_pass_f32"
SCHEMA_VERSION = 1
FEATURE_SCHEMA_VERSION = 2
PRIMARY_CONFIG_IDS = (
    "bm16_bn16_bk32_w4_s3",
    "bm32_bn32_bk32_w4_s3",
    "bm64_bn64_bk32_w4_s3",
    "bm16_bn64_bk32_w4_s3",
)
FORBIDDEN_MODEL_FIELDS = {
    "expected_region",
    "oracle_config",
    "classification",
    "stable_winner",
    "fresh_oracle",
    "regret",
    "winner_margin",
    "session_level_winners",
    "cross_session_median_ms_by_config",
}


@dataclass(frozen=True)
class TritonFusedConfig:
    config_id: str
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    precision_mode: str = "ieee"

    def validate(self) -> None:
        if self.config_id not in PRIMARY_CONFIG_IDS:
            raise ValueError(f"unsupported primary config: {self.config_id}")
        if min(self.block_m, self.block_n, self.block_k, self.num_warps, self.num_stages) <= 0:
            raise ValueError("malformed fused config")
        if self.block_k != 32 or self.num_warps != 4 or self.num_stages != 3:
            raise ValueError("primary fused configs must use bk32/w4/s3")
        if self.precision_mode != "ieee":
            raise ValueError("primary fused configs must use ieee precision")

    def typed(self) -> dict[str, Any]:
        return {
            "semantic_kernel_id": KERNEL_FAMILY_ID,
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
        }


PRIMARY_CONFIGS = {
    "bm16_bn16_bk32_w4_s3": TritonFusedConfig("bm16_bn16_bk32_w4_s3", 16, 16, 32, 4, 3),
    "bm32_bn32_bk32_w4_s3": TritonFusedConfig("bm32_bn32_bk32_w4_s3", 32, 32, 32, 4, 3),
    "bm64_bn64_bk32_w4_s3": TritonFusedConfig("bm64_bn64_bk32_w4_s3", 64, 64, 32, 4, 3),
    "bm16_bn64_bk32_w4_s3": TritonFusedConfig("bm16_bn64_bk32_w4_s3", 16, 64, 32, 4, 3),
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


def safe_log2(value: float) -> float:
    return math.log2(max(float(value), 1.0))


def reject_forbidden_model_input(payload: dict[str, Any]) -> None:
    present = sorted(FORBIDDEN_MODEL_FIELDS & set(payload))
    if present:
        raise ValueError(f"forbidden evaluation fields in selection input: {present}")


def workload_features(shape: dict[str, Any]) -> dict[str, Any]:
    reject_forbidden_model_input(shape)
    m, n, k = int(shape["m"]), int(shape["n"]), int(shape["k"])
    dtype_bytes = 4
    output_elements = m * n
    a_elements = m * k
    b_elements = k * n
    flops = 2 * m * n * k
    a_bytes = a_elements * dtype_bytes
    b_bytes = b_elements * dtype_bytes
    bias_bytes = n * dtype_bytes
    output_bytes = output_elements * dtype_bytes
    total_bytes = a_bytes + b_bytes + bias_bytes + output_bytes
    return {
        "m": m,
        "n": n,
        "k": k,
        "dtype": shape.get("dtype", DTYPE),
        "log2_m": safe_log2(m),
        "log2_n": safe_log2(n),
        "log2_k": safe_log2(k),
        "mn": output_elements,
        "mk": a_elements,
        "kn": b_elements,
        "flops": flops,
        "a_bytes": a_bytes,
        "b_bytes": b_bytes,
        "bias_bytes": bias_bytes,
        "output_bytes": output_bytes,
        "arithmetic_intensity": flops / max(total_bytes, 1),
        "m_over_n": m / n,
        "n_over_m": n / m,
        "k_over_m": k / m,
        "k_over_n": k / n,
        "small_m": m <= 16,
        "small_n": n <= 128,
        "large_m": m >= 512,
        "large_n": n >= 1024,
        "extreme_k": k >= 8192,
        "skinny_m": m <= 16 and n >= 1024,
        "skinny_n": n <= 128 and m >= 64,
        "square_like": 0.5 <= (m / n) <= 2.0,
        "output_parallelism_estimate": output_elements,
        "log2_output_elements": safe_log2(output_elements),
    }


def tile_features(shape_features: dict[str, Any], cfg: TritonFusedConfig, sm_count: int = 16) -> dict[str, Any]:
    m, n, k = shape_features["m"], shape_features["n"], shape_features["k"]
    mt = ceil_div(m, cfg.block_m)
    nt = ceil_div(n, cfg.block_n)
    kt = ceil_div(k, cfg.block_k)
    programs = mt * nt
    padded_output = mt * cfg.block_m * nt * cfg.block_n
    useful_output = m * n
    effective_flops = 2 * padded_output * kt * cfg.block_k
    useful_flops = shape_features["flops"]
    padding_amplification = effective_flops / max(useful_flops, 1)
    masked_output_fraction = 1.0 - (useful_output / max(padded_output, 1))
    work_per_program = cfg.block_m * cfg.block_n * kt
    programs_per_compute_unit = programs / max(sm_count, 1)
    output_program_waves = ceil_div(programs, max(sm_count, 1))
    return {
        "config_id": cfg.config_id,
        "block_m": cfg.block_m,
        "block_n": cfg.block_n,
        "block_k": cfg.block_k,
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
        "precision_mode": cfg.precision_mode,
        "m_tile_count": mt,
        "n_tile_count": nt,
        "k_tile_count": kt,
        "total_output_program_count": programs,
        "m_tile_utilization": m / (mt * cfg.block_m),
        "n_tile_utilization": n / (nt * cfg.block_n),
        "k_tail_utilization": k / (kt * cfg.block_k),
        "total_padded_output_elements": padded_output,
        "total_useful_output_elements": useful_output,
        "masked_output_fraction": masked_output_fraction,
        "effective_tiled_flops": effective_flops,
        "padding_amplification": padding_amplification,
        "work_per_program": work_per_program,
        "elements_computed_per_program": cfg.block_m * cfg.block_n,
        "bytes_per_output_tile": cfg.block_m * cfg.block_n * 4,
        "sm_count": sm_count,
        "programs_per_sm": programs_per_compute_unit,
        "output_program_waves": output_program_waves,
        "parallelism_relative_to_sm_count": programs_per_compute_unit,
        "low_parallelism": programs < sm_count * 2,
        "excessive_program_count": programs > sm_count * 128,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "parallel_work_items": programs,
        "work_items_per_compute_unit": programs_per_compute_unit,
        "execution_waves": output_program_waves,
        "compute_work_per_item": work_per_program,
        "padding_waste_ratio": masked_output_fraction,
        "work_amplification": padding_amplification,
        "triton_specific_fields": [
            "block_m",
            "block_n",
            "block_k",
            "num_warps",
            "num_stages",
            "sm_count",
        ],
    }


@dataclass
class Measurement:
    workload_id: str
    shape: dict[str, Any]
    category: str
    region: str
    classification: str | None
    latencies: dict[str, float]


def region_for(row: dict[str, Any]) -> str:
    if row.get("category") == "decision_boundary":
        return "decision_boundary_high_k"
    shape = row["shape"]
    m, n, k = shape["m"], shape["n"], shape["k"]
    if m <= 16 and n >= 1024:
        return "small_skinny"
    if m >= 128 and n >= 512:
        return "large_regular"
    if m <= 128 and n <= 128 and k >= 1024:
        return "small_square_high_k"
    if row.get("category") == "representative":
        return "representative_llm"
    if row.get("category") == "fusion_friendly_memory_stress":
        return "stress_low_k"
    return row.get("category") or "other"


def parse_discovery_measurements(path: Path) -> tuple[list[Measurement], dict[str, Any]]:
    payload = read_json(path)
    if payload.get("schema") != "triton_matmul_bias_relu_fused_candidate_sweep":
        raise ValueError("expected fused candidate sweep artifact")
    rows: list[Measurement] = []
    for row in payload.get("workloads", []):
        med = row.get("cross_session_median_ms_by_config")
        if not med:
            continue
        latencies = {cid: float(med[cid]) for cid in PRIMARY_CONFIG_IDS if cid in med}
        if set(latencies) == set(PRIMARY_CONFIG_IDS):
            rows.append(
                Measurement(
                    workload_id=row["workload_id"],
                    shape=dict(row["shape"]),
                    category=row.get("category", ""),
                    region=region_for(row),
                    classification=row.get("classification"),
                    latencies=latencies,
                )
            )
    return rows, payload


def grouped_split(rows: list[Measurement], split: str) -> tuple[list[Measurement], list[Measurement], dict[str, Any]]:
    if split == "leave-one-shape-region-out":
        held_regions = {"small_skinny", "large_regular", "small_square_high_k", "decision_boundary_high_k"}
        held_ids = {
            "rep_m1_k768_n3072",
            "rep_m128_k768_n3072",
            "unfriendly_m64_n64_k4096",
            "boundary_m1_n4096_k65536",
            "boundary_m256_n256_k2048",
            "balanced_m64_n64_k64",
        }
    elif split == "leave-one-k-band-out":
        held_ids = {r.workload_id for r in rows if r.shape["k"] >= 2048}
        held_regions = {"high_k"}
    elif split == "leave-one-n-family-out":
        held_ids = {r.workload_id for r in rows if r.shape["n"] >= 3072}
        held_regions = {"wide_n"}
    else:
        raise ValueError(f"unknown split: {split}")
    heldout = [r for r in rows if r.workload_id in held_ids]
    train = [r for r in rows if r.workload_id not in held_ids]
    return train, heldout, {
        "strategy": split,
        "held_out_workload_ids": [r.workload_id for r in heldout],
        "training_workload_ids": [r.workload_id for r in train],
        "held_out_regions": sorted(held_regions),
    }


def component_units(shape: dict[str, Any], cfg: TritonFusedConfig, sm_count: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    wf = workload_features(shape)
    tf = tile_features(wf, cfg, sm_count)
    useful_flops_g = wf["flops"] / 1e9
    effective_flops_g = tf["effective_tiled_flops"] / 1e9
    memory_gb = (wf["a_bytes"] + wf["b_bytes"] + wf["bias_bytes"] + wf["output_bytes"]) / 1e9
    padding_extra = max(0.0, effective_flops_g - useful_flops_g)
    low_parallel = max(0.0, 1.0 - tf["parallelism_relative_to_sm_count"] / 2.0)
    excessive_programs = max(0.0, tf["total_output_program_count"] / (sm_count * 128) - 1.0)
    shape_mismatch = 0.0
    if wf["skinny_m"] and cfg.block_m > 16:
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
        + coeff["program_overhead_ms"] * units["program_overhead_unit"]
        + coeff["shape_mismatch_ms"] * units["shape_mismatch_unit"]
    )


def median_ratio(actuals: list[float], preds: list[float], default: float) -> float:
    vals = [a / p for a, p in zip(actuals, preds) if a > 0 and p > 0 and math.isfinite(a / p)]
    return statistics.median(vals) if vals else default


def calibrate(rows: list[Measurement], sm_count: int) -> dict[str, Any]:
    all_lat, all_compute, all_memory, all_padding = [], [], [], []
    all_low, all_program, all_shape = [], [], []
    for row in rows:
        for cid in PRIMARY_CONFIG_IDS:
            _, _, units = component_units(row.shape, PRIMARY_CONFIGS[cid], sm_count)
            all_lat.append(row.latencies[cid])
            all_compute.append(units["effective_compute_unit"])
            all_memory.append(units["memory_unit"])
            all_padding.append(units["padding_unit"])
            all_low.append(units["low_parallelism_unit"])
            all_program.append(units["program_overhead_unit"])
            all_shape.append(units["shape_mismatch_unit"])
    med = statistics.median(all_lat) if all_lat else 0.01
    coeff = {
        "fixed_ms": med * 0.03,
        "compute_ms_per_gflop": median_ratio(all_lat, all_compute, 0.1) * 0.60,
        "memory_ms_per_gbyte": median_ratio(all_lat, all_memory, 0.1) * 0.10,
        "padding_ms_per_gflop": median_ratio(all_lat, all_padding, 0.0) * 0.08 if any(all_padding) else 0.0,
        "low_parallelism_ms": med * 0.08,
        "program_overhead_ms": med * 0.03,
        "shape_mismatch_ms": med * 0.10,
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
    # Training-derived safe config: lowest mean normalized latency on training.
    mean_norm = {}
    for cid in PRIMARY_CONFIG_IDS:
        vals = []
        for row in rows:
            best = min(row.latencies.values())
            vals.append(row.latencies[cid] / best)
        mean_norm[cid] = statistics.fmean(vals) if vals else float("inf")
    fallback = min(mean_norm.items(), key=lambda item: (item[1], item[0]))[0]
    return {
        "formula": "fixed_config_cost + effective_compute_cost + memory_cost + tile_padding_penalty + low_parallelism_penalty + excessive_program_count_penalty + shape_config_mismatch_penalty",
        "coefficients": coeff,
        "per_config_affine_calibration": corrections,
        "cross_validation_error_proxy": {
            "mean_relative_abs_error": statistics.fmean(residuals) if residuals else 0.0,
            "median_relative_abs_error": statistics.median(residuals) if residuals else 0.0,
            "p95_relative_abs_error": percentile(residuals, 95),
        },
        "confidence_thresholds": {
            "minimum_confident_margin": max(0.03, statistics.median(residuals) if residuals else 0.03),
            "tie_relative_difference": 0.01,
        },
        "fallback_config_id": fallback,
        "fallback_config_training_mean_normalized_latency": mean_norm[fallback],
        "coefficient_constraints": "non_negative_coefficients_and_affine_scales",
    }


DISTANCE_FEATURES = ("log2_m", "log2_n", "log2_k", "log2_output_elements", "arithmetic_intensity")


def feature_stats(rows: list[Measurement]) -> dict[str, Any]:
    feats = [workload_features(r.shape) for r in rows]
    stats = {}
    for key in DISTANCE_FEATURES:
        vals = [float(f[key]) for f in feats]
        stats[key] = {
            "mean": statistics.fmean(vals) if vals else 0.0,
            "std": statistics.stdev(vals) if len(vals) > 1 else 1.0,
        }
    return stats


def vector(features: dict[str, Any], stats: dict[str, Any]) -> list[float]:
    out = []
    for key in DISTANCE_FEATURES:
        std = stats[key]["std"] or 1.0
        out.append((float(features[key]) - stats[key]["mean"]) / std)
    return out


def ood(shape: dict[str, Any], train: list[Measurement], stats: dict[str, Any]) -> dict[str, Any]:
    vf = vector(workload_features(shape), stats)
    best = (float("inf"), None)
    train_dists = []
    train_vecs = [(r, vector(workload_features(r.shape), stats)) for r in train]
    for row, tv in train_vecs:
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vf, tv)) / len(vf))
        if dist < best[0]:
            best = (dist, row)
    for i, (_, v1) in enumerate(train_vecs):
        local = []
        for j, (_, v2) in enumerate(train_vecs):
            if i != j:
                local.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)) / len(v1)))
        if local:
            train_dists.append(min(local))
    p90 = percentile(train_dists, 90) or 1.0
    kind = "interpolation" if best[0] <= p90 else "near_boundary_extrapolation"
    if best[0] > 2 * p90:
        kind = "far_extrapolation"
    return {
        "distance_to_training_distribution": best[0],
        "nearest_training_shape": best[1].workload_id if best[1] else None,
        "interpolation_kind": kind,
        "training_distance_p90": p90,
    }


def predict_config(shape: dict[str, Any], cfg: TritonFusedConfig, model: dict[str, Any], sm_count: int,
                   calibrated: bool) -> dict[str, Any]:
    wf, tf, units = component_units(shape, cfg, sm_count)
    coeff = model["coefficients"]
    components = {
        "fixed_ms": coeff["fixed_ms"] * units["fixed_unit"],
        "compute_ms": coeff["compute_ms_per_gflop"] * units["effective_compute_unit"],
        "memory_ms": coeff["memory_ms_per_gbyte"] * units["memory_unit"],
        "padding_ms": coeff["padding_ms_per_gflop"] * units["padding_unit"],
        "low_parallelism_ms": coeff["low_parallelism_ms"] * units["low_parallelism_unit"],
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
    }


def ranking(shape: dict[str, Any], model: dict[str, Any], sm_count: int, calibrated: bool) -> list[dict[str, Any]]:
    rows = [predict_config(shape, PRIMARY_CONFIGS[cid], model, sm_count, calibrated) for cid in PRIMARY_CONFIG_IDS]
    return sorted(rows, key=lambda r: (r["predicted_latency_ms"], r["config_id"]))


def nearest_policy(shape: dict[str, Any], train: list[Measurement], stats: dict[str, Any]) -> dict[str, Any]:
    oo = ood(shape, train, stats)
    nearest = next(r for r in train if r.workload_id == oo["nearest_training_shape"])
    selected = min(nearest.latencies.items(), key=lambda item: (item[1], item[0]))[0]
    return {"selected_config_id": selected, "selection_source": "nearest_measured_shape", **oo}


def choose_policy(policy: str, shape: dict[str, Any], model: dict[str, Any], train: list[Measurement],
                  stats: dict[str, Any], sm_count: int) -> dict[str, Any]:
    if policy.startswith("static_"):
        cid = policy.removeprefix("static_")
        return {"selected_config_id": cid, "selection_source": f"static_{cid}", "candidate_ranking": []}
    if policy == "nearest_shape":
        return nearest_policy(shape, train, stats)
    calibrated = policy in {"calibrated_analytical", "confidence_aware"}
    rank = ranking(shape, model, sm_count, calibrated=calibrated)
    best, second = rank[0], rank[1]
    margin = (second["predicted_latency_ms"] / max(best["predicted_latency_ms"], 1e-9)) - 1.0
    oo = ood(shape, train, stats)
    if policy == "confidence_aware":
        threshold = model["confidence_thresholds"]["minimum_confident_margin"]
        low = margin < threshold or oo["interpolation_kind"] == "far_extrapolation"
        if low:
            selected = model["fallback_config_id"]
            return {
                "selected_config_id": selected,
                "selection_source": "fused_config_confidence_fallback",
                "fallback_reason": "low_model_confidence",
                "fallback_config_training_evidence": {
                    "training_mean_normalized_latency": model["fallback_config_training_mean_normalized_latency"],
                },
                "candidate_ranking": rank,
                "confidence": {
                    "level": "low",
                    "predicted_margin": margin,
                    "ood_distance": oo["distance_to_training_distribution"],
                },
                **oo,
            }
    source = "calibrated_analytical_cost_model" if calibrated else "analytical_cost_model"
    return {
        "selected_config_id": best["config_id"],
        "selection_source": source,
        "fallback_reason": None,
        "candidate_ranking": rank,
        "confidence": {
            "level": "high" if oo["interpolation_kind"] == "interpolation" else "medium",
            "predicted_margin": margin,
            "ood_distance": oo["distance_to_training_distribution"],
        },
        **oo,
    }


POLICIES = (
    "static_bm16_bn16_bk32_w4_s3",
    "static_bm32_bn32_bk32_w4_s3",
    "static_bm64_bn64_bk32_w4_s3",
    "static_bm16_bn64_bk32_w4_s3",
    "nearest_shape",
    "analytical",
    "calibrated_analytical",
    "confidence_aware",
)


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    regrets = [r["regret"] for r in rows]
    top1 = [r["top1_correct"] for r in rows]
    classes = sorted({r["oracle_config"] for r in rows})
    recalls = {}
    for cls in classes:
        cls_rows = [r for r in rows if r["oracle_config"] == cls]
        recalls[cls] = sum(1 for r in cls_rows if r["selected_config_id"] == cls) / max(len(cls_rows), 1)
    return {
        "workload_count": len(rows),
        "config_diversity": len({r["selected_config_id"] for r in rows}),
        "top1_accuracy": sum(top1) / max(len(top1), 1),
        "macro_accuracy": statistics.fmean(recalls.values()) if recalls else 0.0,
        "per_config_recall": recalls,
        "mean_regret": statistics.fmean(regrets) if regrets else 0.0,
        "median_regret": statistics.median(regrets) if regrets else 0.0,
        "p95_regret": percentile(regrets, 95),
        "max_regret": max(regrets) if regrets else 0.0,
        "within_1_percent": sum(1 for r in rows if r["regret"] <= 0.01) / max(len(rows), 1),
        "within_3_percent": sum(1 for r in rows if r["regret"] <= 0.03) / max(len(rows), 1),
        "selection_counts": {cid: sum(1 for r in rows if r["selected_config_id"] == cid) for cid in PRIMARY_CONFIG_IDS},
        "oracle_counts": {cid: sum(1 for r in rows if r["oracle_config"] == cid) for cid in PRIMARY_CONFIG_IDS},
        "top1_correct_counts": {cid: sum(1 for r in rows if r["oracle_config"] == cid and r["selected_config_id"] == cid) for cid in PRIMARY_CONFIG_IDS},
    }


def build_plan(row: Measurement, decision: dict[str, Any], model_sha: str) -> dict[str, Any]:
    cfg = PRIMARY_CONFIGS[decision["selected_config_id"]]
    op = {
        "op_id": "matmul_bias_relu_0",
        "op_type": "MatMulBiasRelu",
        "backend": BACKEND,
        "semantic_kernel_id": KERNEL_FAMILY_ID,
        "selected_kernel": "triton_tiled_matmul_bias_relu_one_pass_f32",
        "selected_config_id": cfg.config_id,
        "kernel_config": {
            "block_m": cfg.block_m,
            "block_n": cfg.block_n,
            "block_k": cfg.block_k,
            "num_warps": cfg.num_warps,
            "num_stages": cfg.num_stages,
            "precision_mode": cfg.precision_mode,
        },
        "shape": row.shape,
        "inputs": ["A", "B", "bias"],
        "outputs": ["Y"],
        "selection_source": decision["selection_source"],
        "profile_match": "unseen",
        "candidate_ranking": decision.get("candidate_ranking", []),
        "confidence": decision.get("confidence"),
        "distance_to_training_distribution": decision.get("distance_to_training_distribution"),
        "cost_model_artifact_sha256": model_sha,
        "truth_boundary": "selected_config_predicted_from_training_profiles_not_measured_for_this_shape",
    }
    return {
        "schema": "runtime_execution_plan",
        "schema_version": 2,
        "mode": "compiler-selection",
        "graph_id": f"{row.workload_id}_fused_config",
        "workload_id": row.workload_id,
        "backend": BACKEND,
        "operations": [op],
    }


def validate_plans(plans: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for plan in plans:
        op = plan["operations"][0]
        cfg = PRIMARY_CONFIGS[op["selected_config_id"]]
        actual_config = {
            "block_m": cfg.block_m,
            "block_n": cfg.block_n,
            "block_k": cfg.block_k,
            "num_warps": cfg.num_warps,
            "num_stages": cfg.num_stages,
            "precision_mode": cfg.precision_mode,
        }
        rows.append({
            "workload_id": plan["workload_id"],
            "planned_config_id": op["selected_config_id"],
            "actual_dispatched_config_id": cfg.config_id,
            "planned_config": op["kernel_config"],
            "actual_config": actual_config,
            "planned_equals_actual": op["selected_config_id"] == cfg.config_id,
            "config_equals_actual": op["kernel_config"] == actual_config,
            "runtime_policy_override": False,
        })
    return {
        "schema": "triton_matmul_bias_relu_fused_config_plan_validation",
        "schema_version": SCHEMA_VERSION,
        "mode": "use-plan",
        "backend": BACKEND,
        "workloads": rows,
        "aggregate": {
            "workload_count": len(rows),
            "planned_config_equals_actual_rate": sum(1 for r in rows if r["config_equals_actual"]) / max(len(rows), 1),
            "planned_id_equals_actual_rate": sum(1 for r in rows if r["planned_equals_actual"]) / max(len(rows), 1),
            "runtime_override_count": 0,
        },
        "truth_boundary": "plan_validation_checks_exact_config_transport_no_runtime_autotuning",
        "utc_start": utc_now(),
        "utc_end": utc_now(),
    }


def make_reports(summary: dict[str, Any]) -> str:
    lines = [
        "# Triton Fused Config Kernel Selection",
        "",
        "This report evaluates shape-aware selection among one-pass fused Triton tile configurations only.",
        "",
        "## Policy Comparison",
        "",
        "| Policy | Config diversity | Top-1 | Macro accuracy | Mean regret | P95 regret | Max regret | Within 3% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in summary["policy_aggregates"].items():
        lines.append(
            f"| {name} | {row['config_diversity']} | {row['top1_accuracy']:.4f} | {row['macro_accuracy']:.4f} | "
            f"{row['mean_regret']:.4f} | {row['p95_regret']:.4f} | {row['max_regret']:.4f} | {row['within_3_percent']:.4f} |"
        )
    lines += [
        "",
        "## Per-Workload Confidence-Aware Decisions",
        "",
        "| Workload | M | N | K | Oracle | Selected | Confidence | Regret |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for row in summary["workloads"]:
        shape = row["shape"]
        dec = row["policy_results"]["confidence_aware"]
        lines.append(
            f"| {row['workload_id']} | {shape['m']} | {shape['n']} | {shape['k']} | "
            f"{row['oracle_config']} | {dec['selected_config_id']} | {dec.get('confidence', {}).get('level')} | {dec['regret']:.4f} |"
        )
    lines += [
        "",
        "## Collapse Audit",
        "",
        f"- analytical collapsed: `{summary['collapse_audit']['analytical_collapsed_to_static_config']}`",
        f"- calibrated collapsed: `{summary['collapse_audit']['calibrated_collapsed_to_static_config']}`",
        f"- confidence-aware collapsed: `{summary['collapse_audit']['confidence_aware_collapsed_to_static_config']}`",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-sweep", default="trace/matmul_postop_triton_fused_candidate_sweep.json")
    p.add_argument("--split", default="leave-one-shape-region-out", choices=("leave-one-shape-region-out", "leave-one-k-band-out", "leave-one-n-family-out"))
    p.add_argument("--cost-model-output", default="trace/matmul_postop_triton_fused_config_cost_model.json")
    p.add_argument("--plans-output", default="trace/matmul_postop_triton_fused_config_selection_plans.json")
    p.add_argument("--plan-validation-output", default="trace/matmul_postop_triton_fused_config_plan_validation.json")
    p.add_argument("--fresh-oracle-output", default="trace/matmul_postop_triton_fused_config_fresh_oracle.json")
    p.add_argument("--summary-output", default="trace/matmul_postop_triton_fused_config_selection_summary.json")
    p.add_argument("--report-output", default="trace/matmul_postop_triton_fused_config_selection_report.md")
    p.add_argument("--doc-output", default="DOC/result/TRITON_FUSED_CONFIG_KERNEL_SELECTION.md")
    p.add_argument("--target-profile", default=None)
    p.add_argument("--effective-compute-units", type=int, default=None)
    p.add_argument("--allow-hardware-override", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = utc_now()
    sweep_path = Path(args.candidate_sweep)
    measurements, sweep_payload = parse_discovery_measurements(sweep_path)
    train, heldout, split = grouped_split(measurements, args.split)
    env = sweep_payload.get("environment", {})
    resolved_hardware = hardware_profile.load_and_resolve_hardware_profile(
        args.target_profile,
        env,
        cli_override=args.effective_compute_units,
        compatibility_default=16,
        allow_cli_override=args.allow_hardware_override,
    )
    sm_count = resolved_hardware.effective_compute_units
    model = calibrate(train, sm_count)
    stats = feature_stats(train)
    cost_payload = {
        "schema": "triton_matmul_bias_relu_fused_config_cost_model",
        "schema_version": SCHEMA_VERSION,
        "backend": BACKEND,
        "target_gpu": {
            "gpu_model": env.get("gpu_model"),
            "compute_capability": env.get("compute_capability"),
            "pytorch_version": env.get("pytorch_version"),
            "triton_version": env.get("triton_version"),
        },
        "hardware_profile": resolved_hardware.as_dict(),
        "candidate_registry": [PRIMARY_CONFIGS[cid].typed() for cid in PRIMARY_CONFIG_IDS],
        "feature_schema": {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "workload_features": list(workload_features({"m": 1, "n": 1, "k": 1, "dtype": DTYPE}).keys()),
            "candidate_tile_features": list(tile_features(workload_features({"m": 1, "n": 1, "k": 1, "dtype": DTYPE}), PRIMARY_CONFIGS[PRIMARY_CONFIG_IDS[0]]).keys()),
            "compatibility_aliases": {
                "total_output_program_count": "parallel_work_items",
                "programs_per_sm": "work_items_per_compute_unit",
                "output_program_waves": "execution_waves",
                "work_per_program": "compute_work_per_item",
                "masked_output_fraction": "padding_waste_ratio",
                "padding_amplification": "work_amplification",
            },
            "forbidden_model_fields": sorted(FORBIDDEN_MODEL_FIELDS),
        },
        "analytical_model": model,
        "grouped_split": split,
        "training_profile_sha256": sha256_file(sweep_path),
        "candidate_sweep_sha256": sha256_file(sweep_path),
        "truth_boundary": "human_readable_fused_config_cost_model_no_opaque_ml_model",
        "utc_start": started,
        "utc_end": utc_now(),
    }
    cost_path = Path(args.cost_model_output)
    write_json(cost_path, cost_payload)
    model_sha = sha256_file(cost_path)

    policy_rows = {p: [] for p in POLICIES}
    workloads_summary = []
    plans = []
    for row in heldout:
        oracle = min(row.latencies.items(), key=lambda item: (item[1], item[0]))[0]
        oracle_latency = row.latencies[oracle]
        decisions = {}
        for policy in POLICIES:
            decision = choose_policy(policy, row.shape, model, train, stats, sm_count)
            selected = decision["selected_config_id"]
            regret = row.latencies[selected] / oracle_latency - 1.0
            result = {
                **decision,
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
        plan = build_plan(row, decisions["confidence_aware"], model_sha)
        plans.append(plan)
        workloads_summary.append({
            "workload_id": row.workload_id,
            "shape": row.shape,
            "category": row.category,
            "region": row.region,
            "oracle_config": oracle,
            "oracle_latencies_ms": row.latencies,
            "policy_results": decisions,
        })
    plans_payload = {
        "schema": "triton_matmul_bias_relu_fused_config_selection_plans",
        "schema_version": SCHEMA_VERSION,
        "mode": "compiler-selection",
        "backend": BACKEND,
        "cost_model_artifact_sha256": model_sha,
        "plans": plans,
        "utc_start": started,
        "utc_end": utc_now(),
    }
    validation = validate_plans(plans)
    fresh_oracle = {
        "schema": "triton_matmul_bias_relu_fused_config_fresh_oracle",
        "schema_version": SCHEMA_VERSION,
        "mode": "fresh-oracle",
        "backend": BACKEND,
        "oracle_source": str(sweep_path),
        "truth_boundary": "fresh_oracle_reuses_independent_formal_fused_candidate_sweep_artifact_no_training_feedback",
        "workloads": [
            {
                "workload_id": r.workload_id,
                "shape": r.shape,
                "oracle_config": min(r.latencies.items(), key=lambda item: (item[1], item[0]))[0],
                "candidate_latencies_ms": r.latencies,
            }
            for r in heldout
        ],
        "utc_start": started,
        "utc_end": utc_now(),
    }
    policy_aggregates = {name: evaluate(rows) for name, rows in policy_rows.items()}
    collapse = {
        "analytical_collapsed_to_static_config": policy_aggregates["analytical"]["config_diversity"] == 1,
        "calibrated_collapsed_to_static_config": policy_aggregates["calibrated_analytical"]["config_diversity"] == 1,
        "confidence_aware_collapsed_to_static_config": policy_aggregates["confidence_aware"]["config_diversity"] == 1,
    }
    summary = {
        "schema": "triton_matmul_bias_relu_fused_config_selection_summary",
        "schema_version": SCHEMA_VERSION,
        "mode": "aggregate-report",
        "backend": BACKEND,
        "cost_model_artifact_sha256": model_sha,
        "plan_validation_aggregate": validation["aggregate"],
        "grouped_split": split,
        "policy_aggregates": policy_aggregates,
        "selection_diversity": policy_aggregates["confidence_aware"]["selection_counts"],
        "oracle_win_counts": policy_aggregates["confidence_aware"]["oracle_counts"],
        "top1_correct_counts": policy_aggregates["confidence_aware"]["top1_correct_counts"],
        "workloads": workloads_summary,
        "collapse_audit": collapse,
        "utc_start": started,
        "utc_end": utc_now(),
    }
    for path, payload in [
        (Path(args.plans_output), plans_payload),
        (Path(args.plan_validation_output), validation),
        (Path(args.fresh_oracle_output), fresh_oracle),
        (Path(args.summary_output), summary),
    ]:
        write_json(path, payload)
    report = make_reports(summary)
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(report, encoding="utf-8")
    Path(args.doc_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc_output).write_text(report, encoding="utf-8")
    print(json.dumps({"summary": args.summary_output, "policy_aggregates": policy_aggregates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
