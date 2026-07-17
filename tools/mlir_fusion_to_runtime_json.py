#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

# --- Profile-guided MatMul post-op kernel selection -------------------------
#
# The measured kernel benchmark profile is the JSON document written by
# apps/run_mlir_fused_kernel_benchmark.cpp (benchmark == "matmul_postop_relu",
# schema_version == 2). A measurement is used for selection only when ALL of
# pattern / backend / dtype / M / N / K / kernel_id / kernel configuration
# match exactly and correctness passed. Ranking metric: mean latency
# (mean_latency_ms). Tie-breaking order, documented and deterministic:
#   1. lower p95_ms
#   2. lower coefficient of variation
#   3. fewer runtime dispatches
#   4. fewer intermediate tensors
#   5. stable lexical kernel_id ordering
# When no valid exact-match evidence exists, selection falls back to a
# deterministic safe kernel with an explicit structured fallback reason.

MATMUL_PROFILE_BENCHMARK = "matmul_postop_relu"
SUPPORTED_PROFILE_SCHEMA_VERSIONS = (2,)
PROFILE_RANKING_METRIC = "mean_latency_ms"
# Repeat-to-repeat coefficient of variation acceptance threshold for a
# measurement to count as selection evidence (5%).
MAX_ACCEPTED_CV = 0.05

TIE_BREAKER_ORDER = (
    "p95_ms",
    "coefficient_of_variation",
    "runtime_dispatch_count",
    "intermediate_tensor_count",
    "kernel_id_lexical",
)

POSTOP_SEMANTICS_TO_PATTERN = {
    "bias_shape_N": "matmul_bias_relu",
    "elementwise_add_shape_MxN": "matmul_add_relu",
}

MATMUL_LEGAL_KERNELS = {
    "matmul_bias_relu": (
        "cpu_naive_matmul_bias_relu_unfused_f32",
        "cpu_naive_matmul_bias_relu_one_pass_f32",
        "cpu_tiled_matmul_bias_relu_unfused_f32",
        "cpu_tiled_matmul_bias_relu_one_pass_f32",
    ),
    "matmul_add_relu": (
        "cpu_naive_matmul_add_relu_unfused_f32",
        "cpu_naive_matmul_add_relu_one_pass_f32",
        "cpu_tiled_matmul_add_relu_unfused_f32",
        "cpu_tiled_matmul_add_relu_one_pass_f32",
    ),
}

MATMUL_SAFE_FALLBACK_KERNEL = {
    "matmul_bias_relu": "cpu_tiled_matmul_bias_relu_unfused_f32",
    "matmul_add_relu": "cpu_tiled_matmul_add_relu_unfused_f32",
}

DEFAULT_MATMUL_KERNEL_CONFIG = {"tile_m": 32, "tile_n": 32, "tile_k": 32}

TRITON_PROFILE_SCHEMA = "triton_matmul_bias_relu_fixed_config_profile"
TRITON_PROFILE_SCHEMA_VERSION = 1
TRITON_BACKEND = "triton_cuda"
TRITON_PATTERN = "bias_relu"
TRITON_DTYPE = "f32"
TRITON_PROFILE_RANKING_METRIC = "median_ms"
TRITON_LEGAL_KERNELS = (
    "triton_tiled_matmul_bias_relu_unfused_f32",
    "triton_tiled_matmul_bias_relu_one_pass_f32",
)
TRITON_KERNEL_VARIANTS = {
    "triton_tiled_matmul_bias_relu_unfused_f32": "V1",
    "triton_tiled_matmul_bias_relu_one_pass_f32": "V3",
}


def kernel_static_properties(kernel_id):
    one_pass = kernel_id.endswith("_one_pass_f32")
    return {
        "tiled": kernel_id.startswith("cpu_tiled_"),
        "runtime_dispatch_count": 1 if one_pass else 3,
        "intermediate_tensor_count": 0 if one_pass else 2,
    }


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def normalize_profile_measurement(pattern, variant_key, variant, config, machine, source):
    stats = variant.get("statistics") or {}
    correctness = variant.get("correctness") or {}
    implementation = variant.get("implementation_properties") or {}
    record = {
        "pattern": pattern,
        "backend": "cpu",
        "dtype": config.get("dtype"),
        "m": config.get("m"),
        "n": config.get("n"),
        "k": config.get("k"),
        "kernel_id": variant.get("kernel_id"),
        "variant": variant_key,
        "tile_size": implementation.get("tile_size"),
        "correctness_passed": correctness.get("passed"),
        "warmup_iterations": config.get("warmup"),
        "measured_iterations": config.get("iterations"),
        "repeats": config.get("repeats"),
        "mean_ms": stats.get("mean_ms"),
        "p50_ms": stats.get("p50_ms"),
        "p95_ms": stats.get("p95_ms"),
        "stddev_ms": stats.get("stddev_ms"),
        "coefficient_of_variation": stats.get("coefficient_of_variation"),
        "machine_hostname": machine.get("hostname"),
        "source": source,
    }

    issues = []
    if not record["kernel_id"]:
        issues.append("missing_kernel_id")
    elif not record["kernel_id"].startswith("cpu_"):
        issues.append("kernel_backend_not_cpu")
    if not record["dtype"]:
        issues.append("missing_dtype")
    for dim in ("m", "n", "k"):
        if not positive_int(record[dim]):
            issues.append(f"invalid_shape_{dim}")
    if not positive_int(record["warmup_iterations"]):
        issues.append("invalid_warmup_iterations")
    if not positive_int(record["measured_iterations"]):
        issues.append("invalid_measured_iterations")
    if not positive_int(record["repeats"]):
        issues.append("invalid_repeats")
    for field in ("mean_ms", "p50_ms", "p95_ms"):
        if not finite_positive(record[field]):
            issues.append(f"invalid_{field}")
    cv = record["coefficient_of_variation"]
    if cv is None or isinstance(cv, bool) or not isinstance(cv, (int, float)) or not math.isfinite(cv) or cv < 0:
        issues.append("invalid_coefficient_of_variation")
    elif cv > MAX_ACCEPTED_CV:
        issues.append("cv_above_acceptance_threshold")
    if record["correctness_passed"] is not True:
        issues.append("correctness_not_passed")

    record["issues"] = issues
    record["stats_valid"] = all(issue == "correctness_not_passed" for issue in issues)
    record["usable"] = not issues
    return record


def profile_fingerprint(path):
    payload = path.read_bytes()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


def collect_matmul_profile_document(payload, source):
    doc = {
        "path": source,
        "benchmark": payload.get("benchmark"),
        "schema": payload.get("schema"),
        "schema_version": payload.get("schema_version"),
        "mode": payload.get("mode"),
        "supported": False,
        "issues": [],
        "measurements": [],
    }
    if payload.get("benchmark") != MATMUL_PROFILE_BENCHMARK:
        doc["issues"].append("not_a_matmul_postop_benchmark_document")
        return doc
    if payload.get("schema_version") not in SUPPORTED_PROFILE_SCHEMA_VERSIONS:
        doc["issues"].append("unsupported_profile_schema_version")
        return doc
    if payload.get("mode") == "use-plan":
        # Circular-measurement guard: a plan-driven validation run must never
        # feed back into selection as profile evidence.
        doc["issues"].append("use_plan_output_rejected_as_selection_evidence")
        return doc

    config = payload.get("configuration") or {}
    machine = payload.get("machine") or {}
    for pattern_payload in (payload.get("patterns") or {}).values():
        semantics = pattern_payload.get("postop_semantics")
        pattern = POSTOP_SEMANTICS_TO_PATTERN.get(semantics)
        if pattern is None:
            doc["issues"].append(f"unknown_postop_semantics:{semantics}")
            continue
        for variant_key, variant in (pattern_payload.get("variants") or {}).items():
            doc["measurements"].append(
                normalize_profile_measurement(pattern, variant_key, variant, config, machine, source)
            )
    doc["supported"] = True
    return doc


def triton_config_from_profile(cfg):
    return {
        "block_m": cfg.get("BLOCK_M") or cfg.get("block_m"),
        "block_n": cfg.get("BLOCK_N") or cfg.get("block_n"),
        "block_k": cfg.get("BLOCK_K") or cfg.get("block_k"),
        "num_warps": cfg.get("num_warps"),
        "num_stages": cfg.get("num_stages"),
        "precision_mode": cfg.get("precision_mode"),
    }


def normalize_triton_profile_measurement(workload, variant_key, variant, config, env, source):
    stats = ((variant.get("timing") or {}).get("statistics") or {})
    correctness = variant.get("correctness") or {}
    fixed_config = triton_config_from_profile(workload.get("selected_fixed_config") or config.get("fixed_config") or {})
    shape = workload.get("shape") or {}
    record = {
        "pattern": TRITON_PATTERN,
        "backend": TRITON_BACKEND,
        "dtype": shape.get("dtype"),
        "m": shape.get("m"),
        "n": shape.get("n"),
        "k": shape.get("k"),
        "workload_id": workload.get("workload_id"),
        "kernel_id": variant.get("kernel_id"),
        "variant": variant.get("variant") or variant_key,
        "fixed_config": fixed_config,
        "correctness_passed": correctness.get("passed"),
        "warmup_iterations": config.get("warmup"),
        "measured_iterations": config.get("iterations"),
        "repeats": config.get("repeats"),
        "mean_ms": stats.get("mean_ms"),
        "median_ms": stats.get("median_ms"),
        "p50_ms": stats.get("p50_ms"),
        "p95_ms": stats.get("p95_ms"),
        "stddev_ms": stats.get("stddev_ms"),
        "coefficient_of_variation": stats.get("coefficient_of_variation"),
        "gpu_model": env.get("gpu_model") or env.get("gpu_model_torch"),
        "compute_capability": env.get("compute_capability"),
        "profile_source": source,
    }
    issues = []
    if record["kernel_id"] not in TRITON_LEGAL_KERNELS:
        issues.append("unknown_kernel_id")
    if record["dtype"] != TRITON_DTYPE:
        issues.append("dtype_mismatch")
    for dim in ("m", "n", "k"):
        if not positive_int(record[dim]):
            issues.append(f"invalid_shape_{dim}")
    if record["backend"] != TRITON_BACKEND:
        issues.append("backend_mismatch")
    if record["pattern"] != TRITON_PATTERN:
        issues.append("pattern_mismatch")
    for key in ("block_m", "block_n", "block_k", "num_warps", "num_stages"):
        if not positive_int(record["fixed_config"].get(key)):
            issues.append("config_mismatch")
            break
    if record["fixed_config"].get("precision_mode") != "ieee":
        issues.append("precision_mode_mismatch")
    for field in ("mean_ms", "median_ms", "p50_ms", "p95_ms"):
        if not finite_positive(record[field]):
            issues.append("invalid_profile_statistics")
    cv = record["coefficient_of_variation"]
    if cv is None or isinstance(cv, bool) or not isinstance(cv, (int, float)) or not math.isfinite(cv) or cv < 0:
        issues.append("invalid_profile_statistics")
    if record["correctness_passed"] is not True:
        issues.append("candidate_correctness_failed")
    record["issues"] = sorted(set(issues))
    record["usable"] = not record["issues"]
    return record


def collect_triton_matmul_profile_document(payload, source):
    doc = {
        "path": source,
        "schema": payload.get("schema"),
        "schema_version": payload.get("schema_version"),
        "mode": payload.get("mode"),
        "backend": payload.get("backend"),
        "supported": False,
        "issues": [],
        "measurements": [],
    }
    if payload.get("schema") != TRITON_PROFILE_SCHEMA:
        doc["issues"].append("not_a_triton_matmul_bias_relu_profile")
        return doc
    if payload.get("schema_version") != TRITON_PROFILE_SCHEMA_VERSION:
        doc["issues"].append("unsupported_profile_schema_version")
        return doc
    if payload.get("mode") == "use-plan":
        doc["issues"].append("use_plan_output_rejected_as_selection_evidence")
        return doc
    if payload.get("mode") not in ("candidate-sweep", "fresh-oracle"):
        doc["issues"].append("unsupported_profile_mode")
        return doc
    if payload.get("backend") != TRITON_BACKEND:
        doc["issues"].append("backend_mismatch")
        return doc
    config = payload.get("benchmark_config") or {}
    env = payload.get("environment") or {}
    for workload in payload.get("workloads") or []:
        if workload.get("status") != "completed":
            continue
        for variant_key, variant in (workload.get("variants") or {}).items():
            doc["measurements"].append(
                normalize_triton_profile_measurement(workload, variant_key, variant, config, env, source)
            )
    doc["supported"] = True
    return doc


def candidate_export_entry(kernel_id, measurement):
    props = kernel_static_properties(kernel_id)
    entry = {
        "kernel_id": kernel_id,
        "eligible": False,
        "rank": None,
        "profile_latency_ms": None,
        "profile_p50_ms": None,
        "profile_p95_ms": None,
        "profile_cv": None,
        "runtime_dispatch_count": props["runtime_dispatch_count"],
        "intermediate_tensor_count": props["intermediate_tensor_count"],
    }
    if measurement is not None:
        entry["profile_latency_ms"] = measurement["mean_ms"]
        entry["profile_p50_ms"] = measurement["p50_ms"]
        entry["profile_p95_ms"] = measurement["p95_ms"]
        entry["profile_cv"] = measurement["coefficient_of_variation"]
        entry["correctness_passed"] = measurement["correctness_passed"]
    return entry


def kernel_config_matches(kernel_id, measurement, kernel_config):
    """Tiled kernels must be measured at exactly the planned tile config.

    Naive kernels do not consume the tile configuration, so any measurement of
    a naive kernel is config-compatible by construction.
    """
    if not kernel_static_properties(kernel_id)["tiled"]:
        return True
    tile = measurement.get("tile_size")
    return (
        positive_int(tile)
        and tile == kernel_config["tile_m"]
        and tile == kernel_config["tile_n"]
        and tile == kernel_config["tile_k"]
    )


def matmul_safe_fallback_selection(pattern, shape, profile, reason, detail=None, candidates=None):
    fallback_kernel = MATMUL_SAFE_FALLBACK_KERNEL[pattern]
    shape_bucket = f"{shape['m']}x{shape['k']}x{shape['n']}:{shape['dtype']}"
    selection_export = {
        "policy": "safe_fallback",
        "metric": PROFILE_RANKING_METRIC,
        "selected_value": None,
        "profile_schema_version": None,
        "profile_match": "none",
        "fallback_used": True,
        "fallback_reason": reason,
        "tie_breaker_order": list(TIE_BREAKER_ORDER),
        "cv_acceptance_threshold": MAX_ACCEPTED_CV,
        "profile_source": profile.get("profile_path"),
    }
    if detail:
        selection_export["fallback_detail"] = detail
    return {
        "selected_kernel": fallback_kernel,
        "selected_backend": "cpu",
        "candidate_kernel": fallback_kernel,
        "candidate_backend": "cpu",
        "fallback_kernel": fallback_kernel,
        "fallback_backend": "cpu",
        "profile_status": profile.get("profile_status", "not_provided"),
        "profile_source": profile.get("profile_path"),
        "selection_reason": reason,
        "profile_calibrated": False,
        "shape_bucket": shape_bucket,
        "evidence": None,
        "selection": selection_export,
        "kernel_candidates": candidates if candidates is not None else [],
    }


def resolve_no_eligible_reason(legal_measurements, illegal_kernels):
    if not legal_measurements:
        if illegal_kernels:
            return "all_profiled_candidates_illegal", None
        return "invalid_profile_measurement", "no_legal_candidate_has_a_matching_measurement"
    if all(
        measurement["stats_valid"] and "correctness_not_passed" in measurement["issues"]
        for measurement in legal_measurements
    ):
        return "no_correctness_passing_candidate", None
    issues = sorted({issue for m in legal_measurements for issue in m["issues"]})
    return "invalid_profile_measurement", ",".join(issues) if issues else None


def select_matmul_kernel(pattern, shape, kernel_config, profile):
    """Rank legal kernels for this op by measured profile evidence.

    Exact-match keys: pattern, backend, dtype, M, N, K, kernel_id, kernel
    configuration, correctness_passed == true. Ranking metric: mean latency.
    """
    status = profile.get("profile_status", "not_provided")
    if status in ("not_provided", "missing"):
        detail = "profile_files_missing" if status == "missing" else None
        return matmul_safe_fallback_selection(pattern, shape, profile, "profile_not_provided", detail)

    documents = profile.get("matmul_profile_documents", [])
    supported_docs = [doc for doc in documents if doc["supported"]]
    if not supported_docs:
        issues = sorted({issue for doc in documents for issue in doc["issues"]})
        return matmul_safe_fallback_selection(
            pattern, shape, profile, "unsupported_profile_schema", ",".join(issues) or None
        )

    measurements = [m for doc in supported_docs for m in doc["measurements"]]
    pattern_measurements = [m for m in measurements if m["pattern"] == pattern]
    if not pattern_measurements:
        return matmul_safe_fallback_selection(pattern, shape, profile, "no_matching_pattern")

    exact = [
        m
        for m in pattern_measurements
        if m["m"] == shape["m"]
        and m["n"] == shape["n"]
        and m["k"] == shape["k"]
        and m["dtype"] == shape["dtype"]
        and m["backend"] == "cpu"
    ]
    if not exact:
        return matmul_safe_fallback_selection(pattern, shape, profile, "no_exact_shape_match")

    legal = MATMUL_LEGAL_KERNELS[pattern]
    # Later profile documents override earlier ones for the same kernel_id.
    by_kernel = {}
    for measurement in exact:
        if measurement["kernel_id"]:
            by_kernel[measurement["kernel_id"]] = measurement
    missing_kernel_id = [m for m in exact if not m["kernel_id"]]
    illegal_kernels = sorted(k for k in by_kernel if k not in legal)

    candidates = []
    for kernel_id in legal:
        measurement = by_kernel.get(kernel_id)
        entry = candidate_export_entry(kernel_id, measurement)
        if measurement is None:
            entry["ineligible_reason"] = "no_exact_profile_measurement"
        elif not kernel_config_matches(kernel_id, measurement, kernel_config):
            entry["ineligible_reason"] = "kernel_config_mismatch"
        elif measurement["issues"]:
            entry["ineligible_reason"] = (
                "correctness_failed"
                if measurement["issues"] == ["correctness_not_passed"]
                else "invalid_profile_measurement:" + ",".join(measurement["issues"])
            )
        else:
            entry["eligible"] = True
        candidates.append(entry)

    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        legal_measurements = [
            by_kernel[kernel_id]
            for kernel_id in legal
            if kernel_id in by_kernel
            and kernel_config_matches(kernel_id, by_kernel[kernel_id], kernel_config)
        ]
        if missing_kernel_id and not by_kernel:
            reason, detail = "invalid_profile_measurement", "measurements_missing_kernel_id"
        else:
            reason, detail = resolve_no_eligible_reason(legal_measurements, illegal_kernels)
        return matmul_safe_fallback_selection(pattern, shape, profile, reason, detail, candidates)

    ranked = sorted(
        eligible,
        key=lambda c: (
            c["profile_latency_ms"],
            c["profile_p95_ms"],
            c["profile_cv"],
            c["runtime_dispatch_count"],
            c["intermediate_tensor_count"],
            c["kernel_id"],
        ),
    )
    for index, entry in enumerate(ranked):
        entry["rank"] = index + 1
    ranked_ids = [entry["kernel_id"] for entry in ranked]
    candidates.sort(
        key=lambda c: (c["rank"] is None, c["rank"] if c["rank"] is not None else 0, c["kernel_id"])
    )
    winner = ranked[0]
    winner_measurement = by_kernel[winner["kernel_id"]]
    shape_bucket = f"{shape['m']}x{shape['k']}x{shape['n']}:{shape['dtype']}"

    selection_export = {
        "policy": "profile_guided_latency",
        "metric": PROFILE_RANKING_METRIC,
        "selected_value": winner["profile_latency_ms"],
        "profile_schema_version": SUPPORTED_PROFILE_SCHEMA_VERSIONS[-1],
        "profile_match": "exact",
        "fallback_used": False,
        "tie_breaker_order": list(TIE_BREAKER_ORDER),
        "cv_acceptance_threshold": MAX_ACCEPTED_CV,
        "profile_source": winner_measurement["source"],
        "profile_generation_mode": next(
            (doc["mode"] for doc in supported_docs if doc["path"] == winner_measurement["source"]),
            None,
        ),
        "ranked_kernels": ranked_ids,
    }
    return {
        "selected_kernel": winner["kernel_id"],
        "selected_backend": "cpu",
        "candidate_kernel": winner["kernel_id"],
        "candidate_backend": "cpu",
        "fallback_kernel": MATMUL_SAFE_FALLBACK_KERNEL[pattern],
        "fallback_backend": "cpu",
        "profile_status": profile.get("profile_status", "loaded"),
        "profile_source": winner_measurement["source"],
        "selection_reason": "profile_guided_latency_rank_1",
        "profile_calibrated": True,
        "shape_bucket": shape_bucket,
        "evidence": winner_measurement,
        "selection": selection_export,
        "kernel_candidates": candidates,
    }


def triton_profile_safe_result(shape, profile, reason, detail=None, candidates=None):
    return {
        "workload_id": shape.get("workload_id"),
        "backend": TRITON_BACKEND,
        "profile_match": "none",
        "selection_source": "unsupported_exact_profile",
        "selected_variant": None,
        "selected_kernel": None,
        "selected_config": None,
        "selected_latency_ms": None,
        "fallback_reason": reason,
        "fallback_detail": detail,
        "selection_statistic": TRITON_PROFILE_RANKING_METRIC,
        "reporting_statistic": TRITON_PROFILE_RANKING_METRIC,
        "kernel_candidates": candidates or [],
        "profile_source": profile.get("profile_path"),
    }


def triton_config_matches(lhs, rhs):
    return triton_config_from_profile(lhs) == triton_config_from_profile(rhs)


def triton_measurement_mismatch_reasons(measurement, shape, config, target):
    reasons = []
    if measurement.get("backend") != TRITON_BACKEND:
        reasons.append("backend_mismatch")
    if measurement.get("pattern") != TRITON_PATTERN:
        reasons.append("pattern_mismatch")
    if measurement.get("dtype") != shape.get("dtype"):
        reasons.append("dtype_mismatch")
    if (
        measurement.get("m") != shape.get("m")
        or measurement.get("n") != shape.get("n")
        or measurement.get("k") != shape.get("k")
    ):
        reasons.append("shape_mismatch")
    if target.get("gpu_model") and measurement.get("gpu_model") != target.get("gpu_model"):
        reasons.append("gpu_model_mismatch")
    if target.get("compute_capability") and list(measurement.get("compute_capability") or []) != list(target.get("compute_capability")):
        reasons.append("compute_capability_mismatch")
    if measurement.get("fixed_config", {}).get("precision_mode") != config.get("precision_mode"):
        reasons.append("precision_mode_mismatch")
    if not triton_config_matches(measurement.get("fixed_config") or {}, config):
        reasons.append("config_mismatch")
    if "candidate_correctness_failed" in measurement.get("issues", []):
        reasons.append("candidate_correctness_failed")
    if "invalid_profile_statistics" in measurement.get("issues", []):
        reasons.append("invalid_profile_statistics")
    return sorted(set(reasons))


def select_triton_matmul_bias_relu_kernel(shape, config, profile, target):
    """Select a Triton V1/V3 kernel from exact measured profile evidence.

    Exact-match keys: backend, pattern, dtype, M/N/K, GPU model, compute
    capability, precision mode, and fixed config identity. Ranking metric:
    median latency from PR A.
    """
    status = profile.get("profile_status", "not_provided")
    if status in ("not_provided", "missing"):
        return triton_profile_safe_result(shape, profile, "profile_not_provided")

    documents = profile.get("triton_profile_documents", [])
    supported_docs = [doc for doc in documents if doc["supported"]]
    if not supported_docs:
        issues = sorted({issue for doc in documents for issue in doc["issues"]})
        return triton_profile_safe_result(shape, profile, "unsupported_profile_schema", ",".join(issues) or None)

    measurements = [m for doc in supported_docs for m in doc["measurements"]]
    candidates = []
    for kernel_id in TRITON_LEGAL_KERNELS:
        exact_for_kernel = [
            m for m in measurements
            if m.get("kernel_id") == kernel_id
            and not triton_measurement_mismatch_reasons(m, shape, config, target)
        ]
        measurement = exact_for_kernel[-1] if exact_for_kernel else None
        entry = {
            "kernel_id": kernel_id,
            "variant": TRITON_KERNEL_VARIANTS[kernel_id],
            "eligible": False,
            "rank": None,
            "profile_latency_ms": None,
            "profile_mean_ms": None,
            "profile_p95_ms": None,
            "profile_cv": None,
        }
        if measurement is None:
            related = [m for m in measurements if m.get("kernel_id") == kernel_id]
            mismatch_reasons = sorted({
                reason
                for m in related
                for reason in triton_measurement_mismatch_reasons(m, shape, config, target)
            })
            entry["ineligible_reason"] = ",".join(mismatch_reasons) if mismatch_reasons else "no_exact_profile_measurement"
        elif measurement.get("issues"):
            entry["ineligible_reason"] = ",".join(measurement["issues"])
        else:
            entry.update(
                {
                    "eligible": True,
                    "profile_latency_ms": measurement["median_ms"],
                    "profile_mean_ms": measurement["mean_ms"],
                    "profile_p95_ms": measurement["p95_ms"],
                    "profile_cv": measurement["coefficient_of_variation"],
                    "source": measurement["profile_source"],
                }
            )
        candidates.append(entry)

    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        all_reasons = sorted({c.get("ineligible_reason") for c in candidates if c.get("ineligible_reason")})
        reason = all_reasons[0] if len(all_reasons) == 1 else "no_exact_profile_match"
        return triton_profile_safe_result(shape, profile, reason, ",".join(all_reasons), candidates)

    ranked = sorted(
        eligible,
        key=lambda c: (
            c["profile_latency_ms"],
            c["profile_p95_ms"],
            c["profile_cv"],
            c["kernel_id"],
        ),
    )
    for index, entry in enumerate(ranked):
        entry["rank"] = index + 1
    winner = ranked[0]
    for candidate in candidates:
        if candidate["kernel_id"] == winner["kernel_id"]:
            candidate["rank"] = winner["rank"]
    candidates.sort(key=lambda c: (c["rank"] is None, c["rank"] or 0, c["kernel_id"]))
    return {
        "workload_id": shape.get("workload_id"),
        "backend": TRITON_BACKEND,
        "profile_match": "exact",
        "selection_source": "measured_profile_exact_match",
        "selected_variant": winner["variant"],
        "selected_kernel": winner["kernel_id"],
        "selected_config": dict(config),
        "selected_latency_ms": winner["profile_latency_ms"],
        "fallback_reason": None,
        "selection_statistic": TRITON_PROFILE_RANKING_METRIC,
        "reporting_statistic": TRITON_PROFILE_RANKING_METRIC,
        "kernel_candidates": candidates,
        "profile_source": winner.get("source"),
    }


def detect_fused_matmul(text):
    hir_pattern = re.compile(
        r"(?P<result>%[\w\d_]+)\s*=\s*(?:\"hir\.fused_matmul_bias_relu\"|hir\.fused_matmul_bias_relu)\s*",
        re.MULTILINE,
    )
    hir_matches = list(hir_pattern.finditer(text))
    if hir_matches:
        return hir_matches

    annotated_pattern = re.compile(
        r"(?P<result>%[\w\d_]+)\s*=\s*linalg\.matmul\s*"
        r"\{[^}]*fusion\.candidate\s*=\s*\"matmul_bias_relu\"[^}]*\}",
        re.MULTILINE,
    )
    return list(annotated_pattern.finditer(text))


def detect_rmsnorm(text):
    hir_pattern = re.compile(
        r"(?P<result>%[\w\d_]+)\s*=\s*(?:\"hir\.fused_rmsnorm\"|hir\.fused_rmsnorm)\s*",
        re.MULTILINE,
    )
    hir_matches = list(hir_pattern.finditer(text))
    if hir_matches:
        return hir_matches

    annotated_pattern = re.compile(
        r"(?P<result>%[\w\d_]+)\s*=\s*\"llm\.rmsnorm\"\s*\([^)]*\)",
        re.MULTILINE,
    )
    return list(annotated_pattern.finditer(text))


def detect_fused_qmatmul(text):
    hir_pattern = re.compile(
        r"(?P<result>%[\w\d_]+)\s*=\s*(?:\"hir\.fused_qmatmul_bias_relu\"|hir\.fused_qmatmul_bias_relu)\s*",
        re.MULTILINE,
    )
    return list(hir_pattern.finditer(text))


def op_line(text, match):
    line_start = text.rfind("\n", 0, match.start()) + 1
    line_end = text.find("\n", match.end())
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end]


def string_attr(line, name):
    found = re.search(rf'{re.escape(name)}\s*=\s*"([^"]+)"', line)
    return found.group(1) if found else None


def bool_attr(line, name):
    found = re.search(rf"{re.escape(name)}\s*=\s*(true|false)", line)
    if not found:
        return None
    return found.group(1) == "true"


def parse_tensor_shapes(text, match):
    line = op_line(text, match)
    tensor_pattern = re.compile(
        r"tensor<(?P<d0>\d+)x(?P<d1>\d+)x(?P<dtype>f32|f16|i8)>"
    )
    return [
        {
            "d0": int(found.group("d0")),
            "d1": int(found.group("d1")),
            "dtype": found.group("dtype"),
        }
        for found in tensor_pattern.finditer(line)
    ]


def matmul_shape_from_mlir(text, match, default):
    shapes = parse_tensor_shapes(text, match)
    if len(shapes) < 2:
        return default
    lhs = shapes[0]
    rhs = shapes[1]
    return {
        "m": lhs["d0"],
        "k": lhs["d1"],
        "n": rhs["d1"],
        "dtype": lhs["dtype"],
    }


def load_kernel_profiles(paths):
    if not paths:
        return {
            "profile_status": "not_provided",
            "kernels": {},
            "matmul_profile_documents": [],
            "triton_profile_documents": [],
        }

    profile_paths = [Path(path) for path in paths]
    kernels = {}
    cost_table = {}
    matmul_documents = []
    triton_documents = []
    fingerprints = {}
    loaded = []
    missing = []
    exact_candidates = []

    for profile_path in profile_paths:
        if not profile_path.exists():
            missing.append(str(profile_path))
            continue

        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        loaded.append(str(profile_path))
        fingerprints[str(profile_path)] = profile_fingerprint(profile_path)
        matmul_documents.append(collect_matmul_profile_document(payload, str(profile_path)))
        triton_documents.append(collect_triton_matmul_profile_document(payload, str(profile_path)))
        if payload.get("artifact_type") == "profile_calibrated_cost_table":
            for fusion_candidate, by_backend in payload.get("cost_table", {}).items():
                cost_table.setdefault(fusion_candidate, {}).update(by_backend)
            exact_candidates.extend(payload.get("exact_candidates", []))
        for row in payload.get("kernel_benchmarks", []):
            kernels[row.get("fusion_candidate")] = row

    if not loaded:
        return {
            "profile_status": "missing",
            "profile_path": ",".join(str(path) for path in profile_paths),
            "missing_profiles": missing,
            "kernels": {},
            "matmul_profile_documents": [],
            "triton_profile_documents": [],
        }

    return {
        "profile_status": "loaded",
        "profile_path": ",".join(loaded),
        "missing_profiles": missing,
        "kernels": kernels,
        "cost_table": cost_table,
        "matmul_profile_documents": matmul_documents,
        "triton_profile_documents": triton_documents,
        "profile_fingerprints": fingerprints,
        "exact_candidates": exact_candidates,
    }


def select_exact_rmsnorm_candidate(profile, shape, target_gpu_name, target_compute_capability):
    requested = {
        "operator": "rmsnorm", "semantics": "weighted_rmsnorm",
        "tokens": shape["tokens"], "hidden": shape["hidden"],
        "dtype": {"f32": "fp32", "float32": "fp32"}.get(shape["dtype"], shape["dtype"]),
        "target_gpu_name": target_gpu_name,
        "target_compute_capability": target_compute_capability,
    }
    accepted, rejected = [], []
    for row in profile.get("exact_candidates", []):
        reasons = []
        target = row.get("target") or {}
        for key in ("operator", "semantics", "tokens", "hidden", "dtype"):
            if row.get(key) != requested.get(key):
                reasons.append(f"{key}_mismatch")
        if target.get("gpu_name") != target_gpu_name:
            reasons.append("target_gpu_name_mismatch")
        if str(target.get("compute_capability")) != str(target_compute_capability):
            reasons.append("target_compute_capability_mismatch")
        if row.get("correct") is not True:
            reasons.append("correctness_failed")
        if row.get("selection_ready") is not True:
            reasons.append("not_selection_ready")
        if row.get("measurement_kind") != "measured" or not isinstance(row.get("p50_ms"), (int, float)):
            reasons.append("not_measured")
        (rejected if reasons else accepted).append({"candidate_id": row.get("candidate_id"), "reasons": reasons, "candidate": row})
    if not accepted:
        fallback = {
            "candidate_id": "torch_rmsnorm_fp32_v1", "operator": "rmsnorm",
            "semantics": "weighted_rmsnorm", "backend": "torch",
            "kernel_family": "pytorch_rmsnorm_fallback", "kernel_entry_point": "torch_rmsnorm",
            "dtype": requested["dtype"], "tokens": shape["tokens"], "hidden": shape["hidden"],
            "epsilon": 1e-6, "block_size": None, "num_warps": None, "num_stages": None,
            "target": {"gpu_name": target_gpu_name, "compute_capability": target_compute_capability},
        }
        return {"decision_kind": "rmsnorm_gpu_exact_config_selection", "requested_key": requested,
                "selected_candidate_id": fallback["candidate_id"], "selected_candidate": fallback,
                "selection_metric": "p50_ms", "selected_cost_ms": None, "runner_up_candidate_id": None,
                "runner_up_cost_ms": None, "exact_match": False, "fallback_used": True,
                "selection_reason": "fallback_no_valid_exact_measured_candidate", "rejected_candidates": rejected}
    accepted.sort(key=lambda item: (item["candidate"]["p50_ms"], item["candidate_id"]))
    selected = accepted[0]["candidate"]
    runner_up = accepted[1]["candidate"] if len(accepted) > 1 else None
    return {"decision_kind": "rmsnorm_gpu_exact_config_selection", "requested_key": requested,
            "selected_candidate_id": selected["candidate_id"], "selected_candidate": selected,
            "selection_metric": "p50_ms", "selected_cost_ms": selected["p50_ms"],
            "runner_up_candidate_id": runner_up and runner_up["candidate_id"],
            "runner_up_cost_ms": runner_up and runner_up["p50_ms"], "exact_match": True,
            "fallback_used": False, "selection_reason": "lowest_exact_measured_p50_ms",
            "evidence_artifact": selected.get("source_artifact"), "rejected_candidates": rejected}


def shape_bucket_for(fusion_candidate, shape=None):
    if shape and fusion_candidate == "rmsnorm":
        return f"{shape['tokens']}x{shape['hidden']}:{shape['dtype']}"
    if fusion_candidate == "matmul_bias_relu":
        return "128x128x128:f32"
    if fusion_candidate == "qmatmul_bias_relu":
        return "128x128x128:i8"
    if fusion_candidate == "rmsnorm":
        return "16x4096:f32"
    return "default:f32"


def profile_cost_entry(profile, fusion_candidate, backend, shape=None):
    bucket = shape_bucket_for(fusion_candidate, shape)
    return (
        profile.get("cost_table", {})
        .get(fusion_candidate, {})
        .get(backend, {})
        .get(bucket)
    )


def select_kernel(
    fusion_candidate,
    custom_kernel,
    custom_backend,
    fallback_kernel,
    fallback_backend,
    profile,
    shape=None,
):
    bucket = shape_bucket_for(fusion_candidate, shape)
    table_entry = profile_cost_entry(profile, fusion_candidate, custom_backend, shape)
    evidence = profile.get("kernels", {}).get(fusion_candidate)
    if table_entry:
        evidence = {
            "fusion_candidate": fusion_candidate,
            "custom_kernel": table_entry.get("custom_kernel"),
            "fallback_kernel": table_entry.get("fallback_kernel"),
            "custom_latency_ms": table_entry.get("custom_ms"),
            "fallback_latency_ms": table_entry.get("fallback_ms"),
            "speedup": table_entry.get("speedup"),
            "correct": table_entry.get("correct"),
            "selection_ready": table_entry.get("selection_ready"),
            "shape_bucket": bucket,
            "profile_table_source": table_entry.get("source"),
        }
    if not evidence:
        return {
            "selected_kernel": fallback_kernel,
            "selected_backend": fallback_backend,
            "candidate_kernel": custom_kernel,
            "candidate_backend": custom_backend,
            "fallback_kernel": fallback_kernel,
            "fallback_backend": fallback_backend,
            "profile_status": profile.get("profile_status", "not_provided"),
            "profile_source": profile.get("profile_path"),
            "selection_reason": "fallback_no_profile_evidence",
            "profile_calibrated": False,
            "shape_bucket": bucket,
            "evidence": None,
        }

    custom_ms = evidence.get("custom_latency_ms")
    fallback_ms = evidence.get("fallback_latency_ms")
    custom_wins = (
        isinstance(custom_ms, (int, float))
        and isinstance(fallback_ms, (int, float))
        and custom_ms < fallback_ms
        and evidence.get("correct") is True
        and evidence.get("selection_ready") is not False
    )

    if fusion_candidate == "rmsnorm" and custom_backend in {"CUDA", "Triton"}:
        selection_reason = (
            "gpu_pgo_like_lowest_p95_latency"
            if custom_wins
            else "gpu_pgo_like_profile_guided_fallback"
        )
    else:
        selection_reason = "profile_calibrated_fastest" if custom_wins else "profile_calibrated_fallback"

    return {
        "selected_kernel": custom_kernel if custom_wins else fallback_kernel,
        "selected_backend": custom_backend if custom_wins else fallback_backend,
        "candidate_kernel": custom_kernel,
        "candidate_backend": custom_backend,
        "fallback_kernel": fallback_kernel,
        "fallback_backend": fallback_backend,
        "profile_status": profile.get("profile_status", "loaded"),
        "profile_source": profile.get("profile_path"),
        "selection_reason": selection_reason,
        "profile_calibrated": True,
        "feedback_loop": (
            "gpu_pgo_like_kernel_selection"
            if fusion_candidate == "rmsnorm" and custom_backend in {"CUDA", "Triton"}
            else "profile_calibrated_cost_model"
        ),
        "shape_bucket": bucket,
        "cost_table_entry": table_entry,
        "evidence": evidence,
    }


def build_runtime_dispatch_contract(hir_op_type, runtime_op_type, selection):
    return {
        "op_type": hir_op_type,
        "runtime_op_type": runtime_op_type,
        "runtime_kernel": selection["selected_kernel"],
        "backend": selection["selected_backend"],
        "candidate_kernel": selection["candidate_kernel"],
        "fallback_kernel": selection["fallback_kernel"],
        "profile_source": selection["profile_source"],
        "selection_reason": selection["selection_reason"],
        "profile_calibrated": selection["profile_calibrated"],
        "feedback_loop": selection.get("feedback_loop"),
        "shape_bucket": selection["shape_bucket"],
    }


def portable_accelerator_target_model():
    return {
        "target": "portable_accelerator_v1",
        "base_tile_shape": {"m": 16, "n": 16, "k": 32},
        "memory_hierarchy": "global_sram_register",
        "sram_kb": 256,
        "vector_bytes": 128,
        "alignment_bytes": 128,
        "collective": "none",
        "legality": [
            "matmul result must have one use",
            "bias add result must have one use",
            "bias must be materialized to result shape or represented as legal HIR broadcast",
            "M/N/K exact tile multiples lower directly; near-tile static shapes may use pad_to_tile_with_crop",
            "dtype must be f32 or f16 for this fused floating-point path",
        ],
        "padding_policy": {
            "mode": "pad_to_tile_with_crop",
            "max_compute_overhead_ratio": 1.25,
            "max_output_overhead_ratio": 1.25,
            "realization": "tensor.pad + padded linalg.matmul + tensor.extract_slice",
        },
    }


def dtype_bytes(dtype):
    return {"f32": 4, "f16": 2, "i8": 1}.get(dtype, 4)


def estimate_tile_sram_bytes(tile_m, tile_n, tile_k, bytes_per_element):
    lhs_bytes = tile_m * tile_k * bytes_per_element
    rhs_bytes = tile_k * tile_n * bytes_per_element
    bias_bytes = tile_m * tile_n * bytes_per_element
    out_bytes = tile_m * tile_n * bytes_per_element
    return lhs_bytes + rhs_bytes + bias_bytes + out_bytes


def round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def padding_crop_decision(shape, tile, max_compute_overhead=1.25, max_output_overhead=1.25):
    padded = {
        "m": round_up(shape["m"], tile["m"]),
        "n": round_up(shape["n"], tile["n"]),
        "k": round_up(shape["k"], tile["k"]),
    }
    requires_padding = padded != {"m": shape["m"], "n": shape["n"], "k": shape["k"]}
    compute_overhead = (
        padded["m"] * padded["n"] * padded["k"]
    ) / max(shape["m"] * shape["n"] * shape["k"], 1)
    output_overhead = (
        padded["m"] * padded["n"]
    ) / max(shape["m"] * shape["n"], 1)
    legal = (
        compute_overhead <= max_compute_overhead
        and output_overhead <= max_output_overhead
    )
    reasons = []
    if compute_overhead > max_compute_overhead:
        reasons.append("padding_compute_overhead_too_high")
    if output_overhead > max_output_overhead:
        reasons.append("padding_output_overhead_too_high")
    return {
        "requires_padding_crop": requires_padding,
        "original_shape": {"m": shape["m"], "n": shape["n"], "k": shape["k"]},
        "padded_shape": padded,
        "padding_compute_overhead_ratio": round(compute_overhead, 6),
        "padding_output_overhead_ratio": round(output_overhead, 6),
        "legal": legal,
        "reject_reasons": reasons,
    }


def choose_tile(shape, target):
    bytes_per_element = dtype_bytes(shape["dtype"])
    sram_budget = target["sram_kb"] * 1024
    vector_bytes = target["vector_bytes"]
    candidates = [
        {"m": 16, "n": 16, "k": 32},
        {"m": 16, "n": 32, "k": 32},
        {"m": 32, "n": 16, "k": 32},
        {"m": 32, "n": 32, "k": 32},
        {"m": 16, "n": 64, "k": 32},
        {"m": 64, "n": 16, "k": 32},
        {"m": 64, "n": 32, "k": 32},
        {"m": 32, "n": 64, "k": 32},
        {"m": 64, "n": 64, "k": 32},
        {"m": 16, "n": 32, "k": 128},
        {"m": 16, "n": 64, "k": 128},
        {"m": 32, "n": 32, "k": 128},
        {"m": 32, "n": 64, "k": 128},
        {"m": 64, "n": 32, "k": 128},
        {"m": 64, "n": 64, "k": 128},
    ]

    evaluated = []
    legal = []
    for tile in candidates:
        reasons = []
        padding = padding_crop_decision(shape, tile)
        if padding["requires_padding_crop"] and not padding["legal"]:
            reasons.extend(padding["reject_reasons"])
        if (tile["k"] * bytes_per_element) % vector_bytes != 0:
            reasons.append("K_tile_not_vector_aligned")
        sram_bytes = estimate_tile_sram_bytes(
            tile["m"],
            tile["n"],
            tile["k"],
            bytes_per_element,
        )
        if sram_bytes > sram_budget:
            reasons.append("tile_exceeds_sram_budget")
        flops = 2 * tile["m"] * tile["n"] * tile["k"]
        arithmetic_intensity = flops / max(sram_bytes, 1)
        record = {
            "tile": tile,
            "sram_bytes": sram_bytes,
            "arithmetic_intensity_flops_per_byte": round(arithmetic_intensity, 6),
            "legal": not reasons,
            "reject_reasons": reasons,
            "padding": padding,
        }
        evaluated.append(record)
        if record["legal"]:
            legal.append(record)

    if not legal:
        return {
            "status": "fallback_no_legal_tile",
            "selected_tile": None,
            "selected_sram_bytes": None,
            "decision_reason": "no tile satisfies shape, vector alignment, and SRAM constraints",
            "candidates": evaluated,
        }

    selected = max(
        legal,
        key=lambda item: (
            item["arithmetic_intensity_flops_per_byte"],
            item["tile"]["m"] * item["tile"]["n"],
        ),
    )
    return {
        "status": "selected",
        "selected_tile": selected["tile"],
        "selected_sram_bytes": selected["sram_bytes"],
        "requires_padding_crop": selected["padding"]["requires_padding_crop"],
        "padding": selected["padding"],
        "decision_reason": (
            "selected padded masked/crop tile under SRAM/alignment constraints"
            if selected["padding"]["requires_padding_crop"]
            else "selected highest-intensity legal tile under SRAM/alignment constraints"
        ),
        "candidates": evaluated,
    }


def build_dispatch_descriptor(hir_op_type, runtime_op_type, selection, shape, target):
    tile_decision = choose_tile(shape, target)
    return {
        "descriptor_type": "target.dispatch_descriptor.v1",
        "hir_op": hir_op_type,
        "runtime_op_type": runtime_op_type,
        "target": target["target"],
        "backend": selection["selected_backend"],
        "kernel": selection["selected_kernel"],
        "shape": shape,
        "dtype": shape["dtype"],
        "tile_decision": tile_decision,
        "memory_hierarchy": target["memory_hierarchy"],
        "alignment_bytes": target["alignment_bytes"],
        "vector_bytes": target["vector_bytes"],
    }


def estimate_matmul_bias_relu_cost(m=16, k=128, n=64, dtype="f32"):
    bytes_per_element = dtype_bytes(dtype)
    matmul_flops = 2 * m * k * n
    bias_add_flops = m * n
    relu_flops = m * n
    total_flops = matmul_flops + bias_add_flops + relu_flops

    bytes_a = m * k * bytes_per_element
    bytes_b = k * n * bytes_per_element
    bytes_bias = m * n * bytes_per_element
    bytes_out = m * n * bytes_per_element

    bytes_read = bytes_a + bytes_b + bytes_bias
    bytes_written = bytes_out
    arithmetic_intensity = total_flops / max(bytes_read + bytes_written, 1)

    return {
        "m": m,
        "k": k,
        "n": n,
        "dtype": dtype,
        "estimated_flops": total_flops,
        "estimated_bytes_read": bytes_read,
        "estimated_bytes_written": bytes_written,
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        "source": "profile_calibrated_table",
        "shape_bucket": "128x128x128:f32",
    }


def estimate_rmsnorm_cost(tokens=16, hidden=768, dtype="f16"):
    bytes_per_element = dtype_bytes(dtype)
    elements = tokens * hidden
    total_flops = elements * 4
    bytes_read = elements * bytes_per_element * 2
    bytes_written = elements * bytes_per_element

    return {
        "tokens": tokens,
        "hidden": hidden,
        "dtype": dtype,
        "estimated_flops": total_flops,
        "estimated_bytes_read": bytes_read,
        "estimated_bytes_written": bytes_written,
        "arithmetic_intensity_flops_per_byte": total_flops / max(bytes_read + bytes_written, 1),
        "source": "profile_calibrated_table",
        "shape_bucket": "16x4096:f32",
    }


def build_matmul_op(index, match, profile, source_text):
    result_name = match.group("result")
    hir_op_type = "hir.fused_matmul_bias_relu"
    runtime_op_type = "FusedMatMulAddReLU"
    typed_runtime_op_type = "FusedMatMulBiasRelu"
    shape = matmul_shape_from_mlir(
        source_text,
        match,
        {"m": 16, "k": 128, "n": 64, "dtype": "f32"},
    )
    selection = select_matmul_kernel(
        "matmul_bias_relu",
        shape,
        DEFAULT_MATMUL_KERNEL_CONFIG,
        profile,
    )
    target = portable_accelerator_target_model()
    dispatch_descriptor = build_dispatch_descriptor(
        hir_op_type,
        runtime_op_type,
        selection,
        shape,
        target,
    )
    return {
        "id": index,
        "name": f"fused_matmul_bias_relu_{index}",
        "source_result": result_name,
        "op_type": hir_op_type,
        "legacy_op_type": "FusedMatMulBiasReLU",
        "lowered_op_type": hir_op_type,
        "runtime_op_type": runtime_op_type,
        "runtime_kernel": selection["selected_kernel"],
        "runtime_kernel_backend": selection["selected_backend"],
        "backend": selection["selected_backend"],
        "typed_runtime_op_type": typed_runtime_op_type,
        "selected_kernel": selection["selected_kernel"],
        "kernel_config": dict(DEFAULT_MATMUL_KERNEL_CONFIG),
        "runtime_dispatch_contract": build_runtime_dispatch_contract(
            hir_op_type,
            runtime_op_type,
            selection,
        ),
        "target_model": target,
        "dispatch_descriptor": dispatch_descriptor,
        "fusion_candidate": "matmul_bias_relu",
        "fusion_group": "matmul_bias_relu_0",
        "inputs": ["A", "B", "bias"],
        "outputs": ["output"],
        "cost_model": estimate_matmul_bias_relu_cost(
            shape["m"],
            shape["k"],
            shape["n"],
            shape["dtype"],
        ),
        "kernel_selection": selection,
        "notes": [
            "Detected from MLIR linalg.matmul annotated by MatMulBiasReluFusionPass",
            "Lowered through the shared runtime-aware kernel selection contract",
            "Runtime benchmark evidence selects the custom fused kernel or fallback path",
        ],
    }


def build_rmsnorm_op(index, match, profile, source_text, rmsnorm_backend, target_gpu_name=None, target_compute_capability=None):
    shapes = parse_tensor_shapes(source_text, match)
    shape = {
        "tokens": shapes[0]["d0"] if shapes else 16,
        "hidden": shapes[0]["d1"] if shapes else 4096,
        "dtype": shapes[0]["dtype"] if shapes else "f32",
    }
    exact_selection = None
    if target_gpu_name and target_compute_capability:
        exact_selection = select_exact_rmsnorm_candidate(profile, shape, target_gpu_name, target_compute_capability)
    if exact_selection:
        candidate = exact_selection["selected_candidate"]
        selection = {
            "selected_kernel": candidate["candidate_id"], "selected_backend": candidate["backend"],
            "candidate_kernel": candidate["candidate_id"], "candidate_backend": candidate["backend"],
            "fallback_kernel": "torch_rmsnorm_fp32_v1", "fallback_backend": "torch",
            "profile_status": profile.get("profile_status"), "profile_source": profile.get("profile_path"),
            "selection_reason": exact_selection["selection_reason"], "profile_calibrated": exact_selection["exact_match"],
            "shape_bucket": shape_bucket_for("rmsnorm", shape), "evidence": candidate,
            "exact_config_selection": exact_selection,
        }
    elif rmsnorm_backend == "Metal":
        custom_kernel = "fused_rmsnorm_metal"
        fallback_kernel = "cpu_rmsnorm"
        fallback_backend = "CPU"
    else:
        custom_kernel = "fused_rmsnorm_cuda"
        fallback_kernel = "torch_rmsnorm"
        fallback_backend = "PyTorch"
    if not exact_selection:
        selection = select_kernel("rmsnorm", custom_kernel, rmsnorm_backend, fallback_kernel, fallback_backend, profile, shape)
    result_name = match.group("result")
    hir_op_type = "hir.fused_rmsnorm"
    runtime_op_type = "FusedRMSNorm"
    return {
        "id": index,
        "name": f"fused_rmsnorm_{index}",
        "source_result": result_name,
        "op_type": hir_op_type,
        "legacy_op_type": "FusedRMSNorm",
        "lowered_op_type": hir_op_type,
        "runtime_op_type": runtime_op_type,
        "runtime_kernel": selection["selected_kernel"],
        "runtime_kernel_backend": selection["selected_backend"],
        "backend": selection["selected_backend"],
        "runtime_dispatch_contract": build_runtime_dispatch_contract(
            hir_op_type,
            runtime_op_type,
            selection,
        ),
        "fusion_candidate": "rmsnorm",
        "fusion_group": f"rmsnorm_{index}",
        "inputs": ["hidden_states", "weight"],
        "outputs": [result_name],
        "cost_model": estimate_rmsnorm_cost(
            shape["tokens"],
            shape["hidden"],
            shape["dtype"],
        ),
        "shape": shape,
        "kernel_selection": selection,
        "exact_config_selection": exact_selection,
        "notes": [
            "Detected from MLIR llm.rmsnorm annotated by RMSNormKernelSelectionPass",
            "Lowered to HIR fused RMSNorm candidate",
            "Runtime benchmark evidence selects custom CUDA or PyTorch fallback",
        ],
    }


def build_qmatmul_op(index, match, profile, source_text):
    selection = select_kernel(
        "qmatmul_bias_relu",
        "int8_qmatmul_bias_relu",
        "CPU",
        "fused_matmul_add_relu",
        "CPU",
        profile,
    )
    result_name = match.group("result")
    hir_op_type = "hir.fused_qmatmul_bias_relu"
    runtime_op_type = "FusedQMatMulBiasReLU"
    shape = matmul_shape_from_mlir(
        source_text,
        match,
        {"m": 128, "k": 128, "n": 128, "dtype": "i8"},
    )
    shape["dtype"] = "i8"
    target = portable_accelerator_target_model()
    dispatch_descriptor = build_dispatch_descriptor(
        hir_op_type,
        runtime_op_type,
        selection,
        shape,
        target,
    )
    return {
        "id": index,
        "name": f"fused_qmatmul_bias_relu_{index}",
        "source_result": result_name,
        "op_type": hir_op_type,
        "legacy_op_type": "FusedQMatMulBiasReLU",
        "lowered_op_type": hir_op_type,
        "runtime_op_type": runtime_op_type,
        "runtime_kernel": selection["selected_kernel"],
        "runtime_kernel_backend": selection["selected_backend"],
        "backend": selection["selected_backend"],
        "runtime_dispatch_contract": build_runtime_dispatch_contract(
            hir_op_type,
            runtime_op_type,
            selection,
        ),
        "target_model": target,
        "dispatch_descriptor": dispatch_descriptor,
        "fusion_candidate": "qmatmul_bias_relu",
        "fusion_group": f"qmatmul_bias_relu_{index}",
        "inputs": ["A_int8", "B_int8", "bias"],
        "outputs": [result_name],
        "cost_model": {
            "m": 128,
            "k": 128,
            "n": 128,
            "dtype": "i8",
            "estimated_flops": 2 * 128 * 128 * 128 + 128 * 128 * 2,
            "estimated_bytes_read": 128 * 128 + 128 * 128 + 128 * 128 * 4,
            "estimated_bytes_written": 128 * 128 * 4,
            "arithmetic_intensity_flops_per_byte": (
                (2 * 128 * 128 * 128 + 128 * 128 * 2)
                / max((128 * 128 + 128 * 128 + 128 * 128 * 4 + 128 * 128 * 4), 1)
            ),
            "source": "profile_calibrated_table",
            "shape_bucket": "128x128x128:i8",
            "layout": {
                "input_layout": "NHWC",
                "weight_layout": "blocked_kc",
                "alignment": 128,
            },
        },
        "kernel_selection": selection,
        "notes": [
            "Detected from typed HIR quantized MatMul-Bias-ReLU op",
            "Lowered only when profile metadata marks the INT8 path valid and faster",
            "Layout constraints model mobile DSP/NPU alignment and channel requirements",
        ],
    }


def build_lowered_graph(
    matmul_matches,
    rmsnorm_matches,
    qmatmul_matches,
    source_path,
    profile,
    source_text,
    rmsnorm_backend, target_gpu_name=None, target_compute_capability=None,
):
    ops = []
    for match in matmul_matches:
        ops.append(build_matmul_op(len(ops), match, profile, source_text))
    for match in rmsnorm_matches:
        ops.append(build_rmsnorm_op(
            len(ops),
            match,
            profile,
            source_text,
            rmsnorm_backend,
            target_gpu_name, target_compute_capability,
        ))
    for match in qmatmul_matches:
        ops.append(build_qmatmul_op(len(ops), match, profile, source_text))

    return {
        "format": "hir.lowered_graph.v1",
        "source": str(source_path),
        "kernel_profile": {
            "status": profile.get("profile_status", "not_provided"),
            "source": profile.get("profile_path"),
            "fingerprints": profile.get("profile_fingerprints", {}),
            "documents": [
                {key: doc[key] for key in ("path", "benchmark", "schema_version", "mode", "supported", "issues")}
                for doc in profile.get("matmul_profile_documents", [])
            ],
        },
        "num_ops": len(ops),
        "ops": ops,
    }


def build_execution_plan(lowered_graph):
    steps = []
    operations = []

    for op in lowered_graph["ops"]:
        steps.append({
            "step": op["id"],
            "op_name": op["name"],
            "op_type": op["op_type"],
            "lowered_op_type": op["lowered_op_type"],
            "runtime_op_type": op["runtime_op_type"],
            "runtime_kernel": op["runtime_kernel"],
            "runtime_kernel_backend": op["runtime_kernel_backend"],
            "backend": op["backend"],
            "fusion_candidate": op["fusion_candidate"],
            "fusion_group": op["fusion_group"],
            "runtime_action": "dispatch_selected_kernel",
            "runtime_dispatch_contract": op["runtime_dispatch_contract"],
            "kernel_selection": op["kernel_selection"],
            "target_model": op.get("target_model"),
            "dispatch_descriptor": op.get("dispatch_descriptor"),
            "estimated_launch_overhead_us": 80,
            "estimated_flops": op["cost_model"]["estimated_flops"],
            "arithmetic_intensity_flops_per_byte": op["cost_model"]["arithmetic_intensity_flops_per_byte"],
        })
        if op.get("typed_runtime_op_type") and op.get("selected_kernel"):
            operation = {
                "op_id": op["name"],
                "op_type": op["typed_runtime_op_type"],
                "backend": op["backend"],
                "selected_kernel": op["selected_kernel"],
                "kernel_config": op["kernel_config"],
                "inputs": op["inputs"],
                "outputs": op["outputs"],
            }
            kernel_selection = op.get("kernel_selection") or {}
            if "selection" in kernel_selection:
                operation["selection"] = kernel_selection["selection"]
                operation["kernel_candidates"] = kernel_selection.get("kernel_candidates", [])
            operations.append(operation)

    return {
        "schema_version": 2,
        "schema": "runtime_execution_plan",
        "format": "hir.execution_plan.v2",
        "graph_id": "mlir_matmul_postop_relu",
        "source": lowered_graph["source"],
        "kernel_profile": lowered_graph["kernel_profile"],
        "operations": operations,
        "num_steps": len(steps),
        "steps": steps,
    }


def build_canonical_rmsnorm_execution_plan(lowered_graph):
    rmsnorm_ops = [op for op in lowered_graph["ops"] if op["fusion_candidate"] == "rmsnorm"]
    if len(rmsnorm_ops) != 1 or not rmsnorm_ops[0].get("exact_config_selection"):
        raise ValueError("canonical exact RMSNorm plan requires exactly one exact-selected RMSNorm op")
    op = rmsnorm_ops[0]
    decision = op["exact_config_selection"]
    candidate = decision["selected_candidate"]
    launch = {"block_size": candidate.get("block_size"), "num_warps": candidate.get("num_warps"), "num_stages": candidate.get("num_stages")}
    kernel = {
        "decision_type": "KernelDecision", "scope": "PerOp",
        "selected_kernel": candidate["candidate_id"], "kernel_library": candidate["kernel_family"],
        "lowering_path": "rmsnorm_gpu_exact_config", "kernel_exists": candidate["backend"] != "torch",
        "decision_kind": "rmsnorm_gpu_exact_config_selection",
        "operator": "rmsnorm", "semantics": "weighted_rmsnorm", "backend": candidate["backend"],
        "candidate_id": candidate["candidate_id"], "kernel_family": candidate["kernel_family"],
        "kernel_entry_point": candidate["kernel_entry_point"], "dtype": candidate["dtype"],
        "tokens": candidate["tokens"], "hidden": candidate["hidden"], "epsilon": candidate.get("epsilon", 1e-6),
        "launch_config": launch,
        "artifact": {"source_hash": candidate.get("source_hash"), "measurement_artifact_hash": candidate.get("artifact_hash"),
            "compiled_artifact_hash": candidate.get("compiled_artifact_hash")},
        "target": candidate["target"], "measurement_evidence_ref": candidate.get("source_artifact"),
        "runtime_no_policy_redecision": True,
    }
    return {
        "schema": "execution_plan", "schema_version": "2.0.0",
        "plan_id": f"rmsnorm-exact-{candidate['candidate_id']}-{candidate['tokens']}x{candidate['hidden']}",
        "provenance": {"compiler_tool": "tools/mlir_fusion_to_runtime_json.py",
            "model_spec_ref": str(lowered_graph["source"]),
            "capability_bundle": {"hardware_profile_ref": "hardware/nvidia_gtx1650_maxq.json",
                "backend_profile_refs": [f"backend/{candidate['backend']}.json"], "kernel_profile_refs": []},
            "truth_boundary": "operator_level_weighted_rmsnorm_exact_measured_selection_not_full_model"},
        "model_identity": {"model_id": "operator_level_weighted_rmsnorm"}, "global_decisions": {},
        "function_plans": [{"function_name": "weighted_rmsnorm", "serving_phase": "other",
            "backend": {"decision_type": "BackendDecision", "scope": "Function", "selected_backend": candidate["backend"]},
            "per_op_decisions": [{"op_name": "rmsnorm_0", "op_type": "RMSNorm", "kernel": kernel}]}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="trace/mlir_fused_graph.mlir")
    parser.add_argument("--lowered-output", default="trace/mlir_lowered_graph.json")
    parser.add_argument("--plan-output", default="trace/mlir_execution_plan.json")
    parser.add_argument("--kernel-profile", action="append")
    parser.add_argument("--rmsnorm-backend", choices=["CUDA", "Metal"], default="CUDA")
    parser.add_argument("--rmsnorm-target-gpu-name")
    parser.add_argument("--rmsnorm-target-compute-capability")
    parser.add_argument("--canonical-rmsnorm-plan-output")
    args = parser.parse_args()

    input_path = Path(args.input)
    text = input_path.read_text(encoding="utf-8")

    matmul_matches = detect_fused_matmul(text)
    rmsnorm_matches = detect_rmsnorm(text)
    qmatmul_matches = detect_fused_qmatmul(text)

    if not matmul_matches and not rmsnorm_matches and not qmatmul_matches:
        raise SystemExit(
            "No fusion annotations found. Expected fusion.candidate for "
            "matmul_bias_relu, qmatmul_bias_relu, or rmsnorm."
        )

    profile = load_kernel_profiles(args.kernel_profile)
    lowered_graph = build_lowered_graph(
        matmul_matches,
        rmsnorm_matches,
        qmatmul_matches,
        input_path,
        profile,
        text,
        args.rmsnorm_backend,
        args.rmsnorm_target_gpu_name,
        args.rmsnorm_target_compute_capability,
    )
    execution_plan = build_execution_plan(lowered_graph)

    lowered_output = Path(args.lowered_output)
    plan_output = Path(args.plan_output)

    lowered_output.parent.mkdir(parents=True, exist_ok=True)
    plan_output.parent.mkdir(parents=True, exist_ok=True)

    lowered_output.write_text(json.dumps(lowered_graph, indent=2) + "\n", encoding="utf-8")
    plan_output.write_text(json.dumps(execution_plan, indent=2) + "\n", encoding="utf-8")
    if args.canonical_rmsnorm_plan_output:
        canonical_path = Path(args.canonical_rmsnorm_plan_output)
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_path.write_text(json.dumps(build_canonical_rmsnorm_execution_plan(lowered_graph), indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {lowered_output}")
    print(f"Wrote {plan_output}")


if __name__ == "__main__":
    main()
