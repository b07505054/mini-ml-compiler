#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_triton_matmul_bias_relu_analytical_selection as model  # noqa: E402


CONFIG = {
    "block_m": 16,
    "block_n": 16,
    "block_k": 32,
    "num_warps": 4,
    "num_stages": 3,
    "precision_mode": "ieee",
}


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def measurement(workload_id: str, m: int, n: int, k: int, v1: float, v3: float, held_out: bool = False, category: str = "balanced") -> model.Measurement:
    return model.Measurement(
        workload_id=workload_id,
        shape={"m": m, "n": n, "k": k, "dtype": "f32"},
        config=CONFIG,
        category=category,
        held_out=held_out,
        source="fixture",
        classification=None,
        latencies={"V1": v1, "V3": v3},
    )


def profile_payload(rows: list[model.Measurement]) -> dict:
    workloads = []
    for row in rows:
        workloads.append(
            {
                "workload_id": row.workload_id,
                "category": row.category,
                "held_out": row.held_out,
                "status": "completed",
                "shape": row.shape,
                "selected_fixed_config": {
                    "BLOCK_M": 16,
                    "BLOCK_N": 16,
                    "BLOCK_K": 32,
                    "num_warps": 4,
                    "num_stages": 3,
                    "precision_mode": "ieee",
                },
                "variants": {
                    "V1": {
                        "variant": "V1",
                        "kernel_id": model.V1_KERNEL,
                        "correctness": {"passed": True},
                        "timing": {"statistics": {"median_ms": row.latencies["V1"], "mean_ms": row.latencies["V1"], "p50_ms": row.latencies["V1"], "p95_ms": row.latencies["V1"] * 1.01, "coefficient_of_variation": 0.001}},
                    },
                    "V3": {
                        "variant": "V3",
                        "kernel_id": model.V3_KERNEL,
                        "correctness": {"passed": True},
                        "timing": {"statistics": {"median_ms": row.latencies["V3"], "mean_ms": row.latencies["V3"], "p50_ms": row.latencies["V3"], "p95_ms": row.latencies["V3"] * 1.01, "coefficient_of_variation": 0.001}},
                    },
                },
            }
        )
    return {
        "schema": "triton_matmul_bias_relu_fixed_config_profile",
        "schema_version": 1,
        "mode": "candidate-sweep",
        "backend": "triton_cuda",
        "environment": {"gpu_model": "Fixture GPU", "compute_capability": [7, 5]},
        "benchmark_config": {"warmup": 50, "iterations": 300, "repeats": 5, "fixed_config": {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32, "num_warps": 4, "num_stages": 3, "precision_mode": "ieee"}},
        "workloads": workloads,
    }


def main() -> int:
    shape = {"m": 1, "n": 1024, "k": 65536, "dtype": "f32"}
    features = model.shape_features(shape, CONFIG)
    require(features["small_m"] is True, "small-M feature missing")
    require(features["extreme_k"] is True, "extreme-K feature missing")
    require(features["output_tile_count"] == 64, "output tile count mismatch")
    require(features["k_tile_count"] == 2048, "K tile count mismatch")
    require(0 < features["m_edge_utilization"] <= 1, "edge utilization invalid")

    v1 = model.candidate_features("V1", features, CONFIG)
    v3 = model.candidate_features("V3", features, CONFIG)
    require(v1["expected_launch_count"] == 3 and v3["expected_launch_count"] == 1, "launch count difference missing")
    require(v1["full_size_intermediate_count"] == 2 and v3["full_size_intermediate_count"] == 0, "intermediate difference missing")
    require(v1["estimated_full_size_intermediate_bytes"] > v3["estimated_full_size_intermediate_bytes"], "intermediate bytes mismatch")

    try:
        model.shape_features({**shape, "expected_region": "v1_win"}, CONFIG)
        raise AssertionError("forbidden labels should fail closed")
    except ValueError:
        pass

    train = [
        measurement("train_v3_a", 64, 64, 8192, 0.22, 0.21),
        measurement("train_v3_b", 128, 128, 4096, 0.29, 0.28),
        measurement("train_tie", 1, 11008, 8192, 7.30, 7.29),
        measurement("train_canon", 1024, 1024, 32, 0.32, 0.15),
    ]
    calibrated = model.calibrate_model(train)
    stats = model.training_feature_stats(train)
    decision = model.confidence_decision({"m": 1024, "n": 1024, "k": 32, "dtype": "f32"}, CONFIG, calibrated, train, stats)
    require(decision["selected_variant"] in ("V1", "V3"), "decision must choose executable variant")
    require(len(decision["predicted_candidates"]) == 2, "candidate ranking must include V1 and V3")
    for cand in decision["predicted_candidates"]:
        components = cand["analytical_components"]
        raw_sum = sum(components.values())
        require(raw_sum >= 0, "cost components must be non-negative")
        require(cand["predicted_latency_ms"] >= 0, "predicted cost must be non-negative")

    tie_decision = model.confidence_decision({"m": 1, "n": 11008, "k": 8192, "dtype": "f32"}, CONFIG, calibrated, train, stats)
    if tie_decision["selection_source"] == "confidence_aware_fallback":
        require(tie_decision["selected_variant"] == "V1", "low-confidence fallback must select V1")
        require(tie_decision["truth_boundary"] == "conservative_choice_due_to_low_model_confidence", "fallback truth boundary mismatch")

    policies = model.policy_decisions({"m": 1, "n": 11008, "k": 8192, "dtype": "f32"}, CONFIG, calibrated, train, stats)
    require(policies["always_v3"]["selected_variant"] == "V3", "always-V3 baseline mismatch")
    require(policies["current_v1_fallback"]["selected_variant"] == "V1", "current fallback baseline mismatch")
    require(policies["nearest_profile"]["selection_source"] == "nearest_profile_candidate", "nearest profile baseline missing")
    require(policies["analytical_winner"]["selection_source"] == "analytical_cost_model_no_confidence_fallback", "analytical baseline missing")
    require("confidence_guided" in policies, "confidence-guided policy missing")

    split_train, split_heldout, split = model.split_measurements(
        train
        + [
            measurement("holdout_m1024_n1024_k24", 1024, 1024, 24, 0.3, 0.15, held_out=True, category="k_sweep"),
            measurement("boundary_m1_n4096_k65536", 1, 4096, 65536, 18.5, 18.4, category="decision_boundary"),
        ]
    )
    require(split_train and split_heldout, "training/held-out split failed")
    require("leave_one_k_band_out" in split["primary_future_strategy"], "grouped split recommendation missing")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        profile = tmp_path / "profile.json"
        profile.write_text(json.dumps(profile_payload(train + split_heldout)), encoding="utf-8")
        out = tmp_path / "model.json"
        plans = tmp_path / "plans.json"
        summary = tmp_path / "summary.json"
        report = tmp_path / "report.md"
        doc = tmp_path / "doc.md"
        completed = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "run_triton_matmul_bias_relu_analytical_selection.py"),
                "--profile",
                str(profile),
                "--cost-model-output",
                str(out),
                "--plans-output",
                str(plans),
                "--summary-output",
                str(summary),
                "--report-output",
                str(report),
                "--doc-output",
                str(doc),
                "--fresh-oracle-output",
                str(tmp_path / "oracle.json"),
                "--plan-validation-output",
                str(tmp_path / "validation.json"),
                "--skip-use-plan",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        require(out.exists() and plans.exists() and summary.exists(), completed.stderr)
        payload = json.loads(summary.read_text(encoding="utf-8"))
        require("always_v3" in payload["policy_aggregates"], "always-V3 policy missing from summary")
        require("confidence_guided" in payload["policy_aggregates"], "confidence policy missing from summary")
        plan_op = json.loads(plans.read_text(encoding="utf-8"))["plans"][0]["operations"][0]
        require("predicted_candidates" in plan_op, "ExecutionPlan must transport model metadata")
        require(plan_op["selected_kernel"] in (model.V1_KERNEL, model.V3_KERNEL), "plan selected unknown kernel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
