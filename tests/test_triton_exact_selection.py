#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import mlir_fusion_to_runtime_json as bridge  # noqa: E402
import run_triton_matmul_bias_relu_exact_selection as exact  # noqa: E402


SHAPE = {"m": 64, "n": 64, "k": 64, "dtype": "f32", "workload_id": "balanced_m64_n64_k64"}
CONFIG = {
    "block_m": 16,
    "block_n": 16,
    "block_k": 32,
    "num_warps": 4,
    "num_stages": 3,
    "precision_mode": "ieee",
}
TARGET = {"gpu_model": "Fixture GPU", "compute_capability": [7, 5]}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def variant(kernel_id: str, variant_name: str, median: float, passed: bool = True) -> dict:
    return {
        "variant": variant_name,
        "kernel_id": kernel_id,
        "correctness": {"passed": passed},
        "timing": {
            "statistics": {
                "mean_ms": median * 1.01,
                "median_ms": median,
                "p50_ms": median,
                "p95_ms": median * 1.02,
                "stddev_ms": median * 0.001,
                "coefficient_of_variation": 0.001,
            }
        },
    }


def profile_payload() -> dict:
    return {
        "schema": "triton_matmul_bias_relu_fixed_config_profile",
        "schema_version": 1,
        "mode": "candidate-sweep",
        "backend": "triton_cuda",
        "environment": {
            "hostname": "fixture-host",
            "gpu_model": "Fixture GPU",
            "compute_capability": [7, 5],
            "pytorch_version": "fixture",
            "triton_version": "fixture",
        },
        "benchmark_config": {
            "warmup": 50,
            "iterations": 300,
            "repeats": 5,
            "fixed_config": {
                "BLOCK_M": 16,
                "BLOCK_N": 16,
                "BLOCK_K": 32,
                "num_warps": 4,
                "num_stages": 3,
                "precision_mode": "ieee",
            },
        },
        "workloads": [
            {
                "workload_id": "balanced_m64_n64_k64",
                "category": "balanced",
                "tier": "formal_core",
                "held_out": False,
                "status": "completed",
                "shape": {"m": 64, "n": 64, "k": 64, "dtype": "f32"},
                "selected_fixed_config": {
                    "BLOCK_M": 16,
                    "BLOCK_N": 16,
                    "BLOCK_K": 32,
                    "num_warps": 4,
                    "num_stages": 3,
                    "precision_mode": "ieee",
                },
                "variants": {
                    "V1": variant("triton_tiled_matmul_bias_relu_unfused_f32", "V1", 0.09),
                    "V3": variant("triton_tiled_matmul_bias_relu_one_pass_f32", "V3", 0.03),
                },
            },
            {
                "workload_id": "rep_m128_k4096_n4096",
                "category": "representative",
                "held_out": False,
                "status": "skipped",
                "skip_reason": "resource_limit",
            },
        ],
    }


def load_profile(tmp: Path, payload: dict) -> dict:
    path = tmp / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return bridge.load_kernel_profiles([str(path)])


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        payload = profile_payload()
        profile = load_profile(tmp_path, payload)
        selection = bridge.select_triton_matmul_bias_relu_kernel(SHAPE, CONFIG, profile, TARGET)
        require(selection["profile_match"] == "exact", "Triton profile exact shape match failed")
        require(selection["selected_kernel"] == "triton_tiled_matmul_bias_relu_one_pass_f32", "median ranking failed")
        require(selection["selected_variant"] == "V3", "variant transport failed")

        wrong_gpu = bridge.select_triton_matmul_bias_relu_kernel(SHAPE, CONFIG, profile, {"gpu_model": "Other GPU", "compute_capability": [7, 5]})
        require(wrong_gpu["fallback_reason"] in ("gpu_model_mismatch", "no_exact_profile_match"), "GPU mismatch should reject exact match")
        wrong_cc = bridge.select_triton_matmul_bias_relu_kernel(SHAPE, CONFIG, profile, {"gpu_model": "Fixture GPU", "compute_capability": [8, 0]})
        require(wrong_cc["fallback_reason"] in ("compute_capability_mismatch", "no_exact_profile_match"), "CC mismatch should reject exact match")
        wrong_precision = dict(CONFIG)
        wrong_precision["precision_mode"] = "tf32"
        sel = bridge.select_triton_matmul_bias_relu_kernel(SHAPE, wrong_precision, profile, TARGET)
        require(
            "precision_mode_mismatch" in (sel.get("fallback_reason") or sel.get("fallback_detail") or ""),
            "precision mismatch should reject",
        )

        failed = copy.deepcopy(payload)
        failed["workloads"][0]["variants"]["V3"]["correctness"]["passed"] = False
        selection = bridge.select_triton_matmul_bias_relu_kernel(SHAPE, CONFIG, load_profile(tmp_path, failed), TARGET)
        require(selection["selected_kernel"] == "triton_tiled_matmul_bias_relu_unfused_f32", "failed correctness candidate should be excluded")

        invalid = copy.deepcopy(payload)
        invalid["workloads"][0]["variants"]["V3"]["timing"]["statistics"]["median_ms"] = 0.0
        selection = bridge.select_triton_matmul_bias_relu_kernel(SHAPE, CONFIG, load_profile(tmp_path, invalid), TARGET)
        require(selection["selected_kernel"] == "triton_tiled_matmul_bias_relu_unfused_f32", "invalid latency should be excluded")

        use_plan_profile = copy.deepcopy(payload)
        use_plan_profile["mode"] = "use-plan"
        docs = load_profile(tmp_path, use_plan_profile)["triton_profile_documents"]
        require("use_plan_output_rejected_as_selection_evidence" in docs[0]["issues"], "use-plan output must be rejected")

        cpu_profile = {
            "schema": "kernel_benchmark_profile",
            "schema_version": 2,
            "benchmark": "matmul_postop_relu",
            "mode": "sweep-candidates",
            "machine": {"hostname": "fixture"},
            "configuration": {"warmup": 50, "iterations": 300, "repeats": 5, "dtype": "f32", "m": 64, "n": 64, "k": 64, "tile_size": 32},
            "patterns": {},
        }
        cpu_loaded = load_profile(tmp_path, cpu_profile)
        require(not [d for d in cpu_loaded["triton_profile_documents"] if d["supported"]], "CPU profile cannot be Triton evidence")

        profile_path = tmp_path / "profile_for_plans.json"
        profile_path.write_text(json.dumps(payload), encoding="utf-8")
        plans_path = tmp_path / "plans.json"
        plans_payload, plans = exact.build_selection(profile_path, plans_path)
        require(len(plans) == 1, "skipped workload must not produce a plan")
        op = plans[0]["operations"][0]
        require(op["kernel_config"] == CONFIG, "ExecutionPlan must transport exact Triton config")
        require(op["backend"] == "triton_cuda", "ExecutionPlan backend mismatch")

        validation_payload = {
            "workloads": [
                {
                    "workload_id": "balanced_m64_n64_k64",
                    "actual_dispatched_kernel": op["selected_kernel"],
                    "planned_equals_actual": True,
                    "config_equals_actual": True,
                    "correctness": {"passed": True},
                }
            ]
        }
        oracle = copy.deepcopy(payload)
        oracle["mode"] = "fresh-oracle"
        oracle["artifact_path"] = str(tmp_path / "oracle.json")
        (tmp_path / "oracle.json").write_text(json.dumps(oracle), encoding="utf-8")
        summary = exact.build_summary(plans_payload, validation_payload, oracle, tmp_path / "summary.json", tmp_path / "report.md")
        row = [r for r in summary["workloads"] if r.get("status") == "completed"][0]
        require(row["relative_regret"] == 0.0, "regret must use fresh oracle latency")
        require(row["within_1_percent"] and row["within_3_percent"], "within thresholds incorrect")
        require(summary["aggregates"]["overall"]["completed_workload_count"] == 1, "skipped workload excluded from aggregates")

        completed = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "run_triton_matmul_bias_relu_exact_selection.py"),
                "--profile",
                str(profile_path),
                "--fresh-oracle-output",
                str(profile_path),
                "--skip-use-plan",
                "--skip-fresh-oracle",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        require(completed.returncode != 0, "fresh oracle cannot reuse profile artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
