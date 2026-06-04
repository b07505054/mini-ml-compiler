#!/usr/bin/env python3

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlir_fusion_to_runtime_json as bridge


PROFILE = Path("trace/metal_rmsnorm_benchmark.json")
PLAN = Path("trace/metal_rmsnorm_execution_plan.json")


def main():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    rows = profile["kernel_benchmarks"]
    step = plan["steps"][0]
    loaded_profile = bridge.load_kernel_profiles([str(Path("trace/metal_rmsnorm_cost_table.json"))])
    small_decision = bridge.select_kernel(
        "rmsnorm",
        "fused_rmsnorm_metal",
        "Metal",
        "cpu_rmsnorm",
        "CPU",
        loaded_profile,
        {"tokens": 1, "hidden": 768, "dtype": "f32"},
    )
    large_decision = bridge.select_kernel(
        "rmsnorm",
        "fused_rmsnorm_metal",
        "Metal",
        "cpu_rmsnorm",
        "CPU",
        loaded_profile,
        {"tokens": 16, "hidden": 4096, "dtype": "f32"},
    )
    invalid_profile = {
        "profile_status": "loaded",
        "profile_path": "synthetic_invalid_profile",
        "kernels": {
            "rmsnorm": {
                "custom_latency_ms": 0.01,
                "fallback_latency_ms": 1.0,
                "correct": False,
                "selection_ready": False,
            }
        },
    }
    invalid_decision = bridge.select_kernel(
        "rmsnorm",
        "fused_rmsnorm_metal",
        "Metal",
        "cpu_rmsnorm",
        "CPU",
        invalid_profile,
        {"tokens": 16, "hidden": 4096, "dtype": "f32"},
    )

    assert len(rows) == 12
    assert all(row["correct"] for row in rows)
    assert any(not row["selection_ready"] for row in rows)
    assert any(row["selection_ready"] for row in rows)
    assert step["op_type"] == "hir.fused_rmsnorm"
    assert step["runtime_kernel"] == "fused_rmsnorm_metal"
    assert step["backend"] == "Metal"
    assert step["kernel_selection"]["shape_bucket"] == "16x4096:f32"
    assert step["kernel_selection"]["selection_reason"] == "profile_calibrated_fastest"
    assert step["kernel_selection"]["evidence"]["correct"] is True
    assert small_decision["selected_kernel"] == "cpu_rmsnorm"
    assert small_decision["selection_reason"] == "profile_calibrated_fallback"
    assert large_decision["selected_kernel"] == "fused_rmsnorm_metal"
    assert large_decision["selection_reason"] == "profile_calibrated_fastest"
    assert invalid_decision["selected_kernel"] == "cpu_rmsnorm"

    print(
        "validated MLIR-to-Metal RMSNorm path: "
        f"1x768:f32 -> {small_decision['selected_kernel']}; "
        f"{step['kernel_selection']['shape_bucket']} -> {step['runtime_kernel']}"
    )


if __name__ == "__main__":
    main()
