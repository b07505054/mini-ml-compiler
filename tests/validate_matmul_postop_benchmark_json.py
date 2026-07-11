#!/usr/bin/env python3
"""Validate the MatMul post-op ReLU benchmark JSON contract.

This test intentionally uses tiny timing settings. It checks structure and
implementation invariants, not performance thresholds.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_STATS = {
    "sample_count",
    "mean_ms",
    "median_ms",
    "p50_ms",
    "p95_ms",
    "min_ms",
    "max_ms",
    "stddev_ms",
    "coefficient_of_variation",
}

PATTERNS = ("bias", "elementwise_add")
VARIANTS = (
    "naive_unfused",
    "tiled_unfused",
    "naive_one_pass_fused",
    "tiled_one_pass_fused",
)


def write_plan(path: Path, **overrides: Any) -> None:
    operation = {
        "op_id": "matmul_postop_0",
        "op_type": "FusedMatMulBiasRelu",
        "backend": "cpu",
        "selected_kernel": "cpu_tiled_matmul_bias_relu_one_pass_f32",
        "kernel_config": {"tile_m": 32, "tile_n": 32, "tile_k": 32},
        "inputs": ["A", "B", "bias"],
        "outputs": ["output"],
    }
    operation.update(overrides.pop("operation", {}))
    payload = {
        "schema_version": 2,
        "schema": "runtime_execution_plan",
        "graph_id": "test_graph",
        "operations": [operation],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def expect_failure(args: list[str]) -> None:
    completed = subprocess.run(args, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert completed.returncode != 0, completed.stdout + completed.stderr


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: validate_matmul_postop_benchmark_json.py <benchmark-exe> <json-output> <report-output>",
            file=sys.stderr,
        )
        return 2

    benchmark_exe = Path(sys.argv[1])
    json_output = Path(sys.argv[2])
    report_output = Path(sys.argv[3])

    subprocess.run(
        [
            str(benchmark_exe),
            "--pattern",
            "all",
            "--variant",
            "all",
            "--warmup",
            "1",
            "--iterations",
            "1",
            "--repeats",
            "2",
            "--output",
            str(json_output),
            "--report-output",
            str(report_output),
        ],
        check=True,
    )

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["benchmark"] == "matmul_postop_relu"
    assert payload["configuration"]["warmup"] > 0
    assert payload["configuration"]["repeats"] == 2

    for pattern in PATTERNS:
        pattern_payload = payload["patterns"][pattern]
        assert set(VARIANTS).issubset(pattern_payload["variants"])
        for variant in VARIANTS:
            variant_payload = pattern_payload["variants"][variant]
            assert REQUIRED_STATS.issubset(variant_payload["statistics"])
            assert variant_payload["statistics"]["sample_count"] == 2
            assert variant_payload["correctness"]["passed"] is True
            assert "max_abs_error" in variant_payload["correctness"]
            assert "max_rel_error" in variant_payload["correctness"]

        for fused_variant in ("naive_one_pass_fused", "tiled_one_pass_fused"):
            implementation = pattern_payload["variants"][fused_variant][
                "implementation_properties"
            ]
            assert implementation["post_op_strategy"] == "one_pass_fused"
            assert implementation["intermediate_tensor_count"] == 0
            assert implementation["full_output_post_op_passes"] == 0
            assert implementation["final_output_store_passes"] == 1

        tiled_unfused = pattern_payload["variants"]["tiled_unfused"][
            "implementation_properties"
        ]
        tiled_fused = pattern_payload["variants"]["tiled_one_pass_fused"][
            "implementation_properties"
        ]
        assert tiled_unfused["matmul_strategy"] == tiled_fused["matmul_strategy"]
        assert tiled_unfused["tile_size"] == tiled_fused["tile_size"]

        comparisons = pattern_payload["comparisons"]
        for key in (
            "tiling_speedup",
            "fusion_speedup_naive",
            "fusion_speedup_fair",
            "full_stack_speedup",
        ):
            assert key in comparisons
            assert "speedup" in comparisons[key]
            assert "latency_reduction_percent" in comparisons[key]

    assert report_output.exists()

    valid_plan = json_output.with_suffix(".plan.json")
    use_plan_output = json_output.with_suffix(".use_plan.json")
    use_plan_report = report_output.with_suffix(".use_plan.md")
    write_plan(valid_plan)
    subprocess.run(
        [
            str(benchmark_exe),
            "--mode",
            "use-plan",
            "--execution-plan",
            str(valid_plan),
            "--pattern",
            "bias",
            "--warmup",
            "1",
            "--iterations",
            "1",
            "--repeats",
            "2",
            "--output",
            str(use_plan_output),
            "--report-output",
            str(use_plan_report),
        ],
        check=True,
    )
    use_plan_payload = json.loads(use_plan_output.read_text(encoding="utf-8"))
    planned = use_plan_payload["patterns"]["bias"]["variants"]["tiled_one_pass_fused"]
    trace = planned["runtime_trace"]
    assert trace["planned_kernel"] == "cpu_tiled_matmul_bias_relu_one_pass_f32"
    assert trace["actual_dispatched_kernel"] == trace["planned_kernel"]
    assert trace["dispatch_count"] == 1
    assert trace["plan_matched_runtime"] is True
    assert planned["correctness"]["passed"] is True

    unfused_plan = json_output.with_suffix(".unfused_plan.json")
    unfused_output = json_output.with_suffix(".unfused_use_plan.json")
    write_plan(
        unfused_plan,
        operation={
            "selected_kernel": "cpu_tiled_matmul_bias_relu_unfused_f32",
        },
    )
    subprocess.run(
        [
            str(benchmark_exe),
            "--mode",
            "use-plan",
            "--execution-plan",
            str(unfused_plan),
            "--pattern",
            "bias",
            "--warmup",
            "1",
            "--iterations",
            "1",
            "--repeats",
            "1",
            "--output",
            str(unfused_output),
            "--report-output",
            str(use_plan_report.with_suffix(".unfused.md")),
        ],
        check=True,
    )
    unfused_payload = json.loads(unfused_output.read_text(encoding="utf-8"))
    assert (
        unfused_payload["patterns"]["bias"]["variants"]["tiled_unfused"]["runtime_trace"][
            "dispatch_count"
        ]
        == 3
    )

    add_plan = json_output.with_suffix(".add_plan.json")
    add_output = json_output.with_suffix(".add_use_plan.json")
    write_plan(
        add_plan,
        operation={
            "op_type": "FusedMatMulAddRelu",
            "selected_kernel": "cpu_tiled_matmul_add_relu_one_pass_f32",
            "inputs": ["A", "B", "addend"],
        },
    )
    subprocess.run(
        [
            str(benchmark_exe),
            "--mode",
            "use-plan",
            "--execution-plan",
            str(add_plan),
            "--pattern",
            "elementwise-add",
            "--warmup",
            "1",
            "--iterations",
            "1",
            "--repeats",
            "1",
            "--output",
            str(add_output),
            "--report-output",
            str(use_plan_report.with_suffix(".add.md")),
        ],
        check=True,
    )
    add_payload = json.loads(add_output.read_text(encoding="utf-8"))
    assert "tiled_one_pass_fused" in add_payload["patterns"]["elementwise_add"]["variants"]

    malformed = json_output.with_suffix(".malformed.json")
    malformed.write_text("{not valid json", encoding="utf-8")
    expect_failure([
        str(benchmark_exe),
        "--mode",
        "use-plan",
        "--execution-plan",
        str(malformed),
        "--pattern",
        "bias",
    ])

    unsupported_schema = json_output.with_suffix(".unsupported_schema.json")
    write_plan(unsupported_schema, schema_version=99)
    expect_failure([str(benchmark_exe), "--mode", "use-plan", "--execution-plan", str(unsupported_schema), "--pattern", "bias"])

    unknown_kernel = json_output.with_suffix(".unknown_kernel.json")
    write_plan(unknown_kernel, operation={"selected_kernel": "cpu_unknown_kernel_f32"})
    expect_failure([str(benchmark_exe), "--mode", "use-plan", "--execution-plan", str(unknown_kernel), "--pattern", "bias"])

    invalid_backend = json_output.with_suffix(".invalid_backend.json")
    write_plan(invalid_backend, operation={"backend": "gpu"})
    expect_failure([str(benchmark_exe), "--mode", "use-plan", "--execution-plan", str(invalid_backend), "--pattern", "bias"])

    invalid_tile = json_output.with_suffix(".invalid_tile.json")
    write_plan(invalid_tile, operation={"kernel_config": {"tile_m": 32, "tile_n": 16, "tile_k": 32}})
    expect_failure([str(benchmark_exe), "--mode", "use-plan", "--execution-plan", str(invalid_tile), "--pattern", "bias"])

    missing_tensor = json_output.with_suffix(".missing_tensor.json")
    write_plan(missing_tensor, operation={"inputs": ["A", "B"]})
    expect_failure([str(benchmark_exe), "--mode", "use-plan", "--execution-plan", str(missing_tensor), "--pattern", "bias"])

    expect_failure([
        str(benchmark_exe),
        "--mode",
        "force-variant",
        "--pattern",
        "bias",
        "--variant",
        "naive-unfused",
        "--warmup",
        "0",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
