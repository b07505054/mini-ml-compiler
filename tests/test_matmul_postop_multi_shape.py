#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from matmul_postop_workloads import (  # noqa: E402
    geometric_mean,
    load_manifest,
    postop_shape_for,
    static_cost,
)


def write_manifest(path: Path, workloads: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "matmul_postop_workload_manifest",
                "schema_version": 1,
                "workloads": workloads,
            }
        ),
        encoding="utf-8",
    )


def base_workload(**overrides):
    payload = {
        "workload_id": "w0",
        "category": "balanced",
        "m": 128,
        "n": 128,
        "k": 128,
        "dtype": "f32",
        "patterns": ["bias", "elementwise_add"],
        "tile_configs": [{"tile_m": 32, "tile_n": 32, "tile_k": 32}],
        "representative_reason": "test",
        "tier": "formal_core",
        "status": "active",
    }
    payload.update(overrides)
    return payload


def expect_manifest_failure(workloads: list[dict]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.json"
        write_manifest(path, workloads)
        try:
            load_manifest(path)
        except ValueError:
            return
        raise AssertionError("manifest unexpectedly loaded")


def main() -> int:
    manifest = load_manifest(ROOT / "benchmarks/matmul_postop_workloads.json")
    assert manifest
    assert sum(1 for w in manifest if w.held_out) >= len(manifest) * 0.2

    expect_manifest_failure([base_workload(), base_workload()])
    expect_manifest_failure([base_workload(m=0)])
    expect_manifest_failure([base_workload(tier="unknown")])
    expect_manifest_failure([base_workload(status="skipped_resource_limit")])

    assert postop_shape_for("bias", 16, 3072) == [3072]
    assert postop_shape_for("elementwise_add", 16, 3072) == [16, 3072]

    cost = static_cost("elementwise_add", 128, 128, 128)
    assert cost["matmul_macs"] == 128 * 128 * 128
    assert cost["matmul_flops"] == 2 * 128 * 128 * 128
    assert cost["output_bytes"] == 128 * 128 * 4
    assert cost["logical_intermediate_storage_bytes_eliminated"] == 2 * 128 * 128 * 4
    assert cost["runtime_dispatch_count_unfused"] == 3
    assert cost["runtime_dispatch_count_fused"] == 1
    assert cost["fusion_pressure_score_units"] == "analytical_bytes_eliminated_per_flop"

    assert round(geometric_mean([2.0, 8.0]), 6) == 4.0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        profile = tmp_path / "profile.json"
        validation = tmp_path / "validation.json"
        summary = tmp_path / "summary.json"
        corr = tmp_path / "corr.csv"
        report = tmp_path / "report.md"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "run_matmul_postop_multi_shape_evaluation.py"),
                "--benchmark-exe",
                str(ROOT / "build" / "run_mlir_fused_kernel_benchmark"),
                "--workload-manifest",
                str(ROOT / "benchmarks" / "matmul_postop_workloads.json"),
                "--workload-id",
                "balanced_m128_n128_k128",
                "--workload-id",
                "holdout_m192_n192_k192",
                "--include-held-out",
                "--pattern",
                "bias",
                "--dry-run",
                "--output",
                str(profile),
                "--validation-output",
                str(validation),
                "--summary-output",
                str(summary),
                "--correlation-output",
                str(corr),
                "--report-output",
                str(report),
                "--scratch-dir",
                str(tmp_path / "runs"),
            ],
            check=True,
        )
        payload = json.loads(profile.read_text(encoding="utf-8"))
        assert payload["schema"] == "matmul_postop_multi_shape_evaluation"
        assert len(payload["workloads"]) == 2
        exact = [r for r in payload["workloads"] if not r["held_out"]][0]
        held_out = [r for r in payload["workloads"] if r["held_out"]][0]
        assert exact["fallback_reason"] is None
        assert held_out["fallback_reason"] == "no_exact_shape_match"
        assert held_out["compiler_selected_kernel"] == "cpu_tiled_matmul_bias_relu_unfused_f32"
        assert held_out["selection_regret"] >= 0.0
        assert payload["aggregates"]["exact_profiled"]["top1_accuracy"] == 1.0
        assert json.loads(validation.read_text(encoding="utf-8"))["use_plan_outputs_are_profile_evidence"] is False
        assert "fusion_pressure_score" in corr.read_text(encoding="utf-8")
        assert "Layer 1" in report.read_text(encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
