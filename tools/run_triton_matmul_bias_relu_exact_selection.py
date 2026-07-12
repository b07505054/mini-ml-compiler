#!/usr/bin/env python3
"""PR B Triton exact-profile selection, plan emission, and validation.

This tool consumes the PR A fixed-config Triton candidate-sweep profile and
selects between the already implemented V1/V3 kernels by exact measured
profile evidence. It does not implement shape-aware modeling or autotuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlir_fusion_to_runtime_json as bridge  # noqa: E402
from matmul_postop_workloads import geometric_mean, load_manifest  # noqa: E402


BACKEND = "triton_cuda"
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def profile_target_environment(profile_payload: dict[str, Any]) -> dict[str, Any]:
    env = profile_payload.get("environment") or {}
    return {
        "gpu_model": env.get("gpu_model") or env.get("gpu_model_torch"),
        "compute_capability": env.get("compute_capability"),
        "hostname": env.get("hostname"),
        "pytorch_version": env.get("pytorch_version"),
        "triton_version": env.get("triton_version"),
    }


def selected_config_from_profile(row: dict[str, Any]) -> dict[str, Any]:
    cfg = row["selected_fixed_config"]
    return {
        "block_m": cfg["BLOCK_M"],
        "block_n": cfg["BLOCK_N"],
        "block_k": cfg["BLOCK_K"],
        "num_warps": cfg["num_warps"],
        "num_stages": cfg["num_stages"],
        "precision_mode": cfg["precision_mode"],
    }


def plan_for_row(row: dict[str, Any], selection: dict[str, Any], profile_sha256: str,
                 target: dict[str, Any]) -> dict[str, Any]:
    shape = row["shape"]
    workload_id = row["workload_id"]
    return {
        "schema": "runtime_execution_plan",
        "schema_version": 2,
        "mode": "compiler-selection",
        "graph_id": f"{workload_id}_bias_relu",
        "workload_id": workload_id,
        "backend": BACKEND,
        "profile_artifact_sha256": profile_sha256,
        "target_gpu_identity": {
            "gpu_model": target.get("gpu_model"),
            "compute_capability": target.get("compute_capability"),
        },
        "operations": [
            {
                "op_id": "matmul_bias_relu_0",
                "op_type": "MatMulBiasRelu",
                "backend": BACKEND,
                "selected_kernel": selection["selected_kernel"],
                "variant": selection["selected_variant"],
                "kernel_config": selection["selected_config"],
                "shape": {
                    "m": shape["m"],
                    "n": shape["n"],
                    "k": shape["k"],
                    "dtype": shape["dtype"],
                },
                "inputs": ["A", "B", "bias"],
                "outputs": ["Y"],
                "selection_source": selection["selection_source"],
                "profile_match": selection["profile_match"],
                "fallback_reason": selection["fallback_reason"],
                "selected_latency_ms": selection["selected_latency_ms"],
                "truth_boundary": "selected_from_measured_triton_profile_not_runtime_latency_guarantee",
            }
        ],
    }


def build_selection(profile_path: Path, plans_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile_payload = read_json(profile_path)
    profile_sha = sha256_file(profile_path)
    profile = bridge.load_kernel_profiles([str(profile_path)])
    target = profile_target_environment(profile_payload)
    rows = []
    plans = []
    for row in profile_payload.get("workloads") or []:
        if row.get("status") != "completed":
            rows.append(
                {
                    "workload_id": row.get("workload_id"),
                    "status": "skipped",
                    "skip_reason": row.get("skip_reason"),
                    "category": row.get("category"),
                    "held_out": row.get("held_out"),
                }
            )
            continue
        shape = dict(row["shape"])
        shape["workload_id"] = row["workload_id"]
        config = selected_config_from_profile(row)
        selection = bridge.select_triton_matmul_bias_relu_kernel(shape, config, profile, target)
        selection_row = {
            "workload_id": row["workload_id"],
            "status": "selected" if selection["selected_kernel"] else "unsupported",
            "category": row.get("category"),
            "tier": row.get("tier"),
            "held_out": row.get("held_out"),
            "shape": row["shape"],
            "profile_winner": selection["selected_kernel"],
            "selection": selection,
        }
        if selection["selected_kernel"]:
            plan = plan_for_row(row, selection, profile_sha, target)
            selection_row["plan_index"] = len(plans)
            selection_row["plan_sha256"] = hashlib.sha256(
                json.dumps(plan, sort_keys=True).encode("utf-8")
            ).hexdigest()
            plans.append(plan)
        rows.append(selection_row)

    payload = {
        "schema": "triton_matmul_bias_relu_execution_plans",
        "schema_version": SCHEMA_VERSION,
        "mode": "compiler-selection",
        "backend": BACKEND,
        "profile_path": str(profile_path),
        "profile_sha256": profile_sha,
        "environment": profile_payload.get("environment", {}),
        "benchmark_config": profile_payload.get("benchmark_config", {}),
        "plans": plans,
        "selections": rows,
        "utc_start": utc_now(),
        "utc_end": utc_now(),
    }
    write_json(plans_path, payload)
    # Keep the bundle hash out of the bundle itself. Downstream artifacts carry
    # this value as provenance for the exact plan file they consumed.
    payload["execution_plans_sha256"] = sha256_file(plans_path)
    return payload, plans


def run_plan_validation(runner: Path, plans_payload: dict[str, Any], validation_path: Path,
                        work_dir: Path, warmup: int, iterations: int, repeats: int) -> dict[str, Any]:
    started = utc_now()
    run_dir = validation_path.parent / "matmul_postop_triton_plan_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, plan in enumerate(plans_payload["plans"]):
        plan_path = run_dir / f"{plan['workload_id']}.plan.json"
        out_path = run_dir / f"{plan['workload_id']}.use_plan.json"
        write_json(plan_path, plan)
        cmd = [
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
            str(run_dir / f"{plan['workload_id']}.use_plan.md"),
        ]
        subprocess.run(cmd, cwd=work_dir, check=True)
        payload = read_json(out_path)
        record = payload["workloads"][0]
        record["plan_index"] = index
        record["plan_sha256"] = sha256_file(plan_path)
        records.append(record)

    aggregate = {
        "workload_count": len(records),
        "planned_kernel_equals_actual_rate": sum(1 for r in records if r["planned_equals_actual"]) / max(len(records), 1),
        "planned_config_equals_actual_rate": sum(1 for r in records if r["config_equals_actual"]) / max(len(records), 1),
        "correctness_pass_rate": sum(1 for r in records if r["correctness"]["passed"]) / max(len(records), 1),
    }
    payload = {
        "schema": "triton_matmul_bias_relu_plan_validation",
        "schema_version": SCHEMA_VERSION,
        "mode": "use-plan",
        "backend": BACKEND,
        "profile_sha256": plans_payload["profile_sha256"],
        "execution_plans_sha256": plans_payload.get("execution_plans_sha256"),
        "environment": plans_payload.get("environment", {}),
        "benchmark_config": {**plans_payload.get("benchmark_config", {}), "warmup": warmup, "iterations": iterations, "repeats": repeats},
        "workloads": records,
        "aggregate": aggregate,
        "utc_start": started,
        "utc_end": utc_now(),
    }
    write_json(validation_path, payload)
    return payload


def run_fresh_oracle(runner: Path, manifest: Path, oracle_path: Path, report_path: Path,
                     work_dir: Path, warmup: int, iterations: int, repeats: int) -> dict[str, Any]:
    if oracle_path.resolve() == report_path.resolve():
        raise ValueError("fresh oracle artifact cannot be the report path")
    cmd = [
        sys.executable,
        str(runner),
        "--mode",
        "fresh-oracle",
        "--manifest",
        str(manifest),
        "--all-eligible",
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--repeats",
        str(repeats),
        "--output",
        str(oracle_path),
        "--report-output",
        str(report_path),
    ]
    subprocess.run(cmd, cwd=work_dir, check=True)
    payload = read_json(oracle_path)
    if payload.get("mode") != "fresh-oracle":
        raise ValueError("fresh oracle artifact has wrong mode")
    return payload


def latency_for_kernel(row: dict[str, Any], kernel_id: str) -> float | None:
    for variant in (row.get("variants") or {}).values():
        if variant.get("kernel_id") == kernel_id and variant.get("correctness", {}).get("passed"):
            stats = (variant.get("timing") or {}).get("statistics") or {}
            return stats.get("median_ms")
    return None


def oracle_winner(row: dict[str, Any]) -> tuple[str | None, float | None]:
    candidates = []
    for variant in (row.get("variants") or {}).values():
        if not variant.get("correctness", {}).get("passed"):
            continue
        stats = (variant.get("timing") or {}).get("statistics") or {}
        latency = stats.get("median_ms")
        if isinstance(latency, (int, float)) and latency > 0:
            candidates.append((variant["kernel_id"], latency))
    if not candidates:
        return None, None
    return min(candidates, key=lambda item: (item[1], item[0]))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in rows if r.get("status") == "completed"]
    regrets = [r["relative_regret"] for r in completed if r.get("relative_regret") is not None]
    if not completed:
        return {"completed_workload_count": 0}
    return {
        "completed_workload_count": len(completed),
        "exact_top1_accuracy": sum(1 for r in completed if r["top1_correct"]) / len(completed),
        "mean_regret": statistics.fmean(regrets) if regrets else 0.0,
        "median_regret": statistics.median(regrets) if regrets else 0.0,
        "p95_regret": percentile(regrets, 95),
        "within_1_percent_rate": sum(1 for r in completed if r["within_1_percent"]) / len(completed),
        "within_3_percent_rate": sum(1 for r in completed if r["within_3_percent"]) / len(completed),
    }


def build_summary(plans_payload: dict[str, Any], validation_payload: dict[str, Any] | None,
                  oracle_payload: dict[str, Any], summary_path: Path, report_path: Path) -> dict[str, Any]:
    started = utc_now()
    oracle_by_id = {row["workload_id"]: row for row in oracle_payload.get("workloads", [])}
    validation_by_id = {
        row["workload_id"]: row
        for row in (validation_payload or {}).get("workloads", [])
    }
    rows = []
    for selection in plans_payload.get("selections", []):
        workload_id = selection["workload_id"]
        if selection.get("status") != "selected":
            rows.append({"workload_id": workload_id, "status": selection.get("status"), "skip_reason": selection.get("skip_reason")})
            continue
        selected_kernel = selection["selection"]["selected_kernel"]
        oracle_row = oracle_by_id.get(workload_id)
        if not oracle_row or oracle_row.get("status") != "completed":
            rows.append({"workload_id": workload_id, "status": "missing_oracle"})
            continue
        oracle_kernel, oracle_latency = oracle_winner(oracle_row)
        selected_latency = latency_for_kernel(oracle_row, selected_kernel)
        regret = None
        if selected_latency is not None and oracle_latency:
            regret = selected_latency / oracle_latency - 1.0
        validation = validation_by_id.get(workload_id, {})
        rows.append(
            {
                "workload_id": workload_id,
                "status": "completed",
                "category": selection.get("category"),
                "held_out": selection.get("held_out"),
                "profile_winner": selected_kernel,
                "compiler_selected_kernel": selected_kernel,
                "actual_dispatched_kernel": validation.get("actual_dispatched_kernel"),
                "fresh_oracle_kernel": oracle_kernel,
                "selected_fresh_latency_ms": selected_latency,
                "oracle_fresh_latency_ms": oracle_latency,
                "top1_correct": selected_kernel == oracle_kernel,
                "relative_regret": regret,
                "within_1_percent": regret is not None and regret <= 0.01,
                "within_3_percent": regret is not None and regret <= 0.03,
                "planned_equals_actual": validation.get("planned_equals_actual"),
                "config_equals_actual": validation.get("config_equals_actual"),
                "correctness_passed": (validation.get("correctness") or {}).get("passed"),
            }
        )

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("category"):
            groups.setdefault(row["category"], []).append(row)
    aggregates = {name: aggregate_rows(group_rows) for name, group_rows in groups.items()}
    aggregates["overall"] = aggregate_rows(rows)
    payload = {
        "schema": "triton_matmul_bias_relu_exact_selection_summary",
        "schema_version": SCHEMA_VERSION,
        "mode": "aggregate-report",
        "backend": BACKEND,
        "profile_sha256": plans_payload["profile_sha256"],
        "execution_plans_sha256": plans_payload.get("execution_plans_sha256"),
        "fresh_oracle_sha256": sha256_file(Path(oracle_payload.get("artifact_path", ""))) if oracle_payload.get("artifact_path") else None,
        "environment": plans_payload.get("environment", {}),
        "benchmark_config": oracle_payload.get("benchmark_config", {}),
        "workloads": rows,
        "aggregates": aggregates,
        "utc_start": started,
        "utc_end": utc_now(),
    }
    write_json(summary_path, payload)
    write_report(report_path, payload)
    return payload


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Triton Exact-Profile Kernel Selection",
        "",
        "This PR B report covers exact-profile Triton V1/V3 selection, plan-driven dispatch, and a separate fresh oracle sweep.",
        "",
        "## Aggregates",
        "",
        "| Group | Completed | Top-1 | Mean regret | Median regret | P95 regret | Within 1% | Within 3% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, agg in summary.get("aggregates", {}).items():
        lines.append(
            f"| {name} | {agg.get('completed_workload_count', 0)} | "
            f"{agg.get('exact_top1_accuracy', 0.0):.4f} | {agg.get('mean_regret', 0.0):.6f} | "
            f"{agg.get('median_regret', 0.0):.6f} | {agg.get('p95_regret', 0.0):.6f} | "
            f"{agg.get('within_1_percent_rate', 0.0):.4f} | {agg.get('within_3_percent_rate', 0.0):.4f} |"
        )
    lines += [
        "",
        "## Per-Workload Selection",
        "",
        "| Workload | Profile winner | Compiler selected | Actual dispatched | Fresh oracle | Top-1 | Regret | Within 1% | Within 3% |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("workloads", []):
        if row.get("status") != "completed":
            lines.append(f"| {row.get('workload_id')} | skipped | skipped | skipped | skipped |  |  |  |  |")
            continue
        lines.append(
            f"| {row['workload_id']} | {row['profile_winner']} | {row['compiler_selected_kernel']} | "
            f"{row.get('actual_dispatched_kernel')} | {row.get('fresh_oracle_kernel')} | "
            f"{row['top1_correct']} | {row['relative_regret']:.6f} | {row['within_1_percent']} | {row['within_3_percent']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/matmul_postop_workloads.json")
    parser.add_argument("--profile", default="trace/matmul_postop_triton_fixed_config_profile.json")
    parser.add_argument("--runner", default="tools/run_triton_matmul_bias_relu_benchmark.py")
    parser.add_argument("--execution-plans-output", default="trace/matmul_postop_triton_execution_plans.json")
    parser.add_argument("--plan-validation-output", default="trace/matmul_postop_triton_plan_validation.json")
    parser.add_argument("--fresh-oracle-output", default="trace/matmul_postop_triton_fresh_oracle.json")
    parser.add_argument("--fresh-oracle-report-output", default="trace/matmul_postop_triton_fresh_oracle_report.md")
    parser.add_argument("--summary-output", default="trace/matmul_postop_triton_exact_selection_summary.json")
    parser.add_argument("--report-output", default="trace/matmul_postop_triton_exact_selection_report.md")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-use-plan", action="store_true")
    parser.add_argument("--skip-fresh-oracle", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 50 or args.iterations < 300 or args.repeats < 5:
        raise ValueError("formal PR B runs require warmup>=50, iterations>=300, repeats>=5")
    work_dir = Path.cwd()
    manifest = Path(args.manifest)
    if len(load_manifest(manifest)) != 33:
        raise ValueError("canonical manifest must contain 33 workloads")
    profile_path = Path(args.profile)
    if profile_path.resolve() == Path(args.fresh_oracle_output).resolve():
        raise ValueError("fresh oracle artifact cannot be the same file as profile artifact")
    plans_path = Path(args.execution_plans_output)
    plans_payload, _plans = build_selection(profile_path, plans_path)
    validation_payload = None
    if not args.skip_use_plan:
        validation_payload = run_plan_validation(
            Path(args.runner), plans_payload, Path(args.plan_validation_output),
            work_dir, args.warmup, args.iterations, args.repeats
        )
    if args.skip_fresh_oracle:
        return 0
    oracle_payload = run_fresh_oracle(
        Path(args.runner), manifest, Path(args.fresh_oracle_output),
        Path(args.fresh_oracle_report_output), work_dir, args.warmup, args.iterations, args.repeats
    )
    oracle_payload["artifact_path"] = str(Path(args.fresh_oracle_output))
    build_summary(
        plans_payload,
        validation_payload,
        oracle_payload,
        Path(args.summary_output),
        Path(args.report_output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
