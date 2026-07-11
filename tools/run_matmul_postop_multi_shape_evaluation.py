#!/usr/bin/env python3
"""Run and summarize the multi-shape MatMul post-op evaluation suite.

This is an orchestration layer over apps/run_mlir_fused_kernel_benchmark.cpp.
It does not duplicate kernel implementations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matmul_postop_workloads import (  # noqa: E402
    VALID_CATEGORIES,
    VALID_PATTERNS,
    Workload,
    geometric_mean,
    load_manifest,
    pearson,
    percentile,
    postop_shape_for,
    spearman,
    static_cost,
)

VARIANTS = (
    "naive_unfused",
    "tiled_unfused",
    "naive_one_pass_fused",
    "tiled_one_pass_fused",
)


def latency_reduction_percent(speedup: float) -> float:
    return 0.0 if speedup <= 0 else (1.0 - (1.0 / speedup)) * 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-exe", required=True)
    parser.add_argument("--workload-manifest", default="benchmarks/matmul_postop_workloads.json")
    parser.add_argument("--category", action="append", choices=VALID_CATEGORIES)
    parser.add_argument("--workload-id", action="append")
    parser.add_argument("--pattern", action="append", choices=("bias", "elementwise-add", "elementwise_add", "all"))
    parser.add_argument("--mode", default="sweep-candidates", choices=("sweep-candidates",))
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--output", default="trace/matmul_postop_multi_shape_profile.json")
    parser.add_argument("--validation-output", default="trace/matmul_postop_multi_shape_validation.json")
    parser.add_argument("--summary-output", default="trace/matmul_postop_selection_summary.json")
    parser.add_argument("--correlation-output", default="trace/matmul_postop_correlation.csv")
    parser.add_argument("--report-output", default="trace/matmul_postop_multi_shape_report.md")
    parser.add_argument("--scratch-dir", default="trace/matmul_postop_multi_shape_runs")
    parser.add_argument("--formal-core-only", action="store_true")
    parser.add_argument("--include-held-out", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Generate deterministic synthetic outputs for tests.")
    return parser.parse_args()


def pattern_flag(pattern: str) -> str:
    return "elementwise-add" if pattern == "elementwise_add" else pattern


def kernel_id(pattern: str, variant: str) -> str:
    semantic = "matmul_bias_relu" if pattern == "bias" else "matmul_add_relu"
    prefix = {
        "naive_unfused": "cpu_naive",
        "tiled_unfused": "cpu_tiled",
        "naive_one_pass_fused": "cpu_naive",
        "tiled_one_pass_fused": "cpu_tiled",
    }[variant]
    suffix = "unfused_f32" if variant.endswith("unfused") else "one_pass_f32"
    return f"{prefix}_{semantic}_{suffix}"


def variant_from_kernel(value: str) -> str:
    for variant in VARIANTS:
        if kernel_id("bias", variant) == value or kernel_id("elementwise_add", variant) == value:
            return variant
    raise ValueError(f"unknown kernel id: {value}")


def select_workloads(args: argparse.Namespace, workloads: list[Workload]) -> list[Workload]:
    selected = []
    ids = set(args.workload_id or [])
    categories = set(args.category or [])
    for workload in workloads:
        if ids and workload.workload_id not in ids:
            continue
        if categories and workload.category not in categories:
            continue
        if args.formal_core_only and workload.tier != "formal_core":
            continue
        if workload.held_out and not args.include_held_out:
            continue
        selected.append(workload)
    return selected


def selected_patterns(args: argparse.Namespace, workload: Workload) -> list[str]:
    requested = args.pattern or ["all"]
    normalized = ["elementwise_add" if p == "elementwise-add" else p for p in requested]
    if "all" in normalized:
        return list(workload.patterns)
    return [p for p in workload.patterns if p in normalized]


def budget_for(args: argparse.Namespace, workload: Workload) -> tuple[int, int, int, str | None]:
    budget = workload.budget
    warmup = args.warmup if args.warmup is not None else budget.warmup
    iterations = args.iterations if args.iterations is not None else budget.iterations
    repeats = args.repeats if args.repeats is not None else budget.repeats
    lowered_reason = budget.lowered_reason
    formal = warmup >= 50 and iterations >= 300 and repeats >= 5
    if not formal and lowered_reason is None:
        lowered_reason = "explicit_cli_budget_below_formal_threshold"
    if warmup <= 0:
        raise ValueError("formal suite invocations require warmup > 0")
    return warmup, iterations, repeats, lowered_reason


def run_json(command: list[str], output: Path, dry_payload: dict[str, Any] | None) -> dict[str, Any]:
    if dry_payload is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(dry_payload, indent=2), encoding="utf-8")
    else:
        subprocess.run(command, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def synthetic_profile(workload: Workload, pattern: str, warmup: int, iterations: int, repeats: int) -> dict[str, Any]:
    cost = static_cost(pattern, workload.m, workload.n, workload.k, workload.dtype)
    base = max(0.001, cost["matmul_flops"] / 4.0e9 * 1000.0)
    pressure = cost["fusion_pressure_score"]
    fair_speedup = 1.0 + min(1.2, pressure * 90.0)
    variant_ms = {
        "naive_unfused": base * 1.35,
        "tiled_unfused": base,
        "naive_one_pass_fused": base * 1.20 / fair_speedup,
        "tiled_one_pass_fused": base / fair_speedup,
    }
    variants = {}
    ordered = sorted(variant_ms, key=variant_ms.get)
    for variant in VARIANTS:
        mean = variant_ms[variant]
        variants[variant] = {
            "kernel_id": kernel_id(pattern, variant),
            "rank": ordered.index(variant) + 1,
            "oracle_best": ordered[0] == variant,
            "samples_ms": [mean * (1.0 + i * 0.001) for i in range(repeats)],
            "statistics": {
                "sample_count": repeats,
                "mean_ms": mean,
                "median_ms": mean,
                "p50_ms": mean,
                "p95_ms": mean,
                "min_ms": mean,
                "max_ms": mean,
                "stddev_ms": mean * 0.001,
                "coefficient_of_variation": 0.001,
            },
            "correctness": {
                "passed": True,
                "atol": 0.0001,
                "rtol": 0.0001,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "contains_nan": False,
                "contains_inf": False,
            },
        }
    return {
        "schema": "kernel_benchmark_profile",
        "schema_version": 2,
        "benchmark": "matmul_postop_relu",
        "mode": "sweep-candidates",
        "machine": {"hostname": platform.node(), "uname": platform.platform()},
        "configuration": {
            "warmup": warmup,
            "iterations": iterations,
            "repeats": repeats,
            "dtype": workload.dtype,
            "m": workload.m,
            "n": workload.n,
            "k": workload.k,
            "tile_size": 32,
        },
        "patterns": {
            pattern: {
                "postop_semantics": "bias_shape_N" if pattern == "bias" else "elementwise_add_shape_MxN",
                "variants": variants,
                "comparisons": {
                    "fusion_speedup_fair": {
                        "speedup": variant_ms["tiled_unfused"] / variant_ms["tiled_one_pass_fused"],
                        "latency_reduction_percent": latency_reduction_percent(
                            variant_ms["tiled_unfused"] / variant_ms["tiled_one_pass_fused"]
                        ),
                    },
                    "full_stack_speedup": {
                        "speedup": variant_ms["naive_unfused"] / variant_ms["tiled_one_pass_fused"],
                        "latency_reduction_percent": latency_reduction_percent(
                            variant_ms["naive_unfused"] / variant_ms["tiled_one_pass_fused"]
                        ),
                    },
                },
            }
        },
    }


def make_plan(path: Path, workload: Workload, pattern: str, selected_kernel: str) -> str:
    payload = {
        "schema": "runtime_execution_plan",
        "schema_version": 2,
        "graph_id": f"{workload.workload_id}_{pattern}",
        "workload_shape": {"m": workload.m, "n": workload.n, "k": workload.k, "dtype": workload.dtype},
        "operations": [
            {
                "op_id": "matmul_postop_0",
                "op_type": "FusedMatMulBiasRelu" if pattern == "bias" else "FusedMatMulAddRelu",
                "backend": "cpu",
                "selected_kernel": selected_kernel,
                "kernel_config": {"tile_m": 32, "tile_n": 32, "tile_k": 32},
                "inputs": ["A", "B", "bias" if pattern == "bias" else "addend"],
                "outputs": ["output"],
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats_for(profile: dict[str, Any], pattern: str, variant: str) -> dict[str, Any]:
    return profile["patterns"][pattern]["variants"][variant]["statistics"]


def variant_record(profile: dict[str, Any], pattern: str, variant: str) -> dict[str, Any]:
    return profile["patterns"][pattern]["variants"][variant]


def summarize_selection(workload: Workload, pattern: str, profile: dict[str, Any], selected_kernel: str, fallback_reason: str | None) -> dict[str, Any]:
    variants = profile["patterns"][pattern]["variants"]
    correct = {k: v for k, v in variants.items() if v["correctness"]["passed"]}
    oracle_variant = min(correct, key=lambda v: correct[v]["statistics"]["mean_ms"])
    selected_variant = variant_from_kernel(selected_kernel)
    oracle_latency = correct[oracle_variant]["statistics"]["mean_ms"]
    selected_latency = correct[selected_variant]["statistics"]["mean_ms"]
    tiled_unfused_latency = correct["tiled_unfused"]["statistics"]["mean_ms"]
    fair_speedup = tiled_unfused_latency / correct["tiled_one_pass_fused"]["statistics"]["mean_ms"]
    full_stack_speedup = correct["naive_unfused"]["statistics"]["mean_ms"] / correct["tiled_one_pass_fused"]["statistics"]["mean_ms"]
    regret = (selected_latency - oracle_latency) / oracle_latency
    return {
        "workload_id": workload.workload_id,
        "category": workload.category,
        "tier": workload.tier,
        "held_out": workload.held_out,
        "pattern": pattern,
        "postop_shape": postop_shape_for(pattern, workload.m, workload.n),
        "shape": {"m": workload.m, "n": workload.n, "k": workload.k, "dtype": workload.dtype},
        "static_cost": static_cost(pattern, workload.m, workload.n, workload.k, workload.dtype),
        "compiler_selected_kernel": selected_kernel,
        "oracle_best_kernel": correct[oracle_variant]["kernel_id"],
        "compiler_selected_latency": selected_latency,
        "oracle_best_latency": oracle_latency,
        "tiled_unfused_latency": tiled_unfused_latency,
        "plan_speedup": tiled_unfused_latency / selected_latency,
        "oracle_speedup": tiled_unfused_latency / oracle_latency,
        "fair_fusion_speedup": fair_speedup,
        "fair_fusion_latency_reduction_percent": latency_reduction_percent(fair_speedup),
        "full_stack_speedup": full_stack_speedup,
        "selection_regret": regret,
        "top1_correct": selected_kernel == correct[oracle_variant]["kernel_id"],
        "fallback_reason": fallback_reason,
        "correctness": {variant: correct[variant]["correctness"] for variant in correct},
        "cv_unstable": any(v["statistics"]["coefficient_of_variation"] > 0.05 for v in correct.values()),
        "candidate_count": len(correct),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in records if not r["cv_unstable"]]
    if not usable:
        return {"count": 0}
    regrets = [r["selection_regret"] for r in usable]
    fair = [r["fair_fusion_speedup"] for r in usable if r["fair_fusion_speedup"] > 0]
    plan = [r["plan_speedup"] for r in usable if r["plan_speedup"] > 0]
    return {
        "count": len(usable),
        "excluded_unstable_cv": len(records) - len(usable),
        "top1_accuracy": sum(1 for r in usable if r["top1_correct"]) / len(usable),
        "mean_regret": statistics.fmean(regrets),
        "median_regret": statistics.median(regrets),
        "p95_regret": percentile(regrets, 95),
        "geomean_plan_speedup": geometric_mean(plan),
        "geomean_fair_fusion_speedup": geometric_mean(fair),
        "best_case_fair_fusion_speedup": max(fair),
        "worst_case_fair_fusion_speedup": min(fair),
    }


def write_report(path: Path, payload: dict[str, Any], skipped: list[dict[str, Any]]) -> None:
    exact = payload["aggregates"].get("exact_profiled", {"count": 0})
    fallback = payload["aggregates"].get("unprofiled_fallback", {"count": 0})
    lines = [
        "# MatMul Post-Op Multi-Shape Evaluation",
        "",
        "Static cost values are analytical features, not measured hardware traffic.",
        "",
        "## Summary",
        f"- Completed workload-pattern measurements: `{len(payload['workloads'])}`",
        f"- Skipped workloads: `{len(skipped)}`",
        f"- Exact-profiled top-1 accuracy: `{exact.get('top1_accuracy', 0):.6f}`",
        f"- Exact-profiled mean/median/p95 regret: `{exact.get('mean_regret', 0):.6f}` / `{exact.get('median_regret', 0):.6f}` / `{exact.get('p95_regret', 0):.6f}`",
        f"- Held-out fallback mean regret: `{fallback.get('mean_regret', 0):.6f}`",
        "",
        "## Layer 1: Compiler Correctness",
        "",
        "| Workload | Pattern | Profile match | Selected kernel | Oracle kernel | Fallback | Correct | Planned dispatch |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for record in payload["workloads"]:
        lines.append(
            f"| {record['workload_id']} | {record['pattern']} | "
            f"{'false' if record['fallback_reason'] else 'true'} | "
            f"{record['compiler_selected_kernel']} | {record['oracle_best_kernel']} | "
            f"{record['fallback_reason'] or ''} | true | {record.get('planned_equals_actual_dispatched', True)} |"
        )
    lines += [
        "",
        "## Layer 2: Static Optimization Impact",
        "",
        "The fused path is modeled as reducing runtime dispatch count from 3 to 1, logical intermediate tensors from 2 to 0, and full-output post-op passes from 2 to 0.",
        "",
        "## Layer 3: Measured Runtime Impact",
        "",
        "| Group | Count | Geo mean fair fusion | Geo mean plan speedup | Top-1 | Mean regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, agg in payload["aggregates"].items():
        if agg.get("count", 0):
            lines.append(
                f"| {name} | {agg['count']} | {agg['geomean_fair_fusion_speedup']:.6f} | "
                f"{agg['geomean_plan_speedup']:.6f} | {agg['top1_accuracy']:.6f} | {agg['mean_regret']:.6f} |"
            )
    lines += [
        "",
        "## Correlation Analysis",
        "",
    ]
    for key, value in payload["correlations"].items():
        lines.append(f"- `{key}`: Pearson `{value['pearson']:.6f}`, Spearman `{value['spearman']:.6f}`")
    if skipped:
        lines += ["", "## Skipped Workloads", ""]
        for item in skipped:
            lines.append(f"- `{item['workload_id']}`: {item['skip_reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    workloads = load_manifest(args.workload_manifest)
    selected = select_workloads(args, workloads)
    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    skipped = [
        {"workload_id": w.workload_id, "category": w.category, "skip_reason": w.skip_reason}
        for w in selected
        if w.status == "skipped_resource_limit"
    ]
    started = datetime.now(timezone.utc).isoformat()
    pid = os.getpid()

    for workload in selected:
        if workload.status != "active":
            continue
        warmup, iterations, repeats, lowered_reason = budget_for(args, workload)
        for pattern in selected_patterns(args, workload):
            stem = f"{workload.workload_id}_{pattern}"
            sweep_json = scratch / f"{stem}.sweep.json"
            sweep_md = scratch / f"{stem}.sweep.md"
            command = [
                args.benchmark_exe,
                "--mode", "sweep-candidates",
                "--pattern", pattern_flag(pattern),
                "--variant", "all",
                "--m", str(workload.m),
                "--n", str(workload.n),
                "--k", str(workload.k),
                "--warmup", str(warmup),
                "--iterations", str(iterations),
                "--repeats", str(repeats),
                "--output", str(sweep_json),
                "--report-output", str(sweep_md),
            ]
            profile = run_json(
                command,
                sweep_json,
                synthetic_profile(workload, pattern, warmup, iterations, repeats) if args.dry_run else None,
            )
            selected_variant_for_plan = (
                "tiled_unfused"
                if workload.held_out
                else min(
                    VARIANTS,
                    key=lambda v: stats_for(profile, pattern, v)["mean_ms"],
                )
            )
            selected_kernel = kernel_id(pattern, selected_variant_for_plan)
            fallback_reason = "no_exact_shape_match" if workload.held_out else None
            plan_path = scratch / f"{stem}.plan.json"
            plan_sha = make_plan(plan_path, workload, pattern, selected_kernel)
            use_json = scratch / f"{stem}.use_plan.json"
            use_md = scratch / f"{stem}.use_plan.md"
            use_command = [
                args.benchmark_exe,
                "--mode", "use-plan",
                "--execution-plan", str(plan_path),
                "--pattern", pattern_flag(pattern),
                "--m", str(workload.m),
                "--n", str(workload.n),
                "--k", str(workload.k),
                "--warmup", str(warmup),
                "--iterations", str(iterations),
                "--repeats", str(repeats),
                "--output", str(use_json),
                "--report-output", str(use_md),
            ]
            use_payload = None
            if args.dry_run:
                use_payload = synthetic_profile(workload, pattern, warmup, iterations, repeats)
                selected_variant = variant_from_kernel(selected_kernel)
                use_payload["mode"] = "use-plan"
                use_payload["patterns"][pattern]["variants"] = {
                    selected_variant: use_payload["patterns"][pattern]["variants"][selected_variant]
                }
                use_payload["patterns"][pattern]["variants"][selected_variant]["runtime_trace"] = {
                    "planned_kernel": selected_kernel,
                    "actual_dispatched_kernel": selected_kernel,
                    "dispatch_count": 1 if "one_pass" in selected_kernel else 3,
                    "plan_matched_runtime": True,
                }
            validation = run_json(use_command, use_json, use_payload)
            record = summarize_selection(workload, pattern, profile, selected_kernel, fallback_reason)
            record["benchmark_configuration"] = {
                "warmup": warmup,
                "iterations": iterations,
                "repeats": repeats,
                "lowered_reason": lowered_reason,
            }
            record["profile_artifact_sha256"] = hashlib.sha256(sweep_json.read_bytes()).hexdigest()
            record["execution_plan_sha256"] = plan_sha
            record["process_id"] = pid
            record["utc_start"] = started
            record["utc_end"] = datetime.now(timezone.utc).isoformat()
            selected_variant = variant_from_kernel(selected_kernel)
            runtime_trace = validation["patterns"][pattern]["variants"][selected_variant].get("runtime_trace", {})
            record["planned_equals_actual_dispatched"] = (
                runtime_trace.get("planned_kernel") == runtime_trace.get("actual_dispatched_kernel")
            )
            records.append(record)
            validations.append(
                {
                    "workload_id": workload.workload_id,
                    "pattern": pattern,
                    "use_plan_artifact": str(use_json),
                    "use_plan_is_profile_evidence": False,
                    "planned_equals_actual_dispatched": record["planned_equals_actual_dispatched"],
                    "fallback_reason": fallback_reason,
                }
            )

    aggregates: dict[str, Any] = {}
    for category in VALID_CATEGORIES:
        category_records = [r for r in records if r["category"] == category]
        if category_records:
            aggregates[category] = aggregate(category_records)
    aggregates["exact_profiled"] = aggregate([r for r in records if not r["held_out"]])
    aggregates["unprofiled_fallback"] = aggregate([r for r in records if r["held_out"]])
    for pattern in VALID_PATTERNS:
        pattern_records = [r for r in records if r["pattern"] == pattern]
        if pattern_records:
            aggregates[pattern] = aggregate(pattern_records)

    corr_inputs = {
        "k": [r["shape"]["k"] for r in records],
        "output_bytes": [r["static_cost"]["output_bytes"] for r in records],
        "fusion_pressure_score": [r["static_cost"]["fusion_pressure_score"] for r in records],
        "matmul_flops": [r["static_cost"]["matmul_flops"] for r in records],
        "output_bytes_per_matmul_flop": [
            r["static_cost"]["output_bytes"] / r["static_cost"]["matmul_flops"] for r in records
        ],
    }
    speedups = [r["fair_fusion_speedup"] for r in records]
    correlations = {}
    for name, values in corr_inputs.items():
        try:
            correlations[name] = {"pearson": pearson(values, speedups), "spearman": spearman(values, speedups)}
        except ValueError:
            correlations[name] = {"pearson": 0.0, "spearman": 0.0}

    payload = {
        "schema": "matmul_postop_multi_shape_evaluation",
        "schema_version": 1,
        "machine": {"hostname": platform.node(), "uname": platform.platform()},
        "benchmark_configuration": {"formal_warmup_min": 50, "formal_iterations_min": 300, "formal_repeats_min": 5},
        "workloads": records,
        "skipped_workloads": skipped,
        "aggregates": aggregates,
        "correlations": correlations,
    }

    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.validation_output).write_text(
        json.dumps(
            {
                "schema": "matmul_postop_multi_shape_validation",
                "schema_version": 1,
                "use_plan_outputs_are_profile_evidence": False,
                "validations": validations,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    Path(args.summary_output).write_text(
        json.dumps({"schema": "matmul_postop_selection_summary", "schema_version": 1, "aggregates": aggregates}, indent=2),
        encoding="utf-8",
    )
    with Path(args.correlation_output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "workload_id",
                "pattern",
                "k",
                "output_bytes",
                "fusion_pressure_score",
                "matmul_flops",
                "fair_fusion_speedup",
                "absolute_latency_saved_ms",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "workload_id": record["workload_id"],
                    "pattern": record["pattern"],
                    "k": record["shape"]["k"],
                    "output_bytes": record["static_cost"]["output_bytes"],
                    "fusion_pressure_score": record["static_cost"]["fusion_pressure_score"],
                    "matmul_flops": record["static_cost"]["matmul_flops"],
                    "fair_fusion_speedup": record["fair_fusion_speedup"],
                    "absolute_latency_saved_ms": record["tiled_unfused_latency"] - record["compiler_selected_latency"],
                }
            )
    write_report(Path(args.report_output), payload, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
