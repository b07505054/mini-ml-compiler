#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_optional_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return load_json(path)


def chosen_candidate(planner):
    chosen = [candidate for candidate in planner.get("candidates", []) if candidate.get("chosen")]
    if len(chosen) != 1:
        raise SystemExit("expected exactly one chosen planner candidate")
    return chosen[0]


def candidate_named(planner, name):
    for candidate in planner.get("candidates", []):
        if candidate.get("name") == name:
            return candidate
    return None


def first_step(plan):
    steps = plan.get("steps", [])
    if not steps:
        raise SystemExit(f"{plan.get('source', 'plan')} has no steps")
    return steps[0]


def dispatch_summary(name, plan_path):
    plan = load_json(plan_path)
    step = first_step(plan)
    descriptor = step.get("dispatch_descriptor") or {}
    tile_decision = descriptor.get("tile_decision") or {}
    candidates = tile_decision.get("candidates") or []
    selection = step.get("kernel_selection") or {}
    evidence = selection.get("evidence") or {}
    return {
        "name": name,
        "plan": str(plan_path),
        "hir_op": step.get("op_type"),
        "runtime_kernel": step.get("runtime_kernel"),
        "backend": step.get("backend"),
        "selection_reason": selection.get("selection_reason"),
        "profile_calibrated": selection.get("profile_calibrated"),
        "shape_bucket": selection.get("shape_bucket"),
        "latency": {
            "custom_ms": evidence.get("custom_latency_ms"),
            "fallback_ms": evidence.get("fallback_latency_ms"),
            "speedup": evidence.get("speedup"),
            "correct": evidence.get("correct"),
        },
        "dispatch_descriptor": {
            "target": descriptor.get("target"),
            "shape": descriptor.get("shape"),
            "selected_tile": tile_decision.get("selected_tile"),
            "selected_sram_bytes": tile_decision.get("selected_sram_bytes"),
            "decision_reason": tile_decision.get("decision_reason"),
            "num_legal_tile_candidates": sum(1 for candidate in candidates if candidate.get("legal")),
            "num_rejected_tile_candidates": sum(1 for candidate in candidates if not candidate.get("legal")),
        } if descriptor else None,
    }


def prefetch_summary(path):
    payload = load_optional_json(path)
    if not payload:
        return {
            "available": False,
            "path": str(path),
        }
    faster = payload.get("prefetch_p95_ms", 0) < payload.get("baseline_p95_ms", 0)
    selected = payload.get("selection_ready") is True
    return {
        "available": True,
        "path": str(path),
        "technology": payload.get("technology"),
        "input": payload.get("input"),
        "decision": payload.get("decision"),
        "metric": payload.get("metric"),
        "candidate_kernel": payload.get("candidate_kernel"),
        "fallback_kernel": payload.get("fallback_kernel"),
        "correct": payload.get("correct"),
        "baseline_p95_ms": payload.get("baseline_p95_ms"),
        "prefetch_p95_ms": payload.get("prefetch_p95_ms"),
        "speedup": payload.get("speedup"),
        "selection_ready": selected,
        "selection_reason": payload.get("selection_reason"),
        "decision_matches_measurement": selected == faster,
    }


def build_report(args):
    planner = load_json(args.cost_planner)
    chosen = chosen_candidate(planner)
    all_metal = candidate_named(planner, "all_metal")
    if not all_metal:
        raise SystemExit("expected all_metal candidate for before/after comparison")

    compiler_planner = {
        "format": planner.get("format"),
        "chosen_plan": chosen.get("name"),
        "chosen_latency_ms": chosen.get("total_latency_ms"),
        "all_metal_latency_ms": all_metal.get("total_latency_ms"),
        "all_metal_transfer_cost_ms": all_metal.get("transfer_cost_ms"),
        "decision_delta_vs_all_metal_ms": round(
            all_metal.get("total_latency_ms", 0) - chosen.get("total_latency_ms", 0),
            6,
        ),
        "decision_reason": (
            "CostReport-driven planner keeps memory-heavy CPU ops on CPU because "
            "moving pool/flatten to Metal adds transfer cost."
        ),
        "chosen_op_costs": chosen.get("op_costs", []),
    }

    dispatch_entries = [
        dispatch_summary("MatMul-Bias-ReLU", args.matmul_plan),
        dispatch_summary("RMSNorm", args.rmsnorm_plan),
        dispatch_summary("INT8 QMatMul-Bias-ReLU", args.qmatmul_plan),
    ]
    prefetch = prefetch_summary(args.prefetch_benchmark)

    checks = {
        "planner_uses_cost_report_v2": planner.get("format") == "cost_based_planner.v2",
        "planner_decision_changed_from_all_metal": chosen.get("name") != "all_metal",
        "all_metal_has_transfer_cost": all_metal.get("transfer_cost_ms", 0) > 0,
        "matmul_profile_falls_back_when_fused_slower": dispatch_entries[0]["selection_reason"] == "profile_calibrated_fallback",
        "rmsnorm_profile_selects_cuda": dispatch_entries[1]["runtime_kernel"] == "fused_rmsnorm_cuda",
        "qmatmul_profile_selects_int8": dispatch_entries[2]["runtime_kernel"] == "int8_qmatmul_bias_relu",
        "dispatch_descriptors_have_tiles": all(
            entry["dispatch_descriptor"] is None or entry["dispatch_descriptor"]["selected_tile"]
            for entry in dispatch_entries
        ),
        "prefetch_candidate_profile_valid": (
            prefetch.get("available") is True
            and prefetch.get("correct") is True
            and prefetch.get("decision_matches_measurement") is True
        ),
    }

    return {
        "artifact_type": "compiler_runtime_decision_report",
        "format": "compiler_runtime_decision_report.v1",
        "status": "passed" if all(checks.values()) else "failed",
        "compiler_planner": compiler_planner,
        "kernel_and_dispatch_decisions": dispatch_entries,
        "prefetch_candidate_decision": prefetch,
        "checks": checks,
    }


def write_markdown(report, path):
    planner = report["compiler_planner"]
    lines = [
        "# Compiler Runtime Decision Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Planner Decision",
        "",
        f"- Chosen plan: `{planner['chosen_plan']}`",
        f"- Chosen latency: `{planner['chosen_latency_ms']} ms`",
        f"- All-Metal candidate latency: `{planner['all_metal_latency_ms']} ms`",
        f"- All-Metal transfer cost: `{planner['all_metal_transfer_cost_ms']} ms`",
        f"- Delta vs all-Metal: `{planner['decision_delta_vs_all_metal_ms']} ms`",
        f"- Reason: {planner['decision_reason']}",
        "",
        "## Kernel And Dispatch Decisions",
        "",
        "| Case | HIR op | Kernel | Backend | Selection | Custom ms | Fallback ms | Speedup | Tile | SRAM bytes |",
        "|---|---|---|---|---|---:|---:|---:|---|---:|",
    ]
    for entry in report["kernel_and_dispatch_decisions"]:
        latency = entry["latency"]
        descriptor = entry["dispatch_descriptor"] or {}
        tile = descriptor.get("selected_tile")
        tile_text = ""
        if tile:
            tile_text = f"{tile['m']}x{tile['n']}x{tile['k']}"
        lines.append(
            "| {name} | {op} | {kernel} | {backend} | {reason} | {custom} | {fallback} | {speedup} | {tile} | {sram} |".format(
                name=entry["name"],
                op=entry["hir_op"],
                kernel=entry["runtime_kernel"],
                backend=entry["backend"],
                reason=entry["selection_reason"],
                custom=latency["custom_ms"],
                fallback=latency["fallback_ms"],
                speedup=latency["speedup"],
                tile=tile_text,
                sram=descriptor.get("selected_sram_bytes", ""),
            )
        )

    prefetch = report.get("prefetch_candidate_decision", {})
    if prefetch.get("available"):
        lines.extend([
            "",
            "## CPU Software Prefetch Candidate",
            "",
            f"- Input: `{prefetch.get('input')}`",
            f"- Decision: `{prefetch.get('decision')}`",
            f"- Metric: `{prefetch.get('metric')}`",
            f"- Candidate: `{prefetch.get('candidate_kernel')}`",
            f"- Fallback: `{prefetch.get('fallback_kernel')}`",
            f"- Baseline p95: `{prefetch.get('baseline_p95_ms')}` ms",
            f"- Prefetch p95: `{prefetch.get('prefetch_p95_ms')}` ms",
            f"- Selection ready: `{prefetch.get('selection_ready')}`",
            f"- Selection reason: `{prefetch.get('selection_reason')}`",
        ])

    lines.extend(["", "## Checks", ""])
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-planner", default="trace/cv_cost_based_planner.json")
    parser.add_argument("--matmul-plan", default="trace/mlir_execution_plan.json")
    parser.add_argument("--rmsnorm-plan", default="trace/rmsnorm_execution_plan.json")
    parser.add_argument("--qmatmul-plan", default="trace/qmatmul_execution_plan.json")
    parser.add_argument("--prefetch-benchmark", default="trace/prefetch_matmul_benchmark.json")
    parser.add_argument("--json-output", default="trace/compiler_runtime_decision_report.json")
    parser.add_argument("--markdown-output", default="trace/compiler_runtime_decision_report.md")
    args = parser.parse_args()

    report = build_report(args)
    json_output = Path(args.json_output)
    markdown_output = Path(args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, markdown_output)
    if report["status"] != "passed":
        raise SystemExit(f"decision report failed: {json_output}")
    print(f"Wrote {json_output}")
    print(f"Wrote {markdown_output}")


if __name__ == "__main__":
    main()
