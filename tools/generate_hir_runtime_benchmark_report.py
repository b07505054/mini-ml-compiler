#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def first_step(plan):
    steps = plan.get("steps", [])
    if not steps:
        raise SystemExit(f"{plan.get('source', 'execution plan')} has no execution steps")
    return steps[0]


def build_entry(name, plan_path):
    plan = load_json(plan_path)
    step = first_step(plan)
    contract = step.get("runtime_dispatch_contract", {})
    selection = step.get("kernel_selection", {})
    evidence = selection.get("evidence") or {}
    op_type = step.get("op_type")
    custom_ms = evidence.get("custom_latency_ms")
    fallback_ms = evidence.get("fallback_latency_ms")
    custom_wins = (
        isinstance(custom_ms, (int, float))
        and isinstance(fallback_ms, (int, float))
        and custom_ms < fallback_ms
    )
    expected_kernel = selection.get("candidate_kernel") if custom_wins else selection.get("fallback_kernel")

    checks = {
        "compiler_emitted_typed_hir_op": isinstance(op_type, str) and op_type.startswith("hir."),
        "runtime_dispatch_contract_present": bool(contract),
        "runtime_decision_profile_calibrated": selection.get("profile_calibrated") is True,
        "runtime_decision_matches_benchmark": step.get("runtime_kernel") == expected_kernel,
        "numeric_correctness_passed": evidence.get("correct") is True,
        "benchmark_available": isinstance(custom_ms, (int, float))
        and isinstance(fallback_ms, (int, float)),
    }

    return {
        "name": name,
        "execution_plan": str(plan_path),
        "compiler_emitted_op": op_type,
        "runtime_op_type": step.get("runtime_op_type"),
        "runtime_kernel": step.get("runtime_kernel"),
        "backend": step.get("backend"),
        "fallback_kernel": selection.get("fallback_kernel"),
        "selection_reason": selection.get("selection_reason"),
        "profile_source": selection.get("profile_source"),
        "latency": {
            "custom_ms": evidence.get("custom_latency_ms"),
            "fallback_ms": evidence.get("fallback_latency_ms"),
            "speedup": evidence.get("speedup"),
        },
        "correct": evidence.get("correct"),
        "custom_kernel_was_faster": custom_wins,
        "runtime_dispatch_contract": contract,
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_markdown(report, path):
    lines = [
        "# HIR Runtime Benchmark Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "| HIR op | Runtime kernel | Backend | Custom ms | Baseline ms | Speedup | Correct |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for entry in report["entries"]:
        latency = entry["latency"]
        lines.append(
            "| {op} | {kernel} | {backend} | {custom} | {fallback} | {speedup} | {correct} |".format(
                op=entry["compiler_emitted_op"],
                kernel=entry["runtime_kernel"],
                backend=entry["backend"],
                custom=latency["custom_ms"],
                fallback=latency["fallback_ms"],
                speedup=latency["speedup"],
                correct=entry["correct"],
            )
        )

    lines.extend([
        "",
        "## Validation",
        "",
    ])

    for entry in report["entries"]:
        lines.append(f"### {entry['name']}")
        lines.append("")
        lines.append(f"- Compiler emitted: `{entry['compiler_emitted_op']}`")
        lines.append(f"- Runtime dispatch: `{entry['runtime_kernel']}` on `{entry['backend']}`")
        lines.append(f"- Selection reason: `{entry['selection_reason']}`")
        lines.append(f"- Profile source: `{entry['profile_source']}`")
        for key, value in entry["checks"].items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matmul-plan", default="trace/mlir_execution_plan.json")
    parser.add_argument("--rmsnorm-plan", default="trace/rmsnorm_execution_plan.json")
    parser.add_argument("--qmatmul-plan", default="trace/qmatmul_execution_plan.json")
    parser.add_argument("--json-output", default="trace/hir_runtime_benchmark_report.json")
    parser.add_argument("--markdown-output", default="trace/hir_runtime_benchmark_report.md")
    args = parser.parse_args()

    entries = [
        build_entry("MatMul-Bias-ReLU", args.matmul_plan),
        build_entry("RMSNorm", args.rmsnorm_plan),
    ]
    if Path(args.qmatmul_plan).exists():
        entries.append(build_entry("INT8 QMatMul-Bias-ReLU", args.qmatmul_plan))
    report = {
        "artifact_type": "hir_runtime_benchmark_report",
        "status": "passed" if all(entry["passed"] for entry in entries) else "failed",
        "summary": {
            "num_entries": len(entries),
            "num_passed": sum(1 for entry in entries if entry["passed"]),
            "num_failed": sum(1 for entry in entries if not entry["passed"]),
        },
        "entries": entries,
    }

    json_output = Path(args.json_output)
    markdown_output = Path(args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, markdown_output)

    if report["status"] != "passed":
        raise SystemExit(f"HIR runtime benchmark report failed: {json_output}")

    print(f"Wrote {json_output}")
    print(f"Wrote {markdown_output}")


if __name__ == "__main__":
    main()
