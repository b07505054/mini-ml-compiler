#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


REQUIRED_ARTIFACTS = {
    "execution_plan": "cv_execution_plan_v2.json",
    "static_schedule": "cv_static_schedule.json",
    "subgraph_partition": "cv_subgraph_partition.json",
    "cost_report": "cv_cost_report.json",
    "cost_planner": "cv_cost_based_planner.json",
}

OPTIONAL_ARTIFACTS = {
    "memory_plan": "cv_memory_plan.json",
    "runtime_timeline": "cv_runtime_timeline.json",
    "runtime_replan": "cv_runtime_replan.json",
}

REQUIRED_PLAN_FIELDS = {
    "step_id",
    "lowered_op_type",
    "backend",
    "dependency_steps",
    "memory_offset",
    "launch_config",
}

ALLOWED_BACKENDS = {"CPU", "Metal", "MockGPU", "CUDA"}
HEAVY_OP_MARKERS = {
    "Conv2D",
    "FusedConvBatchNormReLU",
    "Linear",
    "MatMul",
    "RMSNorm",
}
FLOAT_TOLERANCE = 1e-6


class ValidationResult:
    def __init__(self):
        self.checks = []
        self.failures = []
        self.warnings = []

    @property
    def passed(self):
        return not self.failures

    def check(self, group, passed, message):
        self.checks.append(
            {
                "group": group,
                "passed": bool(passed),
                "message": message,
            }
        )
        if not passed:
            self.failures.append(
                {
                    "group": group,
                    "message": message,
                }
            )

    def warn(self, group, message):
        self.warnings.append(
            {
                "group": group,
                "message": message,
            }
        )


def load_json(path, result, group):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result.check(group, False, f"{path} is not valid JSON: {exc}")
    except OSError as exc:
        result.check(group, False, f"failed to read {path}: {exc}")
    return None


def require_list(payload, result, group, name):
    if not isinstance(payload, list):
        result.check(group, False, f"{name} must be a JSON array")
        return []
    result.check(group, True, f"{name} is a JSON array with {len(payload)} entries")
    return payload


def validate_required_inputs(trace_dir, result):
    loaded = {}
    for key, filename in REQUIRED_ARTIFACTS.items():
        path = trace_dir / filename
        if not path.exists():
            result.check("inputs", False, f"required artifact missing: {path}")
            continue
        result.check("inputs", True, f"found required artifact: {filename}")
        loaded[key] = load_json(path, result, "inputs")

    optional = {}
    for key, filename in OPTIONAL_ARTIFACTS.items():
        path = trace_dir / filename
        if path.exists():
            result.check("inputs", True, f"found optional artifact: {filename}")
            optional[key] = load_json(path, result, "inputs")
        else:
            result.warn("inputs", f"optional artifact not present: {filename}")

    return loaded, optional


def validate_execution_plan(plan_payload, result):
    steps = require_list(plan_payload, result, "execution_plan", "execution plan")
    if not steps:
        result.check("execution_plan", False, "execution plan has no steps")
        return set()

    step_ids = []
    plan_backends = set()
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            result.check("execution_plan", False, f"step {index} must be an object")
            continue

        missing = REQUIRED_PLAN_FIELDS - set(step)
        result.check(
            "execution_plan",
            not missing,
            f"step {index} has required fields"
            if not missing
            else f"step {index} missing fields: {sorted(missing)}",
        )
        if missing:
            continue

        step_id = step["step_id"]
        step_ids.append(step_id)

        dependencies = step["dependency_steps"]
        if not isinstance(dependencies, list):
            result.check(
                "execution_plan",
                False,
                f"step {step_id} dependency_steps must be an array",
            )
            dependencies = []

        backend = step["backend"]
        plan_backends.add(backend)
        result.check(
            "execution_plan",
            backend in ALLOWED_BACKENDS,
            f"step {step_id} backend {backend} is allowed",
        )

        memory_offset = step["memory_offset"]
        result.check(
            "execution_plan",
            isinstance(memory_offset, int) and memory_offset >= 0,
            f"step {step_id} memory_offset is non-negative",
        )

    expected_ids = list(range(len(steps)))
    result.check(
        "execution_plan",
        step_ids == expected_ids,
        "step_id values are contiguous from 0"
        if step_ids == expected_ids
        else f"step_id values must be contiguous from 0; found {step_ids}",
    )

    step_id_set = set(step_ids)
    for step in steps:
        if not isinstance(step, dict) or "step_id" not in step:
            continue
        step_id = step["step_id"]
        for dependency in step.get("dependency_steps", []):
            result.check(
                "execution_plan",
                dependency in step_id_set,
                f"step {step_id} dependency {dependency} exists",
            )
            result.check(
                "execution_plan",
                isinstance(dependency, int) and dependency < step_id,
                f"step {step_id} dependency {dependency} precedes current step",
            )

    return plan_backends


def schedule_backends(schedule_payload, result):
    entries = require_list(schedule_payload, result, "backend_placement", "static schedule")
    backends = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            result.check("backend_placement", False, f"schedule entry {index} must be an object")
            continue
        backend = entry.get("backend")
        if backend:
            backends.add(backend)
        result.check(
            "backend_placement",
            backend in ALLOWED_BACKENDS,
            f"schedule entry {index} backend {backend} is allowed",
        )
    return backends


def is_heavy_op(op_type):
    return any(marker in str(op_type) for marker in HEAVY_OP_MARKERS)


def validate_backend_placement(
    plan_payload,
    schedule_payload,
    partition_payload,
    result,
    allow_cpu_heavy_ops,
):
    plan_steps = plan_payload if isinstance(plan_payload, list) else []
    schedule_entries = schedule_payload if isinstance(schedule_payload, list) else []
    seen_backends = schedule_backends(schedule_payload, result)
    seen_backends.update(
        step.get("backend")
        for step in plan_steps
        if isinstance(step, dict) and step.get("backend")
    )
    seen_backends.discard(None)

    partitions = require_list(
        partition_payload,
        result,
        "backend_placement",
        "subgraph partition",
    )
    for index, partition in enumerate(partitions):
        if not isinstance(partition, dict):
            result.check("backend_placement", False, f"partition {index} must be an object")
            continue
        backend = partition.get("backend")
        result.check(
            "backend_placement",
            backend in seen_backends,
            f"partition {index} backend {backend} is seen in schedule or execution plan",
        )

    heavy_cpu_ops = []
    for step in plan_steps:
        if not isinstance(step, dict):
            continue
        if step.get("backend") == "CPU" and is_heavy_op(step.get("lowered_op_type")):
            heavy_cpu_ops.append(step.get("lowered_op_type"))

    for entry in schedule_entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("backend") == "CPU" and is_heavy_op(entry.get("op_type")):
            heavy_cpu_ops.append(entry.get("op_type"))

    heavy_cpu_ops = sorted(set(op for op in heavy_cpu_ops if op))
    message = (
        "no high-compute op has likely unintended CPU fallback"
        if not heavy_cpu_ops
        else "likely unintended CPU fallback for high-compute ops: "
        + ", ".join(heavy_cpu_ops)
    )
    if allow_cpu_heavy_ops and heavy_cpu_ops:
        result.warn("backend_placement", message)
        result.check("backend_placement", True, "CPU heavy-op fallback allowed by flag")
    else:
        result.check("backend_placement", not heavy_cpu_ops, message)


def validate_cost_report(cost_payload, result):
    entries = require_list(cost_payload, result, "cost_report", "cost report")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            result.check("cost_report", False, f"cost report entry {index} must be an object")
            continue
        op_name = entry.get("op_name", f"entry {index}")
        for field in (
            "estimated_read_bytes",
            "estimated_write_bytes",
            "estimated_flops",
            "estimated_kernel_launch_cost",
            "estimated_backend_switch_cost",
        ):
            if field not in entry:
                continue
            value = entry[field]
            result.check(
                "cost_report",
                isinstance(value, (int, float)) and value >= 0,
                f"{op_name} {field} is non-negative",
            )
        if "actual_latency_ms" in entry:
            latency = entry["actual_latency_ms"]
            result.check(
                "cost_report",
                isinstance(latency, (int, float)) and latency >= -1,
                f"{op_name} actual_latency_ms is non-negative or -1 sentinel",
            )


def validate_cost_planner(planner_payload, result):
    summary = {}
    if not isinstance(planner_payload, dict):
        result.check("cost_planner", False, "cost planner report must be an object")
        return summary

    result.check(
        "cost_planner",
        planner_payload.get("format") == "cost_based_planner.v2",
        "cost planner format is cost_based_planner.v2",
    )

    candidates = planner_payload.get("candidates", [])
    result.check(
        "cost_planner",
        isinstance(candidates, list) and bool(candidates),
        "cost planner has candidates",
    )
    if not isinstance(candidates, list) or not candidates:
        return summary

    chosen = [candidate for candidate in candidates if candidate.get("chosen") is True]
    non_chosen = [candidate for candidate in candidates if candidate.get("chosen") is not True]
    result.check("cost_planner", len(chosen) == 1, "exactly one planner candidate is chosen")
    result.check("cost_planner", bool(non_chosen), "planner has at least one non-chosen candidate")

    latencies = []
    for candidate in candidates:
        name = candidate.get("name", "<unnamed>")
        latency = candidate.get("total_latency_ms")
        result.check(
            "cost_planner",
            isinstance(latency, (int, float)) and latency > 0,
            f"candidate {name} has positive total_latency_ms",
        )
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))

        op_costs = candidate.get("op_costs", [])
        result.check(
            "cost_planner",
            isinstance(op_costs, list) and bool(op_costs),
            f"candidate {name} has non-empty op_costs",
        )
        if not isinstance(op_costs, list):
            continue
        for op_cost in op_costs:
            op_name = op_cost.get("op_name", "<unknown>") if isinstance(op_cost, dict) else "<unknown>"
            latency_ms = op_cost.get("latency_ms") if isinstance(op_cost, dict) else None
            result.check(
                "cost_planner",
                isinstance(latency_ms, (int, float)) and latency_ms > 0,
                f"candidate {name} op {op_name} has positive latency_ms",
            )

    if len(chosen) == 1 and latencies:
        chosen_candidate = chosen[0]
        chosen_latency = chosen_candidate.get("total_latency_ms")
        min_latency = min(latencies)
        result.check(
            "cost_planner",
            isinstance(chosen_latency, (int, float))
            and math.isclose(float(chosen_latency), min_latency, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE),
            "chosen planner candidate has minimum total_latency_ms",
        )
        summary = {
            "name": chosen_candidate.get("name"),
            "total_latency_ms": chosen_latency,
            "candidate_count": len(candidates),
        }

    return summary


def validate_memory_plan(memory_payload, result):
    if memory_payload is None:
        result.warn("memory_plan", "cv_memory_plan.json not present; skipped optional memory checks")
        return
    if not isinstance(memory_payload, dict):
        result.check("memory_plan", False, "memory plan must be an object")
        return

    naive = memory_payload.get("naive_float_elements")
    planned = memory_payload.get("planned_peak_float_elements")
    saved = memory_payload.get("saved_float_elements")
    result.check(
        "memory_plan",
        isinstance(naive, (int, float)) and isinstance(planned, (int, float)) and planned <= naive,
        "planned_peak_float_elements <= naive_float_elements",
    )
    result.check(
        "memory_plan",
        isinstance(saved, (int, float)) and saved >= 0,
        "saved_float_elements >= 0",
    )


def print_summary(result):
    groups = []
    for check in result.checks:
        if check["group"] not in groups:
            groups.append(check["group"])

    print("Offline compiler artifact validation")
    for group in groups:
        group_checks = [check for check in result.checks if check["group"] == group]
        failed = [check for check in group_checks if not check["passed"]]
        status = "PASS" if not failed else "FAIL"
        print(f"[{status}] {group}: {len(group_checks) - len(failed)}/{len(group_checks)} checks passed")
        for check in failed:
            print(f"  - {check['message']}")

    for warning in result.warnings:
        print(f"[WARN] {warning['group']}: {warning['message']}")


def write_report(path, result, inputs, chosen_summary):
    payload = {
        "artifact_type": "compiler_artifact_validation_report",
        "passed": result.passed,
        "checks": result.checks,
        "failures": result.failures,
        "warnings": result.warnings,
        "inputs": inputs,
        "chosen_planner": chosen_summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate CV compiler/runtime artifacts before expensive backend "
            "or accelerator benchmarking."
        )
    )
    parser.add_argument("--trace-dir", type=Path, default=Path("trace"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-cpu-heavy-ops", action="store_true")
    args = parser.parse_args()

    result = ValidationResult()
    trace_dir = args.trace_dir
    loaded, optional = validate_required_inputs(trace_dir, result)

    if result.passed:
        plan_backends = validate_execution_plan(loaded.get("execution_plan"), result)
        validate_backend_placement(
            loaded.get("execution_plan"),
            loaded.get("static_schedule"),
            loaded.get("subgraph_partition"),
            result,
            args.allow_cpu_heavy_ops,
        )
        validate_cost_report(loaded.get("cost_report"), result)
        chosen_summary = validate_cost_planner(loaded.get("cost_planner"), result)
        validate_memory_plan(optional.get("memory_plan"), result)
        if plan_backends:
            result.check(
                "backend_placement",
                plan_backends.issubset(ALLOWED_BACKENDS),
                "execution plan backends are in the allowed backend set",
            )
    else:
        chosen_summary = {}

    inputs = {
        "trace_dir": str(trace_dir),
        "required": {
            key: str(trace_dir / filename)
            for key, filename in REQUIRED_ARTIFACTS.items()
        },
        "optional": {
            key: str(trace_dir / filename)
            for key, filename in OPTIONAL_ARTIFACTS.items()
        },
    }

    if args.report:
        write_report(args.report, result, inputs, chosen_summary)

    print_summary(result)
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
