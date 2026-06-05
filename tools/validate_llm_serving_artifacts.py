#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


REQUIRED_FILES = [
    "llm_graph_ir.json",
    "serving_execution_plan.json",
    "kv_cache_plan.json",
    "memory_plan.json",
    "scheduling_plan.json",
    "artifact_provenance.json",
    "candidate_execution_plans.json",
    "serving_framework_contract.json",
    "memory_timeline.json",
    "validation_manifest.json",
]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(condition, name, detail):
    return {
        "name": name,
        "passed": bool(condition),
        "detail": detail,
    }


def validate_artifacts(artifact_dir):
    artifact_dir = Path(artifact_dir)

    results = []

    files = {}
    for filename in REQUIRED_FILES:
        path = artifact_dir / filename
        exists = path.exists()
        results.append(check(
            exists,
            f"{filename}_exists",
            str(path),
        ))

        if exists:
            files[filename] = load_json(path)

    if len(files) != len(REQUIRED_FILES):
        return results

    graph_ir = files["llm_graph_ir.json"]
    execution_plan = files["serving_execution_plan.json"]
    kv_plan = files["kv_cache_plan.json"]
    memory_plan = files["memory_plan.json"]
    scheduling_plan = files["scheduling_plan.json"]
    provenance = files["artifact_provenance.json"]
    candidate_plans = files["candidate_execution_plans.json"]
    serving_framework_contract = files["serving_framework_contract.json"]
    memory_timeline = files["memory_timeline.json"]
    manifest = files["validation_manifest.json"]

    results.append(check(
        graph_ir.get("artifact_type") == "llm_graph_ir",
        "graph_ir_type_valid",
        graph_ir.get("artifact_type"),
    ))

    results.append(check(
        execution_plan.get("artifact_type") == "serving_execution_plan",
        "execution_plan_type_valid",
        execution_plan.get("artifact_type"),
    ))

    phase_names = [phase["name"] for phase in execution_plan.get("phases", [])]
    results.append(check(
        "prefill" in phase_names and "decode" in phase_names,
        "prefill_decode_phase_present",
        phase_names,
    ))

    block_size = kv_plan.get("block_size_tokens", 0)
    num_blocks = kv_plan.get("num_blocks", 0)
    total_capacity = kv_plan.get("total_token_capacity", 0)

    results.append(check(
        block_size > 0,
        "kv_cache_block_size_positive",
        block_size,
    ))

    results.append(check(
        num_blocks > 0,
        "kv_cache_num_blocks_positive",
        num_blocks,
    ))

    results.append(check(
        total_capacity == block_size * num_blocks,
        "kv_cache_capacity_consistent",
        {
            "total_token_capacity": total_capacity,
            "expected": block_size * num_blocks,
        },
    ))

    results.append(check(
        kv_plan.get("allocation_strategy") == "block_based",
        "kv_cache_allocation_block_based",
        kv_plan.get("allocation_strategy"),
    ))

    results.append(check(
        memory_plan.get("peak_prefill_memory_mb", 0) > 0,
        "prefill_memory_positive",
        memory_plan.get("peak_prefill_memory_mb"),
    ))

    results.append(check(
        memory_plan.get("peak_decode_memory_mb", 0) > 0,
        "decode_memory_positive",
        memory_plan.get("peak_decode_memory_mb"),
    ))

    results.append(check(
        memory_plan.get("fits_memory_budget") is True,
        "memory_budget_not_exceeded",
        {
            "peak_decode_memory_mb": memory_plan.get("peak_decode_memory_mb"),
            "memory_budget_mb": memory_plan.get("memory_budget_mb"),
            "fits_memory_budget": memory_plan.get("fits_memory_budget"),
        },
    ))

    queue_names = [
        queue["name"]
        for queue in scheduling_plan.get("queues", [])
    ]

    results.append(check(
        "prefill_queue" in queue_names and "decode_queue" in queue_names,
        "scheduling_queues_present",
        queue_names,
    ))

    results.append(check(
        provenance.get("artifact_type") == "artifact_provenance"
        and bool(provenance.get("outputs")),
        "artifact_provenance_present",
        {
            "artifact_type": provenance.get("artifact_type"),
            "output_count": len(provenance.get("outputs", [])),
        },
    ))

    results.append(check(
        candidate_plans.get("artifact_type") == "candidate_execution_plans"
        and bool(candidate_plans.get("plans")),
        "candidate_plans_present",
        {
            "artifact_type": candidate_plans.get("artifact_type"),
            "plan_count": len(candidate_plans.get("plans", [])),
        },
    ))

    framework_targets = serving_framework_contract.get("framework_targets", {})
    required_frameworks = {"vllm", "sglang", "triton_server", "tensorrt"}
    results.append(check(
        serving_framework_contract.get("artifact_type") == "serving_framework_contract"
        and required_frameworks.issubset(set(framework_targets)),
        "serving_framework_contract_present",
        {
            "artifact_type": serving_framework_contract.get("artifact_type"),
            "framework_targets": sorted(framework_targets),
        },
    ))

    results.append(check(
        memory_timeline.get("artifact_type") == "memory_timeline"
        and bool(memory_timeline.get("events")),
        "memory_timeline_present",
        {
            "artifact_type": memory_timeline.get("artifact_type"),
            "event_count": len(memory_timeline.get("events", [])),
        },
    ))

    expected_outputs = manifest.get("expected_outputs", [])
    results.append(check(
        sorted(expected_outputs) == sorted(REQUIRED_FILES),
        "manifest_expected_outputs_complete",
        expected_outputs,
    ))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", default="trace/llm_artifact_validation_report.json")
    args = parser.parse_args()

    results = validate_artifacts(Path(args.artifacts))
    passed = sum(1 for item in results if item["passed"])
    failed = sum(1 for item in results if not item["passed"])

    report = {
        "artifact_type": "llm_artifact_validation_report",
        "artifacts": args.artifacts,
        "summary": {
            "passed": passed,
            "failed": failed,
            "total": len(results),
            "status": "passed" if failed == 0 else "failed",
        },
        "checks": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(out)
    print(report["summary"]["status"])

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
