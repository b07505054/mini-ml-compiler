#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


REQUIRED_OP_COST_FIELDS = {
    "op_name",
    "op_type",
    "backend",
    "latency_ms",
    "launch_overhead_ms",
    "compute_time_ms",
    "memory_time_ms",
    "transfer_cost_ms",
    "estimated_flops",
    "estimated_bytes",
    "cost_source",
    "decision_reason",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="trace/cv_cost_based_planner.json")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if payload.get("format") != "cost_based_planner.v2":
        raise SystemExit("expected cost_based_planner.v2")

    candidates = payload.get("candidates", [])
    if not candidates:
        raise SystemExit("planner report has no candidates")

    chosen = [candidate for candidate in candidates if candidate.get("chosen")]
    if len(chosen) != 1:
        raise SystemExit("expected exactly one chosen planner candidate")

    for candidate in candidates:
        op_costs = candidate.get("op_costs", [])
        if not op_costs:
            raise SystemExit(f"candidate {candidate.get('name')} has no op_costs")
        for op_cost in op_costs:
            missing = REQUIRED_OP_COST_FIELDS - set(op_cost)
            if missing:
                raise SystemExit(
                    f"candidate {candidate.get('name')} op {op_cost.get('op_name')} "
                    f"missing fields: {sorted(missing)}"
                )
            if op_cost["latency_ms"] <= 0:
                raise SystemExit(f"op {op_cost['op_name']} has non-positive latency")

    if not any(
        op.get("transfer_cost_ms", 0) > 0
        for candidate in candidates
        for op in candidate.get("op_costs", [])
    ):
        raise SystemExit("expected at least one candidate to include transfer cost")

    print(
        "validated cost_based_planner.v2: "
        f"chosen={chosen[0].get('name')} candidates={len(candidates)}"
    )


if __name__ == "__main__":
    main()
