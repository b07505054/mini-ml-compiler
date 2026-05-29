#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


def detect_fused_matmul(text):
    pattern = re.compile(
        r"(?P<result>%[\w\d_]+)\s*=\s*linalg\.matmul\s*"
        r"\{fusion\.candidate\s*=\s*\"matmul_bias_relu\"\}",
        re.MULTILINE,
    )
    return list(pattern.finditer(text))


def build_lowered_graph(matches, source_path):
    ops = []

    for index, match in enumerate(matches):
        result_name = match.group("result")
        ops.append({
            "id": index,
            "name": f"fused_matmul_bias_relu_{index}",
            "source_result": result_name,
            "op_type": "FusedMatMulBiasReLU",
            "lowered_op_type": "hir.fused_matmul_bias_relu",
            "backend": "Metal",
            "fusion_candidate": "matmul_bias_relu",
            "inputs": ["A", "B", "bias"],
            "outputs": [result_name],
            "notes": [
                "Detected from MLIR linalg.matmul annotated by MatMulBiasReluFusionPass",
                "Mapped to the existing heterogeneous runtime planner as a fused accelerator candidate",
            ],
        })

    return {
        "format": "hir.lowered_graph.v1",
        "source": str(source_path),
        "num_ops": len(ops),
        "ops": ops,
    }


def build_execution_plan(lowered_graph):
    steps = []

    for op in lowered_graph["ops"]:
        steps.append({
            "step": op["id"],
            "op_name": op["name"],
            "op_type": op["op_type"],
            "lowered_op_type": op["lowered_op_type"],
            "backend": op["backend"],
            "fusion_candidate": op["fusion_candidate"],
            "runtime_action": "dispatch_fused_kernel",
            "estimated_launch_overhead_us": 80,
        })

    return {
        "format": "hir.execution_plan.v1",
        "source": lowered_graph["source"],
        "num_steps": len(steps),
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="trace/mlir_fused_graph.mlir")
    parser.add_argument("--lowered-output", default="trace/mlir_lowered_graph.json")
    parser.add_argument("--plan-output", default="trace/mlir_execution_plan.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    text = input_path.read_text(encoding="utf-8")

    matches = detect_fused_matmul(text)

    if not matches:
        raise SystemExit(
            "No fused MatMul-Bias-ReLU annotation found. "
            "Expected fusion.candidate = \"matmul_bias_relu\"."
        )

    lowered_graph = build_lowered_graph(matches, input_path)
    execution_plan = build_execution_plan(lowered_graph)

    lowered_output = Path(args.lowered_output)
    plan_output = Path(args.plan_output)

    lowered_output.parent.mkdir(parents=True, exist_ok=True)
    plan_output.parent.mkdir(parents=True, exist_ok=True)

    lowered_output.write_text(json.dumps(lowered_graph, indent=2) + "\n", encoding="utf-8")
    plan_output.write_text(json.dumps(execution_plan, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {lowered_output}")
    print(f"Wrote {plan_output}")


if __name__ == "__main__":
    main()