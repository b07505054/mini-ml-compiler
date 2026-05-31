#!/usr/bin/env python3
import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "ml_graph_compiler_runtime"))

from generate_llm_artifacts import generate_artifacts  # noqa: E402


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def to_graph_operator(op):
    if op.startswith("attention_"):
        return "attention"
    if op == "embed":
        return None
    return op


def analysis_to_config(base_config, analysis):
    config = copy.deepcopy(base_config)

    model = analysis["model"]
    config["model"]["name"] = model["model"]
    config["model"]["num_layers"] = model["num_layers"]
    config["model"]["hidden_size"] = model["hidden_size"]
    config["model"]["num_heads"] = model["num_heads"]
    config["model"]["intermediate_size"] = model["intermediate_size"]
    config["model"]["vocab_size"] = model["vocab_size"]

    graph_ops = []
    for op in analysis["phase_partition"]["prefill_ops"]:
        graph_op = to_graph_operator(op)
        if graph_op and graph_op not in graph_ops:
            graph_ops.append(graph_op)

    for op in analysis["phase_partition"]["decode_ops"]:
        graph_op = to_graph_operator(op)
        if graph_op and graph_op not in graph_ops:
            graph_ops.append(graph_op)

    config["model"]["operators"] = graph_ops

    config["execution"]["phases"] = [
        {
            "name": "prefill",
            "ops": analysis["phase_partition"]["prefill_ops"],
            "preferred_backend": "gpu",
            "source_analysis": "trace/llm_serving_compiler_analysis.json",
        },
        {
            "name": "decode",
            "ops": analysis["phase_partition"]["decode_ops"],
            "preferred_backend": "cpu_or_gpu",
            "source_analysis": "trace/llm_serving_compiler_analysis.json",
        },
    ]

    if analysis["kv_cache_analysis"]["required"]:
        config["kv_cache"]["paged_attention_enabled"] = True
        config["kv_cache"]["block_table_enabled"] = True

    if analysis["runtime_constraints"]["continuous_batching_supported"]:
        config["scheduling"]["scheduler"] = "continuous_batching"

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    analysis = load_json(Path(args.analysis))
    base_config = load_json(Path(args.config))

    config = analysis_to_config(base_config, analysis)
    written = generate_artifacts(config, Path(args.out))

    for filename in written:
        print(filename)


if __name__ == "__main__":
    main()