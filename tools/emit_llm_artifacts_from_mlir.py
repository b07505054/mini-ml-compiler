#!/usr/bin/env python3
import argparse
import copy
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "ml_graph_compiler_runtime"))

from generate_llm_artifacts import generate_artifacts  # noqa: E402


INT_ATTRS = {
    "llm.num_layers": "num_layers",
    "llm.hidden_size": "hidden_size",
    "llm.num_heads": "num_heads",
    "llm.intermediate_size": "intermediate_size",
    "llm.vocab_size": "vocab_size",
}

OP_MAP = {
    "llm.embed": "embed",
    "llm.rmsnorm": "rmsnorm",
    "llm.qkv_projection": "qkv_projection",
    "llm.attention_prefill": "attention_prefill",
    "llm.attention_decode": "attention_decode",
    "llm.mlp": "mlp",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_int_attr(text, attr):
    match = re.search(rf"{re.escape(attr)}\s*=\s*(\d+)\s*:\s*i64", text)
    return int(match.group(1)) if match else None


def extract_ops(text):
    found = []
    for mlir_op, artifact_op in OP_MAP.items():
        if f'"{mlir_op}"' in text:
            found.append(artifact_op)

    graph_ops = []
    for op in found:
        if op == "embed":
            continue
        normalized = "attention" if op.startswith("attention_") else op
        if normalized not in graph_ops:
            graph_ops.append(normalized)

    return found, graph_ops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--analysis-out", default="trace/mlir_llm_serving_analysis.json")
    args = parser.parse_args()

    mlir_path = Path(args.mlir)
    text = mlir_path.read_text(encoding="utf-8")

    config = copy.deepcopy(load_json(Path(args.config)))

    model_match = re.search(r'llm\.model\s*=\s*"([^"]+)"', text)
    if model_match:
        config["model"]["name"] = model_match.group(1)

    for mlir_attr, config_key in INT_ATTRS.items():
        value = extract_int_attr(text, mlir_attr)
        if value is not None:
            config["model"][config_key] = value

    found_ops, graph_ops = extract_ops(text)
    config["model"]["operators"] = graph_ops

    config["execution"]["phases"] = [
        {
            "name": "prefill",
            "ops": ["embed", "attention_prefill", "mlp"],
            "preferred_backend": "gpu",
            "source": str(mlir_path),
        },
        {
            "name": "decode",
            "ops": ["attention_decode", "kv_cache_read", "mlp"],
            "preferred_backend": "cpu_or_gpu",
            "source": str(mlir_path),
        },
    ]

    written = generate_artifacts(config, Path(args.out))

    analysis = {
        "artifact_type": "mlir_llm_serving_analysis",
        "source": str(mlir_path),
        "detected_mlir_ops": found_ops,
        "graph_operators": graph_ops,
        "detected_phases": ["prefill", "decode"],
        "kv_cache_required": True,
    }

    analysis_out = Path(args.analysis_out)
    analysis_out.parent.mkdir(parents=True, exist_ok=True)
    analysis_out.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    for filename in written:
        print(filename)
    print(analysis_out)


if __name__ == "__main__":
    main()