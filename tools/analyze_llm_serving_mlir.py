#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


LLM_OPS = [
    "llm.embed",
    "llm.rmsnorm",
    "llm.qkv_projection",
    "llm.attention_prefill",
    "llm.attention_decode",
    "llm.mlp",
]


INT_ATTRS = [
    "llm.num_layers",
    "llm.hidden_size",
    "llm.num_heads",
    "llm.intermediate_size",
    "llm.vocab_size",
]


def extract_model_attrs(text):
    attrs = {}

    model = re.search(r'llm\.model\s*=\s*"([^"]+)"', text)
    if model:
        attrs["model"] = model.group(1)

    for attr in INT_ATTRS:
        match = re.search(rf"{re.escape(attr)}\s*=\s*(\d+)\s*:\s*i64", text)
        if match:
            attrs[attr.replace("llm.", "")] = int(match.group(1))

    return attrs


def extract_ops(text):
    ops = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        for op in LLM_OPS:
            if f'"{op}"' in line:
                ops.append({
                    "op": op,
                    "line": line_no,
                    "serving_phase": detect_phase(line),
                    "kv_cache_role": detect_kv_role(line),
                })

    return ops


def detect_phase(line):
    if "attention_prefill" in line or 'serving.phase = "prefill"' in line:
        return "prefill"
    if "attention_decode" in line or 'serving.phase = "decode"' in line:
        return "decode"
    return "shared"


def detect_kv_role(line):
    if 'kv_cache.role = "producer"' in line:
        return "producer"
    if 'kv_cache.role = "consumer"' in line:
        return "consumer"
    if "attention_prefill" in line:
        return "producer"
    if "attention_decode" in line:
        return "consumer"
    return "none"


def build_phase_partition(ops):
    prefill = []
    decode = []
    shared = []

    for item in ops:
        name = item["op"].replace("llm.", "")

        if item["serving_phase"] == "prefill":
            prefill.append(name)
        elif item["serving_phase"] == "decode":
            decode.append(name)
        else:
            shared.append(name)

    return {
        "shared_ops": unique(shared),
        "prefill_ops": unique(shared + prefill),
        "decode_ops": unique(shared + decode + ["kv_cache_read"]),
    }


def unique(values):
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def build_analysis(source, text):
    attrs = extract_model_attrs(text)
    ops = extract_ops(text)
    phases = build_phase_partition(ops)

    has_prefill = any(op["op"] == "llm.attention_prefill" for op in ops)
    has_decode = any(op["op"] == "llm.attention_decode" for op in ops)

    return {
        "artifact_type": "llm_serving_compiler_analysis",
        "source": str(source),
        "model": attrs,
        "ops": ops,
        "phase_partition": phases,
        "kv_cache_analysis": {
            "required": has_prefill and has_decode,
            "producer_ops": [
                op for op in ops if op["kv_cache_role"] == "producer"
            ],
            "consumer_ops": [
                op for op in ops if op["kv_cache_role"] == "consumer"
            ],
        },
        "runtime_constraints": {
            "prefill_is_compute_heavy": has_prefill,
            "decode_is_token_step": has_decode,
            "decode_requires_kv_cache_read": has_decode,
            "continuous_batching_supported": has_prefill and has_decode,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir", required=True)
    parser.add_argument("--out", default="trace/llm_serving_compiler_analysis.json")
    args = parser.parse_args()

    mlir_path = Path(args.mlir)
    text = mlir_path.read_text(encoding="utf-8")

    analysis = build_analysis(mlir_path, text)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    print(out)


if __name__ == "__main__":
    main()