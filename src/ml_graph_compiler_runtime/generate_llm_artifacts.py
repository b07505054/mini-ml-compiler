#!/usr/bin/env python3
"""Generate LLM serving/runtime planning artifacts."""

import argparse
import json
import math
from pathlib import Path


DTYPE_BYTES = {
    "fp32": 4,
    "float32": 4,
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "int8": 1,
    "fp8": 1,
}


ARTIFACT_FILES = [
    "llm_graph_ir.json",
    "serving_execution_plan.json",
    "kv_cache_plan.json",
    "memory_plan.json",
    "scheduling_plan.json",
    "validation_manifest.json",
]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def to_mb(num_bytes):
    return int(math.ceil(num_bytes / 1024 / 1024))


def dtype_bytes(dtype):
    key = dtype.lower()
    if key not in DTYPE_BYTES:
        supported = ", ".join(sorted(DTYPE_BYTES))
        raise ValueError(f"Unsupported dtype '{dtype}'. Supported values: {supported}")
    return DTYPE_BYTES[key]


def kv_cache_bytes(model, kv_cache, tokens):
    # KV cache stores both K and V for every layer.
    return (
        int(model["num_layers"])
        * 2
        * int(model["hidden_size"])
        * dtype_bytes(kv_cache["kv_dtype"])
        * int(tokens)
    )


def activation_memory_bytes(model, workload, tokens_per_request, safety_factor):
    batch_size = int(workload["batch_size"])
    hidden_size = int(model["hidden_size"])
    num_layers = int(model["num_layers"])
    bytes_per_value = 2

    return int(
        batch_size
        * int(tokens_per_request)
        * hidden_size
        * bytes_per_value
        * (num_layers + 2)
        * float(safety_factor)
    )


def build_llm_graph_ir(config):
    model = config["model"]
    return {
        "schema_version": config["schema_version"],
        "artifact_type": "llm_graph_ir",
        "model": model["name"],
        "num_layers": model["num_layers"],
        "hidden_size": model["hidden_size"],
        "num_heads": model["num_heads"],
        "intermediate_size": model["intermediate_size"],
        "vocab_size": model["vocab_size"],
        "operators": model["operators"],
        "request_workload": config["request_workload"],
    }


def build_serving_execution_plan(config):
    return {
        "schema_version": config["schema_version"],
        "artifact_type": "serving_execution_plan",
        "model": config["model"]["name"],
        "phases": config["execution"]["phases"],
        "runtime_contract": {
            "prefill_produces_kv_blocks": True,
            "decode_consumes_kv_blocks": True,
            "scheduler_plan": "scheduling_plan.json",
            "kv_cache_plan": "kv_cache_plan.json",
            "memory_plan": "memory_plan.json",
        },
    }


def build_kv_cache_plan(config):
    model = config["model"]
    kv_cache = config["kv_cache"]
    block_size_tokens = int(kv_cache["block_size_tokens"])
    num_blocks = int(kv_cache["num_blocks"])
    bytes_per_token = kv_cache_bytes(model, kv_cache, 1)
    bytes_per_block = kv_cache_bytes(model, kv_cache, block_size_tokens)

    return {
        "schema_version": config["schema_version"],
        "artifact_type": "kv_cache_plan",
        "model": model["name"],
        "block_size_tokens": block_size_tokens,
        "num_blocks": num_blocks,
        "max_context_tokens": kv_cache["max_context_tokens"],
        "total_token_capacity": block_size_tokens * num_blocks,
        "kv_dtype": kv_cache["kv_dtype"],
        "bytes_per_token": bytes_per_token,
        "bytes_per_block": bytes_per_block,
        "memory_mb_at_full_capacity": to_mb(bytes_per_block * num_blocks),
        "eviction_policy": kv_cache["eviction_policy"],
        "allocation_strategy": kv_cache["allocation_strategy"],
        "paged_attention_enabled": kv_cache["paged_attention_enabled"],
        "block_table_enabled": kv_cache["block_table_enabled"],
        "prefix_cache_enabled": kv_cache["prefix_cache_enabled"],
        "sliding_window_enabled": kv_cache["sliding_window_enabled"],
    }


def build_memory_plan(config):
    model = config["model"]
    workload = config["request_workload"]
    kv_cache = config["kv_cache"]
    memory = config["memory"]

    prefill_tokens = int(workload["prefill_tokens_per_request"])
    target_concurrent_requests = int(workload["target_concurrent_requests"])
    reference_tokens = int(memory["kv_cache_reference_tokens"])
    temporary_buffer_mb = int(memory["temporary_buffer_mb"])
    safety_factor = float(memory["activation_safety_factor"])

    prefill_activation_mb = to_mb(
        activation_memory_bytes(model, workload, prefill_tokens, safety_factor)
    )
    decode_activation_mb = to_mb(
        activation_memory_bytes(model, workload, 1, safety_factor)
    )
    kv_cache_reference_mb = to_mb(kv_cache_bytes(model, kv_cache, reference_tokens))
    kv_cache_target_mb = to_mb(
        kv_cache_bytes(model, kv_cache, target_concurrent_requests * prefill_tokens)
    )

    peak_prefill_memory_mb = prefill_activation_mb + temporary_buffer_mb
    peak_decode_memory_mb = decode_activation_mb + kv_cache_target_mb + temporary_buffer_mb

    return {
        "schema_version": config["schema_version"],
        "artifact_type": "memory_plan",
        "model": model["name"],
        "peak_prefill_memory_mb": peak_prefill_memory_mb,
        "peak_decode_memory_mb": peak_decode_memory_mb,
        f"kv_cache_memory_mb_at_{reference_tokens}_tokens": kv_cache_reference_mb,
        "kv_cache_memory_mb_at_target_concurrency": kv_cache_target_mb,
        "activation_memory_mb": {
            "prefill": prefill_activation_mb,
            "decode_step": decode_activation_mb,
        },
        "temporary_buffer_mb": temporary_buffer_mb,
        "reuse_enabled": memory["reuse_enabled"],
        "memory_budget_mb": memory["memory_budget_mb"],
        "fits_memory_budget": peak_decode_memory_mb <= int(memory["memory_budget_mb"]),
    }


def build_scheduling_plan(config):
    scheduling = config["scheduling"]
    workload = config["request_workload"]
    return {
        "schema_version": config["schema_version"],
        "artifact_type": "scheduling_plan",
        "scheduler": scheduling["scheduler"],
        "queues": scheduling["queues"],
        "max_batch_size": scheduling["max_batch_size"],
        "decode_step_tokens": scheduling["decode_step_tokens"],
        "preemption_enabled": scheduling["preemption_enabled"],
        "workload_shape": {
            "batch_size": workload["batch_size"],
            "prefill_tokens_per_request": workload["prefill_tokens_per_request"],
            "decode_tokens_per_request": workload["decode_tokens_per_request"],
            "target_concurrent_requests": workload["target_concurrent_requests"],
        },
        "dashboard_signals": [
            "prefill_queue_depth",
            "decode_queue_depth",
            "active_sequences",
            "tokens_scheduled_per_step",
            "kv_blocks_allocated",
        ],
    }


def build_validation_manifest(config):
    return {
        "schema_version": config["schema_version"],
        "artifact_type": "validation_manifest",
        "model": config["model"]["name"],
        "expected_outputs": ARTIFACT_FILES,
        "checks": [
            "artifact_schema_valid",
            "prefill_decode_phase_present",
            "kv_cache_capacity_valid",
            "kv_cache_block_size_positive",
            "memory_budget_not_exceeded",
            "scheduling_queues_present",
        ],
        "demo_contract": {
            "apple_demo_entry_artifacts": ARTIFACT_FILES,
            "primary_dashboard_views": [
                "model_graph",
                "prefill_decode_timeline",
                "kv_cache_blocks",
                "memory_breakdown",
                "scheduler_behavior",
                "validation_status",
            ],
        },
    }


def generate_artifacts(config, out_dir):
    artifacts = {
        "llm_graph_ir.json": build_llm_graph_ir(config),
        "serving_execution_plan.json": build_serving_execution_plan(config),
        "kv_cache_plan.json": build_kv_cache_plan(config),
        "memory_plan.json": build_memory_plan(config),
        "scheduling_plan.json": build_scheduling_plan(config),
        "validation_manifest.json": build_validation_manifest(config),
    }

    for filename, data in artifacts.items():
        write_json(out_dir / filename, data)

    return list(artifacts.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_json(Path(args.config))
    written = generate_artifacts(config, Path(args.out))

    for filename in written:
        print(filename)


if __name__ == "__main__":
    main()
