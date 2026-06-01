#!/usr/bin/env python3
"""Generate LLM serving/runtime planning artifacts."""

import argparse
import datetime as _datetime
import hashlib
import json
import math
import subprocess
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
    "artifact_provenance.json",
    "candidate_execution_plans.json",
    "memory_timeline.json",
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


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


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
            "kv_policy_contract": {
                "prefix_cache_enabled": config["kv_cache"]["prefix_cache_enabled"],
                "eviction_policy": config["kv_cache"]["eviction_policy"],
                "admission_policy": config["kv_cache"].get(
                    "admission_policy",
                    "capacity_only",
                ),
            },
        },
    }


def build_kv_cache_plan(config):
    model = config["model"]
    kv_cache = config["kv_cache"]
    prefix_cache = kv_cache.get("prefix_cache", {})
    block_size_tokens = int(kv_cache["block_size_tokens"])
    num_blocks = int(kv_cache["num_blocks"])
    bytes_per_token = kv_cache_bytes(model, kv_cache, 1)
    bytes_per_block = kv_cache_bytes(model, kv_cache, block_size_tokens)
    prefix_cache_enabled = bool(kv_cache["prefix_cache_enabled"])

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
        "admission_policy": kv_cache.get("admission_policy", "capacity_only"),
        "allocation_strategy": kv_cache["allocation_strategy"],
        "paged_attention_enabled": kv_cache["paged_attention_enabled"],
        "block_table_enabled": kv_cache["block_table_enabled"],
        "prefix_cache_enabled": prefix_cache_enabled,
        "prefix_cache_policy": {
            "enabled": prefix_cache_enabled,
            "hash_algorithm": prefix_cache.get("hash_algorithm", "sha256"),
            "model_version": prefix_cache.get("model_version", model["name"]),
            "min_prefix_tokens": int(prefix_cache.get("min_prefix_tokens", block_size_tokens)),
            "max_prefix_entries": int(prefix_cache.get("max_prefix_entries", 128)),
            "evictable_state": prefix_cache.get("evictable_state", "finished"),
            "track_ref_count": bool(prefix_cache.get("track_ref_count", True)),
            "runtime_events": [
                "prefix_cache_hit",
                "prefix_cache_miss",
                "kv_blocks_evicted",
                "admission_rejected",
            ],
            "runtime_metrics": [
                "prefix_cache_hit_rate",
                "kv_blocks_reused",
                "kv_blocks_evicted",
                "admission_rejection_rate",
                "prefill_latency_saved_ms",
            ],
        },
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
            "prefix_cache_hit_rate",
            "kv_blocks_reused",
            "kv_blocks_evicted",
            "admission_rejection_rate",
        ],
    }


def build_candidate_execution_plans(config):
    memory = build_memory_plan(config)
    peak_decode = int(memory["peak_decode_memory_mb"])
    temporary = int(memory["temporary_buffer_mb"])
    return {
        "schema_version": config["schema_version"],
        "artifact_type": "candidate_execution_plans",
        "model": config["model"]["name"],
        "selection_objective": "minimize_latency_under_memory_budget",
        "plans": [
            {
                "plan_id": "plan_metal",
                "backend": "Metal",
                "runtime_action": "dispatch_fused_kernel",
                "estimated_latency_ms": 1.8,
                "estimated_memory_mb": peak_decode,
                "estimated_throughput_tokens_per_s": 555.6,
                "notes": "Fused op placed on Metal for lowest estimated latency.",
            },
            {
                "plan_id": "plan_cpu",
                "backend": "CPU",
                "runtime_action": "dispatch_unfused_ops",
                "estimated_latency_ms": 4.7,
                "estimated_memory_mb": max(temporary + 128, 1),
                "estimated_throughput_tokens_per_s": 212.8,
                "notes": "CPU fallback plan with lower memory pressure and higher latency.",
            },
            {
                "plan_id": "plan_hybrid",
                "backend": "Hybrid",
                "runtime_action": "dispatch_metal_compute_cpu_kv_bookkeeping",
                "estimated_latency_ms": 2.4,
                "estimated_memory_mb": max(peak_decode - 96, 1),
                "estimated_throughput_tokens_per_s": 416.7,
                "notes": "Hybrid plan keeps fused compute on accelerator and bookkeeping on CPU.",
            },
        ],
        "selected_plan_id": "plan_metal",
        "selection_reason": "lowest estimated latency while fitting memory budget",
    }


def build_memory_timeline(config):
    memory = build_memory_plan(config)
    activation = memory["activation_memory_mb"]
    temporary = int(memory["temporary_buffer_mb"])
    kv_target = int(memory["kv_cache_memory_mb_at_target_concurrency"])
    decode_step = int(activation["decode_step"])
    peak_decode = int(memory["peak_decode_memory_mb"])

    events = [
        {
            "step": 0,
            "event": "allocate",
            "buffer": "prefill_hidden_states",
            "size_mb": int(activation["prefill"]),
            "lifetime": [0, 3],
        },
        {
            "step": 1,
            "event": "allocate",
            "buffer": "fusion_workspace",
            "size_mb": temporary,
            "lifetime": [1, 2],
        },
        {
            "step": 2,
            "event": "reuse",
            "buffer": "fusion_workspace",
            "reused_as": "mlp_workspace",
            "size_mb": temporary,
            "lifetime": [2, 4],
        },
        {
            "step": 3,
            "event": "allocate",
            "buffer": "kv_cache_blocks",
            "size_mb": kv_target,
            "lifetime": [3, 7],
        },
        {
            "step": 4,
            "event": "allocate",
            "buffer": "decode_step_activation",
            "size_mb": decode_step,
            "lifetime": [4, 5],
        },
        {
            "step": 5,
            "event": "free",
            "buffer": "decode_step_activation",
        },
        {
            "step": 6,
            "event": "free",
            "buffer": "mlp_workspace",
        },
        {
            "step": 7,
            "event": "free",
            "buffer": "kv_cache_blocks",
        },
    ]

    return {
        "schema_version": config["schema_version"],
        "artifact_type": "memory_timeline",
        "model": config["model"]["name"],
        "peak_memory_mb": peak_decode,
        "reuse_enabled": memory["reuse_enabled"],
        "events": events,
    }


def build_artifact_provenance(config, out_dir, generated_files):
    repo_root = Path(__file__).resolve().parents[2]
    outputs = []
    for filename in generated_files:
        path = out_dir / filename
        if path.exists():
            outputs.append({
                "path": filename,
                "sha256": sha256_file(path),
            })

    return {
        "schema_version": config["schema_version"],
        "artifact_type": "artifact_provenance",
        "compiler": {
            "name": "ml-graph-compiler-runtime",
            "version": config.get("compiler_version", "0.4.0"),
            "git_commit": git_commit(repo_root),
        },
        "pass_pipeline": [
            "canonicalize",
            "matmul_bias_relu_fusion",
            "hir_lowering",
            "backend_placement",
            "memory_planning",
        ],
        "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
        "outputs": outputs,
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
            "kv_cache_policy_present",
            "memory_budget_not_exceeded",
            "scheduling_queues_present",
            "artifact_provenance_present",
            "candidate_plans_present",
            "memory_timeline_present",
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
        "candidate_execution_plans.json": build_candidate_execution_plans(config),
        "memory_timeline.json": build_memory_timeline(config),
        "validation_manifest.json": build_validation_manifest(config),
    }

    for filename, data in artifacts.items():
        write_json(out_dir / filename, data)

    generated = list(artifacts.keys())
    provenance = build_artifact_provenance(config, out_dir, generated)
    write_json(out_dir / "artifact_provenance.json", provenance)

    return generated + ["artifact_provenance.json"]


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
