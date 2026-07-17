#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_SHAPES = {
    "matmul_bias_relu": {"m": 128, "k": 128, "n": 128, "dtype": "f32"},
    "matmul_bias_relu_prefetch": {"m": 256, "k": 256, "n": 256, "dtype": "f32"},
    "qmatmul_bias_relu": {"m": 128, "k": 128, "n": 128, "dtype": "i8"},
}


def shape_bucket(fusion_candidate, row):
    shape = row.get("representative_shape") or row.get("shape") or DEFAULT_SHAPES.get(fusion_candidate, {})
    dtype = shape.get("dtype", "f32")
    if dtype == "float32":
        dtype = "f32"
    if fusion_candidate == "rmsnorm":
        return f"{shape.get('tokens', '?')}x{shape.get('hidden', '?')}:{dtype}"
    if fusion_candidate in {"matmul_bias_relu", "matmul_bias_relu_prefetch", "qmatmul_bias_relu"}:
        return f"{shape.get('m', '?')}x{shape.get('k', '?')}x{shape.get('n', '?')}:{dtype}"
    return f"default:{dtype}"


def backend_for(fusion_candidate, row):
    if row.get("backend"):
        return str(row["backend"]).lower()
    if row.get("custom_backend"):
        return row["custom_backend"]
    custom_kernel = row.get("custom_kernel", "")
    lowered = custom_kernel.lower()
    for backend in ("triton", "cuda", "metal", "torch", "cpu"):
        if backend in lowered:
            return backend
    return "cpu"


def normalize_dtype(value):
    return {"float32": "fp32", "f32": "fp32", "float16": "fp16", "f16": "fp16"}.get(value, value)


def exact_row(row, source, artifact_hash, environment):
    shape = row.get("shape", {})
    tokens = row.get("tokens", shape.get("tokens"))
    hidden = row.get("hidden", shape.get("hidden"))
    dtype = normalize_dtype(row.get("dtype", shape.get("dtype")))
    return {
        "candidate_id": row.get("candidate_id"),
        "operator": row.get("operator", row.get("fusion_candidate")),
        "semantics": row.get("semantics"),
        "backend": backend_for(row.get("fusion_candidate"), row),
        "kernel_family": row.get("kernel_family"),
        "kernel_entry_point": row.get("kernel_entry_point"),
        "dtype": dtype,
        "tokens": tokens,
        "hidden": hidden,
        "epsilon": row.get("epsilon"),
        "block_size": row.get("block_size"),
        "num_warps": row.get("num_warps"),
        "num_stages": row.get("num_stages"),
        "source_hash": row.get("source_hash"),
        "artifact_hash": artifact_hash,
        "target": {
            "gpu_name": environment.get("gpu_name"),
            "compute_capability": environment.get("compute_capability"),
        },
        "mean_ms": row.get("custom_latency_ms", row.get("mean_ms")),
        "p50_ms": row.get("custom_p50_ms", row.get("p50_ms")),
        "p95_ms": row.get("custom_p95_ms", row.get("p95_ms")),
        "effective_bandwidth_gbps": row.get("custom_effective_bandwidth_gbps", row.get("effective_bandwidth_gbps")),
        "correct": row.get("correct"),
        "selection_ready": row.get("selection_ready"),
        "failure_reason": row.get("failure_reason"),
        "source_artifact": source,
        "measurement_kind": "measured" if row.get("custom_p50_ms") is not None else "unavailable",
    }


def add_row(cost_table, row, source):
    fusion_candidate = row.get("fusion_candidate")
    if not fusion_candidate:
        return

    backend = backend_for(fusion_candidate, row)
    bucket = shape_bucket(fusion_candidate, row)
    custom_ms = row.get("custom_latency_ms")
    fallback_ms = row.get("fallback_latency_ms")
    custom_wins = isinstance(custom_ms, (int, float)) and isinstance(fallback_ms, (int, float)) and custom_ms < fallback_ms

    cost_table.setdefault(fusion_candidate, {}).setdefault(backend, {})[bucket] = {
        "custom_kernel": row.get("custom_kernel"),
        "fallback_kernel": row.get("fallback_kernel"),
        "custom_ms": custom_ms,
        "fallback_ms": fallback_ms,
        "speedup": row.get("speedup"),
        "correct": row.get("correct"),
        "selection_ready": row.get("selection_ready"),
        "winner": row.get("custom_kernel") if custom_wins else row.get("fallback_kernel"),
        "source": source,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--output", default="trace/profile_calibrated_cost_table.json")
    args = parser.parse_args()

    cost_table = {}
    exact_candidates = []
    sources = []
    for profile in args.profile:
        profile_path = Path(profile)
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        artifact_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        sources.append(str(profile_path))
        for row in payload.get("kernel_benchmarks", []):
            add_row(cost_table, row, str(profile_path))
        for row in payload.get("sweep", []):
            add_row(cost_table, row, str(profile_path))
        environment = payload.get("environment", {})
        for row in payload.get("exact_candidates", []):
            normalized = exact_row(row, str(profile_path), artifact_hash, environment)
            if normalized["candidate_id"]:
                exact_candidates.append(normalized)

    output = {
        "artifact_type": "profile_calibrated_cost_table",
        "format": "hir.profile_cost_table.v2",
        "sources": sources,
        "cost_table": cost_table,
        "exact_candidates": exact_candidates,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
