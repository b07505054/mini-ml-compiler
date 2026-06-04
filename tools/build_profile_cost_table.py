#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


DEFAULT_SHAPES = {
    "matmul_bias_relu": {"m": 128, "k": 128, "n": 128, "dtype": "f32"},
    "qmatmul_bias_relu": {"m": 128, "k": 128, "n": 128, "dtype": "i8"},
}


def shape_bucket(fusion_candidate, row):
    shape = row.get("representative_shape") or row.get("shape") or DEFAULT_SHAPES.get(fusion_candidate, {})
    dtype = shape.get("dtype", "f32")
    if dtype == "float32":
        dtype = "f32"
    if fusion_candidate == "rmsnorm":
        return f"{shape.get('tokens', '?')}x{shape.get('hidden', '?')}:{dtype}"
    if fusion_candidate in {"matmul_bias_relu", "qmatmul_bias_relu"}:
        return f"{shape.get('m', '?')}x{shape.get('k', '?')}x{shape.get('n', '?')}:{dtype}"
    return f"default:{dtype}"


def backend_for(fusion_candidate, row):
    if row.get("custom_backend"):
        return row["custom_backend"]
    custom_kernel = row.get("custom_kernel", "")
    if "cuda" in custom_kernel:
        return "CUDA"
    if "metal" in custom_kernel:
        return "Metal"
    return "CPU"


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
    sources = []
    for profile in args.profile:
        profile_path = Path(profile)
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        sources.append(str(profile_path))
        for row in payload.get("kernel_benchmarks", []):
            add_row(cost_table, row, str(profile_path))
        for row in payload.get("sweep", []):
            add_row(cost_table, row, str(profile_path))

    output = {
        "artifact_type": "profile_calibrated_cost_table",
        "format": "hir.profile_cost_table.v1",
        "sources": sources,
        "cost_table": cost_table,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
