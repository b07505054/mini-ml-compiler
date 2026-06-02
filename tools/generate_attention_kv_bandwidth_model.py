#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def estimate_decode_attention(batch, heads, context, head_dim, dtype_bytes, block_size):
    kv_bytes_per_token = 2 * heads * context * head_dim * dtype_bytes
    q_bytes_per_token = heads * head_dim * dtype_bytes
    output_bytes_per_token = heads * head_dim * dtype_bytes
    score_flops_per_token = 2 * heads * context * head_dim
    value_flops_per_token = 2 * heads * context * head_dim
    total_flops_per_token = score_flops_per_token + value_flops_per_token
    total_bytes_per_token = kv_bytes_per_token + q_bytes_per_token + output_bytes_per_token
    blocks_per_sequence = (context + block_size - 1) // block_size
    return {
        "batch": batch,
        "heads": heads,
        "context": context,
        "head_dim": head_dim,
        "dtype_bytes": dtype_bytes,
        "block_size": block_size,
        "blocks_per_sequence": blocks_per_sequence,
        "kv_bytes_per_token": kv_bytes_per_token,
        "q_bytes_per_token": q_bytes_per_token,
        "output_bytes_per_token": output_bytes_per_token,
        "total_bytes_per_token": total_bytes_per_token,
        "score_flops_per_token": score_flops_per_token,
        "value_flops_per_token": value_flops_per_token,
        "total_flops_per_token": total_flops_per_token,
        "arithmetic_intensity_flops_per_byte": total_flops_per_token / max(total_bytes_per_token, 1),
        "bottleneck": "kv_cache_memory_bandwidth",
        "layout_notes": [
            "Decode attention has a small query but streams K and V across the context window.",
            "Paged KV layout trades contiguous reads for allocation/reuse flexibility.",
            "Tile/block size controls locality, page-table pressure, and wasted reads on partial blocks.",
            "A FlashAttention-inspired implementation reduces intermediate score materialization but still must read KV bytes.",
        ],
    }


def write_markdown(path, report):
    rows = report["sweep"]
    lines = [
        "# Decode Attention KV-Cache Bandwidth Model",
        "",
        "This model is a GEMM-adjacent attention case study for serving-time decode.",
        "It focuses on KV-cache read bandwidth rather than implementing a full FlashAttention kernel.",
        "",
        "| Context | Heads | Head dim | KV bytes/token | FLOPs/token | AI FLOPs/byte | Blocks | Bottleneck |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {context} | {heads} | {head_dim} | {kv} | {flops} | {ai:.3f} | {blocks} | {bottleneck} |".format(
                context=row["context"],
                heads=row["heads"],
                head_dim=row["head_dim"],
                kv=row["kv_bytes_per_token"],
                flops=row["total_flops_per_token"],
                ai=row["arithmetic_intensity_flops_per_byte"],
                blocks=row["blocks_per_sequence"],
                bottleneck=row["bottleneck"],
            )
        )
    lines.extend([
        "",
        "## Compiler/Runtime Relevance",
        "",
        "- Compiler lowering can select layouts and tile sizes that preserve coalesced KV reads.",
        "- Runtime scheduling must account for KV-cache memory pressure during decode.",
        "- Paged KV improves reuse/admission behavior but introduces block lookup and partial-block overhead.",
        "- This model explains why attention decode is often bandwidth-bound even when compute kernels are optimized.",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="trace/attention_kv_bandwidth_model.json")
    parser.add_argument("--markdown-output", default="trace/attention_kv_bandwidth_model.md")
    parser.add_argument("--contexts", default="128,512,2048,4096")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    sweep = [
        estimate_decode_attention(
            args.batch,
            args.heads,
            int(context),
            args.head_dim,
            args.dtype_bytes,
            args.block_size,
        )
        for context in args.contexts.split(",")
        if context
    ]
    report = {
        "artifact_type": "attention_kv_bandwidth_model",
        "format": "hir.attention_kv_bandwidth.v1",
        "status": "modeled",
        "sweep": sweep,
    }

    output = Path(args.output)
    markdown_output = Path(args.markdown_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(markdown_output, report)
    print(f"Wrote {output}")
    print(f"Wrote {markdown_output}")


if __name__ == "__main__":
    main()
