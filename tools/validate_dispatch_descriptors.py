#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def validate_plan(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    checked = 0
    for step in payload.get("steps", []):
        descriptor = step.get("dispatch_descriptor")
        if not descriptor:
            continue
        checked += 1
        if descriptor.get("descriptor_type") != "target.dispatch_descriptor.v1":
            raise SystemExit(f"{path}: invalid descriptor type")
        target = step.get("target_model") or {}
        decision = descriptor.get("tile_decision") or {}
        if decision.get("status") != "selected":
            raise SystemExit(f"{path}: expected selected tile decision")
        tile = decision.get("selected_tile") or {}
        shape = descriptor.get("shape") or {}
        if not tile:
            raise SystemExit(f"{path}: missing selected tile")
        padding = decision.get("padding") or {}
        if decision.get("requires_padding_crop"):
            padded = padding.get("padded_shape") or {}
            for dim in ("m", "n", "k"):
                if padded.get(dim, 0) % tile.get(dim, 1) != 0:
                    raise SystemExit(f"{path}: selected tile does not divide padded shape {dim}")
            if padding.get("padding_compute_overhead_ratio", 99.0) > 1.25:
                raise SystemExit(f"{path}: padded tile exceeds compute overhead threshold")
            if padding.get("padding_output_overhead_ratio", 99.0) > 1.25:
                raise SystemExit(f"{path}: padded tile exceeds output overhead threshold")
        else:
            for dim in ("m", "n", "k"):
                if shape.get(dim) % tile.get(dim, 1) != 0:
                    raise SystemExit(f"{path}: selected tile does not divide shape {dim}")
        if decision.get("selected_sram_bytes", 0) > target.get("sram_kb", 0) * 1024:
            raise SystemExit(f"{path}: selected tile exceeds SRAM budget")
        vector_bytes = descriptor.get("vector_bytes", 1)
        dtype_bytes = {"f32": 4, "f16": 2, "i8": 1}.get(descriptor.get("dtype"), 4)
        if (tile.get("k", 0) * dtype_bytes) % vector_bytes != 0:
            raise SystemExit(f"{path}: selected K tile is not vector aligned")
        candidates = decision.get("candidates", [])
        if not candidates:
            raise SystemExit(f"{path}: tile decision has no candidates")
        if not any(not candidate.get("legal") for candidate in candidates):
            raise SystemExit(f"{path}: expected at least one rejected tile candidate")
    if checked == 0:
        raise SystemExit(f"{path}: no dispatch descriptors found")
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        action="append",
        default=["trace/mlir_execution_plan.json", "trace/qmatmul_execution_plan.json"],
    )
    args = parser.parse_args()

    total = 0
    for plan in args.plan:
        total += validate_plan(plan)
    print(f"validated {total} target dispatch descriptor(s)")


if __name__ == "__main__":
    main()
