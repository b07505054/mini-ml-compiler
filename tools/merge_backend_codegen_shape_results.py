#!/usr/bin/env python3
"""Merge per-shape JSON output from aarch64_matmul_bias_relu_harness (run
once per shape, in its own process -- see the harness header comment) into
one combined benchmark_results.json, plus a smaller correctness_results.json
extract.

Usage:
  merge_backend_codegen_shape_results.py <shape1.json> <shape2.json> ... \
      --out benchmark_results.json --correctness-out correctness_results.json
"""
import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--correctness-out", required=True)
    args = parser.parse_args()

    shapes = []
    for path in args.inputs:
        with open(path) as f:
            doc = json.load(f)
        if len(doc["shapes"]) != 1:
            print(f"error: expected exactly one shape in {path}, got {len(doc['shapes'])}",
                  file=sys.stderr)
            return 1
        shapes.append(doc["shapes"][0])

    all_correct = all(s["correct"] for s in shapes)
    merged = {"shapes": shapes, "all_correct": all_correct}
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")

    correctness = {
        "shapes": [
            {
                "shape": s["shape"],
                "correct": s["correct"],
                "max_abs_error": s["max_abs_error"],
                "reference_checksum": s["reference_checksum"],
                "generated_checksum": s["generated_checksum"],
            }
            for s in shapes
        ],
        "all_correct": all_correct,
    }
    with open(args.correctness_out, "w") as f:
        json.dump(correctness, f, indent=2)
        f.write("\n")

    print(f"wrote {args.out} and {args.correctness_out}; all_correct={all_correct}")
    return 0 if all_correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
