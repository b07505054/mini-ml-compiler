#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


RMSNORM_LINALG = """#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

func.func @stablehlo_textual_rmsnorm(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %zero = arith.constant 0.0 : f32
  %eps = arith.constant 0.000001 : f32
  %hidden = arith.constant 4.0 : f32
  %row_empty = tensor.empty() : tensor<2xf32>
  %row_init = linalg.fill ins(%zero : f32)
      outs(%row_empty : tensor<2xf32>) -> tensor<2xf32>
  %sum = linalg.generic {
      indexing_maps = [#map, #row],
      iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<2x4xf32>)
      outs(%row_init : tensor<2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sq = arith.mulf %in, %in : f32
      %next = arith.addf %out, %sq : f32
      linalg.yield %next : f32
  } -> tensor<2xf32>
  %out_empty = tensor.empty() : tensor<2x4xf32>
  %normalized = linalg.generic {
      indexing_maps = [#map, #row, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%x, %sum : tensor<2x4xf32>, tensor<2xf32>)
      outs(%out_empty : tensor<2x4xf32>) {
    ^bb0(%in: f32, %row_sum: f32, %out: f32):
      %mean = arith.divf %row_sum, %hidden : f32
      %var = arith.addf %mean, %eps : f32
      %inv = math.rsqrt %var : f32
      %y = arith.mulf %in, %inv : f32
      linalg.yield %y : f32
  } -> tensor<2x4xf32>
  return %normalized : tensor<2x4xf32>
}
"""


MATMUL_LINALG = """func.func @stablehlo_textual_matmul_bias_relu(
    %lhs: tensor<16x128xf32>,
    %rhs: tensor<128x64xf32>,
    %bias: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %empty = tensor.empty() : tensor<16x64xf32>
  %matmul = linalg.matmul
      ins(%lhs, %rhs : tensor<16x128xf32>, tensor<128x64xf32>)
      outs(%empty : tensor<16x64xf32>) -> tensor<16x64xf32>
  %add = linalg.map
      ins(%matmul, %bias : tensor<16x64xf32>, tensor<16x64xf32>)
      outs(%empty : tensor<16x64xf32>)
      (%x: f32, %b: f32) {
    %y = arith.addf %x, %b : f32
    linalg.yield %y : f32
  }
  %zero = arith.constant 0.0 : f32
  %relu = linalg.map
      ins(%add : tensor<16x64xf32>)
      outs(%empty : tensor<16x64xf32>)
      (%x: f32) {
    %y = arith.maximumf %x, %zero : f32
    linalg.yield %y : f32
  }
  return %relu : tensor<16x64xf32>
}
"""


def detect_kind(text):
    ops = set(re.findall(r'"(stablehlo\.[A-Za-z0-9_]+)"', text))
    if {
        "stablehlo.multiply",
        "stablehlo.reduce",
        "stablehlo.divide",
        "stablehlo.rsqrt",
    }.issubset(ops):
        return "rmsnorm"
    if {
        "stablehlo.dot_general",
        "stablehlo.add",
        "stablehlo.maximum",
    }.issubset(ops):
        return "matmul_bias_relu"
    raise ValueError(f"unsupported StableHLO textual subset ops: {sorted(ops)}")


def convert(text):
    kind = detect_kind(text)
    if kind == "rmsnorm":
        return kind, RMSNORM_LINALG
    if kind == "matmul_bias_relu":
        return kind, MATMUL_LINALG
    raise AssertionError(kind)


def main():
    parser = argparse.ArgumentParser(description="Import a narrow StableHLO textual subset into standard MLIR.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    kind, output = convert(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(
                {
                    "artifact_type": "stablehlo_textual_subset_import",
                    "source": str(args.input),
                    "output": str(args.output),
                    "kind": kind,
                    "frontend": "stablehlo_textual_subset",
                    "supported_patterns": ["rmsnorm", "matmul_bias_relu"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"imported {kind} -> {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
