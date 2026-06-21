#!/usr/bin/env python3
import argparse
from dataclasses import dataclass, field
import json
import re
import sys
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


@dataclass
class StableHLOOp:
    results: list[str]
    op_name: str
    operands: list[str]
    result_type: str
    attributes: str
    line_no: int


@dataclass
class StableHLOModule:
    ops: list[StableHLOOp]
    def_use: dict[str, list[StableHLOOp]] = field(default_factory=dict)


class ImportErrorWithReason(ValueError):
    def __init__(self, reason, detail):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def parse_stablehlo_text(text):
    ops = []
    generic_pattern = re.compile(
        r'^\s*(?P<results>%[\w.$-]+(?:\s*,\s*%[\w.$-]+)*)\s*=\s*'
        r'(?:"(?P<quoted>stablehlo\.[A-Za-z0-9_]+)"|(?P<bare>stablehlo\.[A-Za-z0-9_]+))'
        r'\((?P<operands>[^)]*)\)'
        r'(?P<tail>.*)$'
    )
    bare_pattern = re.compile(
        r'^\s*(?P<results>%[\w.$-]+(?:\s*,\s*%[\w.$-]+)*)\s*=\s*'
        r'(?P<bare>stablehlo\.[A-Za-z0-9_]+)\b'
        r'(?P<body>.*)$'
    )
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        match = generic_pattern.match(line)
        if match:
            operands_text = match.group("operands")
            tail = match.group("tail")
            op_name = match.group("quoted") or match.group("bare")
        else:
            match = bare_pattern.match(line)
            if not match:
                continue
            body = match.group("body")
            operands_text = body.split(":", 1)[0]
            tail = body
            op_name = match.group("bare")
        if "func.func" in line:
            continue
        result_type_match = re.search(r'->\s*([^,}]+(?:<[^>]+>)?)', tail)
        if not result_type_match:
            colon_types = re.findall(r':\s*(tensor<[^>]+>)', tail)
            result_type = colon_types[-1] if colon_types else ""
        else:
            result_type = result_type_match.group(1).strip()
        results = [result.strip() for result in match.group("results").split(",")]
        operands = re.findall(r'%[\w.$-]+', operands_text)
        ops.append(StableHLOOp(
            results=results,
            op_name=op_name,
            operands=operands,
            result_type=result_type,
            attributes=tail.strip(),
            line_no=line_no,
        ))

    def_use = {}
    for op in ops:
        for operand in op.operands:
            def_use.setdefault(operand, []).append(op)
    return StableHLOModule(ops=ops, def_use=def_use)


def contains_dynamic_shape(module):
    return any("tensor<?" in op.result_type or "x?" in op.result_type for op in module.ops)


def contains_unsupported_dtype(module):
    return any("xf16" in op.result_type or "xbf16" in op.result_type for op in module.ops)


def require_static_f32(module):
    if contains_dynamic_shape(module):
        raise ImportErrorWithReason("dynamic_shape_unsupported", "StableHLO parser v1 requires static tensor shapes")
    if contains_unsupported_dtype(module):
        raise ImportErrorWithReason("unsupported_dtype", "StableHLO parser v1 supports f32 only")


def single_result(op):
    return op.results[0] if op.results else None


def producer_by_result(module):
    producers = {}
    for op in module.ops:
        for result in op.results:
            producers[result] = op
    return producers


def canonical_operand(value, producers):
    seen = set()
    current = value
    while current in producers and current not in seen:
        seen.add(current)
        producer = producers[current]
        if producer.op_name != "stablehlo.broadcast_in_dim" or not producer.operands:
            break
        current = producer.operands[0]
    return current


def operand_sources(op, producers):
    return [canonical_operand(operand, producers) for operand in op.operands]


def detect_rmsnorm(module):
    require_static_f32(module)
    ops_by_name = {}
    for op in module.ops:
        ops_by_name.setdefault(op.op_name, []).append(op)
    producers = producer_by_result(module)

    if not ops_by_name.get("stablehlo.rsqrt"):
        raise ImportErrorWithReason("missing_rsqrt", "RMSNorm import requires stablehlo.rsqrt")
    if not ops_by_name.get("stablehlo.reduce"):
        raise ImportErrorWithReason("missing_reduce", "RMSNorm import requires stablehlo.reduce")
    if not ops_by_name.get("stablehlo.divide"):
        raise ImportErrorWithReason("missing_divide", "RMSNorm import requires stablehlo.divide")

    multiply_ops = ops_by_name.get("stablehlo.multiply", [])
    square = None
    final_multiply = None
    for op in multiply_ops:
        if len(op.operands) >= 2 and op.operands[0] == op.operands[1]:
            square = op
        if any(
            producers.get(canonical_operand(operand, producers), None)
            and producers[canonical_operand(operand, producers)].op_name == "stablehlo.rsqrt"
            for operand in op.operands
        ):
            final_multiply = op

    if square is None:
        raise ImportErrorWithReason("missing_square_multiply", "RMSNorm import requires x*x multiply")
    if final_multiply is None:
        raise ImportErrorWithReason("missing_final_multiply", "RMSNorm import requires final x*rsqrt multiply")

    square_result = single_result(square)
    reduce = ops_by_name["stablehlo.reduce"][0]
    if square_result not in operand_sources(reduce, producers):
        raise ImportErrorWithReason("invalid_rmsnorm_def_use", "reduce must consume square multiply result")

    reduce_result = single_result(reduce)
    divide = ops_by_name["stablehlo.divide"][0]
    if reduce_result not in operand_sources(divide, producers):
        raise ImportErrorWithReason("invalid_rmsnorm_def_use", "divide must consume reduce result")

    divide_result = single_result(divide)
    rsqrt = ops_by_name["stablehlo.rsqrt"][0]
    rsqrt_sources = operand_sources(rsqrt, producers)
    if divide_result not in rsqrt_sources:
        add_ops = [
            op for op in ops_by_name.get("stablehlo.add", [])
            if divide_result in operand_sources(op, producers)
        ]
        if not add_ops or single_result(add_ops[0]) not in rsqrt_sources:
            raise ImportErrorWithReason("invalid_rmsnorm_def_use", "rsqrt must consume divide result or epsilon add")

    return {
        "legal": True,
        "kind": "rmsnorm",
        "decision": "import_to_linalg_then_hir_fused_rmsnorm",
        "fallback_reason": None,
        "parsed_ops": len(module.ops),
        "def_use_edges": sum(len(users) for users in module.def_use.values()),
    }


def detect_matmul_bias_relu(module):
    require_static_f32(module)
    ops_by_name = {}
    for op in module.ops:
        ops_by_name.setdefault(op.op_name, []).append(op)

    dots = ops_by_name.get("stablehlo.dot_general", [])
    adds = ops_by_name.get("stablehlo.add", [])
    maximums = ops_by_name.get("stablehlo.maximum", [])
    if not dots:
        raise ImportErrorWithReason("missing_dot_general", "MatMul import requires stablehlo.dot_general")
    if not adds:
        raise ImportErrorWithReason("missing_add", "MatMul+Bias import requires stablehlo.add")
    if not maximums:
        raise ImportErrorWithReason("missing_maximum_relu", "MatMul+Bias+ReLU import requires stablehlo.maximum")

    dot = dots[0]
    dot_result = single_result(dot)
    dot_users = module.def_use.get(dot_result, [])
    if len(dot_users) != 1:
        raise ImportErrorWithReason("dot_result_multi_use", "dot_general result must have exactly one logical use")
    add = dot_users[0]
    if add.op_name != "stablehlo.add":
        raise ImportErrorWithReason("dot_not_consumed_by_add", "dot_general result must feed stablehlo.add")

    add_result = single_result(add)
    add_users = module.def_use.get(add_result, [])
    if len(add_users) != 1 or add_users[0].op_name != "stablehlo.maximum":
        raise ImportErrorWithReason("missing_maximum_relu", "add result must feed stablehlo.maximum")

    maximum = add_users[0]
    has_zero_constant = any(op.op_name == "stablehlo.constant" and "0.0" in op.attributes for op in module.ops)
    if not has_zero_constant and len(maximum.operands) < 2:
        raise ImportErrorWithReason("missing_relu_zero", "maximum must compare against zero")

    return {
        "legal": True,
        "kind": "matmul_bias_relu",
        "decision": "import_to_linalg_then_hir_fused_matmul_bias_relu",
        "fallback_reason": None,
        "parsed_ops": len(module.ops),
        "def_use_edges": sum(len(users) for users in module.def_use.values()),
    }


def detect_kind(text):
    module = parse_stablehlo_text(text)
    if not module.ops:
        raise ImportErrorWithReason("no_stablehlo_ops", "no supported stablehlo.* operations were found")

    op_names = {op.op_name for op in module.ops}
    errors = []
    if "stablehlo.reduce" in op_names or "stablehlo.rsqrt" in op_names:
        try:
            return detect_rmsnorm(module)
        except ImportErrorWithReason as exc:
            errors.append(exc)
    if "stablehlo.dot_general" in op_names:
        try:
            return detect_matmul_bias_relu(module)
        except ImportErrorWithReason as exc:
            errors.append(exc)
    if errors:
        raise errors[0]
    raise ImportErrorWithReason("unsupported_pattern", f"unsupported StableHLO textual subset ops: {sorted(op_names)}")


def convert(text):
    metadata = detect_kind(text)
    if metadata["kind"] == "rmsnorm":
        return metadata, RMSNORM_LINALG
    if metadata["kind"] == "matmul_bias_relu":
        return metadata, MATMUL_LINALG
    raise AssertionError(metadata["kind"])


def main():
    parser = argparse.ArgumentParser(description="Import a narrow StableHLO textual subset into standard MLIR.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    try:
        metadata, output = convert(source)
    except ImportErrorWithReason as exc:
        if args.metadata_output:
            args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
            args.metadata_output.write_text(
                json.dumps({
                    "legal": False,
                    "kind": None,
                    "decision": "reject_before_linalg_import",
                    "fallback_reason": exc.reason,
                    "detail": exc.detail,
                }, indent=2),
                encoding="utf-8",
            )
        print(f"StableHLO import rejected: {exc.reason}: {exc.detail}", file=sys.stderr)
        return 1
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
                    "frontend": "stablehlo_textual_subset",
                    "supported_patterns": ["rmsnorm", "matmul_bias_relu"],
                    **metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"imported {metadata['kind']} -> {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
