#!/usr/bin/env python3
"""Check structural readiness for GenericGraphIR lowering to existing MLIR dialects."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import verify_graph_ir

TRUTH_BOUNDARY = (
    "lowering_contract_only_no_mlir_emission_no_domain_recognition_"
    "no_execution_plan_generation"
)

# These strategies use dialects already registered by project pass code.
# A strategy here is a contract decision, not an implemented MLIR emitter.
LOWERING_STRATEGIES: dict[str, dict[str, Any]] = {
    "nn.conv2d": {
        "target": "linalg.conv_2d_nchw_fchw or linalg.generic",
        "mode": "direct_or_decompose",
        "attrs": ["pads", "strides", "dilations", "groups", "kernel_shape"],
    },
    "nn.conv_transpose2d": {
        "target": "linalg.generic + arith",
        "mode": "direct_specialized",
        "attrs": [
            "pads", "strides", "dilations", "groups", "kernel_shape",
            "output_padding", "output_shape",
        ],
    },
    "nn.maxpool2d": {
        "target": "linalg.pooling_nchw_max or linalg.generic",
        "mode": "direct_or_decompose",
        "attrs": ["kernel_shape", "pads", "strides", "dilations", "ceil_mode"],
    },
    "nn.add": {"target": "linalg.generic + arith.addf/addi", "mode": "direct", "attrs": []},
    "nn.sub": {"target": "linalg.generic + arith.subf/subi", "mode": "direct", "attrs": []},
    "nn.mul": {"target": "linalg.generic + arith.mulf/muli", "mode": "direct", "attrs": []},
    "nn.div": {"target": "linalg.generic + arith.divf/divsi/divui", "mode": "direct", "attrs": []},
    "nn.matmul": {"target": "linalg.matmul/linalg.batch_matmul", "mode": "direct", "attrs": []},
    "nn.gemm": {
        "target": "linalg.matmul + linalg.generic + arith",
        "mode": "decompose",
        "attrs": ["alpha", "beta", "transA", "transB"],
    },
    "nn.reshape": {
        "target": "tensor.expand_shape/tensor.collapse_shape/tensor.reshape",
        "mode": "direct_or_decompose",
        "attrs": ["allowzero", "target_shape"],
    },
    "nn.transpose": {
        "target": "linalg.transpose",
        "mode": "direct",
        "attrs": ["perm"],
    },
    "nn.concat": {
        "target": "tensor.insert_slice",
        "mode": "decompose",
        "attrs": ["axis"],
    },
    "nn.resize": {
        "target": "tensor.generate + tensor.extract + arith",
        "mode": "direct_specialized",
        "attrs": ["mode", "coordinate_transformation_mode", "nearest_mode"],
    },
    "nn.softmax": {
        "target": "linalg.generic reductions + arith + math.exp",
        "mode": "decompose",
        "attrs": ["axis"],
    },
    "nn.sigmoid": {
        "target": "linalg.generic + arith + math.exp",
        "mode": "decompose",
        "attrs": [],
    },
    "nn.relu": {
        "target": "linalg.generic + arith.maximumf/maxsi/maxui",
        "mode": "direct",
        "attrs": [],
    },
    "nn.split": {
        "target": "tensor.extract_slice",
        "mode": "decompose",
        "attrs": ["axis"],
    },
    "nn.slice": {
        "target": "tensor.extract_slice",
        "mode": "direct_or_decompose",
        "attrs": ["starts", "ends"],
    },
    "nn.constant": {"target": "arith.constant", "mode": "direct", "attrs": []},
    "nn.identity": {"target": "SSA value forwarding", "mode": "direct", "attrs": []},
    "nn.unknown": {"target": None, "mode": "unsupported", "attrs": []},
}


def _strategy_limitation(op: str, attrs: Any) -> str | None:
    if not isinstance(attrs, dict):
        return "canonical_attrs are unavailable"
    if op == "nn.conv_transpose2d":
        if attrs.get("groups") != 1:
            return "only groups=1 is selected"
        if attrs.get("dilations") != [1, 1]:
            return "only unit dilation is selected"
        if attrs.get("pads") != [0, 0, 0, 0]:
            return "only zero padding is selected"
        if attrs.get("output_padding") != [0, 0]:
            return "only zero output_padding is selected"
        if attrs.get("kernel_shape") != attrs.get("strides"):
            return "only kernel_shape equal to strides is selected"
        if attrs.get("output_shape") not in (None, []):
            return "explicit output_shape is not selected"
    elif op == "nn.resize":
        if attrs.get("mode") != "nearest":
            return "only nearest mode is selected"
        if attrs.get("coordinate_transformation_mode") != "asymmetric":
            return "only asymmetric coordinate transformation is selected"
        if attrs.get("nearest_mode") != "floor":
            return "only floor nearest mode is selected"
        if attrs.get("scales") != [1.0, 1.0, 2.0, 2.0]:
            return "only static rank-4 2x spatial scale [1,1,2,2] is selected"
    return None


def _tensor_records(graph_ir: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for field in ("values", "initializers"):
        for record in graph_ir.get(field, []):
            if isinstance(record, dict) and isinstance(record.get("name"), str):
                records[record["name"]] = record
    return records


def _shape_missing(record: dict[str, Any] | None) -> bool:
    if not record or not isinstance(record.get("shape"), list):
        return True
    return any(
        not isinstance(dim, dict) or dim.get("kind") == "unknown"
        for dim in record["shape"]
    )


def _dtype_missing(record: dict[str, Any] | None) -> bool:
    return not record or record.get("dtype") in (None, "", "unknown")


def _node_summary(node: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    return {
        "id": node.get("id"),
        "name": node.get("name", ""),
        "op": node.get("op", ""),
        "source_node_id": node.get("source_node_id"),
        "source_op_type": node.get("source_op_type", ""),
        "source_name": node.get("source_name", ""),
        "reasons": reasons,
    }


def check_lowering_contract(graph_ir: dict[str, Any]) -> dict[str, Any]:
    verify_result = verify_graph_ir.verify_generic_graph_ir(graph_ir)
    base = {
        "contract_status": "invalid_generic_graph_ir",
        "supported_ops": [],
        "unsupported_ops": [],
        "blocking_nodes": [],
        "missing_required_attrs": [],
        "missing_shapes": [],
        "missing_dtypes": [],
        "preferred_mlir_targets": {},
        "verifier": {
            "passed": verify_result.passed,
            "errors": verify_result.errors,
        },
        "truth_boundary": TRUTH_BOUNDARY,
    }
    if not verify_result.passed:
        return base

    records = _tensor_records(graph_ir)
    present_supported: set[str] = set()
    present_unsupported: set[str] = set()
    target_counts: Counter[str] = Counter()
    blocking_nodes: list[dict[str, Any]] = []
    missing_attrs: list[dict[str, Any]] = []
    missing_shapes: list[dict[str, Any]] = []
    missing_dtypes: list[dict[str, Any]] = []

    for node in graph_ir.get("nodes", []):
        op = node.get("op", "")
        strategy = LOWERING_STRATEGIES.get(op)
        reasons: list[str] = []
        if strategy is None or strategy["mode"] == "unsupported":
            present_unsupported.add(op)
            reasons.append("no existing-MLIR lowering strategy selected")
        else:
            limitation = _strategy_limitation(op, node.get("canonical_attrs"))
            if limitation:
                present_unsupported.add(op)
                reasons.append(
                    "existing-MLIR strategy does not cover canonical semantics: "
                    + limitation
                )
            else:
                present_supported.add(op)
                target_counts[strategy["target"]] += 1

        attrs = node.get("canonical_attrs")
        required_attrs = strategy["attrs"] if strategy else []
        missing_for_node = [
            attr for attr in required_attrs
            if not isinstance(attrs, dict) or attr not in attrs
        ]
        if missing_for_node:
            missing_attrs.append({
                "node_id": node.get("id"),
                "node_name": node.get("name", ""),
                "op": op,
                "attributes": missing_for_node,
            })
            reasons.append("missing required canonical attrs: " + ", ".join(missing_for_node))

        if node.get("shape_inference_status") != "inferred":
            reasons.append(
                "shape inference status is "
                + repr(node.get("shape_inference_status", "missing"))
            )

        tensor_names = [
            name for name in node.get("inputs", []) + node.get("outputs", [])
            if isinstance(name, str) and name
        ]
        node_missing_shapes = sorted({name for name in tensor_names if _shape_missing(records.get(name))})
        node_missing_dtypes = sorted({name for name in tensor_names if _dtype_missing(records.get(name))})
        if node_missing_shapes:
            missing_shapes.append({
                "node_id": node.get("id"),
                "node_name": node.get("name", ""),
                "op": op,
                "values": node_missing_shapes,
            })
            reasons.append("missing required shapes: " + ", ".join(node_missing_shapes))
        if node_missing_dtypes:
            missing_dtypes.append({
                "node_id": node.get("id"),
                "node_name": node.get("name", ""),
                "op": op,
                "values": node_missing_dtypes,
            })
            reasons.append("missing required dtypes: " + ", ".join(node_missing_dtypes))

        if reasons:
            blocking_nodes.append(_node_summary(node, reasons))

    base.update({
        "contract_status": (
            "needs_lowering_support"
            if blocking_nodes
            else "ready_for_existing_mlir_lowering"
        ),
        "supported_ops": sorted(present_supported),
        "unsupported_ops": sorted(present_unsupported),
        "blocking_nodes": blocking_nodes,
        "missing_required_attrs": missing_attrs,
        "missing_shapes": missing_shapes,
        "missing_dtypes": missing_dtypes,
        "preferred_mlir_targets": dict(sorted(target_counts.items())),
    })
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    try:
        graph_ir = json.loads(args.input_path.read_text(encoding="utf-8"))
        if not isinstance(graph_ir, dict):
            raise ValueError("root JSON value must be an object")
        report = check_lowering_contract(graph_ir)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"check_generic_lowering_contract: wrote {args.out}")
    print(f"  status: {report['contract_status']}")
    return 0 if report["contract_status"] != "invalid_generic_graph_ir" else 1


if __name__ == "__main__":
    raise SystemExit(main())
