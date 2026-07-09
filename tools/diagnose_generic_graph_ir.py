#!/usr/bin/env python3
"""Produce a diagnostics/readiness report for shape-annotated GenericGraphIR."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import verify_graph_ir

TRUTH_BOUNDARY = "diagnostics_only_no_domain_recognition_no_mlir_lowering_no_execution_plan_generation"


def _nodes(graph_ir: dict[str, Any]) -> list[dict[str, Any]]:
    return graph_ir.get("nodes", []) if isinstance(graph_ir.get("nodes"), list) else []


def _values(graph_ir: dict[str, Any]) -> list[dict[str, Any]]:
    return graph_ir.get("values", []) if isinstance(graph_ir.get("values"), list) else []


def _initializers(graph_ir: dict[str, Any]) -> list[dict[str, Any]]:
    return graph_ir.get("initializers", []) if isinstance(graph_ir.get("initializers"), list) else []


def _op_histogram(nodes: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(node.get("op", "")) for node in nodes))


def _shape_status_histogram(nodes: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(node.get("shape_inference_status", "unknown")) for node in nodes))


def _node_summary(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": node.get("id"),
        "name": node.get("name", ""),
        "op": node.get("op", ""),
        "source_node_id": node.get("source_node_id"),
        "source_op_type": node.get("source_op_type", ""),
        "source_name": node.get("source_name", ""),
        "shape_inference_notes": list(node.get("shape_inference_notes", [])),
    }


def _metadata_counts(graph_ir: dict[str, Any]) -> dict[str, int]:
    missing_dtype = 0
    missing_shape = 0
    unknown_dtype = 0
    unknown_shape = 0
    records_by_name: dict[str, dict[str, Any]] = {}
    for record in _values(graph_ir):
        name = record.get("name")
        if isinstance(name, str) and name:
            records_by_name[name] = record
    for record in _initializers(graph_ir):
        name = record.get("name")
        if isinstance(name, str) and name:
            records_by_name[name] = record

    for record in records_by_name.values():
        dtype = record.get("dtype")
        shape = record.get("shape")
        if "dtype" not in record:
            missing_dtype += 1
        elif dtype in ("", "unknown", None):
            unknown_dtype += 1
        if "shape" not in record:
            missing_shape += 1
        elif not isinstance(shape, list) or any(dim.get("kind") == "unknown" for dim in shape if isinstance(dim, dict)):
            unknown_shape += 1

    return {
        "missing_dtype": missing_dtype,
        "unknown_dtype": unknown_dtype,
        "missing_shape": missing_shape,
        "unknown_shape": unknown_shape,
    }


def _top_initializers(graph_ir: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    records = []
    for init in _initializers(graph_ir):
        records.append(
            {
                "name": init.get("name", ""),
                "dtype": init.get("dtype", "unknown"),
                "shape": init.get("shape", []),
                "raw_data_bytes": int(init.get("raw_data_bytes", 0) or 0),
            }
        )
    return sorted(records, key=lambda item: item["raw_data_bytes"], reverse=True)[:limit]


def _readiness(
    verifier_passed: bool,
    unknown_op_count: int,
    shape_histogram: dict[str, int],
) -> str:
    if not verifier_passed:
        return "invalid_ir"
    if unknown_op_count > 0:
        return "needs_op_support"
    if shape_histogram.get("error", 0) or shape_histogram.get("unknown", 0) or shape_histogram.get("partially_inferred", 0):
        return "needs_shape_support"
    return "ready_for_generic_lowering"


def diagnose_generic_graph_ir(graph_ir: dict[str, Any]) -> dict[str, Any]:
    verify_result = verify_graph_ir.verify_generic_graph_ir(graph_ir)
    nodes = _nodes(graph_ir)
    unknown_nodes = [node for node in nodes if node.get("op") == "nn.unknown"]
    shape_hist = _shape_status_histogram(nodes)

    report = {
        "graph_name": graph_ir.get("graph", {}).get("name", ""),
        "node_count": len(nodes),
        "value_count": len(_values(graph_ir)),
        "initializer_count": len(_initializers(graph_ir)),
        "op_histogram": _op_histogram(nodes),
        "unknown_op_count": len(unknown_nodes),
        "unknown_source_op_types": sorted({
            str(node.get("source_op_type", ""))
            for node in unknown_nodes
            if node.get("source_op_type", "")
        }),
        "shape_inference_status_histogram": shape_hist,
        "shape_error_nodes": [
            _node_summary(node)
            for node in nodes
            if node.get("shape_inference_status") == "error"
        ],
        "shape_unknown_nodes": [
            _node_summary(node)
            for node in nodes
            if node.get("shape_inference_status") == "unknown"
        ],
        "shape_partially_inferred_nodes": [
            _node_summary(node)
            for node in nodes
            if node.get("shape_inference_status") == "partially_inferred"
        ],
        "metadata_counts": _metadata_counts(graph_ir),
        "top_initializers_by_raw_data_bytes": _top_initializers(graph_ir),
        "verifier": {
            "passed": verify_result.passed,
            "errors": verify_result.errors,
        },
        "frontend_readiness_status": _readiness(
            verify_result.passed,
            len(unknown_nodes),
            shape_hist,
        ),
        "truth_boundary": TRUTH_BOUNDARY,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True, type=Path,
                        help="Path to shape-annotated GenericGraphIR JSON")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output diagnostics JSON report")
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"error: --in path does not exist: {args.input_path}", file=sys.stderr)
        return 1

    try:
        graph_ir = json.loads(args.input_path.read_text(encoding="utf-8"))
        report = diagnose_generic_graph_ir(graph_ir)
    except json.JSONDecodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"diagnose_generic_graph_ir: wrote {args.out}")
    print(f"  readiness: {report['frontend_readiness_status']}")
    return 0 if report["frontend_readiness_status"] != "invalid_ir" else 1


if __name__ == "__main__":
    raise SystemExit(main())
