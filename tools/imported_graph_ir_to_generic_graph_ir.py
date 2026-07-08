#!/usr/bin/env python3
"""ImportedGraphIR v0 -> GenericGraphIR v0 JSON.

This tool normalizes source-faithful imported nodes into a small
compiler-owned, model-agnostic op vocabulary. It does not perform domain
recognition or lower into a domain dialect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "generic_graph_ir"
SCHEMA_VERSION = "0.1.0"
SUPPORTED_IMPORTED_SCHEMA = "imported_graph_ir"
TRUTH_BOUNDARY = "imported_graph_ir_normalized_no_domain_recognition"

ONNX_TO_GENERIC_OP = {
    "Conv": "nn.conv2d",
    "Add": "nn.add",
    "Mul": "nn.mul",
    "MatMul": "nn.matmul",
    "Gemm": "nn.gemm",
    "Reshape": "nn.reshape",
    "Transpose": "nn.transpose",
    "Concat": "nn.concat",
    "Resize": "nn.resize",
    "Sigmoid": "nn.sigmoid",
    "Relu": "nn.relu",
    "Softmax": "nn.softmax",
    "Identity": "nn.identity",
}
UNKNOWN_OP = "nn.unknown"


class ImportedGraphIRToGenericGraphIRError(Exception):
    """Raised when ImportedGraphIR is malformed or unsupported."""


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ImportedGraphIRToGenericGraphIRError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ImportedGraphIRToGenericGraphIRError(f"{label} must be an array")
    return value


def _source_graph_ref(imported: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": imported.get("schema", ""),
        "schema_version": imported.get("schema_version", ""),
        "graph_name": imported.get("graph", {}).get("name", ""),
    }


def _normalize_node(node: dict[str, Any]) -> dict[str, Any]:
    source_op_type = str(node.get("op_type", ""))
    generic_op = ONNX_TO_GENERIC_OP.get(source_op_type, UNKNOWN_OP)
    source_id = node.get("id")
    return {
        "id": source_id,
        "name": node.get("name", ""),
        "op": generic_op,
        "inputs": list(node.get("inputs", [])),
        "outputs": list(node.get("outputs", [])),
        "attributes": list(node.get("attributes", [])),
        "source_node_id": source_id,
        "source_op_type": source_op_type,
        "source_name": node.get("source_name", node.get("name", "")),
        "source_domain": node.get("domain", ""),
    }


def _normalize_values(values: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for value in values:
        value = _require_dict(value, "graph.values[]")
        normalized.append(
            {
                "name": value.get("name", ""),
                "source_name": value.get("source_name", value.get("name", "")),
                "dtype": value.get("dtype", "unknown"),
                "shape": list(value.get("shape", [])),
            }
        )
    return normalized


def _normalize_initializers(initializers: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for init in initializers:
        init = _require_dict(init, "graph.initializers[]")
        record = {
            "name": init.get("name", ""),
            "source_name": init.get("source_name", init.get("name", "")),
            "dtype": init.get("dtype", "unknown"),
            "shape": list(init.get("shape", [])),
        }
        if "data_location" in init:
            record["data_location"] = init["data_location"]
        if "raw_data_bytes" in init:
            record["raw_data_bytes"] = init["raw_data_bytes"]
        normalized.append(record)
    return normalized


def convert_imported_graph_ir(imported: dict[str, Any]) -> dict[str, Any]:
    if imported.get("schema") != SUPPORTED_IMPORTED_SCHEMA:
        raise ImportedGraphIRToGenericGraphIRError(
            f"expected schema '{SUPPORTED_IMPORTED_SCHEMA}', got '{imported.get('schema')}'"
        )

    graph = _require_dict(imported.get("graph"), "graph")
    nodes = _require_list(graph.get("nodes", []), "graph.nodes")
    values = _require_list(graph.get("values", []), "graph.values")
    initializers = _require_list(graph.get("initializers", []), "graph.initializers")

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "graph": {
            "name": graph.get("name", ""),
            "source_name": graph.get("source_name", graph.get("name", "")),
            "inputs": list(graph.get("inputs", [])),
            "outputs": list(graph.get("outputs", [])),
            "opset_version": graph.get("opset_version"),
            "opset_imports": list(graph.get("opset_imports", [])),
            "source_graph": _source_graph_ref(imported),
        },
        "nodes": [_normalize_node(_require_dict(node, "graph.nodes[]")) for node in nodes],
        "values": _normalize_values(values),
        "initializers": _normalize_initializers(initializers),
        "provenance": {
            "source_schema": imported.get("schema", ""),
            "source_schema_version": imported.get("schema_version", ""),
            "source_truth_boundary": imported.get("provenance", {}).get("truth_boundary", ""),
            "truth_boundary": TRUTH_BOUNDARY,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True, type=Path,
                        help="Path to ImportedGraphIR JSON")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output GenericGraphIR JSON")
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"error: --in path does not exist: {args.input_path}", file=sys.stderr)
        return 1

    try:
        imported = json.loads(args.input_path.read_text(encoding="utf-8"))
        generic = convert_imported_graph_ir(imported)
    except (json.JSONDecodeError, ImportedGraphIRToGenericGraphIRError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(generic, indent=2) + "\n", encoding="utf-8")
    print(f"imported_graph_ir_to_generic_graph_ir: wrote {args.out}")
    print(f"  graph: {generic['graph']['name']}")
    print(f"  nodes: {len(generic['nodes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
