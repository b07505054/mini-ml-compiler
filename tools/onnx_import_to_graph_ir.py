#!/usr/bin/env python3
"""ONNX protobuf -> ImportedGraphIR v0 JSON.

This tool is a generic importer boundary. It preserves ONNX graph structure
and metadata without assigning model-family semantics or lowering into a
domain dialect.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "imported_graph_ir"
SCHEMA_VERSION = "0.1.0"
TRUTH_BOUNDARY = "onnx_protobuf_metadata_preserved_no_domain_recognition"


class OnnxImportToGraphIRError(Exception):
    """Raised when ONNX metadata cannot be converted into ImportedGraphIR."""


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _dtype_from_elem_type(elem_type: int, onnx_mod: Any) -> str:
    if elem_type == 0:
        return "unknown"
    try:
        return onnx_mod.TensorProto.DataType.Name(elem_type).lower()
    except ValueError:
        return f"unknown_{elem_type}"


def _shape_from_tensor_type(tensor_type: Any) -> list[dict[str, Any]]:
    shape = []
    if not tensor_type.HasField("shape"):
        return shape
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append({"kind": "static", "value": int(dim.dim_value)})
        elif dim.HasField("dim_param"):
            shape.append({"kind": "symbolic", "value": dim.dim_param})
        else:
            shape.append({"kind": "unknown", "value": None})
    return shape


def _value_info_to_record(value_info: Any, onnx_mod: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": value_info.name,
        "source_name": value_info.name,
        "dtype": "unknown",
        "shape": [],
    }
    value_type = value_info.type
    if value_type.HasField("tensor_type"):
        tensor_type = value_type.tensor_type
        record["dtype"] = _dtype_from_elem_type(tensor_type.elem_type, onnx_mod)
        record["shape"] = _shape_from_tensor_type(tensor_type)
    return record


def _tensor_shape(dims: Any) -> list[dict[str, Any]]:
    return [{"kind": "static", "value": int(dim)} for dim in dims]


def _initializer_to_record(init: Any, onnx_mod: Any) -> dict[str, Any]:
    return {
        "name": init.name,
        "source_name": init.name,
        "dtype": _dtype_from_elem_type(init.data_type, onnx_mod),
        "shape": _tensor_shape(init.dims),
        "data_location": int(init.data_location),
        "raw_data_bytes": len(init.raw_data),
    }


def _attribute_value(attr: Any, onnx_mod: Any) -> Any:
    attr_type = attr.type
    attr_type_name = onnx_mod.AttributeProto.AttributeType.Name(attr_type).lower()
    if attr_type_name == "float":
        return float(attr.f)
    if attr_type_name == "int":
        return int(attr.i)
    if attr_type_name == "string":
        return attr.s.decode("utf-8", errors="replace")
    if attr_type_name == "floats":
        return [float(v) for v in attr.floats]
    if attr_type_name == "ints":
        return [int(v) for v in attr.ints]
    if attr_type_name == "strings":
        return [v.decode("utf-8", errors="replace") for v in attr.strings]
    if attr_type_name == "tensor":
        return {
            "name": attr.t.name,
            "dtype": _dtype_from_elem_type(attr.t.data_type, onnx_mod),
            "shape": _tensor_shape(attr.t.dims),
        }
    if attr_type_name == "graph":
        return {"name": attr.g.name, "node_count": len(attr.g.node)}
    if attr_type_name == "sparse_tensor":
        return {"name": attr.sparse_tensor.values.name}
    if attr_type_name == "type_proto":
        return str(attr.tp)
    return str(attr)


def _attribute_to_record(attr: Any, onnx_mod: Any) -> dict[str, Any]:
    attr_type = onnx_mod.AttributeProto.AttributeType.Name(attr.type).lower()
    return {
        "name": attr.name,
        "type": attr_type,
        "value": _attribute_value(attr, onnx_mod),
    }


def _node_to_record(index: int, node: Any, onnx_mod: Any) -> dict[str, Any]:
    return {
        "id": index,
        "name": node.name,
        "source_name": node.name,
        "op_type": node.op_type,
        "domain": node.domain,
        "inputs": list(node.input),
        "outputs": list(node.output),
        "attributes": [_attribute_to_record(attr, onnx_mod) for attr in node.attribute],
    }


def _opset_imports(model: Any) -> list[dict[str, Any]]:
    return [
        {"domain": opset.domain, "version": int(opset.version)}
        for opset in model.opset_import
    ]


def _default_opset_version(model: Any) -> int | None:
    for opset in model.opset_import:
        if opset.domain == "":
            return int(opset.version)
    return None


def import_onnx_to_graph_ir(onnx_path: Path, onnx_mod: Any) -> dict[str, Any]:
    model = onnx_mod.load(str(onnx_path))
    graph = model.graph

    values: dict[str, dict[str, Any]] = {}
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        values[value_info.name] = _value_info_to_record(value_info, onnx_mod)
    for init in graph.initializer:
        values.setdefault(
            init.name,
            {
                "name": init.name,
                "source_name": init.name,
                "dtype": _dtype_from_elem_type(init.data_type, onnx_mod),
                "shape": _tensor_shape(init.dims),
            },
        )

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "graph": {
            "name": graph.name,
            "source_name": graph.name,
            "opset_version": _default_opset_version(model),
            "opset_imports": _opset_imports(model),
            "inputs": [value_info.name for value_info in graph.input],
            "outputs": [value_info.name for value_info in graph.output],
            "nodes": [
                _node_to_record(index, node, onnx_mod)
                for index, node in enumerate(graph.node)
            ],
            "values": [
                values[name]
                for name in sorted(values)
            ],
            "initializers": [
                _initializer_to_record(init, onnx_mod)
                for init in graph.initializer
            ],
        },
        "provenance": {
            "source_format": "onnx",
            "source_file": onnx_path.name,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "ir_version": int(model.ir_version),
            "truth_boundary": TRUTH_BOUNDARY,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path, help="Path to a .onnx model")
    parser.add_argument("--out", required=True, type=Path, help="Output ImportedGraphIR JSON")
    args = parser.parse_args()

    if not module_available("onnx"):
        print("error: install the 'onnx' package to run this importer", file=sys.stderr)
        return 1

    import onnx  # noqa: PLC0415

    if not args.onnx.exists():
        print(f"error: --onnx path does not exist: {args.onnx}", file=sys.stderr)
        return 1

    try:
        graph_ir = import_onnx_to_graph_ir(args.onnx, onnx)
    except OnnxImportToGraphIRError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(graph_ir, indent=2) + "\n", encoding="utf-8")
    print(f"onnx_import_to_graph_ir: wrote {args.out}")
    print(f"  graph: {graph_ir['graph']['name']}")
    print(f"  nodes: {len(graph_ir['graph']['nodes'])}")
    print(f"  initializers: {len(graph_ir['graph']['initializers'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
