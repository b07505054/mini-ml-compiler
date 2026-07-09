#!/usr/bin/env python3
"""Canonicalize GenericGraphIR v0 JSON.

The pass normalizes selected nn.* attributes into compiler-owned
canonical_attrs while preserving original source attributes.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import verify_graph_ir

SCHEMA = "generic_graph_ir"
SCHEMA_VERSION = "0.1.0"
CANONICALIZATION_VERSION = "0.1.0"
TRUTH_BOUNDARY = "generic_graph_ir_attribute_canonicalization_no_domain_recognition"


class GenericGraphIRCanonicalizationError(Exception):
    """Raised when GenericGraphIR cannot be canonicalized."""


def _attr_map(attrs: list[dict[str, Any]]) -> dict[str, Any]:
    return {attr.get("name", ""): attr.get("value") for attr in attrs if isinstance(attr, dict)}


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise GenericGraphIRCanonicalizationError(f"expected integer attribute, got {value!r}")


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise GenericGraphIRCanonicalizationError(f"expected numeric attribute, got {value!r}")


def _as_string(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    raise GenericGraphIRCanonicalizationError(f"expected string attribute, got {value!r}")


def _as_int_list(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if not isinstance(value, list):
        raise GenericGraphIRCanonicalizationError(f"expected integer list attribute, got {value!r}")
    return [_as_int(item, 0) for item in value]


def _canonical_conv2d(attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "pads": _as_int_list(attrs.get("pads"), [0, 0, 0, 0]),
        "strides": _as_int_list(attrs.get("strides"), [1, 1]),
        "dilations": _as_int_list(attrs.get("dilations"), [1, 1]),
        "groups": _as_int(attrs.get("group"), 1),
        "kernel_shape": _as_int_list(attrs.get("kernel_shape"), []),
    }


def _canonical_conv_transpose2d(attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "pads": _as_int_list(attrs.get("pads"), [0, 0, 0, 0]),
        "strides": _as_int_list(attrs.get("strides"), [1, 1]),
        "dilations": _as_int_list(attrs.get("dilations"), [1, 1]),
        "groups": _as_int(attrs.get("group"), 1),
        "kernel_shape": _as_int_list(attrs.get("kernel_shape"), []),
        "output_padding": _as_int_list(attrs.get("output_padding"), [0, 0]),
        "output_shape": _as_int_list(attrs.get("output_shape"), []),
    }


def _canonical_maxpool2d(attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "kernel_shape": _as_int_list(attrs.get("kernel_shape"), []),
        "pads": _as_int_list(attrs.get("pads"), [0, 0, 0, 0]),
        "strides": _as_int_list(attrs.get("strides"), [1, 1]),
        "dilations": _as_int_list(attrs.get("dilations"), [1, 1]),
        "ceil_mode": _as_int(attrs.get("ceil_mode"), 0),
    }


def _canonical_transpose(attrs: dict[str, Any]) -> dict[str, Any]:
    return {"perm": _as_int_list(attrs.get("perm"), [])}


def _canonical_concat(attrs: dict[str, Any]) -> dict[str, Any]:
    return {"axis": _as_int(attrs.get("axis"), 0)}


def _canonical_split(attrs: dict[str, Any]) -> dict[str, Any]:
    out = {"axis": _as_int(attrs.get("axis"), 0)}
    if attrs.get("split") is not None:
        out["split"] = _as_int_list(attrs.get("split"), [])
    return out


def _canonical_slice(attrs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name in ["axes", "starts", "ends", "steps"]:
        if attrs.get(name) is not None:
            out[name] = _as_int_list(attrs.get(name), [])
    return out


def _canonical_resize(attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": _as_string(attrs.get("mode"), "nearest"),
        "coordinate_transformation_mode": _as_string(
            attrs.get("coordinate_transformation_mode"),
            "half_pixel",
        ),
        "nearest_mode": _as_string(attrs.get("nearest_mode"), "round_prefer_floor"),
    }


def _canonical_softmax(attrs: dict[str, Any]) -> dict[str, Any]:
    return {"axis": _as_int(attrs.get("axis"), -1)}


def _canonical_gemm(attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "alpha": _as_float(attrs.get("alpha"), 1.0),
        "beta": _as_float(attrs.get("beta"), 1.0),
        "transA": _as_int(attrs.get("transA"), 0),
        "transB": _as_int(attrs.get("transB"), 0),
    }


def _canonical_reshape(attrs: dict[str, Any]) -> dict[str, Any]:
    return {"allowzero": _as_int(attrs.get("allowzero"), 0)}


CANONICALIZERS = {
    "nn.conv2d": _canonical_conv2d,
    "nn.conv_transpose2d": _canonical_conv_transpose2d,
    "nn.maxpool2d": _canonical_maxpool2d,
    "nn.transpose": _canonical_transpose,
    "nn.concat": _canonical_concat,
    "nn.split": _canonical_split,
    "nn.slice": _canonical_slice,
    "nn.resize": _canonical_resize,
    "nn.softmax": _canonical_softmax,
    "nn.gemm": _canonical_gemm,
    "nn.reshape": _canonical_reshape,
}


def _record_map(records: list[Any]) -> dict[str, dict[str, Any]]:
    return {
        record["name"]: record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }


def _literal_values(record: dict[str, Any] | None) -> list[Any] | None:
    if not record:
        return None
    values = record.get("literal_values")
    return list(values) if isinstance(values, list) else None


def _literal_from_input(node: dict[str, Any], tensor_records: dict[str, dict[str, Any]], index: int) -> list[Any] | None:
    inputs = node.get("inputs", [])
    if index >= len(inputs) or not isinstance(inputs[index], str) or inputs[index] == "":
        return None
    return _literal_values(tensor_records.get(inputs[index]))


def _literal_int_list(values: list[Any] | None) -> list[int] | None:
    if values is None:
        return None
    out = []
    for value in values:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            out.append(value)
        elif isinstance(value, float) and value.is_integer():
            out.append(int(value))
        else:
            return None
    return out


def _literal_number_list(values: list[Any] | None) -> list[int | float] | None:
    if values is None:
        return None
    out = []
    for value in values:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            out.append(value)
        elif isinstance(value, float):
            out.append(value)
        else:
            return None
    return out


def _augment_shape_bearing_attrs(node: dict[str, Any], tensor_records: dict[str, dict[str, Any]]) -> None:
    op = node.get("op")
    attrs = node.setdefault("canonical_attrs", {})
    if op == "nn.reshape":
        target_shape = _literal_int_list(_literal_from_input(node, tensor_records, 1))
        if target_shape is not None:
            attrs["target_shape"] = target_shape
    elif op == "nn.slice":
        starts = _literal_int_list(_literal_from_input(node, tensor_records, 1))
        ends = _literal_int_list(_literal_from_input(node, tensor_records, 2))
        axes = _literal_int_list(_literal_from_input(node, tensor_records, 3))
        steps = _literal_int_list(_literal_from_input(node, tensor_records, 4))
        if starts is not None:
            attrs["starts"] = starts
        if ends is not None:
            attrs["ends"] = ends
        if axes is not None:
            attrs["axes"] = axes
        if steps is not None:
            attrs["steps"] = steps
    elif op == "nn.resize":
        scales = _literal_number_list(_literal_from_input(node, tensor_records, 2))
        sizes = _literal_int_list(_literal_from_input(node, tensor_records, 3))
        if scales is not None:
            attrs["scales"] = scales
        if sizes is not None:
            attrs["sizes"] = sizes
    elif op == "nn.split" and "split" not in attrs:
        split = _literal_int_list(_literal_from_input(node, tensor_records, 1))
        if split is not None:
            attrs["split"] = split


def canonicalize_generic_graph_ir(generic: dict[str, Any]) -> dict[str, Any]:
    result = verify_graph_ir.verify_generic_graph_ir(generic)
    if not result.passed:
        raise GenericGraphIRCanonicalizationError("; ".join(result.errors))
    if generic.get("schema") != SCHEMA or generic.get("schema_version") != SCHEMA_VERSION:
        raise GenericGraphIRCanonicalizationError("unsupported GenericGraphIR schema")

    out = copy.deepcopy(generic)
    tensor_records = _record_map(out.get("values", []))
    tensor_records.update(_record_map(out.get("initializers", [])))
    for node in out["nodes"]:
        attrs = node.get("attributes", [])
        if "source_attributes" not in node:
            node["source_attributes"] = copy.deepcopy(attrs)
        attr_values = _attr_map(attrs)
        canonicalizer = CANONICALIZERS.get(node.get("op"))
        if canonicalizer:
            node["canonical_attrs"] = canonicalizer(attr_values)
        else:
            node["canonical_attrs"] = {}
        _augment_shape_bearing_attrs(node, tensor_records)
        node["canonicalized"] = True
        node["canonicalization_version"] = CANONICALIZATION_VERSION

    provenance = out.setdefault("provenance", {})
    provenance["canonicalization_version"] = CANONICALIZATION_VERSION
    provenance["canonicalization_truth_boundary"] = TRUTH_BOUNDARY
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True, type=Path,
                        help="Path to GenericGraphIR JSON")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output canonicalized GenericGraphIR JSON")
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"error: --in path does not exist: {args.input_path}", file=sys.stderr)
        return 1

    try:
        generic = json.loads(args.input_path.read_text(encoding="utf-8"))
        canonical = canonicalize_generic_graph_ir(generic)
        verify_result = verify_graph_ir.verify_generic_graph_ir(canonical)
        if not verify_result.passed:
            raise GenericGraphIRCanonicalizationError("; ".join(verify_result.errors))
    except (json.JSONDecodeError, GenericGraphIRCanonicalizationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
    print(f"canonicalize_generic_graph_ir: wrote {args.out}")
    print(f"  graph: {canonical['graph']['name']}")
    print(f"  nodes: {len(canonical['nodes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
