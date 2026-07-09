#!/usr/bin/env python3
"""Verify ImportedGraphIR v0 or GenericGraphIR v0 JSON.

The verifier checks structural correctness only. It does not perform domain
recognition, target planning, or lowering.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMPORTED_SCHEMA = "imported_graph_ir"
GENERIC_SCHEMA = "generic_graph_ir"
SCHEMA_VERSION = "0.1.0"

SUPPORTED_GENERIC_OPS = {
    "nn.conv2d",
    "nn.conv_transpose2d",
    "nn.add",
    "nn.sub",
    "nn.mul",
    "nn.div",
    "nn.matmul",
    "nn.gemm",
    "nn.reshape",
    "nn.transpose",
    "nn.concat",
    "nn.split",
    "nn.slice",
    "nn.resize",
    "nn.maxpool2d",
    "nn.sigmoid",
    "nn.relu",
    "nn.softmax",
    "nn.identity",
    "nn.constant",
    "nn.unknown",
}

FORBIDDEN_SCHEMA_FIELD_TERMS = (
    "qwen",
    "llm",
    "yolo",
    "cv",
    "kv_cache",
    "attention",
    "backbone",
    "neck",
    "head",
)


@dataclass
class VerificationResult:
    passed: bool
    errors: list[str]


class GraphIRVerificationError(Exception):
    """Raised by callers that want exception-style validation."""


def _is_obj(value: Any) -> bool:
    return isinstance(value, dict)


def _is_list(value: Any) -> bool:
    return isinstance(value, list)


def _require_keys(obj: dict[str, Any], keys: list[str], label: str, errors: list[str]) -> None:
    for key in keys:
        if key not in obj:
            errors.append(f"{label} missing required field '{key}'")


def _names(records: list[Any], label: str, errors: list[str]) -> set[str]:
    out: set[str] = set()
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not _is_obj(record):
            errors.append(f"{label}[{index}] must be an object")
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"{label}[{index}] missing non-empty string field 'name'")
            continue
        if name in seen:
            errors.append(f"{label} duplicate name '{name}'")
        seen.add(name)
        out.add(name)
    return out


def _verify_shape_record(shape: Any, label: str, errors: list[str]) -> None:
    if not _is_list(shape):
        errors.append(f"{label}.shape must be an array")
        return
    for index, dim in enumerate(shape):
        if not _is_obj(dim):
            errors.append(f"{label}.shape[{index}] must be an object")
            continue
        _require_keys(dim, ["kind", "value"], f"{label}.shape[{index}]", errors)
        if dim.get("kind") not in {"static", "symbolic", "unknown"}:
            errors.append(f"{label}.shape[{index}].kind is not recognized: {dim.get('kind')!r}")


def _verify_value_records(records: list[Any], label: str, errors: list[str]) -> set[str]:
    names = _names(records, label, errors)
    for index, record in enumerate(records):
        if not _is_obj(record):
            continue
        _require_keys(record, ["name", "source_name", "dtype", "shape"], f"{label}[{index}]", errors)
        if "dtype" in record and not isinstance(record["dtype"], str):
            errors.append(f"{label}[{index}].dtype must be a string")
        if "shape" in record:
            _verify_shape_record(record["shape"], f"{label}[{index}]", errors)
    return names


def _verify_attribute_records(attrs: Any, label: str, errors: list[str]) -> None:
    if not _is_list(attrs):
        errors.append(f"{label}.attributes must be an array")
        return
    for index, attr in enumerate(attrs):
        if not _is_obj(attr):
            errors.append(f"{label}.attributes[{index}] must be an object")
            continue
        _require_keys(attr, ["name", "type", "value"], f"{label}.attributes[{index}]", errors)
        try:
            json.dumps(attr, sort_keys=True)
        except (TypeError, ValueError) as exc:
            errors.append(f"{label}.attributes[{index}] is not stable JSON: {exc}")


def _verify_node_ids(nodes: list[Any], errors: list[str]) -> set[Any]:
    ids: set[Any] = set()
    for index, node in enumerate(nodes):
        if not _is_obj(node):
            errors.append(f"nodes[{index}] must be an object")
            continue
        if "id" not in node:
            errors.append(f"nodes[{index}] missing required field 'id'")
            continue
        node_id = node["id"]
        if node_id in ids:
            errors.append(f"duplicate node id '{node_id}'")
        ids.add(node_id)
    return ids


def _verify_dataflow(
    nodes: list[Any],
    graph_inputs: set[str],
    graph_outputs: list[str],
    value_names: set[str],
    initializer_names: set[str],
    errors: list[str],
    graph_allows_duplicate_outputs: bool = False,
) -> None:
    available = set(graph_inputs) | set(value_names) | set(initializer_names)
    produced: set[str] = set()

    for index, node in enumerate(nodes):
        if not _is_obj(node):
            continue
        inputs = node.get("inputs")
        outputs = node.get("outputs")
        if not _is_list(inputs):
            errors.append(f"nodes[{index}].inputs must be an array")
            inputs = []
        if not _is_list(outputs):
            errors.append(f"nodes[{index}].outputs must be an array")
            outputs = []

        for name in inputs:
            if name == "":
                continue
            if not isinstance(name, str):
                errors.append(f"nodes[{index}].inputs contains non-string value {name!r}")
                continue
            if name not in available:
                errors.append(f"nodes[{index}] input '{name}' is unresolved")

        node_allows_duplicate_outputs = bool(node.get("allow_duplicate_outputs", False))
        for name in outputs:
            if name == "":
                continue
            if not isinstance(name, str):
                errors.append(f"nodes[{index}].outputs contains non-string value {name!r}")
                continue
            if (
                name in produced
                and not graph_allows_duplicate_outputs
                and not node_allows_duplicate_outputs
            ):
                errors.append(f"node output '{name}' is produced more than once")
            produced.add(name)
            available.add(name)

    declared = set(graph_inputs) | set(value_names) | set(initializer_names) | produced
    for name in graph_outputs:
        if name not in declared:
            errors.append(f"graph output '{name}' is neither produced nor declared")


def _verify_top_schema(ir: dict[str, Any], schema: str, errors: list[str]) -> None:
    _require_keys(ir, ["schema", "schema_version", "graph", "provenance"], "root", errors)
    if ir.get("schema") != schema:
        errors.append(f"root.schema must be '{schema}', got {ir.get('schema')!r}")
    if ir.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"root.schema_version must be '{SCHEMA_VERSION}', got {ir.get('schema_version')!r}")
    if "graph" in ir and not _is_obj(ir["graph"]):
        errors.append("root.graph must be an object")
    if "provenance" in ir and not _is_obj(ir["provenance"]):
        errors.append("root.provenance must be an object")


def verify_imported_graph_ir(ir: dict[str, Any]) -> VerificationResult:
    errors: list[str] = []
    _verify_top_schema(ir, IMPORTED_SCHEMA, errors)

    graph = ir.get("graph") if _is_obj(ir.get("graph")) else {}
    _require_keys(graph, ["inputs", "outputs", "nodes", "values", "initializers"], "graph", errors)

    inputs = graph.get("inputs", [])
    outputs = graph.get("outputs", [])
    nodes = graph.get("nodes", [])
    values = graph.get("values", [])
    initializers = graph.get("initializers", [])

    for field_name, value in (
        ("graph.inputs", inputs),
        ("graph.outputs", outputs),
        ("graph.nodes", nodes),
        ("graph.values", values),
        ("graph.initializers", initializers),
    ):
        if not _is_list(value):
            errors.append(f"{field_name} must be an array")

    if not _is_list(inputs):
        inputs = []
    if not _is_list(outputs):
        outputs = []
    if not _is_list(nodes):
        nodes = []
    if not _is_list(values):
        values = []
    if not _is_list(initializers):
        initializers = []

    _verify_node_ids(nodes, errors)
    value_names = _verify_value_records(values, "graph.values", errors)
    initializer_names = _verify_value_records(initializers, "graph.initializers", errors)

    for index, node in enumerate(nodes):
        if not _is_obj(node):
            continue
        _require_keys(
            node,
            ["id", "name", "source_name", "op_type", "domain", "inputs", "outputs", "attributes"],
            f"graph.nodes[{index}]",
            errors,
        )
        _verify_attribute_records(node.get("attributes"), f"graph.nodes[{index}]", errors)

    graph_inputs = {name for name in inputs if isinstance(name, str)}
    graph_outputs = [name for name in outputs if isinstance(name, str)]
    _verify_dataflow(
        nodes,
        graph_inputs,
        graph_outputs,
        value_names,
        initializer_names,
        errors,
        graph_allows_duplicate_outputs=bool(graph.get("allow_duplicate_outputs", False)),
    )

    return VerificationResult(not errors, errors)


def _schema_field_paths(obj: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            paths.append(path)
            paths.extend(_schema_field_paths(value, path))
    elif isinstance(obj, list):
        for value in obj:
            paths.extend(_schema_field_paths(value, prefix))
    return paths


def _verify_no_forbidden_schema_field_terms(ir: dict[str, Any], errors: list[str]) -> None:
    for path in _schema_field_paths(ir):
        lowered = path.lower()
        for term in FORBIDDEN_SCHEMA_FIELD_TERMS:
            if term in lowered:
                errors.append(f"schema field path '{path}' contains forbidden domain term '{term}'")


def verify_generic_graph_ir(ir: dict[str, Any]) -> VerificationResult:
    errors: list[str] = []
    _verify_top_schema(ir, GENERIC_SCHEMA, errors)
    _require_keys(ir, ["nodes", "values", "initializers"], "root", errors)
    _verify_no_forbidden_schema_field_terms(ir, errors)

    graph = ir.get("graph") if _is_obj(ir.get("graph")) else {}
    _require_keys(graph, ["inputs", "outputs"], "graph", errors)
    inputs = graph.get("inputs", [])
    outputs = graph.get("outputs", [])
    nodes = ir.get("nodes", [])
    values = ir.get("values", [])
    initializers = ir.get("initializers", [])

    for field_name, value in (
        ("graph.inputs", inputs),
        ("graph.outputs", outputs),
        ("root.nodes", nodes),
        ("root.values", values),
        ("root.initializers", initializers),
    ):
        if not _is_list(value):
            errors.append(f"{field_name} must be an array")

    if not _is_list(inputs):
        inputs = []
    if not _is_list(outputs):
        outputs = []
    if not _is_list(nodes):
        nodes = []
    if not _is_list(values):
        values = []
    if not _is_list(initializers):
        initializers = []

    _verify_node_ids(nodes, errors)
    value_names = _verify_value_records(values, "root.values", errors)
    initializer_names = _verify_value_records(initializers, "root.initializers", errors)
    missing_initializer_values = initializer_names - value_names
    if missing_initializer_values:
        errors.append(
            "initializers missing corresponding value metadata: "
            + ", ".join(sorted(missing_initializer_values))
        )

    for index, node in enumerate(nodes):
        if not _is_obj(node):
            continue
        _require_keys(
            node,
            [
                "id",
                "name",
                "op",
                "inputs",
                "outputs",
                "attributes",
                "source_node_id",
                "source_op_type",
                "source_name",
            ],
            f"nodes[{index}]",
            errors,
        )
        op = node.get("op")
        if op not in SUPPORTED_GENERIC_OPS:
            errors.append(f"nodes[{index}].op is unsupported: {op!r}")
        _verify_attribute_records(node.get("attributes"), f"nodes[{index}]", errors)

    graph_inputs = {name for name in inputs if isinstance(name, str)}
    graph_outputs = [name for name in outputs if isinstance(name, str)]
    _verify_dataflow(
        nodes,
        graph_inputs,
        graph_outputs,
        value_names,
        initializer_names,
        errors,
        graph_allows_duplicate_outputs=bool(graph.get("allow_duplicate_outputs", False)),
    )

    return VerificationResult(not errors, errors)


def verify_graph_ir(ir: dict[str, Any]) -> VerificationResult:
    schema = ir.get("schema") if isinstance(ir, dict) else None
    if schema == IMPORTED_SCHEMA:
        return verify_imported_graph_ir(ir)
    if schema == GENERIC_SCHEMA:
        return verify_generic_graph_ir(ir)
    return VerificationResult(False, [f"unsupported or missing schema: {schema!r}"])


def assert_valid_graph_ir(ir: dict[str, Any]) -> None:
    result = verify_graph_ir(ir)
    if not result.passed:
        raise GraphIRVerificationError("; ".join(result.errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Graph IR JSON file(s) to verify")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable validation results")
    args = parser.parse_args()

    results = []
    overall_ok = True
    for path in args.paths:
        if not path.exists():
            result = VerificationResult(False, [f"path does not exist: {path}"])
        else:
            try:
                ir = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(ir, dict):
                    result = VerificationResult(False, ["root JSON value must be an object"])
                else:
                    result = verify_graph_ir(ir)
            except json.JSONDecodeError as exc:
                result = VerificationResult(False, [f"JSON parse error: {exc}"])
        overall_ok = overall_ok and result.passed
        results.append({"path": str(path), "passed": result.passed, "errors": result.errors})

    if args.json:
        print(json.dumps({"passed": overall_ok, "results": results}, indent=2))
    else:
        for item in results:
            if item["passed"]:
                print(f"PASS {item['path']}")
            else:
                print(f"FAIL {item['path']}", file=sys.stderr)
                for error in item["errors"]:
                    print(f"  - {error}", file=sys.stderr)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
