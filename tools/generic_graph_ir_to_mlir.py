#!/usr/bin/env python3
"""Emit existing-dialect MLIR from shape-annotated GenericGraphIR v0.

This is a deliberately small first emitter. It accepts only static f32 tensor
forms for the initial elementwise subset and emits upstream MLIR dialects:
func, tensor, linalg, arith, and math.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import check_generic_lowering_contract

SUPPORTED_EMITTER_OPS = {
    "nn.constant",
    "nn.identity",
    "nn.add",
    "nn.sub",
    "nn.mul",
    "nn.div",
    "nn.conv2d",
    "nn.conv_transpose2d",
    "nn.relu",
    "nn.sigmoid",
    "nn.reshape",
    "nn.maxpool2d",
    "nn.softmax",
    "nn.transpose",
    "nn.resize",
    "nn.concat",
    "nn.slice",
    "nn.split",
}

DTYPE_TO_MLIR = {
    "float": "f32",
    "float32": "f32",
    "fp32": "f32",
    "f32": "f32",
}

BINARY_ARITH = {
    "nn.add": "arith.addf",
    "nn.sub": "arith.subf",
    "nn.mul": "arith.mulf",
    "nn.div": "arith.divf",
}

TRUTH_BOUNDARY_COMMENT = (
    "generic_graph_ir_to_existing_mlir_no_domain_recognition_"
    "no_execution_plan_generation"
)


class GenericGraphIRToMLIRError(Exception):
    """Raised when GenericGraphIR cannot be emitted as v0 MLIR."""


def _sanitize_symbol(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_$.-]", "_", name or "graph")
    base = re.sub(r"^[^A-Za-z_$]", "_", base)
    return base or "graph"


def _ssa_name(name: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_$.-]", "_", name or "v")
    base = re.sub(r"^[^A-Za-z_$]", "_", base)
    candidate = base or "v"
    index = 0
    while candidate in used:
        index += 1
        candidate = f"{base}_{index}"
    used.add(candidate)
    return f"%{candidate}"


def _static_dims(record: dict[str, Any]) -> list[int]:
    dims = []
    shape = record.get("shape")
    if not isinstance(shape, list):
        raise GenericGraphIRToMLIRError(f"value '{record.get('name')}' is missing static shape")
    for dim in shape:
        if not isinstance(dim, dict) or dim.get("kind") != "static" or not isinstance(dim.get("value"), int):
            raise GenericGraphIRToMLIRError(f"value '{record.get('name')}' has non-static shape")
        dims.append(int(dim["value"]))
    return dims


def _mlir_elem_type(record: dict[str, Any]) -> str:
    dtype = str(record.get("dtype", ""))
    elem_type = DTYPE_TO_MLIR.get(dtype)
    if elem_type is None:
        raise GenericGraphIRToMLIRError(
            f"value '{record.get('name')}' has unsupported dtype '{dtype}' for emitter v0"
        )
    return elem_type


def _mlir_tensor_type(record: dict[str, Any]) -> str:
    dims = _static_dims(record)
    elem_type = _mlir_elem_type(record)
    if not dims:
        return f"tensor<{elem_type}>"
    return "tensor<" + "x".join(str(dim) for dim in dims) + f"x{elem_type}>"


def _identity_affine_map(rank: int) -> str:
    dims = ", ".join(f"d{i}" for i in range(rank))
    return f"affine_map<({dims}) -> ({dims})>"


def _iterator_types(rank: int) -> str:
    return "[" + ", ".join('"parallel"' for _ in range(rank)) + "]"


def _softmax_iterator_types(rank: int, axis: int) -> str:
    return "[" + ", ".join('"reduction"' if index == axis else '"parallel"' for index in range(rank)) + "]"


def _element_count(dims: list[int]) -> int:
    count = 1
    for dim in dims:
        count *= dim
    return count


def _normalize_axis(axis: Any, rank: int, op_name: str) -> int:
    if not isinstance(axis, int):
        raise GenericGraphIRToMLIRError(f"{op_name} axis must be an integer")
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        raise GenericGraphIRToMLIRError(f"{op_name} axis {axis} is outside rank {rank}")
    return normalized


def _static_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        raise GenericGraphIRToMLIRError(f"{label} must be a static integer list")
    out = []
    for item in value:
        if not isinstance(item, int):
            raise GenericGraphIRToMLIRError(f"{label} must contain only integers")
        out.append(item)
    return out


def _list_text(values: list[int]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def _dense_i64_vector(values: list[int]) -> str:
    return "dense<[" + ", ".join(str(value) for value in values) + f"]> : vector<{len(values)}xi64>"


def _drop_axis_affine_map(rank: int, axis: int) -> str:
    dims = ", ".join(f"d{i}" for i in range(rank))
    results = [f"d{i}" for i in range(rank) if i != axis]
    return f"affine_map<({dims}) -> ({', '.join(results)})>"


def _reshape_resolved_target(
    target: Any,
    allowzero: Any,
    input_dims: list[int],
) -> list[int]:
    if not isinstance(target, list):
        raise GenericGraphIRToMLIRError("nn.reshape requires canonical target_shape")
    if any(not isinstance(dim, int) for dim in target):
        raise GenericGraphIRToMLIRError("nn.reshape target_shape must contain only integers")
    if allowzero not in (0, 1):
        raise GenericGraphIRToMLIRError("nn.reshape allowzero must be 0 or 1")
    if sum(dim == -1 for dim in target) > 1:
        raise GenericGraphIRToMLIRError("nn.reshape target_shape may contain at most one -1")
    if any(dim < -1 for dim in target):
        raise GenericGraphIRToMLIRError("nn.reshape target_shape contains an invalid negative dimension")
    if allowzero == 1 and -1 in target and 0 in target:
        raise GenericGraphIRToMLIRError("nn.reshape cannot combine allowzero=1, zero dimensions, and -1")

    resolved = list(target)
    for index, dim in enumerate(resolved):
        if dim == 0 and allowzero == 0:
            if index >= len(input_dims):
                raise GenericGraphIRToMLIRError("nn.reshape zero copy dimension is outside input rank")
            resolved[index] = input_dims[index]

    infer_index = resolved.index(-1) if -1 in resolved else None
    if infer_index is not None:
        explicit = [dim for index, dim in enumerate(resolved) if index != infer_index]
        explicit_count = _element_count(explicit)
        input_count = _element_count(input_dims)
        if explicit_count == 0 or input_count % explicit_count != 0:
            raise GenericGraphIRToMLIRError("nn.reshape input size is incompatible with target_shape")
        resolved[infer_index] = input_count // explicit_count
    return resolved


def _derive_collapse_reassociation(input_dims: list[int], output_dims: list[int]) -> list[list[int]] | None:
    groups: list[list[int]] = []
    input_index = 0
    for output_dim in output_dims:
        if input_index >= len(input_dims):
            return None
        product = input_dims[input_index]
        group = [input_index]
        input_index += 1
        while input_index < len(input_dims) and product < output_dim:
            product *= input_dims[input_index]
            group.append(input_index)
            input_index += 1
        if not group:
            return None
        if product != output_dim:
            return None
        groups.append(group)
    if input_index != len(input_dims):
        return None
    return groups


def _derive_expand_reassociation(input_dims: list[int], output_dims: list[int]) -> list[list[int]] | None:
    groups: list[list[int]] = []
    output_index = 0
    for input_dim in input_dims:
        if output_index >= len(output_dims):
            return None
        product = output_dims[output_index]
        group = [output_index]
        output_index += 1
        while output_index < len(output_dims) and product < input_dim:
            product *= output_dims[output_index]
            group.append(output_index)
            output_index += 1
        if not group:
            return None
        if product != input_dim:
            return None
        groups.append(group)
    if output_index != len(output_dims):
        return None
    return groups


def _format_reassociation(groups: list[list[int]]) -> str:
    return "[" + ", ".join("[" + ", ".join(str(index) for index in group) + "]" for group in groups) + "]"


def _record_map(graph_ir: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for field in ("values", "initializers"):
        for record in graph_ir.get(field, []):
            if isinstance(record, dict) and isinstance(record.get("name"), str):
                records[record["name"]] = record
    return records


def _node_by_output(graph_ir: dict[str, Any]) -> dict[str, dict[str, Any]]:
    producers: dict[str, dict[str, Any]] = {}
    for node in graph_ir.get("nodes", []):
        if not isinstance(node, dict):
            continue
        for output in node.get("outputs", []):
            if isinstance(output, str) and output:
                producers[output] = node
    return producers


def _data_inputs_for_node(node: dict[str, Any]) -> list[str]:
    inputs = [name for name in node.get("inputs", []) if isinstance(name, str) and name]
    op = node.get("op", "")
    if op in {"nn.constant"}:
        return []
    if op in {"nn.reshape", "nn.resize", "nn.slice", "nn.split"}:
        return inputs[:1]
    if op == "nn.conv2d":
        return inputs[:3]
    if op == "nn.conv_transpose2d":
        return inputs[:3]
    return inputs


def _same_static_tensor_types(records: dict[str, dict[str, Any]], names: list[str]) -> str:
    if not names:
        raise GenericGraphIRToMLIRError("internal error: no tensor names supplied")
    tensor_type = _mlir_tensor_type(records[names[0]])
    for name in names[1:]:
        other = _mlir_tensor_type(records[name])
        if other != tensor_type:
            raise GenericGraphIRToMLIRError(
                "emitter v0 requires identical static tensor types for "
                + ", ".join(names)
            )
    return tensor_type


def _broadcast_affine_map(input_dims: list[int], output_dims: list[int], label: str) -> str:
    if len(input_dims) > len(output_dims):
        raise GenericGraphIRToMLIRError(
            f"unsupported broadcast for {label}: input rank exceeds output rank"
        )
    out_rank = len(output_dims)
    input_rank = len(input_dims)
    dims = ", ".join(f"d{i}" for i in range(out_rank))
    results = []
    offset = out_rank - input_rank
    for index, input_dim in enumerate(input_dims):
        output_dim = output_dims[offset + index]
        if input_dim == output_dim:
            results.append(f"d{offset + index}")
        elif input_dim == 1:
            results.append("0")
        else:
            raise GenericGraphIRToMLIRError(
                f"unsupported broadcast for {label}: dimension {input_dim} cannot map to {output_dim}"
            )
    return f"affine_map<({dims}) -> ({', '.join(results)})>"


def _validate_same_elem_type(records: dict[str, dict[str, Any]], names: list[str]) -> str:
    elem_type = _mlir_elem_type(records[names[0]])
    for name in names[1:]:
        other = _mlir_elem_type(records[name])
        if other != elem_type:
            raise GenericGraphIRToMLIRError(
                "emitter v0 requires identical element types for " + ", ".join(names)
            )
    return elem_type


def _format_f32(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise GenericGraphIRToMLIRError(f"non-numeric f32 literal {value!r}")
    return f"{float(value):.6e}"


def _nested_dense_literal(values: list[Any], dims: list[int]) -> str:
    expected = 1
    for dim in dims:
        expected *= dim
    if len(values) != expected:
        raise GenericGraphIRToMLIRError(
            f"literal value count {len(values)} does not match tensor shape element count {expected}"
        )

    def build(offset: int, shape: list[int]) -> tuple[str, int]:
        if not shape:
            return _format_f32(values[offset]), offset + 1
        items = []
        cursor = offset
        for _ in range(shape[0]):
            item, cursor = build(cursor, shape[1:])
            items.append(item)
        return "[" + ", ".join(items) + "]", cursor

    literal, _ = build(0, dims)
    return literal


def _source_comment(node: dict[str, Any]) -> str:
    return (
        f"    // source_node_id={node.get('source_node_id')} "
        f"source_op_type={node.get('source_op_type', '')} "
        f"source_name={node.get('source_name', '')}"
    )


PROVENANCE_CONTRACT = "generic_emitter_source_attrs_v1"

# Op roles for dispatch-unit grouping (Phase 26). Every top-level emitted op
# receives exactly one role; helper ops never become runtime dispatch units.
_ROLE_DISPATCH_ROOT = "dispatch_root"
_ROLE_INTERNAL = "dispatch_internal_compute"
_ROLE_TENSOR_CONTRACT = "tensor_contract_operation"
_ROLE_ALLOCATION = "allocation_helper"
_ROLE_SCALAR = "scalar_helper"
_ROLE_VIEW = "view_operation"

_DEF_LINE_RE = re.compile(r"^    (%[A-Za-z0-9_$.\-]+) = ([a-z_]+\.[a-z0-9_.]+)")


def _escape_attr_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _op_role(mnemonic: str, is_root: bool) -> str:
    if is_root:
        return _ROLE_DISPATCH_ROOT
    if mnemonic in ("tensor.empty", "linalg.fill"):
        return _ROLE_ALLOCATION
    if mnemonic == "arith.constant":
        return _ROLE_SCALAR
    if mnemonic == "tensor.pad":
        return _ROLE_TENSOR_CONTRACT
    if mnemonic == "tensor.insert_slice":
        # Non-final concat inserts perform real writes into the output buffer.
        return _ROLE_INTERNAL
    if mnemonic in (
        "tensor.extract_slice",
        "tensor.collapse_shape",
        "tensor.expand_shape",
        "tensor.cast",
        "tensor.generate",
    ):
        return _ROLE_VIEW
    if mnemonic.startswith("linalg."):
        return _ROLE_INTERNAL
    return "unresolved"


def _provenance_attr_text(node: dict[str, Any], role: str) -> str:
    node_id = node.get("id")
    imported_id = node.get("source_node_id", node_id)
    return (
        f"source.graph_node_id = {node_id} : i64, "
        f"source.imported_node_id = {imported_id} : i64, "
        f'source.op_type = "{_escape_attr_string(str(node.get("source_op_type", "")))}", '
        f'source.generic_op = "{_escape_attr_string(str(node.get("op", "")))}", '
        f'source.onnx_name = "{_escape_attr_string(str(node.get("source_name", "")))}", '
        f'source.dispatch_group = "dg_{node_id}", '
        f'source.op_role = "{role}"'
    )


def _inject_before_last_type_colon(line: str, attr_text: str) -> str:
    index = line.rfind(" : ")
    if index < 0:
        raise GenericGraphIRToMLIRError(
            f"internal error: cannot place provenance attrs on line: {line.strip()}"
        )
    return line[:index] + " {" + attr_text + "}" + line[index:]


def _attach_provenance(
    lines: list[str],
    node: dict[str, Any],
    root_ssa_names: set[str],
) -> list[str]:
    """Attach source.* provenance attrs to every top-level op emitted for one node.

    The emitted-line grammar is fully owned by this file, so placement is
    resolved per op mnemonic. Region-carrying ops (tensor.pad, tensor.generate)
    receive their attr-dict on the region-close line, which is where MLIR
    expects it.
    """
    annotated: list[str] = []
    pending_close: str | None = None  # attr text awaiting a region-close line
    for line in lines:
        if pending_close is not None and line.startswith("    } : "):
            annotated.append("    } {" + pending_close + "}" + line[len("    }"):])
            pending_close = None
            continue
        match = _DEF_LINE_RE.match(line)
        if match is None:
            annotated.append(line)
            continue
        ssa_name, mnemonic = match.group(1), match.group(2)
        role = _op_role(mnemonic, ssa_name in root_ssa_names)
        attr_text = _provenance_attr_text(node, role)
        if mnemonic == "arith.constant":
            annotated.append(
                line.replace("arith.constant ", "arith.constant {" + attr_text + "} ", 1)
            )
        elif mnemonic == "tensor.empty":
            annotated.append(
                line.replace("tensor.empty() : ", "tensor.empty() {" + attr_text + "} : ", 1)
            )
        elif mnemonic == "linalg.fill":
            annotated.append(
                line.replace("linalg.fill ins(", "linalg.fill {" + attr_text + "} ins(", 1)
            )
        elif mnemonic in ("linalg.conv_2d_nchw_fchw", "linalg.pooling_nchw_max"):
            annotated.append(line.replace("} ins(", ", " + attr_text + "} ins(", 1))
        elif mnemonic == "linalg.transpose":
            annotated.append(line + " {" + attr_text + "}")
        elif mnemonic == "linalg.generic":
            if not line.rstrip().endswith("linalg.generic {"):
                raise GenericGraphIRToMLIRError(
                    f"internal error: unexpected linalg.generic line: {line.strip()}"
                )
            annotated.append(line + attr_text + ",")
        elif mnemonic in ("tensor.pad", "tensor.generate"):
            annotated.append(line)
            pending_close = attr_text
        elif mnemonic in (
            "tensor.extract_slice",
            "tensor.insert_slice",
            "tensor.collapse_shape",
            "tensor.expand_shape",
            "tensor.cast",
        ):
            annotated.append(_inject_before_last_type_colon(line, attr_text))
        else:
            raise GenericGraphIRToMLIRError(
                f"internal error: no provenance placement rule for op '{mnemonic}'"
            )
    if pending_close is not None:
        raise GenericGraphIRToMLIRError(
            "internal error: region-close line for provenance attrs not found"
        )
    return annotated


def _argument_roles(graph_ir: dict[str, Any]) -> dict[str, str]:
    """Classify every function argument: model_input, weight, bias, initializer."""
    roles: dict[str, str] = {}
    graph = graph_ir.get("graph", {})
    for name in graph.get("inputs", []):
        if isinstance(name, str):
            roles[name] = "model_input"
    for node in graph_ir.get("nodes", []):
        if node.get("op") in ("nn.conv2d", "nn.conv_transpose2d"):
            inputs = [n for n in node.get("inputs", []) if isinstance(n, str)]
            if len(inputs) >= 2:
                roles.setdefault(inputs[1], "weight")
            if len(inputs) >= 3:
                roles.setdefault(inputs[2], "bias")
    return roles


def _emit_binary(
    node: dict[str, Any],
    records: dict[str, dict[str, Any]],
    ssa: dict[str, str],
    output_ssa: str,
) -> list[str]:
    inputs = node.get("inputs", [])
    outputs = node.get("outputs", [])
    if len(inputs) != 2 or len(outputs) != 1:
        raise GenericGraphIRToMLIRError(f"{node.get('op')} requires two inputs and one output")
    for name in inputs:
        if name not in ssa:
            raise GenericGraphIRToMLIRError(f"input '{name}' has no SSA value")
    elem_type = _validate_same_elem_type(records, [inputs[0], inputs[1], outputs[0]])
    lhs_type = _mlir_tensor_type(records[inputs[0]])
    rhs_type = _mlir_tensor_type(records[inputs[1]])
    output_type = _mlir_tensor_type(records[outputs[0]])
    lhs_dims = _static_dims(records[inputs[0]])
    rhs_dims = _static_dims(records[inputs[1]])
    output_dims = _static_dims(records[outputs[0]])
    rank = len(output_dims)
    lhs_map = _broadcast_affine_map(lhs_dims, output_dims, inputs[0])
    rhs_map = _broadcast_affine_map(rhs_dims, output_dims, inputs[1])
    output_map = _identity_affine_map(rank)
    op = BINARY_ARITH[str(node["op"])]
    suffix = output_ssa[1:]
    return [
        f"    %empty_{output_ssa[1:]} = tensor.empty() : {output_type}",
        f"    {output_ssa} = linalg.generic {{",
        f"      indexing_maps = [{lhs_map}, {rhs_map}, {output_map}],",
        f"      iterator_types = {_iterator_types(rank)}",
        f"    }} ins({ssa[inputs[0]]}, {ssa[inputs[1]]} : {lhs_type}, {rhs_type}) "
        f"outs(%empty_{output_ssa[1:]} : {output_type}) {{",
        f"    ^bb0(%lhs_{suffix}: {elem_type}, %rhs_{suffix}: {elem_type}, %unused_{suffix}: {elem_type}):",
        f"      %result_{suffix} = {op} %lhs_{suffix}, %rhs_{suffix} : {elem_type}",
        f"      linalg.yield %result_{suffix} : {elem_type}",
        f"    }} -> {output_type}",
    ]


def _emit_unary(
    node: dict[str, Any],
    records: dict[str, dict[str, Any]],
    ssa: dict[str, str],
    output_ssa: str,
) -> list[str]:
    inputs = node.get("inputs", [])
    outputs = node.get("outputs", [])
    if len(inputs) != 1 or len(outputs) != 1:
        raise GenericGraphIRToMLIRError(f"{node.get('op')} requires one input and one output")
    if inputs[0] not in ssa:
        raise GenericGraphIRToMLIRError(f"input '{inputs[0]}' has no SSA value")
    tensor_type = _same_static_tensor_types(records, [inputs[0], outputs[0]])
    rank = len(_static_dims(records[outputs[0]]))
    elem_type = _mlir_elem_type(records[outputs[0]])
    affine_map = _identity_affine_map(rank)
    suffix = output_ssa[1:]
    lines = [
        f"    %empty_{output_ssa[1:]} = tensor.empty() : {tensor_type}",
        f"    {output_ssa} = linalg.generic {{",
        f"      indexing_maps = [{affine_map}, {affine_map}],",
        f"      iterator_types = {_iterator_types(rank)}",
        f"    }} ins({ssa[inputs[0]]} : {tensor_type}) outs(%empty_{output_ssa[1:]} : {tensor_type}) {{",
        f"    ^bb0(%input_{suffix}: {elem_type}, %unused_{suffix}: {elem_type}):",
    ]
    if node["op"] == "nn.relu":
        lines.extend([
            f"      %zero_{suffix} = arith.constant 0.000000e+00 : {elem_type}",
            f"      %result_{suffix} = arith.maximumf %input_{suffix}, %zero_{suffix} : {elem_type}",
        ])
    elif node["op"] == "nn.sigmoid":
        lines.extend([
            f"      %zero_{suffix} = arith.constant 0.000000e+00 : {elem_type}",
            f"      %one_{suffix} = arith.constant 1.000000e+00 : {elem_type}",
            f"      %neg_{suffix} = arith.subf %zero_{suffix}, %input_{suffix} : {elem_type}",
            f"      %exp_{suffix} = math.exp %neg_{suffix} : {elem_type}",
            f"      %denominator_{suffix} = arith.addf %one_{suffix}, %exp_{suffix} : {elem_type}",
            f"      %result_{suffix} = arith.divf %one_{suffix}, %denominator_{suffix} : {elem_type}",
        ])
    else:
        raise GenericGraphIRToMLIRError(f"internal error: unsupported unary op {node['op']}")
    lines.extend([
        f"      linalg.yield %result_{suffix} : {elem_type}",
        f"    }} -> {tensor_type}",
    ])
    return lines


def _emit_constant(
    node: dict[str, Any],
    records: dict[str, dict[str, Any]],
    output_ssa: str,
) -> list[str]:
    outputs = node.get("outputs", [])
    if len(outputs) != 1:
        raise GenericGraphIRToMLIRError("nn.constant requires one output")
    record = records.get(outputs[0])
    if record is None:
        raise GenericGraphIRToMLIRError(f"constant output '{outputs[0]}' has no value metadata")
    tensor_type = _mlir_tensor_type(record)
    if _mlir_elem_type(record) != "f32":
        raise GenericGraphIRToMLIRError("emitter v0 only supports f32 constants")
    values = record.get("literal_values")
    if not isinstance(values, list):
        raise GenericGraphIRToMLIRError(f"constant output '{outputs[0]}' has no literal_values")
    literal = _nested_dense_literal(values, _static_dims(record))
    return [f"    {output_ssa} = arith.constant dense<{literal}> : {tensor_type}"]


def _emit_reshape(
    node: dict[str, Any],
    records: dict[str, dict[str, Any]],
    ssa: dict[str, str],
    output_ssa: str,
) -> list[str]:
    inputs = node.get("inputs", [])
    outputs = node.get("outputs", [])
    if len(inputs) not in (1, 2) or len(outputs) != 1:
        raise GenericGraphIRToMLIRError("nn.reshape requires one data input, optional static shape input, and one output")
    data_input = inputs[0]
    if data_input not in ssa:
        raise GenericGraphIRToMLIRError(f"input '{data_input}' has no SSA value")
    elem_type = _validate_same_elem_type(records, [data_input, outputs[0]])
    if elem_type != "f32":
        raise GenericGraphIRToMLIRError("emitter v0 only supports f32 reshape")
    input_dims = _static_dims(records[data_input])
    output_dims = _static_dims(records[outputs[0]])
    attrs = node.get("canonical_attrs")
    if not isinstance(attrs, dict):
        raise GenericGraphIRToMLIRError("nn.reshape missing canonical_attrs")
    resolved_target = _reshape_resolved_target(attrs.get("target_shape"), attrs.get("allowzero", 0), input_dims)
    if resolved_target != output_dims:
        raise GenericGraphIRToMLIRError(
            f"nn.reshape resolved target_shape {resolved_target} does not match output shape {output_dims}"
        )
    if _element_count(input_dims) != _element_count(output_dims):
        raise GenericGraphIRToMLIRError(
            f"nn.reshape element count mismatch: {input_dims} -> {output_dims}"
        )
    input_type = _mlir_tensor_type(records[data_input])
    output_type = _mlir_tensor_type(records[outputs[0]])
    if input_dims == output_dims:
        return [f"    {output_ssa} = tensor.cast {ssa[data_input]} : {input_type} to {output_type}"]
    collapse = _derive_collapse_reassociation(input_dims, output_dims)
    if collapse is not None:
        return [
            f"    {output_ssa} = tensor.collapse_shape {ssa[data_input]} {_format_reassociation(collapse)} : "
            f"{input_type} into {output_type}"
        ]
    expand = _derive_expand_reassociation(input_dims, output_dims)
    if expand is not None:
        output_shape = "[" + ", ".join(str(dim) for dim in output_dims) + "]"
        return [
            f"    {output_ssa} = tensor.expand_shape {ssa[data_input]} {_format_reassociation(expand)} "
            f"output_shape {output_shape} : {input_type} into {output_type}"
        ]
    raise GenericGraphIRToMLIRError(
        f"unsupported nn.reshape reassociation/remapping for emitter v0: {input_dims} -> {output_dims}"
    )


def _conv2d_output_dim(input_dim: int, kernel: int, pad_before: int, pad_after: int, stride: int, dilation: int) -> int:
    return (input_dim + pad_before + pad_after - dilation * (kernel - 1) - 1) // stride + 1


def _emit_conv2d(
    node: dict[str, Any],
    records: dict[str, dict[str, Any]],
    ssa: dict[str, str],
    output_ssa: str,
) -> list[str]:
    inputs = node.get("inputs", [])
    outputs = node.get("outputs", [])
    if len(inputs) not in (2, 3) or len(outputs) != 1:
        raise GenericGraphIRToMLIRError("nn.conv2d requires input, weight, optional bias, and one output")
    input_name, weight_name = inputs[0], inputs[1]
    bias_name = inputs[2] if len(inputs) == 3 else None
    for name in [input_name, weight_name, *( [bias_name] if bias_name else [] )]:
        if name not in ssa:
            raise GenericGraphIRToMLIRError(f"input '{name}' has no SSA value")
    elem_names = [input_name, weight_name, outputs[0]]
    if bias_name:
        elem_names.append(bias_name)
    elem_type = _validate_same_elem_type(records, elem_names)
    if elem_type != "f32":
        raise GenericGraphIRToMLIRError("nn.conv2d emitter supports only f32")

    input_dims = _static_dims(records[input_name])
    weight_dims = _static_dims(records[weight_name])
    output_dims = _static_dims(records[outputs[0]])
    if len(input_dims) != 4 or len(weight_dims) != 4 or len(output_dims) != 4:
        raise GenericGraphIRToMLIRError("nn.conv2d emitter supports only rank-4 NCHW input, FCHW weight, and output")
    attrs = node.get("canonical_attrs")
    if not isinstance(attrs, dict):
        raise GenericGraphIRToMLIRError("nn.conv2d missing canonical_attrs")
    groups = attrs.get("groups", 1)
    if not isinstance(groups, int):
        raise GenericGraphIRToMLIRError("nn.conv2d groups must be an integer")
    if groups != 1:
        raise GenericGraphIRToMLIRError("nn.conv2d emitter supports only groups=1 standard convolution")
    kernel = _static_list(attrs.get("kernel_shape"), "nn.conv2d kernel_shape")
    pads = _static_list(attrs.get("pads"), "nn.conv2d pads")
    strides = _static_list(attrs.get("strides"), "nn.conv2d strides")
    dilations = _static_list(attrs.get("dilations"), "nn.conv2d dilations")
    if len(kernel) != 2 or len(pads) != 4 or len(strides) != 2 or len(dilations) != 2:
        raise GenericGraphIRToMLIRError("nn.conv2d canonical attributes have invalid lengths")
    if any(value <= 0 for value in kernel + strides + dilations):
        raise GenericGraphIRToMLIRError("nn.conv2d kernel_shape, strides, and dilations must be positive")
    if any(value < 0 for value in pads):
        raise GenericGraphIRToMLIRError("nn.conv2d pads must be non-negative")

    n, input_channels, input_h, input_w = input_dims
    output_channels, weight_channels, kh, kw = weight_dims
    out_n, out_c, out_h, out_w = output_dims
    if kernel != [kh, kw]:
        raise GenericGraphIRToMLIRError(
            f"nn.conv2d kernel_shape {kernel} does not match weight spatial shape {[kh, kw]}"
        )
    if input_channels != weight_channels * groups:
        raise GenericGraphIRToMLIRError(
            f"nn.conv2d channel mismatch: input channels {input_channels} != weight channels {weight_channels} * groups {groups}"
        )
    if out_n != n or out_c != output_channels:
        raise GenericGraphIRToMLIRError(
            f"nn.conv2d output N/C shape {output_dims[:2]} does not match expected {[n, output_channels]}"
        )
    pt, pl, pb, pr = pads
    sh, sw = strides
    dh, dw = dilations
    expected_h = _conv2d_output_dim(input_h, kh, pt, pb, sh, dh)
    expected_w = _conv2d_output_dim(input_w, kw, pl, pr, sw, dw)
    if expected_h <= 0 or expected_w <= 0:
        raise GenericGraphIRToMLIRError("nn.conv2d output spatial dimensions must be positive")
    if [out_h, out_w] != [expected_h, expected_w]:
        raise GenericGraphIRToMLIRError(
            f"nn.conv2d output shape {output_dims} does not match convolution result {[n, output_channels, expected_h, expected_w]}"
        )
    if bias_name:
        bias_dims = _static_dims(records[bias_name])
        if bias_dims != [output_channels]:
            raise GenericGraphIRToMLIRError(
                f"nn.conv2d bias shape {bias_dims} must be rank-1 with output channel length {output_channels}"
            )

    input_type = _mlir_tensor_type(records[input_name])
    weight_type = _mlir_tensor_type(records[weight_name])
    output_type = _mlir_tensor_type(records[outputs[0]])
    suffix = output_ssa[1:]
    padded_h = input_h + pt + pb
    padded_w = input_w + pl + pr
    padded_type = f"tensor<{n}x{input_channels}x{padded_h}x{padded_w}x{elem_type}>"
    lines = [
        f"    %zero_{suffix} = arith.constant 0.000000e+00 : {elem_type}",
    ]
    conv_input_ssa = ssa[input_name]
    conv_input_type = input_type
    if pads != [0, 0, 0, 0]:
        lines.extend([
            f"    %padded_{suffix} = tensor.pad {ssa[input_name]} low[0, 0, {pt}, {pl}] high[0, 0, {pb}, {pr}] {{",
            f"    ^bb0(%pad_n_{suffix}: index, %pad_c_{suffix}: index, %pad_h_{suffix}: index, %pad_w_{suffix}: index):",
            f"      tensor.yield %zero_{suffix} : {elem_type}",
            f"    }} : {input_type} to {padded_type}",
        ])
        conv_input_ssa = f"%padded_{suffix}"
        conv_input_type = padded_type
    lines.extend([
        f"    %empty_{suffix} = tensor.empty() : {output_type}",
        f"    %init_{suffix} = linalg.fill ins(%zero_{suffix} : {elem_type}) "
        f"outs(%empty_{suffix} : {output_type}) -> {output_type}",
    ])
    conv_ssa = output_ssa if bias_name is None else f"%conv_{suffix}"
    lines.append(
        f"    {conv_ssa} = linalg.conv_2d_nchw_fchw "
        f"{{dilations = {_dense_i64_vector(dilations)}, strides = {_dense_i64_vector(strides)}}} "
        f"ins({conv_input_ssa}, {ssa[weight_name]} : {conv_input_type}, {weight_type}) "
        f"outs(%init_{suffix} : {output_type}) -> {output_type}"
    )
    if bias_name:
        bias_type = _mlir_tensor_type(records[bias_name])
        identity = _identity_affine_map(4)
        bias_map = "affine_map<(d0, d1, d2, d3) -> (d1)>"
        lines.extend([
            f"    %bias_empty_{suffix} = tensor.empty() : {output_type}",
            f"    {output_ssa} = linalg.generic {{",
            f"      indexing_maps = [{identity}, {bias_map}, {identity}],",
            f"      iterator_types = {_iterator_types(4)}",
            f"    }} ins({conv_ssa}, {ssa[bias_name]} : {output_type}, {bias_type}) "
            f"outs(%bias_empty_{suffix} : {output_type}) {{",
            f"    ^bb0(%conv_value_{suffix}: {elem_type}, %bias_value_{suffix}: {elem_type}, %unused_bias_{suffix}: {elem_type}):",
            f"      %biased_{suffix} = arith.addf %conv_value_{suffix}, %bias_value_{suffix} : {elem_type}",
            f"      linalg.yield %biased_{suffix} : {elem_type}",
            f"    }} -> {output_type}",
        ])
    return lines


def _conv_transpose2d_output_dim(
    input_dim: int,
    kernel: int,
    pad_before: int,
    pad_after: int,
    stride: int,
    dilation: int,
    output_padding: int,
) -> int:
    return stride * (input_dim - 1) + output_padding + dilation * (kernel - 1) + 1 - pad_before - pad_after


def _emit_conv_transpose2d(
    node: dict[str, Any],
    records: dict[str, dict[str, Any]],
    ssa: dict[str, str],
    output_ssa: str,
) -> list[str]:
    inputs = node.get("inputs", [])
    outputs = node.get("outputs", [])
    if len(inputs) not in (2, 3) or len(outputs) != 1:
        raise GenericGraphIRToMLIRError(
            "nn.conv_transpose2d requires input, weight, optional bias, and one output"
        )
    input_name, weight_name = inputs[0], inputs[1]
    bias_name = inputs[2] if len(inputs) == 3 else None
    for name in [input_name, weight_name, *( [bias_name] if bias_name else [] )]:
        if name not in ssa:
            raise GenericGraphIRToMLIRError(f"input '{name}' has no SSA value")
    elem_names = [input_name, weight_name, outputs[0]]
    if bias_name:
        elem_names.append(bias_name)
    elem_type = _validate_same_elem_type(records, elem_names)
    if elem_type != "f32":
        raise GenericGraphIRToMLIRError("nn.conv_transpose2d emitter supports only f32")

    input_dims = _static_dims(records[input_name])
    weight_dims = _static_dims(records[weight_name])
    output_dims = _static_dims(records[outputs[0]])
    if len(input_dims) != 4 or len(weight_dims) != 4 or len(output_dims) != 4:
        raise GenericGraphIRToMLIRError(
            "nn.conv_transpose2d emitter supports only rank-4 NCHW input/output and ONNX IOHW weight"
        )
    attrs = node.get("canonical_attrs")
    if not isinstance(attrs, dict):
        raise GenericGraphIRToMLIRError("nn.conv_transpose2d missing canonical_attrs")
    groups = attrs.get("groups", 1)
    if not isinstance(groups, int):
        raise GenericGraphIRToMLIRError("nn.conv_transpose2d groups must be an integer")
    if groups != 1:
        raise GenericGraphIRToMLIRError("nn.conv_transpose2d emitter supports only groups=1")
    kernel = _static_list(attrs.get("kernel_shape"), "nn.conv_transpose2d kernel_shape")
    pads = _static_list(attrs.get("pads"), "nn.conv_transpose2d pads")
    strides = _static_list(attrs.get("strides"), "nn.conv_transpose2d strides")
    dilations = _static_list(attrs.get("dilations"), "nn.conv_transpose2d dilations")
    output_padding = _static_list(attrs.get("output_padding"), "nn.conv_transpose2d output_padding")
    output_shape = attrs.get("output_shape", [])
    if (
        len(kernel) != 2
        or len(pads) != 4
        or len(strides) != 2
        or len(dilations) != 2
        or len(output_padding) != 2
    ):
        raise GenericGraphIRToMLIRError("nn.conv_transpose2d canonical attributes have invalid lengths")
    if any(value <= 0 for value in kernel + strides + dilations):
        raise GenericGraphIRToMLIRError("nn.conv_transpose2d kernel_shape, strides, and dilations must be positive")
    if any(value < 0 for value in pads + output_padding):
        raise GenericGraphIRToMLIRError("nn.conv_transpose2d pads and output_padding must be non-negative")
    if output_shape not in (None, []):
        raise GenericGraphIRToMLIRError("nn.conv_transpose2d emitter does not support explicit output_shape")
    if kernel != [2, 2] or strides != [2, 2]:
        raise GenericGraphIRToMLIRError("nn.conv_transpose2d emitter supports only kernel_shape=strides=[2,2]")
    if kernel != strides:
        raise GenericGraphIRToMLIRError("nn.conv_transpose2d emitter supports only non-overlapping kernel_shape == strides")
    if dilations != [1, 1]:
        raise GenericGraphIRToMLIRError("nn.conv_transpose2d emitter supports only dilations=[1,1]")
    if pads != [0, 0, 0, 0]:
        raise GenericGraphIRToMLIRError("nn.conv_transpose2d emitter supports only zero pads")
    if output_padding != [0, 0]:
        raise GenericGraphIRToMLIRError("nn.conv_transpose2d emitter supports only zero output_padding")

    n, input_channels, input_h, input_w = input_dims
    weight_input_channels, output_channels_per_group, kh, kw = weight_dims
    out_n, out_c, out_h, out_w = output_dims
    if weight_input_channels != input_channels:
        raise GenericGraphIRToMLIRError(
            f"nn.conv_transpose2d channel mismatch: weight input channels {weight_input_channels} "
            f"must equal input channels {input_channels}"
        )
    if kernel != [kh, kw]:
        raise GenericGraphIRToMLIRError(
            f"nn.conv_transpose2d kernel_shape {kernel} does not match weight spatial shape {[kh, kw]}"
        )
    output_channels = output_channels_per_group * groups
    if out_n != n or out_c != output_channels:
        raise GenericGraphIRToMLIRError(
            f"nn.conv_transpose2d output N/C shape {output_dims[:2]} does not match expected {[n, output_channels]}"
        )
    pt, pl, pb, pr = pads
    sh, sw = strides
    dh, dw = dilations
    oph, opw = output_padding
    expected_h = _conv_transpose2d_output_dim(input_h, kh, pt, pb, sh, dh, oph)
    expected_w = _conv_transpose2d_output_dim(input_w, kw, pl, pr, sw, dw, opw)
    if [out_h, out_w] != [expected_h, expected_w]:
        raise GenericGraphIRToMLIRError(
            f"nn.conv_transpose2d output shape {output_dims} does not match transposed convolution result "
            f"{[n, output_channels, expected_h, expected_w]}"
        )
    if bias_name:
        bias_dims = _static_dims(records[bias_name])
        if bias_dims != [output_channels]:
            raise GenericGraphIRToMLIRError(
                f"nn.conv_transpose2d bias shape {bias_dims} must be rank-1 with output channel length {output_channels}"
            )

    input_type = _mlir_tensor_type(records[input_name])
    weight_type = _mlir_tensor_type(records[weight_name])
    output_type = _mlir_tensor_type(records[outputs[0]])
    suffix = output_ssa[1:]
    identity = _identity_affine_map(4)
    lines: list[str] = []
    if bias_name:
        bias_type = _mlir_tensor_type(records[bias_name])
        bias_map = "affine_map<(n, oc, oh, ow) -> (oc)>"
        lines.extend([
            f"    %empty_{suffix} = tensor.empty() : {output_type}",
            f"    %init_{suffix} = linalg.generic {{",
            f"      indexing_maps = [{bias_map}, {identity}],",
            f"      iterator_types = {_iterator_types(4)}",
            f"    }} ins({ssa[bias_name]} : {bias_type}) outs(%empty_{suffix} : {output_type}) {{",
            f"    ^bb0(%bias_value_{suffix}: {elem_type}, %unused_init_{suffix}: {elem_type}):",
            f"      linalg.yield %bias_value_{suffix} : {elem_type}",
            f"    }} -> {output_type}",
        ])
    else:
        lines.extend([
            f"    %zero_{suffix} = arith.constant 0.000000e+00 : {elem_type}",
            f"    %empty_{suffix} = tensor.empty() : {output_type}",
            f"    %init_{suffix} = linalg.fill ins(%zero_{suffix} : {elem_type}) "
            f"outs(%empty_{suffix} : {output_type}) -> {output_type}",
        ])
    input_map = "affine_map<(n, oc, oh, ow, ic) -> (n, ic, oh floordiv 2, ow floordiv 2)>"
    weight_map = "affine_map<(n, oc, oh, ow, ic) -> (ic, oc, oh mod 2, ow mod 2)>"
    output_map = "affine_map<(n, oc, oh, ow, ic) -> (n, oc, oh, ow)>"
    lines.extend([
        f"    {output_ssa} = linalg.generic {{",
        f"      indexing_maps = [{input_map}, {weight_map}, {output_map}],",
        f"      iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"parallel\", \"reduction\"]",
        f"    }} ins({ssa[input_name]}, {ssa[weight_name]} : {input_type}, {weight_type}) "
        f"outs(%init_{suffix} : {output_type}) {{",
        f"    ^bb0(%input_value_{suffix}: {elem_type}, %weight_value_{suffix}: {elem_type}, %acc_{suffix}: {elem_type}):",
        f"      %product_{suffix} = arith.mulf %input_value_{suffix}, %weight_value_{suffix} : {elem_type}",
        f"      %sum_{suffix} = arith.addf %acc_{suffix}, %product_{suffix} : {elem_type}",
        f"      linalg.yield %sum_{suffix} : {elem_type}",
        f"    }} -> {output_type}",
    ])
    return lines


def _emit_maxpool2d(
    node: dict[str, Any],
    records: dict[str, dict[str, Any]],
    ssa: dict[str, str],
    output_ssa: str,
) -> list[str]:
    inputs = node.get("inputs", [])
    outputs = node.get("outputs", [])
    if len(inputs) != 1 or len(outputs) != 1:
        raise GenericGraphIRToMLIRError("nn.maxpool2d requires one input and one output")
    if inputs[0] not in ssa:
        raise GenericGraphIRToMLIRError(f"input '{inputs[0]}' has no SSA value")
    elem_type = _validate_same_elem_type(records, [inputs[0], outputs[0]])
    if elem_type != "f32":
        raise GenericGraphIRToMLIRError("nn.maxpool2d emitter supports only f32")
    input_dims = _static_dims(records[inputs[0]])
    output_dims = _static_dims(records[outputs[0]])
    if len(input_dims) != 4 or len(output_dims) != 4:
        raise GenericGraphIRToMLIRError("nn.maxpool2d emitter supports only rank-4 NCHW tensors")
    attrs = node.get("canonical_attrs")
    if not isinstance(attrs, dict):
        raise GenericGraphIRToMLIRError("nn.maxpool2d missing canonical_attrs")
    kernel = _static_list(attrs.get("kernel_shape"), "nn.maxpool2d kernel_shape")
    pads = _static_list(attrs.get("pads"), "nn.maxpool2d pads")
    strides = _static_list(attrs.get("strides"), "nn.maxpool2d strides")
    dilations = _static_list(attrs.get("dilations"), "nn.maxpool2d dilations")
    if len(kernel) != 2 or len(pads) != 4 or len(strides) != 2 or len(dilations) != 2:
        raise GenericGraphIRToMLIRError("nn.maxpool2d canonical attributes have invalid lengths")
    if any(value <= 0 for value in kernel + strides + dilations):
        raise GenericGraphIRToMLIRError("nn.maxpool2d kernel_shape, strides, and dilations must be positive")
    if any(value < 0 for value in pads):
        raise GenericGraphIRToMLIRError("nn.maxpool2d pads must be non-negative")
    if attrs.get("ceil_mode", 0) not in (0, False):
        raise GenericGraphIRToMLIRError("nn.maxpool2d emitter supports only ceil_mode=0")

    n, c, h, w = input_dims
    kh, kw = kernel
    pt, pl, pb, pr = pads
    sh, sw = strides
    dh, dw = dilations
    padded_h = h + pt + pb
    padded_w = w + pl + pr
    expected_h = (padded_h - dh * (kh - 1) - 1) // sh + 1
    expected_w = (padded_w - dw * (kw - 1) - 1) // sw + 1
    if output_dims != [n, c, expected_h, expected_w]:
        raise GenericGraphIRToMLIRError(
            f"nn.maxpool2d output shape {output_dims} does not match floor-mode pooling result "
            f"{[n, c, expected_h, expected_w]}"
        )

    input_type = _mlir_tensor_type(records[inputs[0]])
    padded_type = f"tensor<{n}x{c}x{padded_h}x{padded_w}x{elem_type}>"
    kernel_type = f"tensor<{kh}x{kw}x{elem_type}>"
    output_type = _mlir_tensor_type(records[outputs[0]])
    suffix = output_ssa[1:]
    return [
        f"    %neg_inf_{suffix} = arith.constant 0xFF800000 : {elem_type}",
        f"    %padded_{suffix} = tensor.pad {ssa[inputs[0]]} low[0, 0, {pt}, {pl}] high[0, 0, {pb}, {pr}] {{",
        f"    ^bb0(%pad_n_{suffix}: index, %pad_c_{suffix}: index, %pad_h_{suffix}: index, %pad_w_{suffix}: index):",
        f"      tensor.yield %neg_inf_{suffix} : {elem_type}",
        f"    }} : {input_type} to {padded_type}",
        f"    %empty_{suffix} = tensor.empty() : {output_type}",
        f"    %init_{suffix} = linalg.fill ins(%neg_inf_{suffix} : {elem_type}) "
        f"outs(%empty_{suffix} : {output_type}) -> {output_type}",
        f"    %kernel_{suffix} = tensor.empty() : {kernel_type}",
        f"    {output_ssa} = linalg.pooling_nchw_max "
        f"{{dilations = {_dense_i64_vector(dilations)}, strides = {_dense_i64_vector(strides)}}} "
        f"ins(%padded_{suffix}, %kernel_{suffix} : {padded_type}, {kernel_type}) "
        f"outs(%init_{suffix} : {output_type}) -> {output_type}",
    ]


def _emit_softmax(
    node: dict[str, Any],
    records: dict[str, dict[str, Any]],
    ssa: dict[str, str],
    output_ssa: str,
) -> list[str]:
    inputs = node.get("inputs", [])
    outputs = node.get("outputs", [])
    if len(inputs) != 1 or len(outputs) != 1:
        raise GenericGraphIRToMLIRError("nn.softmax requires one input and one output")
    if inputs[0] not in ssa:
        raise GenericGraphIRToMLIRError(f"input '{inputs[0]}' has no SSA value")
    tensor_type = _same_static_tensor_types(records, [inputs[0], outputs[0]])
    elem_type = _mlir_elem_type(records[outputs[0]])
    if elem_type != "f32":
        raise GenericGraphIRToMLIRError("nn.softmax emitter supports only f32")
    dims = _static_dims(records[outputs[0]])
    attrs = node.get("canonical_attrs")
    axis = _normalize_axis(attrs.get("axis") if isinstance(attrs, dict) else None, len(dims), "nn.softmax")
    reduced_dims = [dim for index, dim in enumerate(dims) if index != axis]
    reduced_type = f"tensor<{elem_type}>" if not reduced_dims else "tensor<" + "x".join(str(dim) for dim in reduced_dims) + f"x{elem_type}>"
    identity = _identity_affine_map(len(dims))
    reduced_map = _drop_axis_affine_map(len(dims), axis)
    suffix = output_ssa[1:]
    return [
        f"    %neg_inf_{suffix} = arith.constant 0xFF800000 : {elem_type}",
        f"    %zero_{suffix} = arith.constant 0.000000e+00 : {elem_type}",
        f"    %max_empty_{suffix} = tensor.empty() : {reduced_type}",
        f"    %max_init_{suffix} = linalg.fill ins(%neg_inf_{suffix} : {elem_type}) "
        f"outs(%max_empty_{suffix} : {reduced_type}) -> {reduced_type}",
        f"    %max_{suffix} = linalg.generic {{",
        f"      indexing_maps = [{identity}, {reduced_map}],",
        f"      iterator_types = {_softmax_iterator_types(len(dims), axis)}",
        f"    }} ins({ssa[inputs[0]]} : {tensor_type}) outs(%max_init_{suffix} : {reduced_type}) {{",
        f"    ^bb0(%input_max_{suffix}: {elem_type}, %acc_max_{suffix}: {elem_type}):",
        f"      %new_max_{suffix} = arith.maximumf %input_max_{suffix}, %acc_max_{suffix} : {elem_type}",
        f"      linalg.yield %new_max_{suffix} : {elem_type}",
        f"    }} -> {reduced_type}",
        f"    %exp_empty_{suffix} = tensor.empty() : {tensor_type}",
        f"    %exp_{suffix} = linalg.generic {{",
        f"      indexing_maps = [{identity}, {reduced_map}, {identity}],",
        f"      iterator_types = {_iterator_types(len(dims))}",
        f"    }} ins({ssa[inputs[0]]}, %max_{suffix} : {tensor_type}, {reduced_type}) "
        f"outs(%exp_empty_{suffix} : {tensor_type}) {{",
        f"    ^bb0(%input_exp_{suffix}: {elem_type}, %max_value_{suffix}: {elem_type}, %unused_exp_{suffix}: {elem_type}):",
        f"      %shifted_{suffix} = arith.subf %input_exp_{suffix}, %max_value_{suffix} : {elem_type}",
        f"      %exp_value_{suffix} = math.exp %shifted_{suffix} : {elem_type}",
        f"      linalg.yield %exp_value_{suffix} : {elem_type}",
        f"    }} -> {tensor_type}",
        f"    %sum_empty_{suffix} = tensor.empty() : {reduced_type}",
        f"    %sum_init_{suffix} = linalg.fill ins(%zero_{suffix} : {elem_type}) "
        f"outs(%sum_empty_{suffix} : {reduced_type}) -> {reduced_type}",
        f"    %sum_{suffix} = linalg.generic {{",
        f"      indexing_maps = [{identity}, {reduced_map}],",
        f"      iterator_types = {_softmax_iterator_types(len(dims), axis)}",
        f"    }} ins(%exp_{suffix} : {tensor_type}) outs(%sum_init_{suffix} : {reduced_type}) {{",
        f"    ^bb0(%exp_sum_{suffix}: {elem_type}, %acc_sum_{suffix}: {elem_type}):",
        f"      %new_sum_{suffix} = arith.addf %acc_sum_{suffix}, %exp_sum_{suffix} : {elem_type}",
        f"      linalg.yield %new_sum_{suffix} : {elem_type}",
        f"    }} -> {reduced_type}",
        f"    %softmax_empty_{suffix} = tensor.empty() : {tensor_type}",
        f"    {output_ssa} = linalg.generic {{",
        f"      indexing_maps = [{identity}, {reduced_map}, {identity}],",
        f"      iterator_types = {_iterator_types(len(dims))}",
        f"    }} ins(%exp_{suffix}, %sum_{suffix} : {tensor_type}, {reduced_type}) "
        f"outs(%softmax_empty_{suffix} : {tensor_type}) {{",
        f"    ^bb0(%exp_div_{suffix}: {elem_type}, %sum_value_{suffix}: {elem_type}, %unused_div_{suffix}: {elem_type}):",
        f"      %result_{suffix} = arith.divf %exp_div_{suffix}, %sum_value_{suffix} : {elem_type}",
        f"      linalg.yield %result_{suffix} : {elem_type}",
        f"    }} -> {tensor_type}",
    ]


def _emit_transpose(
    node: dict[str, Any],
    records: dict[str, dict[str, Any]],
    ssa: dict[str, str],
    output_ssa: str,
) -> list[str]:
    inputs = node.get("inputs", [])
    outputs = node.get("outputs", [])
    if len(inputs) != 1 or len(outputs) != 1:
        raise GenericGraphIRToMLIRError("nn.transpose requires one input and one output")
    if inputs[0] not in ssa:
        raise GenericGraphIRToMLIRError(f"input '{inputs[0]}' has no SSA value")
    _validate_same_elem_type(records, [inputs[0], outputs[0]])
    input_dims = _static_dims(records[inputs[0]])
    output_dims = _static_dims(records[outputs[0]])
    attrs = node.get("canonical_attrs")
    perm = attrs.get("perm") if isinstance(attrs, dict) else None
    if not isinstance(perm, list) or sorted(perm) != list(range(len(input_dims))):
        raise GenericGraphIRToMLIRError(f"nn.transpose has invalid permutation {perm!r}")
    expected_output = [input_dims[index] for index in perm]
    if expected_output != output_dims:
        raise GenericGraphIRToMLIRError(
            f"nn.transpose output shape {output_dims} does not match permutation result {expected_output}"
        )
    input_type = _mlir_tensor_type(records[inputs[0]])
    output_type = _mlir_tensor_type(records[outputs[0]])
    perm_text = "[" + ", ".join(str(index) for index in perm) + "]"
    return [
        f"    %empty_{output_ssa[1:]} = tensor.empty() : {output_type}",
        f"    {output_ssa} = linalg.transpose ins({ssa[inputs[0]]} : {input_type}) "
        f"outs(%empty_{output_ssa[1:]} : {output_type}) permutation = {perm_text}",
    ]


def _emit_resize(
    node: dict[str, Any],
    records: dict[str, dict[str, Any]],
    ssa: dict[str, str],
    output_ssa: str,
) -> list[str]:
    inputs = node.get("inputs", [])
    outputs = node.get("outputs", [])
    if len(inputs) < 1 or len(outputs) != 1:
        raise GenericGraphIRToMLIRError("nn.resize requires at least one input and one output")
    if inputs[0] not in ssa:
        raise GenericGraphIRToMLIRError(f"input '{inputs[0]}' has no SSA value")
    attrs = node.get("canonical_attrs")
    if not isinstance(attrs, dict):
        raise GenericGraphIRToMLIRError("nn.resize missing canonical_attrs")
    if attrs.get("mode") != "nearest":
        raise GenericGraphIRToMLIRError("nn.resize emitter supports only nearest mode")
    if attrs.get("coordinate_transformation_mode") != "asymmetric":
        raise GenericGraphIRToMLIRError("nn.resize emitter supports only asymmetric coordinate transformation")
    if attrs.get("nearest_mode") != "floor":
        raise GenericGraphIRToMLIRError("nn.resize emitter supports only floor nearest mode")
    if attrs.get("scales") != [1.0, 1.0, 2.0, 2.0]:
        raise GenericGraphIRToMLIRError("nn.resize emitter supports only static NCHW [1,1,2,2] scales")
    elem_type = _validate_same_elem_type(records, [inputs[0], outputs[0]])
    if elem_type != "f32":
        raise GenericGraphIRToMLIRError("emitter v0 only supports f32 resize")
    input_dims = _static_dims(records[inputs[0]])
    output_dims = _static_dims(records[outputs[0]])
    if len(input_dims) != 4 or len(output_dims) != 4:
        raise GenericGraphIRToMLIRError("nn.resize emitter supports only rank-4 NCHW tensors")
    if output_dims != [input_dims[0], input_dims[1], input_dims[2] * 2, input_dims[3] * 2]:
        raise GenericGraphIRToMLIRError(
            f"nn.resize output shape {output_dims} is not NCHW 2x spatial resize of {input_dims}"
        )
    input_type = _mlir_tensor_type(records[inputs[0]])
    output_type = _mlir_tensor_type(records[outputs[0]])
    return [
        f"    %two_{output_ssa[1:]} = arith.constant 2 : index",
        f"    {output_ssa} = tensor.generate {{",
        f"    ^bb0(%n_{output_ssa[1:]}: index, %c_{output_ssa[1:]}: index, "
        f"%oh_{output_ssa[1:]}: index, %ow_{output_ssa[1:]}: index):",
        f"      %ih_{output_ssa[1:]} = arith.divui %oh_{output_ssa[1:]}, %two_{output_ssa[1:]} : index",
        f"      %iw_{output_ssa[1:]} = arith.divui %ow_{output_ssa[1:]}, %two_{output_ssa[1:]} : index",
        f"      %value_{output_ssa[1:]} = tensor.extract {ssa[inputs[0]]}"
        f"[%n_{output_ssa[1:]}, %c_{output_ssa[1:]}, %ih_{output_ssa[1:]}, %iw_{output_ssa[1:]}] : {input_type}",
        f"      tensor.yield %value_{output_ssa[1:]} : {elem_type}",
        f"    }} : {output_type}",
    ]


def _emit_slice(
    node: dict[str, Any],
    records: dict[str, dict[str, Any]],
    ssa: dict[str, str],
    output_ssa: str,
) -> list[str]:
    inputs = node.get("inputs", [])
    outputs = node.get("outputs", [])
    if len(inputs) < 1 or len(outputs) != 1:
        raise GenericGraphIRToMLIRError("nn.slice requires at least one input and one output")
    if inputs[0] not in ssa:
        raise GenericGraphIRToMLIRError(f"input '{inputs[0]}' has no SSA value")
    _validate_same_elem_type(records, [inputs[0], outputs[0]])
    input_dims = _static_dims(records[inputs[0]])
    output_dims = _static_dims(records[outputs[0]])
    attrs = node.get("canonical_attrs")
    if not isinstance(attrs, dict):
        raise GenericGraphIRToMLIRError("nn.slice missing canonical_attrs")
    starts = _static_list(attrs.get("starts"), "nn.slice starts")
    ends = _static_list(attrs.get("ends"), "nn.slice ends")
    axes_raw = attrs.get("axes")
    axes = (
        list(range(len(starts)))
        if axes_raw is None
        else [_normalize_axis(axis, len(input_dims), "nn.slice") for axis in _static_list(axes_raw, "nn.slice axes")]
    )
    steps_raw = attrs.get("steps")
    steps = [1] * len(starts) if steps_raw is None else _static_list(steps_raw, "nn.slice steps")
    if not (len(starts) == len(ends) == len(axes) == len(steps)):
        raise GenericGraphIRToMLIRError("nn.slice starts/ends/axes/steps lengths are incompatible")
    offsets = [0] * len(input_dims)
    sizes = list(input_dims)
    strides = [1] * len(input_dims)
    for start, end, axis, step in zip(starts, ends, axes, steps):
        if step != 1:
            raise GenericGraphIRToMLIRError("nn.slice emitter supports only unit positive steps")
        if start < 0 or end < 0:
            raise GenericGraphIRToMLIRError("nn.slice emitter does not support negative starts/ends")
        if end < start or end > input_dims[axis]:
            raise GenericGraphIRToMLIRError("nn.slice bounds are inconsistent with input shape")
        offsets[axis] = start
        sizes[axis] = end - start
        strides[axis] = step
    if sizes != output_dims:
        raise GenericGraphIRToMLIRError(
            f"nn.slice inferred output shape {output_dims} does not match slice sizes {sizes}"
        )
    input_type = _mlir_tensor_type(records[inputs[0]])
    output_type = _mlir_tensor_type(records[outputs[0]])
    return [
        f"    {output_ssa} = tensor.extract_slice {ssa[inputs[0]]}"
        f"{_list_text(offsets)} {_list_text(sizes)} {_list_text(strides)} : {input_type} to {output_type}"
    ]


def _split_sizes(node: dict[str, Any], input_dim: int, output_count: int) -> list[int]:
    attrs = node.get("canonical_attrs")
    if not isinstance(attrs, dict):
        raise GenericGraphIRToMLIRError("nn.split missing canonical_attrs")
    split = attrs.get("split")
    if split is not None:
        sizes = _static_list(split, "nn.split split")
        if len(sizes) != output_count:
            raise GenericGraphIRToMLIRError("nn.split split size count must match output count")
    else:
        if output_count <= 0 or input_dim % output_count != 0:
            raise GenericGraphIRToMLIRError("nn.split equal split is not statically divisible")
        sizes = [input_dim // output_count] * output_count
    if sum(sizes) != input_dim:
        raise GenericGraphIRToMLIRError("nn.split sizes do not sum to input dimension")
    return sizes


def _emit_split(
    node: dict[str, Any],
    records: dict[str, dict[str, Any]],
    ssa: dict[str, str],
    output_ssas: list[str],
) -> list[str]:
    inputs = node.get("inputs", [])
    outputs = node.get("outputs", [])
    if len(inputs) < 1 or len(outputs) < 1:
        raise GenericGraphIRToMLIRError("nn.split requires one input and at least one output")
    if len(output_ssas) != len(outputs):
        raise GenericGraphIRToMLIRError("internal error: split SSA/output count mismatch")
    if inputs[0] not in ssa:
        raise GenericGraphIRToMLIRError(f"input '{inputs[0]}' has no SSA value")
    _validate_same_elem_type(records, [inputs[0], *outputs])
    input_dims = _static_dims(records[inputs[0]])
    attrs = node.get("canonical_attrs")
    axis = _normalize_axis(attrs.get("axis") if isinstance(attrs, dict) else None, len(input_dims), "nn.split")
    sizes_along_axis = _split_sizes(node, input_dims[axis], len(outputs))
    input_type = _mlir_tensor_type(records[inputs[0]])
    offsets = [0] * len(input_dims)
    strides = [1] * len(input_dims)
    running = 0
    lines: list[str] = []
    for output_name, output_ssa, split_size in zip(outputs, output_ssas, sizes_along_axis):
        output_dims = _static_dims(records[output_name])
        expected_dims = list(input_dims)
        expected_dims[axis] = split_size
        if output_dims != expected_dims:
            raise GenericGraphIRToMLIRError(
                f"nn.split output shape {output_dims} does not match expected {expected_dims}"
            )
        offsets[axis] = running
        output_type = _mlir_tensor_type(records[output_name])
        lines.append(
            f"    {output_ssa} = tensor.extract_slice {ssa[inputs[0]]}"
            f"{_list_text(offsets)} {_list_text(output_dims)} {_list_text(strides)} : {input_type} to {output_type}"
        )
        running += split_size
    return lines


def _emit_concat(
    node: dict[str, Any],
    records: dict[str, dict[str, Any]],
    ssa: dict[str, str],
    output_ssa: str,
) -> list[str]:
    inputs = node.get("inputs", [])
    outputs = node.get("outputs", [])
    if len(inputs) < 1 or len(outputs) != 1:
        raise GenericGraphIRToMLIRError("nn.concat requires at least one input and one output")
    for input_name in inputs:
        if input_name not in ssa:
            raise GenericGraphIRToMLIRError(f"input '{input_name}' has no SSA value")
    _validate_same_elem_type(records, [*inputs, outputs[0]])
    output_dims = _static_dims(records[outputs[0]])
    rank = len(output_dims)
    attrs = node.get("canonical_attrs")
    axis = _normalize_axis(attrs.get("axis") if isinstance(attrs, dict) else None, rank, "nn.concat")
    input_dims_list = [_static_dims(records[name]) for name in inputs]
    running = 0
    for dims in input_dims_list:
        if len(dims) != rank:
            raise GenericGraphIRToMLIRError("nn.concat input ranks are incompatible")
        for dim_index, (input_dim, output_dim) in enumerate(zip(dims, output_dims)):
            if dim_index == axis:
                continue
            if input_dim != output_dim:
                raise GenericGraphIRToMLIRError("nn.concat non-axis dimensions are incompatible")
        running += dims[axis]
    if running != output_dims[axis]:
        raise GenericGraphIRToMLIRError("nn.concat accumulated axis size does not match output shape")
    output_type = _mlir_tensor_type(records[outputs[0]])
    lines = [f"    %empty_{output_ssa[1:]} = tensor.empty() : {output_type}"]
    offsets = [0] * rank
    strides = [1] * rank
    running = 0
    current = f"%empty_{output_ssa[1:]}"
    for index, (input_name, input_dims) in enumerate(zip(inputs, input_dims_list)):
        offsets[axis] = running
        result_ssa = output_ssa if index == len(inputs) - 1 else f"%concat_{output_ssa[1:]}_{index}"
        input_type = _mlir_tensor_type(records[input_name])
        lines.append(
            f"    {result_ssa} = tensor.insert_slice {ssa[input_name]} into {current}"
            f"{_list_text(offsets)} {_list_text(input_dims)} {_list_text(strides)} : "
            f"{input_type} into {output_type}"
        )
        current = result_ssa
        running += input_dims[axis]
    return lines


def _validate_emitter_subset(graph_ir: dict[str, Any]) -> None:
    report = check_generic_lowering_contract.check_lowering_contract(graph_ir)
    if report["contract_status"] == "invalid_generic_graph_ir":
        raise GenericGraphIRToMLIRError(
            "GenericGraphIR verifier failed: " + "; ".join(report["verifier"]["errors"])
        )
    if report["contract_status"] != "ready_for_existing_mlir_lowering":
        reasons = []
        for node in report["blocking_nodes"]:
            reasons.append(f"{node.get('id')}:{node.get('op')}({', '.join(node.get('reasons', []))})")
        raise GenericGraphIRToMLIRError(
            "lowering contract is not ready: " + report["contract_status"] + " " + "; ".join(reasons)
        )


def emit_mlir(
    graph_ir: dict[str, Any],
    allow_partial: bool = False,
    model_artifact: str | None = None,
) -> str:
    _validate_emitter_subset(graph_ir)
    records = _record_map(graph_ir)
    producers = _node_by_output(graph_ir)
    argument_roles = _argument_roles(graph_ir)
    graph = graph_ir.get("graph", {})
    graph_inputs = list(graph.get("inputs", []))
    graph_outputs = list(graph.get("outputs", []))
    initializer_names = {
        record["name"]
        for record in graph_ir.get("initializers", [])
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    external_arg_names = list(graph_inputs)
    for node in graph_ir.get("nodes", []):
        for input_name in _data_inputs_for_node(node):
            if (
                input_name in initializer_names
                and input_name not in producers
                and input_name not in external_arg_names
            ):
                external_arg_names.append(input_name)

    used_ssa: set[str] = set()
    ssa: dict[str, str] = {}
    arg_parts = []
    for arg_index, input_name in enumerate(external_arg_names):
        record = records.get(input_name)
        if record is None:
            raise GenericGraphIRToMLIRError(f"graph input '{input_name}' has no value metadata")
        value = _ssa_name(input_name, used_ssa)
        ssa[input_name] = value
        arg_role = argument_roles.get(input_name, "initializer")
        arg_attrs = (
            f'source.name = "{_escape_attr_string(input_name)}", '
            f'source.arg_role = "{arg_role}", '
            f"source.arg_index = {arg_index} : i64"
        )
        arg_parts.append(f"{value}: {_mlir_tensor_type(record)} {{{arg_attrs}}}")

    return_types = []
    for output_name in graph_outputs:
        record = records.get(output_name)
        if record is None:
            raise GenericGraphIRToMLIRError(f"graph output '{output_name}' has no value metadata")
        return_types.append(_mlir_tensor_type(record))

    module_attrs = [f'source.provenance_contract = "{PROVENANCE_CONTRACT}"']
    if model_artifact:
        module_attrs.append(
            f'source.model_artifact = "{_escape_attr_string(model_artifact)}"'
        )
    lines = [
        "// Generated from GenericGraphIR v0.",
        f"// truth_boundary={TRUTH_BOUNDARY_COMMENT}",
        "module attributes {" + ", ".join(module_attrs) + "} {",
        f"  func.func @{_sanitize_symbol(str(graph.get('name', 'graph')))}("
        + ", ".join(arg_parts)
        + ") -> "
        + (return_types[0] if len(return_types) == 1 else "(" + ", ".join(return_types) + ")")
        + " {",
    ]

    for node in graph_ir.get("nodes", []):
        op = node.get("op", "")
        if op not in SUPPORTED_EMITTER_OPS:
            message = f"emitter v0 does not support op '{op}' at node {node.get('id')}"
            if not allow_partial:
                raise GenericGraphIRToMLIRError(message)
            lines.append(f"    // unsupported: {message}")
            continue
        lines.append(_source_comment(node))
        outputs = list(node.get("outputs", []))
        if op == "nn.identity":
            if len(node.get("inputs", [])) != 1 or len(outputs) != 1:
                raise GenericGraphIRToMLIRError("nn.identity requires one input and one output")
            input_name = node["inputs"][0]
            if input_name not in ssa:
                raise GenericGraphIRToMLIRError(f"input '{input_name}' has no SSA value")
            _same_static_tensor_types(records, [input_name, outputs[0]])
            ssa[outputs[0]] = ssa[input_name]
            lines.append(f"    // nn.identity forwards {ssa[input_name]} as {outputs[0]}")
            continue

        if op == "nn.split":
            if not outputs:
                raise GenericGraphIRToMLIRError("nn.split requires at least one output")
            output_ssas = [_ssa_name(output, used_ssa) for output in outputs]
            lines.extend(
                _attach_provenance(
                    _emit_split(node, records, ssa, output_ssas), node, set(output_ssas)
                )
            )
            for output_name, output_ssa in zip(outputs, output_ssas):
                ssa[output_name] = output_ssa
            continue

        if len(outputs) != 1:
            raise GenericGraphIRToMLIRError(f"emitter v0 requires one output for op '{op}'")
        output_ssa = _ssa_name(outputs[0], used_ssa)
        if op == "nn.constant":
            node_lines = _emit_constant(node, records, output_ssa)
        elif op == "nn.conv2d":
            node_lines = _emit_conv2d(node, records, ssa, output_ssa)
        elif op == "nn.conv_transpose2d":
            node_lines = _emit_conv_transpose2d(node, records, ssa, output_ssa)
        elif op in BINARY_ARITH:
            node_lines = _emit_binary(node, records, ssa, output_ssa)
        elif op in {"nn.relu", "nn.sigmoid"}:
            node_lines = _emit_unary(node, records, ssa, output_ssa)
        elif op == "nn.reshape":
            node_lines = _emit_reshape(node, records, ssa, output_ssa)
        elif op == "nn.maxpool2d":
            node_lines = _emit_maxpool2d(node, records, ssa, output_ssa)
        elif op == "nn.softmax":
            node_lines = _emit_softmax(node, records, ssa, output_ssa)
        elif op == "nn.transpose":
            node_lines = _emit_transpose(node, records, ssa, output_ssa)
        elif op == "nn.resize":
            node_lines = _emit_resize(node, records, ssa, output_ssa)
        elif op == "nn.slice":
            node_lines = _emit_slice(node, records, ssa, output_ssa)
        elif op == "nn.concat":
            node_lines = _emit_concat(node, records, ssa, output_ssa)
        else:
            raise GenericGraphIRToMLIRError(f"internal error: unsupported op '{op}'")
        lines.extend(_attach_provenance(node_lines, node, {output_ssa}))
        ssa[outputs[0]] = output_ssa

    missing_returns = [name for name in graph_outputs if name not in ssa and name in producers]
    if missing_returns:
        raise GenericGraphIRToMLIRError(
            "graph outputs were not emitted: " + ", ".join(missing_returns)
        )
    return_values = []
    for output_name in graph_outputs:
        if output_name in ssa:
            return_values.append(ssa[output_name])
        elif output_name in graph_inputs:
            return_values.append(ssa[output_name])
        else:
            raise GenericGraphIRToMLIRError(f"graph output '{output_name}' has no SSA value")
    lines.append("    return " + ", ".join(return_values) + " : " + ", ".join(return_types))
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Emit comments for unsupported nodes when possible.")
    parser.add_argument("--model-artifact", default=None,
                        help="Model artifact reference (e.g. models/yolo-seg.onnx) "
                             "stamped as a module attribute for runtime weight binding.")
    args = parser.parse_args()

    try:
        graph_ir = json.loads(args.input_path.read_text(encoding="utf-8"))
        if not isinstance(graph_ir, dict):
            raise GenericGraphIRToMLIRError("root JSON value must be an object")
        mlir_text = emit_mlir(graph_ir, allow_partial=args.allow_partial,
                              model_artifact=args.model_artifact)
    except (OSError, json.JSONDecodeError, GenericGraphIRToMLIRError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(mlir_text, encoding="utf-8")
    print(f"generic_graph_ir_to_mlir: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
