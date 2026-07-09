#!/usr/bin/env python3
"""Infer simple shape/type metadata for canonicalized GenericGraphIR v0.

This pass is model-agnostic. It does not perform domain recognition, target
planning, or lowering.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import verify_graph_ir

INFERENCE_VERSION = "0.1.0"
TRUTH_BOUNDARY = "generic_graph_ir_static_shape_type_inference_v0_no_domain_recognition"


class GenericGraphIRShapeInferenceError(Exception):
    """Raised when the input graph is not a valid GenericGraphIR document."""


def _dim_record(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"kind": "unknown", "value": None}
    if isinstance(value, int):
        return {"kind": "static", "value": value}
    if isinstance(value, str):
        return {"kind": "symbolic", "value": value}
    return {"kind": "unknown", "value": None}


def _shape_from_dims(dims: list[Any]) -> list[dict[str, Any]]:
    return [_dim_record(dim) for dim in dims]


def _dims_from_shape(shape: Any) -> list[Any] | None:
    if not isinstance(shape, list):
        return None
    dims: list[Any] = []
    for dim in shape:
        if not isinstance(dim, dict):
            return None
        value = dim.get("value")
        if dim.get("kind") == "static" and isinstance(value, int):
            dims.append(value)
        elif dim.get("kind") == "symbolic" and isinstance(value, str):
            dims.append(value)
        else:
            dims.append(None)
    return dims


def _is_known_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _status_for_dims(dims: list[Any] | None, notes: list[str]) -> str:
    if dims is None:
        return "unknown"
    if any(dim is None or isinstance(dim, str) for dim in dims):
        return "partially_inferred"
    if notes:
        return "partially_inferred"
    return "inferred"


def _output_record(name: str, dtype: str, dims: list[Any] | None) -> dict[str, Any]:
    return {
        "name": name,
        "source_name": name,
        "dtype": dtype or "unknown",
        "shape": _shape_from_dims(dims or []),
    }


def _value_map(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {value["name"]: value for value in graph.get("values", []) if isinstance(value, dict)}


def _initializer_map(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        init["name"]: init
        for init in graph.get("initializers", [])
        if isinstance(init, dict)
    }


def _lookup_tensor(name: str, values: dict[str, Any], initializers: dict[str, Any]) -> dict[str, Any] | None:
    return values.get(name) or initializers.get(name)


def _input_tensors(node: dict[str, Any], values: dict[str, Any], initializers: dict[str, Any]) -> list[dict[str, Any] | None]:
    return [
        _lookup_tensor(name, values, initializers) if isinstance(name, str) and name else None
        for name in node.get("inputs", [])
    ]


def _tensor_dims(tensor: dict[str, Any] | None) -> list[Any] | None:
    if not tensor:
        return None
    return _dims_from_shape(tensor.get("shape"))


def _tensor_dtype(tensor: dict[str, Any] | None) -> str:
    if not tensor:
        return "unknown"
    dtype = tensor.get("dtype")
    return dtype if isinstance(dtype, str) else "unknown"


def _broadcast_dim(a: Any, b: Any) -> Any:
    if a == 1:
        return b
    if b == 1:
        return a
    if _is_known_int(a) and _is_known_int(b):
        if a == b:
            return a
        raise ValueError(f"incompatible broadcast dimensions {a} and {b}")
    if a == b:
        return a
    return None


def _broadcast_shapes(a: list[Any], b: list[Any]) -> list[Any]:
    out: list[Any] = []
    max_rank = max(len(a), len(b))
    pa = [1] * (max_rank - len(a)) + list(a)
    pb = [1] * (max_rank - len(b)) + list(b)
    for da, db in zip(pa, pb):
        out.append(_broadcast_dim(da, db))
    return out


def _conv2d(node: dict[str, Any], tensors: list[dict[str, Any] | None]) -> tuple[list[Any] | None, str, list[str]]:
    notes: list[str] = []
    x_dims = _tensor_dims(tensors[0] if len(tensors) > 0 else None)
    w_dims = _tensor_dims(tensors[1] if len(tensors) > 1 else None)
    dtype = _tensor_dtype(tensors[0] if tensors else None)
    attrs = node.get("canonical_attrs", {})
    if x_dims is None or w_dims is None:
        return None, dtype, ["missing input or weight shape"]
    if len(x_dims) != 4 or len(w_dims) != 4:
        raise ValueError("nn.conv2d expects rank-4 input and weight tensors")

    pads = attrs.get("pads", [0, 0, 0, 0])
    strides = attrs.get("strides", [1, 1])
    dilations = attrs.get("dilations", [1, 1])
    groups = attrs.get("groups", 1)
    kernel_shape = attrs.get("kernel_shape") or w_dims[2:4]
    if len(pads) != 4 or len(strides) != 2 or len(dilations) != 2 or len(kernel_shape) != 2:
        raise ValueError("nn.conv2d canonical attributes have invalid lengths")

    n, c, h, width = x_dims
    out_channels, in_per_group, _, _ = w_dims
    if _is_known_int(c) and _is_known_int(in_per_group) and _is_known_int(groups):
        if c != in_per_group * groups:
            raise ValueError("nn.conv2d input channels do not match weight channels * groups")

    def out_dim(input_dim: Any, k: Any, pad0: Any, pad1: Any, stride: Any, dilation: Any) -> Any:
        if all(_is_known_int(v) for v in [input_dim, k, pad0, pad1, stride, dilation]):
            return math.floor((input_dim + pad0 + pad1 - dilation * (k - 1) - 1) / stride + 1)
        return None

    oh = out_dim(h, kernel_shape[0], pads[0], pads[2], strides[0], dilations[0])
    ow = out_dim(width, kernel_shape[1], pads[1], pads[3], strides[1], dilations[1])
    return [n, out_channels, oh, ow], dtype, notes


def _maxpool2d(node: dict[str, Any], tensors: list[dict[str, Any] | None]) -> tuple[list[Any] | None, str, list[str]]:
    dims = _tensor_dims(tensors[0] if tensors else None)
    dtype = _tensor_dtype(tensors[0] if tensors else None)
    attrs = node.get("canonical_attrs", {})
    if dims is None:
        return None, dtype, ["missing input shape"]
    if len(dims) != 4:
        raise ValueError("nn.maxpool2d expects rank-4 input tensor")

    kernel_shape = attrs.get("kernel_shape", [])
    pads = attrs.get("pads", [0, 0, 0, 0])
    strides = attrs.get("strides", [1, 1])
    dilations = attrs.get("dilations", [1, 1])
    ceil_mode = attrs.get("ceil_mode", 0)
    if len(kernel_shape) != 2 or len(pads) != 4 or len(strides) != 2 or len(dilations) != 2:
        raise ValueError("nn.maxpool2d canonical attributes have invalid lengths")

    n, c, h, width = dims

    def out_dim(input_dim: Any, k: Any, pad0: Any, pad1: Any, stride: Any, dilation: Any) -> Any:
        if all(_is_known_int(v) for v in [input_dim, k, pad0, pad1, stride, dilation]):
            value = (input_dim + pad0 + pad1 - dilation * (k - 1) - 1) / stride + 1
            return math.ceil(value) if ceil_mode else math.floor(value)
        return None

    return [
        n,
        c,
        out_dim(h, kernel_shape[0], pads[0], pads[2], strides[0], dilations[0]),
        out_dim(width, kernel_shape[1], pads[1], pads[3], strides[1], dilations[1]),
    ], dtype, []


def _conv_transpose2d(node: dict[str, Any], tensors: list[dict[str, Any] | None]) -> tuple[list[Any] | None, str, list[str]]:
    x_dims = _tensor_dims(tensors[0] if len(tensors) > 0 else None)
    w_dims = _tensor_dims(tensors[1] if len(tensors) > 1 else None)
    dtype = _tensor_dtype(tensors[0] if tensors else None)
    attrs = node.get("canonical_attrs", {})
    if x_dims is None or w_dims is None:
        return None, dtype, ["missing input or weight shape"]
    if len(x_dims) != 4 or len(w_dims) != 4:
        raise ValueError("nn.conv_transpose2d expects rank-4 input and weight tensors")

    pads = attrs.get("pads", [0, 0, 0, 0])
    strides = attrs.get("strides", [1, 1])
    dilations = attrs.get("dilations", [1, 1])
    groups = attrs.get("groups", 1)
    output_padding = attrs.get("output_padding", [0, 0])
    output_shape = attrs.get("output_shape", [])
    kernel_shape = attrs.get("kernel_shape") or w_dims[2:4]
    if (
        len(pads) != 4
        or len(strides) != 2
        or len(dilations) != 2
        or len(output_padding) != 2
        or len(kernel_shape) != 2
    ):
        raise ValueError("nn.conv_transpose2d canonical attributes have invalid lengths")

    n, _, h, width = x_dims
    _, out_channels_per_group, _, _ = w_dims
    out_channels = out_channels_per_group * groups if _is_known_int(out_channels_per_group) and _is_known_int(groups) else None

    if len(output_shape) >= 2:
        oh, ow = output_shape[-2], output_shape[-1]
    else:
        def out_dim(input_dim: Any, k: Any, pad0: Any, pad1: Any, stride: Any, dilation: Any, out_pad: Any) -> Any:
            if all(_is_known_int(v) for v in [input_dim, k, pad0, pad1, stride, dilation, out_pad]):
                return stride * (input_dim - 1) + out_pad + dilation * (k - 1) + 1 - pad0 - pad1
            return None

        oh = out_dim(h, kernel_shape[0], pads[0], pads[2], strides[0], dilations[0], output_padding[0])
        ow = out_dim(width, kernel_shape[1], pads[1], pads[3], strides[1], dilations[1], output_padding[1])

    return [n, out_channels, oh, ow], dtype, []


def _eltwise(node: dict[str, Any], tensors: list[dict[str, Any] | None]) -> tuple[list[Any] | None, str, list[str]]:
    if len(tensors) < 2:
        return None, _tensor_dtype(tensors[0] if tensors else None), ["missing second input"]
    lhs = _tensor_dims(tensors[0])
    rhs = _tensor_dims(tensors[1])
    if lhs is None or rhs is None:
        return None, _tensor_dtype(tensors[0]), ["missing input shape"]
    return _broadcast_shapes(lhs, rhs), _tensor_dtype(tensors[0]), []


def _matmul(node: dict[str, Any], tensors: list[dict[str, Any] | None]) -> tuple[list[Any] | None, str, list[str]]:
    if len(tensors) < 2:
        return None, _tensor_dtype(tensors[0] if tensors else None), ["missing rhs input"]
    a = _tensor_dims(tensors[0])
    b = _tensor_dims(tensors[1])
    if a is None or b is None:
        return None, _tensor_dtype(tensors[0]), ["missing input shape"]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("nn.matmul expects rank >= 2 inputs")
    if _is_known_int(a[-1]) and _is_known_int(b[-2]) and a[-1] != b[-2]:
        raise ValueError("nn.matmul inner dimensions are incompatible")
    batch = _broadcast_shapes(a[:-2], b[:-2]) if (len(a) > 2 or len(b) > 2) else []
    return batch + [a[-2], b[-1]], _tensor_dtype(tensors[0]), []


def _gemm(node: dict[str, Any], tensors: list[dict[str, Any] | None]) -> tuple[list[Any] | None, str, list[str]]:
    if len(tensors) < 2:
        return None, _tensor_dtype(tensors[0] if tensors else None), ["missing rhs input"]
    a = _tensor_dims(tensors[0])
    b = _tensor_dims(tensors[1])
    if a is None or b is None:
        return None, _tensor_dtype(tensors[0]), ["missing input shape"]
    if len(a) != 2 or len(b) != 2:
        raise ValueError("nn.gemm expects rank-2 lhs and rhs inputs")
    attrs = node.get("canonical_attrs", {})
    a_rows, a_cols = (a[1], a[0]) if attrs.get("transA", 0) else (a[0], a[1])
    b_rows, b_cols = (b[1], b[0]) if attrs.get("transB", 0) else (b[0], b[1])
    if _is_known_int(a_cols) and _is_known_int(b_rows) and a_cols != b_rows:
        raise ValueError("nn.gemm inner dimensions are incompatible")
    return [a_rows, b_cols], _tensor_dtype(tensors[0]), []


def _reshape(node: dict[str, Any], tensors: list[dict[str, Any] | None]) -> tuple[list[Any] | None, str, list[str]]:
    attrs = node.get("canonical_attrs", {})
    target = attrs.get("shape") or attrs.get("target_shape")
    input_dims = _tensor_dims(tensors[0] if tensors else None)
    dtype = _tensor_dtype(tensors[0] if tensors else None)
    if target is None:
        return None, dtype, ["reshape target shape unavailable"]
    if not isinstance(target, list):
        raise ValueError("nn.reshape target shape must be a list")
    if any(not _is_known_int(dim) for dim in target):
        raise ValueError("nn.reshape target shape must contain integers")

    allowzero = attrs.get("allowzero", 0)
    if allowzero not in (0, 1):
        raise ValueError("nn.reshape allowzero must be 0 or 1")
    if sum(dim == -1 for dim in target) > 1:
        raise ValueError("nn.reshape target shape may contain at most one -1 dimension")
    if any(dim < -1 for dim in target):
        raise ValueError("nn.reshape target shape contains an invalid negative dimension")
    if allowzero == 1 and -1 in target and 0 in target:
        raise ValueError("nn.reshape cannot combine allowzero=1, zero dimensions, and -1")

    resolved = list(target)
    for index, dim in enumerate(resolved):
        if dim != 0 or allowzero == 1:
            continue
        if input_dims is None or index >= len(input_dims):
            resolved[index] = None
        else:
            resolved[index] = input_dims[index]

    infer_index = resolved.index(-1) if -1 in resolved else None
    if infer_index is not None:
        known_input_size = (
            math.prod(input_dims)
            if input_dims is not None and all(_is_known_int(dim) and dim >= 0 for dim in input_dims)
            else None
        )
        explicit_dims = [dim for index, dim in enumerate(resolved) if index != infer_index]
        known_output_size = (
            math.prod(explicit_dims)
            if all(_is_known_int(dim) and dim >= 0 for dim in explicit_dims)
            else None
        )
        if known_input_size is None or known_output_size is None:
            resolved[infer_index] = None
        elif known_output_size == 0 or known_input_size % known_output_size != 0:
            raise ValueError("nn.reshape input size is incompatible with target shape")
        else:
            resolved[infer_index] = known_input_size // known_output_size
    elif (
        input_dims is not None
        and all(_is_known_int(dim) and dim >= 0 for dim in input_dims)
        and all(_is_known_int(dim) and dim >= 0 for dim in resolved)
        and math.prod(input_dims) != math.prod(resolved)
    ):
        raise ValueError("nn.reshape input size is incompatible with target shape")

    notes = ["reshape inferred dimension is unresolved"] if any(dim is None for dim in resolved) else []
    return resolved, dtype, notes


def _transpose(node: dict[str, Any], tensors: list[dict[str, Any] | None]) -> tuple[list[Any] | None, str, list[str]]:
    dims = _tensor_dims(tensors[0] if tensors else None)
    if dims is None:
        return None, _tensor_dtype(tensors[0] if tensors else None), ["missing input shape"]
    perm = node.get("canonical_attrs", {}).get("perm") or list(reversed(range(len(dims))))
    if len(perm) != len(dims) or sorted(perm) != list(range(len(dims))):
        raise ValueError("nn.transpose perm is incompatible with input rank")
    return [dims[i] for i in perm], _tensor_dtype(tensors[0]), []


def _concat(node: dict[str, Any], tensors: list[dict[str, Any] | None]) -> tuple[list[Any] | None, str, list[str]]:
    shapes = [_tensor_dims(tensor) for tensor in tensors if tensor is not None]
    if not shapes or any(shape is None for shape in shapes):
        return None, _tensor_dtype(tensors[0] if tensors else None), ["missing input shape"]
    rank = len(shapes[0])
    axis = node.get("canonical_attrs", {}).get("axis", 0)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError("nn.concat axis is outside input rank")
    for shape in shapes:
        if len(shape) != rank:
            raise ValueError("nn.concat input ranks are incompatible")
    out = list(shapes[0])
    concat_total: int | None = 0
    for shape in shapes:
        for i, dim in enumerate(shape):
            if i == axis:
                continue
            if _is_known_int(out[i]) and _is_known_int(dim) and out[i] != dim:
                raise ValueError("nn.concat non-axis dimensions are incompatible")
            if out[i] != dim:
                out[i] = None
        axis_dim = shape[axis]
        if _is_known_int(axis_dim) and concat_total is not None:
            concat_total += axis_dim
        else:
            concat_total = None
    out[axis] = concat_total
    return out, _tensor_dtype(tensors[0]), []


def _split_output_dims(node: dict[str, Any], tensors: list[dict[str, Any] | None], output_count: int) -> tuple[list[list[Any]] | None, str, list[str]]:
    dims = _tensor_dims(tensors[0] if tensors else None)
    dtype = _tensor_dtype(tensors[0] if tensors else None)
    attrs = node.get("canonical_attrs", {})
    if dims is None:
        return None, dtype, ["missing input shape"]
    if output_count == 0:
        return [], dtype, []
    rank = len(dims)
    axis = attrs.get("axis", 0)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError("nn.split axis is outside input rank")

    split = attrs.get("split")
    if isinstance(split, list) and len(split) == output_count:
        outputs = []
        for size in split:
            out = list(dims)
            out[axis] = size
            outputs.append(out)
        return outputs, dtype, []

    axis_dim = dims[axis]
    if _is_known_int(axis_dim) and axis_dim % output_count == 0:
        split_size = axis_dim // output_count
        outputs = []
        for _ in range(output_count):
            out = list(dims)
            out[axis] = split_size
            outputs.append(out)
        return outputs, dtype, []

    outputs = []
    for _ in range(output_count):
        out = list(dims)
        out[axis] = None
        outputs.append(out)
    return outputs, dtype, ["split sizes unavailable"]


def _slice(node: dict[str, Any], tensors: list[dict[str, Any] | None]) -> tuple[list[Any] | None, str, list[str]]:
    dims = _tensor_dims(tensors[0] if tensors else None)
    dtype = _tensor_dtype(tensors[0] if tensors else None)
    attrs = node.get("canonical_attrs", {})
    if dims is None:
        return None, dtype, ["missing input shape"]

    starts = attrs.get("starts")
    ends = attrs.get("ends")
    if not isinstance(starts, list) or not isinstance(ends, list):
        return list(dims), dtype, ["slice starts/ends unavailable"]
    axes = attrs.get("axes")
    if not isinstance(axes, list) or not axes:
        axes = list(range(len(starts)))
    steps = attrs.get("steps")
    if not isinstance(steps, list) or not steps:
        steps = [1] * len(starts)
    if not (len(starts) == len(ends) == len(axes) == len(steps)):
        raise ValueError("nn.slice starts/ends/axes/steps lengths are incompatible")

    out = list(dims)
    notes = []
    rank = len(dims)
    for start, end, axis, step in zip(starts, ends, axes, steps):
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ValueError("nn.slice axis is outside input rank")
        if not _is_known_int(step) or step == 0:
            raise ValueError("nn.slice step must be a non-zero integer")
        dim = dims[axis]
        if not all(_is_known_int(v) for v in [dim, start, end]) or step < 0:
            out[axis] = None
            notes.append("slice dimension partially inferred")
            continue
        clamped_start = min(max(start if start >= 0 else dim + start, 0), dim)
        clamped_end = min(max(end if end >= 0 else dim + end, 0), dim)
        if clamped_end <= clamped_start:
            out[axis] = 0
        else:
            out[axis] = math.ceil((clamped_end - clamped_start) / step)
    return out, dtype, notes


def _resize(node: dict[str, Any], tensors: list[dict[str, Any] | None]) -> tuple[list[Any] | None, str, list[str]]:
    attrs = node.get("canonical_attrs", {})
    if isinstance(attrs.get("sizes"), list):
        return list(attrs["sizes"]), _tensor_dtype(tensors[0] if tensors else None), []
    dims = _tensor_dims(tensors[0] if tensors else None)
    if dims is None:
        return None, _tensor_dtype(tensors[0] if tensors else None), ["missing input shape"]
    if isinstance(attrs.get("scales"), list) and len(attrs["scales"]) == len(dims):
        out = []
        for dim, scale in zip(dims, attrs["scales"]):
            if _is_known_int(dim) and isinstance(scale, (int, float)):
                out.append(int(math.floor(dim * scale)))
            else:
                out.append(None)
        return out, _tensor_dtype(tensors[0]), []
    return dims, _tensor_dtype(tensors[0]), ["resize sizes/scales unavailable; preserving input rank"]


def _unary(node: dict[str, Any], tensors: list[dict[str, Any] | None]) -> tuple[list[Any] | None, str, list[str]]:
    return _tensor_dims(tensors[0] if tensors else None), _tensor_dtype(tensors[0] if tensors else None), []


INFER_RULES = {
    "nn.conv2d": _conv2d,
    "nn.conv_transpose2d": _conv_transpose2d,
    "nn.maxpool2d": _maxpool2d,
    "nn.add": _eltwise,
    "nn.sub": _eltwise,
    "nn.mul": _eltwise,
    "nn.div": _eltwise,
    "nn.matmul": _matmul,
    "nn.gemm": _gemm,
    "nn.reshape": _reshape,
    "nn.transpose": _transpose,
    "nn.concat": _concat,
    "nn.slice": _slice,
    "nn.resize": _resize,
    "nn.softmax": _unary,
    "nn.sigmoid": _unary,
    "nn.relu": _unary,
    "nn.identity": _unary,
}


def infer_generic_graph_shapes(generic: dict[str, Any]) -> dict[str, Any]:
    result = verify_graph_ir.verify_generic_graph_ir(generic)
    if not result.passed:
        raise GenericGraphIRShapeInferenceError("; ".join(result.errors))

    out = copy.deepcopy(generic)
    values = _value_map(out)
    initializers = _initializer_map(out)

    for node in out["nodes"]:
        op = node.get("op")
        output_names = [name for name in node.get("outputs", []) if isinstance(name, str) and name]
        tensors = _input_tensors(node, values, initializers)
        notes: list[str] = []
        status = "unknown"
        inferred_outputs: list[dict[str, Any]] = []

        if op == "nn.unknown":
            notes.append("no inference rule for nn.unknown")
        elif op == "nn.constant":
            for name in output_names:
                existing = values.get(name)
                if existing:
                    record = copy.deepcopy(existing)
                    inferred_outputs.append(record)
                    values[name] = record
            status = "inferred" if inferred_outputs else "unknown"
            if not inferred_outputs:
                notes.append("constant output metadata unavailable")
        elif op == "nn.split":
            try:
                output_dims, dtype, rule_notes = _split_output_dims(node, tensors, len(output_names))
                notes.extend(rule_notes)
                status = _status_for_dims(
                    None if output_dims is None else [dim for dims in output_dims for dim in dims],
                    notes,
                )
                if output_dims is not None:
                    for name, dims in zip(output_names, output_dims):
                        record = _output_record(name, dtype, dims)
                        inferred_outputs.append(record)
                        values[name] = record
            except ValueError as exc:
                status = "error"
                notes.append(str(exc))
        elif op in INFER_RULES:
            try:
                dims, dtype, rule_notes = INFER_RULES[op](node, tensors)
                notes.extend(rule_notes)
                status = _status_for_dims(dims, notes)
                for name in output_names:
                    record = _output_record(name, dtype, dims)
                    inferred_outputs.append(record)
                    values[name] = record
            except ValueError as exc:
                status = "error"
                notes.append(str(exc))
        else:
            notes.append(f"no inference rule for {op}")

        if not inferred_outputs and status != "error":
            for name in output_names:
                existing = values.get(name)
                if existing:
                    inferred_outputs.append(copy.deepcopy(existing))

        node["shape_inference_status"] = status
        node["inferred_outputs"] = inferred_outputs
        node["shape_inference_notes"] = notes

    out["values"] = list(values.values())
    provenance = out.setdefault("provenance", {})
    provenance["shape_inference_version"] = INFERENCE_VERSION
    provenance["shape_inference_truth_boundary"] = TRUTH_BOUNDARY
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True, type=Path,
                        help="Path to canonicalized GenericGraphIR JSON")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output GenericGraphIR JSON with shape/type annotations")
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"error: --in path does not exist: {args.input_path}", file=sys.stderr)
        return 1

    try:
        generic = json.loads(args.input_path.read_text(encoding="utf-8"))
        inferred = infer_generic_graph_shapes(generic)
        verify_result = verify_graph_ir.verify_generic_graph_ir(inferred)
        if not verify_result.passed:
            raise GenericGraphIRShapeInferenceError("; ".join(verify_result.errors))
    except (json.JSONDecodeError, GenericGraphIRShapeInferenceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inferred, indent=2) + "\n", encoding="utf-8")
    print(f"infer_generic_graph_shapes: wrote {args.out}")
    print(f"  graph: {inferred['graph']['name']}")
    print(f"  nodes: {len(inferred['nodes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
