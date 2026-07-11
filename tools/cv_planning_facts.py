#!/usr/bin/env python3
"""Build planner-facing CV semantic/tensor facts from annotated upstream MLIR.

This is an analysis/reporting utility. It does not select a backend, choose a
kernel, allocate memory slots, generate an ExecutionPlan, or require legacy
cv.* operations.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TRUTH_BOUNDARY = (
    "cv_planning_facts_only_no_backend_selection_no_kernel_selection_"
    "no_memory_slot_assignment_no_execution_plan_generation_no_measured_performance"
)

TENSOR_RE = re.compile(r"tensor<([^>]+)>")
VALUE_RE = re.compile(r"%[A-Za-z0-9_.$-]+")
ATTR_RE = re.compile(
    r'cv\.([A-Za-z_][\w.]*)\s*=\s*("[^"]*"|\[[^\]]*\]|[0-9]+|true|false)'
)
OP_RE = re.compile(r"\b(func|tensor|linalg|arith|math)\.[A-Za-z_][\w]*\b")


@dataclass
class TensorType:
    shape: list[int | str]
    dtype: str


@dataclass
class CVTensorPlanningFact:
    tensor_id: str
    shape: list[int | str]
    dtype: str
    layout: str
    byte_size: int | None
    producer: str | None
    consumers: list[str]
    is_graph_input: bool = False
    is_graph_output: bool = False
    is_initializer: bool = False
    is_temporary: bool = False
    semantic_role: str | None = None
    lifetime_start: int | None = None
    lifetime_end: int | None = None
    ownership: str = "compiler_temporary"
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class CVOutputPlanningFact:
    tensor_id: str
    output_role: str
    shape: list[int | str]
    dtype: str
    producer_region: str | None
    postprocess_required: bool
    postprocess_boundary: str | None
    ownership_expectation: str
    layout: str


@dataclass
class CVRegionPlanningFact:
    region_id: str
    semantic_role: str
    recognition_confidence: str
    operation_count: int
    operation_ids: list[str]
    input_tensor_ids: list[str]
    output_tensor_ids: list[str]
    input_shapes: list[list[int | str]]
    output_shapes: list[list[int | str]]
    dominant_dtype: str | None
    feature_scales: list[str]
    estimated_flops: int
    estimated_read_bytes: int
    estimated_write_bytes: int
    estimated_weight_bytes: int
    estimated_temporary_bytes: int
    candidate_execution_domains: list[dict[str, Any]]
    quantization_eligibility: dict[str, Any]
    fusion_eligibility: dict[str, Any]
    planning_notes: list[str]


@dataclass
class CVModelPlanningFacts:
    model_family: str
    function_name: str
    graph_input_count: int
    graph_output_count: int
    regions: list[CVRegionPlanningFact]
    outputs: list[CVOutputPlanningFact]
    tensors: list[CVTensorPlanningFact]
    operation_summary: dict[str, Any]
    unresolved_facts: list[dict[str, Any]]
    provenance: dict[str, Any]
    memory_summary: dict[str, Any]
    cost_summary: dict[str, Any]
    truth_boundary: str = TRUTH_BOUNDARY


@dataclass
class OperationFact:
    op_id: str
    op_name: str
    position: int
    line: int
    result_ids: list[str]
    operand_ids: list[str]
    result_types: list[TensorType]
    operand_types: list[TensorType]
    attrs: dict[str, Any]
    body: str
    source_provenance: dict[str, Any]
    costs: dict[str, int | str | bool]


def parse_attr_value(value: str) -> Any:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        return re.findall(r'"([^"]*)"', value)
    if value.isdigit():
        return int(value)
    if value == "true":
        return True
    if value == "false":
        return False
    return value


def parse_tensor_type(text: str) -> TensorType | None:
    match = TENSOR_RE.search(text)
    if not match:
        return None
    body = match.group(1)
    parts = body.split("x")
    if not parts:
        return None
    dtype = parts[-1]
    shape: list[int | str] = []
    for dim in parts[:-1]:
        if dim == "?":
            shape.append("?")
        elif dim:
            try:
                shape.append(int(dim))
            except ValueError:
                shape.append(dim)
    return TensorType(shape=shape, dtype=dtype)


def parse_all_tensor_types(text: str) -> list[TensorType]:
    result: list[TensorType] = []
    for match in TENSOR_RE.finditer(text):
        parsed = parse_tensor_type("tensor<" + match.group(1) + ">")
        if parsed:
            result.append(parsed)
    return result


def dtype_bytes(dtype: str) -> int | None:
    return {"f32": 4, "float": 4, "f16": 2, "bf16": 2, "i8": 1, "int8": 1}.get(dtype)


def element_count(shape: list[int | str]) -> int | None:
    total = 1
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            return None
        total *= dim
    return total


def tensor_bytes(t: TensorType | None) -> int | None:
    if t is None:
        return None
    elems = element_count(t.shape)
    width = dtype_bytes(t.dtype)
    if elems is None or width is None:
        return None
    return elems * width


def infer_layout(t: TensorType | None) -> str:
    if t is None:
        return "unknown"
    if len(t.shape) == 4:
        return "NCHW"
    if len(t.shape) == 3:
        return "NCX"
    if len(t.shape) == 1:
        return "C"
    if len(t.shape) == 0:
        return "scalar"
    return "ranked_unknown_layout"


def split_top_level_args(text: str) -> list[str]:
    args: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in "(<[{":
            depth += 1
        elif ch in ")>]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            args.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail:
        args.append(tail)
    return args


def extract_function_signature(text: str) -> tuple[str, list[tuple[str, TensorType]], list[TensorType], dict[str, Any]]:
    match = re.search(
        r"func\.func\s+@([A-Za-z0-9_.$-]+)\((.*?)\)\s*->\s*\((.*?)\)\s*(?:attributes\s*\{(.*?)\})?\s*\{",
        text,
        re.S,
    )
    if not match:
        raise ValueError("could not find func.func signature")
    name = match.group(1)
    args_text = match.group(2)
    returns_text = match.group(3)
    attrs_text = match.group(4) or ""
    args: list[tuple[str, TensorType]] = []
    for item in split_top_level_args(args_text):
        m = re.match(r"(%[A-Za-z0-9_.$-]+)\s*:\s*(tensor<[^>]+>)", item)
        if not m:
            continue
        parsed = parse_tensor_type(m.group(2))
        if parsed:
            args.append((m.group(1), parsed))
    returns = [t for t in parse_all_tensor_types(returns_text)]
    attrs = {f"cv.{k}": parse_attr_value(v) for k, v in ATTR_RE.findall(attrs_text)}
    return name, args, returns, attrs


def collect_operation_blocks(text: str) -> list[tuple[int, str]]:
    lines = text.splitlines()
    starts: list[int] = []
    for i, line in enumerate(lines):
        if re.match(r"^    %[A-Za-z0-9_.$-]+(?:\s*,\s*%[A-Za-z0-9_.$-]+)*\s*=", line):
            starts.append(i)
        elif re.match(r"^    (func\.)?return\b", line):
            starts.append(i)
    blocks: list[tuple[int, str]] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        blocks.append((start + 1, "\n".join(lines[start:end])))
    return blocks


def source_comments_before(text: str) -> dict[int, dict[str, Any]]:
    comments: dict[int, dict[str, Any]] = {}
    latest: dict[str, Any] | None = None
    for i, line in enumerate(text.splitlines(), start=1):
        m = re.search(r"source_node_id=([^\s]+)\s+source_op_type=([^\s]+)\s+source_name=(.*)$", line)
        if m:
            latest = {
                "source_node_id": m.group(1),
                "source_op_type": m.group(2),
                "source_name": m.group(3).strip(),
            }
            continue
        if latest and re.match(r"\s*%[A-Za-z0-9_.$-]+", line):
            comments[i] = latest
            latest = None
    return comments


def parse_operations(text: str) -> list[OperationFact]:
    comments = source_comments_before(text)
    ops: list[OperationFact] = []
    for pos, (line, block) in enumerate(collect_operation_blocks(text)):
        first = block.strip().splitlines()[0]
        op_match = OP_RE.search(first)
        if first.startswith("func.return") or first.startswith("return"):
            op_name = "func.return"
            results: list[str] = []
        elif op_match:
            op_name = op_match.group(0)
            lhs = first.split("=", 1)[0]
            results = VALUE_RE.findall(lhs)
        else:
            continue
        result_types: list[TensorType] = []
        arrow = re.search(r"->\s*(tensor<[^>]+>|\([^)]+\))", block, re.S)
        if arrow:
            result_types = parse_all_tensor_types(arrow.group(1))
        elif op_name == "tensor.empty":
            result_types = parse_all_tensor_types(block)
        elif results:
            all_types = parse_all_tensor_types(block)
            if len(all_types) >= len(results):
                result_types = all_types[-len(results):]

        rhs = first.split("=", 1)[1] if "=" in first else first
        operand_ids = [v for v in VALUE_RE.findall(rhs) if v not in results]
        attrs = {f"cv.{k}": parse_attr_value(v) for k, v in ATTR_RE.findall(block)}
        operand_types = parse_all_tensor_types(re.sub(r"->.*", "", block, flags=re.S))
        ops.append(
            OperationFact(
                op_id=f"op_{pos:04d}",
                op_name=op_name,
                position=pos,
                line=line,
                result_ids=[normalize_value_id(v) for v in results],
                operand_ids=[normalize_value_id(v) for v in operand_ids],
                result_types=result_types,
                operand_types=operand_types,
                attrs=attrs,
                body=block,
                source_provenance=comments.get(line, {"source": "unavailable_or_stripped_by_mlir_opt"}),
                costs={},
            )
        )
    return ops


def normalize_value_id(value: str) -> str:
    return value[1:] if value.startswith("%") else value


def estimate_operation_cost(op: OperationFact) -> dict[str, int | str | bool]:
    out = op.result_types[0] if op.result_types else None
    out_elems = element_count(out.shape) if out else None
    out_bytes = tensor_bytes(out) or 0
    input_bytes = sum(tensor_bytes(t) or 0 for t in op.operand_types)
    weight_bytes = 0
    flops = 0
    comparisons = 0
    moved_bytes = 0
    materializing = bool(op.result_types)
    kind = "unknown"

    if op.op_name == "linalg.conv_2d_nchw_fchw" and len(op.operand_types) >= 2 and out:
        kind = "conv2d"
        inp, weight = op.operand_types[0], op.operand_types[1]
        weight_bytes = tensor_bytes(weight) or 0
        if len(out.shape) == 4 and len(weight.shape) == 4 and all(isinstance(d, int) for d in out.shape + weight.shape):
            n, f, oh, ow = out.shape  # type: ignore[misc]
            _, c, kh, kw = weight.shape  # type: ignore[misc]
            macs = n * f * oh * ow * c * kh * kw
            flops = 2 * macs
        input_bytes = tensor_bytes(inp) or 0
    elif op.op_name == "linalg.pooling_nchw_max" and out and len(op.operand_types) >= 1:
        kind = "pooling"
        kernel = [int(x) for x in re.findall(r"window_dimensions\s*=\s*dense<\[?([0-9,\s]+)\]?", op.body)[:1] for x in x.replace(",", " ").split()]
        k = math.prod(kernel) if kernel else 25
        if out_elems is not None:
            comparisons = out_elems * max(k - 1, 0)
            flops = comparisons
    elif op.op_name == "linalg.generic" and out:
        if "math.exp" in op.body or ("arith.divf" in op.body and "softmax" in str(op.attrs).lower()):
            kind = "softmax_or_exp_elementwise"
            flops = (out_elems or 0) * 5
        elif "arith.addf" in op.body or "arith.mulf" in op.body or "arith.subf" in op.body or "arith.divf" in op.body:
            kind = "elementwise"
            body_ops = sum(op.body.count(token) for token in ["arith.addf", "arith.mulf", "arith.subf", "arith.divf", "math.exp"])
            flops = (out_elems or 0) * max(body_ops, 1)
        else:
            kind = "generic"
            flops = out_elems or 0
    elif op.op_name in {"tensor.collapse_shape", "tensor.expand_shape"}:
        kind = "reshape_view_like"
        moved_bytes = 0
        materializing = False
    elif op.op_name in {"tensor.extract_slice"}:
        kind = "slice_view_like"
        moved_bytes = out_bytes
        materializing = False
    elif op.op_name in {"tensor.insert_slice"}:
        kind = "concat_or_slice_update"
        moved_bytes = input_bytes + out_bytes
    elif op.op_name == "tensor.generate":
        kind = "resize_or_generate"
        moved_bytes = input_bytes + out_bytes
    elif op.op_name == "tensor.pad":
        kind = "pad"
        moved_bytes = input_bytes + out_bytes
    elif op.op_name == "tensor.empty":
        kind = "allocation_marker"
    elif op.op_name == "linalg.fill":
        kind = "fill"
        moved_bytes = out_bytes
    elif op.op_name == "func.return":
        kind = "return"

    if moved_bytes == 0 and kind not in {"reshape_view_like", "slice_view_like", "return"}:
        moved_bytes = input_bytes + out_bytes

    return {
        "kind": kind,
        "flops": flops,
        "comparisons": comparisons,
        "read_bytes": input_bytes,
        "write_bytes": out_bytes,
        "weight_bytes": weight_bytes,
        "moved_bytes": moved_bytes,
        "materializing_in_current_ir": materializing,
    }


def domain_candidates(op_names: set[str], semantic_role: str, dtype: str | None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if dtype not in {None, "f32"}:
        return [{"domain": "unsupported_for_current_target_profile", "reason": f"unsupported dtype {dtype}"}]
    tensor_compute = any(name.startswith("linalg.") for name in op_names)
    view_like = op_names and all(name.startswith("tensor.") for name in op_names)
    if tensor_compute:
        candidates.append({"domain": "accelerator_candidate", "reason": "static f32 tensor compute in upstream linalg"})
        candidates.append({"domain": "cpu_candidate", "reason": "portable upstream linalg/tensor lowering remains available"})
    if view_like:
        candidates.append({"domain": "transfer_or_view_operation", "reason": "region contains tensor data movement or view-like operations"})
        candidates.append({"domain": "cpu_candidate", "reason": "portable host materialization is available if needed"})
    if "output" in semantic_role:
        candidates.append({"domain": "host_postprocess_candidate", "reason": "graph output boundary may feed external CV postprocess"})
    if not candidates:
        candidates.append({"domain": "cpu_candidate", "reason": "conservative default for static upstream operation region"})
    return candidates


def quantization_eligibility(op_names: set[str], semantic_role: str, dtype: str | None) -> dict[str, Any]:
    if dtype != "f32":
        return {"status": "unknown", "reason": f"dtype {dtype or 'unknown'} has no Phase 23 quant policy"}
    if "detection_output" in semantic_role or "segmentation_prototype" in semantic_role:
        return {"status": "unknown", "reason": "model outputs require explicit output precision policy"}
    if any(name == "linalg.conv_2d_nchw_fchw" for name in op_names):
        return {"status": "eligible", "reason": "static f32 convolution region may be quantization candidate"}
    if any(name == "linalg.generic" for name in op_names):
        return {"status": "unknown", "reason": "generic elementwise ops need pattern-specific precision policy"}
    return {"status": "ineligible", "reason": "view/data-movement region has no arithmetic quantization target"}


def fusion_eligibility(op_names: list[str], semantic_role: str) -> dict[str, Any]:
    names = set(op_names)
    patterns: list[str] = []
    if "linalg.conv_2d_nchw_fchw" in names and "linalg.generic" in names:
        patterns.append("conv_plus_bias_or_activation_candidate")
    if op_names.count("linalg.generic") >= 2:
        patterns.append("elementwise_chain_candidate")
    if {"tensor.collapse_shape", "tensor.extract_slice"} & names:
        patterns.append("reshape_slice_view_chain_candidate")
    if "tensor.generate" in names and "tensor.insert_slice" in names:
        patterns.append("resize_concat_boundary_candidate")
    if patterns:
        return {"status": "eligible", "candidate_patterns": patterns, "reason": "static topology exposes conservative fusion candidates"}
    return {"status": "unknown", "candidate_patterns": [], "reason": f"no Phase 23 fusion pattern for {semantic_role}"}


def load_shape_ir(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def shape_from_ir_dim_list(dims: list[dict[str, Any]]) -> list[int | str]:
    result: list[int | str] = []
    for dim in dims:
        if dim.get("kind") == "static":
            result.append(int(dim["value"]))
        else:
            result.append("?")
    return result


def build_facts(mlir_text: str, shape_ir: dict[str, Any] | None = None, input_path: str | None = None) -> CVModelPlanningFacts:
    shape_ir = shape_ir or {}
    function_name, args, returns, func_attrs = extract_function_signature(mlir_text)
    ops = parse_operations(mlir_text)
    for op in ops:
        op.costs = estimate_operation_cost(op)

    graph = shape_ir.get("graph", {})
    initializer_records = shape_ir.get("initializers", [])
    initializer_count = len(initializer_records)
    initializer_bytes = sum(int(rec.get("raw_data_bytes") or 0) for rec in initializer_records)
    graph_input_count = len(graph.get("inputs", [])) or max(0, len(args) - initializer_count)
    graph_output_count = len(returns)

    value_types: dict[str, TensorType] = {}
    producers: dict[str, str | None] = {}
    producer_positions: dict[str, int] = {}
    consumers: dict[str, list[str]] = defaultdict(list)
    semantic_by_value: dict[str, str] = {}
    known_ssa_ids = {normalize_value_id(name) for name, _ in args}

    for idx, (name, typ) in enumerate(args):
        tid = normalize_value_id(name)
        value_types[tid] = typ
        producers[tid] = None
        producer_positions[tid] = 0

    for op in ops:
        known_ssa_ids.update(op.result_ids)
        for rid, typ in zip(op.result_ids, op.result_types):
            value_types[rid] = typ
            producers[rid] = op.op_id
            producer_positions[rid] = op.position
            if op.attrs.get("cv.semantic_role"):
                semantic_by_value[rid] = op.attrs["cv.semantic_role"]
        for oid in op.operand_ids:
            if oid in value_types:
                consumers[oid].append(op.op_id)

    return_operands: list[str] = []
    for op in ops:
        if op.op_name == "func.return":
            return_operands = op.operand_ids
            break

    output_tensor_ids = set(return_operands)
    initializer_names = {rec.get("name") for rec in initializer_records}
    tensor_facts: list[CVTensorPlanningFact] = []
    unresolved: list[dict[str, Any]] = []

    for idx, (tid, typ) in enumerate(value_types.items()):
        b = tensor_bytes(typ)
        if b is None:
            unresolved.append({"kind": "tensor_byte_size", "tensor_id": tid, "reason": "dynamic shape or unsupported dtype"})
        is_graph_input = idx < graph_input_count and producers.get(tid) is None
        is_initializer = producers.get(tid) is None and not is_graph_input
        lifetime_start = producer_positions.get(tid, 0)
        consumer_positions = [next((op.position for op in ops if op.op_id == cid), lifetime_start) for cid in consumers.get(tid, [])]
        lifetime_end = max(consumer_positions + ([len(ops)] if tid in output_tensor_ids else [lifetime_start]))
        ownership = "external_graph_input" if is_graph_input else "model_state_initializer" if is_initializer else "external_graph_output" if tid in output_tensor_ids else "compiler_temporary"
        tensor_facts.append(
            CVTensorPlanningFact(
                tensor_id=tid,
                shape=typ.shape,
                dtype=typ.dtype,
                layout=infer_layout(typ),
                byte_size=b,
                producer=producers.get(tid),
                consumers=consumers.get(tid, []),
                is_graph_input=is_graph_input,
                is_graph_output=tid in output_tensor_ids,
                is_initializer=is_initializer,
                is_temporary=producers.get(tid) is not None and tid not in output_tensor_ids,
                semantic_role=semantic_by_value.get(tid),
                lifetime_start=lifetime_start,
                lifetime_end=lifetime_end,
                ownership=ownership,
                provenance={"initializer_name": tid if tid in initializer_names else None, "identity": "mlir_value_id"},
            )
        )

    tensor_by_id = {t.tensor_id: t for t in tensor_facts}

    outputs: list[CVOutputPlanningFact] = []
    for out_index, tid in enumerate(return_operands):
        tf = tensor_by_id.get(tid)
        producer = next((op for op in ops if op.op_id == producers.get(tid)), None)
        role = producer.attrs.get("cv.output_role") if producer else None
        if not role:
            role = f"unannotated_output_{out_index}"
            unresolved.append({"kind": "output_role", "tensor_id": tid, "reason": "missing cv.output_role"})
        outputs.append(
            CVOutputPlanningFact(
                tensor_id=tid,
                output_role=role,
                shape=tf.shape if tf else [],
                dtype=tf.dtype if tf else "unknown",
                producer_region=producer.attrs.get("cv.region_id") if producer else None,
                postprocess_required=True,
                postprocess_boundary=producer.attrs.get("cv.postprocess_boundary") if producer else None,
                ownership_expectation="caller_visible_output",
                layout=tf.layout if tf else "unknown",
            )
        )

    region_ops: dict[str, list[OperationFact]] = defaultdict(list)
    for op in ops:
        rid = op.attrs.get("cv.region_id")
        if rid:
            region_ops[rid].append(op)

    regions: list[CVRegionPlanningFact] = []
    for region_id, rops in sorted(region_ops.items()):
        op_ids = [op.op_id for op in rops]
        internal_results = {rid for op in rops for rid in op.result_ids}
        region_inputs = sorted({oid for op in rops for oid in op.operand_ids if oid in value_types and oid not in internal_results})
        region_outputs = sorted({rid for rid in internal_results if any(cid not in op_ids for cid in consumers.get(rid, [])) or rid in output_tensor_ids})
        output_shapes = [tensor_by_id[tid].shape for tid in region_outputs if tid in tensor_by_id]
        input_shapes = [tensor_by_id[tid].shape for tid in region_inputs if tid in tensor_by_id]
        dtype_counts = Counter(tensor_by_id[tid].dtype for tid in region_inputs + region_outputs if tid in tensor_by_id)
        dominant_dtype = dtype_counts.most_common(1)[0][0] if dtype_counts else None
        semantic_counts = Counter(op.attrs.get("cv.semantic_role", "unknown") for op in rops)
        semantic_role = semantic_counts.most_common(1)[0][0]
        confidence_counts = Counter(op.attrs.get("cv.recognition_confidence", "unknown") for op in rops)
        confidence = confidence_counts.most_common(1)[0][0]
        op_names = [op.op_name for op in rops]
        estimated_flops = sum(int(op.costs.get("flops", 0)) for op in rops)
        read_bytes = sum(int(op.costs.get("read_bytes", 0)) for op in rops)
        write_bytes = sum(int(op.costs.get("write_bytes", 0)) for op in rops)
        weight_bytes = sum(int(op.costs.get("weight_bytes", 0)) for op in rops)
        temporary_bytes = sum((tensor_by_id[tid].byte_size or 0) for tid in internal_results if tid in tensor_by_id and tid not in output_tensor_ids)
        notes = sorted({e for op in rops for e in op.attrs.get("cv.recognition_evidence", []) if isinstance(op.attrs.get("cv.recognition_evidence", []), list)})
        regions.append(
            CVRegionPlanningFact(
                region_id=region_id,
                semantic_role=semantic_role,
                recognition_confidence=confidence,
                operation_count=len(rops),
                operation_ids=op_ids,
                input_tensor_ids=region_inputs,
                output_tensor_ids=region_outputs,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                dominant_dtype=dominant_dtype,
                feature_scales=sorted({op.attrs.get("cv.feature_scale") for op in rops if op.attrs.get("cv.feature_scale")}),
                estimated_flops=estimated_flops,
                estimated_read_bytes=read_bytes,
                estimated_write_bytes=write_bytes,
                estimated_weight_bytes=weight_bytes,
                estimated_temporary_bytes=temporary_bytes,
                candidate_execution_domains=domain_candidates(set(op_names), semantic_role, dominant_dtype),
                quantization_eligibility=quantization_eligibility(set(op_names), semantic_role, dominant_dtype),
                fusion_eligibility=fusion_eligibility(op_names, semantic_role),
                planning_notes=notes,
            )
        )

    temporary_tensors = [t for t in tensor_facts if t.is_temporary and t.byte_size is not None]
    total_temporary_bytes = sum(t.byte_size or 0 for t in temporary_tensors)
    peak_live = 0
    for pos in range(len(ops) + 1):
        live = sum(t.byte_size or 0 for t in temporary_tensors if (t.lifetime_start or 0) <= pos <= (t.lifetime_end or 0))
        peak_live = max(peak_live, live)

    op_hist = Counter(op.op_name for op in ops)
    cost_by_region = {r.region_id: r.estimated_flops for r in regions}
    memory_summary = {
        "total_tensor_bytes": sum(t.byte_size or 0 for t in tensor_facts),
        "total_initializer_bytes": initializer_bytes,
        "total_temporary_bytes": total_temporary_bytes,
        "peak_live_temporary_bytes": peak_live,
        "top_temporary_tensors_by_size": [
            {
                "tensor_id": t.tensor_id,
                "byte_size": t.byte_size,
                "shape": t.shape,
                "producer": t.producer,
                "lifetime_start": t.lifetime_start,
                "lifetime_end": t.lifetime_end,
            }
            for t in sorted(temporary_tensors, key=lambda x: x.byte_size or 0, reverse=True)[:10]
        ],
        "top_region_memory_footprints": [
            {
                "region_id": r.region_id,
                "estimated_temporary_bytes": r.estimated_temporary_bytes,
                "estimated_read_bytes": r.estimated_read_bytes,
                "estimated_write_bytes": r.estimated_write_bytes,
                "estimated_weight_bytes": r.estimated_weight_bytes,
            }
            for r in sorted(regions, key=lambda x: x.estimated_temporary_bytes, reverse=True)
        ],
        "truth_boundary": "static_lifetime_analysis_no_slot_allocation_no_runtime_validation",
    }
    cost_summary = {
        "flop_convention": "1 MAC = 2 FLOPs",
        "estimated_total_flops": sum(int(op.costs.get("flops", 0)) for op in ops),
        "estimated_flops_by_region": cost_by_region,
        "estimated_total_read_bytes": sum(int(op.costs.get("read_bytes", 0)) for op in ops),
        "estimated_total_write_bytes": sum(int(op.costs.get("write_bytes", 0)) for op in ops),
        "truth_boundary": "static_analytical_estimates_not_measured_latency",
    }

    if not regions:
        unresolved.append({"kind": "semantic_regions", "reason": "no cv.region_id attributes found"})
    for op in ops:
        for oid in op.operand_ids:
            if oid in known_ssa_ids:
                continue
            unresolved.append({"kind": "unresolved_producer", "operation_id": op.op_id, "tensor_id": oid})

    return CVModelPlanningFacts(
        model_family=func_attrs.get("cv.model_family", "unknown"),
        function_name=function_name,
        graph_input_count=graph_input_count,
        graph_output_count=graph_output_count,
        regions=regions,
        outputs=outputs,
        tensors=tensor_facts,
        operation_summary={
            "operation_histogram": dict(sorted(op_hist.items())),
            "operation_count": len(ops),
            "region_operation_count": sum(len(v) for v in region_ops.values()),
            "tensor_count": len(tensor_facts),
        },
        unresolved_facts=unresolved[:200],
        provenance={
            "input_mlir": input_path,
            "shape_ir_available": bool(shape_ir),
            "shape_ir_graph": graph,
            "source_name_semantic_dependency": "none",
            "legacy_cv_reuse": {
                "CVMemoryPlanningPass": "not_reused_depends_on_legacy_cv_op_names_and_attrs",
                "CVExecutionDomainPlanningPass": "not_reused_depends_on_legacy_cv_op_names",
                "CVExecutionPlanBuilder": "not_reused_execution_plan_schema_out_of_scope",
                "ShapeCostModel": "concept_reused_static_shape_dtype_accounting_not_llm_matmul_formulas",
            },
        },
        memory_summary=memory_summary,
        cost_summary=cost_summary,
    )


def build_report(facts: CVModelPlanningFacts) -> dict[str, Any]:
    payload = asdict(facts)
    payload["schema"] = "cv_planning_facts"
    payload["schema_version"] = "0.1.0"
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_mlir", type=Path, required=True)
    parser.add_argument("--shape-ir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    shape_ir = load_shape_ir(args.shape_ir) if args.shape_ir else {}
    facts = build_facts(args.input_mlir.read_text(encoding="utf-8"), shape_ir, str(args.input_mlir))
    report = build_report(facts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"cv planning facts: {args.out}")
    print(f"  model_family: {facts.model_family}")
    print(f"  regions: {len(facts.regions)}")
    print(f"  tensors: {len(facts.tensors)}")
    print(f"  total_initializer_bytes: {facts.memory_summary['total_initializer_bytes']}")
    print(f"  peak_live_temporary_bytes: {facts.memory_summary['peak_live_temporary_bytes']}")
    print(f"  unresolved_facts: {len(facts.unresolved_facts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
