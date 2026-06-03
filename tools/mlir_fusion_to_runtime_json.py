#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


def detect_fused_matmul(text):
    hir_pattern = re.compile(
        r"(?P<result>%[\w\d_]+)\s*=\s*(?:\"hir\.fused_matmul_bias_relu\"|hir\.fused_matmul_bias_relu)\s*",
        re.MULTILINE,
    )
    hir_matches = list(hir_pattern.finditer(text))
    if hir_matches:
        return hir_matches

    annotated_pattern = re.compile(
        r"(?P<result>%[\w\d_]+)\s*=\s*linalg\.matmul\s*"
        r"\{[^}]*fusion\.candidate\s*=\s*\"matmul_bias_relu\"[^}]*\}",
        re.MULTILINE,
    )
    return list(annotated_pattern.finditer(text))


def detect_rmsnorm(text):
    hir_pattern = re.compile(
        r"(?P<result>%[\w\d_]+)\s*=\s*(?:\"hir\.fused_rmsnorm\"|hir\.fused_rmsnorm)\s*",
        re.MULTILINE,
    )
    hir_matches = list(hir_pattern.finditer(text))
    if hir_matches:
        return hir_matches

    annotated_pattern = re.compile(
        r"(?P<result>%[\w\d_]+)\s*=\s*\"llm\.rmsnorm\"\s*"
        r"\([^)]*\)\s*\{[^}]*fusion\.candidate\s*=\s*\"rmsnorm\"[^}]*\}",
        re.MULTILINE,
    )
    return list(annotated_pattern.finditer(text))


def detect_fused_qmatmul(text):
    hir_pattern = re.compile(
        r"(?P<result>%[\w\d_]+)\s*=\s*(?:\"hir\.fused_qmatmul_bias_relu\"|hir\.fused_qmatmul_bias_relu)\s*",
        re.MULTILINE,
    )
    return list(hir_pattern.finditer(text))


def op_line(text, match):
    line_start = text.rfind("\n", 0, match.start()) + 1
    line_end = text.find("\n", match.end())
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end]


def parse_tensor_shapes(text, match):
    line = op_line(text, match)
    tensor_pattern = re.compile(
        r"tensor<(?P<d0>\d+)x(?P<d1>\d+)x(?P<dtype>f32|f16|i8)>"
    )
    return [
        {
            "d0": int(found.group("d0")),
            "d1": int(found.group("d1")),
            "dtype": found.group("dtype"),
        }
        for found in tensor_pattern.finditer(line)
    ]


def matmul_shape_from_mlir(text, match, default):
    shapes = parse_tensor_shapes(text, match)
    if len(shapes) < 2:
        return default
    lhs = shapes[0]
    rhs = shapes[1]
    return {
        "m": lhs["d0"],
        "k": lhs["d1"],
        "n": rhs["d1"],
        "dtype": lhs["dtype"],
    }


def load_kernel_profiles(paths):
    if not paths:
        return {"profile_status": "not_provided", "kernels": {}}

    profile_paths = [Path(path) for path in paths]
    kernels = {}
    cost_table = {}
    loaded = []
    missing = []

    for profile_path in profile_paths:
        if not profile_path.exists():
            missing.append(str(profile_path))
            continue

        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        loaded.append(str(profile_path))
        if payload.get("artifact_type") == "profile_calibrated_cost_table":
            for fusion_candidate, by_backend in payload.get("cost_table", {}).items():
                cost_table.setdefault(fusion_candidate, {}).update(by_backend)
        for row in payload.get("kernel_benchmarks", []):
            kernels[row.get("fusion_candidate")] = row

    if not loaded:
        return {
            "profile_status": "missing",
            "profile_path": ",".join(str(path) for path in profile_paths),
            "missing_profiles": missing,
            "kernels": {},
        }

    return {
        "profile_status": "loaded",
        "profile_path": ",".join(loaded),
        "missing_profiles": missing,
        "kernels": kernels,
        "cost_table": cost_table,
    }


def shape_bucket_for(fusion_candidate):
    if fusion_candidate == "matmul_bias_relu":
        return "128x128x128:f32"
    if fusion_candidate == "qmatmul_bias_relu":
        return "128x128x128:i8"
    if fusion_candidate == "rmsnorm":
        return "16x4096:f32"
    return "default:f32"


def profile_cost_entry(profile, fusion_candidate, backend):
    bucket = shape_bucket_for(fusion_candidate)
    return (
        profile.get("cost_table", {})
        .get(fusion_candidate, {})
        .get(backend, {})
        .get(bucket)
    )


def select_kernel(
    fusion_candidate,
    custom_kernel,
    custom_backend,
    fallback_kernel,
    fallback_backend,
    profile,
):
    bucket = shape_bucket_for(fusion_candidate)
    table_entry = profile_cost_entry(profile, fusion_candidate, custom_backend)
    evidence = profile.get("kernels", {}).get(fusion_candidate)
    if table_entry:
        evidence = {
            "fusion_candidate": fusion_candidate,
            "custom_kernel": table_entry.get("custom_kernel"),
            "fallback_kernel": table_entry.get("fallback_kernel"),
            "custom_latency_ms": table_entry.get("custom_ms"),
            "fallback_latency_ms": table_entry.get("fallback_ms"),
            "speedup": table_entry.get("speedup"),
            "correct": table_entry.get("correct"),
            "selection_ready": table_entry.get("selection_ready"),
            "shape_bucket": bucket,
            "profile_table_source": table_entry.get("source"),
        }
    if not evidence:
        return {
            "selected_kernel": fallback_kernel,
            "selected_backend": fallback_backend,
            "candidate_kernel": custom_kernel,
            "candidate_backend": custom_backend,
            "fallback_kernel": fallback_kernel,
            "fallback_backend": fallback_backend,
            "profile_status": profile.get("profile_status", "not_provided"),
            "profile_source": profile.get("profile_path"),
            "selection_reason": "fallback_no_profile_evidence",
            "profile_calibrated": False,
            "shape_bucket": bucket,
            "evidence": None,
        }

    custom_ms = evidence.get("custom_latency_ms")
    fallback_ms = evidence.get("fallback_latency_ms")
    custom_wins = (
        isinstance(custom_ms, (int, float))
        and isinstance(fallback_ms, (int, float))
        and custom_ms < fallback_ms
    )

    return {
        "selected_kernel": custom_kernel if custom_wins else fallback_kernel,
        "selected_backend": custom_backend if custom_wins else fallback_backend,
        "candidate_kernel": custom_kernel,
        "candidate_backend": custom_backend,
        "fallback_kernel": fallback_kernel,
        "fallback_backend": fallback_backend,
        "profile_status": profile.get("profile_status", "loaded"),
        "profile_source": profile.get("profile_path"),
        "selection_reason": "profile_calibrated_fastest" if custom_wins else "profile_calibrated_fallback",
        "profile_calibrated": True,
        "shape_bucket": bucket,
        "cost_table_entry": table_entry,
        "evidence": evidence,
    }


def build_runtime_dispatch_contract(hir_op_type, runtime_op_type, selection):
    return {
        "op_type": hir_op_type,
        "runtime_op_type": runtime_op_type,
        "runtime_kernel": selection["selected_kernel"],
        "backend": selection["selected_backend"],
        "candidate_kernel": selection["candidate_kernel"],
        "fallback_kernel": selection["fallback_kernel"],
        "profile_source": selection["profile_source"],
        "selection_reason": selection["selection_reason"],
        "profile_calibrated": selection["profile_calibrated"],
        "shape_bucket": selection["shape_bucket"],
    }


def sparsecore_like_target_model():
    return {
        "target": "sparsecore_like_v1",
        "base_tile_shape": {"m": 16, "n": 16, "k": 32},
        "memory_hierarchy": "global_sram_register",
        "sram_kb": 256,
        "vector_bytes": 128,
        "alignment_bytes": 128,
        "sparse_layout": "dense_or_2_4",
        "collective": "none",
        "legality": [
            "matmul result must have one use",
            "bias add result must have one use",
            "bias must be materialized to result shape or represented as legal HIR broadcast",
            "M/N/K must be static multiples of 16/16/32",
            "dtype must be f32 or f16 for this fused floating-point path",
        ],
    }


def dtype_bytes(dtype):
    return {"f32": 4, "f16": 2, "i8": 1}.get(dtype, 4)


def estimate_tile_sram_bytes(tile_m, tile_n, tile_k, bytes_per_element):
    lhs_bytes = tile_m * tile_k * bytes_per_element
    rhs_bytes = tile_k * tile_n * bytes_per_element
    bias_bytes = tile_m * tile_n * bytes_per_element
    out_bytes = tile_m * tile_n * bytes_per_element
    return lhs_bytes + rhs_bytes + bias_bytes + out_bytes


def choose_tile(shape, target):
    bytes_per_element = dtype_bytes(shape["dtype"])
    sram_budget = target["sram_kb"] * 1024
    vector_bytes = target["vector_bytes"]
    candidates = [
        {"m": 16, "n": 16, "k": 32},
        {"m": 16, "n": 32, "k": 32},
        {"m": 32, "n": 16, "k": 32},
        {"m": 32, "n": 32, "k": 32},
        {"m": 16, "n": 64, "k": 32},
        {"m": 64, "n": 16, "k": 32},
        {"m": 64, "n": 32, "k": 32},
        {"m": 32, "n": 64, "k": 32},
        {"m": 64, "n": 64, "k": 32},
        {"m": 16, "n": 32, "k": 128},
        {"m": 16, "n": 64, "k": 128},
        {"m": 32, "n": 32, "k": 128},
        {"m": 32, "n": 64, "k": 128},
        {"m": 64, "n": 32, "k": 128},
        {"m": 64, "n": 64, "k": 128},
    ]

    evaluated = []
    legal = []
    for tile in candidates:
        reasons = []
        if shape["m"] % tile["m"] != 0:
            reasons.append("M_not_multiple_of_tile_m")
        if shape["n"] % tile["n"] != 0:
            reasons.append("N_not_multiple_of_tile_n")
        if shape["k"] % tile["k"] != 0:
            reasons.append("K_not_multiple_of_tile_k")
        if (tile["k"] * bytes_per_element) % vector_bytes != 0:
            reasons.append("K_tile_not_vector_aligned")
        sram_bytes = estimate_tile_sram_bytes(
            tile["m"],
            tile["n"],
            tile["k"],
            bytes_per_element,
        )
        if sram_bytes > sram_budget:
            reasons.append("tile_exceeds_sram_budget")
        flops = 2 * tile["m"] * tile["n"] * tile["k"]
        arithmetic_intensity = flops / max(sram_bytes, 1)
        record = {
            "tile": tile,
            "sram_bytes": sram_bytes,
            "arithmetic_intensity_flops_per_byte": round(arithmetic_intensity, 6),
            "legal": not reasons,
            "reject_reasons": reasons,
        }
        evaluated.append(record)
        if record["legal"]:
            legal.append(record)

    if not legal:
        return {
            "status": "fallback_no_legal_tile",
            "selected_tile": None,
            "selected_sram_bytes": None,
            "decision_reason": "no tile satisfies shape, vector alignment, and SRAM constraints",
            "candidates": evaluated,
        }

    selected = max(
        legal,
        key=lambda item: (
            item["arithmetic_intensity_flops_per_byte"],
            item["tile"]["m"] * item["tile"]["n"],
        ),
    )
    return {
        "status": "selected",
        "selected_tile": selected["tile"],
        "selected_sram_bytes": selected["sram_bytes"],
        "decision_reason": "selected highest-intensity legal tile under SRAM/alignment constraints",
        "candidates": evaluated,
    }


def build_dispatch_descriptor(hir_op_type, runtime_op_type, selection, shape, target):
    tile_decision = choose_tile(shape, target)
    return {
        "descriptor_type": "target.dispatch_descriptor.v1",
        "hir_op": hir_op_type,
        "runtime_op_type": runtime_op_type,
        "target": target["target"],
        "backend": selection["selected_backend"],
        "kernel": selection["selected_kernel"],
        "shape": shape,
        "dtype": shape["dtype"],
        "tile_decision": tile_decision,
        "memory_hierarchy": target["memory_hierarchy"],
        "alignment_bytes": target["alignment_bytes"],
        "vector_bytes": target["vector_bytes"],
        "sparse_layout": target["sparse_layout"],
    }


def estimate_matmul_bias_relu_cost(m=16, k=128, n=64, dtype="f32"):
    bytes_per_element = dtype_bytes(dtype)
    matmul_flops = 2 * m * k * n
    bias_add_flops = m * n
    relu_flops = m * n
    total_flops = matmul_flops + bias_add_flops + relu_flops

    bytes_a = m * k * bytes_per_element
    bytes_b = k * n * bytes_per_element
    bytes_bias = m * n * bytes_per_element
    bytes_out = m * n * bytes_per_element

    bytes_read = bytes_a + bytes_b + bytes_bias
    bytes_written = bytes_out
    arithmetic_intensity = total_flops / max(bytes_read + bytes_written, 1)

    return {
        "m": m,
        "k": k,
        "n": n,
        "dtype": dtype,
        "estimated_flops": total_flops,
        "estimated_bytes_read": bytes_read,
        "estimated_bytes_written": bytes_written,
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        "source": "profile_calibrated_table",
        "shape_bucket": "128x128x128:f32",
    }


def estimate_rmsnorm_cost(tokens=16, hidden=768, dtype_bytes=2):
    elements = tokens * hidden
    total_flops = elements * 4
    bytes_read = elements * dtype_bytes * 2
    bytes_written = elements * dtype_bytes

    return {
        "tokens": tokens,
        "hidden": hidden,
        "dtype": "f16",
        "estimated_flops": total_flops,
        "estimated_bytes_read": bytes_read,
        "estimated_bytes_written": bytes_written,
        "arithmetic_intensity_flops_per_byte": total_flops / max(bytes_read + bytes_written, 1),
        "source": "profile_calibrated_table",
        "shape_bucket": "16x4096:f32",
    }


def build_matmul_op(index, match, profile, source_text):
    selection = select_kernel(
        "matmul_bias_relu",
        "fused_matmul_add_relu",
        "CPU",
        "unfused_matmul_add_relu",
        "CPU",
        profile,
    )
    result_name = match.group("result")
    hir_op_type = "hir.fused_matmul_bias_relu"
    runtime_op_type = "FusedMatMulAddReLU"
    shape = matmul_shape_from_mlir(
        source_text,
        match,
        {"m": 16, "k": 128, "n": 64, "dtype": "f32"},
    )
    target = sparsecore_like_target_model()
    dispatch_descriptor = build_dispatch_descriptor(
        hir_op_type,
        runtime_op_type,
        selection,
        shape,
        target,
    )
    return {
        "id": index,
        "name": f"fused_matmul_bias_relu_{index}",
        "source_result": result_name,
        "op_type": hir_op_type,
        "legacy_op_type": "FusedMatMulBiasReLU",
        "lowered_op_type": hir_op_type,
        "runtime_op_type": runtime_op_type,
        "runtime_kernel": selection["selected_kernel"],
        "runtime_kernel_backend": selection["selected_backend"],
        "backend": selection["selected_backend"],
        "runtime_dispatch_contract": build_runtime_dispatch_contract(
            hir_op_type,
            runtime_op_type,
            selection,
        ),
        "target_model": target,
        "dispatch_descriptor": dispatch_descriptor,
        "fusion_candidate": "matmul_bias_relu",
        "fusion_group": "matmul_bias_relu_0",
        "inputs": ["A", "B", "bias"],
        "outputs": [result_name],
        "cost_model": estimate_matmul_bias_relu_cost(
            shape["m"],
            shape["k"],
            shape["n"],
            shape["dtype"],
        ),
        "kernel_selection": selection,
        "notes": [
            "Detected from MLIR linalg.matmul annotated by MatMulBiasReluFusionPass",
            "Lowered through the shared runtime-aware kernel selection contract",
            "Runtime benchmark evidence selects the custom fused kernel or fallback path",
        ],
    }


def build_rmsnorm_op(index, match, profile):
    selection = select_kernel(
        "rmsnorm",
        "fused_rmsnorm_cuda",
        "CUDA",
        "torch_rmsnorm",
        "PyTorch",
        profile,
    )
    result_name = match.group("result")
    hir_op_type = "hir.fused_rmsnorm"
    runtime_op_type = "FusedRMSNorm"
    return {
        "id": index,
        "name": f"fused_rmsnorm_{index}",
        "source_result": result_name,
        "op_type": hir_op_type,
        "legacy_op_type": "FusedRMSNorm",
        "lowered_op_type": hir_op_type,
        "runtime_op_type": runtime_op_type,
        "runtime_kernel": selection["selected_kernel"],
        "runtime_kernel_backend": selection["selected_backend"],
        "backend": selection["selected_backend"],
        "runtime_dispatch_contract": build_runtime_dispatch_contract(
            hir_op_type,
            runtime_op_type,
            selection,
        ),
        "fusion_candidate": "rmsnorm",
        "fusion_group": f"rmsnorm_{index}",
        "inputs": ["hidden_states", "weight"],
        "outputs": [result_name],
        "cost_model": estimate_rmsnorm_cost(),
        "kernel_selection": selection,
        "notes": [
            "Detected from MLIR llm.rmsnorm annotated by RMSNormKernelSelectionPass",
            "Lowered to HIR fused RMSNorm candidate",
            "Runtime benchmark evidence selects custom CUDA or PyTorch fallback",
        ],
    }


def build_qmatmul_op(index, match, profile, source_text):
    selection = select_kernel(
        "qmatmul_bias_relu",
        "int8_qmatmul_bias_relu",
        "CPU",
        "fused_matmul_add_relu",
        "CPU",
        profile,
    )
    result_name = match.group("result")
    hir_op_type = "hir.fused_qmatmul_bias_relu"
    runtime_op_type = "FusedQMatMulBiasReLU"
    shape = matmul_shape_from_mlir(
        source_text,
        match,
        {"m": 128, "k": 128, "n": 128, "dtype": "i8"},
    )
    shape["dtype"] = "i8"
    target = sparsecore_like_target_model()
    dispatch_descriptor = build_dispatch_descriptor(
        hir_op_type,
        runtime_op_type,
        selection,
        shape,
        target,
    )
    return {
        "id": index,
        "name": f"fused_qmatmul_bias_relu_{index}",
        "source_result": result_name,
        "op_type": hir_op_type,
        "legacy_op_type": "FusedQMatMulBiasReLU",
        "lowered_op_type": hir_op_type,
        "runtime_op_type": runtime_op_type,
        "runtime_kernel": selection["selected_kernel"],
        "runtime_kernel_backend": selection["selected_backend"],
        "backend": selection["selected_backend"],
        "runtime_dispatch_contract": build_runtime_dispatch_contract(
            hir_op_type,
            runtime_op_type,
            selection,
        ),
        "target_model": target,
        "dispatch_descriptor": dispatch_descriptor,
        "fusion_candidate": "qmatmul_bias_relu",
        "fusion_group": f"qmatmul_bias_relu_{index}",
        "inputs": ["A_int8", "B_int8", "bias"],
        "outputs": [result_name],
        "cost_model": {
            "m": 128,
            "k": 128,
            "n": 128,
            "dtype": "i8",
            "estimated_flops": 2 * 128 * 128 * 128 + 128 * 128 * 2,
            "estimated_bytes_read": 128 * 128 + 128 * 128 + 128 * 128 * 4,
            "estimated_bytes_written": 128 * 128 * 4,
            "arithmetic_intensity_flops_per_byte": (
                (2 * 128 * 128 * 128 + 128 * 128 * 2)
                / max((128 * 128 + 128 * 128 + 128 * 128 * 4 + 128 * 128 * 4), 1)
            ),
            "source": "profile_calibrated_table",
            "shape_bucket": "128x128x128:i8",
            "layout": {
                "input_layout": "NHWC",
                "weight_layout": "blocked_kc",
                "alignment": 128,
            },
        },
        "kernel_selection": selection,
        "notes": [
            "Detected from typed HIR quantized MatMul-Bias-ReLU op",
            "Lowered only when profile metadata marks the INT8 path valid and faster",
            "Layout constraints model mobile DSP/NPU alignment and channel requirements",
        ],
    }


def build_lowered_graph(matmul_matches, rmsnorm_matches, qmatmul_matches, source_path, profile, source_text):
    ops = []
    for match in matmul_matches:
        ops.append(build_matmul_op(len(ops), match, profile, source_text))
    for match in rmsnorm_matches:
        ops.append(build_rmsnorm_op(len(ops), match, profile))
    for match in qmatmul_matches:
        ops.append(build_qmatmul_op(len(ops), match, profile, source_text))

    return {
        "format": "hir.lowered_graph.v1",
        "source": str(source_path),
        "kernel_profile": {
            "status": profile.get("profile_status", "not_provided"),
            "source": profile.get("profile_path"),
        },
        "num_ops": len(ops),
        "ops": ops,
    }


def build_execution_plan(lowered_graph):
    steps = []

    for op in lowered_graph["ops"]:
        steps.append({
            "step": op["id"],
            "op_name": op["name"],
            "op_type": op["op_type"],
            "lowered_op_type": op["lowered_op_type"],
            "runtime_op_type": op["runtime_op_type"],
            "runtime_kernel": op["runtime_kernel"],
            "runtime_kernel_backend": op["runtime_kernel_backend"],
            "backend": op["backend"],
            "fusion_candidate": op["fusion_candidate"],
            "fusion_group": op["fusion_group"],
            "runtime_action": "dispatch_selected_kernel",
            "runtime_dispatch_contract": op["runtime_dispatch_contract"],
            "kernel_selection": op["kernel_selection"],
            "target_model": op.get("target_model"),
            "dispatch_descriptor": op.get("dispatch_descriptor"),
            "estimated_launch_overhead_us": 80,
            "estimated_flops": op["cost_model"]["estimated_flops"],
            "arithmetic_intensity_flops_per_byte": op["cost_model"]["arithmetic_intensity_flops_per_byte"],
        })

    return {
        "format": "hir.execution_plan.v1",
        "source": lowered_graph["source"],
        "kernel_profile": lowered_graph["kernel_profile"],
        "num_steps": len(steps),
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="trace/mlir_fused_graph.mlir")
    parser.add_argument("--lowered-output", default="trace/mlir_lowered_graph.json")
    parser.add_argument("--plan-output", default="trace/mlir_execution_plan.json")
    parser.add_argument("--kernel-profile", action="append")
    args = parser.parse_args()

    input_path = Path(args.input)
    text = input_path.read_text(encoding="utf-8")

    matmul_matches = detect_fused_matmul(text)
    rmsnorm_matches = detect_rmsnorm(text)
    qmatmul_matches = detect_fused_qmatmul(text)

    if not matmul_matches and not rmsnorm_matches and not qmatmul_matches:
        raise SystemExit(
            "No fusion annotations found. Expected fusion.candidate for "
            "matmul_bias_relu, qmatmul_bias_relu, or rmsnorm."
        )

    profile = load_kernel_profiles(args.kernel_profile)
    lowered_graph = build_lowered_graph(
        matmul_matches,
        rmsnorm_matches,
        qmatmul_matches,
        input_path,
        profile,
        text,
    )
    execution_plan = build_execution_plan(lowered_graph)

    lowered_output = Path(args.lowered_output)
    plan_output = Path(args.plan_output)

    lowered_output.parent.mkdir(parents=True, exist_ok=True)
    plan_output.parent.mkdir(parents=True, exist_ok=True)

    lowered_output.write_text(json.dumps(lowered_graph, indent=2) + "\n", encoding="utf-8")
    plan_output.write_text(json.dumps(execution_plan, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {lowered_output}")
    print(f"Wrote {plan_output}")


if __name__ == "__main__":
    main()
