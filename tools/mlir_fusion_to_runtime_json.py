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


def load_kernel_profiles(paths):
    if not paths:
        return {"profile_status": "not_provided", "kernels": {}}

    profile_paths = [Path(path) for path in paths]
    kernels = {}
    loaded = []
    missing = []

    for profile_path in profile_paths:
        if not profile_path.exists():
            missing.append(str(profile_path))
            continue

        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        loaded.append(str(profile_path))
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
    }


def select_kernel(
    fusion_candidate,
    custom_kernel,
    custom_backend,
    fallback_kernel,
    fallback_backend,
    profile,
):
    evidence = profile.get("kernels", {}).get(fusion_candidate)
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
    }


def estimate_matmul_bias_relu_cost(m=1, k=128, n=64, dtype_bytes=4):
    matmul_flops = 2 * m * k * n
    bias_add_flops = m * n
    relu_flops = m * n
    total_flops = matmul_flops + bias_add_flops + relu_flops

    bytes_a = m * k * dtype_bytes
    bytes_b = k * n * dtype_bytes
    bytes_bias = m * n * dtype_bytes
    bytes_out = m * n * dtype_bytes

    bytes_read = bytes_a + bytes_b + bytes_bias
    bytes_written = bytes_out
    arithmetic_intensity = total_flops / max(bytes_read + bytes_written, 1)

    return {
        "m": m,
        "k": k,
        "n": n,
        "dtype": "f32",
        "estimated_flops": total_flops,
        "estimated_bytes_read": bytes_read,
        "estimated_bytes_written": bytes_written,
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
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
    }


def build_matmul_op(index, match, profile):
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
        "fusion_candidate": "matmul_bias_relu",
        "fusion_group": "matmul_bias_relu_0",
        "inputs": ["A", "B", "bias"],
        "outputs": [result_name],
        "cost_model": estimate_matmul_bias_relu_cost(),
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


def build_lowered_graph(matmul_matches, rmsnorm_matches, source_path, profile):
    ops = []
    for match in matmul_matches:
        ops.append(build_matmul_op(len(ops), match, profile))
    for match in rmsnorm_matches:
        ops.append(build_rmsnorm_op(len(ops), match, profile))

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

    if not matmul_matches and not rmsnorm_matches:
        raise SystemExit(
            "No fusion annotations found. Expected fusion.candidate for "
            "matmul_bias_relu or rmsnorm."
        )

    profile = load_kernel_profiles(args.kernel_profile)
    lowered_graph = build_lowered_graph(
        matmul_matches,
        rmsnorm_matches,
        input_path,
        profile,
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
