#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_optional_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return load_json(path)


def load_text(path):
    return Path(path).read_text(encoding="utf-8")


def first_step(plan):
    steps = plan.get("steps", [])
    if not steps:
        raise SystemExit("RMSNorm execution plan has no steps")
    return steps[0]


def first_rmsnorm_benchmark(profile):
    for benchmark in profile.get("kernel_benchmarks", []):
        if benchmark.get("fusion_candidate") == "rmsnorm":
            return benchmark
    return {}


def metric(evidence, benchmark, key):
    value = evidence.get(key)
    if value is not None:
        return value
    return benchmark.get(key)


def write_markdown(path, report):
    decision = report["runtime_decision"]
    perf = report["performance"]
    lines = [
        "# RMSNorm Compiler-Runtime Case Study",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Pipeline",
        "",
        "```text",
        "llm.rmsnorm",
        "  -> RMSNormKernelSelectionPass",
        "  -> hir.fused_rmsnorm",
        "  -> profile-calibrated runtime dispatch contract",
        "  -> fused_rmsnorm_cuda vs torch_rmsnorm",
        "```",
        "",
        "## Runtime Decision",
        "",
        f"- HIR op: `{decision['hir_op']}`",
        f"- Selected kernel: `{decision['selected_kernel']}`",
        f"- Fallback kernel: `{decision['fallback_kernel']}`",
        f"- Backend: `{decision['backend']}`",
        f"- Selection reason: `{decision['selection_reason']}`",
        f"- Feedback loop: `{decision.get('feedback_loop')}`",
        f"- Profile source: `{decision['profile_source']}`",
        "",
        "## Performance Evidence",
        "",
        f"- Shape bucket: `{perf['shape_bucket']}`",
        f"- Custom latency: `{perf['custom_latency_ms']} ms`",
        f"- PyTorch latency: `{perf['fallback_latency_ms']} ms`",
        f"- Speedup: `{perf['speedup']}x`",
        f"- Correct: `{perf['correct']}`",
        f"- Custom effective bandwidth: `{perf['custom_effective_bandwidth_gbps']} GB/s`",
        f"- PyTorch effective bandwidth: `{perf['fallback_effective_bandwidth_gbps']} GB/s`",
        f"- Bytes/token: `{perf['bytes_per_token']}`",
        f"- FLOPs/token: `{perf['flops_per_token']}`",
        f"- Arithmetic intensity: `{perf['arithmetic_intensity_flops_per_byte']} FLOPs/byte`",
        "",
        "## Roofline Interpretation",
        "",
        "- RMSNorm is memory-bound: arithmetic intensity is low and each token streams input, weight, and output data.",
        "- The custom CUDA path reduces framework overhead and uses a shape-specialized reduction/writeback kernel.",
        "- The compiler does not assume the custom kernel wins; it selects `fused_rmsnorm_cuda` because GPU PGO-like runtime evidence says it is faster and correct for the shape bucket.",
    ]
    if report.get("technology_gate"):
        gate = report["technology_gate"]
        lines.extend([
            "",
            "## GPU PGO-like Gate",
            "",
            f"- Input: `{gate.get('input')}`",
            f"- Decision: `{gate.get('decision')}`",
            f"- Metric: `{gate.get('metric')}`",
        ])
    if report.get("serving_impact"):
        impact = report["serving_impact"]
        lines.extend([
            "",
            "## Serving Impact Projection",
            "",
            f"- Baseline TPOT p95: `{impact.get('baseline_tpot_p95_ms')}` ms/token",
            f"- Projected TPOT p95: `{impact.get('projected_tpot_p95_ms')}` ms/token",
            f"- TPOT delta: `{impact.get('tpot_delta_ms')}` ms/token",
            f"- Projected tokens/sec gain: `{impact.get('projected_tokens_per_second_gain')}`",
        ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir-input", default="mlir_passes/test/rmsnorm_kernel_selection.mlir")
    parser.add_argument("--hir-mlir", default="trace/rmsnorm_fused_graph.mlir")
    parser.add_argument("--execution-plan", default="trace/rmsnorm_execution_plan.json")
    parser.add_argument("--runtime-profile", default="/Users/allen/Documents/Codex/project/heterogeneous-inference-runtime/results/cuda_transformer/rmsnorm_benchmark.json")
    parser.add_argument("--gpu-pgo-report", default="/Users/allen/Documents/Codex/project/heterogeneous-inference-runtime/results/cuda_transformer/gpu_pgo_like_rmsnorm_report.json")
    parser.add_argument("--json-output", default="trace/rmsnorm_compiler_runtime_case_study.json")
    parser.add_argument("--markdown-output", default="trace/rmsnorm_compiler_runtime_case_study.md")
    args = parser.parse_args()

    plan = load_json(args.execution_plan)
    profile = load_json(args.runtime_profile)
    gpu_pgo = load_optional_json(args.gpu_pgo_report)
    step = first_step(plan)
    selection = step.get("kernel_selection", {})
    evidence = selection.get("evidence") or {}
    benchmark = first_rmsnorm_benchmark(profile)

    checks = {
        "input_contains_llm_rmsnorm": '"llm.rmsnorm"' in load_text(args.mlir_input),
        "lowered_to_hir_fused_rmsnorm": "hir.fused_rmsnorm" in load_text(args.hir_mlir),
        "selected_fused_cuda_kernel": step.get("runtime_kernel") == "fused_rmsnorm_cuda",
        "fallback_is_torch_rmsnorm": selection.get("fallback_kernel") == "torch_rmsnorm",
        "runtime_profile_measured": profile.get("profile_status") == "measured",
        "correctness_passed": evidence.get("correct") is True,
        "profile_calibrated": selection.get("profile_calibrated") is True,
        "gpu_pgo_like_gate_passed": gpu_pgo.get("technology_gate", {}).get("passes_gate") is True,
    }

    report = {
        "artifact_type": "rmsnorm_compiler_runtime_case_study",
        "status": "passed" if all(checks.values()) else "failed",
        "pipeline": [
            "llm.rmsnorm",
            "RMSNormKernelSelectionPass",
            "hir.fused_rmsnorm",
            "profile_calibrated_runtime_dispatch_contract",
            "fused_rmsnorm_cuda",
        ],
        "runtime_decision": {
            "hir_op": step.get("op_type"),
            "selected_kernel": step.get("runtime_kernel"),
            "fallback_kernel": selection.get("fallback_kernel"),
            "backend": step.get("backend"),
            "selection_reason": selection.get("selection_reason"),
            "feedback_loop": selection.get("feedback_loop") or step.get("runtime_dispatch_contract", {}).get("feedback_loop"),
            "profile_source": selection.get("profile_source"),
        },
        "technology_gate": gpu_pgo.get("technology_gate", {}),
        "serving_impact": gpu_pgo.get("serving_impact", {}),
        "performance": {
            "shape_bucket": selection.get("shape_bucket"),
            "custom_latency_ms": metric(evidence, benchmark, "custom_latency_ms"),
            "fallback_latency_ms": metric(evidence, benchmark, "fallback_latency_ms"),
            "speedup": metric(evidence, benchmark, "speedup"),
            "correct": metric(evidence, benchmark, "correct"),
            "custom_effective_bandwidth_gbps": metric(evidence, benchmark, "custom_effective_bandwidth_gbps"),
            "fallback_effective_bandwidth_gbps": metric(evidence, benchmark, "fallback_effective_bandwidth_gbps"),
            "bytes_per_token": metric(evidence, benchmark, "bytes_per_token"),
            "flops_per_token": metric(evidence, benchmark, "flops_per_token"),
            "arithmetic_intensity_flops_per_byte": metric(evidence, benchmark, "arithmetic_intensity_flops_per_byte"),
        },
        "checks": checks,
    }

    json_output = Path(args.json_output)
    markdown_output = Path(args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(markdown_output, report)

    if report["status"] != "passed":
        raise SystemExit(f"RMSNorm case study failed: {json_output}")

    print(f"Wrote {json_output}")
    print(f"Wrote {markdown_output}")


if __name__ == "__main__":
    main()
