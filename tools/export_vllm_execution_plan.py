#!/usr/bin/env python3
"""Export a vLLM-facing execution plan for Qwen on a single GTX 1650 Max-Q.

This artifact is compiler planning intent only. It does not compile model weights,
modify vLLM, run inference, or report measured performance.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "vllm_plans" / "qwen_0_5b_gtx1650_plan.json"
DEFAULT_CAPABILITIES_ROOT = REPO_ROOT.parent / "ml-platform-capabilities" / "profiles"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TRUTH_BOUNDARY = "Execution planning artifact only; not measured performance."
REQUIRED_SECTIONS = (
    "producer",
    "source_artifacts",
    "model",
    "hardware_profile",
    "backend_profile",
    "workload_profile",
    "batch_policy",
    "prefix_policy",
    "memory_policy",
    "quantization_policy",
    "speculative_policy",
    "runtime_config",
)


class VLLMExecutionPlanExportError(ValueError):
    """Raised when the compiler cannot build a valid vLLM execution plan."""


def build_plan(
    *,
    created_at_utc: str | None = None,
    git_commit: str | None = None,
    capabilities_root: Path | None = None,
) -> dict[str, Any]:
    """Build the deterministic vLLM execution plan payload."""
    created_at_utc = created_at_utc or datetime.now(timezone.utc).isoformat()
    capabilities_root = capabilities_root or DEFAULT_CAPABILITIES_ROOT
    capabilities = load_capabilities(capabilities_root)
    plan = {
        "artifact_type": "vllm_execution_plan",
        "schema_version": "1.0.0",
        "plan_id": "qwen_0_5b_gtx1650_v1",
        "producer": {
            "repo": "ml-graph-compiler-runtime",
            "tool": "tools/export_vllm_execution_plan.py",
            "git_commit": git_commit,
            "created_at_utc": created_at_utc,
        },
        "truth_boundary": TRUTH_BOUNDARY,
        "planning_provenance": {
            "planner_layer": "backend_independent_execution_planning",
            "runtime_target": "vllm",
            "capability_source": "../ml-platform-capabilities/profiles",
            "model_weights_modified": False,
            "backend_modified": False,
            "performance_measured": False,
        },
        "source_artifacts": [
            "configs/models/qwen_0_5b_spec.json",
            "configs/vllm_serving_request_trace.json",
            "artifacts/vllm_serving_plan/compiler_serving_plan.json",
            "artifacts/vllm_serving_plan/kv_cache_layout_plan.json",
            "artifacts/vllm_serving_plan/cuda_graph_bucket_plan.json",
            _source_path(capabilities_root, "hardware/nvidia_gtx1650_maxq.json"),
            _source_path(capabilities_root, "backend/vllm.json"),
            *[
                _source_path(capabilities_root, f"kernels/{name}.json")
                for name in capabilities["backend"].get("supported_kernel_libraries", [])
            ],
        ],
        "model": {
            "model_id": MODEL_ID,
            "tokenizer": MODEL_ID,
            "dtype": "float16",
            "quantization": "none",
            "trust_remote_code": False,
        },
        "hardware_profile": _hardware_profile(capabilities["hardware"]),
        "backend_profile": _backend_profile(capabilities["backend"]),
        "kernel_profile": _kernel_profile(capabilities["kernels"]),
        "workload_profile": {
            "request_count": 32,
            "shared_prefix": True,
            "prompt_len_bucket": "short_to_medium",
            "output_len": 128,
        },
        "batch_policy": {
            "max_num_seqs": 4,
            "max_num_batched_tokens": 2048,
            "enable_chunked_prefill": True,
        },
        "prefix_policy": {
            "enable_prefix_caching": True,
            "group_by_shared_prefix": True,
        },
        "memory_policy": {
            "gpu_memory_utilization": 0.75,
            "max_model_len": 2048,
            "block_size": 16,
            "swap_space": 2,
        },
        "quantization_policy": {
            "dtype": "float16",
            "quantization": "none",
        },
        "speculative_policy": {
            "enabled": False,
            "draft_model": None,
            "num_speculative_tokens": None,
        },
        "runtime_config": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "served_model_name": "qwen-0.5b-compiler-plan",
        },
        "metrics_schema": {
            "ttft_ms": True,
            "tpot_ms": True,
            "throughput_tok_s": True,
            "request_latency_ms": True,
            "gpu_memory_used_mb": True,
            "prefix_cache_hit_rate": True,
            "queue_wait_ms": True,
        },
        "feedback": {
            "baseline_metrics_artifact": None,
            "compiler_plan_metrics_artifact": None,
        },
        "non_claims": [
            "no_measured_speedup_claim",
            "no_qwen_weight_modification",
            "no_vllm_fork_or_scheduler_modification",
        ],
    }
    validate_plan(plan)
    return plan


def load_capabilities(root: Path) -> dict[str, Any]:
    """Load shared capability definitions from ../ml-platform-capabilities/profiles."""
    hardware = _load_json(root / "hardware" / "nvidia_gtx1650_maxq.json")
    backend = _load_json(root / "backend" / "vllm.json")
    kernel_libraries = backend.get("supported_kernel_libraries", [])
    if not isinstance(kernel_libraries, list) or not kernel_libraries:
        raise VLLMExecutionPlanExportError("backend/vllm.json must declare supported_kernel_libraries")
    kernels = [_load_json(root / "kernels" / f"{name}.json") for name in kernel_libraries]
    return {"hardware": hardware, "backend": backend, "kernels": kernels}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VLLMExecutionPlanExportError(f"missing shared capability profile: {path}") from exc
    except json.JSONDecodeError as exc:
        raise VLLMExecutionPlanExportError(f"invalid shared capability profile: {path}: {exc}") from exc


def validate_plan(plan: dict[str, Any]) -> None:
    errors: list[str] = []
    if plan.get("artifact_type") != "vllm_execution_plan":
        errors.append("artifact_type must be vllm_execution_plan")
    if plan.get("schema_version") != "1.0.0":
        errors.append("schema_version must be 1.0.0")
    if plan.get("truth_boundary") != TRUTH_BOUNDARY:
        errors.append("truth_boundary must state planning intent only")
    for section in REQUIRED_SECTIONS:
        if section not in plan:
            errors.append(f"missing required section: {section}")
    if not plan.get("producer"):
        errors.append("producer is required")
    if not plan.get("source_artifacts"):
        errors.append("source_artifacts is required")
    if _contains_measured_speedup_claim(plan):
        errors.append("plan must not contain measured speedup claims")
    if errors:
        raise VLLMExecutionPlanExportError("; ".join(errors))


def write_plan(path: Path, plan: dict[str, Any]) -> None:
    validate_plan(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def _hardware_profile(profile: dict[str, Any]) -> dict[str, Any]:
    attrs = profile.get("attributes", {})
    memory = profile.get("memory", {})
    vendor = profile.get("vendor")
    model = profile.get("model")
    gpu_name = f"{vendor} {model}" if vendor and model and not str(model).startswith(str(vendor)) else model
    return {
        "profile_id": profile.get("hardware_id"),
        "gpu_name": gpu_name,
        "vendor": vendor,
        "family": profile.get("family"),
        "vram_gb": memory.get("vram_gb", round(memory.get("vram_mb", 0) / 1024, 3)),
        "vram_mb": memory.get("vram_mb"),
        "cuda_runtime": attrs.get("cuda_runtime"),
        "compute_capability": attrs.get("compute_capability"),
        "single_gpu": attrs.get("multi_gpu") is False,
        "capabilities": {
            "tensor_core_support": attrs.get("tensor_core_support"),
            "fp16": attrs.get("fp16"),
            "bf16": attrs.get("bf16"),
            "fp8": attrs.get("fp8"),
            "nvfp4": attrs.get("nvfp4"),
            "mxfp4": attrs.get("mxfp4"),
            "int8": attrs.get("int8"),
            "int4": attrs.get("int4"),
            "multi_gpu": attrs.get("multi_gpu"),
        },
    }


def _backend_profile(profile: dict[str, Any]) -> dict[str, Any]:
    supports = profile.get("supports", {})
    return {
        "backend": "vllm",
        "backend_name": profile.get("backend"),
        "backend_id": profile.get("backend_id"),
        "backend_api": profile.get("backend_api"),
        "supported_features": supports.get("features", []),
        "supported_quantization": supports.get("quantization", []),
        "supported_speculative_decoding": supports.get("speculative_decoding", {}),
        "supported_prefix_cache": supports.get("prefix_cache", {}),
        "supported_chunked_prefill": supports.get("chunked_prefill", {}),
        "supported_paged_attention": supports.get("paged_attention", {}),
        "supported_tensor_parallel": supports.get("tensor_parallel", {}),
        "supported_pipeline_parallel": supports.get("pipeline_parallel", {}),
        "does_not_claim": profile.get("does_not_claim", []),
    }


def _kernel_profile(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    available = []
    for profile in profiles:
        provider = profile.get("profile_id")
        for kernel in profile.get("kernels", []):
            available.append({
                "provider": provider,
                "operation": kernel.get("operation"),
                "availability": kernel.get("availability"),
                "support_status": kernel.get("support_status"),
                "supported_precisions": kernel.get("supported_precisions", []),
                "supported_features": kernel.get("supported_features", []),
                "measured": kernel.get("measured", False),
            })
    return {
        "providers": [profile.get("profile_id") for profile in profiles],
        "available_kernels": available,
    }


def _source_path(root: Path, relative: str) -> str:
    path = root / relative
    return os.path.relpath(path, REPO_ROOT)


def _contains_measured_speedup_claim(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered_key = str(key).lower()
            if lowered_key in {"speedup", "measured_speedup", "performance_claim"}:
                return True
            if _contains_measured_speedup_claim(item):
                return True
    if isinstance(value, list):
        return any(_contains_measured_speedup_claim(item) for item in value)
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--capabilities-root", type=Path, default=DEFAULT_CAPABILITIES_ROOT)
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--created-at-utc", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan = build_plan(
        created_at_utc=args.created_at_utc,
        git_commit=args.git_commit,
        capabilities_root=args.capabilities_root,
    )
    write_plan(args.output, plan)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
