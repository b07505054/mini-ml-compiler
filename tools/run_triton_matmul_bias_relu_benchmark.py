#!/usr/bin/env python3
"""Fixed-config Triton MatMul + Bias + ReLU fair-fusion benchmark.

PR A scope only: real Triton execution for V1 tiled-unfused and V3 tiled
one-pass fused candidates. This runner does not perform compiler selection,
ExecutionPlan emission, autotuning, or cost-modeling.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matmul_postop_workloads import (  # noqa: E402
    VALID_CATEGORIES,
    canonical_workloads,
    decision_boundary_workloads,
    geometric_mean,
    load_manifest,
)


SCHEMA = "triton_matmul_bias_relu_fixed_config_profile"
BOUNDARY_SCHEMA = "triton_matmul_bias_relu_decision_boundary_profile"
SCHEMA_VERSION = 1
BACKEND = "triton_cuda"
DTYPE = "f32"
PATTERN = "bias"
ATOL = 1e-3
RTOL = 1e-3
MEMORY_SAFETY_FRACTION = 0.80
TIE_RELATIVE_DIFFERENCE = 0.01
STABLE_CV_LIMIT = 0.05

V1_METADATA = {
    "variant": "V1",
    "kernel_id": "triton_tiled_matmul_bias_relu_unfused_f32",
    "runtime_operations": 3,
    "expected_launches": 3,
    "full_size_intermediates": 2,
    "fusion": "none",
}

V3_METADATA = {
    "variant": "V3",
    "kernel_id": "triton_tiled_matmul_bias_relu_one_pass_f32",
    "runtime_operations": 1,
    "expected_launches": 1,
    "full_size_intermediates": 0,
    "fusion": "one_pass_epilogue",
}


def run_command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    return output or None


def git_commit_hash() -> str | None:
    return run_command(["git", "rev-parse", "HEAD"])


def git_status_short() -> str:
    return run_command(["git", "status", "--short"]) or ""


def nvidia_smi_value(query: str) -> str | None:
    output = run_command(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if not output:
        return None
    return output.splitlines()[0].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("candidate-sweep", "use-plan", "fresh-oracle", "decision-boundary-sweep"),
        default="candidate-sweep",
    )
    parser.add_argument("--manifest", default="benchmarks/matmul_postop_workloads.json")
    parser.add_argument("--workload-id", action="append")
    parser.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--decision-boundary-only", action="store_true")
    parser.add_argument("--include-decision-boundary", action="store_true")
    parser.add_argument("--execution-plan")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="trace/matmul_postop_triton_fixed_config_profile.json")
    parser.add_argument("--report-output", default="trace/matmul_postop_triton_fixed_config_report.md")
    parser.add_argument("--smoke-test-mode", action="store_true")
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=16)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--precision-mode", choices=("ieee", "tf32", "tf32x3"), default="ieee")
    parser.add_argument("--memory-safety-fraction", type=float, default=MEMORY_SAFETY_FRACTION)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument(
        "--candidate-order",
        choices=("v1-then-v3", "v3-then-v1", "alternating", "randomized"),
        default="v1-then-v3",
    )
    parser.add_argument("--tie-threshold", type=float, default=TIE_RELATIVE_DIFFERENCE)
    parser.add_argument("--stable-cv-limit", type=float, default=STABLE_CV_LIMIT)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("warmup", "iterations", "repeats", "block_m", "block_n", "block_k", "num_warps", "num_stages"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    formal = args.warmup >= 50 and args.iterations >= 300 and args.repeats >= 5
    if not formal and not args.smoke_test_mode:
        raise ValueError("formal runs require warmup>=50, iterations>=300, repeats>=5")
    if args.warmup <= 0 and not args.smoke_test_mode:
        raise ValueError("warmup must be nonzero unless --smoke-test-mode is set")
    if args.mode == "use-plan" and not args.execution_plan:
        raise ValueError("--execution-plan is required in use-plan mode")
    if args.mode != "use-plan" and not args.all_eligible and not args.workload_id:
        raise ValueError("select --all-eligible or at least one --workload-id")
    if args.mode == "decision-boundary-sweep" and args.sessions < 3 and not args.smoke_test_mode:
        raise ValueError("formal decision-boundary classification requires --sessions >= 3")
    if args.sessions <= 0:
        raise ValueError("--sessions must be positive")
    if args.tie_threshold <= 0.0:
        raise ValueError("--tie-threshold must be positive")
    if args.stable_cv_limit <= 0.0:
        raise ValueError("--stable-cv-limit must be positive")
    if not (0.0 < args.memory_safety_fraction <= 1.0):
        raise ValueError("--memory-safety-fraction must be in (0, 1]")


def fixed_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "BLOCK_M": args.block_m,
        "BLOCK_N": args.block_n,
        "BLOCK_K": args.block_k,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "precision_mode": args.precision_mode,
        "input_dtype": "torch.float32",
        "accumulator_dtype": "fp32",
        "tf32_enabled": args.precision_mode != "ieee",
        "config_source": "single_fixed_config",
    }


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_times(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "sample_count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
            "stddev_ms": None,
            "coefficient_of_variation": None,
        }
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "sample_count": len(values),
        "mean_ms": mean,
        "median_ms": percentile(values, 50),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "min_ms": min(values),
        "max_ms": max(values),
        "stddev_ms": stddev,
        "coefficient_of_variation": stddev / mean if mean > 0 else 0.0,
    }


def import_torch_triton() -> tuple[Any | None, Any | None, Any | None, str | None]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, None, None, f"torch import failed: {exc!r}"
    try:
        import triton
        import triton.language as tl
    except Exception as exc:  # pragma: no cover - environment dependent
        return torch, None, None, f"triton import failed: {exc!r}"
    return torch, triton, tl, None


def environment_metadata(torch: Any | None, triton: Any | None) -> dict[str, Any]:
    nvidia_smi_text = run_command(["nvidia-smi"])
    env: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "process_id": os.getpid(),
        "git_commit": git_commit_hash(),
        "git_status_short": git_status_short(),
        "driver": nvidia_smi_value("driver_version"),
        "nvidia_smi_summary": nvidia_smi_text.splitlines()[2].strip() if nvidia_smi_text and len(nvidia_smi_text.splitlines()) > 2 else nvidia_smi_text,
        "gpu_model": nvidia_smi_value("name"),
        "vram_total_mb_nvidia_smi": nvidia_smi_value("memory.total"),
    }
    if torch is not None:
        env["pytorch_version"] = getattr(torch, "__version__", None)
        env["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env.update(
                {
                    "gpu_model_torch": torch.cuda.get_device_name(0),
                    "compute_capability": list(torch.cuda.get_device_capability(0)),
                    "vram_total_bytes": int(props.total_memory),
                    "tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
                }
            )
    if triton is not None:
        env["triton_version"] = getattr(triton, "__version__", "unknown")
    return env


def unavailable_payload(reason: str, args: argparse.Namespace, started: str, env: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "mode": "candidate-sweep",
        "profile_status": "unavailable",
        "backend": BACKEND,
        "unavailable_reason": reason,
        "environment": env,
        "benchmark_config": benchmark_config(args),
        "workloads": [],
        "aggregates": {},
        "utc_start": started,
        "utc_end": datetime.now(timezone.utc).isoformat(),
    }


def benchmark_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "seed": args.seed,
        "pattern": PATTERN,
        "dtype": DTYPE,
        "primary_speedup_statistic": "median_ms",
        "also_reports_mean": True,
        "fixed_config": fixed_config(args),
        "memory_safety_fraction": args.memory_safety_fraction,
        "execution_plan": args.execution_plan,
        "sessions": args.sessions,
        "candidate_order": args.candidate_order,
        "tie_threshold": args.tie_threshold,
        "stable_cv_limit": args.stable_cv_limit,
    }


def lower_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_m": cfg["BLOCK_M"],
        "block_n": cfg["BLOCK_N"],
        "block_k": cfg["BLOCK_K"],
        "num_warps": cfg["num_warps"],
        "num_stages": cfg["num_stages"],
        "precision_mode": cfg["precision_mode"],
    }


def upper_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "BLOCK_M": cfg.get("BLOCK_M", cfg.get("block_m")),
        "BLOCK_N": cfg.get("BLOCK_N", cfg.get("block_n")),
        "BLOCK_K": cfg.get("BLOCK_K", cfg.get("block_k")),
        "num_warps": cfg.get("num_warps"),
        "num_stages": cfg.get("num_stages"),
        "precision_mode": cfg.get("precision_mode"),
        "input_dtype": cfg.get("input_dtype", "torch.float32"),
        "accumulator_dtype": cfg.get("accumulator_dtype", "fp32"),
        "tf32_enabled": cfg.get("tf32_enabled", cfg.get("precision_mode") != "ieee"),
        "config_source": cfg.get("config_source", "execution_plan"),
    }


def config_matches(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
    return lower_config(upper_config(lhs)) == lower_config(upper_config(rhs))


def estimate_required_bytes(m: int, n: int, k: int, dtype_bytes: int = 4) -> dict[str, int]:
    a = m * k * dtype_bytes
    b = k * n * dtype_bytes
    bias = n * dtype_bytes
    output = m * n * dtype_bytes
    return {
        "A": a,
        "B": b,
        "bias": bias,
        "v1_matmul_intermediate": output,
        "v1_bias_intermediate": output,
        "output": output,
        "reference": output,
        "total": a + b + bias + (4 * output),
    }


def select_workloads(args: argparse.Namespace) -> list[Any]:
    workloads = load_manifest(args.manifest)
    if args.decision_boundary_only or args.mode == "decision-boundary-sweep":
        workloads = decision_boundary_workloads(workloads)
    elif not args.include_decision_boundary:
        workloads = canonical_workloads(workloads)
    ids = set(args.workload_id or [])
    if args.all_eligible:
        selected = workloads
    else:
        selected = [w for w in workloads if w.workload_id in ids]
    missing = ids - {w.workload_id for w in selected}
    if missing:
        raise ValueError(f"unknown workload id(s): {sorted(missing)}")
    return selected


def candidate_order(args: argparse.Namespace, session_index: int, workload_index: int) -> list[str]:
    if args.candidate_order == "v1-then-v3":
        return ["V1", "V3"]
    if args.candidate_order == "v3-then-v1":
        return ["V3", "V1"]
    if args.candidate_order == "alternating":
        return ["V1", "V3"] if (session_index + workload_index) % 2 == 0 else ["V3", "V1"]
    rng = random.Random(args.seed + session_index * 1009 + workload_index * 9176)
    order = ["V1", "V3"]
    rng.shuffle(order)
    return order


def gpu_state_snapshot(torch: Any | None = None) -> dict[str, Any]:
    state = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "temperature_gpu_c": nvidia_smi_value("temperature.gpu"),
        "graphics_clock_mhz": nvidia_smi_value("clocks.gr"),
        "memory_clock_mhz": nvidia_smi_value("clocks.mem"),
        "power_draw_w": nvidia_smi_value("power.draw"),
        "power_limit_w": nvidia_smi_value("power.limit"),
        "utilization_gpu_percent": nvidia_smi_value("utilization.gpu"),
        "memory_free_mb": nvidia_smi_value("memory.free"),
        "driver": nvidia_smi_value("driver_version"),
    }
    if torch is not None and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        state["torch_free_memory_bytes"] = int(free_bytes)
        state["torch_total_memory_bytes"] = int(total_bytes)
    return state


def make_generators(torch: Any, seed: int, device: str) -> Callable[[int, int, int], tuple[Any, Any, Any]]:
    def allocate(m: int, n: int, k: int) -> tuple[Any, Any, Any]:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + (m * 1000003) + (n * 9176) + k)
        a = torch.randn((m, k), device=device, dtype=torch.float32, generator=gen)
        b = torch.randn((k, n), device=device, dtype=torch.float32, generator=gen)
        bias = torch.randn((n,), device=device, dtype=torch.float32, generator=gen)
        return a, b, bias

    return allocate


def build_triton_kernels(triton: Any, tl: Any):
    # The V3 source block is intentionally explicit for code-inspection tests:
    # bias is added and ReLU is applied to the accumulator before the only
    # tl.store in the one-pass kernel.
    @triton.jit
    def _matmul_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                       INPUT_PRECISION: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            a = tl.load(a_ptr + offs_m[:, None] * K + k_idxs[None, :],
                        mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K), other=0.0)
            b = tl.load(b_ptr + k_idxs[:, None] * N + offs_n[None, :],
                        mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b, input_precision=INPUT_PRECISION)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    @triton.jit
    def _bias_kernel(c_ptr, bias_ptr, d_ptr, M: tl.constexpr, N: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        values = tl.load(c_ptr + offs_m[:, None] * N + offs_n[None, :],
                         mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        tl.store(d_ptr + offs_m[:, None] * N + offs_n[None, :], values + bias[None, :],
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    @triton.jit
    def _relu_kernel(d_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        values = tl.load(d_ptr + offs_m[:, None] * N + offs_n[None, :],
                         mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        out = tl.maximum(values, 0.0)
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    @triton.jit
    def _matmul_bias_relu_one_pass_kernel(a_ptr, b_ptr, bias_ptr, out_ptr,
                                          M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                          BLOCK_K: tl.constexpr, INPUT_PRECISION: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            a = tl.load(a_ptr + offs_m[:, None] * K + k_idxs[None, :],
                        mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K), other=0.0)
            b = tl.load(b_ptr + k_idxs[:, None] * N + offs_n[None, :],
                        mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b, input_precision=INPUT_PRECISION)
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
        acc = tl.maximum(acc, 0.0)
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    return _matmul_kernel, _bias_kernel, _relu_kernel, _matmul_bias_relu_one_pass_kernel


def launch_grid(m: int, n: int, cfg: dict[str, Any]) -> tuple[int, int]:
    return (math.ceil(m / cfg["BLOCK_M"]), math.ceil(n / cfg["BLOCK_N"]))


def make_variant_functions(torch: Any, triton: Any, kernels: tuple[Any, Any, Any, Any], cfg: dict[str, Any],
                           m: int, n: int, k: int, a: Any, b: Any, bias: Any):
    matmul_kernel, bias_kernel, relu_kernel, one_pass_kernel = kernels
    grid = launch_grid(m, n, cfg)
    matmul_out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    bias_out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    v1_out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    v3_out = torch.empty((m, n), device=a.device, dtype=torch.float32)

    common = {
        "M": m,
        "N": n,
        "K": k,
        "BLOCK_M": cfg["BLOCK_M"],
        "BLOCK_N": cfg["BLOCK_N"],
        "BLOCK_K": cfg["BLOCK_K"],
        "num_warps": cfg["num_warps"],
        "num_stages": cfg["num_stages"],
        "INPUT_PRECISION": cfg["precision_mode"],
    }

    def run_v1() -> Any:
        matmul_kernel[grid](a, b, matmul_out, **common)
        bias_kernel[grid](matmul_out, bias, bias_out, M=m, N=n, BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"])
        relu_kernel[grid](bias_out, v1_out, M=m, N=n, BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"])
        return v1_out

    def run_v3() -> Any:
        one_pass_kernel[grid](a, b, bias, v3_out, **common)
        return v3_out

    allocations = {
        "v1_matmul_intermediate_allocated_outside_timing": True,
        "v1_bias_intermediate_allocated_outside_timing": True,
        "v3_full_size_intermediates": 0,
    }
    return run_v1, run_v3, allocations


def correctness(torch: Any, actual: Any, reference: Any) -> dict[str, Any]:
    diff = actual - reference
    abs_err = torch.abs(diff)
    rel_err = abs_err / torch.clamp(torch.abs(reference), min=1e-12)
    contains_nan = bool(torch.isnan(actual).any().item() or torch.isnan(reference).any().item())
    contains_inf = bool(torch.isinf(actual).any().item() or torch.isinf(reference).any().item())
    passed = bool(torch.allclose(actual, reference, atol=ATOL, rtol=RTOL)) and not contains_nan and not contains_inf
    return {
        "passed": passed,
        "atol": ATOL,
        "rtol": RTOL,
        "max_abs_error": float(torch.max(abs_err).item()),
        "max_rel_error": float(torch.max(rel_err).item()),
        "contains_nan": contains_nan,
        "contains_inf": contains_inf,
        "output_shape": list(actual.shape),
    }


def measure_gpu(torch: Any, fn: Callable[[], Any], warmup: int, iterations: int, repeats: int) -> dict[str, Any]:
    samples: list[float] = []
    for _ in range(repeats):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)) / iterations)
    return {"samples_ms": samples, "statistics": summarize_times(samples)}


def benchmark_workload(torch: Any, triton: Any, tl: Any, workload: Any, args: argparse.Namespace,
                       free_bytes: int, total_bytes: int, order: list[str] | None = None) -> dict[str, Any]:
    cfg = fixed_config(args)
    required = estimate_required_bytes(workload.m, workload.n, workload.k)
    base = {
        "workload_id": workload.workload_id,
        "category": workload.category,
        "tier": workload.tier,
        "held_out": workload.held_out,
        "profile_role": "held_out" if workload.held_out else "exact_profiled",
        "shape": {"m": workload.m, "n": workload.n, "k": workload.k, "dtype": DTYPE},
        "pattern": PATTERN,
        "bias_shape": [workload.n],
        "selected_fixed_config": cfg,
        "estimated_required_bytes": required,
        "available_gpu_memory_bytes": free_bytes,
        "total_gpu_memory_bytes": total_bytes,
        "subgroup": getattr(workload, "subgroup", None),
        "benchmark_purpose": getattr(workload, "benchmark_purpose", "fusion_coverage"),
        "expected_region": getattr(workload, "expected_region", None),
    }
    if workload.status != "active":
        return {**base, "status": "skipped", "skip_reason": workload.skip_reason or workload.status}
    if workload.dtype != DTYPE or PATTERN not in workload.patterns:
        return {**base, "status": "skipped", "skip_reason": "unsupported_dtype_or_pattern_for_pr_a"}
    if required["total"] > int(free_bytes * args.memory_safety_fraction):
        return {
            **base,
            "status": "skipped",
            "skip_reason": "estimated_required_bytes_exceeds_memory_safety_fraction",
        }

    allocate = make_generators(torch, args.seed, "cuda")
    a, b, bias = allocate(workload.m, workload.n, workload.k)
    if a.dtype != torch.float32 or b.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError("PR A supports fp32 tensors only")
    if a.device != b.device or a.device != bias.device or a.device.type != "cuda":
        raise ValueError("all tensors must be on the same CUDA device")
    if a.shape != (workload.m, workload.k) or b.shape != (workload.k, workload.n) or bias.shape != (workload.n,):
        raise ValueError("invalid A/B/Bias shapes")

    kernels = build_triton_kernels(triton, tl)
    run_v1, run_v3, allocation_flags = make_variant_functions(
        torch, triton, kernels, cfg, workload.m, workload.n, workload.k, a, b, bias
    )

    jit_start = time.perf_counter()
    v1_out = run_v1()
    v3_out = run_v3()
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - jit_start) * 1000.0

    with torch.no_grad():
        reference = torch.relu(torch.matmul(a, b) + bias)
        torch.cuda.synchronize()
        v1_correctness = correctness(torch, v1_out, reference)
        v3_correctness = correctness(torch, v3_out, reference)

    order = order or ["V1", "V3"]
    timings: dict[str, Any] = {"V1": None, "V3": None}
    fns = {"V1": run_v1, "V3": run_v3}
    correctness_by_variant = {"V1": v1_correctness, "V3": v3_correctness}
    for variant in order:
        if correctness_by_variant[variant]["passed"]:
            timings[variant] = measure_gpu(torch, fns[variant], args.warmup, args.iterations, args.repeats)
    v1_timing = timings["V1"]
    v3_timing = timings["V3"]

    speedup = None
    mean_speedup = None
    status = "completed"
    if not v1_correctness["passed"] or not v3_correctness["passed"]:
        status = "failed_correctness"
    elif v1_timing and v3_timing:
        speedup = v1_timing["statistics"]["median_ms"] / v3_timing["statistics"]["median_ms"]
        mean_speedup = v1_timing["statistics"]["mean_ms"] / v3_timing["statistics"]["mean_ms"]

    return {
        **base,
        "status": status,
        "skip_reason": None,
        "first_call_jit_wall_time_ms": first_call_ms,
        "candidate_execution_order": order,
        "allocation": allocation_flags,
        "variants": {
            "V1": {**V1_METADATA, "correctness": v1_correctness, "timing": v1_timing},
            "V3": {**V3_METADATA, "correctness": v3_correctness, "timing": v3_timing},
        },
        "fair_fusion_speedup_median": speedup,
        "fair_fusion_speedup_mean": mean_speedup,
        "comparison": "V1_median_ms / V3_median_ms",
    }


def winner_from_medians(v1_ms: float, v3_ms: float, tie_threshold: float) -> str:
    relative = (v1_ms - v3_ms) / v1_ms
    if abs(relative) <= tie_threshold:
        return "tie"
    return "V3" if relative > 0.0 else "V1"


def classify_boundary_workload(session_records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if len(session_records) < 3 and not args.smoke_test_mode:
        return {
            "final_classification": "unstable",
            "classification_confidence": "low",
            "stability_reason": "requires_at_least_three_independent_sessions",
        }
    completed = [r for r in session_records if r.get("status") == "completed"]
    if len(completed) != len(session_records):
        return {
            "final_classification": "unstable",
            "classification_confidence": "low",
            "stability_reason": "skipped_or_failed_session_prevents_classification",
        }
    v1_medians = [r["variants"]["V1"]["timing"]["statistics"]["median_ms"] for r in completed]
    v3_medians = [r["variants"]["V3"]["timing"]["statistics"]["median_ms"] for r in completed]
    v1_cvs = [r["variants"]["V1"]["timing"]["statistics"]["coefficient_of_variation"] for r in completed]
    v3_cvs = [r["variants"]["V3"]["timing"]["statistics"]["coefficient_of_variation"] for r in completed]
    session_winners = [
        winner_from_medians(v1, v3, args.tie_threshold)
        for v1, v3 in zip(v1_medians, v3_medians)
    ]
    cross_v1 = statistics.median(v1_medians)
    cross_v3 = statistics.median(v3_medians)
    relative = (cross_v1 - cross_v3) / cross_v1
    speedup = cross_v1 / cross_v3
    max_cv = max(v1_cvs + v3_cvs)
    winner_counts = {name: session_winners.count(name) for name in ("V1", "V3", "tie")}

    if abs(relative) <= args.tie_threshold and all(w == "tie" for w in session_winners):
        final = "statistical_tie"
        confidence = "medium" if max_cv <= args.stable_cv_limit else "low"
        reason = "all_sessions_within_tie_threshold"
    elif relative < -args.tie_threshold and all(w == "V1" for w in session_winners) and max_cv <= args.stable_cv_limit:
        final = "stable_v1_win"
        confidence = "high"
        reason = "all_sessions_select_v1_and_cv_within_limit"
    elif relative > args.tie_threshold and all(w == "V3" for w in session_winners) and max_cv <= args.stable_cv_limit:
        final = "stable_v3_win"
        confidence = "high"
        reason = "all_sessions_select_v3_and_cv_within_limit"
    else:
        final = "unstable"
        confidence = "low"
        if max_cv > args.stable_cv_limit:
            reason = "coefficient_of_variation_exceeds_stable_limit"
        else:
            reason = "session_level_winner_changes_or_margin_is_mixed"

    return {
        "cross_session_v1_median_ms": cross_v1,
        "cross_session_v3_median_ms": cross_v3,
        "cross_session_speedup_v1_over_v3": speedup,
        "relative_difference": relative,
        "session_level_winner_counts": winner_counts,
        "session_level_winners": session_winners,
        "max_candidate_cv": max_cv,
        "final_classification": final,
        "classification_confidence": confidence,
        "stability_reason": reason,
    }


def run_decision_boundary_sweep(torch: Any, triton: Any, tl: Any, workloads: list[Any],
                                args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sessions = []
    by_workload: dict[str, list[dict[str, Any]]] = {w.workload_id: [] for w in workloads}
    for session_index in range(args.sessions):
        session_args = copy.copy(args)
        session_args.seed = args.seed + session_index * 1009
        session = {
            "session_index": session_index,
            "seed": session_args.seed,
            "candidate_order_policy": args.candidate_order,
            "utc_start": datetime.now(timezone.utc).isoformat(),
            "gpu_state_before": gpu_state_snapshot(torch),
            "workload_results": [],
        }
        for workload_index, workload in enumerate(workloads):
            torch.cuda.empty_cache()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            order = candidate_order(args, session_index, workload_index)
            record = benchmark_workload(
                torch, triton, tl, workload, session_args, int(free_bytes), int(total_bytes), order
            )
            record["session_index"] = session_index
            record["session_seed"] = session_args.seed
            record["candidate_order_policy"] = args.candidate_order
            session["workload_results"].append(
                {
                    "workload_id": workload.workload_id,
                    "status": record.get("status"),
                    "candidate_execution_order": order,
                }
            )
            by_workload[workload.workload_id].append(record)
            torch.cuda.empty_cache()
        session["gpu_state_after"] = gpu_state_snapshot(torch)
        session["utc_end"] = datetime.now(timezone.utc).isoformat()
        sessions.append(session)

    workloads_payload = []
    for workload in workloads:
        records = by_workload[workload.workload_id]
        base_record = records[0]
        classification = classify_boundary_workload(records, args)
        workloads_payload.append(
            {
                "workload_id": workload.workload_id,
                "category": workload.category,
                "group": workload.group,
                "subgroup": workload.subgroup,
                "benchmark_purpose": workload.benchmark_purpose,
                "profile_role": workload.profile_role,
                "expected_region": workload.expected_region,
                "backend_eligibility": workload.backend_eligibility,
                "shape": base_record["shape"],
                "pattern": PATTERN,
                "bias_shape": [workload.n],
                "selected_fixed_config": base_record["selected_fixed_config"],
                "estimated_required_bytes": base_record["estimated_required_bytes"],
                "available_gpu_memory_bytes": base_record["available_gpu_memory_bytes"],
                "total_gpu_memory_bytes": base_record["total_gpu_memory_bytes"],
                "status": "completed" if all(r.get("status") == "completed" for r in records) else "incomplete",
                "sessions": records,
                **classification,
            }
        )
    return sessions, workloads_payload


def validate_triton_plan(plan: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != "runtime_execution_plan":
        raise ValueError("unsupported execution plan schema")
    if plan.get("backend") != BACKEND:
        raise ValueError("wrong backend")
    operations = plan.get("operations") or []
    if len(operations) != 1:
        raise ValueError("expected exactly one operation")
    op = operations[0]
    if op.get("backend") != BACKEND:
        raise ValueError("wrong operation backend")
    if op.get("op_type") != "MatMulBiasRelu":
        raise ValueError("wrong pattern")
    if op.get("selected_kernel") not in {V1_METADATA["kernel_id"], V3_METADATA["kernel_id"]}:
        raise ValueError("unknown kernel ID")
    shape = op.get("shape") or {}
    if shape.get("dtype") != DTYPE:
        raise ValueError("unsupported dtype")
    for dim in ("m", "n", "k"):
        if not isinstance(shape.get(dim), int) or shape[dim] <= 0:
            raise ValueError("invalid shape")
    cfg = op.get("kernel_config") or {}
    required_cfg = {"block_m", "block_n", "block_k", "num_warps", "num_stages", "precision_mode"}
    if set(cfg) < required_cfg:
        raise ValueError("invalid config")
    if any(not isinstance(cfg[key], int) or cfg[key] <= 0 for key in ("block_m", "block_n", "block_k", "num_warps", "num_stages")):
        raise ValueError("invalid config")
    if cfg["precision_mode"] != "ieee":
        raise ValueError("precision mismatch")
    if op.get("inputs") != ["A", "B", "bias"] or op.get("outputs") != ["Y"]:
        raise ValueError("missing tensor IDs")
    target = op.get("target_gpu_identity") or plan.get("target_gpu_identity") or {}
    plan_gpu = target.get("gpu_model")
    env_gpu = env.get("gpu_model") or env.get("gpu_model_torch")
    if plan_gpu and env_gpu and plan_gpu != env_gpu:
        raise ValueError("GPU mismatch")
    plan_cc = target.get("compute_capability")
    env_cc = env.get("compute_capability")
    if plan_cc and env_cc and list(plan_cc) != list(env_cc):
        raise ValueError("compute capability mismatch")
    return op


def run_plan_workload(torch: Any, triton: Any, tl: Any, plan: dict[str, Any], args: argparse.Namespace,
                      env: dict[str, Any], free_bytes: int, total_bytes: int) -> dict[str, Any]:
    op = validate_triton_plan(plan, env)
    shape = op["shape"]
    cfg = upper_config(op["kernel_config"])
    cli_cfg = fixed_config(args)
    if not config_matches(cfg, cli_cfg):
        raise ValueError("plan config does not match runner fixed config")

    required = estimate_required_bytes(shape["m"], shape["n"], shape["k"])
    if required["total"] > int(free_bytes * args.memory_safety_fraction):
        raise ValueError("estimated required bytes exceed memory safety fraction")

    allocate = make_generators(torch, args.seed, "cuda")
    a, b, bias = allocate(shape["m"], shape["n"], shape["k"])
    kernels = build_triton_kernels(triton, tl)
    run_v1, run_v3, allocation_flags = make_variant_functions(
        torch, triton, kernels, cfg, shape["m"], shape["n"], shape["k"], a, b, bias
    )
    fn_by_kernel = {
        V1_METADATA["kernel_id"]: run_v1,
        V3_METADATA["kernel_id"]: run_v3,
    }
    selected_kernel = op["selected_kernel"]
    fn = fn_by_kernel[selected_kernel]

    jit_start = time.perf_counter()
    actual = fn()
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - jit_start) * 1000.0
    with torch.no_grad():
        reference = torch.relu(torch.matmul(a, b) + bias)
        torch.cuda.synchronize()
        correctness_result = correctness(torch, actual, reference)
    timing = measure_gpu(torch, fn, args.warmup, args.iterations, args.repeats) if correctness_result["passed"] else None
    actual_config = lower_config(cfg)
    planned_config = op["kernel_config"]
    return {
        "workload_id": plan.get("workload_id") or plan.get("graph_id"),
        "planned_backend": op.get("backend"),
        "actual_backend": BACKEND,
        "planned_kernel": selected_kernel,
        "actual_dispatched_kernel": selected_kernel,
        "planned_config": planned_config,
        "actual_config": actual_config,
        "planned_equals_actual": True,
        "config_equals_actual": planned_config == actual_config,
        "correctness": correctness_result,
        "timing": timing,
        "fallback_reason": op.get("fallback_reason"),
        "selection_source": op.get("selection_source"),
        "profile_match": op.get("profile_match"),
        "first_call_jit_wall_time_ms": first_call_ms,
        "allocation": allocation_flags,
        "shape": shape,
        "estimated_required_bytes": required,
        "available_gpu_memory_bytes": free_bytes,
        "total_gpu_memory_bytes": total_bytes,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        r for r in records
        if r.get("status") == "completed"
        and r.get("fair_fusion_speedup_median")
        and r["variants"]["V1"]["correctness"]["passed"]
        and r["variants"]["V3"]["correctness"]["passed"]
    ]
    skipped = [r for r in records if r.get("status") == "skipped"]
    failed = [r for r in records if r.get("status") == "failed_correctness"]
    if not completed:
        return {
            "completed_workload_count": 0,
            "skipped_workload_count": len(skipped),
            "failed_correctness_count": len(failed),
            "correctness_pass_rate": 0.0,
        }
    speedups = [r["fair_fusion_speedup_median"] for r in completed]
    return {
        "completed_workload_count": len(completed),
        "skipped_workload_count": len(skipped),
        "failed_correctness_count": len(failed),
        "correctness_pass_rate": len(completed) / max(len(completed) + len(failed), 1),
        "geomean_v1_over_v3_speedup": geometric_mean(speedups),
        "median_v1_over_v3_speedup": statistics.median(speedups),
        "best_speedup": max(speedups),
        "worst_speedup": min(speedups),
        "v3_win_rate": sum(1 for v in speedups if v > 1.0) / len(speedups),
    }


def build_aggregates(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for category in VALID_CATEGORIES:
        groups[category] = [r for r in records if r.get("category") == category]
    groups["exact_profiled"] = [r for r in records if not r.get("held_out")]
    groups["held_out"] = [r for r in records if r.get("held_out")]
    return {name: aggregate(rows) for name, rows in groups.items() if rows}


def build_boundary_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "stable_v1_win": 0,
        "stable_v3_win": 0,
        "statistical_tie": 0,
        "unstable": 0,
        "incomplete": 0,
    }
    for row in records:
        if row.get("status") != "completed":
            counts["incomplete"] += 1
        else:
            counts[row.get("final_classification", "unstable")] = counts.get(row.get("final_classification", "unstable"), 0) + 1
    return {
        "workload_count": len(records),
        "completed_workload_count": sum(1 for row in records if row.get("status") == "completed"),
        **counts,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    if payload.get("mode") == "use-plan":
        agg = payload.get("aggregate", {})
        lines = [
            "# Triton Plan-Driven MatMul-Bias-ReLU Validation",
            "",
            "This report is use-plan validation only. It is rejected as profile-selection evidence.",
            "",
            f"- planned kernel == actual kernel rate: `{agg.get('planned_kernel_equals_actual_rate')}`",
            f"- planned config == actual config rate: `{agg.get('planned_config_equals_actual_rate')}`",
            f"- correctness pass rate: `{agg.get('correctness_pass_rate')}`",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    if payload.get("mode") == "decision-boundary-sweep":
        env = payload["environment"]
        agg = payload.get("aggregate", {})
        lines = [
            "# Triton MatMul-Bias-ReLU Decision Boundary Benchmark",
            "",
            "This report is measured boundary evidence only. It does not train or evaluate an analytical selector.",
            "",
            "## Environment",
            "",
        ]
        for key in ("hostname", "gpu_model", "compute_capability", "driver", "torch_cuda", "pytorch_version", "triton_version", "git_commit"):
            lines.append(f"- {key}: `{env.get(key)}`")
        cfg = payload["fixed_config"]
        lines += [
            "",
            "## Methodology",
            "",
            f"- warmup: `{payload['benchmark_config']['warmup']}`",
            f"- iterations: `{payload['benchmark_config']['iterations']}`",
            f"- repeats: `{payload['benchmark_config']['repeats']}`",
            f"- independent sessions: `{payload['benchmark_config']['sessions']}`",
            f"- candidate order: `{payload['benchmark_config']['candidate_order']}`",
            f"- tie threshold: `{payload['classification_threshold']['tie_relative_difference']}`",
            f"- fixed config: `{cfg['block_m']}x{cfg['block_n']}x{cfg['block_k']}/w{cfg['num_warps']}/s{cfg['num_stages']}/{cfg['precision_mode']}`",
            "",
            "## Class Balance",
            "",
            f"- stable V1 wins: `{agg.get('stable_v1_win', 0)}`",
            f"- stable V3 wins: `{agg.get('stable_v3_win', 0)}`",
            f"- statistical ties: `{agg.get('statistical_tie', 0)}`",
            f"- unstable: `{agg.get('unstable', 0)}`",
            f"- incomplete: `{agg.get('incomplete', 0)}`",
            "",
            "## Accepted Boundary Workloads",
            "",
            "| Workload | M | N | K | Est. bytes | Session winners | V1 median ms | V3 median ms | V1/V3 | Rel diff | Classification | Confidence |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for row in payload["workloads"]:
            shape = row["shape"]
            lines.append(
                f"| {row['workload_id']} | {shape['m']} | {shape['n']} | {shape['k']} | "
                f"{row['estimated_required_bytes']['total']} | "
                f"{','.join(row.get('session_level_winners', []))} | "
                f"{row.get('cross_session_v1_median_ms', 0.0):.6f} | "
                f"{row.get('cross_session_v3_median_ms', 0.0):.6f} | "
                f"{row.get('cross_session_speedup_v1_over_v3', 0.0):.4f} | "
                f"{row.get('relative_difference', 0.0) * 100:.2f}% | "
                f"{row.get('final_classification')} | {row.get('classification_confidence')} |"
            )
        lines += [
            "",
            "## Canonical Versus Boundary Suite",
            "",
            "The canonical suite remains the fusion-benefit, representative, stress, and correctness coverage set. "
            "The decision-boundary tier is separate and is intended to evaluate whether a compiler can distinguish V1, V3, and tie regions.",
            "",
            "## Boundary Interpretation",
            "",
            "These workloads intentionally emphasize very small M and very large K. In that region, MatMul work dominates and "
            "the launch/intermediate-store savings of one-pass fusion shrink toward measurement noise. This is descriptive "
            "analysis only; no analytical model is fit in this PR.",
            "",
            "## Future Split Recommendation",
            "",
            "PR C should use grouped held-out splits, preferably leave-one-K-band-out first, with leave-one-N-family-out as a stress check. "
            "The held-out group must contain V1, V3, and tie/near-boundary cases.",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines = [
        "# Triton Fixed-Config MatMul-Bias-ReLU Fair Fusion Benchmark",
        "",
        "This report covers PR A only: Triton execution and fair V1/V3 fusion evidence. It does not claim compiler-selection improvement.",
        "",
        "## Environment",
        "",
    ]
    env = payload["environment"]
    for key in ("hostname", "gpu_model", "compute_capability", "driver", "torch_cuda", "pytorch_version", "triton_version", "git_commit"):
        lines.append(f"- {key}: `{env.get(key)}`")
    lines += [
        "",
        "## Aggregate Results",
        "",
        "| Group | Completed | Skipped | Correctness pass rate | Geo mean speedup | Median speedup | Best | Worst | V3 win rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, agg in payload.get("aggregates", {}).items():
        lines.append(
            f"| {name} | {agg.get('completed_workload_count', 0)} | {agg.get('skipped_workload_count', 0)} | "
            f"{agg.get('correctness_pass_rate', 0.0):.4f} | {agg.get('geomean_v1_over_v3_speedup', 0.0):.4f} | "
            f"{agg.get('median_v1_over_v3_speedup', 0.0):.4f} | {agg.get('best_speedup', 0.0):.4f} | "
            f"{agg.get('worst_speedup', 0.0):.4f} | {agg.get('v3_win_rate', 0.0):.4f} |"
        )
    lines += [
        "",
        "## Per-Workload Results",
        "",
        "| Workload | M | N | K | Config | V1 median ms | V3 median ms | V1/V3 | Correct | Status |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["workloads"]:
        shape = row["shape"]
        cfg = row["selected_fixed_config"]
        cfg_label = f"{cfg['BLOCK_M']}x{cfg['BLOCK_N']}x{cfg['BLOCK_K']}/w{cfg['num_warps']}/s{cfg['num_stages']}/{cfg['precision_mode']}"
        if row.get("status") == "completed":
            v1 = row["variants"]["V1"]["timing"]["statistics"]["median_ms"]
            v3 = row["variants"]["V3"]["timing"]["statistics"]["median_ms"]
            correct = row["variants"]["V1"]["correctness"]["passed"] and row["variants"]["V3"]["correctness"]["passed"]
            speedup = row["fair_fusion_speedup_median"]
            lines.append(
                f"| {row['workload_id']} | {shape['m']} | {shape['n']} | {shape['k']} | {cfg_label} | "
                f"{v1:.6f} | {v3:.6f} | {speedup:.4f} | {correct} | completed |"
            )
        else:
            lines.append(
                f"| {row['workload_id']} | {shape['m']} | {shape['n']} | {shape['k']} | {cfg_label} |  |  |  | false | "
                f"{row.get('status')}: {row.get('skip_reason')} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_args(args)
    started = datetime.now(timezone.utc).isoformat()
    torch, triton, tl, import_error = import_torch_triton()
    env = environment_metadata(torch, triton)
    output = Path(args.output)
    report = Path(args.report_output)
    if import_error:
        payload = unavailable_payload(import_error, args, started, env)
        write_payload(output, payload)
        write_report(report, payload)
        print(json.dumps(payload, indent=2))
        return 0
    if not torch.cuda.is_available():
        payload = unavailable_payload("CUDA is not available", args, started, env)
        write_payload(output, payload)
        write_report(report, payload)
        print(json.dumps(payload, indent=2))
        return 0

    torch.backends.cuda.matmul.allow_tf32 = args.precision_mode != "ieee"
    torch.manual_seed(args.seed)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if args.mode == "use-plan":
        plan_path = Path(args.execution_plan)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        record = run_plan_workload(torch, triton, tl, plan, args, env, int(free_bytes), int(total_bytes))
        records = [record]
        payload = {
            "schema": "triton_matmul_bias_relu_plan_validation",
            "schema_version": 1,
            "mode": "use-plan",
            "profile_status": "not_profile_evidence",
            "backend": BACKEND,
            "environment": environment_metadata(torch, triton),
            "benchmark_config": benchmark_config(args),
            "execution_plan_path": str(plan_path),
            "execution_plan_sha256": plan_sha256,
            "workloads": records,
            "aggregate": {
                "workload_count": len(records),
                "planned_kernel_equals_actual_rate": sum(1 for r in records if r["planned_equals_actual"]) / len(records),
                "planned_config_equals_actual_rate": sum(1 for r in records if r["config_equals_actual"]) / len(records),
                "correctness_pass_rate": sum(1 for r in records if r["correctness"]["passed"]) / len(records),
            },
            "utc_start": started,
            "utc_end": datetime.now(timezone.utc).isoformat(),
        }
        write_payload(output, payload)
        write_report(report, payload)
        print(json.dumps(payload, indent=2))
        return 0

    workloads = select_workloads(args)
    if args.mode == "decision-boundary-sweep":
        sessions, records = run_decision_boundary_sweep(torch, triton, tl, workloads, args)
        payload = {
            "schema": BOUNDARY_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "mode": "decision-boundary-sweep",
            "profile_status": "measured",
            "backend": BACKEND,
            "candidate_set": ["V1", "V3"],
            "classification_threshold": {
                "tie_relative_difference": args.tie_threshold,
                "stable_cv_limit": args.stable_cv_limit,
            },
            "fixed_config": lower_config(fixed_config(args)),
            "environment": environment_metadata(torch, triton),
            "benchmark_config": benchmark_config(args),
            "workload_manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            "sessions": sessions,
            "workloads": records,
            "aggregate": build_boundary_aggregate(records),
            "truth_boundary": "measured_boundary_evidence_not_compiler_selection_input",
            "utc_start": started,
            "utc_end": datetime.now(timezone.utc).isoformat(),
        }
        write_payload(output, payload)
        write_report(report, payload)
        print(json.dumps(payload, indent=2))
        return 0

    records = [
        benchmark_workload(torch, triton, tl, workload, args, int(free_bytes), int(total_bytes))
        for workload in workloads
    ]
    payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "profile_status": "measured",
        "backend": BACKEND,
        "environment": environment_metadata(torch, triton),
        "benchmark_config": benchmark_config(args),
        "workload_manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "workloads": records,
        "aggregates": build_aggregates(records),
        "utc_start": started,
        "utc_end": datetime.now(timezone.utc).isoformat(),
    }
    write_payload(output, payload)
    write_report(report, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
