#!/usr/bin/env python3
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLUGIN = ROOT / "build-mlir-codex" / "HIRMatMulBiasReluFusionPass.dylib"
IMPORTER = ROOT / "tools" / "import_stablehlo_subset.py"
RMSNORM_STABLEHLO = ROOT / "trace" / "jax_stablehlo_rmsnorm.mlir"
MATMUL_STABLEHLO = ROOT / "trace" / "jax_stablehlo_matmul_bias_relu.mlir"
RESULT_JSON = ROOT / "trace" / "jax_frontend_pipeline_result.json"

RMSNORM_TO_HIR = "builtin.module(stablehlo-compatible-rmsnorm-import)"
MATMUL_TO_HIR = "builtin.module(hir-canonicalize,matmul-bias-relu-fusion,hir-fusion-lowering,hir-verify-fused-ops)"
HIR_TO_LLVM = (
    "builtin.module("
    "hir-rmsnorm-to-linalg,"
    "one-shot-bufferize{bufferize-function-boundaries},"
    "convert-linalg-to-loops,"
    "convert-scf-to-cf,"
    "convert-index-to-llvm,"
    "convert-math-to-llvm,"
    "convert-arith-to-llvm,"
    "finalize-memref-to-llvm,"
    "convert-func-to-llvm,"
    "convert-cf-to-llvm,"
    "reconcile-unrealized-casts"
    ")"
)
MATMUL_HIR_TO_LLVM = (
    "builtin.module("
    "hir-matmul-bias-relu-to-linalg,"
    "one-shot-bufferize{bufferize-function-boundaries},"
    "convert-linalg-to-loops,"
    "convert-scf-to-cf,"
    "convert-index-to-llvm,"
    "convert-math-to-llvm,"
    "convert-arith-to-llvm,"
    "finalize-memref-to-llvm,"
    "convert-func-to-llvm,"
    "convert-cf-to-llvm,"
    "reconcile-unrealized-casts"
    ")"
)

RMSNORM_MAIN = """func.func @main() -> f32 attributes {llvm.emit_c_interface} {
  %x = arith.constant dense<[
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 0.0, -2.0, 4.0]
  ]> : tensor<2x4xf32>
  %out = hir.fused_rmsnorm %x {
    frontend.source = "jax_stablehlo",
    fusion.candidate = "rmsnorm",
    kernel.selection = "native_cpu",
    lowering.source = "llm.rmsnorm"
  } : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %value = tensor.extract %out[%c0, %c1] : tensor<2x4xf32>
  return %value : f32
}
"""

MATMUL_MAIN = """func.func @main() -> f32 attributes {llvm.emit_c_interface} {
  %lhs = arith.constant dense<[
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 0.0, -2.0, 4.0]
  ]> : tensor<2x4xf32>
  %rhs = arith.constant dense<[
    [1.0, 0.0, -1.0],
    [0.5, 2.0, 1.0],
    [1.0, -1.0, 0.0],
    [0.0, 1.0, 2.0]
  ]> : tensor<4x3xf32>
  %bias = arith.constant dense<[
    [0.1, -10.0, 0.2],
    [-1.0, 0.5, 0.0]
  ]> : tensor<2x3xf32>
  %out = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "native_cpu",
    lowering.source = "linalg.matmul_add_relu"
  } : (tensor<2x4xf32>, tensor<4x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %c0 = arith.constant 0 : index
  %value = tensor.extract %out[%c0, %c0] : tensor<2x3xf32>
  return %value : f32
}
"""

DYNAMIC_SHAPE_FIXTURE = ROOT / "mlir_passes" / "test" / "stablehlo_bad_matmul_dynamic_shape.mlir"


def command_path(env_name, default_name):
    override = os.environ.get(env_name)
    if override:
        return override
    found = shutil.which(default_name)
    if found:
        return found
    brew = Path("/opt/homebrew/opt/llvm/bin") / default_name
    return str(brew) if brew.exists() else None


def run(args):
    return subprocess.run(
        args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def mlir_opt(mlir_opt_path, plugin, input_path, pipeline):
    return run([
        mlir_opt_path,
        str(input_path),
        f"--load-dialect-plugin={plugin}",
        f"--load-pass-plugin={plugin}",
        f"--pass-pipeline={pipeline}",
    ])


def write_result(payload):
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def median(values):
    values = sorted(values)
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def benchmark(callable_fn, runs=25, warmup=5):
    for _ in range(warmup):
        callable_fn()
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        callable_fn()
        latencies.append((time.perf_counter() - start) * 1000.0)
    return {
        "runs": runs,
        "warmup": warmup,
        "p50_ms": round(median(latencies), 5),
        "min_ms": round(min(latencies), 5),
        "max_ms": round(max(latencies), 5),
    }


def time_once(callable_fn):
    start = time.perf_counter()
    value = callable_fn()
    return value, (time.perf_counter() - start) * 1000.0


def describe_device(device):
    return {
        "repr": str(device),
        "id": getattr(device, "id", None),
        "platform": getattr(device, "platform", None),
        "device_kind": getattr(device, "device_kind", None),
    }


def object_type(value):
    return f"{type(value).__module__}.{type(value).__name__}"


def memory_stats_to_dict(stats):
    if stats is None:
        return None
    fields = [
        "generated_code_size_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_generated_code_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
    ]
    return {field: getattr(stats, field) for field in fields if hasattr(stats, field)}


def safe_call(callable_fn, default=None):
    try:
        return callable_fn()
    except Exception:
        return default


def summarize_stablehlo(text):
    matched_ops = re.findall(r'\b(stablehlo\.[A-Za-z0-9_]+)\b', text)
    ops = sorted(set(matched_ops))
    return {
        "contains_stablehlo": "stablehlo." in text,
        "contains_reduce": "stablehlo.reduce" in text,
        "contains_dot_general": "stablehlo.dot_general" in text,
        "contains_maximum": "stablehlo.maximum" in text,
        "op_count": len(matched_ops),
        "unique_ops": ops,
    }


def benchmark_compiled_jax(callable_fn, sample_args, runs=25, warmup=5):
    lowered, lower_latency_ms = time_once(lambda: callable_fn.lower(*sample_args))
    compiled, compile_latency_ms = time_once(lowered.compile)
    runtime_executable = safe_call(compiled.runtime_executable)
    memory_analysis = safe_call(compiled.memory_analysis)
    cost_analysis = safe_call(compiled.cost_analysis, {})
    first_output, first_run_latency_ms = time_once(lambda: compiled(*sample_args).block_until_ready())
    warm_latency = benchmark(lambda: compiled(*sample_args).block_until_ready(), runs=runs, warmup=warmup)
    return {
        "lowered": lowered,
        "compiled": compiled,
        "first_output": first_output,
        "lower_latency_ms": round(lower_latency_ms, 5),
        "compile_latency_ms": round(compile_latency_ms, 5),
        "first_run_latency_ms": round(first_run_latency_ms, 5),
        "warm_run_latency_ms": warm_latency,
        "compiled_executable_type": object_type(compiled),
        "runtime_executable_type": object_type(runtime_executable) if runtime_executable is not None else None,
        "memory_analysis": memory_stats_to_dict(memory_analysis),
        "cost_analysis": cost_analysis,
    }


def run_importer(importer_path, source, output, metadata_output):
    return run([
        sys.executable,
        str(importer_path),
        str(source),
        "--output",
        str(output),
        "--metadata-output",
        str(metadata_output),
    ])


def build_dynamic_shape_fallback(importer_path, tmp_dir):
    if not DYNAMIC_SHAPE_FIXTURE.exists():
        return {
            "input": str(DYNAMIC_SHAPE_FIXTURE),
            "decision": "fallback_to_jax_pjrt_reference",
            "fallback_reason": "dynamic_shape_fixture_missing",
            "lowering_attempted": False,
        }
    output = tmp_dir / "dynamic_shape_should_not_import.mlir"
    metadata_output = tmp_dir / "dynamic_shape_rejection.json"
    importer = run_importer(importer_path, DYNAMIC_SHAPE_FIXTURE, output, metadata_output)
    metadata = load_json(metadata_output) if metadata_output.exists() else {}
    return {
        "input": str(DYNAMIC_SHAPE_FIXTURE),
        "decision": "fallback_to_jax_pjrt_reference",
        "fallback_reason": metadata.get("fallback_reason") or "unknown",
        "detail": metadata.get("detail"),
        "lowering_attempted": False,
        "importer_rejected": importer.returncode != 0,
        "linalg_output_emitted": output.exists(),
    }


def build_shape_specialization_cases(jax, jnp, rmsnorm):
    cases = []
    for tokens, hidden in [(1, 64), (16, 64), (128, 64)]:
        x = jnp.arange(tokens * hidden, dtype=jnp.float32).reshape((tokens, hidden)) / 100.0
        compiled = benchmark_compiled_jax(jax.jit(rmsnorm), (x,), runs=10, warmup=2)
        eager = rmsnorm(x)
        max_error = float(jnp.max(jnp.abs(eager - compiled["first_output"])))
        cases.append({
            "shape_bucket": f"{tokens}x{hidden}:f32",
            "input_shape": [tokens, hidden],
            "compiler_runtime_decision": "compile shape-specialized JAX executable and reuse it for warm runs",
            "compile_latency_ms": compiled["compile_latency_ms"],
            "first_run_latency_ms": compiled["first_run_latency_ms"],
            "warm_run_latency_ms": compiled["warm_run_latency_ms"],
            "max_error_vs_jax_eager": max_error,
            "correct": max_error < 1.0e-5,
            "memory_analysis": compiled["memory_analysis"],
            "cost_analysis": compiled["cost_analysis"],
        })
    return cases


def build_tiny_transformer_case(jax, jnp, rmsnorm):
    def tiny_block(x, weight, bias):
        normalized = rmsnorm(x)
        return jnp.maximum(jnp.matmul(normalized, weight) + bias, 0.0)

    tokens, hidden, output = 16, 64, 32
    x = jnp.arange(tokens * hidden, dtype=jnp.float32).reshape((tokens, hidden)) / 100.0
    weight = jnp.ones((hidden, output), dtype=jnp.float32) * 0.01
    bias = jnp.ones((tokens, output), dtype=jnp.float32) * 0.1
    lowered = jax.jit(tiny_block).lower(x, weight, bias)
    stablehlo_text = str(lowered.compiler_ir("stablehlo"))
    compiled = benchmark_compiled_jax(jax.jit(tiny_block), (x, weight, bias), runs=10, warmup=2)
    eager = tiny_block(x, weight, bias)
    max_error = float(jnp.max(jnp.abs(eager - compiled["first_output"])))
    return {
        "input": "JAX tiny block: RMSNorm -> MatMul -> Bias -> ReLU",
        "compiler_decision": "export StableHLO and verify RMSNorm and dot/add/relu patterns are visible for local HIR import candidates",
        "runtime_boundary": "JAX/PJRT-backed CPU compile/run reference for a multi-op block",
        "shape": {
            "tokens": tokens,
            "hidden": hidden,
            "output": output,
            "dtype": "f32",
        },
        "stablehlo_summary": summarize_stablehlo(stablehlo_text),
        "lowered_object_type": object_type(lowered),
        "compiled_executable_type": compiled["compiled_executable_type"],
        "runtime_executable_type": compiled["runtime_executable_type"],
        "compile_latency_ms": compiled["compile_latency_ms"],
        "first_run_latency_ms": compiled["first_run_latency_ms"],
        "warm_run_latency_ms": compiled["warm_run_latency_ms"],
        "memory_analysis": compiled["memory_analysis"],
        "cost_analysis": compiled["cost_analysis"],
        "max_error_vs_jax_eager": max_error,
        "correct": max_error < 1.0e-5,
    }


def tiled_matmul_bias_relu(lhs, rhs, bias, tile_m, tile_n, tile_k):
    import numpy as np

    m, k = lhs.shape
    _, n = rhs.shape
    out = np.zeros((m, n), dtype=np.float32)
    for mi in range(0, m, tile_m):
        for nj in range(0, n, tile_n):
            m_end = min(mi + tile_m, m)
            n_end = min(nj + tile_n, n)
            block = np.zeros((m_end - mi, n_end - nj), dtype=np.float32)
            for kk in range(0, k, tile_k):
                k_end = min(kk + tile_k, k)
                block += lhs[mi:m_end, kk:k_end] @ rhs[kk:k_end, nj:n_end]
            block += bias[mi:m_end, nj:n_end]
            out[mi:m_end, nj:n_end] = np.maximum(block, 0.0)
    return out


def round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def padding_crop_metadata(m, n, k, tile_m, tile_n, tile_k):
    padded_m = round_up(m, tile_m)
    padded_n = round_up(n, tile_n)
    padded_k = round_up(k, tile_k)
    compute = (padded_m * padded_n * padded_k) / max(m * n * k, 1)
    output = (padded_m * padded_n) / max(m * n, 1)
    return {
        "requires_padding_crop": (padded_m, padded_n, padded_k) != (m, n, k),
        "original_shape": {"m": m, "n": n, "k": k},
        "padded_shape": {"m": padded_m, "n": padded_n, "k": padded_k},
        "padding_compute_overhead_ratio": round(compute, 6),
        "padding_output_overhead_ratio": round(output, 6),
    }


def build_tile_autotuning_case():
    import numpy as np

    candidates = [
        {"tile_m": 8, "tile_n": 16, "tile_k": 32},
        {"tile_m": 16, "tile_n": 16, "tile_k": 32},
        {"tile_m": 16, "tile_n": 32, "tile_k": 32},
        {"tile_m": 32, "tile_n": 32, "tile_k": 64},
    ]
    shapes = [(16, 128, 64), (127, 255, 129), (17, 65, 33), (128, 128, 128)]
    shape_results = []
    rng = np.random.default_rng(7)
    for m, k, n in shapes:
        lhs = rng.normal(size=(m, k)).astype(np.float32)
        rhs = rng.normal(size=(k, n)).astype(np.float32)
        bias = rng.normal(size=(m, n)).astype(np.float32)
        expected = np.maximum(lhs @ rhs + bias, 0.0)
        candidate_results = []
        for candidate in candidates:
            padding = padding_crop_metadata(
                m,
                n,
                k,
                candidate["tile_m"],
                candidate["tile_n"],
                candidate["tile_k"],
            )
            legal = (
                padding["padding_compute_overhead_ratio"] <= 1.25 and
                padding["padding_output_overhead_ratio"] <= 1.25
            )
            if not legal:
                reason = (
                    "fallback_padding_compute_overhead_too_high"
                    if padding["padding_compute_overhead_ratio"] > 1.25
                    else "fallback_padding_output_overhead_too_high"
                )
                candidate_results.append({
                    **candidate,
                    "legal": False,
                    "correct": False,
                    "reject_reason": reason,
                    **padding,
                })
                continue
            first = tiled_matmul_bias_relu(lhs, rhs, bias, **candidate)
            max_error = float(np.max(np.abs(first - expected)))
            correct = max_error < 1.0e-4
            timing = benchmark(lambda: tiled_matmul_bias_relu(lhs, rhs, bias, **candidate), runs=7, warmup=2)
            candidate_results.append({
                **candidate,
                "legal": True,
                "correct": correct,
                "max_error": max_error,
                "latency_ms": timing,
                "decision": (
                    "selected_padded_crop_tile"
                    if padding["requires_padding_crop"]
                    else "selected_exact_tile"
                ),
                **padding,
            })
        valid = [item for item in candidate_results if item.get("legal") and item.get("correct")]
        selected = min(valid, key=lambda item: item["latency_ms"]["p50_ms"]) if valid else None
        shape_results.append({
            "shape_bucket": f"{m}x{k}x{n}:f32",
            "input": "hir.fused_matmul_bias_relu candidate tile search",
            "compiler_decision": "select fastest correct legal tile candidate for MatMul-Bias-ReLU CPU lowering",
            "metric": "median latency over repeated runs plus correctness",
            "candidates": candidate_results,
            "selected_tile": {
                key: selected[key] for key in ["tile_m", "tile_n", "tile_k", "latency_ms"]
            } if selected else None,
            "fallback_selected": selected is None,
        })
    return {
        "input": "HIR fused MatMul-Bias-ReLU shape buckets",
        "decision": "measured tile search chooses lowering/runtime tile metadata",
        "metric": "correctness-gated median latency",
        "shape_results": shape_results,
    }


def export_jax_stablehlo():
    import jax
    import jax.numpy as jnp
    import numpy as np

    def rmsnorm(x):
        return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1.0e-6)

    def matmul_bias_relu(lhs, rhs, bias):
        return jnp.maximum(jnp.matmul(lhs, rhs) + bias, 0.0)

    host_x = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 0.0, -2.0, 4.0],
        ],
        dtype=np.float32,
    )
    x = jnp.asarray(host_x)
    lhs = jnp.ones((16, 128), dtype=jnp.float32)
    rhs = jnp.ones((128, 64), dtype=jnp.float32)
    bias = jnp.ones((16, 64), dtype=jnp.float32)
    rmsnorm_text = str(jax.jit(rmsnorm).lower(x).compiler_ir("stablehlo"))
    matmul_text = str(jax.jit(matmul_bias_relu).lower(lhs, rhs, bias).compiler_ir("stablehlo"))

    RMSNORM_STABLEHLO.parent.mkdir(parents=True, exist_ok=True)
    RMSNORM_STABLEHLO.write_text(rmsnorm_text, encoding="utf-8")
    MATMUL_STABLEHLO.write_text(matmul_text, encoding="utf-8")

    expected = float(rmsnorm(x)[0, 1])
    lowered, lower_latency_ms = time_once(lambda: jax.jit(rmsnorm).lower(x))
    compiled, compile_latency_ms = time_once(lowered.compile)
    runtime_executable = safe_call(compiled.runtime_executable)
    compiled_memory = safe_call(compiled.memory_analysis)
    compiled_cost = safe_call(compiled.cost_analysis, {})

    device_input, host_to_device_ms = time_once(lambda: jax.device_put(host_x))
    device_input.block_until_ready()
    first_output, first_run_latency_ms = time_once(lambda: compiled(device_input).block_until_ready())
    compiled_value = float(first_output[0, 1])
    warm_latency = benchmark(lambda: compiled(device_input).block_until_ready())
    materialized, device_to_host_ms = time_once(lambda: np.asarray(first_output))
    end_to_end_latency = benchmark(
        lambda: np.asarray(compiled(jax.device_put(host_x)).block_until_ready()),
        runs=10,
        warmup=2,
    )
    device = getattr(device_input, "device", None)
    if callable(device):
        device = device()
    return {
        "jax_version": jax.__version__,
        "jax_backend_platform": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_device_details": [describe_device(device) for device in jax.devices()],
        "rmsnorm_input": x,
        "rmsnorm_expected_scalar": expected,
        "rmsnorm_compiled_scalar": compiled_value,
        "jax_compiled_latency_ms": warm_latency,
        "runtime_boundary": {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "device_details": [describe_device(device) for device in jax.devices()],
            "input_device": str(device),
            "input_array_type": object_type(device_input),
            "lowered_object_type": object_type(lowered),
            "compiled_executable_type": object_type(compiled),
            "runtime_executable_type": object_type(runtime_executable) if runtime_executable is not None else None,
            "lower_latency_ms": round(lower_latency_ms, 5),
            "compile_latency_ms": round(compile_latency_ms, 5),
            "first_run_latency_ms": round(first_run_latency_ms, 5),
            "warm_run_latency_ms": warm_latency,
            "memory_analysis": memory_stats_to_dict(compiled_memory),
            "cost_analysis": compiled_cost,
            "claim_level": "jax_pjrt_backed_reference_not_custom_pjrt_runtime",
        },
        "device_buffer_timing": {
            "input": "NumPy host array placed onto JAX device array",
            "runtime_boundary": "host placement -> compiled executable consumes device array -> host materialization",
            "host_to_device_placement_ms": round(host_to_device_ms, 5),
            "compiled_device_execution_first_run_ms": round(first_run_latency_ms, 5),
            "compiled_device_execution_warm_ms": warm_latency,
            "device_to_host_materialization_ms": round(device_to_host_ms, 5),
            "end_to_end_host_to_host_ms": end_to_end_latency,
            "materialized_shape": list(materialized.shape),
            "materialized_dtype": str(materialized.dtype),
        },
        "shape_specialization": {
            "input": "JAX RMSNorm shape buckets",
            "compiler_runtime_decision": "compile per static shape and reuse compiled executable for warm runs",
            "metric": "compile latency, first-run latency, warm-run latency, memory/cost analysis, correctness",
            "cases": build_shape_specialization_cases(jax, jnp, rmsnorm),
        },
        "tile_autotuning": build_tile_autotuning_case(),
        "tiny_transformer_block": build_tiny_transformer_case(jax, jnp, rmsnorm),
        "rmsnorm_text": rmsnorm_text,
        "matmul_text": matmul_text,
    }


def main():
    base = {
        "artifact_type": "jax_frontend_pipeline_result",
        "source": "tools/run_jax_stablehlo_pipeline.py",
        "input": "JAX lowered StableHLO from RMSNorm and MatMul-Bias-ReLU functions",
        "compiler_decision": "import supported StableHLO patterns into HIR fused ops and lower RMSNorm to LLVM CPU",
        "metric": "StableHLO export, HIR fused-op detection, LLVM lowering success, JAX/PJRT-backed CPU runtime-boundary timing, mlir-runner correctness",
        "stablehlo_files": {
            "rmsnorm": str(RMSNORM_STABLEHLO),
            "matmul_bias_relu": str(MATMUL_STABLEHLO),
        },
    }
    if importlib.util.find_spec("jax") is None:
        payload = {
            **base,
            "status": "skipped_missing_jax",
            "reason": "Install JAX in the repo venv: .venv/bin/python -m pip install -U 'jax[cpu]'",
        }
        write_result(payload)
        print(payload["status"])
        return 0

    mlir_opt_path = command_path("MLIR_OPT", "mlir-opt")
    mlir_runner = command_path("MLIR_RUNNER", "mlir-runner")
    plugin = Path(os.environ.get("PLUGIN", str(DEFAULT_PLUGIN)))
    runner_utils = Path(os.environ.get(
        "MLIR_C_RUNNER_UTILS",
        "/opt/homebrew/opt/llvm/lib/libmlir_c_runner_utils.dylib",
    ))
    missing = []
    for name, value in [("mlir-opt", mlir_opt_path), ("mlir-runner", mlir_runner)]:
        if not value:
            missing.append(name)
    for path in [plugin, runner_utils, IMPORTER]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        payload = {**base, "status": "failed_missing_tool", "reason": "missing " + ", ".join(missing)}
        write_result(payload)
        print(payload["reason"])
        return 1

    try:
        exported = export_jax_stablehlo()
    except Exception as exc:
        payload = {**base, "status": "failed_jax_export", "reason": str(exc)}
        write_result(payload)
        print(payload["reason"])
        return 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        rmsnorm_linalg = tmp / "jax_rmsnorm_linalg.mlir"
        matmul_linalg = tmp / "jax_matmul_linalg.mlir"
        parser_metadata = {}
        for key, source, output in [
            ("rmsnorm", RMSNORM_STABLEHLO, rmsnorm_linalg),
            ("matmul_bias_relu", MATMUL_STABLEHLO, matmul_linalg),
        ]:
            metadata_output = tmp / f"{key}_stablehlo_parser.json"
            importer = run_importer(IMPORTER, source, output, metadata_output)
            if importer.returncode != 0:
                parser_metadata[key] = load_json(metadata_output) if metadata_output.exists() else {
                    "legal": False,
                    "decision": "reject_before_linalg_import",
                    "fallback_reason": "unknown_import_failure",
                }
                payload = {
                    **base,
                    "status": "failed_stablehlo_import",
                    "reason": importer.stderr or importer.stdout,
                    "stablehlo_parser": parser_metadata,
                }
                write_result(payload)
                print(payload["reason"])
                return 1
            parser_metadata[key] = load_json(metadata_output)
        dynamic_shape_fallback = build_dynamic_shape_fallback(IMPORTER, tmp)

        rmsnorm_hir_result = mlir_opt(mlir_opt_path, plugin, rmsnorm_linalg, RMSNORM_TO_HIR)
        if rmsnorm_hir_result.returncode != 0:
            payload = {**base, "status": "failed_rmsnorm_hir", "reason": rmsnorm_hir_result.stderr}
            write_result(payload)
            print(payload["reason"])
            return 1
        matmul_hir_result = mlir_opt(mlir_opt_path, plugin, matmul_linalg, MATMUL_TO_HIR)
        if matmul_hir_result.returncode != 0:
            payload = {**base, "status": "failed_matmul_hir", "reason": matmul_hir_result.stderr}
            write_result(payload)
            print(payload["reason"])
            return 1

        main_hir = tmp / "jax_rmsnorm_main_hir.mlir"
        main_hir.write_text(RMSNORM_MAIN, encoding="utf-8")
        llvm_result = mlir_opt(mlir_opt_path, plugin, main_hir, HIR_TO_LLVM)
        if llvm_result.returncode != 0:
            payload = {**base, "status": "failed_rmsnorm_llvm", "reason": llvm_result.stderr}
            write_result(payload)
            print(payload["reason"])
            return 1
        lowered = tmp / "jax_rmsnorm_main_llvm.mlir"
        lowered.write_text(llvm_result.stdout, encoding="utf-8")

        runner_args = [
            mlir_runner,
            str(lowered),
            "-e",
            "main",
            "--entry-point-result=f32",
            f"--shared-libs={runner_utils}",
        ]
        runner = run(runner_args)
        if runner.returncode != 0:
            payload = {**base, "status": "failed_execution", "reason": runner.stderr or runner.stdout}
            write_result(payload)
            print(payload["reason"])
            return 1
        runner_latency = benchmark(lambda: run(runner_args), runs=10, warmup=2)

        matmul_main_hir = tmp / "jax_matmul_main_hir.mlir"
        matmul_main_hir.write_text(MATMUL_MAIN, encoding="utf-8")
        matmul_llvm_result = mlir_opt(mlir_opt_path, plugin, matmul_main_hir, MATMUL_HIR_TO_LLVM)
        if matmul_llvm_result.returncode != 0:
            payload = {**base, "status": "failed_matmul_llvm", "reason": matmul_llvm_result.stderr}
            write_result(payload)
            print(payload["reason"])
            return 1
        matmul_lowered = tmp / "jax_matmul_main_llvm.mlir"
        matmul_lowered.write_text(matmul_llvm_result.stdout, encoding="utf-8")
        matmul_runner_args = [
            mlir_runner,
            str(matmul_lowered),
            "-e",
            "main",
            "--entry-point-result=f32",
            f"--shared-libs={runner_utils}",
        ]
        matmul_runner = run(matmul_runner_args)
        if matmul_runner.returncode != 0:
            payload = {**base, "status": "failed_matmul_execution", "reason": matmul_runner.stderr or matmul_runner.stdout}
            write_result(payload)
            print(payload["reason"])
            return 1
        matmul_runner_latency = benchmark(lambda: run(matmul_runner_args), runs=10, warmup=2)

    actual = float(runner.stdout.strip())
    expected = exported["rmsnorm_expected_scalar"]
    jax_compiled = exported["rmsnorm_compiled_scalar"]
    abs_error = abs(actual - expected)
    abs_error_vs_jax = abs(actual - jax_compiled)
    comparison_passed = abs_error < 1.0e-5 and abs_error_vs_jax < 1.0e-5
    matmul_actual = float(matmul_runner.stdout.strip())
    matmul_expected = 5.1
    matmul_abs_error = abs(matmul_actual - matmul_expected)
    matmul_correct = matmul_abs_error < 1.0e-5
    payload = {
        **base,
        "status": "ok" if comparison_passed and matmul_correct else "failed_correctness",
        "jax_version": exported["jax_version"],
        "jax_backend_platform": exported["jax_backend_platform"],
        "jax_devices": exported["jax_devices"],
        "jax_device_details": exported["jax_device_details"],
        "stablehlo_export": {
            "rmsnorm_contains_stablehlo": "stablehlo." in exported["rmsnorm_text"],
            "matmul_contains_stablehlo": "stablehlo." in exported["matmul_text"],
            "rmsnorm_contains_reduce": "stablehlo.reduce" in exported["rmsnorm_text"],
            "matmul_contains_dot_general": "stablehlo.dot_general" in exported["matmul_text"],
        },
        "hir_lowering": {
            "rmsnorm_contains_fused_op": "hir.fused_rmsnorm" in rmsnorm_hir_result.stdout,
            "matmul_contains_fused_op": "hir.fused_matmul_bias_relu" in matmul_hir_result.stdout,
        },
        "stablehlo_parser": {
            "input": "JAX compiler_ir(\"stablehlo\") for RMSNorm and MatMul-Bias-ReLU",
            "rmsnorm": {
                "legal": parser_metadata["rmsnorm"]["legal"],
                "decision": parser_metadata["rmsnorm"]["decision"],
                "fallback_reason": parser_metadata["rmsnorm"]["fallback_reason"],
                "parsed_ops": parser_metadata["rmsnorm"]["parsed_ops"],
                "def_use_edges": parser_metadata["rmsnorm"]["def_use_edges"],
                "lowered_hir_op": "hir.fused_rmsnorm" if "hir.fused_rmsnorm" in rmsnorm_hir_result.stdout else None,
            },
            "matmul_bias_relu": {
                "legal": parser_metadata["matmul_bias_relu"]["legal"],
                "decision": parser_metadata["matmul_bias_relu"]["decision"],
                "fallback_reason": parser_metadata["matmul_bias_relu"]["fallback_reason"],
                "parsed_ops": parser_metadata["matmul_bias_relu"]["parsed_ops"],
                "def_use_edges": parser_metadata["matmul_bias_relu"]["def_use_edges"],
                "lowered_hir_op": "hir.fused_matmul_bias_relu"
                if "hir.fused_matmul_bias_relu" in matmul_hir_result.stdout else None,
            },
        },
        "llvm_lowering": {
            "rmsnorm_hir_removed": "hir.fused_rmsnorm" not in llvm_result.stdout,
            "contains_llvm_func": "llvm.func" in llvm_result.stdout,
            "matmul_bias_relu_hir_removed": "hir.fused_matmul_bias_relu" not in matmul_llvm_result.stdout,
            "matmul_contains_llvm_func": "llvm.func" in matmul_llvm_result.stdout,
        },
        "execution": {
            "numpy_or_jax_eager_reference_scalar": expected,
            "jax_compiled_scalar": jax_compiled,
            "hir_llvm_runner_scalar": actual,
            "max_error_vs_numpy": abs_error,
            "max_error_vs_jax": abs_error_vs_jax,
            "allclose_atol": 1.0e-5,
            "correct": comparison_passed,
            "execution_comparison_passed": comparison_passed,
            "jax_compiled_latency_ms": exported["jax_compiled_latency_ms"],
            "hir_llvm_runner_latency_ms": runner_latency,
            "latency_note": "HIR/LLVM is measured through mlir-runner subprocess launch; it is a sanity comparison, not a claimed speedup over JAX.",
        },
        "matmul_bias_relu_execution": {
            "input": "constant HIR fused MatMul-Bias-ReLU scalar extraction",
            "compiler_decision": "lower hir.fused_matmul_bias_relu to linalg.matmul plus linalg.generic add/relu, then bufferize and lower to LLVM",
            "expected_scalar": matmul_expected,
            "hir_llvm_runner_scalar": matmul_actual,
            "max_error_vs_reference": matmul_abs_error,
            "correct": matmul_correct,
            "hir_removed": "hir.fused_matmul_bias_relu" not in matmul_llvm_result.stdout,
            "contains_llvm_func": "llvm.func" in matmul_llvm_result.stdout,
            "latency_ms": matmul_runner_latency,
        },
        "runtime_boundary": exported["runtime_boundary"],
        "device_buffer_timing": exported["device_buffer_timing"],
        "shape_specialization": exported["shape_specialization"],
        "dynamic_shape_fallback": dynamic_shape_fallback,
        "tile_autotuning": exported["tile_autotuning"],
        "tiny_transformer_block": exported["tiny_transformer_block"],
    }
    write_result(payload)
    print(
        f"status={payload['status']} jax={exported['jax_version']} "
        f"max_error_vs_numpy={abs_error:.8g} max_error_vs_jax={abs_error_vs_jax:.8g}"
    )
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
