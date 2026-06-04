#!/usr/bin/env python3
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLUGIN = ROOT / "build-mlir-codex" / "HIRMatMulBiasReluFusionPass.dylib"
REPORT_JSON = ROOT / "trace" / "hir_rmsnorm_execution_engine_report.json"
REPORT_MD = ROOT / "trace" / "hir_rmsnorm_execution_engine_report.md"

LOWERING_PIPELINE = (
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

MLIR_SOURCE = r"""
func.func @main() -> f32 attributes {llvm.emit_c_interface} {
  %x = arith.constant dense<[
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 0.0, -2.0, 4.0]
  ]> : tensor<2x4xf32>
  %out = hir.fused_rmsnorm %x {
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


def reference_value():
    row = [1.0, 2.0, 3.0, 4.0]
    mean_square = sum(x * x for x in row) / len(row)
    return row[1] * (1.0 / math.sqrt(mean_square + 1.0e-6))


def write_report(payload):
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# HIR RMSNorm ExecutionEngine Report",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Pipeline",
        "",
        "`hir.fused_rmsnorm -> linalg.generic/math.rsqrt -> LLVM dialect -> mlir-runner`",
        "",
        "## Result",
        "",
        f"- Expected: `{payload.get('expected')}`",
        f"- Actual: `{payload.get('actual')}`",
        f"- Absolute error: `{payload.get('abs_error')}`",
        f"- Tolerance: `{payload.get('tolerance')}`",
        "",
        "## Tools",
        "",
        f"- mlir-opt: `{payload.get('mlir_opt')}`",
        f"- mlir-runner: `{payload.get('mlir_runner')}`",
        f"- plugin: `{payload.get('plugin')}`",
    ]
    if payload.get("reason"):
        lines.extend(["", f"Reason: `{payload['reason']}`"])
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    mlir_opt = command_path("MLIR_OPT", "mlir-opt")
    mlir_runner = command_path("MLIR_RUNNER", "mlir-runner")
    plugin = Path(os.environ.get("PLUGIN", str(DEFAULT_PLUGIN)))
    runner_utils = Path(os.environ.get(
        "MLIR_C_RUNNER_UTILS",
        "/opt/homebrew/opt/llvm/lib/libmlir_c_runner_utils.dylib",
    ))

    base_payload = {
        "artifact_type": "hir_rmsnorm_execution_engine_report",
        "source": "tools/run_hir_rmsnorm_execution_engine.py",
        "mlir_opt": mlir_opt,
        "mlir_runner": mlir_runner,
        "plugin": str(plugin),
        "runner_utils": str(runner_utils),
        "pipeline": LOWERING_PIPELINE,
    }

    missing = []
    if not mlir_opt:
        missing.append("mlir-opt")
    if not mlir_runner:
        missing.append("mlir-runner")
    if not plugin.exists():
        missing.append(str(plugin))
    if not runner_utils.exists():
        missing.append(str(runner_utils))
    if missing:
        payload = {
            **base_payload,
            "status": "skipped",
            "reason": "missing " + ", ".join(missing),
        }
        write_report(payload)
        print(payload["reason"])
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "hir_rmsnorm_main.mlir"
        lowered = Path(tmp) / "hir_rmsnorm_main_llvm.mlir"
        source.write_text(MLIR_SOURCE, encoding="utf-8")

        opt = run([
            mlir_opt,
            str(source),
            f"--load-dialect-plugin={plugin}",
            f"--load-pass-plugin={plugin}",
            f"--pass-pipeline={LOWERING_PIPELINE}",
        ])
        if opt.returncode != 0:
            payload = {
                **base_payload,
                "status": "failed_lowering",
                "reason": opt.stderr.strip() or opt.stdout.strip(),
            }
            write_report(payload)
            print(payload["reason"])
            return 1
        lowered.write_text(opt.stdout, encoding="utf-8")

        runner = run([
            mlir_runner,
            str(lowered),
            "-e",
            "main",
            "--entry-point-result=f32",
            f"--shared-libs={runner_utils}",
        ])
        if runner.returncode != 0:
            payload = {
                **base_payload,
                "status": "failed_execution",
                "reason": runner.stderr.strip() or runner.stdout.strip(),
            }
            write_report(payload)
            print(payload["reason"])
            return 1

    actual = float(runner.stdout.strip())
    expected = reference_value()
    abs_error = abs(actual - expected)
    tolerance = 1.0e-5
    payload = {
        **base_payload,
        "status": "ok" if abs_error <= tolerance else "failed_correctness",
        "expected": expected,
        "actual": actual,
        "abs_error": abs_error,
        "tolerance": tolerance,
    }
    write_report(payload)
    print(f"expected={expected:.8f} actual={actual:.8f} abs_error={abs_error:.8g}")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
