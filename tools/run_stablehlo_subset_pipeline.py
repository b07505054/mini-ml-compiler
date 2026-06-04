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
RMSNORM_INPUT = ROOT / "mlir_passes" / "test" / "stablehlo_textual_rmsnorm.mlir"
MATMUL_INPUT = ROOT / "mlir_passes" / "test" / "stablehlo_textual_matmul_bias_relu.mlir"
REPORT_JSON = ROOT / "trace" / "stablehlo_subset_pipeline_report.json"
REPORT_MD = ROOT / "trace" / "stablehlo_subset_pipeline_report.md"
IMPORTER = ROOT / "tools" / "import_stablehlo_subset.py"

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

RMSNORM_MAIN = """func.func @main() -> f32 attributes {llvm.emit_c_interface} {
  %x = arith.constant dense<[
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 0.0, -2.0, 4.0]
  ]> : tensor<2x4xf32>
  %out = hir.fused_rmsnorm %x {
    frontend.source = "stablehlo_textual_subset",
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
    return row[1] / math.sqrt(sum(x * x for x in row) / len(row) + 1.0e-6)


def mlir_opt(mlir_opt_path, plugin, input_path, pipeline):
    return run([
        mlir_opt_path,
        str(input_path),
        f"--load-dialect-plugin={plugin}",
        f"--load-pass-plugin={plugin}",
        f"--pass-pipeline={pipeline}",
    ])


def write_reports(payload):
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# StableHLO Textual Subset Pipeline Report",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Pipeline",
        "",
        "`stablehlo textual subset -> linalg/arith/math -> HIR -> LLVM dialect -> mlir-runner`",
        "",
        "## Results",
        "",
        f"- RMSNorm HIR contains `hir.fused_rmsnorm`: `{payload.get('rmsnorm_hir_contains_fused_op')}`",
        f"- MatMul HIR contains `hir.fused_matmul_bias_relu`: `{payload.get('matmul_hir_contains_fused_op')}`",
        f"- RMSNorm LLVM contains `llvm.func`: `{payload.get('rmsnorm_llvm_contains_llvm_func')}`",
        f"- Expected: `{payload.get('expected')}`",
        f"- Actual: `{payload.get('actual')}`",
        f"- Abs error: `{payload.get('abs_error')}`",
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
    mlir_opt_path = command_path("MLIR_OPT", "mlir-opt")
    mlir_runner = command_path("MLIR_RUNNER", "mlir-runner")
    plugin = Path(os.environ.get("PLUGIN", str(DEFAULT_PLUGIN)))
    runner_utils = Path(os.environ.get(
        "MLIR_C_RUNNER_UTILS",
        "/opt/homebrew/opt/llvm/lib/libmlir_c_runner_utils.dylib",
    ))
    base = {
        "artifact_type": "stablehlo_subset_pipeline_report",
        "source": "tools/run_stablehlo_subset_pipeline.py",
        "frontend": "stablehlo_textual_subset",
        "mlir_opt": mlir_opt_path,
        "mlir_runner": mlir_runner,
        "plugin": str(plugin),
    }
    missing = []
    for name, value in [("mlir-opt", mlir_opt_path), ("mlir-runner", mlir_runner)]:
        if not value:
            missing.append(name)
    for path in [plugin, runner_utils]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        payload = {**base, "status": "skipped", "reason": "missing " + ", ".join(missing)}
        write_reports(payload)
        print(payload["reason"])
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rmsnorm_linalg = tmp / "rmsnorm_linalg.mlir"
        matmul_linalg = tmp / "matmul_linalg.mlir"
        subprocess.check_call([
            os.environ.get("PYTHON", "python3"),
            str(IMPORTER),
            str(RMSNORM_INPUT),
            "--output",
            str(rmsnorm_linalg),
        ])
        subprocess.check_call([
            os.environ.get("PYTHON", "python3"),
            str(IMPORTER),
            str(MATMUL_INPUT),
            "--output",
            str(matmul_linalg),
        ])

        rmsnorm_hir_result = mlir_opt(mlir_opt_path, plugin, rmsnorm_linalg, RMSNORM_TO_HIR)
        if rmsnorm_hir_result.returncode != 0:
            payload = {**base, "status": "failed_rmsnorm_hir", "reason": rmsnorm_hir_result.stderr}
            write_reports(payload)
            print(payload["reason"])
            return 1
        matmul_hir_result = mlir_opt(mlir_opt_path, plugin, matmul_linalg, MATMUL_TO_HIR)
        if matmul_hir_result.returncode != 0:
            payload = {**base, "status": "failed_matmul_hir", "reason": matmul_hir_result.stderr}
            write_reports(payload)
            print(payload["reason"])
            return 1

        main_hir = tmp / "rmsnorm_main_hir.mlir"
        main_hir.write_text(RMSNORM_MAIN, encoding="utf-8")
        llvm_result = mlir_opt(mlir_opt_path, plugin, main_hir, HIR_TO_LLVM)
        if llvm_result.returncode != 0:
            payload = {**base, "status": "failed_rmsnorm_llvm", "reason": llvm_result.stderr}
            write_reports(payload)
            print(payload["reason"])
            return 1
        lowered = tmp / "rmsnorm_main_llvm.mlir"
        lowered.write_text(llvm_result.stdout, encoding="utf-8")

        runner = run([
            mlir_runner,
            str(lowered),
            "-e",
            "main",
            "--entry-point-result=f32",
            f"--shared-libs={runner_utils}",
        ])
        if runner.returncode != 0:
            payload = {**base, "status": "failed_execution", "reason": runner.stderr or runner.stdout}
            write_reports(payload)
            print(payload["reason"])
            return 1

    expected = reference_value()
    actual = float(runner.stdout.strip())
    abs_error = abs(actual - expected)
    payload = {
        **base,
        "status": "ok" if abs_error < 1.0e-5 else "failed_correctness",
        "rmsnorm_hir_contains_fused_op": "hir.fused_rmsnorm" in rmsnorm_hir_result.stdout,
        "matmul_hir_contains_fused_op": "hir.fused_matmul_bias_relu" in matmul_hir_result.stdout,
        "rmsnorm_llvm_contains_llvm_func": "llvm.func" in llvm_result.stdout,
        "expected": expected,
        "actual": actual,
        "abs_error": abs_error,
    }
    write_reports(payload)
    print(f"status={payload['status']} expected={expected:.8f} actual={actual:.8f}")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
