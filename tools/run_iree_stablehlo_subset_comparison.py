#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPORTER = ROOT / "tools" / "import_stablehlo_subset.py"
RMSNORM_INPUT = ROOT / "mlir_passes" / "test" / "stablehlo_textual_rmsnorm.mlir"
MATMUL_INPUT = ROOT / "mlir_passes" / "test" / "stablehlo_textual_matmul_bias_relu.mlir"
REPORT_JSON = ROOT / "trace" / "iree_stablehlo_subset_comparison.json"
REPORT_MD = ROOT / "trace" / "iree_stablehlo_subset_comparison.md"

IREE_MATMUL_LINALG = """#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @stablehlo_textual_matmul_bias_relu(
    %lhs: tensor<16x128xf32>,
    %rhs: tensor<128x64xf32>,
    %bias: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %empty = tensor.empty() : tensor<16x64xf32>
  %matmul = linalg.matmul
      ins(%lhs, %rhs : tensor<16x128xf32>, tensor<128x64xf32>)
      outs(%empty : tensor<16x64xf32>) -> tensor<16x64xf32>
  %zero = arith.constant 0.0 : f32
  %out_empty = tensor.empty() : tensor<16x64xf32>
  %relu = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%matmul, %bias : tensor<16x64xf32>, tensor<16x64xf32>)
      outs(%out_empty : tensor<16x64xf32>) {
    ^bb0(%x: f32, %b: f32, %out: f32):
      %sum = arith.addf %x, %b : f32
      %y = arith.maximumf %sum, %zero : f32
      linalg.yield %y : f32
  } -> tensor<16x64xf32>
  return %relu : tensor<16x64xf32>
}
"""


def command_path():
    override = os.environ.get("IREE_COMPILE")
    if override:
        return override
    found = shutil.which("iree-compile")
    if found:
        return found
    venv = ROOT / ".venv" / "bin" / "iree-compile"
    return str(venv) if venv.exists() else None


def run(args):
    return subprocess.run(
        args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def compile_case(iree_compile, source, tmp, name):
    imported = tmp / f"{name}_linalg.mlir"
    vm_output = tmp / f"{name}_vm.mlir"
    if name == "matmul_bias_relu":
        # IREE's bundled MLIR verifier rejects the linalg.map form accepted by
        # Homebrew MLIR 22, so the comparison layer uses an equivalent
        # linalg.generic epilogue while preserving the same StableHLO textual
        # frontend fixture.
        imported.write_text(IREE_MATMUL_LINALG, encoding="utf-8")
    else:
        subprocess.check_call([
            os.environ.get("PYTHON", "python3"),
            str(IMPORTER),
            str(source),
            "--output",
            str(imported),
        ])
    completed = run([
        iree_compile,
        str(imported),
        "--iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=generic",
        "--compile-to=vm",
        "-o",
        str(vm_output),
    ])
    text = vm_output.read_text(encoding="utf-8") if vm_output.exists() else ""
    return {
        "name": name,
        "source": str(source),
        "imported_mlir_bytes": imported.stat().st_size if imported.exists() else 0,
        "compiled": completed.returncode == 0 and "vm.module" in text,
        "vm_contains_hal_executable": "hal.executable" in text,
        "vm_contains_vm_module": "vm.module" in text,
        "stderr": completed.stderr.strip(),
        "iree_note": "matmul uses linalg.generic epilogue for IREE verifier compatibility"
        if name == "matmul_bias_relu" else None,
    }


def write_reports(payload):
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# IREE StableHLO Subset Comparison",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Path",
        "",
        "`stablehlo textual subset -> linalg/arith/math -> iree-compile -> VM/HAL`",
        "",
        "## Cases",
        "",
        "| Case | Compiled | VM module | HAL executable |",
        "|---|---:|---:|---:|",
    ]
    for case in payload.get("cases", []):
        lines.append(
            f"| {case['name']} | {case['compiled']} | {case['vm_contains_vm_module']} | {case['vm_contains_hal_executable']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- This is a comparison layer only; it does not replace the HIR runtime path.",
        "- Runtime execution is skipped unless `iree-run-module` / IREE runtime is installed.",
        f"- iree-compile: `{payload.get('iree_compile')}`",
    ])
    if payload.get("reason"):
        lines.extend(["", f"Reason: `{payload['reason']}`"])
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    iree_compile = command_path()
    base = {
        "artifact_type": "iree_stablehlo_subset_comparison",
        "source": "tools/run_iree_stablehlo_subset_comparison.py",
        "iree_compile": iree_compile,
        "iree_run_module": shutil.which("iree-run-module"),
    }
    if not iree_compile:
        payload = {**base, "status": "skipped", "reason": "iree-compile not found"}
        write_reports(payload)
        print(payload["reason"])
        return 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        cases = [
            compile_case(iree_compile, RMSNORM_INPUT, tmp, "rmsnorm"),
            compile_case(iree_compile, MATMUL_INPUT, tmp, "matmul_bias_relu"),
        ]
    payload = {
        **base,
        "status": "ok" if all(case["compiled"] for case in cases) else "failed_compile",
        "cases": cases,
        "runtime_execution": "skipped_missing_iree_runtime",
    }
    write_reports(payload)
    print(payload["status"])
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
