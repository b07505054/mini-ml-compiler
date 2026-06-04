#!/usr/bin/env python3
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "trace" / "openxla_toolchain_status.json"
REPORT_MD = ROOT / "trace" / "openxla_toolchain_status.md"


def command_version(path):
    if not path:
        return None
    for args in ([path, "--version"], [path, "-version"]):
        completed = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output = (completed.stdout or completed.stderr or "").strip()
        if completed.returncode == 0 and output:
            return output.splitlines()[0]
    return None


def python_module_status(name):
    spec = importlib.util.find_spec(name)
    return {
        "available": spec is not None,
        "origin": spec.origin if spec and spec.origin else None,
    }


def write_reports(payload):
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# OpenXLA Toolchain Status",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Commands",
        "",
    ]
    for name, data in payload["commands"].items():
        lines.append(
            f"- `{name}`: available=`{data['available']}`, path=`{data['path']}`, version=`{data['version']}`"
        )
    lines.extend(["", "## Python Modules", ""])
    for name, data in payload["python_modules"].items():
        lines.append(f"- `{name}`: available=`{data['available']}`, origin=`{data['origin']}`")
    lines.extend([
        "",
        "## Compiler Path",
        "",
        "- StableHLO-native tests are skipped until `stablehlo-opt` is installed.",
        "- The current repo uses StableHLO-compatible Linalg/Arith decompositions for FileCheck coverage.",
        "- This keeps the MLIR/HIR/LLVM executable path testable without vendoring OpenXLA tools.",
    ])
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    commands = {}
    for name in ["stablehlo-opt", "torch-mlir-opt", "mlir-opt", "mlir-runner"]:
        path = shutil.which(name)
        if not path and name.startswith("mlir-"):
            fallback = Path("/opt/homebrew/opt/llvm/bin") / name
            path = str(fallback) if fallback.exists() else None
        commands[name] = {
            "available": path is not None,
            "path": path,
            "version": command_version(path),
        }

    modules = {
        name: python_module_status(name)
        for name in ["jax", "tensorflow", "torch_mlir", "stablehlo"]
    }
    has_stablehlo = commands["stablehlo-opt"]["available"] or modules["stablehlo"]["available"]
    payload = {
        "artifact_type": "openxla_toolchain_status",
        "source": "tools/check_openxla_toolchain.py",
        "status": "stablehlo_available" if has_stablehlo else "stablehlo_unavailable_skip_native_tests",
        "commands": commands,
        "python_modules": modules,
    }
    write_reports(payload)
    print(payload["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
