#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "trace" / "torch_mlir_tiny_transformer_probe.json"
REPORT_MD = ROOT / "trace" / "torch_mlir_tiny_transformer_probe.md"


def module_available(name):
    return importlib.util.find_spec(name) is not None


def write_reports(payload):
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Torch-MLIR Tiny Transformer Probe",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Intended Frontend",
        "",
        "`PyTorch tiny block -> torch-mlir -> Linalg/StableHLO -> HIR -> CPU/Metal`",
        "",
        "## Tool Status",
        "",
        f"- torch: `{payload['modules']['torch']}`",
        f"- torch_mlir: `{payload['modules']['torch_mlir']}`",
    ]
    if payload.get("reason"):
        lines.extend(["", f"Reason: `{payload['reason']}`"])
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    modules = {
        "torch": module_available("torch"),
        "torch_mlir": module_available("torch_mlir"),
    }
    if not all(modules.values()):
        payload = {
            "artifact_type": "torch_mlir_tiny_transformer_probe",
            "source": "tools/run_torch_mlir_tiny_transformer_probe.py",
            "status": "skipped_missing_torch_mlir_toolchain",
            "modules": modules,
            "reason": "Install torch and torch_mlir to export the tiny transformer block.",
        }
        write_reports(payload)
        print(payload["status"])
        return 0

    # Keep this block minimal and deterministic. It is only reached when the
    # optional frontend toolchain is installed.
    import torch
    import torch_mlir

    class TinyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x):
            rms = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1.0e-6)
            return torch.relu(self.linear(rms * self.weight))

    model = TinyBlock().eval()
    example = torch.randn(2, 4)
    module = torch_mlir.compile(model, example, output_type="linalg-on-tensors")
    text = str(module)
    payload = {
        "artifact_type": "torch_mlir_tiny_transformer_probe",
        "source": "tools/run_torch_mlir_tiny_transformer_probe.py",
        "status": "ok",
        "modules": modules,
        "contains_linalg": "linalg." in text,
        "contains_torch": "torch." in text,
        "mlir_preview": text[:2000],
    }
    write_reports(payload)
    print(payload["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
