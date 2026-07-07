#!/usr/bin/env python3
"""HF Qwen2.5-0.5B-Instruct -> AWQ quantized checkpoint export (Python edge tooling).

This is Python/HF edge tooling, not compiler core, per this repo's
"Compiler Core Policy: Zero Python / Zero JSON" (see CLAUDE.md). Quantizing a
HuggingFace checkpoint with AutoAWQ is inherently a Python/PyTorch-ecosystem
operation; there is no C++ compiler-core path for it.

Current status (be precise about what this does and does not do):

  IMPLEMENTED here:
    - A hard dependency check for AutoAWQ. If it is missing, this script
      fails with a non-zero exit code and install instructions -- it never
      writes a fake quantized artifact or a fake provenance.json.
    - A real AutoAWQ quantization run (model load, calibration, quantize,
      save) when the toolchain is installed and a GPU/network/checkpoint are
      reachable.
    - A provenance.json recording source model, method, calibration note,
      tool/version, created_at, and an explicit truth_boundary.

  NOT IMPLEMENTED here (explicitly, do not assume otherwise):
    - Any accuracy evaluation of the quantized checkpoint (perplexity, task
      benchmarks). This script only produces the artifact; it does not
      verify quantization quality.
    - Any claim that GTX 1650 (Turing, cc 7.5) has native INT4 tensor-core
      support. Running the resulting artifact under vLLM's AWQ path on that
      hardware is an experimental forced-quant configuration -- see
      configs/target_profiles/nvidia_gtx1650_maxq_awq_forced.json.
    - GPTQ export. Only AWQ is implemented; see docs/future_work.md.

AutoAWQ has no CPU/macOS quantization kernel path -- this script's export
step must run on a CUDA-capable Linux host. On a machine without AutoAWQ (or
without CUDA), running this script fails clearly at the dependency check
rather than silently producing nothing or a fake artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "qwen_awq"

REQUIRED_MODULES = ("awq", "transformers")

INSTALL_INSTRUCTIONS = """\
AutoAWQ is required for this export and is not installed in this environment.

Install it (requires a CUDA-capable Linux host -- AutoAWQ has no CPU/macOS
quantization kernel path):
    .venv/bin/pip install autoawq transformers

See https://github.com/casper-hansen/AutoAWQ for hardware/driver requirements.
"""


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=128)
    parser.add_argument("--calib-dataset", default="pileval")
    parser.add_argument("--calib-samples", type=int, default=128)
    args = parser.parse_args()

    missing = [name for name in REQUIRED_MODULES if not module_available(name)]
    if missing:
        print(f"error: missing required dependency: {', '.join(missing)}", file=sys.stderr)
        print(INSTALL_INSTRUCTIONS, file=sys.stderr)
        print(
            "refusing to write a quantized artifact or provenance.json -- "
            "no fake artifact will be produced.",
            file=sys.stderr,
        )
        return 1

    from awq import AutoAWQForCausalLM
    import awq as awq_pkg
    from transformers import AutoTokenizer

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": "GEMM",
    }

    try:
        model = AutoAWQForCausalLM.from_pretrained(args.model_id, safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=False)
        model.quantize(tokenizer, quant_config=quant_config, calib_data=args.calib_dataset)
        model.save_quantized(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
    except Exception as exc:  # noqa: BLE001 -- report plainly, do not fake success
        print(f"error: AWQ export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("refusing to write provenance.json for a failed export.", file=sys.stderr)
        return 1

    provenance = {
        "artifact_type": "qwen_awq_export_provenance",
        "source_model": args.model_id,
        "method": "awq",
        "tool": "autoawq",
        "tool_version": getattr(awq_pkg, "__version__", "unknown"),
        "quant_config": quant_config,
        "calibration_note": (
            f"AutoAWQ's built-in calibration pipeline, dataset='{args.calib_dataset}', "
            f"~{args.calib_samples} samples requested via AutoAWQ's default calibration "
            "loader; this script does not construct a custom calibration set."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "truth_boundary": (
            "real_autoawq_export_local_run_not_accuracy_evaluated_not_measured_serving_latency"
        ),
        "not_claimed": [
            "no accuracy evaluation (perplexity, task benchmarks) was run on this artifact",
            "no measured vLLM serving latency for this artifact",
            "GTX 1650 (Turing, cc 7.5) has no native INT4 tensor-core path; serving this "
            "artifact under vLLM --quantization awq on that hardware is an experimental "
            "forced-quant configuration, not a claim of native hardware support",
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    print(f"AWQ export complete: {output_dir}")
    print(f"provenance: {output_dir / 'provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
