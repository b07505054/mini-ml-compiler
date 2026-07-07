#!/usr/bin/env python3
"""HF Qwen2.5-0.5B-Instruct -> ONNX export probe (Python edge tooling).

This is Python/HF edge tooling, not compiler core, per this repo's
"Compiler Core Policy: Zero Python / Zero JSON" (see CLAUDE.md). It exists
because there is no reasonable C++ path to load a HuggingFace checkpoint;
exporting a HF model to ONNX is inherently a Python/PyTorch-ecosystem
operation, same as tools/run_jax_stablehlo_pipeline.py already is for JAX.

Current status (be precise about what this does and does not do):

  IMPLEMENTED here:
    - Toolchain availability probe (torch, transformers, optimum, onnx).
    - Real HF -> ONNX export via optimum, when the toolchain is installed
      and the checkpoint/network is reachable.
    - Real ONNX graph introspection (node count, op-type histogram,
      input/output names) written to a report for manual inspection.

  NOT IMPLEMENTED here (explicitly, do not assume otherwise):
    - Automatic per-role bucketing of ONNX nodes into
      embedding / decoder_layer / final_norm / lm_head groups. That mapping
      is nontrivial (real HF ONNX graphs do not self-label node roles) and
      is real future engineering, not attempted in this script.
    - Any output compatible with the qwen-onnx-to-serving-mlir importer's
      graph-facts JSON schema. This script's output is a diagnostic report,
      not compiler input. The importer currently consumes the hand-authored
      fixture at configs/models/qwen_0_5b_onnx_graph_facts.json.
    - Any write to the C++ compiler pipeline. This script is not called by
      tools/run_qwen_compiler_pipeline.sh or any CMake/CTest target.

This script does not claim a full HuggingFace/PyTorch/ONNX import path
exists; it only proves (when the optional toolchain is present) that real
export and real graph introspection are possible on this machine.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "trace" / "qwen_onnx_export_probe.json"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def write_report(payload: dict) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def probe_modules() -> dict:
    return {
        "torch": module_available("torch"),
        "transformers": module_available("transformers"),
        "optimum": module_available("optimum"),
        "onnx": module_available("onnx"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the exported ONNX model to (default: temp dir).",
    )
    args = parser.parse_args()

    modules = probe_modules()
    if not all(modules.values()):
        payload = {
            "artifact_type": "qwen_onnx_export_probe",
            "source": "tools/export_qwen_onnx.py",
            "status": "skipped_missing_toolchain",
            "modules": modules,
            "reason": (
                "Install torch, transformers, optimum, and onnx to attempt a "
                "real HF -> ONNX export and graph introspection."
            ),
            "not_claimed": [
                "no ONNX export was attempted",
                "no per-role node bucketing was attempted",
                "this repo's compiler importer still consumes the hand-authored "
                "fixture at configs/models/qwen_0_5b_onnx_graph_facts.json",
            ],
        }
        write_report(payload)
        print(payload["status"])
        return 0

    # Keep this block minimal; only reached when the optional toolchain is
    # installed. Real export can still fail (no network, gated checkpoint,
    # unsupported optimum/transformers version) -- report that plainly
    # rather than raising.
    import onnx
    from optimum.exporters.onnx import main_export

    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="qwen_onnx_export_"))

    try:
        main_export(
            model_name_or_path=args.model_id,
            output=str(output_dir),
            task="text-generation-with-past",
        )
    except Exception as exc:  # noqa: BLE001 -- report, do not crash the probe
        payload = {
            "artifact_type": "qwen_onnx_export_probe",
            "source": "tools/export_qwen_onnx.py",
            "status": "export_failed",
            "modules": modules,
            "model_id": args.model_id,
            "error": f"{type(exc).__name__}: {exc}",
            "not_claimed": [
                "no ONNX model was produced",
                "this repo's compiler importer still consumes the hand-authored "
                "fixture at configs/models/qwen_0_5b_onnx_graph_facts.json",
            ],
        }
        write_report(payload)
        print(payload["status"])
        return 0

    onnx_files = sorted(output_dir.glob("*.onnx"))
    graph_summaries = []
    for onnx_path in onnx_files:
        model = onnx.load(str(onnx_path))
        op_type_counts: dict[str, int] = {}
        for node in model.graph.node:
            op_type_counts[node.op_type] = op_type_counts.get(node.op_type, 0) + 1
        graph_summaries.append({
            "file": onnx_path.name,
            "node_count": len(model.graph.node),
            "op_type_histogram": op_type_counts,
            "inputs": [i.name for i in model.graph.input],
            "outputs": [o.name for o in model.graph.output],
        })

    payload = {
        "artifact_type": "qwen_onnx_export_probe",
        "source": "tools/export_qwen_onnx.py",
        "status": "exported_report_only",
        "modules": modules,
        "model_id": args.model_id,
        "output_dir": str(output_dir),
        "graphs": graph_summaries,
        "not_claimed": [
            "no per-role (embedding/decoder_layer/final_norm/lm_head) node "
            "bucketing was performed on these graphs",
            "this repo's compiler importer still consumes the hand-authored "
            "fixture at configs/models/qwen_0_5b_onnx_graph_facts.json, not "
            "this export's output",
        ],
    }
    write_report(payload)
    print(payload["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
