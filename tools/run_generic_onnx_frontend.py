#!/usr/bin/env python3
"""Run the generic ONNX frontend pipeline through shape/type annotation.

This driver composes the Phase 1-5 tools. It does not perform domain
recognition, MLIR lowering, or execution-plan generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import canonicalize_generic_graph_ir
import diagnose_generic_graph_ir
import imported_graph_ir_to_generic_graph_ir
import infer_generic_graph_shapes
import onnx_import_to_graph_ir
import verify_graph_ir

TRUTH_BOUNDARY = (
    "generic_onnx_frontend_metadata_only_no_domain_recognition_"
    "no_mlir_lowering_no_execution_plan_generation"
)

STAGES = ("imported", "generic", "canonicalized", "shapes")


class GenericOnnxFrontendPipelineError(Exception):
    """Raised when the generic frontend pipeline cannot continue."""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _verify_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = verify_graph_ir.verify_graph_ir(payload)
    return {"passed": result.passed, "errors": result.errors}


def _nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema") == "imported_graph_ir":
        return payload.get("graph", {}).get("nodes", [])
    return payload.get("nodes", [])


def _op_name(node: dict[str, Any], schema: str) -> str:
    if schema == "imported_graph_ir":
        return str(node.get("op_type", ""))
    return str(node.get("op", ""))


def _op_histogram(payload: dict[str, Any]) -> dict[str, int]:
    schema = payload.get("schema", "")
    return dict(Counter(_op_name(node, schema) for node in _nodes(payload)))


def _unknown_op_count(payload: dict[str, Any]) -> int:
    if payload.get("schema") != "generic_graph_ir":
        return 0
    return sum(1 for node in payload.get("nodes", []) if node.get("op") == "nn.unknown")


def _shape_summary(payload: dict[str, Any]) -> dict[str, int]:
    counts = Counter()
    if payload.get("schema") == "generic_graph_ir":
        for node in payload.get("nodes", []):
            status = node.get("shape_inference_status")
            if status:
                counts[str(status)] += 1
    return dict(counts)


def _stage_report(
    name: str,
    path: Path,
    payload: dict[str, Any] | None,
    status: str,
    verifier_status: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    report = {
        "status": status,
        "artifact_path": str(path),
        "verifier": verifier_status or {"passed": False, "errors": []},
        "node_count": 0,
        "op_histogram": {},
        "unknown_op_count": 0,
    }
    if payload is not None:
        report["node_count"] = len(_nodes(payload))
        report["op_histogram"] = _op_histogram(payload)
        report["unknown_op_count"] = _unknown_op_count(payload)
        if name == "shapes":
            report["shape_inference_summary"] = _shape_summary(payload)
    if error:
        report["error"] = error
    return report


def _should_stop(current: str, stop_after: str) -> bool:
    return STAGES.index(current) >= STAGES.index(stop_after)


def _check_or_raise(stage: str, verifier_status: dict[str, Any], keep_going: bool) -> None:
    if verifier_status["passed"] or keep_going:
        return
    raise GenericOnnxFrontendPipelineError(
        f"{stage} verification failed: {'; '.join(verifier_status['errors'])}"
    )


def run_pipeline(
    onnx_path: Path,
    out_dir: Path,
    prefix: str,
    stop_after: str = "shapes",
    keep_going: bool = False,
) -> dict[str, Any]:
    if stop_after not in STAGES:
        raise GenericOnnxFrontendPipelineError(f"invalid stop stage: {stop_after}")
    if not onnx_import_to_graph_ir.module_available("onnx"):
        raise GenericOnnxFrontendPipelineError("install the 'onnx' package to run this pipeline")
    if not onnx_path.exists():
        raise GenericOnnxFrontendPipelineError(f"input ONNX path does not exist: {onnx_path}")

    import onnx  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "imported": out_dir / f"{prefix}.imported_graph_ir.json",
        "generic": out_dir / f"{prefix}.generic_graph_ir.json",
        "canonicalized": out_dir / f"{prefix}.canonical_generic_graph_ir.json",
        "shapes": out_dir / f"{prefix}.shape_generic_graph_ir.json",
        "diagnostics": out_dir / f"{prefix}.diagnostics_report.json",
        "report": out_dir / f"{prefix}.frontend_report.json",
    }
    report: dict[str, Any] = {
        "input_path": str(onnx_path),
        "artifact_paths": {name: str(path) for name, path in paths.items()},
        "stages": {},
        "truth_boundary": TRUTH_BOUNDARY,
    }

    try:
        imported = onnx_import_to_graph_ir.import_onnx_to_graph_ir(onnx_path, onnx)
        _write_json(paths["imported"], imported)
        imported_verify = _verify_payload(imported)
        report["stages"]["imported"] = _stage_report("imported", paths["imported"], imported, "completed", imported_verify)
        _check_or_raise("imported", imported_verify, keep_going)
        if _should_stop("imported", stop_after):
            _write_json(paths["report"], report)
            return report

        generic = imported_graph_ir_to_generic_graph_ir.convert_imported_graph_ir(imported)
        _write_json(paths["generic"], generic)
        generic_verify = _verify_payload(generic)
        report["stages"]["generic"] = _stage_report("generic", paths["generic"], generic, "completed", generic_verify)
        _check_or_raise("generic", generic_verify, keep_going)
        if _should_stop("generic", stop_after):
            _write_json(paths["report"], report)
            return report

        canonical = canonicalize_generic_graph_ir.canonicalize_generic_graph_ir(generic)
        _write_json(paths["canonicalized"], canonical)
        canonical_verify = _verify_payload(canonical)
        report["stages"]["canonicalized"] = _stage_report(
            "canonicalized", paths["canonicalized"], canonical, "completed", canonical_verify
        )
        _check_or_raise("canonicalized", canonical_verify, keep_going)
        if _should_stop("canonicalized", stop_after):
            _write_json(paths["report"], report)
            return report

        shaped = infer_generic_graph_shapes.infer_generic_graph_shapes(canonical)
        _write_json(paths["shapes"], shaped)
        shaped_verify = _verify_payload(shaped)
        report["stages"]["shapes"] = _stage_report("shapes", paths["shapes"], shaped, "completed", shaped_verify)
        _check_or_raise("shapes", shaped_verify, keep_going)

        diagnostics = diagnose_generic_graph_ir.diagnose_generic_graph_ir(shaped)
        _write_json(paths["diagnostics"], diagnostics)
        report["diagnostics"] = {
            "artifact_path": str(paths["diagnostics"]),
            "frontend_readiness_status": diagnostics["frontend_readiness_status"],
            "truth_boundary": diagnostics["truth_boundary"],
        }

    except Exception as exc:
        if not keep_going:
            report["pipeline_status"] = "failed"
            report["error"] = str(exc)
            _write_json(paths["report"], report)
            raise
        report["pipeline_status"] = "completed_with_errors"
        report["error"] = str(exc)
        _write_json(paths["report"], report)
        return report

    report["pipeline_status"] = "completed"
    _write_json(paths["report"], report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path, help="Input ONNX model path")
    parser.add_argument("out_dir", type=Path, help="Output artifact directory")
    parser.add_argument("--prefix", default=None, help="Output artifact filename prefix")
    parser.add_argument("--stop-after", choices=STAGES, default="shapes")
    parser.add_argument("--keep-going", action="store_true", default=False)
    args = parser.parse_args()

    prefix = args.prefix or args.onnx.stem
    try:
        report = run_pipeline(
            args.onnx,
            args.out_dir,
            prefix,
            stop_after=args.stop_after,
            keep_going=args.keep_going,
        )
    except GenericOnnxFrontendPipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"run_generic_onnx_frontend: wrote {report['artifact_paths']['report']}")
    print(f"  status: {report.get('pipeline_status', 'completed')}")
    return 0 if report.get("pipeline_status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
