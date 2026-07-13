#!/usr/bin/env python3
"""IR-rooted Triton candidate provider in strict shadow mode.

This tool adapts existing Triton fused MatMul+Bias+ReLU selection artifacts
into compiler-internal ImplementationCandidate-shaped records. It never writes
canonical ExecutionPlan data and never invokes Triton or GPU benchmarking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "triton_ir_rooted_candidate_shadow_analysis"
SCHEMA_VERSION = 1
PROVIDER_ID = "triton_candidate_provider_shadow"
PROVIDER_VERSION = "a6.v1"
SUPPORTED_KERNEL = "triton_tiled_matmul_bias_relu_one_pass_f32"
SUPPORTED_SEMANTIC_KERNEL = "triton_matmul_bias_relu_one_pass_f32"
SUPPORTED_OP = "MatMulBiasRelu"
SUPPORTED_BACKEND = "cuda"
SUPPORTED_DTYPE = "f32"
TRUTH_BOUNDARY = (
    "shadow_only_non_authoritative_existing_triton_artifacts_not_canonical_"
    "executionplan_not_runtime_dispatch"
)


@dataclass(frozen=True)
class IrFusedRegion:
    semantic_target_ref: str
    function_ref: str
    op_name: str
    shape: dict[str, Any]
    dtype: str
    source_graph_node_id: str | None = None
    source_imported_node_id: str | None = None
    source_op_type: str | None = None
    source_generic_op: str | None = None
    source_onnx_name: str | None = None
    source_dispatch_group: str | None = None
    source_op_role: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return {
        "m": int(shape["m"]),
        "n": int(shape["n"]),
        "k": int(shape["k"]),
        "dtype": str(shape.get("dtype", SUPPORTED_DTYPE)),
    }


def parse_mlir_fused_regions(path: Path) -> list[IrFusedRegion]:
    text = path.read_text(encoding="utf-8")
    function_ref = "unknown_function"
    func_match = re.search(r"func\.func\s+@([A-Za-z_.$-][A-Za-z0-9_.$-]*)", text)
    if func_match:
        function_ref = func_match.group(1)

    regions: list[IrFusedRegion] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "hir.fused_matmul_bias_relu" not in line:
            continue
        # Keep parsing intentionally simple: A6 only needs static shaped
        # fixture/provenance MLIR and must not become a production MLIR parser.
        window = "\n".join(lines[max(0, idx - 2): min(len(lines), idx + 12)])
        tensors = re.findall(r"tensor<([^>]+)>", window)
        shape = None
        dtype = SUPPORTED_DTYPE
        if tensors:
            dims = tensors[0].split("x")
            if len(dims) >= 3 and dims[-1] in {"f32", "fp32"}:
                # LHS is MxK; RHS is KxN when present.
                lhs = tensors[0].split("x")
                rhs = tensors[1].split("x") if len(tensors) > 1 else []
                try:
                    m = int(lhs[0])
                    k = int(lhs[1])
                    n = int(rhs[1]) if len(rhs) >= 2 else int(lhs[1])
                    dtype = "f32" if lhs[-1] == "f32" else lhs[-1]
                    shape = {"m": m, "n": n, "k": k, "dtype": dtype}
                except ValueError:
                    shape = None
        attrs = {}
        # Attributes may be on the same line or nearby in simple test fixtures.
        for key in [
            "source.graph_node_id",
            "source.imported_node_id",
            "source.op_type",
            "source.generic_op",
            "source.onnx_name",
            "source.dispatch_group",
            "source.op_role",
        ]:
            pattern = re.escape(key) + r"\s*=\s*(?:\"([^\"]*)\"|([0-9]+)\s*:\s*i64)"
            m_attr = re.search(pattern, window)
            if m_attr:
                attrs[key] = m_attr.group(1) if m_attr.group(1) is not None else m_attr.group(2)
        if shape is None:
            shape = {"m": -1, "n": -1, "k": -1, "dtype": dtype}
        semantic_ref = (
            f"{function_ref}:source_graph_node_id={attrs['source.graph_node_id']}"
            if "source.graph_node_id" in attrs
            else f"{function_ref}:hir.fused_matmul_bias_relu:{idx}"
        )
        regions.append(
            IrFusedRegion(
                semantic_target_ref=semantic_ref,
                function_ref=function_ref,
                op_name="hir.fused_matmul_bias_relu",
                shape=shape,
                dtype=shape.get("dtype", dtype),
                source_graph_node_id=attrs.get("source.graph_node_id"),
                source_imported_node_id=attrs.get("source.imported_node_id"),
                source_op_type=attrs.get("source.op_type"),
                source_generic_op=attrs.get("source.generic_op"),
                source_onnx_name=attrs.get("source.onnx_name"),
                source_dispatch_group=attrs.get("source.dispatch_group"),
                source_op_role=attrs.get("source.op_role"),
            )
        )
    return regions


def artifact_plan_by_workload(plans_payload: dict[str, Any], workload_id: str | None) -> dict[str, Any] | None:
    plans = plans_payload.get("plans", [])
    if not plans:
        return None
    if workload_id is None:
        return plans[0]
    for plan in plans:
        if plan.get("workload_id") == workload_id:
            return plan
    return None


def selected_operation(plan: dict[str, Any]) -> dict[str, Any] | None:
    ops = plan.get("operations") or []
    return ops[0] if ops else None


def resolve_mapping(regions: list[IrFusedRegion], plan: dict[str, Any] | None) -> dict[str, Any]:
    if not plan:
        return {
            "status": "AMBIGUOUS_MAPPING",
            "resolved": False,
            "reason": "missing_triton_plan",
            "missing_provenance": ["triton_plan"],
        }
    op = selected_operation(plan)
    if not op:
        return {
            "status": "AMBIGUOUS_MAPPING",
            "resolved": False,
            "reason": "missing_triton_operation",
            "missing_provenance": ["triton_operation"],
        }
    explicit = op.get("ir_mapping") or op.get("source_ir") or {}
    if explicit:
        for region in regions:
            if explicit.get("source_graph_node_id") is not None and str(explicit["source_graph_node_id"]) != str(region.source_graph_node_id):
                continue
            if explicit.get("source_dispatch_group") and explicit["source_dispatch_group"] != region.source_dispatch_group:
                continue
            if explicit.get("function_ref") and explicit["function_ref"] != region.function_ref:
                continue
            return {
                "status": "VERIFIED_DIRECT_MAPPING",
                "resolved": True,
                "semantic_target_ref": region.semantic_target_ref,
                "function_ref": region.function_ref,
                "reason": "artifact_contains_explicit_ir_mapping",
            }
        return {
            "status": "AMBIGUOUS_MAPPING",
            "resolved": False,
            "reason": "explicit_ir_mapping_does_not_match_mlir",
            "missing_provenance": ["matching_source_graph_node_id_or_dispatch_group"],
        }

    shape = canonical_shape(op.get("shape", {}))
    matches = [
        r for r in regions
        if r.op_name == "hir.fused_matmul_bias_relu"
        and r.dtype in {"f32", "fp32"}
        and canonical_shape(r.shape) == shape
    ]
    if len(matches) == 1 and len(regions) == 1:
        return {
            "status": "VERIFIED_DERIVED_MAPPING",
            "resolved": True,
            "semantic_target_ref": matches[0].semantic_target_ref,
            "function_ref": matches[0].function_ref,
            "reason": "single_fused_region_shape_dtype_op_match",
            "truth_boundary": "derived_mapping_requires_single_candidate_region_not_sufficient_for_multi_region_graphs",
        }
    return {
        "status": "AMBIGUOUS_MAPPING",
        "resolved": False,
        "reason": "triton_artifact_lacks_ir_provenance_or_shape_is_not_unique",
        "candidate_region_count": len(regions),
        "shape_match_count": len(matches),
        "missing_provenance": [
            "source.graph_node_id",
            "source.imported_node_id",
            "source.dispatch_group",
            "function_ref",
            "model_or_graph_identity",
        ],
    }


def evidence_refs(paths: dict[str, Path]) -> list[dict[str, Any]]:
    mapping = [
        ("measured_candidate_evidence", "candidate_sweep"),
        ("calibration_model", "cost_model"),
        ("shadow_selection_plan", "plans"),
        ("plan_dispatch_validation", "plan_validation"),
        ("evaluation_summary", "summary"),
    ]
    refs = []
    for kind, key in mapping:
        path = paths.get(key)
        if path and path.exists():
            refs.append({
                "evidence_kind": kind,
                "artifact": str(path),
                "sha256": sha256_file(path),
                "truth_boundary": "referenced_existing_artifact_not_copied_into_candidate",
            })
    return refs


def parse_expected_hashes(values: list[str] | None) -> dict[str, str]:
    expected: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError("--expect-sha256 must use name=hex format")
        name, value = item.split("=", 1)
        expected[name] = value
    return expected


def artifact_integrity(paths: dict[str, Path], expected: dict[str, str]) -> dict[str, Any]:
    checks = []
    ok = True
    for name, expected_hash in expected.items():
        path = paths.get(name)
        if path is None or not path.exists():
            checks.append({"name": name, "status": "missing_artifact", "expected_sha256": expected_hash})
            ok = False
            continue
        actual = sha256_file(path)
        status = "ok" if actual == expected_hash else "sha256_mismatch"
        if status != "ok":
            ok = False
        checks.append({
            "name": name,
            "path": str(path),
            "status": status,
            "expected_sha256": expected_hash,
            "actual_sha256": actual,
        })
    return {
        "status": "ok" if ok else "rejected_artifact",
        "checks": checks,
    }


def candidate_id(semantic_target_ref: str, op: dict[str, Any], rank: dict[str, Any]) -> str:
    cfg = rank["config_id"]
    tile = rank.get("tile_features", {})
    shape = op.get("shape", {})
    return ":".join([
        semantic_target_ref or "unresolved_ir_root",
        "scope=fused_region",
        "backend=cuda",
        "triton_generated_fused_kernel",
        "contract=triton_kernel_config_contract_shadow",
        f"kernel={op.get('semantic_kernel_id') or SUPPORTED_SEMANTIC_KERNEL}",
        f"config={cfg}",
        f"tile=bm{tile.get('block_m')}_bn{tile.get('block_n')}_bk{tile.get('block_k')}",
        f"warps={tile.get('num_warps')}",
        f"stages={tile.get('num_stages')}",
        f"dtype={shape.get('dtype', SUPPORTED_DTYPE)}",
        "target=nvidia_gtx1650_maxq",
    ])


def build_candidates(
    plan: dict[str, Any],
    mapping: dict[str, Any],
    refs: list[dict[str, Any]],
    integrity: dict[str, Any],
) -> list[dict[str, Any]]:
    op = selected_operation(plan)
    if not op:
        return []
    ranking = op.get("candidate_ranking") or op.get("predicted_candidates") or []
    semantic_ref = mapping.get("semantic_target_ref", "")
    candidates = []
    for rank in ranking:
        cfg = rank.get("config_id") or rank.get("variant")
        tile = rank.get("tile_features") or rank.get("candidate_features") or {}
        c = {
            "candidate_id": candidate_id(semantic_ref, op, {**rank, "config_id": cfg, "tile_features": tile}),
            "provider_id": PROVIDER_ID,
            "provider_version": PROVIDER_VERSION,
            "scope": "fused_region",
            "semantic_target_ref": semantic_ref,
            "backend": SUPPORTED_BACKEND,
            "implementation_kind": "triton_generated_fused_kernel",
            "runtime_contract_kind": "triton_kernel_config_contract_shadow",
            "kernel_id": op.get("semantic_kernel_id") or SUPPORTED_SEMANTIC_KERNEL,
            "selected_kernel": op.get("selected_kernel", SUPPORTED_KERNEL),
            "config_id": cfg,
            "tile": {
                "block_m": tile.get("block_m"),
                "block_n": tile.get("block_n"),
                "block_k": tile.get("block_k"),
            },
            "num_warps": tile.get("num_warps"),
            "num_stages": tile.get("num_stages"),
            "dtype": (op.get("shape") or {}).get("dtype", SUPPORTED_DTYPE),
            "target_gpu": "NVIDIA GeForce GTX 1650 with Max-Q Design",
            "compute_capability": [7, 5],
            "evidence_refs": refs,
            "confidence": op.get("confidence") or {
                "level": op.get("confidence_level"),
                "score": op.get("confidence_score"),
                "distance_to_training_distribution": op.get("distance_to_training_distribution"),
                "interpolation_kind": op.get("interpolation_kind"),
            },
            "truth_boundary": TRUTH_BOUNDARY,
        }
        if integrity.get("status") == "rejected_artifact":
            c["feasibility"] = {
                "status": "rejected",
                "reason": "rejected_artifact",
            }
        elif mapping.get("resolved"):
            c["feasibility"] = {
                "status": "feasible_predicted" if op.get("profile_match") == "unseen" else "feasible_verified",
                "reason": "shadow_candidate_ir_mapping_resolved_artifacts_valid",
            }
        else:
            c["feasibility"] = {
                "status": "deferred",
                "reason": "deferred_missing_mapping",
            }
        candidates.append(c)
    return candidates


def shadow_policy_result(plan: dict[str, Any], candidates: list[dict[str, Any]], mapping: dict[str, Any]) -> dict[str, Any]:
    op = selected_operation(plan) if plan else None
    selected_config = op.get("selected_config_id") if op else None
    selected = next((c for c in candidates if c.get("config_id") == selected_config), None)
    return {
        "policy_id": "triton_repaired_calibrated_selector_shadow",
        "mode": "shadow_only_non_authoritative",
        "selected_candidate_id": selected.get("candidate_id") if selected else None,
        "considered_candidate_ids": [c["candidate_id"] for c in candidates],
        "selection_source": op.get("selection_source") if op else None,
        "profile_match": op.get("profile_match") if op else None,
        "confidence": op.get("confidence") if op else None,
        "mapping_status": mapping.get("status"),
        "production_plan_affected": False,
        "runtime_dispatch_affected": False,
        "truth_boundary": (
            "shadow_policy_result_not_consumed_by_canonical_planselection_"
            "or_executionplan_or_runtime"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    mlir_path = Path(args.mlir)
    paths = {
        "candidate_sweep": Path(args.candidate_sweep),
        "cost_model": Path(args.cost_model),
        "plans": Path(args.plans),
        "plan_validation": Path(args.plan_validation),
        "summary": Path(args.summary) if args.summary else None,
    }
    expected_hashes = parse_expected_hashes(args.expect_sha256)
    integrity = artifact_integrity({k: v for k, v in paths.items() if v is not None}, expected_hashes)
    plans_payload = read_json(paths["plans"])
    plan = artifact_plan_by_workload(plans_payload, args.workload_id)
    regions = parse_mlir_fused_regions(mlir_path)
    mapping = resolve_mapping(regions, plan)
    refs = evidence_refs({k: v for k, v in paths.items() if v is not None})
    candidates = build_candidates(plan, mapping, refs, integrity) if plan else []
    policy = shadow_policy_result(plan, candidates, mapping) if plan else {}
    payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "compiler_commit": args.compiler_commit,
        "utc_timestamp": utc_now(),
        "provider": {"provider_id": PROVIDER_ID, "provider_version": PROVIDER_VERSION},
        "source_mlir": str(mlir_path),
        "source_triton_artifacts": [
            {"name": k, "path": str(v), "sha256": sha256_file(v)}
            for k, v in paths.items() if v is not None and v.exists()
        ],
        "artifact_integrity": integrity,
        "ir_regions": [r.__dict__ for r in regions],
        "ir_mapping": mapping,
        "candidates": candidates,
        "shadow_policy_result": policy,
        "production_plan_affected": False,
        "runtime_dispatch_affected": False,
        "truth_boundary": TRUTH_BOUNDARY,
    }
    if args.output:
        write_json(Path(args.output), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mlir", required=True)
    p.add_argument("--candidate-sweep", default="trace/matmul_postop_triton_fused_candidate_sweep.json")
    p.add_argument("--cost-model", default="trace/matmul_postop_triton_fused_config_repair_cost_model.json")
    p.add_argument("--plans", default="trace/matmul_postop_triton_fused_config_repair_plans.json")
    p.add_argument("--plan-validation", default="trace/matmul_postop_triton_fused_config_repair_plan_validation.json")
    p.add_argument("--summary", default="trace/matmul_postop_triton_fused_config_repair_summary.json")
    p.add_argument("--workload-id")
    p.add_argument("--compiler-commit", default="unknown")
    p.add_argument("--output", default="trace/triton_shadow_candidate_analysis.json")
    p.add_argument("--expect-sha256", action="append")
    return p.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
