#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_triton_shadow_candidate_provider as shadow  # noqa: E402


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def write_mlir(path: Path, duplicate: bool = False, shape: tuple[int, int, int] = (1, 3072, 768)) -> None:
    m, n, k = shape
    op = f"""
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {{
    source.graph_node_id = 7 : i64,
    source.imported_node_id = 70 : i64,
    source.op_type = "MatMulBiasRelu",
    source.generic_op = "nn.matmul_bias_relu",
    source.onnx_name = "/fused0",
    source.dispatch_group = "dg_7",
    source.op_role = "dispatch_root"
  }} : (tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>, tensor<{n}xf32>) -> tensor<{m}x{n}xf32>
"""
    op2 = ""
    if duplicate:
        op2 = f"""
  %1 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {{
    source.graph_node_id = 8 : i64,
    source.imported_node_id = 80 : i64,
    source.op_type = "MatMulBiasRelu",
    source.generic_op = "nn.matmul_bias_relu",
    source.onnx_name = "/fused1",
    source.dispatch_group = "dg_8",
    source.op_role = "dispatch_root"
  }} : (tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>, tensor<{n}xf32>) -> tensor<{m}x{n}xf32>
"""
    path.write_text(
        f"""
module {{
  func.func @main(%lhs: tensor<{m}x{k}xf32>, %rhs: tensor<{k}x{n}xf32>, %bias: tensor<{n}xf32>) -> tensor<{m}x{n}xf32> {{
{op}{op2}
    return %0 : tensor<{m}x{n}xf32>
  }}
}}
""",
        encoding="utf-8",
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_artifacts(tmp: Path) -> dict[str, Path]:
    names = {
        "candidate_sweep": "trace/matmul_postop_triton_fused_candidate_sweep.json",
        "cost_model": "trace/matmul_postop_triton_fused_config_repair_cost_model.json",
        "plans": "trace/matmul_postop_triton_fused_config_repair_plans.json",
        "plan_validation": "trace/matmul_postop_triton_fused_config_repair_plan_validation.json",
        "summary": "trace/matmul_postop_triton_fused_config_repair_summary.json",
    }
    out = {}
    for key, rel in names.items():
        dst = tmp / Path(rel).name
        dst.write_bytes((ROOT / rel).read_bytes())
        out[key] = dst
    return out


def run_shadow(tmp: Path, mlir: Path, paths: dict[str, Path], workload: str = "rep_m1_k768_n3072", extra=None) -> dict:
    output = tmp / "shadow.json"
    cmd = [
        sys.executable,
        str(TOOLS / "run_triton_shadow_candidate_provider.py"),
        "--mlir", str(mlir),
        "--candidate-sweep", str(paths["candidate_sweep"]),
        "--cost-model", str(paths["cost_model"]),
        "--plans", str(paths["plans"]),
        "--plan-validation", str(paths["plan_validation"]),
        "--summary", str(paths["summary"]),
        "--workload-id", workload,
        "--compiler-commit", "test",
        "--output", str(output),
    ]
    if extra:
        cmd.extend(extra)
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(output.read_text(encoding="utf-8"))


def mutate_plan_with_mapping(paths: dict[str, Path], graph_node_id: int = 7) -> None:
    payload = json.loads(paths["plans"].read_text(encoding="utf-8"))
    plan = payload["plans"][0]
    op = plan["operations"][0]
    op["ir_mapping"] = {
        "function_ref": "main",
        "source_graph_node_id": graph_node_id,
        "source_dispatch_group": f"dg_{graph_node_id}",
    }
    paths["plans"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        paths = copy_artifacts(tmp)
        mlir = tmp / "single.mlir"
        write_mlir(mlir)

        result = run_shadow(tmp, mlir, paths)
        require(result["schema"] == shadow.SCHEMA, "schema mismatch")
        require(result["production_plan_affected"] is False, "production plan affected")
        require(result["runtime_dispatch_affected"] is False, "runtime dispatch affected")
        require(result["ir_mapping"]["status"] == "VERIFIED_DERIVED_MAPPING", "single-op mapping should derive")
        require(result["shadow_policy_result"]["mode"] == "shadow_only_non_authoritative", "policy must be shadow")
        require(result["shadow_policy_result"]["production_plan_affected"] is False, "policy leaked to production")
        require(result["candidates"], "candidates missing")
        require(len(result["shadow_policy_result"]["considered_candidate_ids"]) == len(result["candidates"]), "policy/candidate mismatch")
        ids = [c["candidate_id"] for c in result["candidates"]]
        require(len(ids) == len(set(ids)), "candidate IDs must be unique")
        require(all("predicted_latency_ms" not in cid for cid in ids), "latency leaked into candidate ID")
        require(all(c["runtime_contract_kind"] == "triton_kernel_config_contract_shadow" for c in result["candidates"]), "wrong contract kind")
        require(all(c["feasibility"]["status"] == "feasible_predicted" for c in result["candidates"]), "expected predicted feasibility")

        # Ambiguous mapping: same shape appears twice, so shape-only mapping must not bind.
        duplicate_mlir = tmp / "duplicate.mlir"
        write_mlir(duplicate_mlir, duplicate=True)
        ambiguous = run_shadow(tmp, duplicate_mlir, paths)
        require(ambiguous["ir_mapping"]["status"] == "AMBIGUOUS_MAPPING", "duplicate mapping must stay ambiguous")
        require(all(c["feasibility"]["reason"] == "deferred_missing_mapping" for c in ambiguous["candidates"]), "ambiguous candidates must defer")
        require(ambiguous["shadow_policy_result"]["production_plan_affected"] is False, "ambiguous result leaked")

        # Explicit artifact mapping succeeds when provenance matches.
        (tmp / "explicit").mkdir(exist_ok=True)
        explicit_paths = copy_artifacts(tmp / "explicit")
        mutate_plan_with_mapping(explicit_paths, 7)
        explicit = run_shadow(tmp, mlir, explicit_paths)
        require(explicit["ir_mapping"]["status"] == "VERIFIED_DIRECT_MAPPING", "explicit mapping should bind")
        require(explicit["candidates"][0]["semantic_target_ref"].startswith("main:source_graph_node_id=7"), "semantic root missing")

        (tmp / "wrong").mkdir(exist_ok=True)
        wrong_paths = copy_artifacts(tmp / "wrong")
        mutate_plan_with_mapping(wrong_paths, 99)
        wrong = run_shadow(tmp, mlir, wrong_paths)
        require(wrong["ir_mapping"]["status"] == "AMBIGUOUS_MAPPING", "wrong mapping must reject")

        # Evidence references are by artifact hash and measured/predicted kinds stay distinct.
        kinds = {r["evidence_kind"] for r in result["candidates"][0]["evidence_refs"]}
        require("measured_candidate_evidence" in kinds, "measured evidence missing")
        require("calibration_model" in kinds, "calibration evidence missing")
        require("shadow_selection_plan" in kinds, "shadow plan evidence missing")
        require("plan_dispatch_validation" in kinds, "plan validation evidence missing")
        require(all("samples_ms" not in json.dumps(c) for c in result["candidates"]), "raw samples leaked into candidates")
        require(all("oracle_config" not in json.dumps(c) for c in result["candidates"]), "oracle leaked into candidates")
        for c in result["candidates"]:
            identity = c["candidate_id"]
            require("regret" not in identity, "evaluation field leaked into identity")

        # Artifact hash mismatch rejects candidate feasibility without modifying artifacts.
        bad_hash = "0" * 64
        mismatch = run_shadow(tmp, mlir, paths, extra=["--expect-sha256", f"cost_model={bad_hash}"])
        require(mismatch["artifact_integrity"]["status"] == "rejected_artifact", "hash mismatch must reject")
        require(all(c["feasibility"]["reason"] == "rejected_artifact" for c in mismatch["candidates"]), "artifact rejection missing")

        # Low-confidence/unseen fields remain shadow evidence, not production plan data.
        policy = result["shadow_policy_result"]
        require(policy["runtime_dispatch_affected"] is False, "runtime dispatch affected")
        require(result.get("schema") != "runtime_execution_plan", "shadow output must not be ExecutionPlan")

        # Provider module does not import Triton or benchmark.
        source = (TOOLS / "run_triton_shadow_candidate_provider.py").read_text(encoding="utf-8")
        forbidden = ["import " + "triton", "import " + "torch", "triton" + ".jit", "cuda" + ".synchronize"]
        for token in forbidden:
            require(token not in source, f"forbidden provider token present: {token}")

        # Production plan files are not modified by the shadow tool.
        before = sha(ROOT / "trace/matmul_postop_triton_fused_config_repair_plans.json")
        _ = run_shadow(tmp, mlir, paths)
        after = sha(ROOT / "trace/matmul_postop_triton_fused_config_repair_plans.json")
        require(before == after, "raw Triton plan artifact changed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
