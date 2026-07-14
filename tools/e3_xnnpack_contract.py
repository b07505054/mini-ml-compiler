#!/usr/bin/env python3
"""Generate an E3 compiler-owned ExecuTorch/XNNPACK comparison contract.

This tool is intentionally artifact-driven: feasibility is derived from the
provided PTE/runner/source identities, and policy selection is restricted to the
real requested-thread variants that the common runner can request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

EXECUTORCH_TAG = "v1.3.1"
EXECUTORCH_COMMIT = "e2f18eb23c45bd22ca332b0b8b49a81de304b472"
XNNPACK_COMMIT = "1adaa7c709d4839d29e1f219cb962b01c9e6a905"
PROVIDER_ID = "executorch_xnnpack_candidate_provider"
RUNTIME_CONTRACT = "executorch_xnnpack_runner_contract"
POLICY_ID = "e3a_static_xnnpack_requested_thread_policy"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def compiler_commit(repo: Path, override: str | None) -> str:
    if override:
        return override
    try:
        return git_head(repo)
    except Exception:
        return "unknown"


def candidate(shape: str, threads: int, pte: Path, pte_sha: str, runner_sha: str) -> dict:
    cid = (
        f"fused_matmul_bias_relu:scope=fused_region:backend=cpu:library=xnnpack:"
        f"external_library_delegate:contract={RUNTIME_CONTRACT}:dtype=fp32:"
        f"pte={pte_sha[:12]}:threads={threads}:axis=runtime_threadpool:"
        f"strategy=xnnpack_requested_threads:target=raspberry-pi5-cortex-a76-cpu"
    )
    return {
        "candidate_id": cid,
        "provider_id": PROVIDER_ID,
        "scope_kind": "fused_region",
        "semantic_target_ref": "fused_matmul_bias_relu",
        "backend": "cpu",
        "library": "xnnpack",
        "implementation_kind": "external_library_delegate",
        "runtime_contract_kind": RUNTIME_CONTRACT,
        "dtype": "fp32",
        "shape": shape,
        "artifact": {"pte_ref": str(pte), "pte_sha256": pte_sha, "runner_sha256": runner_sha},
        "provenance": {
            "executorch_tag": EXECUTORCH_TAG,
            "executorch_commit": EXECUTORCH_COMMIT,
            "xnnpack_commit": XNNPACK_COMMIT,
        },
        "requested_thread_count": threads,
        "thread_schedule": {
            "thread_count": threads,
            "partition_axis": "runtime_threadpool",
            "partition_strategy": "xnnpack_requested_threads",
        },
        "feasibility": {"status": "unknown", "reason": "provider_enumerated_requires_artifact_feasibility"},
        "truth_boundary": "compiler_enumerated_candidate_not_runtime_execution",
    }


def feasible(cand: dict, args: argparse.Namespace, pte_sha: str, runner_sha: str) -> dict:
    reasons = []
    if cand["semantic_target_ref"] != "fused_matmul_bias_relu":
        reasons.append("wrong_semantic_scope")
    if cand["dtype"] != "fp32":
        reasons.append("wrong_dtype")
    if args.target_profile != "raspberry-pi5-cortex-a76-cpu":
        reasons.append("target_profile_mismatch")
    if cand["artifact"]["pte_sha256"] != pte_sha:
        reasons.append("pte_hash_mismatch")
    if cand["artifact"]["runner_sha256"] != runner_sha:
        reasons.append("runner_hash_mismatch")
    if args.executorch_commit != EXECUTORCH_COMMIT:
        reasons.append("executorch_commit_mismatch")
    if args.xnnpack_commit != XNNPACK_COMMIT:
        reasons.append("xnnpack_commit_mismatch")
    if not args.xnnpack_delegated:
        reasons.append("xnnpack_delegation_unproven")
    if not args.input_binding_compatible:
        reasons.append("input_binding_incompatible")
    if cand["requested_thread_count"] not in (1, 4):
        reasons.append("unsupported_requested_thread_mode")
    if cand["requested_thread_count"] > args.physical_compute_units:
        reasons.append("rejected_exceeds_compute_units")
    return {
        "status": "feasible" if not reasons else "rejected",
        "reason": "executorch_xnnpack_artifacts_validated" if not reasons else reasons[0],
        "all_reasons": reasons,
    }


def canonical_hash(contract: dict) -> str:
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, help="MxNxK, e.g. 64x64x64")
    ap.add_argument("--pte", required=True, type=Path)
    ap.add_argument("--runner", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--selected-threads", required=True, type=int, choices=(1, 4))
    ap.add_argument("--target-profile", default="raspberry-pi5-cortex-a76-cpu")
    ap.add_argument("--executorch-commit", default=EXECUTORCH_COMMIT)
    ap.add_argument("--xnnpack-commit", default=XNNPACK_COMMIT)
    ap.add_argument("--physical-compute-units", type=int, default=4)
    ap.add_argument("--xnnpack-delegated", action="store_true")
    ap.add_argument("--input-binding-compatible", action="store_true")
    ap.add_argument("--compiler-commit")
    args = ap.parse_args()

    pte_sha = sha_file(args.pte)
    runner_sha = sha_file(args.runner)
    candidates = [candidate(args.shape, t, args.pte, pte_sha, runner_sha) for t in (1, 4)]
    for cand in candidates:
        cand["feasibility"] = feasible(cand, args, pte_sha, runner_sha)
    selected = next(c for c in candidates if c["requested_thread_count"] == args.selected_threads)
    if selected["feasibility"]["status"] != "feasible":
        raise SystemExit(f"selected candidate is not feasible: {selected['feasibility']}")
    rejected = [{"candidate_id": c["candidate_id"], "reason": "not_selected_by_static_e3a_policy"} for c in candidates if c is not selected]
    compiler_repo = Path(__file__).resolve().parents[1]
    contract = {
        "schema": "e3_compiler_xnnpack_comparison_contract",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_ir_identity": {"semantic_graph": "fused_matmul_bias_relu", "shape": args.shape, "dtype": "fp32"},
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate": selected,
        "considered_candidates": candidates,
        "provider_id": PROVIDER_ID,
        "backend": "cpu",
        "library": "xnnpack",
        "runner_contract": RUNTIME_CONTRACT,
        "pte": {"path": str(args.pte), "sha256": pte_sha},
        "runner": {"path": str(args.runner), "sha256": runner_sha},
        "executorch": {"tag": EXECUTORCH_TAG, "commit": EXECUTORCH_COMMIT},
        "xnnpack": {"commit": XNNPACK_COMMIT},
        "requested_thread_mode": {"kind": "explicit", "threads": args.selected_threads},
        "target_profile": {"id": args.target_profile, "arch": "aarch64", "device": "raspberry-pi5-cortex-a76-cpu"},
        "policy_result": {
            "policy_id": POLICY_ID,
            "policy_version": "e3a.v1",
            "selected_candidate_id": selected["candidate_id"],
            "considered_candidate_ids": [c["candidate_id"] for c in candidates],
            "rejected_candidates": rejected,
            "selection_reason": "static_requested_thread_selection_no_performance_claim",
            "objective_summary": "E3A validation path; calibration may replace this only after evidence freeze",
            "truth_boundary": "compiler_policy_result_not_measured_runtime",
        },
        "truth_boundary": "compiler_generated_contract_runtime_must_validate_all_identities",
        "compiler": {"repo": str(compiler_repo), "commit": compiler_commit(compiler_repo, args.compiler_commit), "tool": "tools/e3_xnnpack_contract.py"},
    }
    contract["contract_sha256"] = canonical_hash(contract)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(contract["contract_sha256"])


if __name__ == "__main__":
    main()
