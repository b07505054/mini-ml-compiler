#!/usr/bin/env python3
"""Slice 18A exact AArch64 candidate artifact/evidence/plan integration.

This deliberately wraps the existing MLIR compiler and MIR analyzers.  It
does not parse MIR in serving passes and does not introduce a second lowering
pipeline.  Scope is fixed to fused f32 32x32x32, tile 8x8x8, uk={1,2,4}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import statistics
import random

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "mlir_passes/test/backend_codegen/matmul_bias_relu_tiled_32x32x32.mlir"
COMPILE = ROOT / "mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh"
EXTRACT = ROOT / "tools/extract_aarch64_candidate_mir.py"
ANALYZE = ROOT / "tools/analyze_aarch64_candidate_mir.py"
TRIPLE, CPU, PROFILE = "aarch64-linux-gnu", "cortex-a76", "raspberry-pi5-cortex-a76-cpu"
ENTRY, ABI = "_mlir_ciface_matmul_bias_relu_tiled_32x32x32", "mlir_ciface_memref_f32_v1"
PIPELINE, LOOP = "aarch64_tiled_scheduled_v1", "tiled_mnk_row_major_v1"
FAMILY = "hir.fused_matmul_bias_relu"
UKS = (1, 2, 4)


class EvidenceError(ValueError):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(argv: list[str]) -> str:
    p = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
    if p.returncode:
        raise EvidenceError(f"command failed ({' '.join(argv)}):\n{p.stdout}\n{p.stderr}")
    return p.stdout


def candidate_id(uk: int) -> str:
    if uk not in UKS:
        raise EvidenceError(f"unsupported schedule_unroll_k={uk}")
    return f"tile8x8x8_uk{uk}"


def identity(uk: int) -> dict:
    return {
        "candidate_id": candidate_id(uk), "operator": FAMILY,
        "kernel_family": "aarch64_generated_fused_matmul_bias_relu",
        "dtype": "f32", "shape": {"m": 32, "n": 32, "k": 32},
        "target": {"triple": TRIPLE, "cpu": CPU, "features": [],
                   "target_profile_id": PROFILE},
        "lowering": {"pipeline_id": PIPELINE, "tile_m": 8, "tile_n": 8,
                     "tile_k": 8, "schedule_unroll_k": uk,
                     "vector_width_bits": 128, "loop_order_id": LOOP},
        "microkernel_id": "hir_fused_matmul_bias_relu_tiled_scheduled_v1",
        "entry_point": ENTRY, "abi_version": ABI,
    }


def artifact_paths(out: Path, uk: int) -> dict[str, Path]:
    cid = candidate_id(uk)
    stem = f"matmul_bias_relu_32x32x32_tm8_tn8_tk8_uk{uk}"
    prefix = f"32x32x32_tm8_tn8_tk8_uk{uk}_greedy_misched-default"
    d = out / cid
    return {
        "dir": d, "llvm_dialect": d / f"{stem}_llvm.mlir",
        "llvm_ir": d / f"{stem}.ll",
        "mir_post_isel": d / f"{prefix}_post_isel.mir",
        "mir_pre_scheduler": d / f"{prefix}_pre_scheduler.mir",
        "mir_pre_ra": d / f"{prefix}_pre_ra.mir",
        "mir_post_ra": d / f"{prefix}_post_ra.mir",
        "mir_post_prologue_epilogue": d / f"{prefix}_post_prologue_epilogue.mir",
        "assembly": d / f"{prefix}.s", "object": d / f"{prefix}.o",
        "register_metrics": d / "register_metrics.json",
        "evidence": d / "backend_evidence.json",
    }


def _tool_version(tool: str) -> str | None:
    p = subprocess.run([tool, "--version"], text=True, capture_output=True)
    return p.stdout.splitlines()[0] if p.returncode == 0 and p.stdout else None


def build_candidate(out: Path, uk: int) -> dict:
    p = artifact_paths(out, uk)
    p["dir"].mkdir(parents=True, exist_ok=True)
    stem = f"matmul_bias_relu_32x32x32_tm8_tn8_tk8_uk{uk}"
    run(["bash", str(COMPILE), "--variant", "tiled-scheduled", "--tile-m", "8",
         "--tile-n", "8", "--tile-k", "8", "--schedule-unroll-k", str(uk),
         str(FIXTURE), str(p["dir"]), stem])
    run(["python3", str(EXTRACT), "--llvm-ir", str(p["llvm_ir"]), "--cpu", CPU,
         "--shape", "32x32x32", "--tile-m", "8", "--tile-n", "8", "--tile-k", "8",
         "--schedule-unroll-k", str(uk), "--output-dir", str(p["dir"])])
    run(["python3", str(ANALYZE), "--post-isel", str(p["mir_post_isel"]),
         "--pre-ra", str(p["mir_pre_ra"]), "--post-ra", str(p["mir_post_ra"]),
         "--post-prologue-epilogue", str(p["mir_post_prologue_epilogue"]),
         "--output", str(p["register_metrics"])])
    metrics = json.loads(p["register_metrics"].read_text())
    stages = metrics["stages"]
    asm = p["assembly"].read_text()
    static_count = sum(1 for line in asm.splitlines()
                       if line.strip() and not line.lstrip().startswith((".", "#", "//")))
    fmla = len(re.findall(r"^\s*fmla\s", asm, re.M))
    text_size = None
    size = subprocess.run(["llvm-size", "-A", str(p["object"])], text=True, capture_output=True)
    if size.returncode == 0:
        m = re.search(r"^\.text\s+(\d+)", size.stdout, re.M)
        text_size = int(m.group(1)) if m else None
    rel = lambda x: os.path.relpath(x, ROOT)
    refs = {k + "_ref": rel(v) for k, v in p.items()
            if k not in {"dir", "evidence", "register_metrics"}}
    hashes = {k + "_sha256": sha(v) for k, v in p.items()
              if k not in {"dir", "evidence"} and v.is_file()}
    git_rev = run(["git", "rev-parse", "HEAD"]).strip()
    dirty = bool(run(["git", "status", "--short"]).strip())
    evidence = {
        "schema_version": 1, **identity(uk),
        "artifacts": {**refs, **hashes, "backend_evidence_ref": rel(p["evidence"])},
        "static_backend_evidence": {
            "object_size_bytes": p["object"].stat().st_size,
            "text_size_bytes": text_size, "static_instruction_count": static_count,
            "fmla_count": fmla,
            "spill_store_count": stages["post_ra"]["spill_stores"],
            "reload_load_count": stages["post_ra"]["spill_reloads"],
            "spill_slot_bytes": stages["post_ra"]["spill_slot_bytes"],
            "physical_vector_registers_referenced":
                stages["post_ra"]["physical_vector_registers_referenced"],
            "approximate_peak_live_vector_registers":
                stages["pre_ra"]["approx_peak_live_vector_registers"],
        },
        "estimated_backend_evidence": {
            "llvm_mca_estimated_cycles": None, "methodology": None},
        "measured_backend_evidence": None,
        "validation": {"codegen_succeeded": True, "llvm_ir_verified": True,
                       "correctness_passed": None, "measured_on_target": False},
        "provenance": {"compiler_revision": git_rev,
                       "llvm_version": _tool_version("llc"),
                       "working_tree_clean": not dirty},
    }
    p["evidence"].write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def validate_evidence(e: dict, *, root: Path | None = None) -> list[str]:
    root = ROOT if root is None else root
    reasons = []
    try:
        uk = int(e["lowering"]["schedule_unroll_k"])
        expected = identity(uk)
    except Exception as ex:
        return [f"malformed_identity:{ex}"]
    for key in ("candidate_id", "operator", "kernel_family", "dtype", "shape",
                "target", "lowering", "microkernel_id", "entry_point", "abi_version"):
        if e.get(key) != expected.get(key):
            reasons.append(f"{key}_mismatch")
    a = e.get("artifacts", {})
    obj_ref, expected_sha = a.get("object_ref"), a.get("object_sha256")
    if not obj_ref or not expected_sha:
        reasons.append("missing_object_identity")
    else:
        obj = root / obj_ref
        if not obj.is_file():
            reasons.append("object_missing")
        elif sha(obj) != expected_sha:
            reasons.append("object_sha256_mismatch")
    v = e.get("validation", {})
    if v.get("codegen_succeeded") is not True:
        reasons.append("codegen_failed")
    if v.get("llvm_ir_verified") is not True:
        reasons.append("llvm_ir_not_verified")
    if v.get("correctness_passed") is False:
        reasons.append("correctness_failed")
    return reasons


def select(evidence: list[dict]) -> dict:
    trace = []
    survivors = []
    for e in evidence:
        reasons = validate_evidence(e)
        row = {"candidate_id": e.get("candidate_id"), "accepted": not reasons,
               "reasons": reasons}
        trace.append(row)
        if not reasons:
            survivors.append(e)
    if not survivors:
        raise EvidenceError("all candidates failed hard gates")
    measured = [e for e in survivors if isinstance(e.get("measured_backend_evidence"), dict)
                and e["measured_backend_evidence"].get("target_profile_id") == PROFILE
                and e["measured_backend_evidence"].get("correctness_passed") is True
                and e["measured_backend_evidence"].get("latency_p50_ms") is not None]
    if measured:
        mode = "exact_raspberry_pi_measurement"
        key = lambda e: (e["measured_backend_evidence"]["latency_p50_ms"], e["candidate_id"])
    else:
        mode = "deterministic_static_lexicographic_estimate"
        def key(e):
            s, est = e["static_backend_evidence"], e["estimated_backend_evidence"]
            return (est.get("llvm_mca_estimated_cycles") is None,
                    est.get("llvm_mca_estimated_cycles") or 0,
                    s.get("reload_load_count") is None, s.get("reload_load_count") or 0,
                    s.get("text_size_bytes") is None, s.get("text_size_bytes") or 0,
                    s.get("object_size_bytes") is None, s.get("object_size_bytes") or 0,
                    e["candidate_id"])
    ranked = sorted(measured or survivors, key=key)
    return {"selection_mode": mode, "selected_candidate_id": ranked[0]["candidate_id"],
            "selector_trace": trace,
            "ranked_candidate_ids": [e["candidate_id"] for e in ranked],
            "truth_boundary": "measured_exact_target" if measured else
                "static_backend_evidence_estimate_not_performance_calibrated"}


def execution_plan(selected: dict, decision: dict, selection_ref: str) -> dict:
    native = {
        "decision_kind": "aarch64_native_exact_candidate_selection",
        **{k: selected[k] for k in ("candidate_id", "operator", "kernel_family", "dtype",
                                    "shape", "target", "lowering", "microkernel_id",
                                    "entry_point", "abi_version")},
        "object_ref": selected["artifacts"]["object_ref"],
        "object_sha256": selected["artifacts"]["object_sha256"],
        "backend_evidence_ref": selected["artifacts"]["backend_evidence_ref"],
        "correctness_evidence_ref": None, "measurement_evidence_ref": None,
        "selection_mode": decision["selection_mode"],
        "selection_trace_ref": selection_ref, "runtime_no_redecision": True,
    }
    for key in ("static_selected_candidate", "calibrated_selected_candidate",
                "measurement_evidence_ref", "measurement_policy_ref",
                "benchmark_protocol_version", "target_fingerprint"):
        if decision.get(key) is not None:
            native[key] = decision[key]
    return {
        "schema": "execution_plan", "schema_version": "2.0.0",
        "plan_id": f"aarch64-native-32x32x32-{selected['candidate_id']}",
        "provenance": {"compiler_tool": "aarch64_backend_evidence.py",
                       "model_spec_ref":
                           "mlir_passes/test/backend_codegen/matmul_bias_relu_tiled_32x32x32.mlir",
                       "capability_bundle": {"hardware_profile_ref":
                           "hardware/raspberry_pi5_cortex_a76.json",
                           "backend_profile_refs": [], "kernel_profile_refs": []},
                       "truth_boundary": "exact generated operator object; fixed shape and target only"},
        "model_identity": {"model_id": "fused_matmul_bias_relu_32x32x32"},
        "global_decisions": {"quantization": {}, "memory": {}, "serving": {}},
        "function_plans": [{
            "function_name": "matmul_bias_relu_tiled_32x32x32",
            "serving_phase": "other", "backend": {"selected_backend": "aarch64_native_object"},
            "per_op_decisions": [{"op_name": "fused_matmul_bias_relu",
                "op_type": FAMILY, "native_execution": native}],
        }],
    }


PROTOCOL_VERSION = "slice19_aarch64_native_batched_v1"
RUNTIME_REVISION = "5b56607cf84d8acda2691f02762f50d30332a8d1"


def measurement_plan(evidence: dict, *, selection_mode="measurement_candidate") -> dict:
    decision = {"selection_mode": selection_mode}
    return execution_plan(evidence, decision, "measurement_protocol.json")


def validate_measurement(e: dict, measurement: dict) -> list[str]:
    reasons = validate_evidence(e)
    if measurement.get("schema_version") != 1:
        reasons.append("measurement_schema_mismatch")
    if measurement.get("benchmark_protocol_version") != PROTOCOL_VERSION:
        reasons.append("benchmark_protocol_mismatch")
    a = measurement.get("artifact_identity", {})
    w = measurement.get("workload_identity", {})
    t = measurement.get("target_fingerprint", {})
    expected = {
        "candidate_id": e["candidate_id"],
        "object_sha256": e["artifacts"]["object_sha256"],
        "entry_point": e["entry_point"], "abi_version": e["abi_version"],
        "compiler_revision": e["provenance"]["compiler_revision"],
        "runtime_revision": RUNTIME_REVISION,
    }
    for key, value in expected.items():
        actual = measurement.get("candidate_id") if key == "candidate_id" else a.get(key)
        if actual != value:
            reasons.append(f"measurement_{key}_mismatch")
    expected_w = {"operator": e["operator"], "kernel_family": e["kernel_family"],
                  "dtype": e["dtype"],
                  "shape": [32,32,32], "tile": [8,8,8],
                  "schedule_unroll_k": e["lowering"]["schedule_unroll_k"],
                  "vector_width_bits": 128, "loop_order_id": LOOP,
                  "lowering_pipeline_id": PIPELINE}
    for key, value in expected_w.items():
        if w.get(key) != value:
            reasons.append(f"measurement_{key}_mismatch")
    if (t.get("architecture") != "aarch64" or t.get("cpu") != CPU
            or t.get("features") != []):
        reasons.append("measurement_target_mismatch")
    c = measurement.get("correctness", {})
    if c.get("passed") is not True: reasons.append("measurement_correctness_failed")
    if c.get("guard_buffers_intact") is not True: reasons.append("measurement_guard_failed")
    sessions = measurement.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        reasons.append("measurement_sessions_unavailable")
    else:
        for s in sessions:
            proof=s.get("identity_proof",{})
            if proof.get("runtime_redecision_count") != 0:
                reasons.append("measurement_runtime_redecision")
                break
    return list(dict.fromkeys(reasons))


def aggregate_sessions(sessions: list[dict], seed=19) -> dict:
    p50=[s["metrics"]["p50_ms"] for s in sessions]
    p95=[s["metrics"]["p95_ms"] for s in sessions]
    means=[s["metrics"]["mean_ms"] for s in sessions]
    rng=random.Random(seed); boots=[]
    for _ in range(10000):
        sample=[means[rng.randrange(len(means))] for _ in means]
        boots.append(statistics.mean(sample))
    boots.sort()
    return {"median_of_session_p50_ms":statistics.median(p50),
            "median_of_session_p95_ms":statistics.median(p95),
            "mean_of_session_means_ms":statistics.mean(means),
            "stddev_across_session_means_ms":statistics.stdev(means),
            "standard_error_ms":statistics.stdev(means)/(len(means)**.5),
            "bootstrap_ci95_low_ms":boots[249],
            "bootstrap_ci95_high_ms":boots[9749],
            "session_count":len(sessions)}


def calibrated_select(evidence: list[dict], measurements: dict[str,dict],
                      equivalence_pct=3.0) -> dict:
    static=select(evidence)
    accepted=[]; rejected=[]; rows=[]
    for e in evidence:
        m=measurements.get(e["candidate_id"])
        reasons=["measurement_missing"] if m is None else validate_measurement(e,m)
        trace={"candidate_id":e["candidate_id"],"accepted":not reasons,"reasons":reasons}
        (accepted if not reasons else rejected).append(trace)
        if not reasons:
            rows.append((e,m,m["aggregate"]))
    if not rows:
        raise EvidenceError("no exact compatible measurements")
    rows.sort(key=lambda x:(x[2]["median_of_session_p50_ms"],x[0]["candidate_id"]))
    best=rows[0]; tied=[]
    for row in rows:
        rel=(row[2]["median_of_session_p50_ms"]/best[2]["median_of_session_p50_ms"]-1)*100
        overlap=not (row[2]["bootstrap_ci95_low_ms"]>best[2]["bootstrap_ci95_high_ms"]
                     or best[2]["bootstrap_ci95_low_ms"]>row[2]["bootstrap_ci95_high_ms"])
        if rel <= equivalence_pct or overlap:
            tied.append(row)
    if len(tied)>1:
        tied.sort(key=lambda x:(x[0]["static_backend_evidence"]["text_size_bytes"],
                                x[0]["static_backend_evidence"]["object_size_bytes"],
                                x[0]["candidate_id"]))
        winner=tied[0]; reason="session_uncertainty_or_3pct_equivalence_tie_code_size_break"
    else:
        winner=best; reason="distinguishable_exact_target_session_measurement"
    return {"selection_mode":"exact_target_calibrated",
      "static_selected_candidate":static["selected_candidate_id"],
      "calibrated_selected_candidate":winner[0]["candidate_id"],
      "selected_candidate_id":winner[0]["candidate_id"],
      "selection_changed":winner[0]["candidate_id"]!=static["selected_candidate_id"],
      "equivalence_threshold_pct":equivalence_pct,
      "uncertainty_method":"deterministic_session_mean_bootstrap_10000_seed19",
      "selection_reason":reason,"accepted_measurements":accepted,
      "rejected_measurements":rejected,
      "candidate_aggregates":{e["candidate_id"]:a for e,_,a in rows}}
def build_all(out: Path) -> tuple[list[dict], dict, dict]:
    evidence = [build_candidate(out, uk) for uk in UKS]
    decision = select(evidence)
    selection_path = out / "selection_trace.json"
    selection_path.write_text(json.dumps(decision, indent=2) + "\n")
    chosen = next(e for e in evidence if e["candidate_id"] == decision["selected_candidate_id"])
    plan = execution_plan(chosen, decision, os.path.relpath(selection_path, ROOT))
    (out / "execution_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return evidence, decision, plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    _, decision, _ = build_all(Path(args.output_dir).resolve())
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
