#!/usr/bin/env python3
"""Stage 16: multi-domain calibration and compatibility analysis.

Builds CandidateEvidenceRecords for 4 independent domains (primary
32x32x32/8x8x8, cube64 64x64x64/8x8x8, altk 32x32x32/8x8x4, rect
32x64x32/8x8x8), each with real static/backend evidence (Stage 12 for
existing candidates, freshly-extracted for new uk4/rect candidates) and
real measured evidence (Stage 13 for existing, freshly-measured for new),
then:
  - builds one multi-domain profile JSON
  - runs static + calibrated ranking PER DOMAIN (never merged)
  - proves cross-domain evidence rejection (compatibility matrix)
  - runs the Stage 15 selector for each domain via the real evidence
  - writes every required Stage 16 artifact
"""
import dataclasses
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import aarch64_schedule_candidate_model as cm  # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "backend_codegen", "aarch64_matmul_bias_relu_schedule_multidomain")
STAGE12_JSON = os.path.join(REPO_ROOT, "artifacts", "backend_codegen", "aarch64_matmul_bias_relu_scheduling", "schedule_comparison_results.json")
STAGE13_JSON = os.path.join(REPO_ROOT, "artifacts", "backend_codegen", "aarch64_matmul_bias_relu_pi_scheduling", "pi_validation_results.json")
S16_PI_JSON = os.path.join(ART, "pi_multidomain_results.json")
S16_EVIDENCE_DIR = "/tmp/s16_evidence"

NEW_STATIC_CANDIDATES = {
    # label: (shape, tile_m, tile_n, tile_k, unroll_k, mir_evidence_dir_name)
    "cube64_uk4": ("64x64x64", 8, 8, 8, 4, "cube64_uk4"),
    "altk_uk4": ("32x32x32", 8, 8, 4, 4, "altk_uk4"),
    "rect_uk1": ("32x64x32", 8, 8, 8, 1, "rect_uk1"),
    "rect_uk2": ("32x64x32", 8, 8, 8, 2, "rect_uk2"),
    "rect_uk4": ("32x64x32", 8, 8, 8, 4, "rect_uk4"),
}

DOMAIN_LABELS = {
    "primary": {"uk1": "primary_unroll1", "uk2": "primary_unroll2", "uk4": "primary_full_unroll"},
    "cube64": {"uk1": "cube64_uk1", "uk2": "cube64_uk2", "uk4": "cube64_uk4"},
    "altk": {"uk1": "altk_uk1", "uk2": "altk_uk2", "uk4": "altk_uk4"},
    "rect": {"uk1": "rect_uk1", "uk2": "rect_uk2", "uk4": "rect_uk4"},
}
DOMAIN_SHAPE_TILE = {
    "primary": ("32x32x32", 8, 8, 8),
    "cube64": ("64x64x64", 8, 8, 8),
    "altk": ("32x32x32", 8, 8, 4),
    "rect": ("32x64x32", 8, 8, 8),
}


def build_static_evidence_from_metrics(key, mir_dir_name, label):
    reg_path = os.path.join(S16_EVIDENCE_DIR, mir_dir_name, "register_metrics.json")
    sched_path = os.path.join(S16_EVIDENCE_DIR, mir_dir_name, "schedule_metrics.json")
    with open(reg_path) as f:
        reg = json.load(f)
    with open(sched_path) as f:
        sched = json.load(f)
    cid = key.canonical_id()

    def mk(v, level="mir", note=""):
        return cm.ev(v, level, reg_path, "stage16_multidomain_v1", "aarch64-linux-gnu/cortex-a76", "stage16", cid, note=note)

    tile_k_trip_pre = key.shape_k // key.tile_k
    k_trip_post = tile_k_trip_pre // key.schedule_unroll_k
    m_trip = key.shape_m // key.tile_m
    n_trip = key.shape_n // key.tile_n
    static_ir = cm.StaticIRBEvidence(
        m_trip_count=mk(m_trip, "mlir"), n_trip_count=mk(n_trip, "mlir"),
        k_trip_count_post_unroll=mk(k_trip_post, "mlir"),
        surviving_loop_count=mk((1 if m_trip > 1 else 0) + (1 if n_trip > 1 else 0) + (0 if k_trip_post <= 1 else 1), "mlir"),
        k_loop_collapsed=mk(k_trip_post <= 1, "mlir"),
        static_vector_contract_count=mk(sched["fmla_count"], "llvm_ir"),
        estimated_dynamic_k_loop_iterations=mk(k_trip_post, "mlir"),
        estimated_dynamic_loop_control_reduction_pct=mk((1.0 - k_trip_post / tile_k_trip_pre) * 100.0, "mlir"),
    )
    llvm_backend = cm.LLVMBackendEvidence(
        pre_ra_approx_peak_live_vector_registers=mk(reg["stages"]["pre_ra"]["approx_peak_live_vector_registers"]),
        physical_vector_registers_post_ra=mk(reg["stages"]["post_ra"]["physical_vector_registers_referenced"]),
        spill_store_count=mk(reg["comparison"]["spill_stores_inserted_by_ra"]),
        reload_load_count=mk(reg["comparison"]["reload_loads_inserted_by_ra"]),
        stack_frame_bytes=mk(reg["final_stack_frame_bytes"]),
        object_bytes=mk(os.path.getsize(os.path.join(ART, "compiled", f"{label}.o")), "assembly"),
        static_fmla_asm_count=mk(sched.get("assembly_cross_check", {}).get("asm_fmla_count", sched["fmla_count"]), "assembly"),
        accumulator_chains=mk(sched.get("accumulator_chains")),
        max_accumulator_chain_length=mk(sched.get("max_accumulator_chain_length")),
        llvm_mca_block_rthroughput=mk(None, "llvm_mca", note="not collected for Stage 16 candidates"),
    )
    return static_ir, llvm_backend


def build_measured_evidence(key, pi_record):
    cid = key.canonical_id()
    agg = pi_record["benchmark_aggregate"]
    fg = pi_record["measurement_groups"][0]["benchmark"]

    def mk(v, note=""):
        return cm.ev(v, "raspberry_pi_measured", S16_PI_JSON, "stage16_multidomain_pi_v1", "raspberry_pi_5_cortex_a76", "stage16", cid, note=note)

    return cm.MeasuredHardwareEvidence(
        median_latency_ms=mk(agg["median_of_medians_ms"]), p95_latency_ms=mk(fg.get("p95_ms")),
        mean_latency_ms=mk(fg.get("mean_ms")), stddev_latency_ms=mk(fg.get("stddev_ms")),
        cv=mk(agg["cv_of_medians"]), correctness_pass=mk(pi_record["correctness_pass"]),
        hardware_identity=mk("raspberry_pi_5_cortex_a76"),
    )


def main():
    os.makedirs(ART, exist_ok=True)

    stage12_records = cm.load_stage12_records(STAGE12_JSON)
    stage13_alias = cm.load_stage13_measured(STAGE13_JSON, stage12_records)
    by_key = {r.key: r for r in stage12_records.values()}

    with open(S16_PI_JSON) as f:
        s16_pi = json.load(f)["candidates"]

    # Build/augment records for the 5 genuinely new candidates.
    for label, (shape, tm, tn, tk, uk, mir_dir) in NEW_STATIC_CANDIDATES.items():
        shape_m, shape_n, shape_k = (int(x) for x in shape.split("x"))
        key = cm.CandidateKey(shape_m=shape_m, shape_n=shape_n, shape_k=shape_k, tile_m=tm, tile_n=tn, tile_k=tk, schedule_unroll_k=uk)
        static_ir, llvm_backend = build_static_evidence_from_metrics(key, mir_dir, label)
        measured = build_measured_evidence(key, s16_pi[label]) if label in s16_pi else None
        by_key[key] = cm.CandidateEvidenceRecord(key=key, label=label, static_ir=static_ir, llvm_backend=llvm_backend, measured=measured)

    # Attach fresh Stage 16 Pi measurements to the EXISTING (Stage 12/13)
    # records too, where this run re-measured them (cube64_uk1/uk2,
    # altk_uk1/uk2), so domain rankings use one consistent measurement
    # session rather than mixing Stage 13 and Stage 16 sessions within a
    # domain.
    for label in ("cube64_uk1", "cube64_uk2", "altk_uk1", "altk_uk2"):
        domain_key = label.rsplit("_", 1)
        shape_tile = DOMAIN_SHAPE_TILE["cube64" if label.startswith("cube64") else "altk"]
        shape_m, shape_n, shape_k = (int(x) for x in shape_tile[0].split("x"))
        uk = 1 if label.endswith("uk1") else 2
        key = cm.CandidateKey(shape_m=shape_m, shape_n=shape_n, shape_k=shape_k,
                               tile_m=shape_tile[1], tile_n=shape_tile[2], tile_k=shape_tile[3], schedule_unroll_k=uk)
        if key in by_key and label in s16_pi:
            by_key[key] = dataclasses.replace(by_key[key], measured=build_measured_evidence(key, s16_pi[label]))

    # ---- Domain summary + per-domain ranking (static + calibrated) ----
    domain_summary = {}
    static_ranking = {}
    calibrated_ranking = {}
    measured_pool_all = [(r.key, r.measured) for r in by_key.values() if r.measured]

    for domain_name, ukmap in DOMAIN_LABELS.items():
        shape, tm, tn, tk = DOMAIN_SHAPE_TILE[domain_name]
        shape_m, shape_n, shape_k = (int(x) for x in shape.split("x"))
        domain_records = []
        rows = []
        for ukname, label in ukmap.items():
            uk = int(ukname[2:])
            key = cm.CandidateKey(shape_m=shape_m, shape_n=shape_n, shape_k=shape_k, tile_m=tm, tile_n=tn, tile_k=tk, schedule_unroll_k=uk)
            if key not in by_key:
                continue
            r = by_key[key]
            domain_records.append(r)
            rows.append({
                "label": label, "unroll_k": uk,
                "spills": r.llvm_backend.spill_store_count.value, "reloads": r.llvm_backend.reload_load_count.value,
                "stack_bytes": r.llvm_backend.stack_frame_bytes.value, "object_bytes": r.llvm_backend.object_bytes.value,
                "median_ms": r.measured.median_latency_ms.value if r.measured else None,
                "correctness_pass": r.measured.correctness_pass.value if r.measured else None,
            })

        measured_present = [r for r in domain_records if r.measured]
        winner_label, winner_class = None, "inconclusive"
        if measured_present:
            fastest = min(measured_present, key=lambda r: r.measured.median_latency_ms.value)
            others = [r for r in measured_present if r is not fastest]
            beats_all = all(
                ((o.measured.median_latency_ms.value - fastest.measured.median_latency_ms.value) / o.measured.median_latency_ms.value * 100.0)
                > max(3.0, 2 * 100 * max(fastest.measured.cv.value or 0, o.measured.cv.value or 0))
                for o in others
            ) if others else True
            winner_label = fastest.label
            winner_class = f"uk{fastest.key.schedule_unroll_k}_winner" if beats_all else "statistically_tied"

        domain_summary[domain_name] = {"shape": shape, "tile": {"m": tm, "n": tn, "k": tk}, "rows": rows, "winner": winner_label, "classification": winner_class}

        static_ranked = cm.rank_candidates(domain_records, cm.RANKING_MODE_STATIC_SOFT_PENALTY)
        static_ranking[domain_name] = [cm.breakdown_to_dict(b) for b in static_ranked]

        domain_pool = [(r.key, r.measured) for r in domain_records if r.measured]  # exact-domain-only pool, never cross-domain
        calibrated_ranked = cm.rank_candidates(domain_records, cm.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=domain_pool)
        calibrated_ranking[domain_name] = [cm.breakdown_to_dict(b) for b in calibrated_ranked]

    with open(os.path.join(ART, "domain_summary.json"), "w") as f:
        json.dump(domain_summary, f, indent=2)
    with open(os.path.join(ART, "static_ranking.json"), "w") as f:
        json.dump(static_ranking, f, indent=2)
    with open(os.path.join(ART, "calibrated_ranking.json"), "w") as f:
        json.dump(calibrated_ranking, f, indent=2)

    # ---- Multi-domain profile (task section 8/14) ----
    multi_domain_profile = {
        "schema_version": "stage16_multidomain_profile_v1",
        "benchmark_methodology_version": cm.BENCHMARK_METHODOLOGY_VERSION,
        "hardware_identity": "raspberry_pi_5_cortex_a76",
        "domains": {},
    }
    for domain_name, ukmap in DOMAIN_LABELS.items():
        shape, tm, tn, tk = DOMAIN_SHAPE_TILE[domain_name]
        domain_entries = {}
        for ukname, label in ukmap.items():
            uk = int(ukname[2:])
            shape_m, shape_n, shape_k = (int(x) for x in shape.split("x"))
            key = cm.CandidateKey(shape_m=shape_m, shape_n=shape_n, shape_k=shape_k, tile_m=tm, tile_n=tn, tile_k=tk, schedule_unroll_k=uk)
            if key not in by_key or not by_key[key].measured:
                continue
            r = by_key[key]
            domain_entries[r.key.canonical_id()] = {
                "label": label, "median_latency_ms": r.measured.median_latency_ms.value,
                "cv": r.measured.cv.value, "correctness_pass": r.measured.correctness_pass.value,
                "spills": r.llvm_backend.spill_store_count.value, "reloads": r.llvm_backend.reload_load_count.value,
            }
        multi_domain_profile["domains"][domain_name] = {
            "domain_identity": {"target_arch": "aarch64", "target_cpu": "cortex-a76", "target_features": "none",
                                 "dtype": "f32", "tile": {"m": tm, "n": tn, "k": tk}, "shape": shape},
            "candidates": domain_entries,
        }
    with open(os.path.join(ART, "multi_domain_profile.json"), "w") as f:
        json.dump(multi_domain_profile, f, indent=2)

    # ---- Cross-domain rejection report (task section 10) ----
    rejection_report = []
    for q_domain, (q_shape, q_tm, q_tn, q_tk) in DOMAIN_SHAPE_TILE.items():
        q_shape_m, q_shape_n, q_shape_k = (int(x) for x in q_shape.split("x"))
        query_key = cm.CandidateKey(shape_m=q_shape_m, shape_n=q_shape_n, shape_k=q_shape_k, tile_m=q_tm, tile_n=q_tn, tile_k=q_tk, schedule_unroll_k=2)
        for e_domain, (e_shape, e_tm, e_tn, e_tk) in DOMAIN_SHAPE_TILE.items():
            if q_domain == e_domain:
                continue
            e_shape_m, e_shape_n, e_shape_k = (int(x) for x in e_shape.split("x"))
            evidence_key = cm.CandidateKey(shape_m=e_shape_m, shape_n=e_shape_n, shape_k=e_shape_k, tile_m=e_tm, tile_n=e_tn, tile_k=e_tk, schedule_unroll_k=2)
            compat = cm.check_compatibility(query_key, evidence_key, cm.BENCHMARK_METHODOLOGY_VERSION)
            rejection_report.append({
                "query_domain": q_domain, "evidence_domain": e_domain,
                "query_key": query_key.canonical_id(), "evidence_key": evidence_key.canonical_id(),
                "compatibility_level": compat["level"], "confidence": compat["confidence"],
                "exact_match_rejected": compat["level"] != cm.EXACT_MATCH,
            })
    with open(os.path.join(ART, "cross_domain_rejection_report.json"), "w") as f:
        json.dump(rejection_report, f, indent=2)

    # ---- Compatibility matrix (full pairwise, all candidates) ----
    compat_matrix = []
    all_keys = list(by_key.keys())
    for qk in all_keys:
        for ek in all_keys:
            if qk == ek:
                continue
            compat = cm.check_compatibility(qk, ek, cm.BENCHMARK_METHODOLOGY_VERSION)
            if compat["level"] != cm.INCOMPATIBLE:
                compat_matrix.append({"query": qk.canonical_id(), "evidence": ek.canonical_id(), "level": compat["level"], "confidence": compat["confidence"]})
    with open(os.path.join(ART, "compatibility_matrix.json"), "w") as f:
        json.dump(compat_matrix, f, indent=2)

    print("Domain winners:", file=sys.stderr)
    for d, s in domain_summary.items():
        print(f"  {d}: winner={s['winner']} class={s['classification']}", file=sys.stderr)
    print(f"\nWrote artifacts to {ART}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
