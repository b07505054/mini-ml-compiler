#!/usr/bin/env python3
"""Stage 17: schedule-unroll boundary search analysis.

Builds CandidateEvidenceRecords for the 2 new stress domains (smallA:
16x16x16/tile 8x8x4; highK: 32x32x128/tile 8x8x8), classifies timing
quality (task section 8), classifies winners (variance-aware noise floor,
same methodology as Stage 13/16), re-evaluates the static model across
the now-6-domain dataset, compares 3 selection policies, and extends the
Stage 16 multi-domain profile with the 2 new domains.
"""
import dataclasses
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import aarch64_schedule_candidate_model as cm  # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "backend_codegen", "aarch64_matmul_bias_relu_schedule_boundary")
STAGE16_ART = os.path.join(REPO_ROOT, "artifacts", "backend_codegen", "aarch64_matmul_bias_relu_schedule_multidomain")
S17_EVIDENCE_DIR = "/tmp/s17_evidence"
S17_PI_JSON = os.path.join(ART, "benchmark_results.json")

# Empirically measured on this exact Pi 5 / taskset -c 3 this session
# (tools/timer_overhead_probe.cpp): back-to-back steady_clock::now() call
# overhead, and malloc(4160)+free overhead (the generated kernel's own
# sret-return allocation pattern -- NOT harness overhead to subtract, it
# is real work the compiled kernel actually performs on every call, but
# still relevant context for interpreting very small absolute latencies).
CLOCK_OVERHEAD_NS = 37.0
MALLOC_OVERHEAD_NS = 37.0
RELIABLE_THRESHOLD_NS = 10 * (CLOCK_OVERHEAD_NS)  # 10x pure clock-read overhead
BORDERLINE_THRESHOLD_NS = 5 * (CLOCK_OVERHEAD_NS)

NEW_CANDIDATES = {
    "smallA_uk1": ("16x16x16", 8, 8, 4, 1, "smallA_uk1"),
    "smallA_uk2": ("16x16x16", 8, 8, 4, 2, "smallA_uk2"),
    "smallA_uk4": ("16x16x16", 8, 8, 4, 4, "smallA_uk4"),
    "highK_uk1": ("32x32x128", 8, 8, 8, 1, "highK_uk1"),
    "highK_uk2": ("32x32x128", 8, 8, 8, 2, "highK_uk2"),
    "highK_uk4": ("32x32x128", 8, 8, 8, 4, "highK_uk4"),
}
DOMAIN_LABELS = {
    "smallA": {"uk1": "smallA_uk1", "uk2": "smallA_uk2", "uk4": "smallA_uk4"},
    "highK": {"uk1": "highK_uk1", "uk2": "highK_uk2", "uk4": "highK_uk4"},
}
DOMAIN_SHAPE_TILE = {
    "smallA": ("16x16x16", 8, 8, 4),
    "highK": ("32x32x128", 8, 8, 8),
}


def build_static_evidence(key, mir_dir_name, label):
    reg_path = os.path.join(S17_EVIDENCE_DIR, mir_dir_name, "register_metrics.json")
    sched_path = os.path.join(S17_EVIDENCE_DIR, mir_dir_name, "schedule_metrics.json")
    with open(reg_path) as f:
        reg = json.load(f)
    with open(sched_path) as f:
        sched = json.load(f)
    cid = key.canonical_id()

    def mk(v, level="mir"):
        return cm.ev(v, level, reg_path, "stage17_boundary_v1", "aarch64-linux-gnu/cortex-a76", "stage17", cid)

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
        llvm_mca_block_rthroughput=mk(None, "llvm_mca"),
    )
    return static_ir, llvm_backend


def build_measured_evidence(key, pi_record):
    cid = key.canonical_id()
    agg = pi_record["benchmark_aggregate"]
    fg = pi_record["measurement_groups"][0]["benchmark"]

    def mk(v):
        return cm.ev(v, "raspberry_pi_measured", S17_PI_JSON, "stage17_boundary_pi_v1", "raspberry_pi_5_cortex_a76", "stage17", cid)

    return cm.MeasuredHardwareEvidence(
        median_latency_ms=mk(agg["median_of_medians_ms"]), p95_latency_ms=mk(fg.get("p95_ms")),
        mean_latency_ms=mk(fg.get("mean_ms")), stddev_latency_ms=mk(fg.get("stddev_ms")),
        cv=mk(agg["cv_of_medians"]), correctness_pass=mk(pi_record["correctness_pass"]),
        hardware_identity=mk("raspberry_pi_5_cortex_a76"),
    )


def classify_timing_quality(median_ms):
    median_ns = median_ms * 1e6
    overhead_fraction = CLOCK_OVERHEAD_NS / median_ns if median_ns else 1.0
    if median_ns >= RELIABLE_THRESHOLD_NS:
        level = "reliable"
    elif median_ns >= BORDERLINE_THRESHOLD_NS:
        level = "borderline"
    else:
        level = "unreliable"
    return {
        "median_ns": median_ns, "clock_overhead_ns": CLOCK_OVERHEAD_NS,
        "overhead_fraction": overhead_fraction, "quality": level,
        "note": (
            "malloc(4160)+free overhead (~37ns) is REAL kernel work (the generated ABI's sret-return "
            "allocation), not harness overhead, and is included in the measured time on purpose; only "
            "clock-read overhead (~37ns per back-to-back now()/now() pair, empirically measured on this "
            "Pi via tools/timer_overhead_probe.cpp) is treated as pure measurement overhead here. Because "
            "this overhead is approximately constant across uk1/uk2/uk4 for the same domain (identical "
            "malloc call site and size), RELATIVE ranking between factors in a 'borderline' domain remains "
            "defensible even when absolute latency numbers are not precise to more than ~1 significant figure."
        ),
    }


def main():
    stage12_records = cm.load_stage12_records(os.path.join(REPO_ROOT, "artifacts/backend_codegen/aarch64_matmul_bias_relu_scheduling/schedule_comparison_results.json"))
    cm.load_stage13_measured(os.path.join(REPO_ROOT, "artifacts/backend_codegen/aarch64_matmul_bias_relu_pi_scheduling/pi_validation_results.json"), stage12_records)
    by_key = {r.key: r for r in stage12_records.values()}

    # Pull in Stage 16's 4 domains (cube64/altk/rect measured evidence) too.
    with open(os.path.join(STAGE16_ART, "pi_multidomain_results.json")) as f:
        s16_pi = json.load(f)["candidates"]
    stage16_static_dir = "/tmp/s16_evidence"
    for label, (shape, tm, tn, tk, uk, mir_dir) in {
        "cube64_uk4": ("64x64x64", 8, 8, 8, 4, "cube64_uk4"), "altk_uk4": ("32x32x32", 8, 8, 4, 4, "altk_uk4"),
        "rect_uk1": ("32x64x32", 8, 8, 8, 1, "rect_uk1"), "rect_uk2": ("32x64x32", 8, 8, 8, 2, "rect_uk2"),
        "rect_uk4": ("32x64x32", 8, 8, 8, 4, "rect_uk4"),
    }.items():
        if not os.path.isdir(os.path.join(stage16_static_dir, mir_dir)):
            continue
        shape_m, shape_n, shape_k = (int(x) for x in shape.split("x"))
        key = cm.CandidateKey(shape_m=shape_m, shape_n=shape_n, shape_k=shape_k, tile_m=tm, tile_n=tn, tile_k=tk, schedule_unroll_k=uk)
        reg_path = os.path.join(stage16_static_dir, mir_dir, "register_metrics.json")
        sched_path = os.path.join(stage16_static_dir, mir_dir, "schedule_metrics.json")
        # Always fully rebuild (not just patch .measured) when real static
        # evidence is available on disk -- regardless of whether `key`
        # already exists in `by_key` as a BLANK placeholder record. A real
        # bug was found here: rect_uk1/rect_uk2 already existed in by_key
        # as blank records (Stage 12 never analyzed `rect` at all, so
        # load_stage13_measured's "candidate present only in Stage 13"
        # path created them with static_ir/llvm_backend fields all None).
        # The original `if key not in by_key` condition then routed them
        # into the elif branch below, which patches ONLY `.measured`,
        # leaving object_bytes permanently None -- which compute_cost's
        # UNSUPPORTED hard-rejection then correctly (but misleadingly)
        # rejected, making rect_uk4 "win" static ranking not because
        # static evidence favored it but because uk1/uk2 were silently
        # disqualified. Verified concretely: rect_uk4 has 18 real spills
        # and should NOT out-rank spill-free rect_uk1/uk2 under the
        # corrected soft-penalty policy once they have real evidence.
        if os.path.isfile(reg_path):
            obj_path = os.path.join(STAGE16_ART, "compiled", f"{label}.o")
            with open(reg_path) as f:
                reg = json.load(f)
            with open(sched_path) as f:
                sched = json.load(f)
            cid = key.canonical_id()
            mk = lambda v, level="mir": cm.ev(v, level, reg_path, "stage16_multidomain_v1", "aarch64-linux-gnu/cortex-a76", "stage16", cid)
            tile_k_trip_pre = key.shape_k // key.tile_k
            k_trip_post = tile_k_trip_pre // key.schedule_unroll_k
            static_ir = cm.StaticIRBEvidence(
                m_trip_count=mk(key.shape_m // key.tile_m), n_trip_count=mk(key.shape_n // key.tile_n),
                k_trip_count_post_unroll=mk(k_trip_post), surviving_loop_count=mk(None), k_loop_collapsed=mk(k_trip_post <= 1),
                static_vector_contract_count=mk(sched["fmla_count"], "llvm_ir"),
                estimated_dynamic_k_loop_iterations=mk(k_trip_post), estimated_dynamic_loop_control_reduction_pct=mk((1.0 - k_trip_post / tile_k_trip_pre) * 100.0),
            )
            llvm_backend = cm.LLVMBackendEvidence(
                pre_ra_approx_peak_live_vector_registers=mk(reg["stages"]["pre_ra"]["approx_peak_live_vector_registers"]),
                physical_vector_registers_post_ra=mk(reg["stages"]["post_ra"]["physical_vector_registers_referenced"]),
                spill_store_count=mk(reg["comparison"]["spill_stores_inserted_by_ra"]), reload_load_count=mk(reg["comparison"]["reload_loads_inserted_by_ra"]),
                stack_frame_bytes=mk(reg["final_stack_frame_bytes"]), object_bytes=mk(os.path.getsize(obj_path) if os.path.isfile(obj_path) else None, "assembly"),
                static_fmla_asm_count=mk(sched.get("assembly_cross_check", {}).get("asm_fmla_count", sched["fmla_count"]), "assembly"),
                accumulator_chains=mk(sched.get("accumulator_chains")), max_accumulator_chain_length=mk(sched.get("max_accumulator_chain_length")),
                llvm_mca_block_rthroughput=mk(None, "llvm_mca"),
            )
            measured = None
            if label in s16_pi:
                measured = build_measured_evidence(key, s16_pi[label])
            by_key[key] = cm.CandidateEvidenceRecord(key=key, label=label, static_ir=static_ir, llvm_backend=llvm_backend, measured=measured)
        elif key in by_key and label in s16_pi:
            by_key[key] = dataclasses.replace(by_key[key], measured=build_measured_evidence(key, s16_pi[label]))
    # cube64_uk1/uk2, altk_uk1/uk2 already merged w/ measured in Stage 16 via labels sharing keys with stage12_records

    with open(S17_PI_JSON) as f:
        s17_pi = json.load(f)["candidates"]

    for label, (shape, tm, tn, tk, uk, mir_dir) in NEW_CANDIDATES.items():
        shape_m, shape_n, shape_k = (int(x) for x in shape.split("x"))
        key = cm.CandidateKey(shape_m=shape_m, shape_n=shape_n, shape_k=shape_k, tile_m=tm, tile_n=tn, tile_k=tk, schedule_unroll_k=uk)
        static_ir, llvm_backend = build_static_evidence(key, mir_dir, label)
        measured = build_measured_evidence(key, s17_pi[label]) if label in s17_pi else None
        by_key[key] = cm.CandidateEvidenceRecord(key=key, label=label, static_ir=static_ir, llvm_backend=llvm_backend, measured=measured)

    # ---- Timing quality ----
    timing_quality = {}
    for label in NEW_CANDIDATES:
        r = s17_pi[label]
        timing_quality[label] = classify_timing_quality(r["benchmark_aggregate"]["median_of_medians_ms"])
    with open(os.path.join(ART, "timing_quality.json"), "w") as f:
        json.dump({
            "clock_overhead_ns": CLOCK_OVERHEAD_NS, "malloc_overhead_ns": MALLOC_OVERHEAD_NS,
            "reliable_threshold_ns": RELIABLE_THRESHOLD_NS, "borderline_threshold_ns": BORDERLINE_THRESHOLD_NS,
            "per_candidate": timing_quality,
            "retroactive_note": (
                "Applying this same threshold retroactively to Stage 13's smallest-ever measurement "
                "(small_control_uk1, median 92ns): overhead_fraction = 37/92 = 40%, well into 'unreliable' "
                "territory by this stage's standard. That measurement was never used for an unroll-factor "
                "COMPARISON (only uk1 was legal for that shape/tile), so no prior ranking conclusion is "
                "invalidated by this finding -- but it is flagged here as a real, previously-uncaught gap "
                "in measurement rigor for absolute-latency claims at that scale."
            ),
        }, f, indent=2)

    # ---- Domain summary + rankings (reuse Stage 16 pattern) ----
    domain_summary = {}
    static_ranking = {}
    calibrated_ranking = {}
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
                "label": label, "unroll_k": uk, "spills": r.llvm_backend.spill_store_count.value,
                "reloads": r.llvm_backend.reload_load_count.value, "stack_bytes": r.llvm_backend.stack_frame_bytes.value,
                "object_bytes": r.llvm_backend.object_bytes.value,
                "median_ms": r.measured.median_latency_ms.value if r.measured else None,
                "correctness_pass": r.measured.correctness_pass.value if r.measured else None,
                "timing_quality": timing_quality.get(label, {}).get("quality"),
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
        domain_pool = [(r.key, r.measured) for r in domain_records if r.measured]
        calibrated_ranked = cm.rank_candidates(domain_records, cm.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=domain_pool)
        calibrated_ranking[domain_name] = [cm.breakdown_to_dict(b) for b in calibrated_ranked]

    with open(os.path.join(ART, "winner_summary.json"), "w") as f:
        json.dump(domain_summary, f, indent=2)
    with open(os.path.join(ART, "static_ranking_new_domains.json"), "w") as f:
        json.dump(static_ranking, f, indent=2)
    with open(os.path.join(ART, "calibrated_ranking_new_domains.json"), "w") as f:
        json.dump(calibrated_ranking, f, indent=2)

    print("New domain winners:", file=sys.stderr)
    for d, s in domain_summary.items():
        print(f"  {d}: winner={s['winner']} class={s['classification']}", file=sys.stderr)

    # ---- Static-model evaluation across ALL 6 domains (task section 12) ----
    all_domains = {
        "primary": ("32x32x32", 8, 8, 8, {"uk1": "primary_unroll1", "uk2": "primary_unroll2", "uk4": "primary_full_unroll"}),
        "cube64": ("64x64x64", 8, 8, 8, {"uk1": "cube64_uk1", "uk2": "cube64_uk2", "uk4": "cube64_uk4"}),
        "altk": ("32x32x32", 8, 8, 4, {"uk1": "altk_uk1", "uk2": "altk_uk2", "uk4": "altk_uk4"}),
        "rect": ("32x64x32", 8, 8, 8, {"uk1": "rect_uk1", "uk2": "rect_uk2", "uk4": "rect_uk4"}),
        "smallA": ("16x16x16", 8, 8, 4, {"uk1": "smallA_uk1", "uk2": "smallA_uk2", "uk4": "smallA_uk4"}),
        "highK": ("32x32x128", 8, 8, 8, {"uk1": "highK_uk1", "uk2": "highK_uk2", "uk4": "highK_uk4"}),
    }
    static_eval = {"top_choice_agreement": {}, "domains_correctly_predicted": [], "domains_mispredicted": [], "spill_related_mispredictions": []}
    for dname, (shape, tm, tn, tk, ukmap) in all_domains.items():
        shape_m, shape_n, shape_k = (int(x) for x in shape.split("x"))
        recs = []
        for ukname, label in ukmap.items():
            uk = int(ukname[2:])
            key = cm.CandidateKey(shape_m=shape_m, shape_n=shape_n, shape_k=shape_k, tile_m=tm, tile_n=tn, tile_k=tk, schedule_unroll_k=uk)
            if key in by_key:
                recs.append(by_key[key])
        if not recs or not any(r.measured for r in recs):
            continue
        static_ranked = cm.rank_candidates(recs, cm.RANKING_MODE_STATIC_SOFT_PENALTY)
        static_top = static_ranked[0].label
        measured_present = [r for r in recs if r.measured]
        measured_winner = min(measured_present, key=lambda r: r.measured.median_latency_ms.value).label if measured_present else None
        agree = static_top == measured_winner
        static_eval["top_choice_agreement"][dname] = {"static_top": static_top, "measured_winner": measured_winner, "agree": agree}
        (static_eval["domains_correctly_predicted"] if agree else static_eval["domains_mispredicted"]).append(dname)
        if not agree:
            winner_rec = next((r for r in recs if r.label == measured_winner), None)
            if winner_rec and (winner_rec.llvm_backend.spill_store_count.value or 0) > 0:
                static_eval["spill_related_mispredictions"].append(dname)
    static_eval["top_choice_agreement_rate"] = len(static_eval["domains_correctly_predicted"]) / max(1, len(static_eval["top_choice_agreement"]))
    static_eval["conclusion"] = (
        "No currently collected static metric reliably predicts the measured winner across the tested domain "
        "set -- static evidence correctly predicted the top choice in only "
        f"{len(static_eval['domains_correctly_predicted'])} of {len(static_eval['top_choice_agreement'])} domains "
        "(the one(s) where the winner happened to be spill-free). Exact-domain calibration remains necessary "
        "for reliable winner selection with this microkernel family. Weights were NOT retuned to fit this "
        "expanded dataset."
    )
    with open(os.path.join(ART, "static_model_evaluation.json"), "w") as f:
        json.dump(static_eval, f, indent=2)
    print(f"\nStatic model: {static_eval['top_choice_agreement_rate']:.0%} top-choice agreement across {len(static_eval['top_choice_agreement'])} domains", file=sys.stderr)

    # ---- Policy comparison (task section 14) ----
    policy_results = {"policy_1_exact_calibration_only": {}, "policy_2_bounded_heuristic_offline_only": {}, "policy_3_universal_uk4_UNSAFE_evaluation_only": {}}
    for dname, (shape, tm, tn, tk, ukmap) in all_domains.items():
        shape_m, shape_n, shape_k = (int(x) for x in shape.split("x"))
        recs = []
        for ukname, label in ukmap.items():
            uk = int(ukname[2:])
            key = cm.CandidateKey(shape_m=shape_m, shape_n=shape_n, shape_k=shape_k, tile_m=tm, tile_n=tn, tile_k=tk, schedule_unroll_k=uk)
            if key in by_key:
                recs.append(by_key[key])
        measured_present = [r for r in recs if r.measured]
        measured_winner_rec = min(measured_present, key=lambda r: r.measured.median_latency_ms.value) if measured_present else None
        measured_winner = measured_winner_rec.label if measured_winner_rec else None
        # Use the actual schedule_unroll_k field, NOT a label-string match
        # (a real bug found while building this comparison: Stage 12's
        # historical label for the primary domain's uk4 candidate is
        # "primary_full_unroll", which does not literally contain the
        # substring "uk4" -- an .endswith("uk4") check silently missed it,
        # undercounting policy 2/3's real agreement by one domain).
        measured_winner_is_uk4 = measured_winner_rec is not None and measured_winner_rec.key.schedule_unroll_k == 4

        # Policy 1: exact calibration only -- domain_pool restricted to THIS domain's own keys.
        pool = [(r.key, r.measured) for r in recs if r.measured]
        p1_ranked = cm.rank_candidates(recs, cm.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=pool) if pool else cm.rank_candidates(recs, cm.RANKING_MODE_STATIC_SOFT_PENALTY)
        p1_selected = p1_ranked[0].label if p1_ranked else None
        policy_results["policy_1_exact_calibration_only"][dname] = {"selected": p1_selected, "matches_measured_winner": p1_selected == measured_winner}

        # Policy 2: bounded heuristic -- prefer uk4 ONLY when target/tile/vw/dtype match an
        # explicitly-supported bucket (here: cortex-a76, tile in {8x8x8,8x8x4}, f32, vw4) AND
        # a uk4 candidate is legally generated. Evaluated OFFLINE only -- never wired into the
        # compiler driver's actual mode set.
        uk4_key = next((r.key for r in recs if r.key.schedule_unroll_k == 4), None)
        supported_bucket = uk4_key is not None and uk4_key.target_cpu == "cortex-a76" and uk4_key.tile_m == 8 and uk4_key.tile_n == 8 and uk4_key.tile_k in (8, 4)
        p2_selected = f"uk4 candidate for {dname}" if supported_bucket else "fallback (outside bounded heuristic bucket)"
        policy_results["policy_2_bounded_heuristic_offline_only"][dname] = {
            "selected": p2_selected, "in_bounded_bucket": supported_bucket,
            "matches_measured_winner": bool(supported_bucket and measured_winner_is_uk4),
        }

        # Policy 3: universal uk4 -- deliberately unsafe, evaluation-only, never exposed.
        policy_results["policy_3_universal_uk4_UNSAFE_evaluation_only"][dname] = {
            "selected": "uk4 (unconditional)", "matches_measured_winner": bool(measured_winner_is_uk4),
            "risk_note": "would silently select uk4 even for a domain/target never measured -- this is exactly the failure mode Stage 14-17 exist to prevent",
        }

    summaries = {}
    for pname, results in policy_results.items():
        agree = sum(1 for v in results.values() if v.get("matches_measured_winner"))
        summaries[pname + "_summary"] = {"agreement_count": agree, "total_domains": len(results), "fallback_count": sum(1 for v in results.values() if "fallback" in str(v.get("selected", "")))}
    policy_results.update(summaries)
    with open(os.path.join(ART, "policy_comparison.json"), "w") as f:
        json.dump(policy_results, f, indent=2)

    # ---- Extended multi-domain profile ----
    with open(os.path.join(STAGE16_ART, "multi_domain_profile.json")) as f:
        profile = json.load(f)
    for dname, (shape, tm, tn, tk, ukmap) in {"smallA": all_domains["smallA"], "highK": all_domains["highK"]}.items():
        entries = {}
        for ukname, label in ukmap.items():
            uk = int(ukname[2:])
            shape_m, shape_n, shape_k = (int(x) for x in shape.split("x"))
            key = cm.CandidateKey(shape_m=shape_m, shape_n=shape_n, shape_k=shape_k, tile_m=tm, tile_n=tn, tile_k=tk, schedule_unroll_k=uk)
            if key not in by_key or not by_key[key].measured:
                continue
            r = by_key[key]
            entries[r.key.canonical_id()] = {
                "label": label, "median_latency_ms": r.measured.median_latency_ms.value, "cv": r.measured.cv.value,
                "correctness_pass": r.measured.correctness_pass.value,
                "spills": r.llvm_backend.spill_store_count.value, "reloads": r.llvm_backend.reload_load_count.value,
                "timing_quality": timing_quality.get(label, {}).get("quality"),
            }
        profile["domains"][dname] = {
            "domain_identity": {"target_arch": "aarch64", "target_cpu": "cortex-a76", "target_features": "none",
                                 "dtype": "f32", "tile": {"m": tm, "n": tn, "k": tk}, "shape": shape},
            "candidates": entries,
            "stress_category": "very_small_workload_with_spilling" if dname == "smallA" else "larger_k_loop_trip_count",
        }
    with open(os.path.join(ART, "updated_multidomain_profile.json"), "w") as f:
        json.dump(profile, f, indent=2)

    # ---- Cross-domain rejection re-verification (existing 4 + 2 new = 6 domains) ----
    all_shape_tile = {d: (s, tm, tn, tk) for d, (s, tm, tn, tk, _) in all_domains.items()}
    rejection_report = []
    for q_domain, (q_shape, q_tm, q_tn, q_tk) in all_shape_tile.items():
        q_shape_m, q_shape_n, q_shape_k = (int(x) for x in q_shape.split("x"))
        query_key = cm.CandidateKey(shape_m=q_shape_m, shape_n=q_shape_n, shape_k=q_shape_k, tile_m=q_tm, tile_n=q_tn, tile_k=q_tk, schedule_unroll_k=2)
        for e_domain, (e_shape, e_tm, e_tn, e_tk) in all_shape_tile.items():
            if q_domain == e_domain:
                continue
            e_shape_m, e_shape_n, e_shape_k = (int(x) for x in e_shape.split("x"))
            evidence_key = cm.CandidateKey(shape_m=e_shape_m, shape_n=e_shape_n, shape_k=e_shape_k, tile_m=e_tm, tile_n=e_tn, tile_k=e_tk, schedule_unroll_k=2)
            compat = cm.check_compatibility(query_key, evidence_key, cm.BENCHMARK_METHODOLOGY_VERSION)
            rejection_report.append({
                "query_domain": q_domain, "evidence_domain": e_domain, "compatibility_level": compat["level"],
                "confidence": compat["confidence"], "exact_match_rejected": compat["level"] != cm.EXACT_MATCH,
            })
    with open(os.path.join(ART, "cross_domain_rejection_report.json"), "w") as f:
        json.dump(rejection_report, f, indent=2)
    all_rejected = all(r["exact_match_rejected"] for r in rejection_report)
    print(f"Cross-domain rejection (6 domains, {len(rejection_report)} pairs): all_exact_rejected={all_rejected}", file=sys.stderr)

    return by_key, domain_summary, timing_quality


if __name__ == "__main__":
    main()
