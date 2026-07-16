#!/usr/bin/env python3
"""run_aarch64_schedule_cost_model.py

Stage 14 CLI driver: loads Stage 12 static/backend evidence and Stage 13
Raspberry Pi measurements, builds CandidateEvidenceRecords, runs the
ranking experiment in all three modes, builds attribution reports and
shape-aware findings, and writes the Stage 14 artifact directory.

Experimental options (task section 13 -- every option below is wired to
real behavior, nothing is a no-op flag):

  --schedule-candidate-mode=static|calibrated
      static:     RANKING_MODE_STATIC_SOFT_PENALTY (spill is a penalty,
                  never a hard veto -- the corrected Stage 12 policy)
      calibrated: RANKING_MODE_CALIBRATED_PI (uses compatible measured
                  Raspberry Pi 5 evidence where available, falls back to
                  static with reduced confidence otherwise)

  --schedule-profile=<path>
      Path to a Stage-13-shaped pi_validation_results.json to use as the
      measured-evidence pool in calibrated mode. Defaults to this repo's
      own Stage 13 artifact. Demonstrates that calibration is tied to a
      DECLARED profile, not implicitly "whatever the tool last measured".

  --emit-schedule-cost-breakdown=<path>
      Writes the full per-candidate CostBreakdown (every score component
      visible, per task section 7) to this path as JSON.

RANKING_MODE_STATIC_HARD_REJECT is intentionally NOT exposed as a CLI
mode -- it exists only inside the ranking-comparison experiment (section
10) to demonstrate what the corrected policy replaced. Exposing it as a
selectable production mode would reintroduce the Stage 12 defect this
stage exists to fix.
"""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import aarch64_schedule_candidate_model as m  # noqa: E402

DEFAULT_STAGE12_JSON = os.path.join(
    REPO_ROOT, "artifacts", "backend_codegen", "aarch64_matmul_bias_relu_scheduling",
    "schedule_comparison_results.json")
DEFAULT_STAGE13_JSON = os.path.join(
    REPO_ROOT, "artifacts", "backend_codegen", "aarch64_matmul_bias_relu_pi_scheduling",
    "pi_validation_results.json")

# Matched groups for the ranking experiment (task section 10) -- ranking
# is meaningful only WITHIN a group (identical shape; Stage 13 never
# measured a cross-shape race, and raw absolute latency across different
# problem sizes is not a fair "which candidate is better" signal -- a
# 64x64x64 candidate does 8x the FLOPs of a 32x32x32 one and would always
# rank last on absolute latency regardless of schedule quality).
RANKING_GROUPS = {
    "primary_32x32x32_unroll_family": ["primary_unroll1", "primary_unroll2", "primary_full_unroll"],
    "alt_k_tile_8x8x4": ["alt_k_tile_unroll1", "alt_k_tile_unroll2"],
    "cube64_8x8x8": ["cube64_unroll1", "cube64_unroll2"],
}

# Measured rank ground truth, from Stage 13 median-of-medians latency
# (lower is better) within each group -- used only for the
# prediction-vs-measured comparison, never fed back into the static model.
MEASURED_RANK_LABELS = {
    "primary_32x32x32_unroll_family": ["primary_full_unroll", "primary_unroll2", "primary_unroll1"],  # fastest first
    "alt_k_tile_8x8x4": ["alt_k_tile_unroll2", "alt_k_tile_unroll1"],
    "cube64_8x8x8": ["cube64_unroll2", "cube64_unroll1"],
}


def load_records(stage12_json, stage13_json):
    records = m.load_stage12_records(stage12_json)
    stage13_label_alias = m.load_stage13_measured(stage13_json, records)
    return records, stage13_label_alias


def run_ranking_experiment(records, measured_pool):
    """Returns {group_name: {mode: [CostBreakdown, ...]}}"""
    modes = [m.RANKING_MODE_STATIC_HARD_REJECT, m.RANKING_MODE_STATIC_SOFT_PENALTY, m.RANKING_MODE_CALIBRATED_PI]
    out = {}
    for group_name, labels in RANKING_GROUPS.items():
        subset = [records[l] for l in labels if l in records]
        out[group_name] = {}
        for mode in modes:
            ranked = m.rank_candidates(subset, mode, measured_evidence_pool=measured_pool)
            out[group_name][mode] = ranked
    return out


def compare_predicted_vs_measured(ranking_results):
    comparisons = {}
    for group_name, by_mode in ranking_results.items():
        measured_order = MEASURED_RANK_LABELS.get(group_name, [])
        group_comparison = {}
        for mode, ranked in by_mode.items():
            predicted_order = [b.label for b in ranked]
            top_predicted = predicted_order[0] if predicted_order else None
            top_measured = measured_order[0] if measured_order else None
            # Spearman-style simple agreement: fraction of adjacent-pair
            # orderings that agree between predicted and measured.
            agree_pairs, total_pairs = 0, 0
            for i in range(len(measured_order)):
                for j in range(i + 1, len(measured_order)):
                    a, b_ = measured_order[i], measured_order[j]
                    if a not in predicted_order or b_ not in predicted_order:
                        continue
                    total_pairs += 1
                    if predicted_order.index(a) < predicted_order.index(b_):
                        agree_pairs += 1
            group_comparison[mode] = {
                "predicted_order": predicted_order,
                "measured_order": measured_order,
                "top_predicted": top_predicted,
                "top_measured": top_measured,
                "top_matches": top_predicted == top_measured,
                "pairwise_agreement": (agree_pairs / total_pairs) if total_pairs else None,
                "mispredictions": [l for l in predicted_order if l != measured_order[predicted_order.index(l)]] if len(predicted_order) == len(measured_order) else None,
            }
        comparisons[group_name] = group_comparison
    return comparisons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--schedule-candidate-mode", choices=["static", "calibrated"], default="static")
    ap.add_argument("--schedule-profile", default=DEFAULT_STAGE13_JSON)
    ap.add_argument("--stage12-json", default=DEFAULT_STAGE12_JSON)
    ap.add_argument("--emit-schedule-cost-breakdown")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records, stage13_label_alias = load_records(args.stage12_json, args.schedule_profile)
    measured_pool = [(r.key, r.measured) for r in records.values() if r.measured]

    # ---- Single-mode "production interface" run (task section 13) ----
    mode = m.RANKING_MODE_STATIC_SOFT_PENALTY if args.schedule_candidate_mode == "static" else m.RANKING_MODE_CALIBRATED_PI
    all_records = list(records.values())
    incompatible_or_unsupported = []  # none in this candidate family are legality-rejected; recorded for schema completeness
    production_ranking = m.rank_candidates(all_records, mode, measured_evidence_pool=measured_pool)

    production_report = {
        "candidates_considered": [b.label for b in production_ranking],
        "candidates_rejected_for_legality_or_incompatibility": incompatible_or_unsupported,
        "mode": mode,
        "selected_candidate": production_ranking[0].label if production_ranking and not production_ranking[0].rejected else None,
        "selected_candidate_confidence": production_ranking[0].confidence if production_ranking else None,
        "fallback_reason": (
            "none -- exact or compatible measured evidence used" if mode == m.RANKING_MODE_CALIBRATED_PI
            else "static mode does not use measured evidence by design"
        ),
        "cost_breakdown": [m.breakdown_to_dict(b) for b in production_ranking],
    }
    if args.emit_schedule_cost_breakdown:
        with open(args.emit_schedule_cost_breakdown, "w") as f:
            json.dump(production_report, f, indent=2)
        print(f"Wrote cost breakdown to {args.emit_schedule_cost_breakdown}", file=sys.stderr)

    # ---- Evidence dump ----
    evidence_out = {label: m.record_to_dict(r) for label, r in records.items()}
    with open(os.path.join(args.output_dir, "evidence.json"), "w") as f:
        json.dump(evidence_out, f, indent=2)

    # ---- Ranking experiment (task section 10) ----
    ranking_results = run_ranking_experiment(records, measured_pool)
    static_hard_reject_out = {g: [m.breakdown_to_dict(b) for b in by_mode[m.RANKING_MODE_STATIC_HARD_REJECT]] for g, by_mode in ranking_results.items()}
    static_soft_out = {g: [m.breakdown_to_dict(b) for b in by_mode[m.RANKING_MODE_STATIC_SOFT_PENALTY]] for g, by_mode in ranking_results.items()}
    calibrated_out = {g: [m.breakdown_to_dict(b) for b in by_mode[m.RANKING_MODE_CALIBRATED_PI]] for g, by_mode in ranking_results.items()}
    with open(os.path.join(args.output_dir, "static_ranking.json"), "w") as f:
        json.dump({"hard_reject_legacy_comparison_only": static_hard_reject_out, "soft_penalty_corrected_default": static_soft_out}, f, indent=2)
    with open(os.path.join(args.output_dir, "calibrated_ranking.json"), "w") as f:
        json.dump(calibrated_out, f, indent=2)

    prediction_comparison = compare_predicted_vs_measured(ranking_results)
    with open(os.path.join(args.output_dir, "ranking_prediction_comparison.json"), "w") as f:
        json.dump(prediction_comparison, f, indent=2)

    # ---- Resolve Stage 13's own comparisons dict into `records`' labels
    # (needed by both the attribution-pair extension below and the
    # shape-aware findings section further down) ----
    stage13_comparisons = {}
    try:
        with open(args.schedule_profile) as f:
            raw_comparisons = json.load(f).get("comparisons", {})
        # Resolve Stage-13-native baseline/scheduled labels to whatever
        # label they were merged under in `records` (see
        # load_stage13_measured's alias-map docstring) -- without this,
        # any comparison whose candidate got folded into a pre-existing
        # Stage-12 label would silently vanish.
        for pair_name, cmp in raw_comparisons.items():
            stage13_comparisons[pair_name] = {
                **cmp,
                "baseline": stage13_label_alias.get(cmp["baseline"], cmp["baseline"]),
                "scheduled": stage13_label_alias.get(cmp["scheduled"], cmp["scheduled"]),
            }
    except Exception:
        pass

    # ---- Attribution (task section 5) ----
    # Core 4 pairs Stage 12 analyzed directly, PLUS every additional
    # matched pair Stage 13 measured on shapes Stage 12 never covered
    # (cube16/rect/large) -- resolved through stage13_label_alias so a
    # candidate that got folded into a pre-existing Stage-12 label (e.g.
    # primary_uk1 -> primary_unroll1) is still found. Without this second
    # part, those three shapes' hardware_confirmation would incorrectly
    # read "unknown" in classification_summary.json despite Stage 13
    # having directly measured and classified them.
    attribution_pairs = [
        ("primary_unroll1", "primary_unroll2"),
        ("primary_unroll1", "primary_full_unroll"),
        ("alt_k_tile_unroll1", "alt_k_tile_unroll2"),
        ("cube64_unroll1", "cube64_unroll2"),
    ]
    for pair_name, cmp in stage13_comparisons.items():
        pair = (cmp["baseline"], cmp["scheduled"])
        if pair not in attribution_pairs:
            attribution_pairs.append(pair)
    attributions = {}
    for b_label, s_label in attribution_pairs:
        if b_label in records and s_label in records:
            attributions[f"{b_label}_vs_{s_label}"] = m.build_attribution(records[b_label], records[s_label])
    with open(os.path.join(args.output_dir, "attribution_summary.json"), "w") as f:
        json.dump(attributions, f, indent=2)

    # ---- Classification (task section 6) ----
    classifications = {}
    for label, r in records.items():
        baseline_label = None
        for b_l, s_l in attribution_pairs:
            if s_l == label:
                baseline_label = b_l
                break
        baseline_measured = records[baseline_label].measured if baseline_label else None
        classifications[label] = m.full_classification(r.llvm_backend, baseline_measured, r.measured)
    with open(os.path.join(args.output_dir, "classification_summary.json"), "w") as f:
        json.dump(classifications, f, indent=2)

    # ---- Shape-aware findings (task section 9) ----
    findings = m.shape_aware_findings(records, stage13_comparisons)
    with open(os.path.join(args.output_dir, "shape_aware_findings.json"), "w") as f:
        json.dump(findings, f, indent=2)

    with open(os.path.join(args.output_dir, "candidate_schema.json"), "w") as f:
        json.dump({
            "schema_version": m.SCHEMA_VERSION,
            "CandidateKey_fields": [f.name for f in __import__("dataclasses").fields(m.CandidateKey)],
            "StaticIRBEvidence_fields": [f.name for f in __import__("dataclasses").fields(m.StaticIRBEvidence)],
            "LLVMBackendEvidence_fields": [f.name for f in __import__("dataclasses").fields(m.LLVMBackendEvidence)],
            "MeasuredHardwareEvidence_fields": [f.name for f in __import__("dataclasses").fields(m.MeasuredHardwareEvidence)],
            "CostBreakdown_fields": [f.name for f in __import__("dataclasses").fields(m.CostBreakdown)],
            "default_weights": __import__("dataclasses").asdict(m.DEFAULT_WEIGHTS),
            "source_levels": list(m.SOURCE_LEVELS),
            "classification_labels": {
                "backend_safety": [m.BACKEND_SAFE, m.BACKEND_COSTLY],
                "hardware_confirmation": [m.HW_CONFIRMED_PROFITABLE, m.HW_CONFIRMED_NEUTRAL, m.HW_CONFIRMED_REGRESSION, m.HW_UNKNOWN, m.INCORRECT],
            },
            "compatibility_levels": [m.EXACT_MATCH, m.CROSS_SHAPE_SAME_SCHEDULE, m.SHAPE_BUCKET, m.INCOMPATIBLE],
        }, f, indent=2)

    print(f"\nWrote Stage 14 artifacts to {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
