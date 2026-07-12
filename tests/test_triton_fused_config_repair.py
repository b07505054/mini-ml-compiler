#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_triton_fused_config_repair as repair  # noqa: E402
import run_triton_fused_config_selection as base  # noqa: E402


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def measurement(wid: str, m: int, n: int, k: int, lat: dict[str, float]) -> base.Measurement:
    return base.Measurement(wid, {"m": m, "n": n, "k": k, "dtype": "f32"}, "fixture", "fixture", None, lat)


def main() -> int:
    shape = {"m": 64, "n": 64, "k": 4096, "dtype": "f32"}
    wf = repair.workload_features(shape)
    require(wf["k_dominance_ratio"] == 1.0, "K-dominance feature mismatch")
    require(wf["k_dominance_ratio"] == wf["reduction_dominance"], "reduction dominance alias mismatch")
    require(wf["output_area_to_k"] == 1.0, "output-area-to-K feature mismatch")
    require(wf["small_square_high_k"], "small-square high-K feature missing")
    try:
        repair.workload_features({**shape, "oracle_winner": "bm16"})
        raise AssertionError("label leakage must reject")
    except ValueError:
        pass

    tf16 = repair.tile_features(wf, repair.PRIMARY_CONFIGS["bm16_bn16_bk32_w4_s3"], 16)
    tf64 = repair.tile_features(wf, repair.PRIMARY_CONFIGS["bm64_bn64_bk32_w4_s3"], 16)
    require(tf16["programs_per_sm"] > tf64["programs_per_sm"], "programs-per-SM should distinguish tile size")
    require(tf16["programs_per_sm"] == tf16["work_items_per_compute_unit"], "programs-per-SM alias mismatch")
    require(tf64["work_per_program"] > tf16["work_per_program"], "work-per-program should increase with tile area")
    require(tf64["work_per_program"] == tf64["compute_work_per_item"], "work-per-item alias mismatch")
    require(tf64["output_program_waves"] == tf64["execution_waves"], "execution wave alias mismatch")
    require(tf64["padding_amplification"] == tf64["work_amplification"], "work amplification alias mismatch")
    require(tf64["k_iterations_per_output_program"] == 128, "K tile count mismatch")

    _, _, u16 = repair.component_units(shape, repair.PRIMARY_CONFIGS["bm16_bn16_bk32_w4_s3"], 16)
    _, _, u64 = repair.component_units(shape, repair.PRIMARY_CONFIGS["bm64_bn64_bk32_w4_s3"], 16)
    require(u64["k_dominant_parallelism_unit"] > u16["k_dominant_parallelism_unit"], "large-tile K-dominant penalty should be larger")

    many_programs = {"m": 4096, "n": 4096, "k": 32, "dtype": "f32"}
    _, _, many16 = repair.component_units(many_programs, repair.PRIMARY_CONFIGS["bm16_bn16_bk32_w4_s3"], 16)
    _, _, many64 = repair.component_units(many_programs, repair.PRIMARY_CONFIGS["bm64_bn64_bk32_w4_s3"], 16)
    require(many16["program_overhead_unit"] > many64["program_overhead_unit"], "tiny-tile program overhead should be larger")

    train = [
        measurement("repair_train_sq32_k4096", 32, 32, 4096, {
            "bm16_bn16_bk32_w4_s3": 0.10, "bm32_bn32_bk32_w4_s3": 0.14, "bm64_bn64_bk32_w4_s3": 0.25, "bm16_bn64_bk32_w4_s3": 0.18,
        }),
        measurement("repair_train_large", 256, 256, 2048, {
            "bm16_bn16_bk32_w4_s3": 0.60, "bm32_bn32_bk32_w4_s3": 0.30, "bm64_bn64_bk32_w4_s3": 0.20, "bm16_bn64_bk32_w4_s3": 0.32,
        }),
        measurement("repair_train_skinny", 1, 4096, 65536, {
            "bm16_bn16_bk32_w4_s3": 12.0, "bm32_bn32_bk32_w4_s3": 12.5, "bm64_bn64_bk32_w4_s3": 15.0, "bm16_bn64_bk32_w4_s3": 10.0,
        }),
    ]
    model = repair.calibrate(train, 16)
    stats = repair.feature_stats(train)
    rank = repair.ranking(shape, model, 16, calibrated=True)
    require(len(rank) == 4, "ranking must include all configs")
    require("k_dominant_parallelism_ms" in rank[0]["components"], "repair component missing")
    dec = repair.choose_policy("repaired_confidence", shape, model, train, stats, 16)
    require(dec["selected_config_id"] in repair.PRIMARY_CONFIG_IDS, "repair policy must stay fused")

    tr, held, split = repair.grouped_split(
        train + [measurement("unfriendly_m64_n64_k4096", 64, 64, 4096, train[0].latencies)],
        "leave-one-small-square-family-out",
    )
    require(held and all("unfriendly_m64" in r.workload_id for r in held), "small-square family split failed")

    eval_rows = [
        {"selected_config_id": "bm16_bn16_bk32_w4_s3", "oracle_config": "bm16_bn16_bk32_w4_s3", "top1_correct": True, "regret": 0.0},
        {"selected_config_id": "bm64_bn64_bk32_w4_s3", "oracle_config": "bm16_bn16_bk32_w4_s3", "top1_correct": False, "regret": 1.0},
    ]
    metrics = repair.evaluate(eval_rows)
    require(metrics["per_config_recall"]["bm16_bn16_bk32_w4_s3"] == 0.5, "16x16 recall mismatch")

    plan = repair.build_plan(train[0], {"selected_config_id": "bm16_bn16_bk32_w4_s3", "selection_source": "fixture", "candidate_ranking": []}, "abc")
    validation = base.validate_plans([plan])
    require(validation["aggregate"]["planned_config_equals_actual_rate"] == 1.0, "plan transport mismatch")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sweep = tmp_path / "sweep.json"
        payload = {
            "schema": "triton_matmul_bias_relu_fused_candidate_sweep",
            "environment": {
                "gpu_model": "NVIDIA GeForce GTX 1650 with Max-Q Design",
                "compute_capability": [7, 5],
                "sm_count": 16,
            },
            "workloads": [
                {"workload_id": r.workload_id, "category": r.category, "shape": r.shape, "classification": "stable_candidate_win", "cross_session_median_ms_by_config": r.latencies}
                for r in train + [measurement("unfriendly_m64_n64_k4096", 64, 64, 4096, train[0].latencies)]
            ],
        }
        sweep.write_text(json.dumps(payload), encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                str(TOOLS / "run_triton_fused_config_repair.py"),
                "--base-sweep", str(sweep),
                "--repair-training-profile", str(tmp_path / "missing.json"),
                "--fresh-oracle", str(tmp_path / "missing_oracle.json"),
                "--cost-model-output", str(tmp_path / "model.json"),
                "--plans-output", str(tmp_path / "plans.json"),
                "--plan-validation-output", str(tmp_path / "validation.json"),
                "--summary-output", str(tmp_path / "summary.json"),
                "--report-output", str(tmp_path / "report.md"),
                "--doc-output", str(tmp_path / "doc.md"),
                "--split", "leave-one-small-square-family-out",
                "--target-profile", str(ROOT / "configs/target_profiles/nvidia_gtx1650_maxq.json"),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        model = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
        require("repaired_calibrated" in summary["policy_aggregates"], "repair summary missing policy")
        require(model["hardware_profile"]["effective_compute_units_source"] == "target_profile", "repair hardware source mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
