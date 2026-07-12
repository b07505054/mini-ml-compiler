#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNNER = TOOLS / "run_triton_fused_config_selection.py"
sys.path.insert(0, str(TOOLS))

import run_triton_fused_config_selection as sel  # noqa: E402


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def measurement(wid: str, m: int, n: int, k: int, region: str, latencies: dict[str, float]) -> sel.Measurement:
    return sel.Measurement(wid, {"m": m, "n": n, "k": k, "dtype": "f32"}, "fixture", region, None, latencies)


def fixture_sweep(path: Path) -> None:
    workloads = []
    rows = [
        measurement("train_skinny", 1, 4096, 8192, "small_skinny", {
            "bm16_bn16_bk32_w4_s3": 3.0,
            "bm32_bn32_bk32_w4_s3": 2.8,
            "bm64_bn64_bk32_w4_s3": 3.5,
            "bm16_bn64_bk32_w4_s3": 2.0,
        }),
        measurement("train_large", 256, 1024, 512, "large_regular", {
            "bm16_bn16_bk32_w4_s3": 2.5,
            "bm32_bn32_bk32_w4_s3": 1.5,
            "bm64_bn64_bk32_w4_s3": 1.0,
            "bm16_bn64_bk32_w4_s3": 1.7,
        }),
        measurement("train_square", 64, 64, 4096, "small_square_high_k", {
            "bm16_bn16_bk32_w4_s3": 1.0,
            "bm32_bn32_bk32_w4_s3": 1.2,
            "bm64_bn64_bk32_w4_s3": 1.6,
            "bm16_bn64_bk32_w4_s3": 1.4,
        }),
        measurement("holdout_m1024_n1024_k24", 1024, 1024, 24, "large_regular", {
            "bm16_bn16_bk32_w4_s3": 2.0,
            "bm32_bn32_bk32_w4_s3": 1.3,
            "bm64_bn64_bk32_w4_s3": 1.0,
            "bm16_bn64_bk32_w4_s3": 1.2,
        }),
    ]
    for row in rows:
        workloads.append({
            "workload_id": row.workload_id,
            "category": row.category,
            "shape": row.shape,
            "classification": "stable_candidate_win",
            "cross_session_median_ms_by_config": row.latencies,
        })
    payload = {
        "schema": "triton_matmul_bias_relu_fused_candidate_sweep",
        "environment": {
            "gpu_model": "NVIDIA GeForce GTX 1650 with Max-Q Design",
            "compute_capability": [7, 5],
            "sm_count": 16,
        },
        "selection_candidate_set": [sel.PRIMARY_CONFIGS[c].typed() for c in sel.PRIMARY_CONFIG_IDS],
        "workloads": workloads,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> int:
    require(len(sel.PRIMARY_CONFIGS) == 4, "exactly four primary configs required")
    for cid, cfg in sel.PRIMARY_CONFIGS.items():
        cfg.validate()
        typed = cfg.typed()
        require(typed["semantic_kernel_id"] == sel.KERNEL_FAMILY_ID, "semantic kernel mismatch")
        require(typed["full_size_intermediates"] == 0, "configs must remain one-pass")
        require(typed["fusion"] == "one_pass_epilogue", "fusion metadata mismatch")
    try:
        sel.TritonFusedConfig("unknown", 16, 16, 32, 4, 3).validate()
        raise AssertionError("unknown config must reject")
    except ValueError:
        pass

    shape = {"m": 1, "n": 4096, "k": 65536, "dtype": "f32"}
    feats = sel.workload_features(shape)
    require(feats["small_m"] and feats["skinny_m"] and feats["extreme_k"], "workload features missing")
    try:
        sel.workload_features({**shape, "oracle_config": "x"})
        raise AssertionError("label leakage must reject")
    except ValueError:
        pass
    tf = sel.tile_features(feats, sel.PRIMARY_CONFIGS["bm16_bn64_bk32_w4_s3"], sm_count=16)
    require(tf["m_tile_count"] == 1, "M tile count mismatch")
    require(tf["n_tile_count"] == 64, "N tile count mismatch")
    require(tf["k_tile_count"] == 2048, "K tile count mismatch")
    require(0 < tf["m_tile_utilization"] <= 1, "edge utilization invalid")
    require(tf["padding_amplification"] >= 1.0, "padding amplification invalid")
    require(tf["programs_per_sm"] == tf["work_items_per_compute_unit"], "programs-per-SM alias mismatch")
    require(tf["output_program_waves"] == tf["execution_waves"], "execution wave alias mismatch")
    require(tf["total_output_program_count"] == tf["parallel_work_items"], "parallel work alias mismatch")
    require(tf["work_per_program"] == tf["compute_work_per_item"], "work per item alias mismatch")
    require(tf["masked_output_fraction"] == tf["padding_waste_ratio"], "padding waste alias mismatch")
    require(tf["padding_amplification"] == tf["work_amplification"], "work amplification alias mismatch")

    train = [
        measurement("train_skinny", 1, 4096, 8192, "small_skinny", {
            "bm16_bn16_bk32_w4_s3": 3.0,
            "bm32_bn32_bk32_w4_s3": 2.8,
            "bm64_bn64_bk32_w4_s3": 3.5,
            "bm16_bn64_bk32_w4_s3": 2.0,
        }),
        measurement("train_large", 256, 1024, 512, "large_regular", {
            "bm16_bn16_bk32_w4_s3": 2.5,
            "bm32_bn32_bk32_w4_s3": 1.5,
            "bm64_bn64_bk32_w4_s3": 1.0,
            "bm16_bn64_bk32_w4_s3": 1.7,
        }),
        measurement("train_square", 64, 64, 4096, "small_square_high_k", {
            "bm16_bn16_bk32_w4_s3": 1.0,
            "bm32_bn32_bk32_w4_s3": 1.2,
            "bm64_bn64_bk32_w4_s3": 1.6,
            "bm16_bn64_bk32_w4_s3": 1.4,
        }),
    ]
    model = sel.calibrate(train, 16)
    stats = sel.feature_stats(train)
    rank = sel.ranking(shape, model, 16, calibrated=True)
    require(len(rank) == 4, "ranking must contain all four configs")
    require(all("components" in r and "tile_features" in r for r in rank), "ranking details missing")
    nearest = sel.choose_policy("nearest_shape", shape, model, train, stats, 16)
    require(nearest["selection_source"] == "nearest_measured_shape", "nearest policy missing")
    conf = sel.choose_policy("confidence_aware", shape, model, train, stats, 16)
    require(conf["selected_config_id"] in sel.PRIMARY_CONFIG_IDS, "confidence policy must stay fused")

    eval_rows = [
        {"selected_config_id": "bm16_bn64_bk32_w4_s3", "oracle_config": "bm16_bn64_bk32_w4_s3", "top1_correct": True, "regret": 0.0},
        {"selected_config_id": "bm64_bn64_bk32_w4_s3", "oracle_config": "bm16_bn64_bk32_w4_s3", "top1_correct": False, "regret": 0.2},
    ]
    metrics = sel.evaluate(eval_rows)
    require(metrics["top1_accuracy"] == 0.5, "top1 metric mismatch")
    require(metrics["macro_accuracy"] == 0.5, "macro accuracy mismatch")
    require(metrics["per_config_recall"]["bm16_bn64_bk32_w4_s3"] == 0.5, "per-config recall mismatch")

    plan = sel.build_plan(train[0], {"selected_config_id": "bm16_bn64_bk32_w4_s3", "selection_source": "fixture", "candidate_ranking": []}, "abc")
    op = plan["operations"][0]
    require(op["semantic_kernel_id"] == sel.KERNEL_FAMILY_ID, "plan semantic kernel missing")
    require(op["selected_config_id"] == "bm16_bn64_bk32_w4_s3", "plan config ID missing")
    validation = sel.validate_plans([plan])
    require(validation["aggregate"]["planned_config_equals_actual_rate"] == 1.0, "plan validation mismatch")

    rows = train + [measurement("rep_m128_k768_n3072", 128, 3072, 768, "large_regular", train[1].latencies)]
    tr, held, split = sel.grouped_split(rows, "leave-one-shape-region-out")
    require(tr and held and split["held_out_workload_ids"], "grouped split failed")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sweep = tmp_path / "sweep.json"
        fixture_sweep(sweep)
        completed = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--candidate-sweep",
                str(sweep),
                "--cost-model-output",
                str(tmp_path / "model.json"),
                "--plans-output",
                str(tmp_path / "plans.json"),
                "--plan-validation-output",
                str(tmp_path / "validation.json"),
                "--fresh-oracle-output",
                str(tmp_path / "oracle.json"),
                "--summary-output",
                str(tmp_path / "summary.json"),
                "--report-output",
                str(tmp_path / "report.md"),
                "--doc-output",
                str(tmp_path / "doc.md"),
                "--target-profile",
                str(ROOT / "configs/target_profiles/nvidia_gtx1650_maxq.json"),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        model = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
        require("confidence_aware" in summary["policy_aggregates"], completed.stderr)
        require(model["hardware_profile"]["effective_compute_units"] == 16, "target-profile hardware not recorded")
        require(model["hardware_profile"]["effective_compute_units_source"] == "target_profile", "hardware source mismatch")
        require(model["feature_schema"]["feature_schema_version"] == 2, "feature schema version missing")
        require((tmp_path / "plans.json").exists(), "plans artifact missing")
        require((tmp_path / "oracle.json").exists(), "oracle artifact missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
