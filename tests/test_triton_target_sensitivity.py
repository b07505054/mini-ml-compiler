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
import run_triton_target_sensitivity_evaluation as sens  # noqa: E402
import target_hardware_profile as hp  # noqa: E402


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def load_model() -> dict:
    base_rows, _ = base.parse_discovery_measurements(ROOT / "trace/matmul_postop_triton_fused_candidate_sweep.json")
    repair_rows = repair.parse_repair_profile(ROOT / "trace/matmul_postop_triton_fused_config_repair_training_profile.json")
    fresh_rows = repair.parse_repair_profile(ROOT / "trace/matmul_postop_triton_fused_config_repair_fresh_oracle.json")
    training = {r.workload_id: r for r in base_rows + repair_rows}
    all_rows = {r.workload_id: r for r in base_rows + repair_rows + fresh_rows}
    train, _, _ = repair.grouped_split(list(all_rows.values()), "leave-one-shape-region-out")
    train = [training[r.workload_id] for r in train if r.workload_id in training]
    return repair.calibrate(train, 16)


def profiles() -> list[hp.ResolvedHardwareProfile]:
    return [
        hp.load_and_resolve_hardware_profile(ROOT / f"configs/target_profiles/synthetic_gpu_{cu}cu.json", {}, compatibility_default=None)
        for cu in (8, 16, 40, 80)
    ]


def main() -> int:
    profs = profiles()
    require([p.effective_compute_units for p in profs] == [8, 16, 40, 80], "synthetic profile CU order")
    require(all(p.profile_kind == "synthetic_analytical" for p in profs), "synthetic profile kind missing")

    model = load_model()
    shape = {"m": 64, "n": 64, "k": 4096, "dtype": "f32"}
    cfg = base.PRIMARY_CONFIGS["bm16_bn16_bk32_w4_s3"]
    feats = [repair.tile_features(repair.workload_features(shape), cfg, p.effective_compute_units) for p in profs]
    require([f["work_items_per_compute_unit"] for f in feats] == [2.0, 1.0, 0.4, 0.2], "work-items per CU should vary")
    require([f["execution_waves"] for f in feats] == [2, 1, 1, 1], "execution waves should vary")
    require(all(f["programs_per_sm"] == f["work_items_per_compute_unit"] for f in feats), "compat alias mismatch")

    same_shape = {"workload_id": "same_shape", "family": "fixture", "shape": shape}
    evaluated = sens.evaluate_workload(same_shape, profs, model)
    require(evaluated["classification"] in {"feature-sensitive", "ranking-sensitive", "selection-sensitive"}, "shape should be hardware sensitive")
    require(len({p["effective_compute_units"] for p in evaluated["profiles"]}) == 4, "only hardware profile should vary")

    rank_case = sens.evaluate_workload(
        {"workload_id": "rank_case", "family": "fixture", "shape": {"m": 32, "n": 32, "k": 1024, "dtype": "f32"}},
        profs,
        model,
    )
    require(rank_case["classification"] == "ranking-sensitive", "expected deterministic ranking-sensitive case")

    select_case = sens.evaluate_workload(
        {"workload_id": "select_case", "family": "fixture", "shape": {"m": 64, "n": 64, "k": 512, "dtype": "f32"}},
        profs,
        model,
    )
    require(select_case["classification"] == "selection-sensitive", "expected deterministic selection-sensitive case")
    require(len(set(select_case["selected_configs_by_profile"].values())) > 1, "top-1 selection should vary")

    try:
        hp.load_and_resolve_hardware_profile(
            ROOT / "configs/target_profiles/synthetic_gpu_8cu.json",
            {"gpu_model": "NVIDIA GeForce GTX 1650 with Max-Q Design", "sm_count": 16},
            compatibility_default=None,
        )
        raise AssertionError("synthetic profiles must not claim real artifact validation")
    except ValueError:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        subprocess.run(
            [
                sys.executable,
                str(TOOLS / "run_triton_target_sensitivity_evaluation.py"),
                "--profiles",
                str(ROOT / "configs/target_profiles/synthetic_gpu_8cu.json"),
                str(ROOT / "configs/target_profiles/synthetic_gpu_16cu.json"),
                str(ROOT / "configs/target_profiles/synthetic_gpu_40cu.json"),
                str(ROOT / "configs/target_profiles/synthetic_gpu_80cu.json"),
                "--output-dir",
                str(out),
                "--doc-output",
                str(out / "doc.md"),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        summary = json.loads((out / "target_sensitivity_summary.json").read_text(encoding="utf-8"))
        require(summary["has_ranking_sensitive_case"], "driver should find ranking sensitivity")
        require(summary["has_selection_sensitive_case"], "driver should find selection sensitivity")
        require((out / "decision_boundary_cases.json").exists(), "boundary artifact missing")
        require("analytical-only" in (out / "report.md").read_text(encoding="utf-8"), "truth boundary missing")

    gtx_model = hp.load_and_resolve_hardware_profile(
        ROOT / "configs/target_profiles/nvidia_gtx1650_maxq.json",
        {"gpu_model": "NVIDIA GeForce GTX 1650 with Max-Q Design", "compute_capability": [7, 5], "sm_count": 16},
        compatibility_default=16,
    )
    require(gtx_model.effective_compute_units == 16, "GTX profile regression")
    require(gtx_model.identity_validation == "matched", "GTX identity validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
