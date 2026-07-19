#!/usr/bin/env python3
"""D6: generate a distributed_profitability_contract_v1 calibration block
from real D5 calibration-split measurements, for embedding into a compiler
target-profile JSON.

    raw D5 calibration results (heterogeneous-inference-runtime)
        -> this script (offline, never run by the compiler itself)
        -> versioned target-profile calibration coefficients JSON

The compiler (DistributedStrategyPlanningPass) never reads benchmark
result files directly -- it only ever reads the small, versioned
coefficients block this script produces, embedded in a target profile
JSON (see configs/target_profiles/nvidia_rtx4090_d6_distributed_profitability.json).

Reuses heterogeneous-inference-runtime's own tested fitting implementation
(deployment.vllm_adapter.tp_cost_model.fit_linear_regression) rather than
reimplementing the regression -- a single source of truth for the fitting
algorithm, cross-checked against that repo's already-published
cost_model_fitted.json as a validation step (see --check-against).

Preserves calibration/held-out separation: only rows whose D5-declared
split is "calibration" are ever read; held-out rows and their labels are
never touched by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

THIS_REPO = Path(__file__).resolve().parents[1]
RUNTIME_REPO = THIS_REPO.parent / "heterogeneous-inference-runtime"
sys.path.insert(0, str(RUNTIME_REPO))

CONTRACT_VERSION = "distributed_profitability_contract_v1"

# The real per-GPU memory and utilization used throughout D4B/D5 on the
# rented 2x RTX 4090 host (nvidia-smi-confirmed, see
# heterogeneous-inference-runtime/results/runtime_paths/distributed_d5_compiler_tp_policy/gpu_inventory_*.json).
GPU_MEMORY_MB_PER_DEVICE = 24564.0
GPU_MEMORY_UTILIZATION = 0.9

D5_SWEEP_SOURCES = [
    ("qwen2.5-0.5b", RUNTIME_REPO / "results/runtime_paths/distributed_d5_compiler_tp_policy",
     "Qwen/Qwen2.5-0.5B-Instruct", ""),
    ("qwen2.5-7b", RUNTIME_REPO / "results/runtime_paths/distributed_d5_compiler_tp_policy/7b",
     "Qwen/Qwen2.5-7B-Instruct", "_7b"),
]


def _git_head(repo: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True,
                          text=True, check=True).stdout.strip()


def _load_calibration_rows():
    from deployment.vllm_adapter.tp_cost_model import MODEL_IDENTITY_FEATURES, build_feature_vector
    import re

    workload_id_re = re.compile(r"^in(\d+)_out(\d+)_c(\d+)$")
    rows = []
    raw_row_identities = []  # for the dataset hash: (model, workload_id, tp_degree, throughput)
    for label, sweep_dir, real_hf_model_id, suffix in D5_SWEEP_SOURCES:
        model_features = MODEL_IDENTITY_FEATURES[real_hf_model_id]
        for tp_degree, fname in ((1, f"tp1_sweep_full{suffix}.json"), (2, f"tp2_sweep_full{suffix}.json")):
            path = sweep_dir / fname
            bundle = json.loads(path.read_text())
            for wr in bundle["workload_results"]:
                if wr["split"] != "calibration":
                    continue
                m = workload_id_re.match(wr["workload_id"])
                input_length, output_length, concurrency = int(m.group(1)), int(m.group(2)), int(m.group(3))
                fv = build_feature_vector(model_features, tp_degree, input_length=input_length,
                                          output_length=output_length, concurrency=concurrency)
                throughput = wr["aggregate_throughput_tokens_per_s"]
                rows.append({"tp_degree": tp_degree, "feature_vector": fv,
                            "aggregate_throughput_tokens_per_s": throughput})
                raw_row_identities.append(f"{label}|{wr['workload_id']}|tp{tp_degree}|{throughput:.10f}")
    dataset_hash = hashlib.sha256("\n".join(sorted(raw_row_identities)).encode()).hexdigest()
    return rows, dataset_hash


def _fit(rows) -> dict:
    from deployment.vllm_adapter.tp_cost_model import fit_linear_regression, FEATURE_NAMES

    out = {}
    for tp_degree in (1, 2):
        subset = [r for r in rows if r["tp_degree"] == tp_degree]
        X = [r["feature_vector"] for r in subset]
        y = [r["aggregate_throughput_tokens_per_s"] for r in subset]
        reg = fit_linear_regression(X, y)
        coeffs = dict(zip(["intercept", *FEATURE_NAMES], reg.coefficients))
        out[f"tp{tp_degree}"] = {"coefficients": coeffs, "n_samples": reg.n_samples, "r_squared": reg.r_squared}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="output path for the calibration block JSON")
    ap.add_argument("--check-against", type=Path, default=None,
                    help="optional: heterogeneous-inference-runtime's cost_model_fitted.json, "
                         "to cross-check this script's independently-fit coefficients match")
    args = ap.parse_args()

    rows, dataset_hash = _load_calibration_rows()
    print(f"loaded {len(rows)} real D5 calibration-split rows (held-out rows never read)")
    fit = _fit(rows)

    if args.check_against:
        published = json.loads(args.check_against.read_text())
        for tp_degree in (1, 2):
            published_coeffs = published["throughput_models"][str(tp_degree)]["coefficients"]
            ours = list(fit[f"tp{tp_degree}"]["coefficients"].values())
            max_diff = max(abs(a - b) for a, b in zip(ours, published_coeffs))
            print(f"tp{tp_degree}: max coefficient diff vs published cost_model_fitted.json = {max_diff:.2e}")
            assert max_diff < 1e-6, (
                f"independently-refit tp{tp_degree} coefficients diverge from the published D5 "
                f"cost_model_fitted.json by {max_diff:.2e} -- refusing to emit a calibration profile "
                "that doesn't match the validated D5 result"
            )
        print("cross-check passed: independently-refit coefficients match the published D5 result")

    block = {
        "contractVersion": CONTRACT_VERSION,
        "calibrationDatasetHash": dataset_hash,
        "calibrationHardwareIdentity": (
            "2x NVIDIA GeForce RTX 4090, PCIe Gen4 (no NVLink), rented Vast.ai host, "
            "confirmed via nvidia-smi during the D4B/D5 real-hardware runs"
        ),
        "calibrationGeneratedAt": "2026-07-19T00:00:00Z",
        "calibrationCompilerCommit": _git_head(THIS_REPO),
        "calibrationRuntimeCommit": _git_head(RUNTIME_REPO),
        "gpuMemoryMbPerDevice": GPU_MEMORY_MB_PER_DEVICE,
        "gpuMemoryUtilization": GPU_MEMORY_UTILIZATION,
        "tieBreakRule": "prefer_lower_tp_degree_within_epsilon",
        "truthBoundary": (
            "linear_regression_calibrated_from_real_d5_measured_throughput_on_2x_rtx4090_"
            "pcie_no_nvlink_54_calibration_rows_both_qwen2.5_0.5b_and_7b_not_a_full_"
            "systems_simulator_not_valid_off_this_calibration_hardware_never_encodes_"
            "held_out_labels"
        ),
        "tp1Coefficients": fit["tp1"]["coefficients"],
        "tp2Coefficients": fit["tp2"]["coefficients"],
        "fitQuality": {"tp1": {"r_squared": fit["tp1"]["r_squared"], "n_samples": fit["tp1"]["n_samples"]},
                      "tp2": {"r_squared": fit["tp2"]["r_squared"], "n_samples": fit["tp2"]["n_samples"]}},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(block, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
