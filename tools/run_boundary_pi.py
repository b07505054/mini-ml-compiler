#!/usr/bin/env python3
"""Stage 17: Pi benchmark matrix for the 2 new stress domains (smallA,
highK), each at uk1/uk2/uk4. Reuses Stage 13's low-level Pi orchestration
functions unmodified. Objects already compiled through the Stage 15
selector's manual mode (see compiled/)."""
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import run_aarch64_schedule_pi_validation as p13  # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "backend_codegen", "aarch64_matmul_bias_relu_schedule_boundary")
COMPILED = os.path.join(ART, "compiled")

PI_HOST = "allen@100.110.37.6"
TASKSET_CORE = 3
FULL_RIGOR_GROUPS = 5

# (label, shape, tile_m, tile_n, tile_k, unroll_k, iterations, warmup, repeated_calls)
CANDIDATES = [
    ("smallA_uk1", "16x16x16", 8, 8, 4, 1, 3000, 300, 500),
    ("smallA_uk2", "16x16x16", 8, 8, 4, 2, 3000, 300, 500),
    ("smallA_uk4", "16x16x16", 8, 8, 4, 4, 3000, 300, 500),
    ("highK_uk1", "32x32x128", 8, 8, 8, 1, 1000, 200, 300),
    ("highK_uk2", "32x32x128", 8, 8, 8, 2, 1000, 200, 300),
    ("highK_uk4", "32x32x128", 8, 8, 8, 4, 1000, 200, 300),
]
DOMAINS = {
    "smallA": ["smallA_uk1", "smallA_uk2", "smallA_uk4"],
    "highK": ["highK_uk1", "highK_uk2", "highK_uk4"],
}


def main():
    out_dir = ART
    pi_dir = f"/tmp/stage17_boundary_pi_{int(time.time())}"

    print(f"[env] capturing Pi environment...", file=sys.stderr)
    env = p13.capture_pi_environment(PI_HOST, out_dir)
    p13.capture_thermal_snapshot(PI_HOST, "before_all", out_dir)

    built = {}
    for (label, shape, tm, tn, tk, uk, iterations, warmup, repeated_calls) in CANDIDATES:
        obj_path = os.path.join(COMPILED, f"{label}.o")
        obj_sha256 = p13.sha256_file(obj_path)
        print(f"[build {label}] ...", file=sys.stderr)
        cand_dir = os.path.join(out_dir, "pi_candidates", label)
        os.makedirs(cand_dir, exist_ok=True)
        build_info = p13.build_on_pi(PI_HOST, pi_dir, shape, label, obj_path, obj_sha256, cand_dir)
        built[label] = {
            "label": label, "shape": shape, "tile": {"m": tm, "n": tn, "k": tk}, "schedule_unroll_k": uk,
            "iterations": iterations, "warmup": warmup, "repeated_calls": repeated_calls,
            "obj_sha256": obj_sha256, "build": build_info, "cand_dir": cand_dir,
        }

    all_group_results = {label: [] for label in built}
    for domain_name, labels in DOMAINS.items():
        print(f"[interleave domain {domain_name}] {labels}", file=sys.stderr)
        for g in range(FULL_RIGOR_GROUPS):
            for label in labels:
                cfg = built[label]
                shape_m, shape_n, shape_k = p13.shape_dims(cfg["shape"])
                dump_path = f"{pi_dir}/{label}_out_g{g}.bin" if g == 0 else None
                r = p13.run_harness(
                    PI_HOST, cfg["build"]["remote_binary"], shape_m, shape_n, shape_k,
                    cfg["tile"]["m"], cfg["tile"]["n"], cfg["tile"]["k"], cfg["schedule_unroll_k"],
                    label, cfg["iterations"], cfg["warmup"], cfg["repeated_calls"],
                    "0x1234", "0x5eed", dump_output_path=dump_path, taskset_core=TASKSET_CORE,
                )
                all_group_results[label].append(r)

    p13.capture_thermal_snapshot(PI_HOST, "after_all", out_dir)

    results = {}
    for label, cfg in built.items():
        group_results = all_group_results[label]
        record = {
            "label": label, "shape": cfg["shape"], "tile": cfg["tile"], "schedule_unroll_k": cfg["schedule_unroll_k"],
            "obj_sha256": cfg["obj_sha256"],
            "measurement_groups": group_results,
            "correctness_pass": all(g["correctness"]["overall_pass"] for g in group_results),
            "benchmark_aggregate": p13.aggregate_measurement_groups([g["benchmark"] for g in group_results]),
        }
        with open(os.path.join(cfg["cand_dir"], "manifest.json"), "w") as f:
            json.dump(record, f, indent=2)
        results[label] = record

    with open(os.path.join(out_dir, "benchmark_results.json"), "w") as f:
        json.dump({"pi_environment": env, "candidates": results}, f, indent=2)

    print(f"\nWrote {os.path.join(out_dir, 'benchmark_results.json')}", file=sys.stderr)
    for label, r in results.items():
        agg = r["benchmark_aggregate"]
        print(f"{label}: pass={r['correctness_pass']} median={agg['median_of_medians_ms']} cv={agg['cv_of_medians']:.4f}", file=sys.stderr)

    p13.ssh_run(PI_HOST, f"rm -rf {pi_dir}", check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
