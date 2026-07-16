#!/usr/bin/env python3
"""merge_tile_candidate_results.py

Merges the four separately-produced evidence sources for the AArch64
tile-candidate slice into one candidate_results.json matching the schema
in the task brief:
  1. Static/structural/backend metrics (generate_aarch64_matmul_tile_candidates.py)
  2. Assembly-derived register-pressure evidence (analyze_register_pressure.py)
  3. Raspberry Pi repeated-call correctness (repeated_call_results.txt, from
     tools/run_tile_candidates_pi_integration.sh)
  4. Raspberry Pi benchmark latency (benchmark_<shape>.json, same script)

Usage:
  python3 tools/merge_tile_candidate_results.py \
    --static /tmp/tile_candidate_static_results.json \
    --register-pressure /tmp/tile_candidate_register_pressure.json \
    --pi-results-dir /tmp/tile_candidate_pi_results \
    --output candidate_results.json
"""
import argparse
import glob
import json
import os
import re


def parse_repeated_call_log(path):
    """Returns {(shapeKey, tileKey): {"passed_1_call": bool, "passed_repeated": bool,
    "repeated_calls": int}}."""
    result = {}
    if not os.path.isfile(path):
        return result
    for line in open(path):
        m = re.match(r"(PASS|FAIL): shape=(\S+) tile=(\S+) .*?(\d+) calls", line)
        if not m:
            continue
        verdict, shape_key, tile_key, calls_str = m.groups()
        calls = int(calls_str)
        key = (shape_key, tile_key)
        result.setdefault(key, {"passed_1_call": None, "passed_repeated": None, "repeated_calls": None})
        if calls == 1:
            result[key]["passed_1_call"] = (verdict == "PASS")
        else:
            result[key]["passed_repeated"] = (verdict == "PASS")
            result[key]["repeated_calls"] = calls
    return result


def parse_benchmarks(pi_results_dir):
    """Returns {(shapeKey, tileKey): {"median_ms":..,"p95_ms":..,"correct":..,"max_abs_error":..}}."""
    result = {}
    for path in glob.glob(os.path.join(pi_results_dir, "benchmark_*.json")):
        shape_key = os.path.basename(path)[len("benchmark_"):-len(".json")]
        data = json.load(open(path))
        for cand in data.get("candidates", []):
            key = (shape_key, cand["tile"])
            result[key] = {
                "median_ms": cand["median_ms"],
                "p95_ms": cand["p95_ms"],
                "correct": cand["correct"],
                "max_abs_error": cand["max_abs_error"],
            }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", required=True)
    ap.add_argument("--register-pressure", required=True)
    ap.add_argument("--pi-results-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    static_data = json.load(open(args.static))["candidates"]
    rp_data = {
        (r["shape"][0], r["shape"][1], r["shape"][2], r["tile"]["m"], r["tile"]["n"], r["tile"]["k"]): r["register_pressure"]
        for r in json.load(open(args.register_pressure))["register_pressure"]
    }
    repeated = parse_repeated_call_log(os.path.join(args.pi_results_dir, "repeated_call_results.txt"))
    benchmarks = parse_benchmarks(args.pi_results_dir)

    merged = []
    for c in static_data:
        m, n, k = c["shape"]
        record = {
            "shape": [m, n, k],
            "tile": c["tile"],
            "legality": c["legality"],
        }
        if not c["legality"]["legal"]:
            merged.append(record)
            continue

        tm, tn, tk = c["tile"]["m"], c["tile"]["n"], c["tile"]["k"]
        shape_key = f"{m}x{n}x{k}"
        tile_key = f"tm{tm}_tn{tn}_tk{tk}"

        rp = rp_data.get((m, n, k, tm, tn, tk))
        rc = repeated.get((shape_key, tile_key), {})
        bm = benchmarks.get((shape_key, tile_key))

        passed = bool(rc.get("passed_1_call")) and bool(rc.get("passed_repeated"))
        max_err = bm["max_abs_error"] if bm else None
        correctness = {
            "passed": passed,
            "max_abs_error": max_err,
            "repeated_calls": rc.get("repeated_calls"),
            "note": None if rc else "no repeated-call evidence found -- Pi integration may not have run for this candidate",
        }

        performance = None
        if bm is not None:
            performance = {"median_ms": bm["median_ms"], "p95_ms": bm["p95_ms"]}
        else:
            performance = {"median_ms": None, "p95_ms": None, "note": "no benchmark evidence found"}

        backend = dict(c.get("backend", {}))
        if rp is not None:
            backend["hot_loop_vector_spills"] = rp["hot_loop_vector_spills"]
            backend["hot_loop_vector_reloads"] = rp["hot_loop_vector_reloads"]
            backend["hot_loop_integer_spills"] = rp["hot_loop_integer_spills"]
            backend["hot_loop_integer_reloads"] = rp["hot_loop_integer_reloads"]
            backend["vector_registers_referenced"] = rp["vector_registers_referenced_count"]
            backend["abi_callee_saved_folded_spills"] = rp["abi_callee_saved_folded_spills"]
            backend["abi_callee_saved_folded_reloads"] = rp["abi_callee_saved_folded_reloads"]
            backend["hot_loop_is_whole_function"] = rp["hot_loop_is_whole_function"]
            backend["register_pressure_evidence_kind"] = rp["evidence_kind"]
        else:
            backend["hot_loop_vector_spills"] = None
            backend["hot_loop_vector_reloads"] = None
            backend["note_register_pressure"] = "no register-pressure evidence found"

        record["object_name"] = c["object_name"]
        record["function_name"] = c["function_name"]
        record["structural"] = c["structural"]
        record["correctness"] = correctness
        record["performance"] = performance
        record["backend"] = backend
        merged.append(record)

    out = {
        "target": {
            "architecture": "aarch64",
            "cpu": "cortex-a76",
            "device": "Raspberry Pi 5",
        },
        "results": merged,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(merged)} merged candidate records to {args.output}")


if __name__ == "__main__":
    main()
