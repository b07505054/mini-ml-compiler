#!/usr/bin/env python3
"""Validate the Phase 1 CPU fused schedule discovery JSON contract.

Runs the discovery tool in --smoke mode (tiny warmup/iterations/repeats) and
checks structure and invariants only — not performance thresholds.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Repaired candidate set (default): the original 4-candidate tier collapsed
# to one dominant candidate on both Apple M5 and remote Intel i5-10210U, so
# the default tool invocation now uses this repaired, wider set (varies
# block_k for the first time; spans small/near-L1/above-L1/rectangular tile
# footprints). See apps/run_cpu_fused_schedule_discovery.cpp make_repaired_candidates().
CANDIDATE_IDS = {
    "bm16_bn16_bk16", "bm32_bn32_bk32", "bm64_bn64_bk32", "bm64_bn64_bk128",
    "bm128_bn128_bk32", "bm128_bn128_bk256", "bm16_bn128_bk32", "bm128_bn16_bk32",
}


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    if len(sys.argv) != 3:
        fail("usage: validate_cpu_fused_schedule_discovery_json.py <benchmark_exe> <output_dir>")
    benchmark_exe, output_dir = sys.argv[1], Path(sys.argv[2])

    subprocess.run(
        [benchmark_exe, "--mode", "discover", "--smoke",
         "--target-profile-id", "test-schema-check", "--output-dir", str(output_dir)],
        check=True,
    )

    environment = json.loads((output_dir / "environment.json").read_text())
    if environment["schema"] != "cpu_fused_schedule_discovery_environment":
        fail("environment.json: wrong schema")
    for artifact_name in ("environment.json", "candidate_contract.json", "workload_manifest.json",
                          "benchmark_measurements.json", "plan_dispatch_validation.json"):
        payload = json.loads((output_dir / artifact_name).read_text())
        prov = payload.get("provenance")
        if not isinstance(prov, dict):
            fail(f"{artifact_name}: missing required 'provenance' block")
        for field in ("target_host", "git_commit", "target_profile_id", "utc_timestamp"):
            if not prov.get(field):
                fail(f"{artifact_name}: provenance missing or empty field '{field}'")
        if prov["target_profile_id"] != "test-schema-check":
            fail(f"{artifact_name}: target_profile_id was not threaded through from CLI")
    for key in ("cpu_model", "os", "arch", "compiler", "benchmark_thread_count"):
        fact = environment.get(key)
        if not isinstance(fact, dict) or "value" not in fact or "source" not in fact:
            fail(f"environment.json: missing or malformed fact '{key}'")
    if environment["benchmark_thread_count"]["value"] != "1":
        fail("environment.json: Phase 1 must fix benchmark_thread_count=1")

    contract = json.loads((output_dir / "candidate_contract.json").read_text())
    candidates = contract["candidates"]
    if {c["candidate_id"] for c in candidates} != CANDIDATE_IDS:
        fail(f"candidate_contract.json: unexpected candidate set {candidates}")
    for c in candidates:
        if c["thread_count"] != 1:
            fail(f"candidate_contract.json: {c['candidate_id']} thread_count must be 1")
        if c["dtype"] != "f32" or c["accumulator_dtype"] != "f32":
            fail(f"candidate_contract.json: {c['candidate_id']} dtype/accumulator must both be f32")
        if c["launch_count"] != 1 or c["full_size_intermediates"] != 0:
            fail(f"candidate_contract.json: {c['candidate_id']} must be one-pass fused with no full intermediates")

    manifest = json.loads((output_dir / "workload_manifest.json").read_text())
    workloads = manifest["workloads"]
    if not workloads:
        fail("workload_manifest.json: no workloads recorded")
    families = {w["family"] for w in workloads}
    expected_families = {
        "small_square", "small_output_high_reduction", "skinny_wide",
        "medium_rectangular", "large_regular", "edge_heavy",
    }
    if not expected_families.issubset(families):
        fail(f"workload_manifest.json: missing families, got {families}")

    measurements = json.loads((output_dir / "benchmark_measurements.json").read_text())
    if len(measurements["workloads"]) != len(workloads):
        fail("benchmark_measurements.json: workload count does not match manifest")

    baseline = measurements.get("fusion_attribution_baseline")
    if not baseline:
        fail("benchmark_measurements.json: missing fusion_attribution_baseline (must be kept separate from schedule oracle)")
    if not baseline["unfused"]["correctness_passed"] or not baseline["fused"]["correctness_passed"]:
        fail("fusion_attribution_baseline: correctness failed")
    if baseline["unfused"]["launch_count"] != 3 or baseline["fused"]["launch_count"] != 1:
        fail("fusion_attribution_baseline: launch counts must be 3 (unfused) and 1 (fused)")
    for wl in measurements["workloads"]:
        cand_ids = {c["candidate_id"] for c in wl["candidates"]}
        if cand_ids != CANDIDATE_IDS:
            fail(f"benchmark_measurements.json: workload {wl['workload_id']} missing candidates: {cand_ids}")
        for c in wl["candidates"]:
            if not c["correctness"]["passed"]:
                fail(
                    f"benchmark_measurements.json: candidate {c['candidate_id']} failed correctness "
                    f"on workload {wl['workload_id']}"
                )
            if len(c["samples_ms"]) != wl["budget"]["repeats"]:
                fail(
                    f"benchmark_measurements.json: candidate {c['candidate_id']} sample count "
                    f"{len(c['samples_ms'])} does not match budget repeats {wl['budget']['repeats']}"
                )
            for sample in c["samples_ms"]:
                if sample < 0:
                    fail("benchmark_measurements.json: negative latency sample")

    plan_validation = json.loads((output_dir / "plan_dispatch_validation.json").read_text())
    if plan_validation["total_override_count"] != 0:
        fail(
            f"plan_dispatch_validation.json: expected zero overrides, got "
            f"{plan_validation['total_override_count']}"
        )
    if len(plan_validation["validations"]) != len(CANDIDATE_IDS):
        fail("plan_dispatch_validation.json: missing per-candidate validation entries")
    for entry in plan_validation["validations"]:
        if not entry["plan_matched_runtime"]:
            fail(f"plan_dispatch_validation.json: candidate {entry['candidate_id']} plan did not match runtime")

    print("OK: cpu_fused_schedule_discovery JSON contract validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
