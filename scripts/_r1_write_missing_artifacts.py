#!/usr/bin/env python3
"""R1: write target_profile_resolution.json and raw_measurements.json,
the two required artifacts not produced directly by the discovery tool
or the session aggregator."""
import json
from pathlib import Path

OUT = Path("trace/remote_intel_cpu_fused_schedule_discovery")

resolution = {
    "schema": "cpu_fused_schedule_target_profile_resolution",
    "schema_version": 1,
    "target_profile_id_used": "remote-intel-i5-10210u",
    "resolution_status": "no_canonical_profile_exists",
    "canonical_target_profiles_directory": "configs/target_profiles/",
    "canonical_profiles_present": ["intel_amx_cpu.json", "intel_ai_server.json"],
    "why_not_used": (
        "Both existing Intel-named profiles (intel_amx_cpu.json, intel_ai_server.json) are "
        "declared/public_docs profiles for Sapphire-Rapids-class server silicon with AMX "
        "(BF16/INT8 tile matmul). The real remote CPU (Intel Core i5-10210U, Comet Lake "
        "mobile, verified via lscpu: family 6 model 142) has NO AMX support "
        "(verified via /proc/cpuinfo flags: avx2 present, no amx* flags, no avx512* flags). "
        "Using either canonical profile for this experiment would misrepresent the real "
        "hardware. This was identified during Phase R0 "
        "(see trace/remote_linux_baseline/cpu_hardware.json)."
    ),
    "target_profile_id_is": (
        "an ad hoc experiment-scoped identifier, NOT backed by a JSON file under "
        "configs/target_profiles/. It exists only to satisfy the per-artifact provenance "
        "requirement (every artifact must record which target it was generated for) and to "
        "make cross-target plan-dispatch rejection meaningful during this experiment."
    ),
    "recommendation": (
        "A real intel_i5_10210u_mobile.json (or similarly scoped) canonical profile should "
        "be authored before this target participates in the joint backend x schedule x "
        "precision x quantization decision model (Phase R8) -- not required for R1's "
        "schedule-discovery-only scope."
    ),
    "real_cpu_facts_verified": {
        "vendor": "GenuineIntel",
        "model_name": "Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz",
        "has_avx2": True,
        "has_avx512": False,
        "has_amx": False,
        "physical_cores": 4,
        "logical_cores": 8,
        "l1d_bytes_per_core": 32768,
        "l2_bytes_per_core": 262144,
        "l3_bytes_total": 6291456,
        "source": "trace/remote_linux_baseline/cpu_hardware.json (Phase R0, live lscpu/getconf query)",
    },
}
(OUT / "target_profile_resolution.json").write_text(json.dumps(resolution, indent=2), encoding="utf-8")

pooled = json.loads((OUT / "benchmark_measurements.json").read_text())
(OUT / "raw_measurements.json").write_text(json.dumps(pooled, indent=2), encoding="utf-8")

print("wrote target_profile_resolution.json and raw_measurements.json")
