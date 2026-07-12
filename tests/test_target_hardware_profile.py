#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import target_hardware_profile as hp  # noqa: E402


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def write_profile(path: Path, block: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "profileId": "fixture-gpu",
                "configuredComputeUnits": "CUDA",
                "hardwareExecutionProfile": block,
            }
        ),
        encoding="utf-8",
    )


def main() -> int:
    gtx = hp.load_target_hardware_execution_profile(
        ROOT / "configs/target_profiles/nvidia_gtx1650_maxq.json"
    )
    require(gtx.target_id == "nvidia_gtx1650_maxq", "GTX target ID should normalize")
    require(gtx.profile_kind is None, "real GTX profile kind should be absent")
    require(gtx.physical_compute_units == 16, "GTX physical compute units")
    require(gtx.effective_compute_units == 16, "GTX effective compute units")
    require(
        gtx.max_concurrent_work_items_per_unit == 1,
        "GTX compatibility concurrent work items",
    )
    require(gtx.supports_latency_hiding is True, "GTX latency hiding")
    require(
        gtx.local_memory_kind == "software_managed_shared_memory",
        "GTX local memory kind",
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        missing = tmp_path / "missing.json"
        write_profile(missing, {})
        prof = hp.load_target_hardware_execution_profile(missing)
        require(prof.effective_compute_units is None, "missing optional fields stay missing")
        resolved = hp.resolve_effective_compute_units(
            prof,
            {"sm_count": 12},
            cli_override=None,
            compatibility_default=16,
        )
        require(resolved.effective_compute_units == 12, "artifact source should be used when profile missing")
        require(resolved.effective_compute_units_source == "artifact_environment", "artifact source label")

        invalid = tmp_path / "invalid.json"
        write_profile(invalid, {"effectiveComputeUnits": 0})
        try:
            hp.load_target_hardware_execution_profile(invalid)
            raise AssertionError("invalid zero compute units must fail")
        except ValueError:
            pass

        malformed = tmp_path / "malformed.json"
        write_profile(malformed, {"localMemoryKind": "cache_or_scratchpad_maybe"})
        try:
            hp.load_target_hardware_execution_profile(malformed)
            raise AssertionError("malformed localMemoryKind must fail")
        except ValueError:
            pass

    resolved = hp.resolve_effective_compute_units(
        gtx,
        {"gpu_model": "NVIDIA GeForce GTX 1650 with Max-Q Design", "compute_capability": [7, 5], "sm_count": 16},
        cli_override=None,
        compatibility_default=99,
    )
    require(resolved.effective_compute_units == 16, "profile beats fallback")
    require(resolved.effective_compute_units_source == "target_profile", "profile source label")
    require(resolved.artifact_compute_units == 16, "artifact units recorded")
    require(resolved.identity_validation == "matched", "matching artifact should validate")

    try:
        hp.resolve_effective_compute_units(
            gtx,
            {"gpu_model": "NVIDIA GeForce GTX 1650 with Max-Q Design", "compute_capability": [7, 5], "sm_count": 12},
            cli_override=None,
            compatibility_default=16,
        )
        raise AssertionError("profile/artifact compute-unit mismatch must fail")
    except ValueError:
        pass

    try:
        hp.resolve_effective_compute_units(
            gtx,
            {},
            cli_override=12,
            compatibility_default=16,
        )
        raise AssertionError("conflicting CLI override must require explicit intent")
    except ValueError:
        pass

    override = hp.resolve_effective_compute_units(
        gtx,
        {},
        cli_override=12,
        compatibility_default=16,
        allow_cli_override=True,
    )
    require(override.effective_compute_units == 12, "explicit CLI override should work")
    require(override.effective_compute_units_source == "cli_override", "CLI source label")

    synthetic = hp.load_target_hardware_execution_profile(
        ROOT / "configs/target_profiles/synthetic_gpu_40cu.json"
    )
    require(synthetic.profile_kind == "synthetic_analytical", "synthetic profile kind")
    require(synthetic.effective_compute_units == 40, "synthetic CU value")
    synthetic_resolved = hp.resolve_effective_compute_units(
        synthetic,
        {},
        cli_override=None,
        compatibility_default=None,
    )
    require(synthetic_resolved.identity_validation == "matched", "synthetic analytical profile without artifact should validate")
    try:
        hp.resolve_effective_compute_units(
            synthetic,
            {"gpu_model": "NVIDIA GeForce GTX 1650 with Max-Q Design", "sm_count": 16},
            cli_override=None,
            compatibility_default=None,
        )
        raise AssertionError("synthetic profile must not match real benchmark artifact")
    except ValueError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
