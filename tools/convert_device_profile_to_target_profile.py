"""
Convert a PocketChef snake_case device profile export to a camelCase compiler
target profile, injecting cost constants from a bootstrap prior artifact.

Usage:
    python3 tools/convert_device_profile_to_target_profile.py \
        --device-profile <path/to/target_device_profile.json> \
        --bootstrap-prior configs/bootstrap_priors/runtime_gpu_bootstrap_prior.json \
        --out configs/target_profiles/<output>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DeviceProfile:
    profile_id: str
    model_identifier: str
    chip_name: str
    metal_max_working_set_mb: int
    configured_compute_units: str
    truth_boundary: str

    @classmethod
    def from_dict(cls, d: dict) -> "DeviceProfile":
        required = ["profile_id", "model_identifier", "chip_name",
                    "metal_max_working_set_mb", "configured_compute_units",
                    "truth_boundary"]
        for k in required:
            if k not in d:
                print(f"ERROR: device profile missing required field: {k}", file=sys.stderr)
                sys.exit(1)
        return cls(
            profile_id=d["profile_id"],
            model_identifier=d["model_identifier"],
            chip_name=d["chip_name"],
            metal_max_working_set_mb=int(d["metal_max_working_set_mb"]),
            configured_compute_units=d["configured_compute_units"],
            truth_boundary=d["truth_boundary"],
        )


@dataclass
class BootstrapPrior:
    prefill_ms_per_token: float
    decode_ms_per_token: float
    source_artifact: str
    source_project: str
    truth_boundary: str

    @classmethod
    def from_dict(cls, d: dict) -> "BootstrapPrior":
        try:
            constants = d["cost_constants"]
            source = d["source"]
            return cls(
                prefill_ms_per_token=float(constants["prefill_ms_per_token"]),
                decode_ms_per_token=float(constants["decode_ms_per_token"]),
                source_artifact=source["artifact"],
                source_project=source["project"],
                truth_boundary=d["truth_boundary"],
            )
        except KeyError as e:
            print(f"ERROR: bootstrap prior missing field: {e}", file=sys.stderr)
            sys.exit(1)


def build_target_profile(device: DeviceProfile, prior: BootstrapPrior) -> dict:
    return {
        "profileId": device.profile_id,
        "modelIdentifier": device.model_identifier,
        "chipName": device.chip_name,
        "metalMaxWorkingSetMB": device.metal_max_working_set_mb,
        "configuredComputeUnits": device.configured_compute_units,
        "staticShapeSupport": True,
        "supportedPrecisions": ["fp16"],
        "pagedKVCompatibleBackends": [],
        "prefillMsPerToken": prior.prefill_ms_per_token,
        "decodeMsPerToken": prior.decode_ms_per_token,
        "costSource": "runtime_project_gpu_bootstrap_prior",
        "costCalibrationSource": f"{prior.source_project}/{prior.source_artifact}",
        "costCalibrationTruthBoundary": prior.truth_boundary,
        "truthBoundary": device.truth_boundary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-profile", required=True, type=Path)
    parser.add_argument("--bootstrap-prior", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    for p, label in [(args.device_profile, "device profile"),
                     (args.bootstrap_prior, "bootstrap prior")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    with open(args.device_profile) as f:
        device = DeviceProfile.from_dict(json.load(f))

    with open(args.bootstrap_prior) as f:
        prior = BootstrapPrior.from_dict(json.load(f))

    target_profile = build_target_profile(device, prior)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(target_profile, f, indent=2)
        f.write("\n")

    print(f"Wrote target profile: {args.out}")
    print(f"  profileId:                   {target_profile['profileId']}")
    print(f"  prefillMsPerToken:           {target_profile['prefillMsPerToken']}")
    print(f"  decodeMsPerToken:            {target_profile['decodeMsPerToken']}")
    print(f"  costSource:                  {target_profile['costSource']}")
    print(f"  costCalibrationTruthBoundary:{target_profile['costCalibrationTruthBoundary']}")
    print(f"  truthBoundary:               {target_profile['truthBoundary']}")


if __name__ == "__main__":
    main()
