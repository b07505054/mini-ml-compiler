#!/usr/bin/env python3
"""
Validation tests for deployment-family target profiles (5-level schema).

Tests:
  1. All 5 profiles parse as valid JSON with required top-level fields.
  2. backendCapabilities (Level 1) structure is present and complete.
  3. Level 2 constraint fields lower correctly (no unknown numeric zero).
  4. Level 3 preference fields are present where expected.
  5. Level 4 runtime model fields roundtrip if present.
  6. Unknown numeric cost fields are absent (absent != 0.0).
  7. representative_devices metadata is present and preserved.

Usage:
  python3 validate_target_profiles.py
  Exit 0 on success, 1 on any failure.
"""
import json
import os
import sys

PROFILES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "configs", "target_profiles",
)

FAMILY_PROFILES = [
    "nvidia_datacenter_gpu.json",
    "amd_instinct_datacenter_gpu.json",
    "intel_ai_server.json",
    "apple_coreml_mobile.json",
    "edge_npu_mobile.json",
]

# Level 1: Hardware Capability — required in every backendCapabilities entry.
REQUIRED_CAPABILITY_FIELDS = [
    "backendName",
    "backendApi",
    "supportedOps",
    "supportedDtypes",
    "accumulationDtypes",
    "supportedQuantModes",
    "preferredActivationLayouts",
    "preferredWeightLayouts",
    "layoutAgnosticOps",
    "supportsLayoutTransform",
    "supportsCast",
    "supportsDequantBoundary",
    "supportsRequantize",
    "supportsFusionPatterns",
    "sourceLevel",
    "truthBoundary",
]

# Level 2: Constraint fields — at least one must be present in each profile's
# primary backend.  Individual fields may be absent when unknown.
LEVEL2_OPTIONAL_FIELDS = [
    "requiredKAlignment",
    "requiredNAlignment",
    "requiredMAlignment",
    "allowedQuantGranularity",
    "requiredActivationQuantMode",
    "requiredWeightQuantMode",
    "maxSupportedRank",
    "requiresStaticShape",
    "requiresConstantWeight",
    "unsupportedOps",
    "fallbackBackend",
]

# Level 3: Preference fields — optional; when present must be arrays or strings.
LEVEL3_OPTIONAL_FIELDS = [
    "acceptableActivationLayouts",
    "acceptableWeightLayouts",
    "preferredDtypes",
    "acceptableDtypes",
    "preferredFusionPatterns",
]

# Level 4: Runtime model fields — optional; when present must be strings.
LEVEL4_OPTIONAL_FIELDS = [
    "runtimeModelLaunchOverhead",
    "runtimeModelDmaTransfer",
    "runtimeModelMemoryHierarchy",
    "runtimeModelOccupancy",
    "runtimeModelRegisterPressure",
    "runtimeModelSharedMemory",
    "runtimeModelSourceLevel",
    "runtimeModelTruthBoundary",
]

# Level 5: Cost model placeholder — optional; when present must be strings.
LEVEL5_OPTIONAL_FIELDS = [
    "costModelKind",
    "costModelInputs",
    "costModelSourceLevel",
    "costModelTruthBoundary",
]

# Cost fields that must be ABSENT when unknown (absent != 0.0).
FORBIDDEN_COST_FIELDS = [
    "layoutTransformCostMs",
    "castCostMs",
    "quantizeCostMs",
    "dequantizeCostMs",
    "requantizeCostMs",
    "backendTransferCostMs",
]


def test_profile(profile_name):
    path = os.path.join(PROFILES_DIR, profile_name)

    # Test 1: JSON parses and required top-level fields are present.
    with open(path) as f:
        data = json.load(f)

    assert "profileId" in data, "missing required top-level field 'profileId'"
    assert "backendCapabilities" in data, "missing required top-level field 'backendCapabilities'"

    # Test 7: representative_devices is present and non-empty.
    assert "representative_devices" in data, "missing 'representative_devices' metadata field"
    assert isinstance(data["representative_devices"], list), \
        "'representative_devices' must be a JSON array"
    assert len(data["representative_devices"]) > 0, \
        "'representative_devices' must be non-empty"

    caps = data["backendCapabilities"]
    assert isinstance(caps, list), "backendCapabilities must be a JSON array"
    assert len(caps) > 0, "backendCapabilities must not be empty"

    # Test 2: Level 1 fields present in all backends.
    for i, cap in enumerate(caps):
        name = cap.get("backendName", f"[{i}]")
        for field in REQUIRED_CAPABILITY_FIELDS:
            assert field in cap, (
                f"backend '{name}': missing required Level 1 field '{field}'"
            )

    # Test 3/4/5: Optional Level 2–5 field types correct when present.
    for i, cap in enumerate(caps):
        name = cap.get("backendName", f"[{i}]")

        for field in LEVEL2_OPTIONAL_FIELDS:
            if field in cap:
                val = cap[field]
                if field in ("requiresStaticShape", "requiresConstantWeight"):
                    assert isinstance(val, bool), (
                        f"backend '{name}': '{field}' must be a bool, got {type(val)}"
                    )
                elif field in ("requiredKAlignment", "requiredNAlignment",
                               "requiredMAlignment", "maxSupportedRank"):
                    assert isinstance(val, (int, float)), (
                        f"backend '{name}': '{field}' must be a number, got {type(val)}"
                    )
                    assert val != 0, (
                        f"backend '{name}': '{field}' = 0 is ambiguous — "
                        f"0 means 'no requirement', absent means 'unknown'. "
                        f"Use null or omit if unknown."
                    )

        for field in LEVEL3_OPTIONAL_FIELDS:
            if field in cap:
                assert isinstance(cap[field], (list, str)), (
                    f"backend '{name}': Level 3 field '{field}' must be a list or string"
                )

        for field in LEVEL4_OPTIONAL_FIELDS:
            if field in cap:
                assert isinstance(cap[field], str), (
                    f"backend '{name}': Level 4 field '{field}' must be a string"
                )

        for field in LEVEL5_OPTIONAL_FIELDS:
            if field in cap:
                assert isinstance(cap[field], str), (
                    f"backend '{name}': Level 5 field '{field}' must be a string"
                )

    # Test 6: Unknown cost fields are absent (absent != 0.0).
    for i, cap in enumerate(caps):
        name = cap.get("backendName", f"[{i}]")
        for cost_field in FORBIDDEN_COST_FIELDS:
            assert cost_field not in cap, (
                f"backend '{name}': cost field '{cost_field}' must be absent "
                f"when unknown — absent means unknown, not 0.0 (free)"
            )

    # Check that each profile has at least one backend with Level 2 data.
    primary = caps[0]
    l2_present = [f for f in LEVEL2_OPTIONAL_FIELDS if f in primary]
    assert len(l2_present) > 0, (
        f"primary backend '{primary.get('backendName', '?')}' has no Level 2 constraint fields — "
        f"expected at least one of: {LEVEL2_OPTIONAL_FIELDS}"
    )

    # Check that Level 4 runtime model source/truth are paired when present.
    for cap in caps:
        name = cap.get("backendName", "?")
        has_source = "runtimeModelSourceLevel" in cap
        has_truth  = "runtimeModelTruthBoundary" in cap
        assert has_source == has_truth, (
            f"backend '{name}': 'runtimeModelSourceLevel' and 'runtimeModelTruthBoundary' "
            f"must both be present or both absent"
        )


def main():
    print("TargetProfileValidationTest (5-level schema):")
    failures = []
    for profile_name in FAMILY_PROFILES:
        try:
            test_profile(profile_name)
            print(f"  [PASS] {profile_name}")
        except (AssertionError, FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"  [FAIL] {profile_name}: {exc}")
            failures.append(profile_name)

    if failures:
        print(f"TargetProfileValidationTest: FAIL ({len(failures)} failure(s))")
        sys.exit(1)
    print("TargetProfileValidationTest: PASS")


if __name__ == "__main__":
    main()
