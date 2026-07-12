#!/usr/bin/env python3
"""Canonical target-profile hardware execution loader for benchmark tooling."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_LOCAL_MEMORY_KINDS = {
    "none",
    "hardware_managed_cache",
    "software_managed_shared_memory",
    "software_managed_scratchpad",
    "explicit_dma_storage",
}


@dataclass(frozen=True)
class TargetHardwareExecutionProfile:
    target_id: str
    profile_kind: str | None
    physical_compute_units: int | None
    effective_compute_units: int | None
    max_concurrent_work_items_per_unit: int | None
    supports_latency_hiding: bool | None
    local_memory_kind: str | None
    provenance: dict[str, str]


@dataclass(frozen=True)
class ResolvedHardwareProfile:
    target_id: str | None
    profile_kind: str | None
    physical_compute_units: int | None
    effective_compute_units: int
    max_concurrent_work_items_per_unit: int | None
    supports_latency_hiding: bool | None
    local_memory_kind: str | None
    effective_compute_units_source: str
    artifact_compute_units: int | None
    compatibility_default: int | None
    identity_validation: str
    provenance: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "profile_kind": self.profile_kind,
            "physical_compute_units": self.physical_compute_units,
            "effective_compute_units": self.effective_compute_units,
            "effective_compute_units_source": self.effective_compute_units_source,
            "max_concurrent_work_items_per_unit": self.max_concurrent_work_items_per_unit,
            "supports_latency_hiding": self.supports_latency_hiding,
            "local_memory_kind": self.local_memory_kind,
            "artifact_compute_units": self.artifact_compute_units,
            "compatibility_default": self.compatibility_default,
            "identity_validation": self.identity_validation,
            "provenance": self.provenance,
        }


def normalize_target_id(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _optional_positive_int(obj: dict[str, Any], field: str) -> int | None:
    if field not in obj:
        return None
    return _positive_int(obj[field], field)


def _optional_bool(obj: dict[str, Any], field: str) -> bool | None:
    if field not in obj:
        return None
    if not isinstance(obj[field], bool):
        raise ValueError(f"{field} must be a boolean")
    return obj[field]


def load_target_hardware_execution_profile(
    path: str | Path,
    expected_target_id: str | None = None,
) -> TargetHardwareExecutionProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profile_id = payload.get("profileId") or payload.get("targetId") or payload.get("target_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("target profile is missing profileId")
    normalized = normalize_target_id(profile_id)
    expected = normalize_target_id(expected_target_id)
    if expected and normalized != expected:
        raise ValueError(f"target profile id mismatch: expected {expected}, found {normalized}")
    profile_kind = payload.get("profileKind") or payload.get("profile_kind")
    if profile_kind is not None and not isinstance(profile_kind, str):
        raise ValueError("profileKind must be a string")

    block = payload.get("hardwareExecutionProfile") or payload.get("hardware_execution_profile") or {}
    if not isinstance(block, dict):
        raise ValueError("hardwareExecutionProfile must be an object when present")

    local_memory_kind = block.get("localMemoryKind", block.get("local_memory_kind"))
    if local_memory_kind is not None:
        if not isinstance(local_memory_kind, str):
            raise ValueError("localMemoryKind must be a string")
        if local_memory_kind not in VALID_LOCAL_MEMORY_KINDS:
            raise ValueError(f"unsupported localMemoryKind: {local_memory_kind}")

    fields = {
        "physical_compute_units": _optional_positive_int(block, "physicalComputeUnits")
        if "physicalComputeUnits" in block
        else _optional_positive_int(block, "physical_compute_units"),
        "effective_compute_units": _optional_positive_int(block, "effectiveComputeUnits")
        if "effectiveComputeUnits" in block
        else _optional_positive_int(block, "effective_compute_units"),
        "max_concurrent_work_items_per_unit": _optional_positive_int(block, "maxConcurrentWorkItemsPerUnit")
        if "maxConcurrentWorkItemsPerUnit" in block
        else _optional_positive_int(block, "max_concurrent_work_items_per_unit"),
        "supports_latency_hiding": _optional_bool(block, "supportsLatencyHiding")
        if "supportsLatencyHiding" in block
        else _optional_bool(block, "supports_latency_hiding"),
    }
    provenance: dict[str, str] = {}
    for name, value in fields.items():
        provenance[name] = "target_profile" if value is not None else "missing"
    provenance["local_memory_kind"] = "target_profile" if local_memory_kind is not None else "missing"
    return TargetHardwareExecutionProfile(
        target_id=normalized or profile_id,
        profile_kind=profile_kind,
        physical_compute_units=fields["physical_compute_units"],
        effective_compute_units=fields["effective_compute_units"],
        max_concurrent_work_items_per_unit=fields["max_concurrent_work_items_per_unit"],
        supports_latency_hiding=fields["supports_latency_hiding"],
        local_memory_kind=local_memory_kind,
        provenance=provenance,
    )


def artifact_compute_units(environment: dict[str, Any]) -> int | None:
    for key in ("effective_compute_units", "sm_count", "num_sms", "compute_units"):
        value = environment.get(key)
        if value is not None:
            return _positive_int(int(value), f"artifact environment {key}")
    return None


def validate_identity(
    profile: TargetHardwareExecutionProfile | None,
    artifact_environment: dict[str, Any],
    artifact_units: int | None,
) -> str:
    if profile is None:
        return "no_target_profile"
    if profile.profile_kind == "synthetic_analytical":
        if artifact_environment.get("gpu_model") or artifact_environment.get("device"):
            return "synthetic_profile_with_real_artifact"
        if artifact_units is not None:
            return "synthetic_profile_with_artifact_units"
    if artifact_units is not None and profile.effective_compute_units is not None:
        if artifact_units != profile.effective_compute_units:
            return "compute_unit_mismatch"
    gpu_model = str(artifact_environment.get("gpu_model") or artifact_environment.get("device") or "")
    capability = artifact_environment.get("compute_capability")
    target_id = profile.target_id
    if gpu_model and "gtx" in target_id and "1650" in target_id:
        model_norm = normalize_target_id(gpu_model) or ""
        if "gtx" not in model_norm or "1650" not in model_norm:
            return "gpu_model_mismatch"
    if capability is not None and "gtx1650" in target_id:
        cap_text = ".".join(str(x) for x in capability) if isinstance(capability, list) else str(capability)
        if cap_text not in {"7.5", "75"}:
            return "compute_capability_mismatch"
    return "matched"


def resolve_effective_compute_units(
    target_profile: TargetHardwareExecutionProfile | None,
    artifact_environment: dict[str, Any] | None,
    cli_override: int | None,
    compatibility_default: int | None,
    allow_cli_override: bool = False,
) -> ResolvedHardwareProfile:
    env = artifact_environment or {}
    artifact_units = artifact_compute_units(env)
    if cli_override is not None:
        cli_override = _positive_int(cli_override, "cli_override")
        if target_profile and target_profile.effective_compute_units is not None:
            if cli_override != target_profile.effective_compute_units and not allow_cli_override:
                raise ValueError(
                    "CLI effective compute-unit override conflicts with target profile; "
                    "pass --allow-hardware-override to use it intentionally"
                )
        source = "cli_override"
        resolved = cli_override
    elif target_profile and target_profile.effective_compute_units is not None:
        source = "target_profile"
        resolved = target_profile.effective_compute_units
    elif artifact_units is not None:
        source = "artifact_environment"
        resolved = artifact_units
    elif compatibility_default is not None:
        source = "compatibility_default"
        resolved = _positive_int(compatibility_default, "compatibility_default")
    else:
        raise ValueError("no effective compute-unit source available")

    identity = validate_identity(target_profile, env, artifact_units)
    if identity.startswith("synthetic_profile"):
        raise ValueError(f"target profile and artifact hardware mismatch: {identity}")
    if identity.endswith("_mismatch") and source != "cli_override":
        raise ValueError(f"target profile and artifact hardware mismatch: {identity}")

    provenance = dict(target_profile.provenance) if target_profile else {}
    provenance["effective_compute_units_resolved"] = source
    return ResolvedHardwareProfile(
        target_id=target_profile.target_id if target_profile else None,
        profile_kind=target_profile.profile_kind if target_profile else None,
        physical_compute_units=target_profile.physical_compute_units if target_profile else None,
        effective_compute_units=resolved,
        max_concurrent_work_items_per_unit=(
            target_profile.max_concurrent_work_items_per_unit if target_profile else None
        ),
        supports_latency_hiding=target_profile.supports_latency_hiding if target_profile else None,
        local_memory_kind=target_profile.local_memory_kind if target_profile else None,
        effective_compute_units_source=source,
        artifact_compute_units=artifact_units,
        compatibility_default=compatibility_default,
        identity_validation=identity,
        provenance=provenance,
    )


def load_and_resolve_hardware_profile(
    target_profile_path: str | Path | None,
    artifact_environment: dict[str, Any] | None,
    cli_override: int | None = None,
    compatibility_default: int | None = 16,
    allow_cli_override: bool = False,
) -> ResolvedHardwareProfile:
    profile = (
        load_target_hardware_execution_profile(target_profile_path)
        if target_profile_path
        else None
    )
    return resolve_effective_compute_units(
        profile,
        artifact_environment,
        cli_override,
        compatibility_default,
        allow_cli_override=allow_cli_override,
    )
