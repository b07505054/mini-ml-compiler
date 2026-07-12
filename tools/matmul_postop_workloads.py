#!/usr/bin/env python3
"""Shared library for the multi-shape MatMul post-op workload suite.

Provides: workload manifest loading/validation, measurement tier budgets,
analytical static cost features, and transparent statistics helpers
(geometric mean, Pearson, Spearman).

All static-cost values are ANALYTICAL features derived from shapes and the
logical kernel structure — never measured hardware traffic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

MANIFEST_SCHEMA = "matmul_postop_workload_manifest"
SUPPORTED_MANIFEST_VERSIONS = (1,)

VALID_CATEGORIES = (
    "representative",
    "fusion_friendly_memory_stress",
    "balanced",
    "fusion_unfriendly_compute_heavy",
    "k_sweep",
    "decision_boundary",
)
VALID_PATTERNS = ("bias", "elementwise_add")
VALID_STATUSES = ("active", "skipped_resource_limit")
VALID_BOUNDARY_EXPECTED_REGIONS = ("v1_win", "tie", "v3_win", "unstable")
EVALUATION_ONLY_FIELDS = {
    "expected_region",
    "measured_winner",
    "final_classification",
    "oracle_latency",
    "oracle_best_kernel",
}

DTYPE_BYTES = {"f32": 4}

# Calibration constant for analytical tier assignment only: measured naive
# 128^3 latency on the remote host was ~1.75 ms => ~2.4 GFLOP/s effective.
NAIVE_EFFECTIVE_GFLOPS = 2.4


@dataclass(frozen=True)
class MeasurementBudget:
    warmup: int
    iterations: int
    repeats: int
    lowered_reason: Optional[str] = None

    def validate(self) -> None:
        if self.warmup <= 0:
            raise ValueError("formal measurement budget requires warmup > 0")
        if self.iterations <= 0 or self.repeats <= 0:
            raise ValueError("iterations and repeats must be positive")
        formal = self.warmup >= 50 and self.iterations >= 300 and self.repeats >= 5
        if not formal and not self.lowered_reason:
            raise ValueError(
                "budgets below warmup>=50/iterations>=300/repeats>=5 must record a lowered_reason"
            )


TIER_BUDGETS = {
    "formal_core": MeasurementBudget(50, 300, 5),
    "extended": MeasurementBudget(
        50, 100, 5, "reduced_iterations_resource_limit_estimated_naive_latency"
    ),
    "resource_heavy": MeasurementBudget(
        10, 30, 3, "reduced_iterations_and_repeats_resource_limit_estimated_naive_latency"
    ),
}


@dataclass
class Workload:
    workload_id: str
    category: str
    m: int
    n: int
    k: int
    dtype: str
    patterns: list[str]
    tile_configs: list[dict[str, int]]
    representative_reason: str
    tier: str
    held_out: bool = False
    status: str = "active"
    skip_reason: Optional[str] = None
    group: Optional[str] = None
    subgroup: Optional[str] = None
    benchmark_purpose: str = "fusion_coverage"
    profile_role: Optional[str] = None
    expected_region: Optional[str] = None
    backend_eligibility: list[str] = field(default_factory=list)
    formal: bool = True

    @property
    def budget(self) -> MeasurementBudget:
        return TIER_BUDGETS[self.tier]

    @property
    def shape_key(self) -> str:
        return f"m{self.m}_n{self.n}_k{self.k}"

    @property
    def is_decision_boundary(self) -> bool:
        return self.category == "decision_boundary" or self.group == "decision_boundary"

    def selection_input(self) -> dict[str, Any]:
        """Label-free shape/candidate input for future compiler selection.

        Evaluation labels such as expected_region, measured winner,
        classifications, and oracle latency intentionally do not appear here.
        """
        return {
            "workload_id": self.workload_id,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "dtype": self.dtype,
            "patterns": list(self.patterns),
            "backend_eligibility": list(self.backend_eligibility),
            "profile_role": self.profile_role,
        }


def postop_shape_for(pattern: str, m: int, n: int) -> list[int]:
    """Bias is [N]; elementwise Add is [M,N]. No semantic mixing."""
    if pattern == "bias":
        return [n]
    if pattern == "elementwise_add":
        return [m, n]
    raise ValueError(f"unknown pattern: {pattern}")


def estimated_naive_latency_ms(m: int, n: int, k: int) -> float:
    return (2.0 * m * n * k) / (NAIVE_EFFECTIVE_GFLOPS * 1e9) * 1000.0


def _validate_workload_entry(entry: dict[str, Any]) -> Workload:
    workload = Workload(
        workload_id=entry["workload_id"],
        category=entry["category"],
        m=entry["m"],
        n=entry["n"],
        k=entry["k"],
        dtype=entry.get("dtype", "f32"),
        patterns=list(entry.get("patterns", [])),
        tile_configs=list(entry.get("tile_configs", [])),
        representative_reason=entry.get("representative_reason", ""),
        tier=entry.get("tier", "formal_core"),
        held_out=bool(entry.get("held_out", False)),
        status=entry.get("status", "active"),
        skip_reason=entry.get("skip_reason"),
        group=entry.get("group"),
        subgroup=entry.get("subgroup"),
        benchmark_purpose=entry.get("benchmark_purpose", "fusion_coverage"),
        profile_role=entry.get("profile_role"),
        expected_region=entry.get("expected_region"),
        backend_eligibility=list(entry.get("backend_eligibility", [])),
        formal=bool(entry.get("formal", True)),
    )
    for dim_name, dim in (("m", workload.m), ("n", workload.n), ("k", workload.k)):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError(f"workload {workload.workload_id}: invalid {dim_name}={dim!r}")
    if workload.category not in VALID_CATEGORIES:
        raise ValueError(f"workload {workload.workload_id}: unknown category {workload.category}")
    if workload.dtype not in DTYPE_BYTES:
        raise ValueError(f"workload {workload.workload_id}: unsupported dtype {workload.dtype}")
    if not workload.patterns or any(p not in VALID_PATTERNS for p in workload.patterns):
        raise ValueError(f"workload {workload.workload_id}: invalid patterns {workload.patterns}")
    if workload.status not in VALID_STATUSES:
        raise ValueError(f"workload {workload.workload_id}: invalid status {workload.status}")
    if workload.is_decision_boundary:
        if workload.benchmark_purpose != "kernel_selection":
            raise ValueError(
                f"workload {workload.workload_id}: decision-boundary workloads require benchmark_purpose=kernel_selection"
            )
        if workload.profile_role != "decision_boundary":
            raise ValueError(
                f"workload {workload.workload_id}: decision-boundary workloads require profile_role=decision_boundary"
            )
        if workload.expected_region not in VALID_BOUNDARY_EXPECTED_REGIONS:
            raise ValueError(
                f"workload {workload.workload_id}: invalid expected_region={workload.expected_region!r}"
            )
        if "triton_cuda" not in workload.backend_eligibility:
            raise ValueError(
                f"workload {workload.workload_id}: decision-boundary workloads require triton_cuda eligibility"
            )
    if workload.status == "skipped_resource_limit" and not workload.skip_reason:
        raise ValueError(
            f"workload {workload.workload_id}: skipped workloads require a structured skip_reason"
        )
    if workload.status == "active":
        if workload.tier not in TIER_BUDGETS:
            raise ValueError(f"workload {workload.workload_id}: unknown tier {workload.tier}")
        workload.budget.validate()
        for tile in workload.tile_configs:
            if any(tile.get(key, 0) <= 0 for key in ("tile_m", "tile_n", "tile_k")):
                raise ValueError(f"workload {workload.workload_id}: invalid tile config {tile}")
        if not workload.tile_configs:
            raise ValueError(f"workload {workload.workload_id}: at least one tile config required")
    return workload


def load_manifest(path: Path | str) -> list[Workload]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"not a workload manifest: schema={payload.get('schema')!r}")
    if payload.get("schema_version") not in SUPPORTED_MANIFEST_VERSIONS:
        raise ValueError(f"unsupported manifest version: {payload.get('schema_version')!r}")
    workloads = [_validate_workload_entry(entry) for entry in payload.get("workloads", [])]
    seen: set[str] = set()
    for workload in workloads:
        if workload.workload_id in seen:
            raise ValueError(f"duplicate workload_id: {workload.workload_id}")
        seen.add(workload.workload_id)
    if not workloads:
        raise ValueError("manifest contains no workloads")
    return workloads


def canonical_workloads(workloads: Sequence[Workload]) -> list[Workload]:
    return [workload for workload in workloads if not workload.is_decision_boundary]


def decision_boundary_workloads(workloads: Sequence[Workload]) -> list[Workload]:
    return [workload for workload in workloads if workload.is_decision_boundary]


def static_cost(pattern: str, m: int, n: int, k: int, dtype: str = "f32") -> dict[str, Any]:
    """Analytical static-cost features. Logical/planned values only —
    never measured hardware traffic, DRAM bytes, or RSS."""
    dtype_bytes = DTYPE_BYTES[dtype]
    macs = m * n * k
    flops = 2 * macs
    output_elements = m * n
    output_bytes = output_elements * dtype_bytes
    postop_elements = n if pattern == "bias" else m * n
    # Logical element accesses in the unfused chain:
    #   matmul writes out; add reads matmul out + postop operand, writes;
    #   relu reads add out, writes final.
    unfused_postop_bytes = (5 * output_elements + postop_elements) * dtype_bytes
    # Fused: postop operand read + single final store.
    fused_postop_bytes = (output_elements + postop_elements) * dtype_bytes
    eliminated = unfused_postop_bytes - fused_postop_bytes  # = 4*M*N*dtype_bytes
    return {
        "matmul_macs": macs,
        "matmul_flops": flops,
        "output_elements": output_elements,
        "output_bytes": output_bytes,
        "logical_intermediate_tensor_count_unfused": 2,
        "logical_intermediate_tensor_count_fused": 0,
        "logical_intermediate_storage_bytes_eliminated": 2 * output_bytes,
        "runtime_dispatch_count_unfused": 3,
        "runtime_dispatch_count_fused": 1,
        "logical_unfused_postop_bytes": unfused_postop_bytes,
        "logical_fused_postop_bytes": fused_postop_bytes,
        "estimated_postop_bytes_eliminated": eliminated,
        "fusion_pressure_score": eliminated / flops,
        "fusion_pressure_score_units": "analytical_bytes_eliminated_per_flop",
        "truth_boundary": "analytical_feature_not_measured_hardware_traffic",
    }


def geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("geometric mean of empty sequence")
    if any(v <= 0 for v in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("pearson requires two equal-length sequences of >= 2 points")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        raise ValueError("pearson undefined for zero-variance input")
    return cov / math.sqrt(var_x * var_y)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for idx in order[i : j + 1]:
            ranks[idx] = average_rank
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(_ranks(xs), _ranks(ys))
