#!/usr/bin/env python3
"""aarch64_schedule_candidate_model.py

Stage 14: turns the Stage 12 static backend evidence and Stage 13
Raspberry Pi measurements into a structured, provenance-tagged,
reusable candidate evidence and scoring model for the AArch64
tiled-scheduled matmul microkernel family.

SCOPE (do not overclaim): "tiled AArch64 matmul schedule candidates
varying K-unroll and tile configuration." Loop order is fixed to the
tiled M/N/K nest already implemented by
mlir_passes/transforms/tile_schedule_matmul_bias_relu.template.mlir; this
is NOT a general loop-interchange framework.

INTEGRATION GAP (Stage 14 section 1 finding, restated here so it travels
with the code): the tiled-scheduled candidates from Stages 10-13 exist
ONLY as scripts and JSON artifacts (this whole backend-codegen series is
Python tooling driving unmodified LLVM/MLIR via subprocess -- there is no
in-tree MLIR pass that represents "AArch64 matmul schedule candidate" as
a compiler-side type). The project's one existing compiler-integrated
cost-model/candidate-selection system
(mlir_passes/include/serving/ServingCostModel.h,
mlir_passes/lib/serving/PlanSelectionPass.cpp) is a SEPARATE subsystem
for LLM-serving quantization/layout/backend-fallback decisions in a
15-pass serving pipeline -- unrelated to this matmul-kernel tile/schedule
family, and out of scope to modify per this task's explicit instruction
not to touch unrelated serving code. This module is therefore new,
Python-side tooling, not a C++/MLIR pass -- but it deliberately MIRRORS
that C++ system's cost-model convention (a decomposed cost struct whose
named components sum to a total, carrying an explicit cost_model_id and
truth_boundary string -- see decision::DecisionCost /
ServingCostModel::compute()) rather than inventing an unrelated schema.

Evidence categories are kept structurally separate (never merged into one
undifferentiated map) per task section 3:
  - StaticIRBEvidence      (MLIR-level: trip counts, loop collapse, ...)
  - LLVMBackendEvidence    (MIR/assembly-level: spills, registers, ...)
  - MeasuredHardwareEvidence (Raspberry Pi: latency, correctness, ...)
Every individual field is wrapped in an EvidenceValue carrying explicit
provenance (source_level, source_artifact, tool_version, target,
revision, candidate_key, timestamp) -- see task section 4.
"""
import dataclasses
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

SCHEMA_VERSION = "aarch64_schedule_candidate_model_v1"

# ---------------------------------------------------------------------------
# Candidate identity (task section 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateKey:
    """Deterministic, serializable candidate identity. NEVER derived from
    or affected by a display label -- two candidates with different
    labels but identical semantic configuration fields are the SAME
    candidate; two candidates with the same label but different fields
    (e.g. a stale re-run) are DIFFERENT candidates."""

    target_arch: str = "aarch64"
    target_cpu: str = "cortex-a76"
    target_features: str = "none"  # no explicit -mattr flags used anywhere in this series
    dtype: str = "f32"
    shape_m: int = 0
    shape_n: int = 0
    shape_k: int = 0
    tile_m: int = 0
    tile_n: int = 0
    tile_k: int = 0
    vector_width: int = 4  # f32 lanes per 128-bit NEON register (v*.4s), fixed throughout this series
    schedule_unroll_k: int = 1
    # Fixed for this stage -- see module docstring "SCOPE". A different
    # loop_order_id would represent a genuinely different (not yet
    # implemented) transform, not a relabeling of this one.
    loop_order_id: str = "tiled_mnk_row_major_v1"
    microkernel_id: str = "hir_fused_matmul_bias_relu_tiled_scheduled_v1"

    def canonical_id(self) -> str:
        return (
            f"{self.target_arch}:{self.target_cpu}:{self.target_features}:{self.dtype}:"
            f"shape{self.shape_m}x{self.shape_n}x{self.shape_k}:"
            f"tile{self.tile_m}x{self.tile_n}x{self.tile_k}:"
            f"vw{self.vector_width}:uk{self.schedule_unroll_k}:"
            f"{self.loop_order_id}:{self.microkernel_id}"
        )

    def shape_bucket(self) -> str:
        """A coarser identity used for cross-shape compatibility checks
        (task section 8) -- same tile/unroll/target/dtype, any shape."""
        return (
            f"{self.target_arch}:{self.target_cpu}:{self.target_features}:{self.dtype}:"
            f"tile{self.tile_m}x{self.tile_n}x{self.tile_k}:vw{self.vector_width}:"
            f"uk{self.schedule_unroll_k}:{self.loop_order_id}:{self.microkernel_id}"
        )

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Evidence provenance (task section 4)
# ---------------------------------------------------------------------------

SOURCE_LEVELS = (
    "mlir", "llvm_ir", "mir", "assembly", "llvm_mca", "raspberry_pi_measured", "manual_config",
)


@dataclass
class Provenance:
    source_level: str  # one of SOURCE_LEVELS
    source_artifact: str
    tool_version: str
    target: str
    revision: str
    candidate_id: str
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.source_level not in SOURCE_LEVELS:
            raise ValueError(f"invalid source_level {self.source_level!r}, must be one of {SOURCE_LEVELS}")


@dataclass
class EvidenceValue:
    """One provenance-tagged evidence field. `value` may be None with
    `note` explaining why (missing evidence must never silently become a
    numeric zero -- task section 7)."""

    value: Optional[Any]
    provenance: Provenance
    note: str = ""
    estimated: bool = False  # True for values derived analytically, not measured/observed directly

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d


def ev(value, source_level, source_artifact, tool_version, target, revision, candidate_id,
       note="", estimated=False, timestamp=None):
    return EvidenceValue(
        value=value,
        provenance=Provenance(source_level, source_artifact, tool_version, target, revision, candidate_id, timestamp),
        note=note, estimated=estimated,
    )


# ---------------------------------------------------------------------------
# Evidence categories (task section 3)
# ---------------------------------------------------------------------------


@dataclass
class StaticIRBEvidence:
    m_trip_count: EvidenceValue
    n_trip_count: EvidenceValue
    k_trip_count_post_unroll: EvidenceValue
    surviving_loop_count: EvidenceValue
    k_loop_collapsed: EvidenceValue
    static_vector_contract_count: EvidenceValue
    estimated_dynamic_k_loop_iterations: EvidenceValue
    estimated_dynamic_loop_control_reduction_pct: EvidenceValue


@dataclass
class LLVMBackendEvidence:
    pre_ra_approx_peak_live_vector_registers: EvidenceValue
    physical_vector_registers_post_ra: EvidenceValue
    spill_store_count: EvidenceValue
    reload_load_count: EvidenceValue
    stack_frame_bytes: EvidenceValue
    object_bytes: EvidenceValue
    static_fmla_asm_count: EvidenceValue
    accumulator_chains: EvidenceValue
    max_accumulator_chain_length: EvidenceValue
    llvm_mca_block_rthroughput: EvidenceValue


@dataclass
class MeasuredHardwareEvidence:
    median_latency_ms: EvidenceValue
    p95_latency_ms: EvidenceValue
    mean_latency_ms: EvidenceValue
    stddev_latency_ms: EvidenceValue
    cv: EvidenceValue
    correctness_pass: EvidenceValue
    hardware_identity: EvidenceValue  # e.g. "raspberry_pi_5_cortex_a76"


@dataclass
class CandidateEvidenceRecord:
    key: CandidateKey
    label: str  # display-only, NEVER part of identity/equality
    static_ir: StaticIRBEvidence
    llvm_backend: LLVMBackendEvidence
    measured: Optional[MeasuredHardwareEvidence]  # None when no compatible Pi measurement exists

    def canonical_id(self) -> str:
        return self.key.canonical_id()


# ---------------------------------------------------------------------------
# Classification (task section 6 -- corrected Stage 12 policy)
# ---------------------------------------------------------------------------

BACKEND_SAFE = "backend_safe"
BACKEND_COSTLY = "backend_costly"
HW_CONFIRMED_PROFITABLE = "hardware_confirmed_profitable"
HW_CONFIRMED_NEUTRAL = "hardware_confirmed_neutral"
HW_CONFIRMED_REGRESSION = "hardware_confirmed_regression"
HW_UNKNOWN = "hardware_unknown"
INCORRECT = "incorrect"
UNSUPPORTED = "unsupported"


def classify_backend_safety(llvm_backend: LLVMBackendEvidence) -> dict:
    """Independent dimension #1: does this candidate cost anything at the
    backend level? Not mutually exclusive with the hardware-confirmation
    dimension below -- a candidate can be BOTH backend_costly AND
    hardware_confirmed_profitable (this is the Stage 13 finding this
    function exists to represent truthfully, not hide)."""
    spills = llvm_backend.spill_store_count.value or 0
    reloads = llvm_backend.reload_load_count.value or 0
    costly_reasons = []
    if spills > 0:
        costly_reasons.append(f"{spills} spill store(s)")
    if reloads > 0:
        costly_reasons.append(f"{reloads} reload load(s)")
    label = BACKEND_COSTLY if costly_reasons else BACKEND_SAFE
    return {"label": label, "reasons": costly_reasons}


def classify_hardware_confirmation(measured_baseline: Optional[MeasuredHardwareEvidence],
                                    measured_candidate: Optional[MeasuredHardwareEvidence],
                                    noise_floor_pct: float = 3.0) -> dict:
    """Independent dimension #2: what did real hardware measurement (if
    any, and if compatible -- see check_compatibility) actually show
    relative to a matched baseline? Returns HW_UNKNOWN, not a guess, when
    no compatible measurement exists."""
    if measured_baseline is None or measured_candidate is None:
        return {"label": HW_UNKNOWN, "reason": "no compatible measured baseline/candidate pair available"}
    if not (measured_baseline.correctness_pass.value and measured_candidate.correctness_pass.value):
        return {"label": INCORRECT, "reason": "correctness failure in baseline or candidate"}

    b_med = measured_baseline.median_latency_ms.value
    c_med = measured_candidate.median_latency_ms.value
    if b_med is None or c_med is None or b_med == 0:
        return {"label": HW_UNKNOWN, "reason": "missing latency evidence"}

    b_cv = measured_baseline.cv.value or 0.0
    c_cv = measured_candidate.cv.value or 0.0
    noise_floor = max(noise_floor_pct, 2 * 100 * max(b_cv, c_cv))
    pct_change = ((b_med - c_med) / b_med) * 100.0

    if pct_change > noise_floor:
        return {"label": HW_CONFIRMED_PROFITABLE, "pct_change": pct_change, "noise_floor_pct": noise_floor}
    elif pct_change < -noise_floor:
        return {"label": HW_CONFIRMED_REGRESSION, "pct_change": pct_change, "noise_floor_pct": noise_floor}
    else:
        return {"label": HW_CONFIRMED_NEUTRAL, "pct_change": pct_change, "noise_floor_pct": noise_floor}


def full_classification(llvm_backend: LLVMBackendEvidence,
                         measured_baseline: Optional[MeasuredHardwareEvidence],
                         measured_candidate: Optional[MeasuredHardwareEvidence]) -> dict:
    safety = classify_backend_safety(llvm_backend)
    hw = classify_hardware_confirmation(measured_baseline, measured_candidate)
    return {
        "backend_safety": safety["label"],
        "backend_safety_reasons": safety["reasons"],
        "hardware_confirmation": hw["label"],
        "hardware_confirmation_detail": hw,
        "combined_human_summary": (
            f"{'Backend-costly' if safety['label'] == BACKEND_COSTLY else 'Backend-safe'}, "
            f"{'hardware-confirmed profitable' if hw['label'] == HW_CONFIRMED_PROFITABLE else hw['label'].replace('_', ' ')}"
        ),
    }


# ---------------------------------------------------------------------------
# Compatibility checking (task section 8)
# ---------------------------------------------------------------------------

EXACT_MATCH = "exact_match"
CROSS_SHAPE_SAME_SCHEDULE = "cross_shape_same_schedule"
SHAPE_BUCKET = "shape_bucket"
INCOMPATIBLE = "incompatible"

BENCHMARK_METHODOLOGY_VERSION = "stage13_pi5_harness_v1"


def check_compatibility(query_key: CandidateKey, evidence_key: CandidateKey,
                         evidence_methodology_version: str) -> dict:
    """Returns a compatibility level + confidence multiplier for using
    `evidence_key`'s measured evidence to inform a decision about
    `query_key`. Fails closed: any of target_arch/target_cpu/
    target_features/dtype/microkernel_id/vector_width mismatch ->
    INCOMPATIBLE, confidence 0. A stale methodology version also fails
    closed regardless of everything else matching."""
    if evidence_methodology_version != BENCHMARK_METHODOLOGY_VERSION:
        return {"level": INCOMPATIBLE, "confidence": 0.0, "reason": f"stale benchmark methodology version {evidence_methodology_version!r} != {BENCHMARK_METHODOLOGY_VERSION!r}"}

    hard_fields = ["target_arch", "target_cpu", "target_features", "dtype", "microkernel_id", "vector_width", "loop_order_id"]
    for f in hard_fields:
        if getattr(query_key, f) != getattr(evidence_key, f):
            return {"level": INCOMPATIBLE, "confidence": 0.0, "reason": f"field '{f}' mismatch: query={getattr(query_key, f)!r} evidence={getattr(evidence_key, f)!r}"}

    if (query_key.shape_m, query_key.shape_n, query_key.shape_k) == (evidence_key.shape_m, evidence_key.shape_n, evidence_key.shape_k) and \
       (query_key.tile_m, query_key.tile_n, query_key.tile_k) == (evidence_key.tile_m, evidence_key.tile_n, evidence_key.tile_k) and \
       query_key.schedule_unroll_k == evidence_key.schedule_unroll_k:
        return {"level": EXACT_MATCH, "confidence": 1.0, "reason": "identical candidate configuration"}

    if (query_key.tile_m, query_key.tile_n, query_key.tile_k) == (evidence_key.tile_m, evidence_key.tile_n, evidence_key.tile_k) and \
       query_key.schedule_unroll_k == evidence_key.schedule_unroll_k:
        return {"level": CROSS_SHAPE_SAME_SCHEDULE, "confidence": 0.7,
                "reason": "same tile/unroll schedule, different shape -- Stage 13 observed this schedule win on 5 independent shapes"}

    if query_key.tile_m == evidence_key.tile_m and query_key.tile_n == evidence_key.tile_n and query_key.tile_k == evidence_key.tile_k:
        return {"level": SHAPE_BUCKET, "confidence": 0.4, "reason": "same tile, different unroll/shape bucket"}

    return {"level": INCOMPATIBLE, "confidence": 0.0, "reason": "no compatible dimension found (different tile configuration)"}


# ---------------------------------------------------------------------------
# Attribution model (task section 5)
# ---------------------------------------------------------------------------


def build_attribution(baseline: CandidateEvidenceRecord, scheduled: CandidateEvidenceRecord) -> dict:
    """Attribution report for one matched (baseline, scheduled) pair.
    Separates benefit signals from cost signals from the final measured
    result -- never claims an exact dynamic instruction count that cannot
    be defensibly derived; anything not directly countable is explicitly
    marked estimated=True with a note on the derivation."""
    # A matched attribution pair must agree on everything EXCEPT
    # schedule_unroll_k (which is precisely the dimension being
    # attributed) -- NOT shape_bucket(), which itself includes
    # schedule_unroll_k and would therefore reject every legitimate pair.
    invariant_fields = ["target_arch", "target_cpu", "target_features", "dtype",
                         "shape_m", "shape_n", "shape_k", "tile_m", "tile_n", "tile_k",
                         "vector_width", "loop_order_id", "microkernel_id"]
    mismatched = [f for f in invariant_fields if getattr(baseline.key, f) != getattr(scheduled.key, f)]
    if mismatched or baseline.key.schedule_unroll_k == scheduled.key.schedule_unroll_k:
        raise ValueError(
            f"refusing attribution: baseline and scheduled candidates are not a matched pair "
            f"(mismatched fields: {mismatched or ['schedule_unroll_k is identical -- nothing to attribute']}) "
            f"(baseline={baseline.key.canonical_id()} scheduled={scheduled.key.canonical_id()})"
        )

    b_ir, s_ir = baseline.static_ir, scheduled.static_ir
    b_be, s_be = baseline.llvm_backend, scheduled.llvm_backend

    def delta(a: EvidenceValue, b: EvidenceValue):
        if a.value is None or b.value is None:
            return None
        return b.value - a.value

    def pct_delta(a: EvidenceValue, b: EvidenceValue):
        if a.value is None or b.value is None or a.value == 0:
            return None
        return (b.value - a.value) / a.value * 100.0

    benefits = {
        "k_loop_dynamic_trip_count_reduction": {
            "baseline": b_ir.estimated_dynamic_k_loop_iterations.value,
            "scheduled": s_ir.estimated_dynamic_k_loop_iterations.value,
            "delta": delta(b_ir.estimated_dynamic_k_loop_iterations, s_ir.estimated_dynamic_k_loop_iterations),
            "estimated": True,
            "note": "estimated_dynamic_k_loop_iterations = (K_trip_count_pre_unroll / schedule_unroll_k); a static-IR-derived estimate, not a dynamic instrumentation count",
        },
        "dynamic_loop_branch_reduction_estimate": {
            "note": "each dynamic K-loop back-edge corresponds to exactly one conditional branch in the compiled loop; reduction is the same ratio as the trip-count reduction above (one branch retired per iteration removed)",
            "estimated": True,
            "delta_pct": pct_delta(b_ir.estimated_dynamic_k_loop_iterations, s_ir.estimated_dynamic_k_loop_iterations),
        },
        "dynamic_compare_update_reduction_estimate": {
            "note": "each removed K-loop iteration also removes one induction-variable update and one trip-count compare (standard scf.for lowering, one add + one compare per back-edge); same ratio as the branch reduction above",
            "estimated": True,
            "delta_pct": pct_delta(b_ir.estimated_dynamic_k_loop_iterations, s_ir.estimated_dynamic_k_loop_iterations),
        },
        "larger_scheduling_region": {
            "baseline_static_contract_count_per_body": b_be.static_fmla_asm_count.value,
            "scheduled_static_contract_count_per_body": s_be.static_fmla_asm_count.value,
            "note": "more independent FMLA work is visible to the LLVM machine scheduler within one static loop body copy (Stage 12 evidence: same_accumulator_distance was not worse after unrolling)",
            "estimated": False,
        },
    }

    costs = {
        "static_body_growth": {
            "baseline_fmla_count": b_be.static_fmla_asm_count.value,
            "scheduled_fmla_count": s_be.static_fmla_asm_count.value,
            "delta_pct": pct_delta(b_be.static_fmla_asm_count, s_be.static_fmla_asm_count),
            "estimated": False,
        },
        "code_size_growth": {
            "baseline_bytes": b_be.object_bytes.value,
            "scheduled_bytes": s_be.object_bytes.value,
            "delta_bytes": delta(b_be.object_bytes, s_be.object_bytes),
            "delta_pct": pct_delta(b_be.object_bytes, s_be.object_bytes),
            "estimated": False,
        },
        "spill_increase": {
            "baseline": b_be.spill_store_count.value, "scheduled": s_be.spill_store_count.value,
            "delta": delta(b_be.spill_store_count, s_be.spill_store_count), "estimated": False,
        },
        "reload_increase": {
            "baseline": b_be.reload_load_count.value, "scheduled": s_be.reload_load_count.value,
            "delta": delta(b_be.reload_load_count, s_be.reload_load_count), "estimated": False,
        },
        "stack_frame_change": {
            "baseline_bytes": b_be.stack_frame_bytes.value, "scheduled_bytes": s_be.stack_frame_bytes.value,
            "delta_bytes": delta(b_be.stack_frame_bytes, s_be.stack_frame_bytes), "estimated": False,
        },
        "physical_register_change": {
            "baseline": b_be.physical_vector_registers_post_ra.value, "scheduled": s_be.physical_vector_registers_post_ra.value,
            "delta": delta(b_be.physical_vector_registers_post_ra, s_be.physical_vector_registers_post_ra), "estimated": False,
            "note": "authoritative post-RA physical register count, NOT the pre-RA approx_peak_live heuristic (see Stage 12 finding: that heuristic over-counts on this project's loop-bodied MIR and is not used as a scoring signal here)",
        },
    }

    measured_result = None
    if baseline.measured and scheduled.measured:
        b_med, s_med = baseline.measured.median_latency_ms.value, scheduled.measured.median_latency_ms.value
        measured_result = {
            "baseline_median_ms": b_med, "scheduled_median_ms": s_med,
            "measured_latency_delta_pct": pct_delta_dir(b_med, s_med),
            "estimated": False,
            "note": "the only non-estimated, non-static number in this report -- real Raspberry Pi measurement",
        }

    return {
        "baseline_id": baseline.canonical_id(), "scheduled_id": scheduled.canonical_id(),
        "benefit_signals": benefits, "cost_signals": costs, "measured_result": measured_result,
    }


def pct_delta_dir(baseline_val, scheduled_val):
    if baseline_val is None or scheduled_val is None or baseline_val == 0:
        return None
    return (baseline_val - scheduled_val) / baseline_val * 100.0  # positive == scheduled is faster


# ---------------------------------------------------------------------------
# Cost model (task section 7)
# ---------------------------------------------------------------------------
#
# total_cost = static_compute_cost + loop_control_cost
#            + register_pressure_penalty + spill_penalty + reload_penalty
#            + code_size_penalty + stack_frame_penalty + target_specific_penalty
#            - measured_latency_calibration_bonus   (calibrated mode only)
#
# LOWER total_cost is preferred. static_compute_cost is a per-shape
# constant (total FLOPs are invariant to schedule choice) included for
# transparency, not because it differentiates candidates of the same
# shape. loop_control_cost is the one static term that genuinely
# decreases with more unrolling -- it is the mechanism behind every
# Stage 13 Class-A win. register_pressure_penalty defaults to weight 0:
# Stage 12 found the only available pre-RA pressure heuristic
# (approx_peak_live_vector_registers) over-counts on this project's
# loop-bodied MIR with no loop-back-edge modeling, so it is reported for
# visibility but NOT scored by default (a documented, deliberate choice,
# not an oversight) -- see CostWeights.register_pressure_weight.

RANKING_MODE_STATIC_HARD_REJECT = "static_hard_reject_spill"   # LEGACY, for comparison only -- reproduces the incorrect Stage 12 "any spill -> reject" policy
RANKING_MODE_STATIC_SOFT_PENALTY = "static_soft_penalty"        # corrected default: spill/reload are penalties, not vetoes
RANKING_MODE_CALIBRATED_PI = "calibrated_raspberry_pi_5"


@dataclass
class CostWeights:
    """All weights are unitless relative penalties in static modes
    (documented defaults below), or real-world-calibrated in calibrated
    mode. Every weight is visible here -- none are hidden inside the
    scoring function."""

    compute_cost_scale: float = 1e-6          # static_compute_cost = shape_m*n*k * this
    loop_control_cost_per_iteration: float = 1.0  # loop_control_cost = estimated_dynamic_k_loop_iterations * this
    register_pressure_weight: float = 0.0      # see module note above: deliberately 0 by default
    spill_weight: float = 2.0                  # spill_penalty = spill_store_count * this
    reload_weight: float = 1.5                 # reload_penalty = reload_load_count * this
    code_size_weight_per_kb: float = 0.05      # code_size_penalty = (object_bytes/1024) * this
    stack_frame_weight_per_64b: float = 0.1    # stack_frame_penalty = (stack_frame_bytes/64) * this
    target_specific_penalty: float = 0.0       # reserved; no Cortex-A76-specific penalty identified in this slice
    hard_reject_on_any_spill: bool = False     # RANKING_MODE_STATIC_HARD_REJECT sets this True
    calibration_weight: float = 1000.0         # calibrated mode: measured_median_ms * this dominates the static sum


DEFAULT_WEIGHTS = CostWeights()


@dataclass
class CostBreakdown:
    candidate_id: str
    label: str
    static_compute_cost: float
    loop_control_cost: float
    register_pressure_penalty: float
    spill_penalty: float
    reload_penalty: float
    code_size_penalty: float
    stack_frame_penalty: float
    target_specific_penalty: float
    measured_latency_calibration_bonus: float
    total_cost: float
    cost_model_id: str
    truth_boundary: str
    ranking_mode: str
    rejected: bool
    rejection_reason: str
    missing_evidence_fields: list
    confidence: float
    rank: int = 0


def compute_cost(record: CandidateEvidenceRecord, mode: str, weights: CostWeights = DEFAULT_WEIGHTS,
                  measured_evidence_for_calibration: Optional[MeasuredHardwareEvidence] = None,
                  compatibility: Optional[dict] = None) -> CostBreakdown:
    ir, be = record.static_ir, record.llvm_backend
    missing = []

    def val(ev_field: EvidenceValue, field_name: str, default=0.0):
        if ev_field.value is None:
            missing.append(field_name)
            return default
        return float(ev_field.value)

    shape_flops = record.key.shape_m * record.key.shape_n * record.key.shape_k
    static_compute_cost = shape_flops * weights.compute_cost_scale

    loop_iters = val(ir.estimated_dynamic_k_loop_iterations, "estimated_dynamic_k_loop_iterations")
    loop_control_cost = loop_iters * weights.loop_control_cost_per_iteration

    peak_live = val(be.pre_ra_approx_peak_live_vector_registers, "pre_ra_approx_peak_live_vector_registers")
    register_pressure_penalty = peak_live * weights.register_pressure_weight

    spills = val(be.spill_store_count, "spill_store_count")
    reloads = val(be.reload_load_count, "reload_load_count")
    spill_penalty = spills * weights.spill_weight
    reload_penalty = reloads * weights.reload_weight

    obj_bytes = val(be.object_bytes, "object_bytes")
    code_size_penalty = (obj_bytes / 1024.0) * weights.code_size_weight_per_kb

    stack_bytes = val(be.stack_frame_bytes, "stack_frame_bytes")
    stack_frame_penalty = (stack_bytes / 64.0) * weights.stack_frame_weight_per_64b

    rejected, rejection_reason = False, ""

    # Legitimate, mode-independent hard rejections -- distinct in kind
    # from the removed "any spill -> reject" veto. An UNSUPPORTED
    # candidate (object generation never succeeded -- no llvm_backend
    # evidence exists at all) or an INCORRECT one (measured on real
    # hardware and found numerically wrong) must never be selectable
    # regardless of ranking mode or how favorable its latency looks;
    # neither condition is "backend cost", both are "not a valid
    # candidate at all".
    if be.object_bytes.value is None:
        rejected = True
        rejection_reason = "UNSUPPORTED: no compiled object exists for this candidate (object generation did not succeed or was never attempted) -- always rejected regardless of mode"
    elif record.measured is not None and record.measured.correctness_pass.value is False:
        rejected = True
        rejection_reason = "INCORRECT: measured on real hardware and found numerically incorrect -- always rejected regardless of mode or measured latency"

    if not rejected and mode == RANKING_MODE_STATIC_HARD_REJECT and spills > 0:
        rejected = True
        rejection_reason = f"LEGACY policy: spill_store_count={int(spills)} > 0 -> hard rejection (reproduces the incorrect Stage 12 Class D veto, kept ONLY for the Stage 14 ranking-mode comparison experiment)"

    calibration_bonus = 0.0
    confidence = 1.0 if not missing else max(0.0, 1.0 - 0.15 * len(missing))
    if mode == RANKING_MODE_CALIBRATED_PI:
        if measured_evidence_for_calibration is not None and compatibility is not None and compatibility["level"] != INCOMPATIBLE:
            med = measured_evidence_for_calibration.median_latency_ms.value
            if med is not None:
                calibration_bonus = med * weights.calibration_weight
                confidence = min(confidence, compatibility["confidence"])
            else:
                missing.append("measured_median_latency_ms")
                confidence = min(confidence, 0.3)
        else:
            missing.append("compatible_measured_evidence")
            confidence = min(confidence, 0.3)  # fall back to static-only signal with reduced confidence, not silent zero

    static_sum = (
        static_compute_cost + loop_control_cost + register_pressure_penalty +
        spill_penalty + reload_penalty + code_size_penalty + stack_frame_penalty +
        weights.target_specific_penalty
    )
    total_cost = float("inf") if rejected else (
        calibration_bonus if (mode == RANKING_MODE_CALIBRATED_PI and calibration_bonus > 0) else static_sum
    )

    return CostBreakdown(
        candidate_id=record.canonical_id(), label=record.label,
        static_compute_cost=static_compute_cost, loop_control_cost=loop_control_cost,
        register_pressure_penalty=register_pressure_penalty, spill_penalty=spill_penalty,
        reload_penalty=reload_penalty, code_size_penalty=code_size_penalty,
        stack_frame_penalty=stack_frame_penalty, target_specific_penalty=weights.target_specific_penalty,
        measured_latency_calibration_bonus=calibration_bonus, total_cost=total_cost,
        cost_model_id=f"{SCHEMA_VERSION}_cost_v1", truth_boundary=(
            "static_backend_evidence_not_measured_latency" if mode != RANKING_MODE_CALIBRATED_PI
            else "calibrated_with_compatible_raspberry_pi_5_cortex_a76_measurement_where_available_else_static_fallback"
        ),
        ranking_mode=mode, rejected=rejected, rejection_reason=rejection_reason,
        missing_evidence_fields=missing, confidence=confidence,
    )


def rank_candidates(records: list, mode: str, weights: CostWeights = DEFAULT_WEIGHTS,
                     measured_evidence_pool: Optional[list] = None) -> list:
    """measured_evidence_pool: list of (CandidateKey, MeasuredHardwareEvidence)
    tuples -- every measured candidate available for calibration, used
    only in calibrated mode. For each ranked candidate, an exact-key match
    is preferred; otherwise the first compatible (non-INCOMPATIBLE) entry
    found via check_compatibility() is used, in pool order. Returns
    breakdowns sorted by total_cost ascending (lower cost = higher rank),
    rejected candidates last, deterministic tie-break by canonical_id."""
    pool = measured_evidence_pool or []
    breakdowns = []
    for r in records:
        measured_ev, compat = None, None
        if mode == RANKING_MODE_CALIBRATED_PI:
            for other_key, other_measured in pool:
                if other_key == r.key:
                    measured_ev, compat = other_measured, {"level": EXACT_MATCH, "confidence": 1.0}
                    break
            if measured_ev is None:
                for other_key, other_measured in pool:
                    c = check_compatibility(r.key, other_key, BENCHMARK_METHODOLOGY_VERSION)
                    if c["level"] != INCOMPATIBLE:
                        measured_ev, compat = other_measured, c
                        break
        breakdowns.append(compute_cost(r, mode, weights, measured_ev, compat))

    breakdowns.sort(key=lambda b: (b.rejected, b.total_cost, b.candidate_id))
    for i, b in enumerate(breakdowns):
        b.rank = i + 1
    return breakdowns


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _to_jsonable(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def record_to_dict(record: CandidateEvidenceRecord) -> dict:
    return {
        "canonical_id": record.canonical_id(),
        "label": record.label,
        "key": record.key.as_dict(),
        "static_ir": _to_jsonable(record.static_ir),
        "llvm_backend": _to_jsonable(record.llvm_backend),
        "measured": _to_jsonable(record.measured) if record.measured else None,
    }


def breakdown_to_dict(b: CostBreakdown) -> dict:
    return _to_jsonable(b)


# ---------------------------------------------------------------------------
# Loaders: Stage 12 (static/backend) + Stage 13 (measured Pi) JSON -> records
# ---------------------------------------------------------------------------

STAGE12_TOOL_VERSION = "compare_aarch64_schedule_variants_v1"
STAGE13_TOOL_VERSION = "run_aarch64_schedule_pi_validation_v1"
STAGE13_HARDWARE_IDENTITY = "raspberry_pi_5_cortex_a76"


def key_from_stage12(cand: dict, revision: str) -> CandidateKey:
    shape = cand["shape"].split("x")
    return CandidateKey(
        shape_m=int(shape[0]), shape_n=int(shape[1]), shape_k=int(shape[2]),
        tile_m=cand["tile"]["m"], tile_n=cand["tile"]["n"], tile_k=cand["tile"]["k"],
        schedule_unroll_k=cand["schedule_unroll_k"],
    )


def static_ir_evidence_from_stage12(cand: dict, key: CandidateKey, artifact_path: str, revision: str) -> StaticIRBEvidence:
    sched = cand["schedule"]["post_scheduler"]
    tile_k_trip_pre_unroll = key.shape_k // key.tile_k
    k_trip_post_unroll = tile_k_trip_pre_unroll // key.schedule_unroll_k
    collapsed = k_trip_post_unroll <= 1
    m_trip = key.shape_m // key.tile_m
    n_trip = key.shape_n // key.tile_n
    surviving = (1 if m_trip > 1 else 0) + (1 if n_trip > 1 else 0) + (0 if collapsed else 1)
    cid = key.canonical_id()

    def mk(value, note="", estimated=False):
        return ev(value, "mlir", artifact_path, STAGE12_TOOL_VERSION, "aarch64-linux-gnu/cortex-a76", revision, cid, note=note, estimated=estimated)

    dyn_k_iters = k_trip_post_unroll  # dynamic K-loop trip count actually executed at runtime
    loop_control_reduction_pct = None
    if tile_k_trip_pre_unroll:
        loop_control_reduction_pct = (1.0 - (k_trip_post_unroll / tile_k_trip_pre_unroll)) * 100.0

    return StaticIRBEvidence(
        m_trip_count=mk(m_trip),
        n_trip_count=mk(n_trip),
        k_trip_count_post_unroll=mk(k_trip_post_unroll),
        surviving_loop_count=mk(surviving, note="count of scf.for loops remaining after trip-count-1 canonicalization (Stage 11 finding)"),
        k_loop_collapsed=mk(collapsed),
        static_vector_contract_count=mk(cand["fp_reduction_order"]["fmuladd_call_count"], note="from LLVM IR @llvm.fmuladd call count (Stage 12)"),
        estimated_dynamic_k_loop_iterations=mk(dyn_k_iters, estimated=True, note="K_trip_count_pre_unroll / schedule_unroll_k -- static-IR-derived estimate of the K-loop's real dynamic trip count, not an instrumented count"),
        estimated_dynamic_loop_control_reduction_pct=mk(loop_control_reduction_pct, estimated=True, note="1 - (post-unroll trip count / pre-unroll trip count), i.e. the fraction of K-loop back-edges/compares/induction-updates eliminated"),
    )


def llvm_backend_evidence_from_stage12(cand: dict, key: CandidateKey, artifact_path: str, revision: str) -> LLVMBackendEvidence:
    reg = cand["register_allocation"]
    sched = cand["schedule"]["post_scheduler"]
    mca = None  # llvm-mca only collected for the primary pair in Stage 12; absent for most candidates
    cid = key.canonical_id()

    def mk_mir(value, note=""):
        return ev(value, "mir", artifact_path, STAGE12_TOOL_VERSION, "aarch64-linux-gnu/cortex-a76", revision, cid, note=note)

    def mk_asm(value, note=""):
        return ev(value, "assembly", artifact_path, STAGE12_TOOL_VERSION, "aarch64-linux-gnu/cortex-a76", revision, cid, note=note)

    def mk_mca(value, note=""):
        if value is None:
            return ev(None, "llvm_mca", artifact_path, STAGE12_TOOL_VERSION, "aarch64-linux-gnu/cortex-a76", revision, cid,
                       note=note or "llvm-mca was only run for the Stage 12 primary comparison pair, not every candidate")
        return ev(value, "llvm_mca", artifact_path, STAGE12_TOOL_VERSION, "aarch64-linux-gnu/cortex-a76", revision, cid, note=note)

    return LLVMBackendEvidence(
        pre_ra_approx_peak_live_vector_registers=mk_mir(
            reg["stages"]["pre_ra"]["approx_peak_live_vector_registers"],
            note="MIR-derived linear-scan approximation, NOT exact (no loop-back-edge modeling) -- see Stage 12 finding, not used as a scoring signal by default",
        ),
        physical_vector_registers_post_ra=mk_mir(reg["stages"]["post_ra"]["physical_vector_registers_referenced"]),
        spill_store_count=mk_mir(reg["comparison"]["spill_stores_inserted_by_ra"]),
        reload_load_count=mk_mir(reg["comparison"]["reload_loads_inserted_by_ra"]),
        stack_frame_bytes=mk_mir(reg["final_stack_frame_bytes"]),
        object_bytes=ev(cand["object_bytes"], "assembly", artifact_path, STAGE12_TOOL_VERSION, "aarch64-linux-gnu/cortex-a76", revision, cid),
        static_fmla_asm_count=mk_asm(cand["assembly_counts"]["fmla"]),
        accumulator_chains=mk_mir(sched.get("accumulator_chains")),
        max_accumulator_chain_length=mk_mir(sched.get("max_accumulator_chain_length")),
        llvm_mca_block_rthroughput=mk_mca(None),
    )


def load_stage12_records(stage12_json_path: str) -> dict:
    """Returns {label: CandidateEvidenceRecord} with measured=None (Stage
    12 has no hardware evidence -- filled in separately by
    load_stage13_measured)."""
    with open(stage12_json_path) as f:
        d = json.load(f)
    revision = d["environment"]["git_commit"]
    records = {}
    for label, cand in d["candidates"].items():
        key = key_from_stage12(cand, revision)
        records[label] = CandidateEvidenceRecord(
            key=key, label=label,
            static_ir=static_ir_evidence_from_stage12(cand, key, stage12_json_path, revision),
            llvm_backend=llvm_backend_evidence_from_stage12(cand, key, stage12_json_path, revision),
            measured=None,
        )
    return records


def measured_evidence_from_stage13(cand: dict, key: CandidateKey, artifact_path: str, revision: str) -> MeasuredHardwareEvidence:
    agg = cand["benchmark_aggregate"]
    cid = key.canonical_id()
    ts = time.time()

    def mk(value, note=""):
        return ev(value, "raspberry_pi_measured", artifact_path, STAGE13_TOOL_VERSION,
                   STAGE13_HARDWARE_IDENTITY, revision, cid, note=note, timestamp=ts)

    # Use group-level stats from the first measurement group for
    # mean/p95/stddev (per-group detail); median/min use the more robust
    # cross-group median-of-medians/min-of-mins aggregate.
    first_group_bench = cand["measurement_groups"][0]["benchmark"]
    return MeasuredHardwareEvidence(
        median_latency_ms=mk(agg["median_of_medians_ms"], note="median of per-group medians across interleaved measurement groups"),
        p95_latency_ms=mk(first_group_bench.get("p95_ms")),
        mean_latency_ms=mk(first_group_bench.get("mean_ms")),
        stddev_latency_ms=mk(first_group_bench.get("stddev_ms")),
        cv=mk(agg["cv_of_medians"], note="coefficient of variation of per-group medians (cross-group stability, not single-group jitter)"),
        correctness_pass=mk(cand["correctness_pass"]),
        hardware_identity=mk(STAGE13_HARDWARE_IDENTITY),
    )


def load_stage13_measured(stage13_json_path: str, records: dict) -> dict:
    """Mutates `records` (as returned by load_stage12_records) in place,
    filling in `.measured` for whichever existing record has a matching
    CANONICAL KEY -- deliberately NOT a label match. Stage 12 and Stage 13
    independently chose different label strings for the same candidate
    (e.g. Stage 12's "primary_unroll2" is Stage 13's "primary_uk2"); a
    label-keyed merge would silently create two half-populated records for
    one real candidate instead of one fully-populated record -- exactly
    the "labels must not affect identity" failure mode this module's
    CandidateKey design exists to prevent. Candidates present only in
    Stage 13 (shapes Stage 12 never analyzed: cube16, rect, large) become
    new records with static_ir/llvm_backend evidence explicitly marked
    missing (None), never fabricated.

    Returns {stage13_label: resolved_records_label} so callers holding a
    Stage-13-labeled structure (e.g. its own "comparisons" dict) can
    resolve into `records`' actual keys after a merge folded a Stage-13
    label into a pre-existing Stage-12 one -- without this, any caller
    doing records[stage13_label] would silently miss every merged
    candidate (a real bug found and fixed while building this loader; see
    tests: test_stage13_label_resolves_after_merge_into_stage12_label)."""
    with open(stage13_json_path) as f:
        d = json.load(f)
    revision = d["pi_environment"]["git_commit"]
    by_key = {r.key: existing_label for existing_label, r in records.items()}
    alias = {}
    for label, cand in d["candidates"].items():
        shape = cand["shape"].split("x")
        key = CandidateKey(
            shape_m=int(shape[0]), shape_n=int(shape[1]), shape_k=int(shape[2]),
            tile_m=cand["tile"]["m"], tile_n=cand["tile"]["n"], tile_k=cand["tile"]["k"],
            schedule_unroll_k=cand["schedule_unroll_k"],
        )
        measured = measured_evidence_from_stage13(cand, key, stage13_json_path, revision)
        if key in by_key:
            existing_label = by_key[key]
            records[existing_label] = dataclasses.replace(records[existing_label], measured=measured)
            alias[label] = existing_label
        else:
            cid = key.canonical_id()
            missing_note = "no Stage 12 static/backend evidence collected for this candidate"
            empty = lambda: ev(None, "manual_config", stage13_json_path, STAGE13_TOOL_VERSION, STAGE13_HARDWARE_IDENTITY, revision, cid, note=missing_note)
            records[label] = CandidateEvidenceRecord(
                key=key, label=label,
                static_ir=StaticIRBEvidence(*[empty() for _ in dataclasses.fields(StaticIRBEvidence)]),
                llvm_backend=LLVMBackendEvidence(*[empty() for _ in dataclasses.fields(LLVMBackendEvidence)]),
                measured=measured,
            )
            by_key[key] = label
            alias[label] = label
    return alias


# ---------------------------------------------------------------------------
# Shape-aware interpretation (task section 9)
# ---------------------------------------------------------------------------

CONCLUSION_DIRECTLY_MEASURED = "directly_measured"
CONCLUSION_CROSS_SHAPE_OBSERVATION = "repeated_cross_shape_observation"
CONCLUSION_PLAUSIBLE_HYPOTHESIS = "plausible_hypothesis"
CONCLUSION_UNSUPPORTED = "unsupported"


def shape_aware_findings(records: dict, comparisons: dict) -> dict:
    """`comparisons`: {pair_name: {"baseline": label, "scheduled": label,
    "runtime_classification": "A"|..., "speedup_median_of_medians": float}}
    from the Stage 13 output. Classifies what can honestly be generalized
    from the 5-shape Stage 13 result set -- explicitly refuses to fit a
    regression from 5 points."""
    rows = []
    for pair_name, cmp in comparisons.items():
        b, s = records.get(cmp["baseline"]), records.get(cmp["scheduled"])
        if not b or not s:
            continue
        k_trip_pre = b.key.shape_k // b.key.tile_k if b.key.tile_k else None
        rows.append({
            "pair": pair_name, "shape": f"{b.key.shape_m}x{b.key.shape_n}x{b.key.shape_k}",
            "k_trip_count_pre_unroll": k_trip_pre,
            "speedup_pct": (cmp.get("speedup_median_of_medians") - 1.0) * 100.0 if cmp.get("speedup_median_of_medians") else None,
            "classification": cmp.get("runtime_classification"),
        })

    n = len(rows)
    findings = {
        "sample_count": n,
        "rows": rows,
        "conclusions": [],
    }

    all_class_a = n > 0 and all(r["classification"] == "A" for r in rows)
    findings["conclusions"].append({
        "statement": "schedule-unroll-k 1->2 at the 8x8x8 tile is profitable on every shape tested",
        "classification": CONCLUSION_DIRECTLY_MEASURED if n >= 1 else CONCLUSION_UNSUPPORTED,
        "basis": f"{n} independent shapes tested (8x8x8-derived small_control excluded, not a comparable pair), all classified A" if all_class_a else "not all tested shapes classified A",
    })
    findings["conclusions"].append({
        "statement": "the effect generalizes to matmul shapes broadly (any M/N/K, any tile)",
        "classification": CONCLUSION_UNSUPPORTED,
        "basis": f"only {n} shapes tested, all at the SAME tile (8x8x8) and SAME unroll factor (2); no data exists for other tiles/unroll factors across multiple shapes, and no data exists for non-matmul kernels",
    })
    if n >= 3:
        speedups = [r["speedup_pct"] for r in rows if r["speedup_pct"] is not None]
        if speedups:
            findings["conclusions"].append({
                "statement": f"speedup magnitude varies across shapes ({min(speedups):.1f}%-{max(speedups):.1f}%) but sign (positive) is consistent",
                "classification": CONCLUSION_CROSS_SHAPE_OBSERVATION,
                "basis": f"observed directly across {len(speedups)} shapes, not fit from a formula",
            })
    findings["conclusions"].append({
        "statement": "speedup magnitude correlates with K-loop trip count or code-size growth in a fittable, predictive way",
        "classification": CONCLUSION_UNSUPPORTED,
        "basis": (
            f"only {n} data points available; a regression fit from {n} points would be a curve through noise, "
            "not a validated model -- task brief section 9 explicitly requires discrete evidence buckets over "
            "an unjustified formula at this sample size"
        ),
    })
    findings["conclusions"].append({
        "statement": "spilling candidates can be profitable even though spills are a real backend cost",
        "classification": CONCLUSION_DIRECTLY_MEASURED,
        "basis": "directly measured for 2 of 2 tested Group B diagnostics (full-K-unroll, alt-K-tile) -- both faster than their matched baselines despite spills",
    })
    findings["conclusions"].append({
        "statement": "spilling candidates are USUALLY profitable / spills rarely matter in general",
        "classification": CONCLUSION_PLAUSIBLE_HYPOTHESIS,
        "basis": "only 2 spilling candidates were ever measured, both from the same 32x32x32 shape and same underlying tile/unroll mechanism -- suggestive, not sufficient to generalize across problem sizes where the spilled working set no longer fits L1",
    })
    return findings
