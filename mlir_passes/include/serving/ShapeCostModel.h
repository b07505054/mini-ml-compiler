#pragma once

// ShapeCostModel — shape-aware, dtype-aware, profile-aware static cost
// formulas (shape_cost_model_v2).
//
// Pure value types and formulas only: no MLIR types, independently testable.
// The MLIR-facing side (reading tensor types, quant attrs, and
// target.static_cost_profile.* module attrs) lives in ServingCostModelPass.
//
// Everything here is a STATIC COMPILER ESTIMATE derived from tensor shapes
// and declared profile numbers. It is not a measured benchmark and not a
// runtime latency guarantee.
//
//   truth_boundary:
//     static_shape_derived_declared_profile_not_measured_not_runtime_validated
//
// V2 formulas (documented approximations, deliberately conservative):
//   matmul_like:    flops = 2*M*K*N; weight_elems = K*N (weights of llm.*
//                   projection ops are not IR operands; K/N are inferred
//                   from the input's and result's last dimensions)
//   normalization:  flops = 4*elems (square + reduce + rsqrt + scale)
//   elementwise:    flops = 1*elems
//   unknown:        no estimate — callers fall back to the V1 fixed model
//
// Boundary traffic model (bytes moved by materialized boundary ops):
//   cast / layout_transform / requant : 2 * output_bytes  (read + write)
//   dequant / dequant_weight          : 2 * float weight bytes
//
// Time estimates exist only when the declared profile provides peak
// FLOPs/bandwidth, and use the roofline form
//   total = max(compute_time, memory_time) + boundary_time
// in integer nanoseconds. No profile numbers -> facts only, never a
// fabricated time.

#include <algorithm>
#include <cstdint>
#include <string>

namespace mlir::hir {

static constexpr const char* kShapeCostModelVersion = "shape_cost_model_v2";
static constexpr const char* kShapeCostTruthBoundary =
    "static_shape_derived_declared_profile_not_measured_not_runtime_validated";

// Dtype width in bits; 0 = unrecognized (no byte estimate possible).
inline int64_t dtypeBits(const std::string& dtype) {
  if (dtype == "fp32" || dtype == "f32")   return 32;
  if (dtype == "fp16" || dtype == "f16")   return 16;
  if (dtype == "bf16")                      return 16;
  if (dtype == "int8" || dtype == "i8")    return 8;
  if (dtype == "int4" || dtype == "i4")    return 4;
  return 0;
}

// Shape-derived, dtype-independent facts for one op.
// Produced by the pass from tensor types; consumed by estimateShapeCost.
struct ShapeFacts {
  // op_kind: "matmul_like" | "normalization" | "elementwise" | "unknown"
  std::string op_kind = "unknown";
  // status: "static_shapes" (usable) | "dynamic_dims_unresolved" |
  //         "no_tensor_result" | "unsupported_op_kind"
  std::string status = "unsupported_op_kind";
  int64_t flops        = 0; // dtype-independent op count
  int64_t input_elems  = 0;
  int64_t output_elems = 0;
  int64_t weight_elems = 0;
  // GEMM dimensions; set only for op_kind == "matmul_like".
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;

  bool usable() const { return status == "static_shapes"; }
};

// MemoryHierarchyProfile — OPTIONAL declared memory-hierarchy metadata.
// Not every backend exposes local memory or DMA details; absence is a valid
// state and must never be papered over with invented values. When
// local_memory_bytes is not declared, tile feasibility is DEFERRED and
// recorded as "deferred_missing_memory_hierarchy" — the plan stays valid.
struct MemoryHierarchyProfile {
  // SRAM / shared memory / scratchpad usable per compute unit for one op's
  // working tiles. 0 = not declared.
  int64_t local_memory_bytes = 0;
  int64_t cache_line_bytes   = 0;
  // Declared capability flags (e.g. cp.async on NVIDIA Ampere+, DMA engines
  // on NPUs). The has_* flags distinguish "declared false" from "not
  // declared at all" — unknown is not the same fact as unavailable. Used
  // only for static feasibility annotations — no async copy or DMA is ever
  // executed or simulated.
  bool supports_async_copy         = false;
  bool supports_dma                = false;
  bool has_async_copy_declaration  = false;
  bool has_dma_declaration         = false;

  bool localMemoryDeclared() const { return local_memory_bytes > 0; }

  // Staging capability as a declared fact:
  //   "async_copy_declared" | "dma_declared" | "declared_unavailable"
  //   (declared and false) | "unknown_not_declared" (no declaration).
  const char* stagingCapability() const {
    if (has_async_copy_declaration && supports_async_copy)
      return "async_copy_declared";
    if (has_dma_declaration && supports_dma)
      return "dma_declared";
    if (has_async_copy_declaration || has_dma_declaration)
      return "declared_unavailable";
    return "unknown_not_declared";
  }
};

// Declared-profile peak numbers (target.static_cost_profile.* module attrs).
// 0 = not declared. All values are theoretical peaks / declared capacities
// from public docs or a declared profile — never measured.
struct StaticCostProfileNums {
  double peak_flops_fp32            = 0.0;
  double peak_flops_fp16            = 0.0;
  double peak_flops_int8            = 0.0;
  double mem_bandwidth_bytes_per_sec = 0.0;
  // Optional declared memory hierarchy (see MemoryHierarchyProfile).
  MemoryHierarchyProfile memory_hierarchy;

  // Peak FLOPs for a dtype, falling back to the nearest wider declared
  // peak. Returns 0 when nothing applicable is declared.
  double peakFlopsFor(int64_t bits) const {
    if (bits <= 8) {
      if (peak_flops_int8 > 0.0) return peak_flops_int8;
      if (peak_flops_fp16 > 0.0) return peak_flops_fp16;
      return peak_flops_fp32;
    }
    if (bits == 16) {
      if (peak_flops_fp16 > 0.0) return peak_flops_fp16;
      return peak_flops_fp32;
    }
    return peak_flops_fp32;
  }

  bool hasBandwidth() const { return mem_bandwidth_bytes_per_sec > 0.0; }
};

// One candidate's shape-derived cost estimate.
struct ShapeCostEstimateResult {
  int64_t flops_estimate             = 0;
  int64_t input_bytes                = 0;
  int64_t output_bytes               = 0;
  int64_t weight_bytes               = 0;
  int64_t total_memory_bytes         = 0;
  int64_t arithmetic_intensity_milli = 0; // flops * 1000 / total_memory_bytes
  // Roofline time estimates in integer nanoseconds; valid only when
  // has_time_estimates is true (declared peak numbers were available).
  bool    has_time_estimates          = false;
  int64_t estimated_compute_cost_nanos  = 0;
  int64_t estimated_memory_cost_nanos   = 0;
  int64_t estimated_boundary_cost_nanos = 0;
  int64_t estimated_total_cost_nanos    = 0;
  // "estimated" | "facts_only_no_profile_numbers"
  std::string status;
};

// ---------------------------------------------------------------------------
// Tile planning (tile_planning_v1) — static local-memory feasibility for
// matmul-like ops. Chooses the largest tile from a fixed conservative menu
// whose A (Mt x Kt), B (Kt x Nt), and C (Mt x Nt) working set fits the
// declared local memory. This models feasibility and reuse-limited global
// traffic only; it does not claim the backend kernel uses this tiling, and
// no codegen or DMA is performed.
//
//   truth_boundary:
//     tile_planning_static_local_memory_model_not_measured_not_codegen
// ---------------------------------------------------------------------------

static constexpr const char* kTilePlanTruthBoundary =
    "tile_planning_static_local_memory_model_not_measured_not_codegen";

struct TilePlanResult {
  bool feasible = false;
  int64_t tile_m = 0, tile_n = 0, tile_k = 0;
  int64_t local_memory_bytes = 0;   // footprint of the selected tile
  int64_t min_footprint_bytes = 0;  // smallest candidate footprint tried
  int64_t rejected_tile_count = 0;
  // True when 2x(A+B) + C also fits — double-buffered staging is feasible.
  // Only meaningful when the profile declares supports_async_copy/dma;
  // recorded as a static fact either way.
  bool double_buffer_fits = false;
  // Reuse-limited global traffic for the selected tiling:
  //   A read ceil(N/Nt) times + B read ceil(M/Mt) times + C written once.
  // This is >= the ideal each-byte-once estimate of shape_cost_model_v2 and
  // is reported alongside it, never replacing it.
  int64_t estimated_global_traffic_bytes = 0;
};

inline int64_t ceilDiv(int64_t a, int64_t b) { return b > 0 ? (a + b - 1) / b : 0; }

inline TilePlanResult planMatmulTiles(int64_t m, int64_t n, int64_t k,
                                      int64_t act_bits, int64_t weight_bits,
                                      int64_t local_memory_bytes) {
  TilePlanResult r;
  if (m <= 0 || n <= 0 || k <= 0 || act_bits <= 0 || weight_bits <= 0 ||
      local_memory_bytes <= 0)
    return r;

  // Conservative fixed menu, largest first; clipped to the op's dims.
  static constexpr int64_t kMenu[][3] = {
      {128, 128, 32}, {64, 64, 32}, {32, 32, 32}, {16, 16, 16}};

  auto footprintOf = [&](int64_t mt, int64_t nt, int64_t kt) {
    return mt * kt * act_bits / 8    // A tile
         + kt * nt * weight_bits / 8 // B tile
         + mt * nt * act_bits / 8;   // C tile
  };

  r.min_footprint_bytes = 0;
  for (const auto& cand : kMenu) {
    int64_t mt = std::min(cand[0], m);
    int64_t nt = std::min(cand[1], n);
    int64_t kt = std::min(cand[2], k);
    int64_t footprint = footprintOf(mt, nt, kt);
    if (r.min_footprint_bytes == 0 || footprint < r.min_footprint_bytes)
      r.min_footprint_bytes = footprint;
    if (footprint > local_memory_bytes) {
      ++r.rejected_tile_count;
      continue;
    }
    r.feasible = true;
    r.tile_m = mt;
    r.tile_n = nt;
    r.tile_k = kt;
    r.local_memory_bytes = footprint;
    int64_t doubleBuffered = 2 * (mt * kt * act_bits / 8 +
                                  kt * nt * weight_bits / 8) +
                             mt * nt * act_bits / 8;
    r.double_buffer_fits = doubleBuffered <= local_memory_bytes;
    r.estimated_global_traffic_bytes =
        m * k * ceilDiv(n, nt) * act_bits / 8      // A panels
      + k * n * ceilDiv(m, mt) * weight_bits / 8   // B panels
      + m * n * act_bits / 8;                      // C written once
    break;
  }
  return r;
}

// Boundary bytes for one boundary-op name.
inline int64_t boundaryBytesFor(const std::string& b, int64_t output_bytes,
                                int64_t float_weight_bytes) {
  if (b == "dequant" || b == "dequant_weight") return 2 * float_weight_bytes;
  if (b == "cast" || b == "layout_transform" || b == "requant")
    return 2 * output_bytes;
  return 0; // unknown boundary op: no byte claim
}

// Compute the V2 estimate for one candidate.
//   facts:        usable shape facts (caller checks facts.usable()).
//   act_bits:     activation/output dtype width in bits (> 0).
//   weight_bits:  weight dtype width in bits (> 0; equals act_bits when no
//                 quantization metadata narrows it).
//   boundary_bytes: precomputed via boundaryBytesFor over the candidate's
//                 required_boundary_ops.
//   nums:         declared profile peaks (may be all-zero).
inline ShapeCostEstimateResult
estimateShapeCost(const ShapeFacts& facts, int64_t act_bits,
                  int64_t weight_bits, int64_t boundary_bytes,
                  const StaticCostProfileNums& nums) {
  ShapeCostEstimateResult r;
  r.flops_estimate = facts.flops;
  r.input_bytes    = facts.input_elems * act_bits / 8;
  r.output_bytes   = facts.output_elems * act_bits / 8;
  r.weight_bytes   = facts.weight_elems * weight_bits / 8;
  r.total_memory_bytes = r.input_bytes + r.output_bytes + r.weight_bytes;
  if (r.total_memory_bytes > 0)
    r.arithmetic_intensity_milli =
        r.flops_estimate * 1000 / r.total_memory_bytes;

  double peak = nums.peakFlopsFor(act_bits);
  if (peak > 0.0 && nums.hasBandwidth()) {
    auto toNanos = [](double seconds) {
      return static_cast<int64_t>(seconds * 1e9 + 0.5);
    };
    r.estimated_compute_cost_nanos =
        toNanos(static_cast<double>(r.flops_estimate) / peak);
    r.estimated_memory_cost_nanos =
        toNanos(static_cast<double>(r.total_memory_bytes) /
                nums.mem_bandwidth_bytes_per_sec);
    r.estimated_boundary_cost_nanos =
        toNanos(static_cast<double>(boundary_bytes) /
                nums.mem_bandwidth_bytes_per_sec);
    r.estimated_total_cost_nanos =
        std::max(r.estimated_compute_cost_nanos,
                 r.estimated_memory_cost_nanos) +
        r.estimated_boundary_cost_nanos;
    r.has_time_estimates = true;
    r.status = "estimated";
  } else {
    r.status = "facts_only_no_profile_numbers";
  }
  return r;
}

} // namespace mlir::hir
