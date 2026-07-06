#pragma once

// ServingCostModel — lightweight value type encapsulating the
// ServingStaticCostModel_v1 formula logic.
//
// Constructed from StaticCostWeights (deployment-context-tunable).
// compute() applies the V1 formula and returns a DecisionCost.
// No MLIR types used; independently testable.
//
// truth_boundary: serving_static_cost_model_v1_not_measured_latency
//
// Compile-time rejection sentinels (not tunable deployment parameters):
//   kUnsupportedPenalty = 100 — must remain unselectable over any working path
//   kUnknownTypePenalty  =  50 — unrecognized type: unknown cost, partial evaluation

#include "capability/CapabilityBundle.h"
#include "decision/Decision.h"

#include <string>
#include <vector>

namespace mlir::hir {

// Compile-time rejection sentinels.
static constexpr int64_t kUnsupportedPenalty = 100;
static constexpr int64_t kUnknownTypePenalty  =  50;

// Cost model identity constants.
static constexpr const char* kCostModelId      = "serving_static_cost_model_v1";
static constexpr const char* kCostTruthBoundary =
    "serving_static_cost_model_v1_not_measured_latency";

struct ServingCostModel {
  explicit ServingCostModel(const StaticCostWeights& w = StaticCostWeights{})
      : w_(w) {}

  // Compute structured cost evidence for one candidate.
  //   candidate_type:  "direct_lower" | "algebraic_decomposition" | etc.
  //   boundary_ops:    required boundary op names ("dequant", "cast", "layout_transform", "requant")
  //   fallback_backend: non-empty when candidate_type == "backend_fallback" with a declared target
  //
  // Returns a DecisionCost where total_cost == exact sum of all component fields.
  // kv_cache_cost is always 0 until a future pass emits kv.transfer attrs.
  DecisionCost compute(const std::string& candidate_type,
                       const std::vector<std::string>& boundary_ops,
                       const std::string& fallback_backend) const {
    DecisionCost cost;
    cost.cost_model_id  = kCostModelId;
    cost.truth_boundary = kCostTruthBoundary;

    int64_t n_bops = 0; // number of boundary ops counted for launch overhead

    // Candidate-type-specific costs.
    if (candidate_type == "direct_lower") {
      cost.compute_cost = 0;
      addBoundaryCosts(cost, boundary_ops, n_bops);
      cost.launch_overhead_cost = n_bops * w_.launch_overhead_per_boundary;
    } else if (candidate_type == "algebraic_decomposition") {
      cost.compute_cost = w_.compute_decomposition;
      addBoundaryCosts(cost, boundary_ops, n_bops);
      cost.launch_overhead_cost = n_bops > 0
          ? n_bops * w_.launch_overhead_per_boundary : 0;
    } else if (candidate_type == "representation_conversion") {
      cost.memory_cost = w_.memory_repr_conv;
      addBoundaryCosts(cost, boundary_ops, n_bops);
      // One launch overhead if more than one distinct boundary op.
      cost.launch_overhead_cost = n_bops > 1 ? w_.launch_overhead_per_boundary : 0;
    } else if (candidate_type == "layout_conversion") {
      // The conversion itself is the primary work; additional boundary ops are extras.
      cost.layout_transform_cost = w_.layout_transform_standalone;
      addBoundaryCosts(cost, boundary_ops, n_bops, /*skip_layout=*/true);
    } else if (candidate_type == "cast_conversion") {
      cost.cast_cost = w_.cast_cost;
      addBoundaryCosts(cost, boundary_ops, n_bops, /*skip_layout=*/false, /*skip_cast=*/true);
    } else if (candidate_type == "backend_fallback") {
      cost.backend_switch_cost  = w_.backend_switch_cost;
      cost.launch_overhead_cost = w_.launch_overhead_fallback;
      addBoundaryCosts(cost, boundary_ops, n_bops);
      // Device transfer applies when a specific fallback backend is declared.
      if (!fallback_backend.empty())
        cost.transfer_cost = w_.transfer_cost_fallback;
    } else if (candidate_type == "unsupported") {
      cost.unsupported_penalty = kUnsupportedPenalty;
    } else {
      // Unknown candidate type: unquantifiable cost, partial evaluation.
      cost.unsupported_penalty = kUnknownTypePenalty;
    }

    // kv_cache_cost: reserved — 0 until a future pass emits kv.transfer attrs.
    cost.kv_cache_cost = 0;

    cost.total_cost = cost.compute_cost
                    + cost.memory_cost
                    + cost.dequant_cost
                    + cost.requant_cost
                    + cost.layout_transform_cost
                    + cost.cast_cost
                    + cost.backend_switch_cost
                    + cost.launch_overhead_cost
                    + cost.kv_cache_cost
                    + cost.transfer_cost
                    + cost.unsupported_penalty;
    return cost;
  }

private:
  StaticCostWeights w_;

  // Assign per-boundary-op costs and increment n_bops.
  // skip_layout: set when candidate IS a layout_conversion (avoids double-counting).
  // skip_cast:   set when candidate IS a cast_conversion (avoids double-counting).
  void addBoundaryCosts(DecisionCost& cost,
                        const std::vector<std::string>& boundary_ops,
                        int64_t& n_bops,
                        bool skip_layout = false,
                        bool skip_cast   = false) const {
    for (const auto& b : boundary_ops) {
      if (b == "dequant" || b == "dequant_weight") {
        cost.dequant_cost += w_.dequant_cost;
        n_bops++;
      } else if (b == "requant") {
        cost.requant_cost += w_.requant_cost;
        n_bops++;
      } else if (b == "layout_transform" && !skip_layout) {
        cost.layout_transform_cost += w_.layout_transform_boundary;
        n_bops++;
      } else if (b == "cast" && !skip_cast) {
        cost.cast_cost += w_.cast_cost;
        n_bops++;
      } else {
        // Unknown boundary op: count for launch overhead but no specific cost.
        n_bops++;
      }
    }
  }
};

} // namespace mlir::hir
