// TilePlanningPass — static local-memory tile feasibility planning
// (tile_planning_v1).
//
// For matmul-like ops with fully static shapes, selects the largest tile
// from a fixed conservative menu whose A/B/C working set fits the declared
// local memory (target.static_cost_profile.local_memory_bytes: SRAM /
// shared memory / scratchpad), records the footprint, double-buffer
// feasibility (relevant when the profile declares supports_async_copy or
// supports_dma), and a reuse-limited global-traffic estimate. Formulas live
// in serving/ShapeCostModel.h (planMatmulTiles).
//
// The pass is deliberately inert when the module declares no local memory —
// no attrs are stamped, so profiles without a memory hierarchy declaration
// keep byte-identical output. Rejections and dynamic shapes are recorded
// explicitly, never guessed.
//
// Truth boundary: this is memory-hierarchy-aware STATIC PLANNING. It does
// not claim the backend kernel uses this tiling, does not perform DMA or
// async copies, does not rewrite IR, and makes no measured-performance
// claim.

#include "serving/OpShapeFacts.h"
#include "serving/ShapeCostModel.h"
#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_TILEPLANNING
#include "FusionPasses.h.inc"

struct TilePlanningPass : impl::TilePlanningBase<TilePlanningPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    if (funcOp.getBody().empty())
      return;

    // Inert without a declared local memory capacity: tile feasibility
    // against an undeclared budget would be a guess.
    StaticCostProfileNums nums = readProfileNums(funcOp->getParentOp());
    if (nums.local_memory_bytes <= 0)
      return;

    std::string effectiveDtype;
    if (auto a =
            funcOp->getAttrOfType<StringAttr>("representation.effective_dtype"))
      effectiveDtype = a.getValue().str();

    auto i64 = IntegerType::get(ctx, 64);
    auto S = [&](StringRef s) { return StringAttr::get(ctx, s); };
    auto I = [&](int64_t v) { return IntegerAttr::get(i64, v); };

    for (Operation &op : funcOp.getBody().front().without_terminator()) {
      if (classifyOpKind(op) != "matmul_like")
        continue; // V1 scope: matmul-like ops only.

      ShapeFacts facts = computeShapeFacts(op);
      op.setAttr("tile.plan.truth_boundary", S(kTilePlanTruthBoundary));

      if (!facts.usable()) {
        // Dynamic dims or unusable shapes: defer honestly, no invented tile.
        op.setAttr("tile.plan.status", S(facts.status));
        continue;
      }

      int64_t actBits =
          dtypeBits(resolveOpActivationDtype(op, effectiveDtype));
      int64_t weightBits = 0;
      if (auto a = op.getAttrOfType<StringAttr>("quant.weight_dtype"))
        weightBits = dtypeBits(a.getValue().str());
      if (weightBits <= 0) weightBits = actBits;
      if (actBits <= 0) {
        op.setAttr("tile.plan.status", S("dtype_unresolved"));
        continue;
      }

      TilePlanResult plan = planMatmulTiles(facts.m, facts.n, facts.k,
                                            actBits, weightBits,
                                            nums.local_memory_bytes);
      if (!plan.feasible) {
        op.setAttr("tile.plan.status", S("no_feasible_tile"));
        op.setAttr("tile.plan.rejected_tile_count",
                   I(plan.rejected_tile_count));
        op.setAttr("tile.plan.rejection_reason",
                   S("smallest_tile_footprint_" +
                     std::to_string(plan.min_footprint_bytes) +
                     "_bytes_exceeds_local_memory_" +
                     std::to_string(nums.local_memory_bytes) + "_bytes"));
        continue;
      }

      op.setAttr("tile.plan.status", S("planned"));
      op.setAttr("tile.plan.shape",
                 ArrayAttr::get(ctx, {I(plan.tile_m), I(plan.tile_n),
                                      I(plan.tile_k)}));
      op.setAttr("tile.plan.local_memory_bytes", I(plan.local_memory_bytes));
      op.setAttr("tile.plan.rejected_tile_count",
                 I(plan.rejected_tile_count));
      op.setAttr("tile.plan.estimated_global_traffic_bytes",
                 I(plan.estimated_global_traffic_bytes));
      // Double-buffered staging feasibility — a static capacity fact. Only
      // actionable on targets declaring async copy / DMA support, so record
      // which capability (if any) the profile declared alongside it.
      op.setAttr("tile.plan.double_buffer_fits",
                 BoolAttr::get(ctx, plan.double_buffer_fits));
      StringRef staging = "none_declared";
      if (nums.supports_async_copy) staging = "async_copy_declared";
      else if (nums.supports_dma)   staging = "dma_declared";
      op.setAttr("tile.plan.staging_capability", S(staging));
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTilePlanningPass() {
  return std::make_unique<TilePlanningPass>();
}

} // namespace mlir::hir
