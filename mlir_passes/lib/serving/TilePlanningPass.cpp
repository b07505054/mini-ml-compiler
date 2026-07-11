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
// The memory hierarchy is OPTIONAL declared metadata: not every backend
// exposes local memory or DMA details. When local_memory_bytes is not
// declared, feasibility is not guessed — matmul-like ops are stamped
// tile.plan.status = "deferred_missing_memory_hierarchy" with an explicit
// deferred_reason, and the plan stays valid. Tile feasibility runs only
// when the op kind is supported, the memory hierarchy is declared, shapes
// are fully static, and the dtype resolves. Rejections and deferrals are
// recorded explicitly, never invented.
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

    StaticCostProfileNums nums = readProfileNums(funcOp->getParentOp());
    const MemoryHierarchyProfile& mh = nums.memory_hierarchy;

    std::string effectiveDtype;
    if (auto a =
            funcOp->getAttrOfType<StringAttr>("representation.effective_dtype"))
      effectiveDtype = a.getValue().str();

    auto i64 = IntegerType::get(ctx, 64);
    auto S = [&](StringRef s) { return StringAttr::get(ctx, s); };
    auto I = [&](int64_t v) { return IntegerAttr::get(i64, v); };

    // Gate order (feasibility runs only when ALL hold, and the recorded
    // status names the first unmet gate):
    //   1. op kind supported (matmul-like; others get no attrs in V1)
    //   2. memory hierarchy declared (else deferred_missing_memory_hierarchy)
    //   3. shapes fully static (else dynamic_dims_unresolved / ...)
    //   4. dtype resolvable (else dtype_unresolved)
    for (Operation &op : funcOp.getBody().front().without_terminator()) {
      if (classifyOpKind(op) != "matmul_like")
        continue; // V1 scope: matmul-like ops only.

      op.setAttr("tile.plan.truth_boundary", S(kTilePlanTruthBoundary));

      if (!mh.localMemoryDeclared()) {
        // Optional metadata is absent: never invent a capacity. Record the
        // deferral so the exported plan says why no tile was planned.
        op.setAttr("tile.plan.status", S("deferred_missing_memory_hierarchy"));
        op.setAttr("tile.plan.deferred_reason",
                   S("local_memory_bytes_not_declared_in_target_profile"));
        continue;
      }

      ShapeFacts facts = computeShapeFacts(op);
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
                                            mh.local_memory_bytes);
      if (!plan.feasible) {
        op.setAttr("tile.plan.status", S("no_feasible_tile"));
        op.setAttr("tile.plan.rejected_tile_count",
                   I(plan.rejected_tile_count));
        op.setAttr("tile.plan.rejection_reason",
                   S("smallest_tile_footprint_" +
                     std::to_string(plan.min_footprint_bytes) +
                     "_bytes_exceeds_local_memory_" +
                     std::to_string(mh.local_memory_bytes) + "_bytes"));
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
      // Double-buffered staging feasibility — a static capacity fact. Its
      // actionability depends on what the profile DECLARED about staging:
      // async_copy_declared / dma_declared / declared_unavailable (declared
      // false) / unknown_not_declared (no declaration — unknown, not
      // unavailable).
      op.setAttr("tile.plan.double_buffer_fits",
                 BoolAttr::get(ctx, plan.double_buffer_fits));
      op.setAttr("tile.plan.staging_capability", S(mh.stagingCapability()));
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTilePlanningPass() {
  return std::make_unique<TilePlanningPass>();
}

} // namespace mlir::hir
