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
#include "serving/MemoryPlanning.h"
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

static DictionaryAttr dictAttr(MLIRContext *ctx,
                               ArrayRef<NamedAttribute> fields) {
  return DictionaryAttr::get(ctx, fields);
}

static void emitMemoryPlacementPlan(Operation &op, MLIRContext *ctx,
                                    StringRef status,
                                    StringRef computeUnit,
                                    StringRef memorySpace,
                                    const TileMemoryFootprint &fp,
                                    bool acceleratorTransfers) {
  auto S = [&](StringRef s) { return StringAttr::get(ctx, s); };
  auto I = [&](int64_t v) { return IntegerAttr::get(IntegerType::get(ctx, 64), v); };
  op.setAttr("memory_placement.status", S(status));
  op.setAttr("memory_placement.compute_unit", S(computeUnit));
  op.setAttr("memory_placement.selected_memory_space", S(memorySpace));
  op.setAttr("memory_placement.input_tile_bytes", I(fp.inputTileBytes));
  op.setAttr("memory_placement.weight_tile_bytes", I(fp.weightTileBytes));
  op.setAttr("memory_placement.output_tile_bytes", I(fp.outputTileBytes));
  op.setAttr("memory_placement.scratch_bytes", I(fp.scratchBytes));
  op.setAttr("memory_placement.padding_bytes", I(fp.paddingBytes));
  op.setAttr("memory_placement.single_buffer_bytes", I(fp.singleBufferBytes));
  op.setAttr("memory_placement.additional_double_buffer_bytes",
             I(fp.additionalDoubleBufferBytes));
  op.setAttr("memory_placement.total_required_local_memory_bytes",
             I(fp.totalRequiredLocalMemoryBytes));
  op.setAttr("memory_placement.truth_boundary",
             S("memory_placement_static_compiler_contract_not_runtime_allocation"));

  SmallVector<Attribute> placements;
  auto placement = [&](StringRef id, StringRef role, int64_t bytes) {
    SmallVector<NamedAttribute> fields;
    fields.emplace_back(S("alignment"), I(64));
    fields.emplace_back(S("buffer_id"), S(id));
    fields.emplace_back(S("byte_count"), I(bytes));
    fields.emplace_back(S("memory_space"), S(memorySpace));
    fields.emplace_back(S("role"), S(role));
    placements.push_back(dictAttr(ctx, fields));
  };
  placement("input_tile", "input", fp.inputTileBytes);
  placement("weight_tile", "weight", fp.weightTileBytes);
  placement("output_tile", "output", fp.outputTileBytes);
  placement("scratch", "scratch", fp.scratchBytes);
  op.setAttr("memory_placement.buffer_placements",
             ArrayAttr::get(ctx, placements));

  SmallVector<Attribute> transfers;
  SmallVector<Attribute> computeDeps;
  if (acceleratorTransfers) {
    auto transfer = [&](StringRef id, StringRef srcBuf, StringRef dstBuf,
                        StringRef srcSpace, StringRef dstSpace, int64_t bytes,
                        ArrayRef<StringRef> deps, StringRef token) {
      SmallVector<Attribute> depAttrs;
      for (StringRef dep : deps) depAttrs.push_back(S(dep));
      SmallVector<NamedAttribute> fields;
      fields.emplace_back(S("alignment"), I(64));
      fields.emplace_back(S("byte_count"), I(bytes));
      fields.emplace_back(S("completion_token"), S(token));
      fields.emplace_back(S("dependency_ids"), ArrayAttr::get(ctx, depAttrs));
      fields.emplace_back(S("destination_buffer"), S(dstBuf));
      fields.emplace_back(S("destination_memory_space"), S(dstSpace));
      fields.emplace_back(S("mode"), S("synchronous"));
      fields.emplace_back(S("source_buffer"), S(srcBuf));
      fields.emplace_back(S("source_memory_space"), S(srcSpace));
      fields.emplace_back(S("transfer_id"), S(id));
      transfers.push_back(dictAttr(ctx, fields));
    };
    transfer("transfer_input_to_local", "input", "input_tile", "host", memorySpace,
             fp.inputTileBytes, {}, "input_ready");
    transfer("transfer_weight_to_local", "weight", "weight_tile", "host", memorySpace,
             fp.weightTileBytes, {}, "weight_ready");
    transfer("transfer_output_to_host", "output_tile", "output", memorySpace, "host",
             fp.outputTileBytes, {"compute_complete"}, "output_ready");
    computeDeps.push_back(S("input_ready"));
    computeDeps.push_back(S("weight_ready"));
  }
  op.setAttr("memory_placement.transfer_operations",
             ArrayAttr::get(ctx, transfers));
  op.setAttr("memory_placement.compute_dependency_ids",
             ArrayAttr::get(ctx, computeDeps));
}

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
        ShapeFacts facts = computeShapeFacts(op);
        int64_t actBits = dtypeBits(resolveOpActivationDtype(op, effectiveDtype));
        int64_t weightBits = actBits;
        if (auto a = op.getAttrOfType<StringAttr>("quant.weight_dtype"))
          if (dtypeBits(a.getValue().str()) > 0) weightBits = dtypeBits(a.getValue().str());
        int64_t outBits = actBits;
        if (auto a = op.getAttrOfType<StringAttr>("quant.output_dtype"))
          if (dtypeBits(a.getValue().str()) > 0) outBits = dtypeBits(a.getValue().str());
        if (facts.usable() && actBits > 0 && weightBits > 0 && outBits > 0) {
          TileMemoryRequest req;
          req.tileM = facts.m;
          req.tileN = facts.n;
          req.tileK = facts.k;
          req.activationDtype = actBits == 32 ? "fp32" : (actBits == 16 ? "fp16" : "int8");
          req.weightDtype = weightBits == 32 ? "fp32" : (weightBits == 16 ? "fp16" : "int8");
          req.outputDtype = outBits == 32 ? "fp32" : (outBits == 16 ? "fp16" : "int8");
          req.alignment = 64;
          TileMemoryFootprint fp = calculateTileMemoryFootprint(req);
          if (fp.ok)
            emitMemoryPlacementPlan(op, ctx, "selected", "cpu",
                                    "cpu_visible_host_memory", fp,
                                    false);
        }
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
      TileMemoryRequest req;
      req.tileM = plan.tile_m;
      req.tileN = plan.tile_n;
      req.tileK = plan.tile_k;
      req.activationDtype = actBits == 32 ? "fp32" : (actBits == 16 ? "fp16" : "int8");
      req.weightDtype = weightBits == 32 ? "fp32" : (weightBits == 16 ? "fp16" : "int8");
      req.outputDtype = req.activationDtype;
      req.alignment = 64;
      req.doubleBufferInputAndWeight = plan.double_buffer_fits &&
          (mh.supports_async_copy || mh.supports_dma);
      TileMemoryFootprint fp = calculateTileMemoryFootprint(req);
      if (fp.ok)
        emitMemoryPlacementPlan(op, ctx, "selected", "synthetic_accelerator",
                                "local_sram", fp, true);
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createTilePlanningPass() {
  return std::make_unique<TilePlanningPass>();
}

} // namespace mlir::hir
