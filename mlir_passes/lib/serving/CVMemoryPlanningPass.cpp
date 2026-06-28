#include "FusionPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <cstdint>
#include <memory>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_CVMEMORYPLANNING
#include "FusionPasses.h.inc"

static constexpr llvm::StringLiteral kTruthBoundary =
    "static_compiler_memory_plan_not_runtime_allocation";
static constexpr llvm::StringLiteral kPlanReason =
    "linear_lifetime_reuse_from_static_shape_metadata";

struct PoolEntry {
  int64_t id;
  int64_t offset;
  int64_t sizeBytes;
  int64_t freedAt;
};

// Linear first-fit reuse policy.
// Returns the first entry where entry.freedAt < lifetimeBegin
// and entry.sizeBytes >= bytesEstimate, or nullptr if none qualifies.
// The policy can be replaced here without touching the rest of the pass.
static PoolEntry *selectReusableBuffer(llvm::SmallVector<PoolEntry> &pool,
                                       int64_t lifetimeBegin,
                                       int64_t bytesEstimate) {
  for (PoolEntry &entry : pool) {
    if (entry.freedAt < lifetimeBegin && entry.sizeBytes >= bytesEstimate)
      return &entry;
  }
  return nullptr;
}

struct CVMemoryPlanningPass
    : impl::CVMemoryPlanningBase<CVMemoryPlanningPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();

    // Gate: require at least one cv.* op; otherwise no function attrs are set.
    bool hasCVOps = false;
    funcOp.walk([&](Operation *op) -> WalkResult {
      if (op->getName().getStringRef().starts_with("cv.")) {
        hasCVOps = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!hasCVOps)
      return;

    // Collect plannable and skipped cv.* ops in block order.
    // Single-block assumption: iterates the entry block only.
    llvm::SmallVector<Operation *> plannableOps;
    int64_t skippedOps = 0;

    for (Operation &op : funcOp.getBody().front()) {
      if (!op.getName().getStringRef().starts_with("cv."))
        continue;
      if (op.getNumResults() != 1 ||
          !op.getAttrOfType<IntegerAttr>("cv.bytes_estimate")) {
        ++skippedOps;
        continue;
      }
      plannableOps.push_back(&op);
    }

    int64_t n = static_cast<int64_t>(plannableOps.size());

    // Build index map for plannable ops.
    llvm::DenseMap<Operation *, int64_t> opIndex;
    opIndex.reserve(n);
    for (int64_t i = 0; i < n; ++i)
      opIndex[plannableOps[i]] = i;

    // Phase 1: compute lifetimes and read bytes estimates.
    llvm::SmallVector<int64_t> lifetimeBegin(n), lifetimeEnd(n), bytesVec(n);
    for (int64_t i = 0; i < n; ++i) {
      lifetimeBegin[i] = i;
      lifetimeEnd[i] = i; // default: no plannable consumer
      bytesVec[i] =
          plannableOps[i]->getAttrOfType<IntegerAttr>("cv.bytes_estimate").getInt();
      // Extend lifetime to the last plannable cv.* consumer.
      for (Operation *user : plannableOps[i]->getResult(0).getUsers()) {
        auto it = opIndex.find(user);
        if (it != opIndex.end())
          lifetimeEnd[i] = std::max(lifetimeEnd[i], it->second);
      }
    }

    // Phase 2: allocate buffer slots with linear first-fit reuse.
    llvm::SmallVector<PoolEntry> pool;
    pool.reserve(n);
    llvm::SmallVector<int64_t> bufferIdVec(n), bufferOffsetVec(n),
        reuseGroupVec(n);
    int64_t nextOffset = 0;
    int64_t nextBufferId = 0;
    int64_t reusedCount = 0;

    for (int64_t i = 0; i < n; ++i) {
      PoolEntry *reuse =
          selectReusableBuffer(pool, lifetimeBegin[i], bytesVec[i]);
      if (reuse) {
        bufferIdVec[i] = reuse->id;
        bufferOffsetVec[i] = reuse->offset;
        reuseGroupVec[i] = reuse->id;
        reuse->freedAt = lifetimeEnd[i];
        ++reusedCount;
      } else {
        int64_t newId = nextBufferId++;
        pool.push_back({newId, nextOffset, bytesVec[i], lifetimeEnd[i]});
        bufferIdVec[i] = newId;
        bufferOffsetVec[i] = nextOffset;
        reuseGroupVec[i] = newId;
        nextOffset += bytesVec[i];
      }
    }

    // Compute peak live bytes: O(N^2) active-set scan over plannable ops.
    int64_t peakBytes = 0;
    for (int64_t t = 0; t < n; ++t) {
      int64_t active = 0;
      for (int64_t j = 0; j < n; ++j) {
        if (lifetimeBegin[j] <= t && t <= lifetimeEnd[j])
          active += bytesVec[j];
      }
      peakBytes = std::max(peakBytes, active);
    }

    // Set per-op attrs on plannable ops.
    auto i64 = IntegerType::get(ctx, 64);
    for (int64_t i = 0; i < n; ++i) {
      Operation *op = plannableOps[i];
      op->setAttr("cv.buffer_id", IntegerAttr::get(i64, bufferIdVec[i]));
      op->setAttr("cv.buffer_offset", IntegerAttr::get(i64, bufferOffsetVec[i]));
      op->setAttr("cv.lifetime_begin", IntegerAttr::get(i64, lifetimeBegin[i]));
      op->setAttr("cv.lifetime_end", IntegerAttr::get(i64, lifetimeEnd[i]));
      op->setAttr("cv.memory_plan.truth_boundary",
                  StringAttr::get(ctx, kTruthBoundary));
      op->setAttr("cv.memory_plan_reason", StringAttr::get(ctx, kPlanReason));
      op->setAttr("cv.reuse_group", IntegerAttr::get(i64, reuseGroupVec[i]));
    }

    // Set function-level summary attrs.
    funcOp->setAttr("cv.memory_plan.buffer_count",
                    IntegerAttr::get(i64, nextBufferId));
    funcOp->setAttr("cv.memory_plan.peak_memory_bytes",
                    IntegerAttr::get(i64, peakBytes));
    funcOp->setAttr("cv.memory_plan.planned_ops", IntegerAttr::get(i64, n));
    funcOp->setAttr("cv.memory_plan.reused_buffer_count",
                    IntegerAttr::get(i64, reusedCount));
    funcOp->setAttr("cv.memory_plan.skipped_ops",
                    IntegerAttr::get(i64, skippedOps));
    funcOp->setAttr("cv.memory_plan.status",
                    StringAttr::get(ctx, "completed"));
    funcOp->setAttr("cv.memory_plan.total_allocated_bytes",
                    IntegerAttr::get(i64, nextOffset));
    funcOp->setAttr("cv.memory_plan.truth_boundary",
                    StringAttr::get(ctx, kTruthBoundary));
  }
};

} // namespace

std::unique_ptr<Pass> createCVMemoryPlanningPass() {
  return std::make_unique<CVMemoryPlanningPass>();
}
} // namespace mlir::hir
