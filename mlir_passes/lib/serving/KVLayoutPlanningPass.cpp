#include "FusionPasses.h"
#include "QuantizationUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <memory>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_KVLAYOUTPLANNING
#include "FusionPasses.h.inc"

struct KVLayoutPlanningPass
    : impl::KVLayoutPlanningBase<KVLayoutPlanningPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();

    // Read model-level and target constraint attrs from the enclosing module.
    int64_t numLayers  = 0;
    int64_t hiddenSize = 0;
    double memoryBudgetMb = 0.0;
    bool hasBudgetConstraint = false;
    Operation *parent = funcOp->getParentOp();
    if (parent) {
      if (auto a = parent->getAttrOfType<IntegerAttr>("llm.num_layers"))
        numLayers = a.getInt();
      if (auto a = parent->getAttrOfType<IntegerAttr>("llm.hidden_size"))
        hiddenSize = a.getInt();
      // Optional: target.memory_budget_mb from a TargetLowering step.
      // Absent → unconstrained; existing producer/consumer logic governs.
      if (auto a = parent->getAttrOfType<FloatAttr>("target.memory_budget_mb")) {
        memoryBudgetMb = a.getValueAsDouble();
        hasBudgetConstraint = true;
      }
    }

    // Read plan dtype from QuantizationPlanningPass output (module attr).
    double dtypeBytes = dtypeBytesFromPlan(parent);

    // Walk for llm.attention_* ops and collect KV role + sequence dims.
    bool hasAttentionOp = false;
    bool isProducer     = false;
    int64_t promptTokens = 0;
    int64_t outputTokens = 0;

    funcOp.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "llm.attention_prefill" && name != "llm.attention_decode")
        return;
      hasAttentionOp = true;
      if (auto r = op->getAttrOfType<StringAttr>("kv_cache.role"))
        isProducer = (r.getValue() == "producer");
      if (auto pt = op->getAttrOfType<IntegerAttr>("serving.prompt_tokens"))
        promptTokens = pt.getInt();
      if (auto ot = op->getAttrOfType<IntegerAttr>("serving.output_tokens"))
        outputTokens = ot.getInt();
    });

    if (!hasAttentionOp)
      return;

    // KV byte estimate (static formula; not measured memory usage):
    //   num_layers * 2 (K+V) * hidden_size * dtype_bytes * seq_tokens / 1MB
    double kvMb = static_cast<double>(numLayers)
                * 2.0
                * static_cast<double>(hiddenSize)
                * dtypeBytes
                * static_cast<double>(promptTokens + outputTokens)
                / (1024.0 * 1024.0);

    // KV layout decision.
    // Budget constraint (from target.memory_budget_mb) takes priority over
    // producer/consumer role. Absent constraint → existing role-based logic.
    const bool budgetConstrained =
        hasBudgetConstraint && (kvMb > memoryBudgetMb * 0.4);
    StringRef layout = budgetConstrained
                           ? "contiguous"
                           : (isProducer ? "paged" : "contiguous");

    Type f64 = Float64Type::get(context);
    funcOp->setAttr("kv.layout", StringAttr::get(context, layout));
    if (budgetConstrained)
      funcOp->setAttr("kv.layout_reason",
                      StringAttr::get(context, "memory_budget_constrained"));
    funcOp->setAttr("kv.byte_estimate_mb",
                    FloatAttr::get(f64, kvMb));
    funcOp->setAttr("kv.dtype_bytes",
                    FloatAttr::get(f64, dtypeBytes));
    funcOp->setAttr("kv.truth_boundary",
                    StringAttr::get(context,
                        "static_formula_estimate_not_measured_memory"));
  }
};

} // namespace

std::unique_ptr<Pass> createKVLayoutPlanningPass() {
  return std::make_unique<KVLayoutPlanningPass>();
}

} // namespace mlir::hir
