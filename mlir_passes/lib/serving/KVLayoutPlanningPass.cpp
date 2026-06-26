#include "FusionPasses.h"

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

// fp16 dtype: 2 bytes per element.
static constexpr double kDtypeBytes = 2.0;

struct KVLayoutPlanningPass
    : impl::KVLayoutPlanningBase<KVLayoutPlanningPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();

    // Read model-level attrs from the enclosing module.
    int64_t numLayers  = 0;
    int64_t hiddenSize = 0;
    if (Operation *parent = funcOp->getParentOp()) {
      if (auto a = parent->getAttrOfType<IntegerAttr>("llm.num_layers"))
        numLayers = a.getInt();
      if (auto a = parent->getAttrOfType<IntegerAttr>("llm.hidden_size"))
        hiddenSize = a.getInt();
    }

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
                * kDtypeBytes
                * static_cast<double>(promptTokens + outputTokens)
                / (1024.0 * 1024.0);

    Type f64 = Float64Type::get(context);
    funcOp->setAttr("kv.layout",
                    StringAttr::get(context, isProducer ? "paged" : "contiguous"));
    funcOp->setAttr("kv.byte_estimate_mb",
                    FloatAttr::get(f64, kvMb));
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
