#include "FusionPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#define GEN_PASS_DEF_WEIGHTCLASSIFICATIONPLANNING
#include "FusionPasses.h.inc"

namespace mlir::hir {
namespace {

static constexpr StringLiteral kTruth =
    "weight_classification_static_def_use_analysis_not_runtime_validated";

static bool isQuantizableOp(llvm::StringRef shortName) {
  static const char *kNames[] = {
      "matmul", "conv", "linear", "dense", "gemm", "dot", "fc",
      "fully_connected", "batch_gemm", "fused_matmul",
  };
  std::string name = shortName.str();
  for (const char *n : kNames)
    if (name.find(n) != std::string::npos) return true;
  return false;
}

enum class WeightKind { ConstantWeight, RuntimeActivation, Unknown };

// Classify a single value by walking one level of the def-use graph.
static WeightKind classifyOperand(mlir::Value v, llvm::StringRef &reason) {
  // Rule 3: function block argument → runtime.
  if (mlir::isa<mlir::BlockArgument>(v)) {
    reason = "function_argument_is_runtime";
    return WeightKind::RuntimeActivation;
  }

  mlir::Operation *def = v.getDefiningOp();
  if (!def) {
    reason = "no_defining_op";
    return WeightKind::Unknown;
  }

  // Rule 4: defining op declares weight.is_constant = true.
  if (auto a = def->getAttrOfType<mlir::BoolAttr>("weight.is_constant")) {
    if (a.getValue()) {
      reason = "declared_constant_weight_attr";
      return WeightKind::ConstantWeight;
    }
    reason = "declared_not_constant";
    return WeightKind::RuntimeActivation;
  }

  // Rule 5: standard constant ops.
  if (def->getName().getStringRef() == "arith.constant") {
    reason = "arith_constant_op";
    return WeightKind::ConstantWeight;
  }
  if (def->hasTrait<mlir::OpTrait::ConstantLike>()) {
    reason = "constant_like_trait";
    return WeightKind::ConstantWeight;
  }

  // Rule 6: known compute dialects → runtime.
  llvm::StringRef opName = def->getName().getStringRef();
  if (opName.starts_with("hir.") || opName.starts_with("llm.") ||
      opName.starts_with("cv.")) {
    reason = "compute_op_produces_runtime_value";
    return WeightKind::RuntimeActivation;
  }
  if (opName.starts_with("arith.")) {
    // Non-constant arith op (e.g. arith.mulf) → runtime.
    reason = "arith_compute_op";
    return WeightKind::RuntimeActivation;
  }
  if (opName.starts_with("func.")) {
    reason = "func_op_is_runtime";
    return WeightKind::RuntimeActivation;
  }

  // Rule 7: everything else → unknown (conservative).
  reason = "unknown_producer_conservative";
  return WeightKind::Unknown;
}

struct WeightClassificationPlanningPass
    : ::impl::WeightClassificationPlanningBase<WeightClassificationPlanningPass> {
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext *ctx  = &getContext();

    if (func.getBody().empty()) return;

    auto S = [&](llvm::StringRef s) { return mlir::StringAttr::get(ctx, s); };
    auto B = [&](bool b)            { return mlir::BoolAttr::get(ctx, b); };

    // Rule 1: function-level override.
    bool funcLevelConstant = false;
    if (auto a = func->getAttrOfType<mlir::BoolAttr>("representation.weights_are_constant"))
      funcLevelConstant = a.getValue();

    mlir::Block &entry = func.getBody().front();
    for (mlir::Operation &op : entry.without_terminator()) {
      llvm::StringRef fullName = op.getName().getStringRef();
      std::string shortName   = fullName.str();
      if (auto dot = fullName.find('.'); dot != llvm::StringRef::npos)
        shortName = fullName.substr(dot + 1).str();

      if (!isQuantizableOp(shortName)) continue;

      // Rule 1: function-level constant declaration.
      if (funcLevelConstant) {
        op.setAttr("weight.classification",        S("constant_weight"));
        op.setAttr("weight.constant_required",     B(true));
        op.setAttr("weight.constant_satisfied",    B(true));
        op.setAttr("weight.classification_reason", S("function_level_weights_are_constant"));
        op.setAttr("weight.truth_boundary",        S(kTruth));
        continue;
      }

      // Rule 2: op-level pre-annotation.
      if (auto a = op.getAttrOfType<mlir::BoolAttr>("weight.is_constant")) {
        bool isConst = a.getValue();
        op.setAttr("weight.classification",
                   S(isConst ? "constant_weight" : "runtime_activation"));
        op.setAttr("weight.constant_required",     B(true));
        op.setAttr("weight.constant_satisfied",    B(isConst));
        op.setAttr("weight.classification_reason", S("declared_constant_weight_attr_on_op"));
        op.setAttr("weight.truth_boundary",        S(kTruth));
        continue;
      }

      // Identify weight operand: RHS (operand 1) for binary ops; operand 0 for unary.
      mlir::Value weightOperand;
      if (op.getNumOperands() >= 2)
        weightOperand = op.getOperand(1);
      else if (op.getNumOperands() == 1)
        weightOperand = op.getOperand(0);

      if (!weightOperand) {
        op.setAttr("weight.classification",        S("unknown"));
        op.setAttr("weight.constant_required",     B(true));
        op.setAttr("weight.constant_satisfied",    B(false));
        op.setAttr("weight.classification_reason", S("no_weight_operand"));
        op.setAttr("weight.truth_boundary",        S(kTruth));
        continue;
      }

      llvm::StringRef reason;
      WeightKind kind = classifyOperand(weightOperand, reason);

      llvm::StringRef classification;
      bool satisfied;
      switch (kind) {
        case WeightKind::ConstantWeight:
          classification = "constant_weight";
          satisfied      = true;
          break;
        case WeightKind::RuntimeActivation:
          classification = "runtime_activation";
          satisfied      = false;
          break;
        default:
          classification = "unknown";
          satisfied      = false;
          break;
      }

      op.setAttr("weight.classification",        S(classification));
      op.setAttr("weight.constant_required",     B(true));
      op.setAttr("weight.constant_satisfied",    B(satisfied));
      op.setAttr("weight.classification_reason", S(reason));
      op.setAttr("weight.truth_boundary",        S(kTruth));
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createWeightClassificationPlanningPass() {
  return std::make_unique<WeightClassificationPlanningPass>();
}

} // namespace mlir::hir
