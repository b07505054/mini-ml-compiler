#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_QUANTIZATIONSTRATEGYPLANNING
#include "FusionPasses.h.inc"

static constexpr StringLiteral kTruth =
    "quantization_strategy_static_not_accuracy_calibrated";

static std::string stripPrefix(llvm::StringRef name) {
  auto pos = name.rfind('.');
  if (pos == llvm::StringRef::npos) return name.str();
  return name.substr(pos + 1).str();
}

static std::string getResultDtype(Operation *op) {
  if (op->getNumResults() == 0) return "";
  Type t = op->getResult(0).getType();
  if (auto tt = dyn_cast<TensorType>(t)) {
    Type elem = tt.getElementType();
    if (elem.isF32())       return "fp32";
    if (elem.isF16())       return "fp16";
    if (elem.isBF16())      return "bf16";
    if (elem.isInteger(8))  return "int8";
    if (elem.isInteger(4))  return "int4";
    if (elem.isInteger(32)) return "int32";
  }
  return "";
}

static bool isQuantizableOp(const std::string &name) {
  static const char *kNames[] = {
      "matmul", "conv", "linear", "dense", "gemm", "dot", "fc",
      "fully_connected", "batch_gemm", "fused_matmul",
  };
  for (const char *n : kNames)
    if (name.find(n) != std::string::npos) return true;
  return false;
}

static bool isAccuracySensitive(const std::string &name) {
  static const char *kNames[] = {
      "softmax", "norm", "normalize", "rmsnorm", "layernorm",
      "batchnorm", "groupnorm", "instancenorm", "layer_norm",
      "group_norm", "batch_norm", "instance_norm", "rms_norm", "embedding",
  };
  for (const char *n : kNames)
    if (name.find(n) != std::string::npos) return true;
  return false;
}

static bool isQuantizedDtype(const std::string &dtype) {
  return dtype == "int8" || dtype == "int4" || dtype == "i8" || dtype == "i4";
}

static bool isFloatDtype(const std::string &dtype) {
  return dtype == "fp16" || dtype == "fp32" || dtype == "bf16" ||
         dtype == "f16" || dtype == "f32";
}

static std::vector<std::string> readStringArr(Operation *op, llvm::StringRef attr) {
  std::vector<std::string> out;
  if (auto a = op->getAttrOfType<ArrayAttr>(attr))
    for (Attribute elem : a)
      if (auto s = dyn_cast<StringAttr>(elem))
        out.push_back(s.getValue().str());
  return out;
}

struct QuantizationStrategyPlanningPass
    : impl::QuantizationStrategyPlanningBase<QuantizationStrategyPlanningPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    Operation *module = funcOp->getParentOp();

    // Backend name from representation pass.
    std::string backend;
    if (auto a = funcOp->getAttrOfType<StringAttr>("representation.source_backend"))
      backend = a.getValue().str();

    // Effective dtype (normalize legacy "f16" -> "fp16").
    std::string effectiveDtype = "fp16";
    if (auto a = funcOp->getAttrOfType<StringAttr>("representation.effective_dtype")) {
      effectiveDtype = a.getValue().str();
      if (effectiveDtype == "f16")  effectiveDtype = "fp16";
      if (effectiveDtype == "f32")  effectiveDtype = "fp32";
      if (effectiveDtype == "i8")   effectiveDtype = "int8";
    }

    // Function-level weight-constant state.
    // representation.weights_are_constant = true  → explicit override for all ops.
    // representation.weights_are_constant = false → old explicit non-constant path.
    // absent                                      → falls through to per-op signal below.
    bool funcWeightsConstant = false;
    bool weightsExplicitlyNonConstant = false;
    if (auto a = funcOp->getAttrOfType<BoolAttr>("representation.weights_are_constant")) {
      if (a.getValue()) funcWeightsConstant = true;
      else              weightsExplicitlyNonConstant = true;
    }

    // Backend capability reads.
    std::string bcPrefix = "target.backend_capabilities." + backend + ".";
    std::vector<std::string> supportedQuantModes;
    std::vector<std::string> supportedDtypes;
    std::vector<std::string> accumDtypes;
    std::vector<std::string> allowedQuantGranularity;
    std::string reqActivationQuantMode;
    std::string reqWeightQuantMode;
    bool backendSupportsDequant = false;

    if (module && !backend.empty()) {
      supportedQuantModes    = readStringArr(module, bcPrefix + "supported_quant_modes");
      supportedDtypes        = readStringArr(module, bcPrefix + "supported_dtypes");
      accumDtypes            = readStringArr(module, bcPrefix + "accumulation_dtypes");
      allowedQuantGranularity = readStringArr(module, bcPrefix + "allowed_quant_granularity");
      if (auto a = module->getAttrOfType<StringAttr>(bcPrefix + "required_activation_quant_mode"))
        reqActivationQuantMode = a.getValue().str();
      if (auto a = module->getAttrOfType<StringAttr>(bcPrefix + "required_weight_quant_mode"))
        reqWeightQuantMode = a.getValue().str();
      if (auto a = module->getAttrOfType<BoolAttr>(bcPrefix + "supports_dequant_boundary"))
        backendSupportsDequant = a.getValue();
    }

    // Derive capability flags.
    bool hasActivationInt8 = false;
    bool hasWeightInt8 = false;
    for (const auto &mode : supportedQuantModes) {
      if (mode == "static_int8") { hasActivationInt8 = true; hasWeightInt8 = true; }
      if (mode == "weight_only") { hasWeightInt8 = true; }
    }
    for (const auto &dtype : supportedDtypes)
      if (dtype == "int8" || dtype == "i8") hasWeightInt8 = true;

    std::string granularity = !allowedQuantGranularity.empty()
        ? allowedQuantGranularity[0] : "per_channel";
    std::string accumDtype = !accumDtypes.empty() ? accumDtypes[0] : "fp32";

    if (funcOp.getBody().empty()) return;
    Block &entry = funcOp.getBody().front();

    for (Operation &op : entry.without_terminator()) {
      std::string opName = stripPrefix(op.getName().getStringRef().str());
      bool quantizable = isQuantizableOp(opName);
      bool accuracySensitive = isAccuracySensitive(opName);

      if (!quantizable && !accuracySensitive) continue;

      std::string resultDtype = getResultDtype(&op);

      std::string strategy, weightDtype, activationDtype, outputDtype, accumDtypeForOp;
      std::string quantMode, weightQuantMode, activationQuantMode;
      std::string decisionReason, fallbackReason, accuracyRisk;
      bool requiresDequantBoundary = false;
      bool requiresRequantBoundary = false;

      // Propagate dequant signal from BoundaryPlanningPass if already set.
      if (auto a = op.getAttrOfType<BoolAttr>("boundary.dequant_required"))
        requiresDequantBoundary = a.getValue();

      // Per-op weight classification from WeightClassificationPlanningPass.
      // When present, overrides the function-level optimistic default.
      bool hasOpConstSat = false;
      bool opConstSatisfied = false;
      bool opWeightUnknown = false;
      if (auto a = op.getAttrOfType<BoolAttr>("weight.constant_satisfied")) {
        hasOpConstSat = true;
        opConstSatisfied = a.getValue();
      }
      if (hasOpConstSat && !opConstSatisfied) {
        if (auto a = op.getAttrOfType<StringAttr>("weight.classification"))
          if (a.getValue() == "unknown") opWeightUnknown = true;
      }

      // Effective constant status (priority: func-level override > per-op > old optimistic).
      bool isConstant;
      bool isUnknownConst;
      if (funcWeightsConstant) {
        isConstant = true;  isUnknownConst = false;
      } else if (hasOpConstSat) {
        isConstant = opConstSatisfied;
        isUnknownConst = !opConstSatisfied && opWeightUnknown;
      } else if (weightsExplicitlyNonConstant) {
        isConstant = false; isUnknownConst = false;
      } else {
        // WeightClassificationPlanningPass did not run (standalone pipeline).
        // Preserve old optimistic default so existing tests pass.
        isConstant = true;  isUnknownConst = false;
      }

      if (accuracySensitive) {
        // Rule 5: accuracy-sensitive ops always fall back to fp16.
        strategy = "fp16_fallback";
        weightDtype = effectiveDtype;
        activationDtype = effectiveDtype;
        outputDtype = !resultDtype.empty() ? resultDtype : effectiveDtype;
        accumDtypeForOp = effectiveDtype;
        quantMode = "none";
        weightQuantMode = "none";
        activationQuantMode = "none";
        decisionReason = "accuracy_sensitive_op";
        fallbackReason = "accuracy_sensitive_op";
        accuracyRisk = "medium";

      } else if (!isConstant && isUnknownConst) {
        // weight.classification = "unknown": weight may not be constant; conservative fallback.
        strategy = "fp16_fallback";
        weightDtype = effectiveDtype;
        activationDtype = effectiveDtype;
        outputDtype = !resultDtype.empty() ? resultDtype : effectiveDtype;
        accumDtypeForOp = effectiveDtype;
        quantMode = "none";
        weightQuantMode = "none";
        activationQuantMode = "none";
        decisionReason = "weight_classification_unknown";
        fallbackReason = "weight_constant_unknown";
        accuracyRisk = "unknown";

      } else if (!isConstant && weightsExplicitlyNonConstant) {
        // Func-level explicit non-constant (representation.weights_are_constant = false).
        // Keeps legacy fallback_reason for backward compat with existing tests.
        strategy = "fp16_fallback";
        weightDtype = effectiveDtype;
        activationDtype = effectiveDtype;
        outputDtype = !resultDtype.empty() ? resultDtype : effectiveDtype;
        accumDtypeForOp = effectiveDtype;
        quantMode = "none";
        weightQuantMode = "none";
        activationQuantMode = "none";
        decisionReason = "weight_not_constant";
        fallbackReason = "requires_constant_weight";
        accuracyRisk = "low";

      } else if (!isConstant) {
        // Per-op weight.constant_satisfied = false from WeightClassificationPlanningPass.
        strategy = "fp16_fallback";
        weightDtype = effectiveDtype;
        activationDtype = effectiveDtype;
        outputDtype = !resultDtype.empty() ? resultDtype : effectiveDtype;
        accumDtypeForOp = effectiveDtype;
        quantMode = "none";
        weightQuantMode = "none";
        activationQuantMode = "none";
        decisionReason = "weight_classification_runtime_activation";
        fallbackReason = "weight_not_constant";
        accuracyRisk = "low";

      } else if (hasWeightInt8) {
        // Rules 1, 2, 3: backend supports int8 weight and weights are constant.
        // Rule 1: always prefer weight-only (fp16 activations) over full int8.
        // Rule 2: int8 weight when backend supports it.
        // Rule 3: fp16 activation when activation int8 is absent or unsafe.
        strategy = "weight_only_int8";
        weightDtype = "int8";
        activationDtype = effectiveDtype;   // fp16 per Rule 1+3
        outputDtype = !resultDtype.empty() ? resultDtype : effectiveDtype;
        // Rule 4: use int32 accumulation for int8 matmul/conv when declared.
        accumDtypeForOp = accumDtype;
        quantMode = "weight_only";
        weightQuantMode = reqWeightQuantMode.empty() ? "weight_only_int8" : reqWeightQuantMode;
        activationQuantMode = reqActivationQuantMode.empty() ? "none" : reqActivationQuantMode;
        decisionReason = hasActivationInt8
            ? "weight_only_int8_prefer_weight_only_per_rule1"
            : "weight_only_int8_backend_lacks_activation_int8";
        fallbackReason = "";
        accuracyRisk = "unknown";  // Rule 10: no calibration

      } else {
        // Backend lacks all int8 support.
        strategy = "fp16_fallback";
        weightDtype = effectiveDtype;
        activationDtype = effectiveDtype;
        outputDtype = !resultDtype.empty() ? resultDtype : effectiveDtype;
        accumDtypeForOp = effectiveDtype;
        quantMode = "none";
        weightQuantMode = "none";
        activationQuantMode = "none";
        decisionReason = "backend_lacks_activation_int8";
        fallbackReason = "backend_lacks_activation_int8";
        accuracyRisk = "low";
      }

      // Rule 6: op produces quantized output but func effective dtype is float → dequant boundary.
      if (isQuantizedDtype(resultDtype) && isFloatDtype(effectiveDtype))
        requiresDequantBoundary = true;

      auto set = [&](llvm::StringRef k, llvm::StringRef v) {
        op.setAttr(k, StringAttr::get(ctx, v));
      };
      auto setBool = [&](llvm::StringRef k, bool v) {
        op.setAttr(k, BoolAttr::get(ctx, v));
      };

      set("quant.strategy",              strategy);
      set("quant.weight_dtype",          weightDtype);
      set("quant.activation_dtype",      activationDtype);
      set("quant.accumulation_dtype",    accumDtypeForOp);
      set("quant.output_dtype",          outputDtype);
      set("quant.granularity",           granularity);
      set("quant.activation_quant_mode", activationQuantMode);
      set("quant.weight_quant_mode",     weightQuantMode);
      setBool("quant.requires_dequant_boundary",  requiresDequantBoundary);
      setBool("quant.requires_requant_boundary",  requiresRequantBoundary);
      set("quant.backend",               backend);
      set("quant.decision_reason",       decisionReason);
      set("quant.fallback_reason",       fallbackReason);
      set("quant.accuracy_risk",         accuracyRisk);
      set("quant.truth_boundary",        kTruth);
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createQuantizationStrategyPlanningPass() {
  return std::make_unique<QuantizationStrategyPlanningPass>();
}

} // namespace mlir::hir
