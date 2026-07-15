#include "FusionPasses.h"
#include "serving/QuantizationCandidateProvider.h"

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

// Ops emitted by an importer (e.g. qwen-onnx-to-serving-mlir,
// qwen-to-serving-mlir) can declare themselves weight-bearing explicitly via
// serving.quantizable, instead of this pass inferring it from the op name.
// This is the preferred path: it lets frontend-specific naming (q_proj,
// mlp, qkv_projection, ...) stay behind the importer instead of leaking into
// this generic planning pass as name heuristics, so they still receive a
// quantization decision instead of silently vanishing from per_op_decisions.
static bool isQuantizableOp(Operation *op, const std::string &name) {
  if (auto a = op->getAttrOfType<BoolAttr>("serving.quantizable"))
    return a.getValue();

  // Fallback for ops with no explicit marker: match a small set of generic,
  // model-family-agnostic linear/conv op-name fragments. This list
  // intentionally excludes any frontend-specific naming convention such as
  // "proj" or "mlp".
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

static std::vector<int64_t> readI64Arr(Operation *op, llvm::StringRef attr) {
  std::vector<int64_t> out;
  if (auto a = op->getAttrOfType<ArrayAttr>(attr))
    for (Attribute elem : a)
      if (auto i = dyn_cast<IntegerAttr>(elem))
        out.push_back(i.getInt());
  return out;
}

static ArrayAttr makeStringArr(MLIRContext *ctx,
                               const std::vector<std::string> &values) {
  SmallVector<Attribute> attrs;
  for (const auto &value : values)
    attrs.push_back(StringAttr::get(ctx, value));
  return ArrayAttr::get(ctx, attrs);
}

static std::string normalizePlanDtype(std::string dtype) {
  if (dtype == "f32") return "fp32";
  if (dtype == "f16") return "fp16";
  if (dtype == "i8") return "int8";
  if (dtype == "i4") return "int4";
  return dtype;
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

    // Backend capability reads. Slice 1 fields are explicit: missing arrays
    // mean unknown/unsupported, never inferred support.
    std::string bcPrefix = "target.backend_capabilities." + backend + ".";
    std::vector<std::string> supportedQuantizationSchemes;
    std::vector<std::string> supportedActivationDtypes;
    std::vector<std::string> supportedWeightDtypes;
    std::vector<std::string> supportedAccumulatorDtypes;
    std::vector<std::string> supportedOutputDtypes;
    std::vector<std::string> allowedQuantGranularity;
    std::vector<int64_t> supportedGroupSizes;
    std::vector<std::string> calibrationAvailableSchemes;
    std::vector<std::string> requiredKernelCapabilities;
    bool supportsPerChannelQuantization = false;

    if (module && !backend.empty()) {
      supportedQuantizationSchemes = readStringArr(module, bcPrefix + "supported_quantization_schemes");
      supportedActivationDtypes = readStringArr(module, bcPrefix + "supported_activation_dtypes");
      supportedWeightDtypes = readStringArr(module, bcPrefix + "supported_weight_dtypes");
      supportedAccumulatorDtypes = readStringArr(module, bcPrefix + "supported_accumulator_dtypes");
      supportedOutputDtypes = readStringArr(module, bcPrefix + "supported_output_dtypes");
      allowedQuantGranularity = readStringArr(module, bcPrefix + "allowed_quant_granularity");
      supportedGroupSizes = readI64Arr(module, bcPrefix + "supported_quantization_group_sizes");
      calibrationAvailableSchemes = readStringArr(module, bcPrefix + "calibration_available_schemes");
      requiredKernelCapabilities = readStringArr(module, bcPrefix + "required_quantization_kernel_capabilities");
      if (auto a = module->getAttrOfType<BoolAttr>(bcPrefix + "supports_per_channel_quantization"))
        supportsPerChannelQuantization = a.getValue();
    }

    if (funcOp.getBody().empty()) return;
    Block &entry = funcOp.getBody().front();

    for (Operation &op : entry.without_terminator()) {
      std::string opName = stripPrefix(op.getName().getStringRef().str());
      bool quantizable = isQuantizableOp(&op, opName);
      bool accuracySensitive = isAccuracySensitive(opName);

      if (!quantizable && !accuracySensitive) continue;

      std::string resultDtype = getResultDtype(&op);

      std::string strategy, weightDtype, activationDtype, outputDtype, accumDtypeForOp;
      std::string quantMode, weightQuantMode, activationQuantMode;
      std::string granularity = "per_tensor";
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

      } else {
        QuantizationCapabilityContext qctx;
        qctx.semanticTargetRef = opName;
        qctx.scopeKind = CandidateScopeKind::Operator;
        if (module)
          if (auto a = module->getAttrOfType<StringAttr>("target.profile_id"))
            qctx.targetProfileId = a.getValue().str();
        qctx.backend = backend.empty() ? "cpu" : backend;
        qctx.activationDtype = "fp32";
        qctx.outputDtype = "fp32";
        qctx.supportedQuantizationSchemes = supportedQuantizationSchemes;
        qctx.supportedActivationDtypes = supportedActivationDtypes;
        qctx.supportedWeightDtypes = supportedWeightDtypes;
        qctx.supportedAccumulatorDtypes = supportedAccumulatorDtypes;
        qctx.supportedOutputDtypes = supportedOutputDtypes;
        qctx.supportedGranularities = allowedQuantGranularity;
        qctx.supportsPerChannel = supportsPerChannelQuantization;
        qctx.supportedGroupSizes = supportedGroupSizes;
        qctx.calibrationAvailableSchemes = calibrationAvailableSchemes;
        qctx.requiredKernelCapabilities = requiredKernelCapabilities;

        StringRef fullName = op.getName().getStringRef();
        StringRef shortName = fullName;
        if (auto dot = fullName.find('.'); dot != StringRef::npos)
          shortName = fullName.substr(dot + 1);
        if (module) {
          if (auto registry = module->getAttrOfType<ArrayAttr>("target.runtime_kernels")) {
            for (Attribute elem : registry) {
              auto dict = dyn_cast<DictionaryAttr>(elem);
              if (!dict) continue;
              auto getS = [&](StringRef key) -> std::string {
                if (auto a = dict.get(key))
                  if (auto st = dyn_cast<StringAttr>(a)) return st.getValue().str();
                return {};
              };
              auto getArr = [&](StringRef key) -> std::vector<std::string> {
                std::vector<std::string> values;
                if (auto a = dict.get(key))
                  if (auto arr = dyn_cast<ArrayAttr>(a))
                    for (Attribute item : arr)
                      if (auto st = dyn_cast<StringAttr>(item))
                        values.push_back(st.getValue().str());
                return values;
              };
              if (getS("op_name") != shortName.str()) continue;
              if (getS("backend") != qctx.backend) continue;
              qctx.kernelId = getS("kernel_id");
              qctx.runtimeKernelQuantModes = getArr("supported_quant_modes");
              qctx.runtimeKernelDtypes = getArr("supported_dtypes");
              break;
            }
          }
        }
        if (qctx.requiredKernelCapabilities.empty() &&
            quantListContains(qctx.runtimeKernelQuantModes, "none"))
          qctx.requiredKernelCapabilities.push_back("quant_kernel.none");

        QuantizationCandidateProvider provider;
        QuantizationProviderResult qres = provider.enumerateAndSelect(qctx);
        const ImplementationCandidate *selected = nullptr;
        for (const auto &candidate : qres.candidates)
          if (candidate.candidateId == qres.policy.selectedCandidateId)
            selected = &candidate;
        if (!selected) {
          strategy = "unsupported";
          weightDtype = "";
          activationDtype = "";
          outputDtype = "";
          accumDtypeForOp = "";
          quantMode = "unsupported";
          weightQuantMode = "unsupported";
          activationQuantMode = "unsupported";
          decisionReason = "no_legal_quantization_candidate";
          fallbackReason = "no_legal_quantization_candidate";
          accuracyRisk = "unknown";
        } else {
          strategy = selected->quantization.scheme;
          granularity = selected->quantization.granularity;
          weightDtype = selected->quantization.weightDtype;
          activationDtype = selected->quantization.activationDtype;
          outputDtype = selected->quantization.outputDtype;
          accumDtypeForOp = selected->quantization.accumulatorDtype;
          quantMode = selected->quantization.scheme == "fp32_baseline" ? "none" : selected->quantization.scheme;
          weightQuantMode = selected->quantization.scheme;
          activationQuantMode = selected->quantization.scheme == "int8_static" ? "int8_static" : "none";
          decisionReason = qres.policy.selectionReason;
          fallbackReason = selected->quantization.scheme == "fp32_baseline" ? "fallback_fp32_selected" : "";
          accuracyRisk = selected->quantization.calibrationRequired &&
                         !selected->quantization.calibrationAvailable
                             ? "unknown"
                             : "low";
        }
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
      auto copyModuleString = [&](llvm::StringRef src, llvm::StringRef dst) {
        if (module)
          if (auto a = module->getAttrOfType<StringAttr>(src))
            op.setAttr(dst, a);
      };
      auto copyModuleFloat = [&](llvm::StringRef src, llvm::StringRef dst) {
        if (module)
          if (auto a = module->getAttrOfType<FloatAttr>(src))
            op.setAttr(dst, a);
      };
      auto copyModuleInt = [&](llvm::StringRef src, llvm::StringRef dst) {
        if (module)
          if (auto a = module->getAttrOfType<IntegerAttr>(src))
            op.setAttr(dst, a);
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

      if (quantizable && isConstant && !accuracySensitive) {
        QuantizationCapabilityContext qctx;
        qctx.semanticTargetRef = opName;
        qctx.scopeKind = CandidateScopeKind::Operator;
        if (module)
          if (auto a = module->getAttrOfType<StringAttr>("target.profile_id"))
            qctx.targetProfileId = a.getValue().str();
        qctx.backend = backend.empty() ? "cpu" : backend;
        qctx.supportedQuantizationSchemes = supportedQuantizationSchemes;
        qctx.supportedActivationDtypes = supportedActivationDtypes;
        qctx.supportedWeightDtypes = supportedWeightDtypes;
        qctx.supportedAccumulatorDtypes = supportedAccumulatorDtypes;
        qctx.supportedOutputDtypes = supportedOutputDtypes;
        qctx.supportedGranularities = allowedQuantGranularity;
        qctx.supportsPerChannel = supportsPerChannelQuantization;
        qctx.supportedGroupSizes = supportedGroupSizes;
        qctx.calibrationAvailableSchemes = calibrationAvailableSchemes;
        qctx.requiredKernelCapabilities = requiredKernelCapabilities;
        StringRef fullName = op.getName().getStringRef();
        StringRef shortName = fullName;
        if (auto dot = fullName.find('.'); dot != StringRef::npos)
          shortName = fullName.substr(dot + 1);
        if (module) {
          if (auto registry = module->getAttrOfType<ArrayAttr>("target.runtime_kernels")) {
            for (Attribute elem : registry) {
              auto dict = dyn_cast<DictionaryAttr>(elem);
              if (!dict) continue;
              auto getS = [&](StringRef key) -> std::string {
                if (auto a = dict.get(key))
                  if (auto st = dyn_cast<StringAttr>(a)) return st.getValue().str();
                return {};
              };
              auto getArr = [&](StringRef key) -> std::vector<std::string> {
                std::vector<std::string> values;
                if (auto a = dict.get(key))
                  if (auto arr = dyn_cast<ArrayAttr>(a))
                    for (Attribute item : arr)
                      if (auto st = dyn_cast<StringAttr>(item))
                        values.push_back(st.getValue().str());
                return values;
              };
              if (getS("op_name") != shortName.str()) continue;
              if (getS("backend") != qctx.backend) continue;
              qctx.kernelId = getS("kernel_id");
              qctx.runtimeKernelQuantModes = getArr("supported_quant_modes");
              qctx.runtimeKernelDtypes = getArr("supported_dtypes");
              break;
            }
          }
        }
        if (qctx.requiredKernelCapabilities.empty() &&
            quantListContains(qctx.runtimeKernelQuantModes, "none"))
          qctx.requiredKernelCapabilities.push_back("quant_kernel.none");
        QuantizationProviderResult qres =
            QuantizationCandidateProvider().enumerateAndSelect(qctx);
        const ImplementationCandidate *selected = nullptr;
        for (const auto &candidate : qres.candidates)
          if (candidate.candidateId == qres.policy.selectedCandidateId)
            selected = &candidate;
        if (selected) {
          set("quant.selected_candidate_id", selected->candidateId);
          set("quant.scheme", selected->quantization.scheme);
          set("quant.required_backend_capability", selected->quantization.requiredBackendCapability);
          set("quant.required_kernel_capability", selected->quantization.requiredKernelCapability);
          if (!selected->kernelId.empty()) set("quant.kernel_id", selected->kernelId);
          if (selected->quantization.scheme == "int8_static_symmetric") {
            set("quant.activation_granularity", "per_tensor");
            set("quant.weight_granularity", "per_tensor");
            copyModuleString("quant.slice3a.calibration_artifact_ref", "quant.calibration_artifact_ref");
            copyModuleString("quant.slice3a.calibration_artifact_id", "quant.calibration_artifact_id");
            copyModuleString("quant.slice3a.calibration_artifact_sha256", "quant.calibration_artifact_sha256");
            copyModuleString("quant.slice3a.workload_id", "quant.workload_id");
            copyModuleFloat("quant.slice3a.activation_scale", "quant.activation_scale");
            copyModuleFloat("quant.slice3a.weight_scale", "quant.weight_scale");
            copyModuleInt("quant.slice3a.activation_zero_point", "quant.activation_zero_point");
            copyModuleInt("quant.slice3a.weight_zero_point", "quant.weight_zero_point");
          }
          setBool("quant.requires_calibration", selected->quantization.calibrationRequired);
          setBool("quant.calibration_available", selected->quantization.calibrationAvailable);
          if (selected->quantization.groupSize > 0)
            op.setAttr("quant.group_size",
                       IntegerAttr::get(IntegerType::get(ctx, 64),
                                        selected->quantization.groupSize));
        }
        set("quant.policy_id", qres.policy.policyId);
        set("quant.selection_reason", qres.policy.selectionReason);
        set("quant.considered_status", qres.policy.selectedCandidateId.empty() ? "unsupported" : "selected");
        op.setAttr("quant.considered_candidate_ids", makeStringArr(ctx, qres.policy.consideredCandidateIds));
        std::vector<std::string> rejectedIds;
        std::vector<std::string> rejectedReasons;
        for (const auto &rej : qres.policy.rejectedCandidates) {
          rejectedIds.push_back(rej.candidateId);
          rejectedReasons.push_back(rej.reason);
        }
        op.setAttr("quant.rejected_candidate_ids", makeStringArr(ctx, rejectedIds));
        op.setAttr("quant.rejected_candidate_reasons", makeStringArr(ctx, rejectedReasons));
      }
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createQuantizationStrategyPlanningPass() {
  return std::make_unique<QuantizationStrategyPlanningPass>();
}

} // namespace mlir::hir
