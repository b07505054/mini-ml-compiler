#include "FusionPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_CVEXECUTIONPLANATTRS
#include "FusionPasses.h.inc"

static constexpr llvm::StringLiteral kTruthBoundary =
    "real_yoloseg_execution_plan_compiler_decisions_from_static_capability_and_analysis_no_runtime_execution_no_measured_performance_no_full_memory_slot_allocation";

static bool hasCVSemanticAnnotation(func::FuncOp funcOp) {
  auto status =
      funcOp->getAttrOfType<StringAttr>("cv.semantic_annotation.status");
  return status && status.getValue() == "completed";
}

static std::vector<std::string> readStringArray(Operation *op,
                                                StringRef attrName) {
  std::vector<std::string> values;
  if (auto arr = op->getAttrOfType<ArrayAttr>(attrName)) {
    for (Attribute attr : arr)
      if (auto str = dyn_cast<StringAttr>(attr))
        values.push_back(str.getValue().str());
  }
  return values;
}

static std::string readString(Operation *op, StringRef attrName) {
  if (auto attr = op->getAttrOfType<StringAttr>(attrName))
    return attr.getValue().str();
  return {};
}

static std::string elementDType(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped)
    return {};
  Type element = shaped.getElementType();
  if (element.isF32())
    return "f32";
  if (element.isF16())
    return "f16";
  if (element.isBF16())
    return "bf16";
  if (element.isInteger(8))
    return "i8";
  if (element.isInteger(32))
    return "i32";
  return {};
}

static std::string resultDType(Operation &op) {
  for (Type type : op.getResultTypes()) {
    std::string dtype = elementDType(type);
    if (!dtype.empty())
      return dtype;
  }
  for (Type type : op.getOperandTypes()) {
    std::string dtype = elementDType(type);
    if (!dtype.empty())
      return dtype;
  }
  return "unknown";
}

static std::string layoutFor(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasRank())
    return "unknown";
  if (shaped.getRank() == 4)
    return "nchw";
  if (shaped.getRank() == 3 || shaped.getRank() == 2)
    return "row_major";
  return "ranked_tensor";
}

static std::string resultLayout(Operation &op) {
  for (Type type : op.getResultTypes()) {
    if (isa<ShapedType>(type))
      return layoutFor(type);
  }
  for (Type type : op.getOperandTypes()) {
    if (isa<ShapedType>(type))
      return layoutFor(type);
  }
  return "unknown";
}

static int64_t byteSize(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return 0;
  int64_t bytesPerElement = 0;
  Type element = shaped.getElementType();
  if (element.isF32() || element.isInteger(32))
    bytesPerElement = 4;
  else if (element.isF16() || element.isBF16())
    bytesPerElement = 2;
  else if (element.isInteger(8))
    bytesPerElement = 1;
  else
    return 0;

  int64_t elements = 1;
  for (int64_t dim : shaped.getShape())
    elements *= dim;
  return elements * bytesPerElement;
}

static bool hasTensorResult(Operation &op) {
  for (Type type : op.getResultTypes())
    if (isa<ShapedType>(type))
      return true;
  return false;
}

static std::string primaryBackend(Operation *moduleOp) {
  std::string preferred = readString(moduleOp, "target.preferred_backend");
  if (!preferred.empty())
    return preferred;
  std::vector<std::string> allowed =
      readStringArray(moduleOp, "target.allowed_backends");
  if (!allowed.empty())
    return allowed.front();
  return {};
}

static std::vector<std::string> fallbackBackends(Operation *moduleOp,
                                                 StringRef primary) {
  std::vector<std::string> result;
  for (const std::string &backend :
       readStringArray(moduleOp, "target.allowed_backends")) {
    if (backend != primary)
      result.push_back(backend);
  }
  return result;
}

static void setString(Operation *op, StringRef name, StringRef value) {
  op->setAttr(name, StringAttr::get(op->getContext(), value));
}

static void setBool(Operation *op, StringRef name, bool value) {
  op->setAttr(name, BoolAttr::get(op->getContext(), value));
}

static void setI64(Operation *op, StringRef name, int64_t value) {
  op->setAttr(name, IntegerAttr::get(IntegerType::get(op->getContext(), 64),
                                    value));
}

static ArrayAttr stringArray(MLIRContext *ctx,
                             const std::vector<std::string> &values) {
  SmallVector<Attribute> attrs;
  for (const std::string &value : values)
    attrs.push_back(StringAttr::get(ctx, value));
  return ArrayAttr::get(ctx, attrs);
}

struct CVExecutionPlanAttrsPass
    : impl::CVExecutionPlanAttrsBase<CVExecutionPlanAttrsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, arith::ArithDialect,
                    math::MathDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!hasCVSemanticAnnotation(funcOp))
      return;

    ModuleOp module = funcOp->getParentOfType<ModuleOp>();
    if (!module ||
        !module->getAttrOfType<StringAttr>("target.profile_id")) {
      funcOp.emitError()
          << "cv-execution-plan-attrs requires target.profile_id; run through "
             "a driver with an explicit target profile";
      signalPassFailure();
      return;
    }

    Operation *moduleOp = module.getOperation();
    std::string backend = primaryBackend(moduleOp);
    if (backend.empty()) {
      funcOp.emitError()
          << "cv-execution-plan-attrs requires target.preferred_backend or "
             "target.allowed_backends";
      signalPassFailure();
      return;
    }

    MLIRContext *ctx = funcOp.getContext();
    std::vector<std::string> fallback = fallbackBackends(moduleOp, backend);

    setString(funcOp, "serving.policy", "cv_full_graph");
    setString(funcOp, "serving.truth_boundary", kTruthBoundary);
    setString(funcOp, "execution_provider.primary", backend);
    funcOp->setAttr("execution_provider.fallback_chain",
                    stringArray(ctx, fallback));
    setString(funcOp, "execution_provider.decision_source",
              "cv-target-profile-static-policy");
    setString(funcOp, "execution_provider.required_precision", "f32");
    setString(funcOp, "execution_provider.required_kv_layout",
              "not_applicable");
    setBool(funcOp, "execution_provider.requires_replay", false);
    setString(funcOp, "execution_provider.truth_boundary",
              "compiler_execution_provider_plan_not_runtime_dispatch");

    setString(funcOp, "representation.effective_dtype", "f32");
    setString(funcOp, "representation.dtype_source",
              "cv_static_upstream_mlir_tensor_contract");
    setString(funcOp, "representation.preferred_activation_layout", "nchw");
    setString(funcOp, "representation.preferred_weight_layout", "fchw");
    setString(funcOp, "representation.source_backend", backend);
    setString(funcOp, "representation.truth_boundary", kTruthBoundary);

    setString(funcOp, "cv.execution_plan.status", "completed");
    setString(funcOp, "cv.execution_plan.pass_order",
              "cv-semantic-annotation,cv-execution-plan-attrs,generic-kernel-lowering-selection,ExecutionPlanBuilder");
    setString(funcOp, "cv.execution_plan.truth_boundary", kTruthBoundary);

    int64_t inputBytes = 0;
    for (Type type : funcOp.getFunctionType().getInputs())
      inputBytes += byteSize(type);

    int64_t outputBytes = 0;
    if (!funcOp.getBody().empty()) {
      if (auto ret = dyn_cast<func::ReturnOp>(
              funcOp.getBody().front().getTerminator())) {
        for (Value operand : ret.getOperands())
          outputBytes += byteSize(operand.getType());
      }
    }

    int64_t temporaryBytes = 0;
    if (!funcOp.getBody().empty()) {
      for (Operation &op : funcOp.getBody().front().without_terminator())
        for (Type type : op.getResultTypes())
          temporaryBytes += byteSize(type);
    }

    setI64(funcOp, "cv.memory.estimated_input_bytes", inputBytes);
    setI64(funcOp, "cv.memory.estimated_output_bytes", outputBytes);
    setI64(funcOp, "cv.memory.estimated_temporary_bytes", temporaryBytes);
    setI64(funcOp, "cv.memory.estimated_total_tensor_bytes",
           inputBytes + outputBytes + temporaryBytes);
    setString(funcOp, "cv.memory.status", "estimated_static_tensor_bytes");
    setString(funcOp, "cv.memory.truth_boundary",
              "static_tensor_byte_estimates_no_slot_allocation");

    if (funcOp.getBody().empty())
      return;

    for (Operation &op : funcOp.getBody().front().without_terminator()) {
      if (!hasTensorResult(op))
        continue;

      std::string dtype = resultDType(op);
      std::string layout = resultLayout(op);

      setString(&op, "layout.effective_layout", layout);
      setString(&op, "layout.required_input_layout", layout);
      setString(&op, "layout.layout_source",
                "cv_static_upstream_mlir_tensor_contract");
      setBool(&op, "layout.transform_required", false);
      setString(&op, "layout.truth_boundary",
                "static_layout_from_ranked_tensor_contract_no_transform");

      setString(&op, "quant.strategy", "none");
      setString(&op, "quant.weight_dtype", dtype);
      setString(&op, "quant.activation_dtype", dtype);
      setString(&op, "quant.accumulation_dtype",
                dtype == "f32" ? "f32" : "");
      setString(&op, "quant.granularity", "not_applicable");
      setString(&op, "quant.accuracy_risk", "none");
      setString(&op, "quant.decision_reason",
                "cv_phase24_no_quantization_configured");
      setString(&op, "quant.truth_boundary",
                "quantization_decision_none_no_calibration_no_quantized_execution");

      setString(&op, "kernel.backend", backend);
      setString(&op, "kernel.decision_reason",
                "seeded_cv_no_kernel_claim_generic_passes_may_refine");
      setBool(&op, "kernel.exists", false);
      setString(&op, "kernel.fallback_backend",
                fallback.empty() ? "" : fallback.front());
      setString(&op, "kernel.library", "");
      setString(&op, "kernel.lowering_status",
                fallback.empty() ? "unsupported" : "fallback_required");
      setString(&op, "kernel.name", "");
      setString(&op, "kernel.required_rewrite", "");
      setString(&op, "kernel.truth_boundary",
                "cv_seeded_kernel_status_not_runtime_kernel_availability");
    }
  }
};

} // namespace

std::unique_ptr<Pass> createCVExecutionPlanAttrsPass() {
  return std::make_unique<CVExecutionPlanAttrsPass>();
}

} // namespace mlir::hir
