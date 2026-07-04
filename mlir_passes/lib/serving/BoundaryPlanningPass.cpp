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

#define GEN_PASS_DEF_BOUNDARYPLANNING
#include "FusionPasses.h.inc"

// Normalize planner dtype strings ("fp16", "int8") to MLIR element type names ("f16", "i8").
static std::string normalizeDtype(const std::string &s) {
  if (s == "fp16" || s == "f16")  return "f16";
  if (s == "fp32" || s == "f32")  return "f32";
  if (s == "bf16")                 return "bf16";
  if (s == "int8" || s == "i8")   return "i8";
  if (s == "int4" || s == "i4")   return "i4";
  return s;
}

// Return the element dtype string of the first result of op, or "" if not a tensor.
static std::string getResultElementDtype(Operation *op) {
  if (op->getNumResults() == 0) return "";
  Type t = op->getResult(0).getType();
  if (auto tt = dyn_cast<TensorType>(t)) {
    Type elem = tt.getElementType();
    if (elem.isF32())        return "f32";
    if (elem.isF16())        return "f16";
    if (elem.isBF16())       return "bf16";
    if (elem.isInteger(8))   return "i8";
    if (elem.isInteger(4))   return "i4";
    if (elem.isInteger(32))  return "i32";
  }
  return "";
}

static bool isFloat(const std::string &dtype) {
  return dtype == "f16" || dtype == "f32" || dtype == "bf16";
}

static bool isQuantized(const std::string &dtype) {
  return dtype == "i8" || dtype == "i4";
}

// Build a comma-separated reason string from the decided and unsupported conditions.
static std::string buildReason(bool castReq, bool layoutReq, bool dequantReq,
                                bool castUnsup, bool layoutUnsup, bool dequantUnsup) {
  std::vector<std::string> parts;
  if (castReq)      parts.push_back("cast");
  if (layoutReq)    parts.push_back("layout_transform");
  if (dequantReq)   parts.push_back("dequant");
  if (castUnsup)    parts.push_back("cast_unsupported");
  if (layoutUnsup)  parts.push_back("layout_transform_unsupported");
  if (dequantUnsup) parts.push_back("dequant_unsupported");
  if (parts.empty()) return "no_boundaries";
  std::string result = parts[0];
  for (size_t i = 1; i < parts.size(); ++i)
    result += "," + parts[i];
  return result;
}

struct BoundaryPlanningPass
    : impl::BoundaryPlanningBase<BoundaryPlanningPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    Operation *module = funcOp->getParentOp();

    // Skip funcs that have no representation planning annotation.
    auto effectiveDtypeAttr =
        funcOp->getAttrOfType<StringAttr>("representation.effective_dtype");
    if (!effectiveDtypeAttr) return;
    std::string effectiveDtype = normalizeDtype(effectiveDtypeAttr.getValue().str());

    // Read source backend.
    std::string backend;
    if (auto a = funcOp->getAttrOfType<StringAttr>("representation.source_backend"))
      backend = a.getValue().str();

    // Read backend capability flags.
    bool supportsCast = false, supportsDequant = false, supportsLayoutTransform = false;
    if (module && !backend.empty()) {
      std::string prefix = "target.backend_capabilities." + backend + ".";
      if (auto a = module->getAttrOfType<BoolAttr>(prefix + "supports_cast"))
        supportsCast = a.getValue();
      if (auto a = module->getAttrOfType<BoolAttr>(prefix + "supports_dequant_boundary"))
        supportsDequant = a.getValue();
      if (auto a = module->getAttrOfType<BoolAttr>(prefix + "supports_layout_transform"))
        supportsLayoutTransform = a.getValue();
    }

    static constexpr StringLiteral kTruthBoundary =
        "boundary_planning_only_no_ir_materialization";

    if (funcOp.getBody().empty()) return;
    Block &entry = funcOp.getBody().front();

    for (Operation &op : entry.without_terminator()) {
      std::string opDtype = getResultElementDtype(&op);

      // Read layout.transform_required annotated by LayoutPlanningPass.
      bool layoutTransformNeeded = false;
      if (auto a = op.getAttrOfType<BoolAttr>("layout.transform_required"))
        layoutTransformNeeded = a.getValue();

      // Determine what boundaries are needed.
      // Cast covers float-to-float precision changes only; int↔float is handled by dequant.
      bool needsCast = !opDtype.empty() && !effectiveDtype.empty()
                       && isFloat(opDtype) && isFloat(effectiveDtype)
                       && opDtype != effectiveDtype;
      bool needsLayoutTransform = layoutTransformNeeded;
      // Dequant boundary: planned dtype is quantized but this op outputs float.
      bool needsDequant = isQuantized(effectiveDtype) && isFloat(opDtype);

      // Apply backend support gates.
      bool castRequired           = needsCast && supportsCast;
      bool layoutTransformRequired = needsLayoutTransform && supportsLayoutTransform;
      bool dequantRequired        = needsDequant && supportsDequant;
      bool materializationRequired = castRequired || layoutTransformRequired || dequantRequired;

      bool castUnsup   = needsCast && !supportsCast;
      bool layoutUnsup = needsLayoutTransform && !supportsLayoutTransform;
      bool dequantUnsup = needsDequant && !supportsDequant;

      std::string reason = buildReason(castRequired, layoutTransformRequired, dequantRequired,
                                       castUnsup, layoutUnsup, dequantUnsup);

      op.setAttr("boundary.cast_required",
                 BoolAttr::get(ctx, castRequired));
      op.setAttr("boundary.dequant_required",
                 BoolAttr::get(ctx, dequantRequired));
      op.setAttr("boundary.layout_transform_required",
                 BoolAttr::get(ctx, layoutTransformRequired));
      op.setAttr("boundary.materialization_required",
                 BoolAttr::get(ctx, materializationRequired));
      op.setAttr("boundary.reason",
                 StringAttr::get(ctx, reason));
      op.setAttr("boundary.truth_boundary",
                 StringAttr::get(ctx, kTruthBoundary));

      if (castUnsup || layoutUnsup || dequantUnsup) {
        std::vector<std::string> unsupReasons;
        if (castUnsup)   unsupReasons.push_back("cast_required_but_unsupported");
        if (layoutUnsup) unsupReasons.push_back("layout_transform_required_but_unsupported");
        if (dequantUnsup) unsupReasons.push_back("dequant_required_but_unsupported");
        std::string joined = unsupReasons[0];
        for (size_t i = 1; i < unsupReasons.size(); ++i)
          joined += "," + unsupReasons[i];
        op.setAttr("boundary.unsupported_reason", StringAttr::get(ctx, joined));
      }
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createBoundaryPlanningPass() {
  return std::make_unique<BoundaryPlanningPass>();
}

} // namespace mlir::hir
