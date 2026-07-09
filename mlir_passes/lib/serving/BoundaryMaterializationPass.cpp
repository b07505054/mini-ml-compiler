// BoundaryMaterializationPass — first IR-materializing pass in the serving
// pipeline.
//
// The 16 planning passes are annotation-only: BoundaryPlanningPass decides
// boundary.cast_required / boundary.dequant_required /
// boundary.layout_transform_required, and PlanSelectionPass records the
// winning candidate's required_boundary_ops, but no pass inserts IR. This
// pass closes that loop for the smallest safe subset: float-to-float
// precision casts. It inserts hir.cast where the plan requires one, and
// explicitly records (rather than silently drops) every requirement it does
// not materialize:
//
//   - dequant boundaries are deferred: an honest hir.dequantize needs
//     scale/zero_point metadata the planning pipeline does not produce.
//   - layout transforms are deferred: no layout-transform op exists in the
//     hir dialect yet.
//   - ops whose selected plan is "unsupported" are skipped: a plan with no
//     viable lowering must not have boundary ops materialized.
//
// Malformed planning attrs are compiler errors, not silent no-ops.
//
// Truth boundary: this pass performs a compiler IR transformation only. It
// makes no runtime execution claim and does not modify planning attrs.

#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <string>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_BOUNDARYMATERIALIZATION
#include "FusionPasses.h.inc"

static constexpr StringLiteral kTruth =
    "compiler_materialized_boundary_op_not_runtime_executed";
static constexpr StringLiteral kPassName = "boundary-materialization";

// Normalize planner dtype strings ("fp16", "int8") to MLIR element type names
// ("f16", "i8"). Mirrors BoundaryPlanningPass so both passes agree on the
// meaning of representation.effective_dtype.
static std::string normalizeDtype(const std::string &s) {
  if (s == "fp16" || s == "f16")  return "f16";
  if (s == "fp32" || s == "f32")  return "f32";
  if (s == "bf16")                 return "bf16";
  if (s == "int8" || s == "i8")   return "i8";
  if (s == "int4" || s == "i4")   return "i4";
  return s;
}

static bool isFloatDtype(const std::string &dtype) {
  return dtype == "f16" || dtype == "f32" || dtype == "bf16";
}

static Type floatTypeFor(const std::string &dtype, MLIRContext *ctx) {
  if (dtype == "f16")  return Float16Type::get(ctx);
  if (dtype == "bf16") return BFloat16Type::get(ctx);
  if (dtype == "f32")  return Float32Type::get(ctx);
  return {};
}

static std::string floatDtypeName(Type elem) {
  if (elem.isF16())  return "f16";
  if (elem.isBF16()) return "bf16";
  if (elem.isF32())  return "f32";
  return "";
}

static bool boolAttrTrue(Operation *op, StringRef key) {
  if (auto a = op->getAttrOfType<BoolAttr>(key))
    return a.getValue();
  return false;
}

struct BoundaryMaterializationPass
    : impl::BoundaryMaterializationBase<BoundaryMaterializationPass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    if (funcOp.getBody().empty())
      return;
    Block &entry = funcOp.getBody().front();

    // Target dtype the planner used when deciding boundary.cast_required.
    // Validated lazily: only an op that actually requires a cast turns a
    // missing/non-float value into an error.
    std::string effectiveDtype;
    if (auto a =
            funcOp->getAttrOfType<StringAttr>("representation.effective_dtype"))
      effectiveDtype = normalizeDtype(a.getValue().str());

    // Snapshot ops first: materialization inserts into the block.
    llvm::SmallVector<Operation *> ops;
    for (Operation &op : entry.without_terminator())
      ops.push_back(&op);

    int64_t materializedCount = 0;
    int64_t deferredCount = 0;
    bool sawAnyRequirement = false;
    bool failed = false;

    for (Operation *op : ops) {
      // Never re-process compiler-materialized ops (idempotent re-run).
      if (op->getAttr("materialized.by"))
        continue;

      bool castRequired    = boolAttrTrue(op, "boundary.cast_required");
      bool dequantRequired = boolAttrTrue(op, "boundary.dequant_required") ||
                             boolAttrTrue(op, "boundary.weight_dequant_required");
      bool layoutRequired =
          boolAttrTrue(op, "boundary.layout_transform_required");

      // Invariant: materialization_required must be backed by a specific flag.
      if (boolAttrTrue(op, "boundary.materialization_required") &&
          !castRequired && !dequantRequired && !layoutRequired) {
        op->emitWarning()
            << kPassName
            << ": boundary.materialization_required = true but no specific "
               "boundary requirement flag is set; nothing materialized";
      }

      if (!castRequired && !dequantRequired && !layoutRequired)
        continue;
      sawAnyRequirement = true;

      // Never materialize boundaries for a plan with no viable lowering.
      if (auto sel =
              op->getAttrOfType<StringAttr>("selected_plan.candidate_type");
          sel && sel.getValue() == "unsupported") {
        op->setAttr("boundary.materialization.skipped_reason",
                    StringAttr::get(ctx, "selected_plan_unsupported"));
        op->emitRemark()
            << kPassName
            << ": boundary requirement skipped because the selected plan is "
               "unsupported (no viable lowering path)";
        continue;
      }

      // Record honest deferrals for the boundary kinds this pass does not
      // materialize yet, so they stay visible in IR and in the exported plan.
      SmallVector<Attribute> deferred;
      if (dequantRequired)
        deferred.push_back(StringAttr::get(ctx, "dequant"));
      if (layoutRequired)
        deferred.push_back(StringAttr::get(ctx, "layout_transform"));
      if (!deferred.empty()) {
        op->setAttr("boundary.materialization.deferred",
                    ArrayAttr::get(ctx, deferred));
        deferredCount += static_cast<int64_t>(deferred.size());
        op->emitRemark()
            << kPassName << ": " << deferred.size()
            << " boundary requirement(s) deferred (dequant needs "
               "scale/zero_point metadata the planner does not produce; no "
               "layout-transform op exists in the hir dialect yet)";
      }

      if (!castRequired)
        continue;

      // Idempotency: cast already materialized on a previous run.
      if (boolAttrTrue(op, "boundary.cast_materialized"))
        continue;

      // --- Contract checks: a required cast we cannot materialize is a
      // compiler error, not a silent no-op. ---
      if (!isFloatDtype(effectiveDtype)) {
        op->emitError()
            << kPassName
            << ": boundary.cast_required = true but the function has no "
               "float representation.effective_dtype (got '"
            << effectiveDtype << "')";
        failed = true;
        continue;
      }
      if (op->getNumResults() == 0) {
        op->emitError() << kPassName
                        << ": boundary.cast_required = true on an op with no "
                           "results";
        failed = true;
        continue;
      }
      auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      std::string fromDtype =
          resultType ? floatDtypeName(resultType.getElementType()) : "";
      if (!resultType || fromDtype.empty()) {
        op->emitError()
            << kPassName
            << ": boundary.cast_required = true but the op's first result is "
               "not a ranked tensor with a float element type";
        failed = true;
        continue;
      }
      if (fromDtype == effectiveDtype) {
        op->emitError()
            << kPassName
            << ": boundary.cast_required = true but source and target element "
               "types are both '"
            << fromDtype << "'";
        failed = true;
        continue;
      }

      // hir.cast is created by name (same technique as
      // LLMFrontendNormalizationPass): where the hir dialect is registered
      // (mlir-opt with the dialect plugin) getOrLoadDialect loads it and the
      // created op resolves to the verified CastOp; in unregistered-dialect
      // drivers (compile-for-target) it stays a named op, matching the rest
      // of that IR. Force-registering the dialect here instead would reject
      // the pseudo hir.* ops those drivers accept.
      if (!ctx->getOrLoadDialect("hir") && !ctx->allowsUnregisteredDialects()) {
        op->emitError()
            << kPassName
            << ": cannot materialize hir.cast — the hir dialect is not "
               "loaded and the context disallows unregistered dialects";
        failed = true;
        continue;
      }

      // --- Materialize hir.cast after the op. ---
      Type targetElem = floatTypeFor(effectiveDtype, ctx);
      auto castType =
          RankedTensorType::get(resultType.getShape(), targetElem);

      OpBuilder builder(ctx);
      builder.setInsertionPointAfter(op);
      OperationState state(op->getLoc(), "hir.cast");
      state.addOperands(op->getResult(0));
      state.addTypes(castType);
      Operation *castOp = builder.create(state);

      castOp->setAttr("cast.from_dtype", StringAttr::get(ctx, fromDtype));
      castOp->setAttr("cast.to_dtype", StringAttr::get(ctx, effectiveDtype));
      castOp->setAttr("compiler.materialized", BoolAttr::get(ctx, true));
      castOp->setAttr("materialized.by", StringAttr::get(ctx, kPassName));
      castOp->setAttr("materialized.from_decision",
                      StringAttr::get(ctx, "boundary.cast_required"));
      castOp->setAttr("materialized.of_op",
                      StringAttr::get(ctx, op->getName().getStringRef()));
      castOp->setAttr("materialized.truth_boundary",
                      StringAttr::get(ctx, kTruth));

      op->getResult(0).replaceAllUsesExcept(castOp->getResult(0), castOp);

      op->setAttr("boundary.cast_materialized", BoolAttr::get(ctx, true));
      op->setAttr("boundary.materialized_ops",
                  ArrayAttr::get(ctx, {StringAttr::get(ctx, "cast")}));
      op->setAttr("boundary.materialization.status",
                  StringAttr::get(ctx, "materialized"));
      ++materializedCount;
    }

    if (failed) {
      signalPassFailure();
      return;
    }

    // Casts may now reach func.return; keep the function type consistent.
    if (materializedCount > 0) {
      Operation *terminator = entry.getTerminator();
      SmallVector<Type> resultTypes(terminator->getOperandTypes());
      auto fnType = funcOp.getFunctionType();
      if (fnType.getResults() != ArrayRef<Type>(resultTypes))
        funcOp.setType(
            FunctionType::get(ctx, fnType.getInputs(), resultTypes));
    }

    // Func-level summary — only on functions that had boundary requirements,
    // so planning-only functions are left byte-identical.
    if (sawAnyRequirement) {
      auto i64 = IntegerType::get(ctx, 64);
      funcOp->setAttr("boundary.materialization.deferred_count",
                      IntegerAttr::get(i64, deferredCount));
      funcOp->setAttr("boundary.materialization.materialized_count",
                      IntegerAttr::get(i64, materializedCount));
      funcOp->setAttr("boundary.materialization.status",
                      StringAttr::get(ctx, "completed"));
      funcOp->setAttr("boundary.materialization.truth_boundary",
                      StringAttr::get(ctx, kTruth));
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createBoundaryMaterializationPass() {
  return std::make_unique<BoundaryMaterializationPass>();
}

} // namespace mlir::hir
