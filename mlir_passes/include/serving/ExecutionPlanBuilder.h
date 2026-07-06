#pragma once

// ExecutionPlanBuilder: reads MLIR attrs already emitted by the serving
// pass pipeline and packs them into a typed ExecutionPlan.
//
// Architecture contract:
//   This builder is a COLLECTOR/PACKER only. It does not make decisions.
//   Every field in the returned plan is sourced from an MLIR attr written
//   by a pass, or read directly from the CapabilityBundle.
//
//   If an attr is absent, the corresponding field is left at its zero
//   value, empty string, or nullopt.  No inference.  No selection.
//   No aggregation across ops or functions.
//
//   Decision Engine: the 15-pass serving pipeline (not this builder).

#include "serving/ExecutionPlan.h"
#include "capability/CapabilityBundle.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include <optional>
#include <vector>

namespace mlir::hir {

class ExecutionPlanBuilder {
public:
  // Collect decisions already emitted by passes and assemble a V2 plan.
  // module:       should have completed at least ServingPhaseAnalysisPass.
  //               Additional pass attrs (kv.*, replay.*, execution_provider.*,
  //               quant.*, lowering.*) are collected when present.
  // capabilities: assembled at the tool boundary before passes ran; used for
  //               bundle refs and fields not expressible as MLIR attrs
  //               (e.g. memory_budget_fraction).
  static ExecutionPlan build(mlir::ModuleOp module,
                               const CapabilityBundle& capabilities,
                               llvm::StringRef plan_id = "");

  // Emit a one-line summary per collected decision.
  // For development inspection only — not a serializer, not schema-stable.
  static void dumpSummary(const ExecutionPlan& plan, llvm::raw_ostream& os);

private:
  // Collect helpers: read existing attrs, return typed structs.
  // Vocabulary: "collect" not "build/decide/select/infer".
  static ModelIdentity    collectModelIdentity(mlir::ModuleOp module);
  static CapabilityBundleRefs collectBundleRefs(mlir::ModuleOp module,
                                                const CapabilityBundle& capabilities);
  static GlobalDecisions  collectGlobalDecisions(mlir::ModuleOp module,
                                                 const CapabilityBundle& capabilities);
  static FunctionPlan   collectFunctionDecisions(mlir::func::FuncOp funcOp,
                                                   const CapabilityBundle& capabilities);
  static std::vector<PerOpDecisionBundle>
  collectPerOpDecisionBundles(mlir::func::FuncOp funcOp);

  // Per-attr translators: read one attr cluster, return one typed Decision.
  // Return nullopt when the primary gate attr is absent.
  static std::optional<ServingDecision>
  attrToServingDecision(mlir::func::FuncOp funcOp);

  static std::optional<MemoryDecision>
  attrToMemoryDecision(mlir::func::FuncOp funcOp, const CapabilityBundle& capabilities);

  static std::optional<QuantizationDecision>
  attrToGlobalQuantDecision(mlir::ModuleOp module);

  static BackendDecision
  attrToBackendDecision(mlir::func::FuncOp funcOp);

  static std::optional<KernelDecision>
  attrToKernelDecision(mlir::Operation* op, int opIndex);

  static std::optional<QuantizationDecision>
  attrToPerOpQuantDecision(mlir::Operation* op, int opIndex);

  static std::optional<FallbackDecision>
  attrToFallbackDecision(mlir::Operation* op, int opIndex);
};

} // namespace mlir::hir
