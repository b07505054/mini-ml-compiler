#include "serving/ServingExecutionPlanBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::hir {
namespace {

ExecutionMode parseExecutionMode(llvm::StringRef s) {
  if (s == "colocated") return ExecutionMode::Colocated;
  if (s == "pd_split")  return ExecutionMode::PDSplit;
  return ExecutionMode::Unknown;
}

Confidence parseConfidence(llvm::StringRef s) {
  if (s == "medium") return Confidence::Medium;
  if (s == "high")   return Confidence::High;
  return Confidence::Low; // default covers "low" and unknown values
}

// Infer serving phase by walking nested ops for llm.attention_prefill or
// llm.attention_decode. Falls back to a serving.phase string attr if present.
ServingPhase inferServingPhase(mlir::func::FuncOp funcOp) {
  ServingPhase phase = ServingPhase::Unknown;

  funcOp.walk([&](mlir::Operation *op) -> mlir::WalkResult {
    llvm::StringRef opName = op->getName().getStringRef();
    if (opName == "llm.attention_prefill") {
      phase = ServingPhase::Prefill;
      return mlir::WalkResult::interrupt();
    }
    if (opName == "llm.attention_decode") {
      phase = ServingPhase::Decode;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  if (phase != ServingPhase::Unknown)
    return phase;

  // Fallback: check serving.phase string attr on any nested op or the func
  funcOp.walk([&](mlir::Operation *op) -> mlir::WalkResult {
    auto a = op->getAttrOfType<mlir::StringAttr>("serving.phase");
    if (!a) return mlir::WalkResult::advance();
    llvm::StringRef v = a.getValue();
    if (v == "prefill")      phase = ServingPhase::Prefill;
    else if (v == "decode")  phase = ServingPhase::Decode;
    if (phase != ServingPhase::Unknown)
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });

  return phase;
}

} // namespace

ServingExecutionPlan ServingExecutionPlanBuilder::build(mlir::ModuleOp module) {
  ServingExecutionPlan plan;

  if (auto a = module->getAttrOfType<mlir::StringAttr>("llm.model"))
    plan.model_name = a.getValue().str();
  if (auto a = module->getAttrOfType<mlir::IntegerAttr>("llm.num_layers"))
    plan.num_layers = a.getInt();
  if (auto a = module->getAttrOfType<mlir::IntegerAttr>("llm.hidden_size"))
    plan.hidden_size = a.getInt();

  module.walk([&](mlir::func::FuncOp funcOp) {
    // Only collect functions annotated by a serving pass.
    if (!funcOp->getAttr("serving.policy"))
      return;

    FunctionExecutionPlan fp;
    fp.function_name = funcOp.getName().str();
    fp.serving_phase = inferServingPhase(funcOp);

    if (auto a = funcOp->getAttrOfType<mlir::StringAttr>("serving.policy"))
      fp.execution_mode = parseExecutionMode(a.getValue());
    if (auto a = funcOp->getAttrOfType<mlir::FloatAttr>("serving.colocated_total_ms"))
      fp.cost_summary.colocated_total_ms = a.getValueAsDouble();
    if (auto a = funcOp->getAttrOfType<mlir::FloatAttr>("serving.pd_split_total_ms"))
      fp.cost_summary.pd_split_total_ms = a.getValueAsDouble();
    if (auto a = funcOp->getAttrOfType<mlir::FloatAttr>("serving.decision_margin_ms"))
      fp.cost_summary.decision_margin_ms = a.getValueAsDouble();
    if (auto a = funcOp->getAttrOfType<mlir::FloatAttr>("serving.decision_margin_pct"))
      fp.cost_summary.decision_margin_pct = a.getValueAsDouble();
    if (auto a = funcOp->getAttrOfType<mlir::StringAttr>("serving.confidence"))
      fp.cost_summary.confidence = parseConfidence(a.getValue());
    if (auto a = funcOp->getAttrOfType<mlir::StringAttr>("serving.cost_source"))
      fp.provenance.cost_source = a.getValue().str();
    if (auto a = funcOp->getAttrOfType<mlir::StringAttr>("serving.truth_boundary"))
      fp.provenance.truth_boundary = a.getValue().str();

    fp.source_passes.push_back("serving-phase-analysis");

    // KV plan from KVLayoutPlanningPass attrs (present when that pass has run).
    if (auto a = funcOp->getAttrOfType<mlir::StringAttr>("kv.layout")) {
      llvm::StringRef v = a.getValue();
      if (v == "paged")           fp.kv_plan.layout = KVLayout::Paged;
      else if (v == "contiguous") fp.kv_plan.layout = KVLayout::Contiguous;
      fp.source_passes.push_back("kv-layout-planning");
    }
    if (auto a = funcOp->getAttrOfType<mlir::FloatAttr>("kv.byte_estimate_mb"))
      fp.kv_plan.kv_byte_estimate_mb = a.getValueAsDouble();

    // Replay plan from ReplayEligibilityPass attrs (present when that pass has run).
    if (auto a = funcOp->getAttrOfType<mlir::BoolAttr>("replay.eligible")) {
      fp.replay_plan.replay_eligible = a.getValue();
      fp.source_passes.push_back("replay-eligibility");
    }
    if (auto a = funcOp->getAttrOfType<mlir::StringAttr>("replay.cuda_graph_bucket"))
      fp.replay_plan.cuda_graph_bucket = a.getValue().str();

    plan.function_plans.push_back(std::move(fp));
  });

  return plan;
}

} // namespace mlir::hir
