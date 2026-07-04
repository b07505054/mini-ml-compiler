#include "serving/ServingExecutionPlanBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
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

  if (auto a = module->getAttrOfType<mlir::StringAttr>("target.profile_id"))
    plan.target_profile_id = a.getValue().str();
  if (auto a = module->getAttrOfType<mlir::StringAttr>("llm.model"))
    plan.model_name = a.getValue().str();
  if (auto a = module->getAttrOfType<mlir::IntegerAttr>("llm.num_layers"))
    plan.num_layers = a.getInt();
  if (auto a = module->getAttrOfType<mlir::IntegerAttr>("llm.hidden_size"))
    plan.hidden_size = a.getInt();
  if (auto a = module->getAttrOfType<mlir::IntegerAttr>("llm.num_attention_heads"))
    plan.num_attention_heads = a.getInt();
  if (auto a = module->getAttrOfType<mlir::IntegerAttr>("llm.num_key_value_heads"))
    plan.num_kv_heads = a.getInt();

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

    // Backend execution plan from ExecutionProviderPlanningPass (present when
    // that pass has run).
    if (auto a = funcOp->getAttrOfType<mlir::StringAttr>(
            "execution_provider.primary")) {
      fp.backend_execution_plan.primary_backend = a.getValue().str();
      fp.source_passes.push_back("execution-provider-planning");
    }
    if (auto a = funcOp->getAttrOfType<mlir::ArrayAttr>(
            "execution_provider.fallback_chain")) {
      for (mlir::Attribute elem : a)
        if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
          fp.backend_execution_plan.fallback_chain.push_back(s.getValue().str());
    }
    if (auto a = funcOp->getAttrOfType<mlir::StringAttr>(
            "execution_provider.decision_source"))
      fp.backend_execution_plan.decision_source = a.getValue().str();
    if (auto a = funcOp->getAttrOfType<mlir::StringAttr>(
            "execution_provider.required_precision"))
      fp.backend_execution_plan.required_precision = a.getValue().str();
    if (auto a = funcOp->getAttrOfType<mlir::StringAttr>(
            "execution_provider.required_kv_layout"))
      fp.backend_execution_plan.required_kv_layout = a.getValue().str();
    if (auto a = funcOp->getAttrOfType<mlir::BoolAttr>(
            "execution_provider.requires_replay"))
      fp.backend_execution_plan.requires_replay = a.getValue();

    // Quantization plan: prefer func-level attrs (quantization.effective_dtype /
    // quantization.dtype_bytes emitted by ServingPhaseAnalysisPass after Commit B),
    // then fall back to module-level attrs from QuantizationPlanningPass.
    {
      auto &qp = fp.quantization_plan;

      // plan_dtype: func effective_dtype → module plan_dtype → default "fp16".
      if (auto a = funcOp->getAttrOfType<mlir::StringAttr>(
              "quantization.effective_dtype"))
        qp.plan_dtype = a.getValue().str();
      else if (auto a = module->getAttrOfType<mlir::StringAttr>(
              "quantization.plan_dtype"))
        qp.plan_dtype = a.getValue().str();
      else
        qp.plan_dtype = "fp16";

      // plan_source: only on the module (set by QuantizationPlanningPass).
      if (auto a = module->getAttrOfType<mlir::StringAttr>(
              "quantization.plan_source"))
        qp.plan_source = a.getValue().str();
      else
        qp.plan_source = "default_dtype";

      // dtype_bytes: func attr (post-cost-connect) → default 2.0.
      if (auto a = funcOp->getAttrOfType<mlir::FloatAttr>(
              "quantization.dtype_bytes"))
        qp.dtype_bytes = a.getValueAsDouble();
      else
        qp.dtype_bytes = 2.0;

      // truth_boundary: func attr → module attr → standard default.
      if (auto a = funcOp->getAttrOfType<mlir::StringAttr>(
              "quantization.truth_boundary"))
        qp.truth_boundary = a.getValue().str();
      else if (auto a = module->getAttrOfType<mlir::StringAttr>(
              "quantization.truth_boundary"))
        qp.truth_boundary = a.getValue().str();
      else
        qp.truth_boundary =
            "precision_selection_from_target_profile_not_calibrated";

      // Track QuantizationPlanningPass only when it actually ran on the module.
      if (module->getAttr("quantization.plan_dtype"))
        fp.source_passes.push_back("quantization-planning");
    }

    // Per-op quantization strategy decisions from QuantizationStrategyPlanningPass.
    // Only collect ops that have quant.strategy set (pass did not run → skip).
    {
      bool strategyPassRan = false;
      int opIndex = 0;
      if (!funcOp.getBody().empty()) {
        mlir::Block &entry = funcOp.getBody().front();
        for (mlir::Operation &op : entry.without_terminator()) {
          auto stratAttr = op.getAttrOfType<mlir::StringAttr>("quant.strategy");
          if (!stratAttr) { ++opIndex; continue; }

          strategyPassRan = true;
          OpQuantizationDecision d;

          d.op_name = "op_" + std::to_string(opIndex);

          d.op_type = op.getName().getStringRef().str();

          auto str = [&](llvm::StringRef k) -> std::string {
            if (auto a = op.getAttrOfType<mlir::StringAttr>(k))
              return a.getValue().str();
            return {};
          };
          auto boolean = [&](llvm::StringRef k) -> bool {
            if (auto a = op.getAttrOfType<mlir::BoolAttr>(k))
              return a.getValue();
            return false;
          };

          d.strategy              = stratAttr.getValue().str();
          d.backend               = str("quant.backend");
          d.weight_dtype          = str("quant.weight_dtype");
          d.activation_dtype      = str("quant.activation_dtype");
          d.accumulation_dtype    = str("quant.accumulation_dtype");
          d.output_dtype          = str("quant.output_dtype");
          d.granularity           = str("quant.granularity");
          d.activation_quant_mode = str("quant.activation_quant_mode");
          d.weight_quant_mode     = str("quant.weight_quant_mode");
          d.requires_dequant_boundary = boolean("quant.requires_dequant_boundary");
          d.requires_requant_boundary = boolean("quant.requires_requant_boundary");
          d.decision_reason       = str("quant.decision_reason");
          d.fallback_reason       = str("quant.fallback_reason");
          d.accuracy_risk         = str("quant.accuracy_risk");
          d.truth_boundary        = str("quant.truth_boundary");
          if (d.truth_boundary.empty())
            d.truth_boundary =
                "quantization_strategy_export_static_not_accuracy_calibrated";

          fp.per_op_quant_decisions.push_back(std::move(d));
          ++opIndex;
        }
      }
      if (strategyPassRan)
        fp.source_passes.push_back("quantization-strategy-planning");
    }

    // Per-op lowering decisions from LoweringDecisionPlanningPass.
    // Only collect ops that have lowering.decision set (pass did not run → skip).
    {
      bool loweringPassRan = false;
      int opIndex = 0;
      if (!funcOp.getBody().empty()) {
        mlir::Block &entry = funcOp.getBody().front();
        for (mlir::Operation &op : entry.without_terminator()) {
          auto decisionAttr = op.getAttrOfType<mlir::StringAttr>("lowering.decision");
          if (!decisionAttr) { ++opIndex; continue; }

          loweringPassRan = true;
          OpLoweringDecision d;

          d.op_name = "op_" + std::to_string(opIndex);
          d.op_type = op.getName().getStringRef().str();

          auto str = [&](llvm::StringRef k) -> std::string {
            if (auto a = op.getAttrOfType<mlir::StringAttr>(k))
              return a.getValue().str();
            return {};
          };
          auto boolean = [&](llvm::StringRef k) -> bool {
            if (auto a = op.getAttrOfType<mlir::BoolAttr>(k))
              return a.getValue();
            return false;
          };

          // Kernel provenance fields (from KernelAvailabilityPlanningPass).
          d.backend               = str("kernel.backend");
          d.kernel_library        = str("kernel.library");
          d.kernel_name           = str("kernel.name");
          d.kernel_exists         = boolean("kernel.exists");
          d.kernel_lowering_status = str("kernel.lowering_status");

          // Lowering decision fields (from LoweringDecisionPlanningPass).
          d.lowering_decision        = decisionAttr.getValue().str();
          d.target_backend           = str("lowering.target_backend");
          d.target_kernel_library    = str("lowering.target_kernel_library");
          d.target_kernel            = str("lowering.target_kernel");
          d.required_rewrite         = str("lowering.required_rewrite");
          d.requires_dequant         = boolean("lowering.requires_dequant");
          d.requires_layout_transform = boolean("lowering.requires_layout_transform");
          d.requires_cast            = boolean("lowering.requires_cast");
          d.reason                   = str("lowering.reason");
          d.truth_boundary           = str("lowering.truth_boundary");
          if (d.truth_boundary.empty())
            d.truth_boundary =
                "lowering_decision_export_static_not_backend_codegen_verified";

          fp.per_op_lowering_decisions.push_back(std::move(d));
          ++opIndex;
        }
      }
      if (loweringPassRan)
        fp.source_passes.push_back("lowering-decision-planning");
    }

    plan.function_plans.push_back(std::move(fp));
  });

  return plan;
}

} // namespace mlir::hir
