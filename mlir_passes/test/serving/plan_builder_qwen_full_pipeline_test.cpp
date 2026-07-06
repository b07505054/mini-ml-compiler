// Full 16-pass pipeline test for ExecutionPlanBuilder (Qwen 2.5-0.5B).
//
// Architecture contract under test:
//   ExecutionPlanBuilder is a COLLECTOR only. It packs attrs already
//   emitted by passes. This test verifies that it can collect real per-op
//   KernelDecisions once the full serving pipeline has run, and that it
//   does NOT synthesize decisions when a pass did not emit the gate attr.
//
// Why target.kernel_libraries.cpu is included in the MLIR:
//   KernelAvailabilityPlanningPass early-exits when kernels.empty() (no
//   library declared for the backend). Without those entries no per-op
//   kernel.lowering_status attr is emitted and attrToKernelDecision never
//   fires. Adding the entries activates the existing pass logic — this is
//   not faking builder output, it is the standard activation mechanism.
//
// Why attention ops do NOT produce QuantizationDecision:
//   QuantizationStrategyPlanningPass gates on isQuantizableOp() (matmul,
//   conv, linear, dense, gemm, dot, fc, batch_gemm, fused_matmul) and
//   isAccuracySensitive() (softmax, norm, rmsnorm, layernorm, …).
//   stripPrefix("llm.attention_prefill") = "attention_prefill" matches
//   neither list, so quant.strategy is never emitted on these ops.
//   The builder correctly returns nullopt for per-op QuantizationDecision.
//
// Known collector gap (do not fix here):
//   PlanSelectionPass emits selected_plan.* attrs per op. attrToKernelDecision
//   currently reads kernel.name/kernel.library from KernelAvailabilityPlanningPass
//   attrs, not from selected_plan.*. This test validates availability+lowering
//   decision collection. Bridging selected_plan.* into the V2 contract is a
//   separate task.
//
// No GoogleTest. No Python. No JSON parsing. Pure C++ + MLIR.

#include "serving/ExecutionPlanBuilder.h"
#include "capability/CapabilityBundle.h"
#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include <cassert>
#include <cstdio>

// Qwen 2.5-0.5B serving MLIR with kernel library declarations.
//
// target.kernel_libraries.cpu provides two entries:
//   - attention_prefill: cpu_ref_attention_prefill / cpu_reference
//   - attention_decode:  cpu_ref_attention_decode  / cpu_reference
//
// Omitting supported_dtypes/supported_layouts/supported_quant_modes is
// intentional: empty lists mean "match any" in parseDictEntry, which causes
// KernelAvailabilityPlanningPass Rule 1 (exact match) to fire for any
// effective_dtype and layout that the upstream passes select.
static const char kQwenModuleWithKernelLib[] = R"mlir(
module attributes {
  llm.model = "qwen2.5-0.5b",
  llm.num_layers = 24 : i64,
  llm.hidden_size = 896 : i64,
  llm.num_attention_heads = 14 : i64,
  llm.num_key_value_heads = 2 : i64,
  "target.kernel_libraries.cpu" = [
    {op_type = "attention_prefill",
     kernel_name = "cpu_ref_attention_prefill",
     kernel_library = "cpu_reference",
     supports_dynamic_shape = true,
     truth_boundary = "declared_profile"},
    {op_type = "attention_decode",
     kernel_name = "cpu_ref_attention_decode",
     kernel_library = "cpu_reference",
     supports_dynamic_shape = true,
     truth_boundary = "declared_profile"}
  ]
} {
  func.func @qwen_prefill(%tokens: tensor<?xi32>) -> tensor<?x896xf16> {
    %0 = "llm.attention_prefill"(%tokens, %tokens, %tokens) {
      kv_cache.role = "producer",
      serving.phase = "prefill",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?x896xf16>
    return %0 : tensor<?x896xf16>
  }

  func.func @qwen_decode(%token: tensor<1xi32>) -> tensor<1x896xf16> {
    %0 = "llm.attention_decode"(%token, %token, %token) {
      kv_cache.role = "consumer",
      serving.phase = "decode",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x896xf16>
    return %0 : tensor<1x896xf16>
  }
}
)mlir";

static mlir::OwningOpRef<mlir::ModuleOp>
runFullServingPipeline(const char *src, mlir::MLIRContext &ctx) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(src, &ctx);
  assert(module && "MLIR parse failed");

  std::puts("  [pass  1/16] ServingPhaseAnalysisPass");
  std::puts("  [pass  2/16] KVLayoutPlanningPass");
  std::puts("  [pass  3/16] ReplayEligibilityPass");
  std::puts("  [pass  4/16] ExecutionProviderPlanningPass");
  std::puts("  [pass  5/16] RepresentationPlanningPass");
  std::puts("  [pass  6/16] LayoutPlanningPass");
  std::puts("  [pass  7/16] BoundaryPlanningPass");
  std::puts("  [pass  8/16] WeightClassificationPlanningPass");
  std::puts("  [pass  9/16] QuantizationStrategyPlanningPass");
  std::puts("  [pass 10/16] KernelAvailabilityPlanningPass");
  std::puts("  [pass 11/16] LoweringDecisionPlanningPass");
  std::puts("  [pass 12/16] QuantizedBoundaryRefinementPass");
  std::puts("  [pass 13/16] AlternativeLoweringPlanningPass");
  std::puts("  [pass 14/16] CandidateGenerationPass");
  std::puts("  [pass 15/16] ServingCostModelPass");
  std::puts("  [pass 16/16] PlanSelectionPass");

  mlir::PassManager pm(&ctx);
  // Order matches servingOptPipeline in ServingPipeline.cpp.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createServingPhaseAnalysisPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createKVLayoutPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createReplayEligibilityPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createExecutionProviderPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createRepresentationPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createLayoutPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createBoundaryPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createWeightClassificationPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createQuantizationStrategyPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createKernelAvailabilityPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createLoweringDecisionPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createQuantizedBoundaryRefinementPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createAlternativeLoweringPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createCandidateGenerationPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createServingCostModelPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createPlanSelectionPass());

  assert(pm.run(module.get()).succeeded() && "PassManager run failed");
  return module;
}

int main() {
  std::puts("=== PlanV2BuilderQwenFullPipelineTest ===");

  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);
  ctx.loadDialect<mlir::func::FuncDialect>();

  std::puts("[1] Running full 16-pass serving pipeline on Qwen 2.5-0.5B MLIR ...");
  auto module = runFullServingPipeline(kQwenModuleWithKernelLib, ctx);

  // Minimal CapabilityBundle: no profile JSON loaded.
  // The builder must not require a fully populated bundle.
  std::puts("[2] Constructing minimal CapabilityBundle ...");
  mlir::hir::CapabilityBundle bundle;
  bundle.hardware.hardware_id              = "test_gtx1650_maxq";
  bundle.deployment.memory_budget_fraction = 0.75;

  std::puts("[3] Building ExecutionPlan ...");
  mlir::hir::ExecutionPlan plan =
      mlir::hir::ExecutionPlanBuilder::build(
          module.get(), bundle, "test_qwen_v2_full");

  // -------------------------------------------------------------------------
  // Schema constants
  // -------------------------------------------------------------------------
  assert(plan.schema         == "execution_plan" && "schema constant mismatch");
  assert(plan.schema_version == "2.0.0"          && "schema_version constant mismatch");
  assert(plan.plan_id        == "test_qwen_v2_full" && "plan_id passthrough failed");

  // -------------------------------------------------------------------------
  // Model identity — Qwen-specific values from module attrs.
  // -------------------------------------------------------------------------
  assert(plan.model_identity.model_id            == "qwen2.5-0.5b" && "model_id mismatch");
  assert(plan.model_identity.num_layers          == 24             && "num_layers mismatch");
  assert(plan.model_identity.hidden_size         == 896            && "hidden_size mismatch");
  assert(plan.model_identity.num_attention_heads == 14             && "num_attention_heads mismatch");
  assert(plan.model_identity.num_kv_heads        == 2              && "num_kv_heads mismatch");
  assert(!plan.model_identity.truth_boundary.empty()               && "model truth_boundary must be non-empty");

  // -------------------------------------------------------------------------
  // Global decisions
  // -------------------------------------------------------------------------
  assert(plan.global_decisions.serving.has_value() &&
         "global serving decision should be present after ServingPhaseAnalysisPass");
  assert(plan.global_decisions.memory.has_value() &&
         "global memory decision should be present after KVLayoutPlanningPass");

  // CalibrationDecision: no V1 pass emits calibration attrs. Must stay nullopt.
  assert(!plan.global_decisions.calibration.has_value() &&
         "calibration must stay nullopt: no V1 attrs for it");

  assert(!plan.global_decisions.serving->meta.truth_boundary.empty() &&
         "global serving truth_boundary must be non-empty");
  assert(!plan.global_decisions.memory->meta.truth_boundary.empty() &&
         "global memory truth_boundary must be non-empty");

  // memory_budget_fraction propagated from CapabilityBundle, not from MLIR attrs.
  assert(plan.global_decisions.memory->memory_budget_fraction == 0.75 &&
         "memory_budget_fraction should come from CapabilityBundle.deployment");

  // -------------------------------------------------------------------------
  // Function plans: exactly prefill and decode.
  // -------------------------------------------------------------------------
  assert(plan.function_plans.size() == 2 && "expected exactly 2 function plans");

  const mlir::hir::FunctionPlan *fp_prefill = nullptr;
  const mlir::hir::FunctionPlan *fp_decode  = nullptr;
  for (const auto& fp : plan.function_plans) {
    if (fp.serving_phase == mlir::hir::ServingPhase::Prefill) fp_prefill = &fp;
    if (fp.serving_phase == mlir::hir::ServingPhase::Decode)  fp_decode  = &fp;
  }
  assert(fp_prefill && "prefill function plan missing");
  assert(fp_decode  && "decode function plan missing");
  assert(fp_prefill->function_name == "qwen_prefill" && "prefill function_name mismatch");
  assert(fp_decode->function_name  == "qwen_decode"  && "decode function_name mismatch");

  // -------------------------------------------------------------------------
  // BackendDecision truth boundary.
  // -------------------------------------------------------------------------
  assert(!fp_prefill->backend.meta.truth_boundary.empty() &&
         "prefill backend truth_boundary must be non-empty");
  assert(!fp_decode->backend.meta.truth_boundary.empty() &&
         "decode backend truth_boundary must be non-empty");
  assert(!fp_prefill->backend.selected_backend.empty() &&
         "prefill selected_backend must be non-empty after ExecutionProviderPlanningPass");
  assert(!fp_decode->backend.selected_backend.empty() &&
         "decode selected_backend must be non-empty after ExecutionProviderPlanningPass");

  // -------------------------------------------------------------------------
  // Per-op decisions: each function has exactly one attention op.
  //
  // KernelAvailabilityPlanningPass found cpu_reference entries for
  // attention_prefill and attention_decode → kernel.lowering_status = "lowerable".
  // LoweringDecisionPlanningPass Rule 1 → lowering.decision = "direct_lower".
  // attrToKernelDecision fires (gate: lowering.decision present).
  // -------------------------------------------------------------------------
  assert(fp_prefill->per_op_decisions.size() == 1 &&
         "prefill should have exactly 1 per-op bundle (one attention_prefill op)");
  assert(fp_decode->per_op_decisions.size() == 1 &&
         "decode should have exactly 1 per-op bundle (one attention_decode op)");

  const auto& prefill_op = fp_prefill->per_op_decisions[0];
  const auto& decode_op  = fp_decode->per_op_decisions[0];

  // KernelDecision: collected from lowering.decision + kernel.* attrs.
  assert(prefill_op.kernel.has_value() &&
         "prefill KernelDecision must be present: LoweringDecisionPlanningPass emits lowering.decision");
  assert(decode_op.kernel.has_value() &&
         "decode KernelDecision must be present: LoweringDecisionPlanningPass emits lowering.decision");

  // lowering_path: from lowering.decision attr. Rule 1 (exact kernel match) → "direct_lower".
  assert(prefill_op.kernel->lowering_path == "direct_lower" &&
         "prefill lowering_path should be direct_lower: cpu_reference kernel matched");
  assert(decode_op.kernel->lowering_path  == "direct_lower" &&
         "decode lowering_path should be direct_lower: cpu_reference kernel matched");

  // kernel_exists: KernelAvailabilityPlanningPass Rule 1 sets kernel.exists = true.
  assert(prefill_op.kernel->kernel_exists == true &&
         "prefill kernel_exists must be true: exact match in kernel library");
  assert(decode_op.kernel->kernel_exists  == true &&
         "decode kernel_exists must be true: exact match in kernel library");

  // kernel_library: from kernel.library attr written by KernelAvailabilityPlanningPass.
  assert(prefill_op.kernel->kernel_library == "cpu_reference" &&
         "prefill kernel_library should be cpu_reference");
  assert(decode_op.kernel->kernel_library  == "cpu_reference" &&
         "decode kernel_library should be cpu_reference");

  // truth_boundary on every kernel decision must be non-empty.
  assert(!prefill_op.kernel->meta.truth_boundary.empty() &&
         "prefill kernel KernelDecision truth_boundary must be non-empty");
  assert(!decode_op.kernel->meta.truth_boundary.empty() &&
         "decode kernel KernelDecision truth_boundary must be non-empty");

  // -------------------------------------------------------------------------
  // meta.evidence.cost: populated by ServingCostModelPass + PlanSelectionPass.
  //
  // For direct_lower with no boundary ops all component costs are 0.
  // cost_model_id and truth_boundary are always set when the cost model ran.
  // -------------------------------------------------------------------------
  assert(prefill_op.kernel->meta.evidence.cost.has_value() &&
         "prefill cost evidence must be present: ServingCostModelPass ran");
  assert(decode_op.kernel->meta.evidence.cost.has_value() &&
         "decode cost evidence must be present: ServingCostModelPass ran");

  assert(prefill_op.kernel->meta.evidence.cost->cost_model_id ==
         "serving_static_cost_model_v1" &&
         "prefill cost_model_id must be serving_static_cost_model_v1");
  assert(decode_op.kernel->meta.evidence.cost->cost_model_id ==
         "serving_static_cost_model_v1" &&
         "decode cost_model_id must be serving_static_cost_model_v1");

  assert(prefill_op.kernel->meta.evidence.cost->total_cost == 0 &&
         "prefill total_cost must be 0: direct_lower, no boundary ops");
  assert(decode_op.kernel->meta.evidence.cost->total_cost == 0 &&
         "decode total_cost must be 0: direct_lower, no boundary ops");

  assert(prefill_op.kernel->meta.evidence.cost->backend_switch_cost == 0 &&
         "prefill backend_switch_cost must be 0: direct_lower path");
  assert(decode_op.kernel->meta.evidence.cost->backend_switch_cost == 0 &&
         "decode backend_switch_cost must be 0: direct_lower path");

  assert(prefill_op.kernel->meta.evidence.cost->truth_boundary ==
         "serving_static_cost_model_v1_not_measured_latency" &&
         "prefill cost truth_boundary must be serving_static_cost_model_v1_not_measured_latency");
  assert(decode_op.kernel->meta.evidence.cost->truth_boundary ==
         "serving_static_cost_model_v1_not_measured_latency" &&
         "decode cost truth_boundary must be serving_static_cost_model_v1_not_measured_latency");

  // -------------------------------------------------------------------------
  // Per-op QuantizationDecision: must be nullopt.
  //
  // QuantizationStrategyPlanningPass skips attention_prefill and
  // attention_decode because neither op name matches isQuantizableOp()
  // (matmul/conv/linear/dense/gemm/dot/fc/batch_gemm/fused_matmul) nor
  // isAccuracySensitive() (softmax/norm/rmsnorm/layernorm/…).
  // quant.strategy is never emitted → attrToPerOpQuantDecision returns nullopt.
  // -------------------------------------------------------------------------
  assert(!prefill_op.quantization.has_value() &&
         "prefill QuantizationDecision must be nullopt: "
         "QuantizationStrategyPlanningPass skips attention ops");
  assert(!decode_op.quantization.has_value() &&
         "decode QuantizationDecision must be nullopt: "
         "QuantizationStrategyPlanningPass skips attention ops");

  // -------------------------------------------------------------------------
  // Per-op FallbackDecision: must be nullopt.
  //
  // attrToFallbackDecision gates on lowering.decision == "fallback_backend".
  // Since lowering.decision == "direct_lower" here, fallback stays nullopt.
  // -------------------------------------------------------------------------
  assert(!prefill_op.fallback.has_value() &&
         "prefill FallbackDecision must be nullopt: lowering_path is direct_lower");
  assert(!decode_op.fallback.has_value() &&
         "decode FallbackDecision must be nullopt: lowering_path is direct_lower");

  // -------------------------------------------------------------------------
  // Per-op LayoutDecision: must be nullopt.
  //
  // No standalone per-op layout attr cluster in V1. attrToKernelDecision does
  // not produce a LayoutDecision; that field stays nullopt by design.
  // -------------------------------------------------------------------------
  assert(!prefill_op.layout.has_value() &&
         "prefill LayoutDecision should stay nullopt: no V1 per-op layout attr");
  assert(!decode_op.layout.has_value() &&
         "decode LayoutDecision should stay nullopt: no V1 per-op layout attr");

  // -------------------------------------------------------------------------
  // dumpSummary: must not crash.
  // -------------------------------------------------------------------------
  std::puts("[4] dumpSummary ...");
  mlir::hir::ExecutionPlanBuilder::dumpSummary(plan, llvm::errs());

  std::puts("PlanV2BuilderQwenFullPipelineTest: PASS");
  return 0;
}
