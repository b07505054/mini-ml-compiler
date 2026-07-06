// CTest unit test for ExecutionPlanBuilder with Qwen 2.5-0.5B model attrs.
//
// Architecture contract under test:
//   ExecutionPlanBuilder is a COLLECTOR only.  It must not make decisions.
//   This test verifies that it packs attrs already emitted by passes without
//   inventing values for absent attrs.
//
// Test strategy:
//   - Run 4-pass serving pipeline: ServingPhaseAnalysis, KVLayoutPlanning,
//     ReplayEligibility, ExecutionProviderPlanning.
//   - Construct a minimal CapabilityBundle manually (no profile loading).
//   - Call ExecutionPlanBuilder::build.
//   - Assert schema version, Qwen-specific model identity, function plan
//     structure, global decisions, and empty per_op_decisions (confirming
//     the builder does not synthesize decisions when passes haven't run).
//
// No GoogleTest.  No Python.  No JSON parsing.  Pure C++ + MLIR.

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

// Qwen 2.5-0.5B serving MLIR.
// Model config matches configs/models/qwen_0_5b_spec.json:
//   layers=24, hidden=896, heads=14, kv_heads=2
// Two functions: prefill (producer) and decode (consumer).
static const char kQwenModule[] = R"mlir(
module attributes {
  llm.model = "qwen2.5-0.5b",
  llm.num_layers = 24 : i64,
  llm.hidden_size = 896 : i64,
  llm.num_attention_heads = 14 : i64,
  llm.num_key_value_heads = 2 : i64
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
runServingPipeline(const char *src, mlir::MLIRContext &ctx) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(src, &ctx);
  assert(module && "MLIR parse failed");

  std::puts("  -> ServingPhaseAnalysisPass");
  std::puts("  -> KVLayoutPlanningPass");
  std::puts("  -> ReplayEligibilityPass");
  std::puts("  -> ExecutionProviderPlanningPass");

  mlir::PassManager pm(&ctx);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createServingPhaseAnalysisPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createKVLayoutPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createReplayEligibilityPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createExecutionProviderPlanningPass());

  assert(pm.run(module.get()).succeeded() && "PassManager run failed");
  return module;
}

int main() {
  std::puts("=== PlanV2BuilderQwenTest ===");

  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);
  ctx.loadDialect<mlir::func::FuncDialect>();

  std::puts("[1] Running 4-pass serving pipeline on Qwen 2.5-0.5B MLIR ...");
  auto module = runServingPipeline(kQwenModule, ctx);

  // Build a minimal CapabilityBundle.  No profile JSON loaded.
  // The builder must not require a fully populated bundle.
  std::puts("[2] Constructing minimal CapabilityBundle ...");
  mlir::hir::CapabilityBundle bundle;
  bundle.hardware.hardware_id         = "test_gtx1650_maxq";
  bundle.deployment.memory_budget_fraction = 0.75;
  // backends, kernels, model, workload left empty — builder must tolerate this.

  std::puts("[3] Building ExecutionPlan ...");
  mlir::hir::ExecutionPlan plan =
      mlir::hir::ExecutionPlanBuilder::build(
          module.get(), bundle, "test_qwen_v2");

  // -------------------------------------------------------------------------
  // Schema constants
  // -------------------------------------------------------------------------
  assert(plan.schema         == "execution_plan" && "schema constant mismatch");
  assert(plan.schema_version == "2.0.0"          && "schema_version constant mismatch");
  assert(plan.plan_id        == "test_qwen_v2"   && "plan_id passthrough failed");

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
  // Provenance: hardware_profile_ref from bundle, not module attrs.
  // -------------------------------------------------------------------------
  assert(plan.provenance.capability_bundle.hardware_profile_ref == "test_gtx1650_maxq"
         && "hardware_profile_ref should come from CapabilityBundle");

  // -------------------------------------------------------------------------
  // Global decisions: serving and memory should be present after the pipeline.
  // -------------------------------------------------------------------------
  assert(plan.global_decisions.serving.has_value() &&
         "global serving decision should be present after ServingPhaseAnalysisPass");
  assert(plan.global_decisions.memory.has_value() &&
         "global memory decision should be present after KVLayoutPlanningPass");

  // Calibration must be nullopt — no V1 pass emits calibration attrs.
  assert(!plan.global_decisions.calibration.has_value() &&
         "calibration should stay nullopt: no V1 attrs for it");

  // Serving topology: "colocated" is the default for a single-device target
  // with no target.preferred_backend or target.profile_id attrs set.
  assert(plan.global_decisions.serving->topology == "colocated" &&
         "serving topology should be colocated (default single-device)");
  assert(!plan.global_decisions.serving->meta.truth_boundary.empty() &&
         "serving decision truth_boundary must be non-empty");

  // Memory budget fraction propagated from CapabilityBundle.
  assert(plan.global_decisions.memory->memory_budget_fraction == 0.75 &&
         "memory_budget_fraction should come from CapabilityBundle.deployment");
  assert(!plan.global_decisions.memory->meta.truth_boundary.empty() &&
         "memory decision truth_boundary must be non-empty");

  // -------------------------------------------------------------------------
  // Function plans: exactly prefill and decode.
  // -------------------------------------------------------------------------
  assert(plan.function_plans.size() == 2 && "expected exactly 2 function plans");

  // Find prefill and decode by serving_phase (order not assumed).
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
  // BackendDecision: collected from execution_provider.primary attrs.
  // ExecutionProviderPlanningPass must have set a non-empty backend.
  // -------------------------------------------------------------------------
  assert(!fp_prefill->backend.meta.truth_boundary.empty() &&
         "prefill backend truth_boundary must be non-empty");
  assert(!fp_decode->backend.meta.truth_boundary.empty() &&
         "decode backend truth_boundary must be non-empty");
  assert(!fp_prefill->backend.selected_backend.empty() &&
         "prefill backend selected_backend should be non-empty after ExecutionProviderPlanningPass");
  assert(!fp_decode->backend.selected_backend.empty() &&
         "decode backend selected_backend should be non-empty after ExecutionProviderPlanningPass");

  // -------------------------------------------------------------------------
  // Per-op decisions: must be empty.
  // QuantizationStrategyPlanningPass and LoweringDecisionPlanningPass did not
  // run, so no quant.strategy or lowering.decision attrs exist on ops.
  // The builder must NOT synthesize per-op decisions when attrs are absent.
  // -------------------------------------------------------------------------
  assert(fp_prefill->per_op_decisions.empty() &&
         "per_op_decisions must be empty: quant/lowering passes did not run");
  assert(fp_decode->per_op_decisions.empty() &&
         "per_op_decisions must be empty: quant/lowering passes did not run");

  // -------------------------------------------------------------------------
  // dumpSummary: must not crash.
  // -------------------------------------------------------------------------
  std::puts("[4] dumpSummary ...");
  mlir::hir::ExecutionPlanBuilder::dumpSummary(plan, llvm::errs());

  std::puts("PlanV2BuilderQwenTest: PASS");
  return 0;
}
