// CTest unit test for ServingCostModelPass — ServingStaticCostModel_v1.
//
// Verifies that the V1 cost model:
//   1. Emits evaluation.cost.* attrs with the correct component values.
//   2. Maintains total_cost == exact sum of all component fields.
//   3. Attaches the correct cost_model_id and truth_boundary.
//   4. Preserves evaluation.penalty_score (V0 backward-compat ranking value).
//   5. Leaves V0 evaluation.status and evaluation.reason unchanged.
//
// Five scenarios:
//   A. direct_lower, no boundary ops → all component costs 0, total 0
//   B. backend_fallback with fallback_backend → backend_switch=20, overhead=2, transfer=5, total=27
//   C. layout_conversion, no boundary → layout_transform_cost=4, total=4
//   D. cast_conversion, no boundary → cast_cost=2, total=2
//   E. unsupported → unsupported_penalty=100, total=100
//
// No GoogleTest. No JSON. Pure C++ + MLIR.

#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include <cassert>
#include <cstdio>
#include <string>

// Each op has exactly one candidate. The pass evaluates it and emits
// evaluation.cost.* attrs into compiler.evaluated_candidates[0].
static const char kTestModule[] = R"mlir(
module {
  func.func @cost_scenarios() {
    "test.op_direct"() {
      compiler.candidates = [{
        candidate_type = "direct_lower",
        required_boundary_ops = [],
        source_op = "test.op_direct",
        truth_boundary = "test"
      }]
    } : () -> ()
    "test.op_fallback"() {
      compiler.candidates = [{
        candidate_type = "backend_fallback",
        fallback_backend = "cpu_reference",
        required_boundary_ops = [],
        source_op = "test.op_fallback",
        truth_boundary = "test"
      }]
    } : () -> ()
    "test.op_layout"() {
      compiler.candidates = [{
        candidate_type = "layout_conversion",
        required_boundary_ops = [],
        source_op = "test.op_layout",
        truth_boundary = "test"
      }]
    } : () -> ()
    "test.op_cast"() {
      compiler.candidates = [{
        candidate_type = "cast_conversion",
        required_boundary_ops = [],
        source_op = "test.op_cast",
        truth_boundary = "test"
      }]
    } : () -> ()
    "test.op_unsupported"() {
      compiler.candidates = [{
        candidate_type = "unsupported",
        required_boundary_ops = [],
        source_op = "test.op_unsupported",
        truth_boundary = "test"
      }]
    } : () -> ()
    func.return
  }
}
)mlir";

// ---------------------------------------------------------------------------
// Helpers to read evaluation.cost.* from an evaluated candidate DictionaryAttr.
// ---------------------------------------------------------------------------
static int64_t getI64Cost(mlir::DictionaryAttr dict, llvm::StringRef key) {
  if (!dict) return -9999;
  if (auto a = dict.get(key))
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a)) return ia.getInt();
  return -9999;
}

static std::string getStrCost(mlir::DictionaryAttr dict, llvm::StringRef key) {
  if (!dict) return "<missing>";
  if (auto a = dict.get(key))
    if (auto sa = mlir::dyn_cast<mlir::StringAttr>(a)) return sa.getValue().str();
  return "<missing>";
}

struct CostFields {
  int64_t compute, memory, dequant, requant, layout_transform;
  int64_t cast, backend_switch, launch_overhead, kv_cache, transfer;
  int64_t unsupported, total;
  std::string model_id, truth_boundary, status, reason;
  int64_t penalty_score;

  explicit CostFields(mlir::DictionaryAttr dict) {
    compute          = getI64Cost(dict, "evaluation.cost.compute");
    memory           = getI64Cost(dict, "evaluation.cost.memory");
    dequant          = getI64Cost(dict, "evaluation.cost.dequant");
    requant          = getI64Cost(dict, "evaluation.cost.requant");
    layout_transform = getI64Cost(dict, "evaluation.cost.layout_transform");
    cast             = getI64Cost(dict, "evaluation.cost.cast");
    backend_switch   = getI64Cost(dict, "evaluation.cost.backend_switch");
    launch_overhead  = getI64Cost(dict, "evaluation.cost.launch_overhead");
    kv_cache         = getI64Cost(dict, "evaluation.cost.kv_cache");
    transfer         = getI64Cost(dict, "evaluation.cost.transfer");
    unsupported      = getI64Cost(dict, "evaluation.cost.unsupported");
    total            = getI64Cost(dict, "evaluation.cost.total");
    model_id         = getStrCost(dict, "evaluation.cost.model_id");
    truth_boundary   = getStrCost(dict, "evaluation.cost.truth_boundary");
    status           = getStrCost(dict, "evaluation.status");
    reason           = getStrCost(dict, "evaluation.reason");
    penalty_score    = getI64Cost(dict, "evaluation.penalty_score");
  }

  int64_t componentSum() const {
    return compute + memory + dequant + requant + layout_transform
         + cast + backend_switch + launch_overhead + kv_cache
         + transfer + unsupported;
  }
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
  std::puts("=== ServingStaticCostModelV1Test ===");

  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);
  ctx.loadDialect<mlir::func::FuncDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(kTestModule, &ctx);
  assert(module && "Failed to parse test module");

  mlir::PassManager pm(&ctx);
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hir::createServingCostModelPass());
  assert(pm.run(*module).succeeded() && "ServingCostModelPass failed");

  // Walk the function and collect evaluated candidate dicts in order.
  mlir::func::FuncOp funcOp;
  module->walk([&](mlir::func::FuncOp f) { funcOp = f; });
  assert(funcOp && "Expected a func.func in test module");

  std::vector<CostFields> results;
  for (mlir::Operation& op : funcOp.getBody().front().without_terminator()) {
    auto arr = op.getAttrOfType<mlir::ArrayAttr>("compiler.evaluated_candidates");
    assert(arr && arr.size() == 1 && "Each test op must have exactly 1 candidate");
    auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(arr[0]);
    assert(dict && "Evaluated candidate must be a DictionaryAttr");
    results.emplace_back(dict);
  }
  assert(results.size() == 5 && "Expected 5 test ops");

  static constexpr char kModelId[] = "serving_static_cost_model_v1";
  static constexpr char kTB[]      = "serving_static_cost_model_v1_not_measured_latency";

  // ---------------------------------------------------------------------------
  // Scenario A: direct_lower, no boundary ops.
  // V1 total: 0; V0 penalty_score: 0.
  // ---------------------------------------------------------------------------
  std::puts("[A] direct_lower");
  {
    const auto& f = results[0];
    assert(f.total == 0 &&             "direct_lower: total must be 0");
    assert(f.compute == 0 &&           "direct_lower: compute must be 0");
    assert(f.backend_switch == 0 &&    "direct_lower: backend_switch must be 0");
    assert(f.componentSum() == f.total && "direct_lower: total must equal component sum");
    assert(f.model_id == kModelId &&   "direct_lower: cost_model_id must be serving_static_cost_model_v1");
    assert(f.truth_boundary == kTB &&  "direct_lower: cost truth_boundary must be V1 string");
    assert(f.status == "evaluated" &&  "direct_lower: V0 status must be evaluated");
    assert(f.penalty_score == 0 &&     "direct_lower: V0 penalty_score must be 0 (backward compat)");
  }

  // ---------------------------------------------------------------------------
  // Scenario B: backend_fallback with fallback_backend="cpu_reference".
  // V1 total: backend_switch(20) + launch_overhead(2) + transfer(5) = 27.
  // V0 penalty_score: 20 (preserved for backward compat, does not include overhead/transfer).
  // ---------------------------------------------------------------------------
  std::puts("[B] backend_fallback with fallback_backend");
  {
    const auto& f = results[1];
    assert(f.backend_switch == 20 &&   "backend_fallback: backend_switch_cost must be 20");
    assert(f.launch_overhead == 2 &&   "backend_fallback: launch_overhead must be 2 (fallback overhead)");
    assert(f.transfer == 5 &&          "backend_fallback: transfer_cost must be 5 when fallback_backend set");
    assert(f.total == 27 &&            "backend_fallback: total must be 27 (20+2+5)");
    assert(f.componentSum() == f.total && "backend_fallback: total must equal component sum");
    assert(f.model_id == kModelId &&   "backend_fallback: cost_model_id must be serving_static_cost_model_v1");
    assert(f.truth_boundary == kTB &&  "backend_fallback: cost truth_boundary must be V1 string");
    assert(f.status == "evaluated" &&  "backend_fallback: V0 status must be evaluated");
    assert(f.penalty_score == 20 &&    "backend_fallback: V0 penalty_score must be 20 (backward compat)");
  }

  // ---------------------------------------------------------------------------
  // Scenario C: layout_conversion, no boundary ops.
  // V1 total: layout_transform_standalone(4) = 4.
  // V0 penalty_score: 4 (same).
  // ---------------------------------------------------------------------------
  std::puts("[C] layout_conversion");
  {
    const auto& f = results[2];
    assert(f.layout_transform == 4 && "layout_conversion: layout_transform_cost must be 4 (standalone)");
    assert(f.total == 4 &&            "layout_conversion: total must be 4");
    assert(f.componentSum() == f.total && "layout_conversion: total must equal component sum");
    assert(f.model_id == kModelId &&  "layout_conversion: cost_model_id correct");
    assert(f.status == "evaluated" && "layout_conversion: V0 status must be evaluated");
    assert(f.penalty_score == 4 &&    "layout_conversion: V0 penalty_score must be 4 (backward compat)");
  }

  // ---------------------------------------------------------------------------
  // Scenario D: cast_conversion, no boundary ops.
  // V1 total: cast_cost(2) = 2.
  // V0 penalty_score: 2 (same).
  // ---------------------------------------------------------------------------
  std::puts("[D] cast_conversion");
  {
    const auto& f = results[3];
    assert(f.cast == 2 &&             "cast_conversion: cast_cost must be 2");
    assert(f.total == 2 &&            "cast_conversion: total must be 2");
    assert(f.componentSum() == f.total && "cast_conversion: total must equal component sum");
    assert(f.model_id == kModelId &&  "cast_conversion: cost_model_id correct");
    assert(f.status == "evaluated" && "cast_conversion: V0 status must be evaluated");
    assert(f.penalty_score == 2 &&    "cast_conversion: V0 penalty_score must be 2 (backward compat)");
  }

  // ---------------------------------------------------------------------------
  // Scenario E: unsupported — compile-time rejection sentinel.
  // V1 total: unsupported_penalty(100) = 100.
  // V0 status: "rejected". V0 penalty_score: 100.
  // ---------------------------------------------------------------------------
  std::puts("[E] unsupported");
  {
    const auto& f = results[4];
    assert(f.unsupported == 100 &&    "unsupported: unsupported_penalty must be 100 (compile-time sentinel)");
    assert(f.total == 100 &&          "unsupported: total must be 100");
    assert(f.componentSum() == f.total && "unsupported: total must equal component sum");
    assert(f.model_id == kModelId &&  "unsupported: cost_model_id correct");
    assert(f.truth_boundary == kTB && "unsupported: cost truth_boundary must be V1 string");
    assert(f.status == "rejected" &&  "unsupported: V0 status must be rejected");
    assert(f.penalty_score == 100 &&  "unsupported: V0 penalty_score must be 100 (backward compat)");
  }

  std::puts("ServingStaticCostModelV1Test: PASS");
  return 0;
}
