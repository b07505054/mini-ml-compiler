// CTest unit test for Phase 26 dispatch-unit materialization in
// ExecutionPlanBuilder.
//
// Architecture contract under test:
//   For CV full-graph functions carrying the emitter's source-provenance
//   contract (source.dispatch_group / source.op_role attrs), the builder
//   groups top-level MLIR ops into DispatchUnits keyed by source identity,
//   classifies every op exactly once, collects the typed tensor ABI, and
//   suppresses the per-op decision list. It remains a COLLECTOR: every
//   value asserted here is sourced from an attr in the test module.
//
// The module below emulates the emitter's op-role vocabulary with tensor
// ops only — grouping and classification are attr-driven, not op-name
// driven. Five source nodes: Conv (multi-op), Sigmoid, Concat, Split
// (multi-output), Softmax (multi-op).
//
// No GoogleTest.  No Python.  No JSON parsing.  Pure C++ + MLIR.

#include "serving/ExecutionPlanBuilder.h"
#include "capability/CapabilityBundle.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <cassert>
#include <cstdio>
#include <string>

static const char kCVModule[] = R"mlir(
module attributes {
  target.profile_id = "test-cv-profile",
  source.provenance_contract = "generic_emitter_source_attrs_v1",
  source.model_artifact = "models/test-model.onnx",
  cv.model_family = "yoloseg"
} {
  func.func @main_graph(
      %img: tensor<1x3x4x4xf32> {source.name = "images", source.arg_role = "model_input", source.arg_index = 0 : i64},
      %w: tensor<1x2x4x4xf32> {source.name = "m0.conv.weight", source.arg_role = "weight", source.arg_index = 1 : i64},
      %b: tensor<2xf32> {source.name = "m0.conv.bias", source.arg_role = "bias", source.arg_index = 2 : i64})
      -> tensor<1x2x4x4xf32>
      attributes {
        serving.policy = "cv_full_graph",
        serving.truth_boundary = "test",
        execution_provider.primary = "coreml",
        execution_provider.fallback_chain = ["metal", "cpu"],
        execution_provider.decision_source = "cv-target-profile-static-policy",
        cv.model_family = "yoloseg",
        cv.execution_plan.status = "completed",
        cv.execution_plan.truth_boundary = "test",
        cv.memory.estimated_input_bytes = 100 : i64,
        cv.memory.estimated_output_bytes = 10 : i64,
        cv.memory.estimated_temporary_bytes = 500 : i64,
        cv.memory.estimated_total_tensor_bytes = 610 : i64,
        cv.memory.model_input_bytes = 60 : i64,
        cv.memory.initializer_bytes = 40 : i64,
        cv.memory.model_output_bytes = 10 : i64,
        cv.memory.total_intermediate_tensor_bytes = 490 : i64,
        cv.memory.total_intermediate_write_bytes = 500 : i64,
        cv.memory.peak_live_temporary_bytes = 123 : i64,
        cv.memory.workspace_bytes = 0 : i64,
        cv.memory.truth_boundary = "test_memory"
      } {
    %c0 = arith.constant {source.graph_node_id = 0 : i64, source.imported_node_id = 0 : i64, source.op_type = "Conv", source.generic_op = "nn.conv2d", source.onnx_name = "/m0/Conv", source.dispatch_group = "dg_0", source.op_role = "scalar_helper"} 0.000000e+00 : f32
    %e0 = tensor.empty() {source.graph_node_id = 0 : i64, source.imported_node_id = 0 : i64, source.op_type = "Conv", source.generic_op = "nn.conv2d", source.onnx_name = "/m0/Conv", source.dispatch_group = "dg_0", source.op_role = "allocation_helper"} : tensor<1x2x4x4xf32>
    %x0 = tensor.extract_slice %img[0, 0, 0, 0] [1, 2, 4, 4] [1, 1, 1, 1] {source.graph_node_id = 0 : i64, source.imported_node_id = 0 : i64, source.op_type = "Conv", source.generic_op = "nn.conv2d", source.onnx_name = "/m0/Conv", source.dispatch_group = "dg_0", source.op_role = "dispatch_internal_compute"} : tensor<1x3x4x4xf32> to tensor<1x2x4x4xf32>
    %s0 = tensor.insert_slice %w into %x0[0, 0, 0, 0] [1, 2, 4, 4] [1, 1, 1, 1] {source.graph_node_id = 0 : i64, source.imported_node_id = 0 : i64, source.op_type = "Conv", source.generic_op = "nn.conv2d", source.onnx_name = "/m0/Conv", source.dispatch_group = "dg_0", source.op_role = "dispatch_internal_compute"} : tensor<1x2x4x4xf32> into tensor<1x2x4x4xf32>
    %r0 = tensor.insert_slice %s0 into %e0[0, 0, 0, 0] [1, 2, 4, 4] [1, 1, 1, 1] {source.graph_node_id = 0 : i64, source.imported_node_id = 0 : i64, source.op_type = "Conv", source.generic_op = "nn.conv2d", source.onnx_name = "/m0/Conv", source.dispatch_group = "dg_0", source.op_role = "dispatch_root", lowering.decision = "fallback_backend", lowering.target_backend = "metal", kernel_selection.status = "rejected_no_kernel_for_op"} : tensor<1x2x4x4xf32> into tensor<1x2x4x4xf32>
    %e1 = tensor.empty() {source.graph_node_id = 1 : i64, source.imported_node_id = 1 : i64, source.op_type = "Sigmoid", source.generic_op = "nn.sigmoid", source.onnx_name = "/m0/Sigmoid", source.dispatch_group = "dg_1", source.op_role = "allocation_helper"} : tensor<1x2x4x4xf32>
    %r1 = tensor.insert_slice %r0 into %e1[0, 0, 0, 0] [1, 2, 4, 4] [1, 1, 1, 1] {source.graph_node_id = 1 : i64, source.imported_node_id = 1 : i64, source.op_type = "Sigmoid", source.generic_op = "nn.sigmoid", source.onnx_name = "/m0/Sigmoid", source.dispatch_group = "dg_1", source.op_role = "dispatch_root", lowering.decision = "fallback_backend", lowering.target_backend = "metal"} : tensor<1x2x4x4xf32> into tensor<1x2x4x4xf32>
    %e2 = tensor.empty() {source.graph_node_id = 2 : i64, source.imported_node_id = 2 : i64, source.op_type = "Concat", source.generic_op = "nn.concat", source.onnx_name = "/m0/Concat", source.dispatch_group = "dg_2", source.op_role = "allocation_helper"} : tensor<1x4x4x4xf32>
    %i2 = tensor.insert_slice %r0 into %e2[0, 0, 0, 0] [1, 2, 4, 4] [1, 1, 1, 1] {source.graph_node_id = 2 : i64, source.imported_node_id = 2 : i64, source.op_type = "Concat", source.generic_op = "nn.concat", source.onnx_name = "/m0/Concat", source.dispatch_group = "dg_2", source.op_role = "dispatch_internal_compute"} : tensor<1x2x4x4xf32> into tensor<1x4x4x4xf32>
    %r2 = tensor.insert_slice %r1 into %i2[0, 2, 0, 0] [1, 2, 4, 4] [1, 1, 1, 1] {source.graph_node_id = 2 : i64, source.imported_node_id = 2 : i64, source.op_type = "Concat", source.generic_op = "nn.concat", source.onnx_name = "/m0/Concat", source.dispatch_group = "dg_2", source.op_role = "dispatch_root", lowering.decision = "fallback_backend", lowering.target_backend = "metal"} : tensor<1x2x4x4xf32> into tensor<1x4x4x4xf32>
    %r3a = tensor.extract_slice %r2[0, 0, 0, 0] [1, 2, 4, 4] [1, 1, 1, 1] {source.graph_node_id = 3 : i64, source.imported_node_id = 3 : i64, source.op_type = "Split", source.generic_op = "nn.split", source.onnx_name = "/m0/Split", source.dispatch_group = "dg_3", source.op_role = "dispatch_root", lowering.decision = "fallback_backend", lowering.target_backend = "metal"} : tensor<1x4x4x4xf32> to tensor<1x2x4x4xf32>
    %r3b = tensor.extract_slice %r2[0, 2, 0, 0] [1, 2, 4, 4] [1, 1, 1, 1] {source.graph_node_id = 3 : i64, source.imported_node_id = 3 : i64, source.op_type = "Split", source.generic_op = "nn.split", source.onnx_name = "/m0/Split", source.dispatch_group = "dg_3", source.op_role = "dispatch_root", lowering.decision = "fallback_backend", lowering.target_backend = "metal"} : tensor<1x4x4x4xf32> to tensor<1x2x4x4xf32>
    %e4 = tensor.empty() {source.graph_node_id = 4 : i64, source.imported_node_id = 4 : i64, source.op_type = "Softmax", source.generic_op = "nn.softmax", source.onnx_name = "/m0/Softmax", source.dispatch_group = "dg_4", source.op_role = "allocation_helper"} : tensor<1x2x4x4xf32>
    %m4 = tensor.insert_slice %r3a into %e4[0, 0, 0, 0] [1, 2, 4, 4] [1, 1, 1, 1] {source.graph_node_id = 4 : i64, source.imported_node_id = 4 : i64, source.op_type = "Softmax", source.generic_op = "nn.softmax", source.onnx_name = "/m0/Softmax", source.dispatch_group = "dg_4", source.op_role = "dispatch_internal_compute"} : tensor<1x2x4x4xf32> into tensor<1x2x4x4xf32>
    %r4 = tensor.insert_slice %r3b into %m4[0, 0, 0, 0] [1, 2, 4, 4] [1, 1, 1, 1] {source.graph_node_id = 4 : i64, source.imported_node_id = 4 : i64, source.op_type = "Softmax", source.generic_op = "nn.softmax", source.onnx_name = "/m0/Softmax", source.dispatch_group = "dg_4", source.op_role = "dispatch_root", lowering.decision = "fallback_backend", lowering.target_backend = "metal"} : tensor<1x2x4x4xf32> into tensor<1x2x4x4xf32>
    return %r4 : tensor<1x2x4x4xf32>
  }
}
)mlir";

static const mlir::hir::DispatchUnit *
findUnit(const mlir::hir::FunctionPlan &fp, const std::string &id) {
  for (const auto &unit : fp.dispatch_units)
    if (unit.dispatch_unit_id == id)
      return &unit;
  return nullptr;
}

int main() {
  mlir::MLIRContext ctx;
  ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect,
                  mlir::arith::ArithDialect>();

  std::puts("[1] Parsing CV module with source provenance attrs ...");
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kCVModule, &ctx);
  assert(module && "test module must parse");

  std::puts("[2] Building ExecutionPlan (collector only, no passes) ...");
  mlir::hir::CapabilityBundle bundle;
  mlir::hir::ExecutionPlan plan =
      mlir::hir::ExecutionPlanBuilder::build(module.get(), bundle,
                                             "dispatch_unit_test_plan");

  assert(plan.function_plans.size() == 1 && "one CV function plan expected");
  const mlir::hir::FunctionPlan &fp = plan.function_plans[0];

  // Per-op decisions are suppressed for CV full-graph functions.
  assert(fp.per_op_decisions.empty() &&
         "CV function must not serialize per-op MLIR decisions");
  std::puts("  [PASS] per-op decision list suppressed for CV function");

  // Five source nodes -> five dispatch units, despite 15 top-level MLIR ops.
  assert(fp.dispatch_units.size() == 5 && "five dispatch units expected");
  std::puts("  [PASS] one dispatch unit per source node");

  // Conv: 5 MLIR ops (const/empty/extract/insert/insert) -> one unit.
  const auto *conv = findUnit(fp, "du_0");
  assert(conv && "conv unit du_0 expected");
  assert(conv->mlir_operation_refs.size() == 5 && "conv folds 5 MLIR ops");
  assert(conv->source_op_type == "Conv");
  assert(conv->operation_family == "nn.conv2d");
  assert(conv->source_graph_node_ids.size() == 1 &&
         conv->source_graph_node_ids[0] == 0);
  assert(conv->source_onnx_node_names.size() == 1 &&
         conv->source_onnx_node_names[0] == "/m0/Conv");
  assert(conv->input_tensor_ids.size() == 1 &&
         conv->input_tensor_ids[0] == "arg_0" &&
         "conv reads the model input");
  assert(conv->initializer_tensor_ids.size() == 1 &&
         conv->initializer_tensor_ids[0] == "arg_1" &&
         "conv weight listed as initializer input");
  assert(conv->output_tensor_ids.size() == 1 && "conv has one output");
  std::puts("  [PASS] Conv: multiple MLIR ops -> one dispatch unit with ABI");

  // Sigmoid: helper + root -> one unit consuming conv's output.
  const auto *sigmoid = findUnit(fp, "du_1");
  assert(sigmoid && sigmoid->operation_family == "nn.sigmoid");
  assert(sigmoid->mlir_operation_refs.size() == 2);
  assert(sigmoid->input_tensor_ids.size() == 1 &&
         sigmoid->input_tensor_ids[0] == "du_0:o0" &&
         "sigmoid consumes conv unit output");
  std::puts("  [PASS] Sigmoid: one dispatch unit, cross-unit tensor id");

  // Concat: empty + 2 inserts -> one unit with two inputs.
  const auto *concat = findUnit(fp, "du_2");
  assert(concat && concat->operation_family == "nn.concat");
  assert(concat->mlir_operation_refs.size() == 3);
  assert(concat->input_tensor_ids.size() == 2);
  assert(concat->output_tensor_ids.size() == 1);
  std::puts("  [PASS] Concat: one dispatch unit");

  // Split: two extract roots -> one multi-output unit.
  const auto *split = findUnit(fp, "du_3");
  assert(split && split->operation_family == "nn.split");
  assert(split->output_tensor_ids.size() == 2 &&
         "split is one multi-output dispatch unit");
  std::puts("  [PASS] Split: one multi-output dispatch unit");

  // Softmax: helper + internal + root -> one unit.
  const auto *softmax = findUnit(fp, "du_4");
  assert(softmax && softmax->operation_family == "nn.softmax");
  assert(softmax->mlir_operation_refs.size() == 3);
  assert(softmax->input_tensor_ids.size() == 2 &&
         "softmax consumes both split outputs");
  std::puts("  [PASS] Softmax: one dispatch unit");

  // Classification totality: 15 ops, every op exactly one class, all
  // assigned; helpers never became units.
  assert(fp.op_classification && "classification expected");
  const auto &cls = *fp.op_classification;
  assert(cls.total_mlir_operations == 15);
  assert(cls.dispatch_root == 6);            // one per unit + extra split root
  assert(cls.dispatch_internal_compute == 4);
  assert(cls.allocation_helper == 4);
  assert(cls.scalar_helper == 1);
  assert(cls.tensor_contract_operation == 0);
  assert(cls.view_operation == 0);
  assert(cls.unresolved == 0);
  assert(cls.operations_assigned_to_units == 15);
  assert(cls.source_graph_node_count == 5);
  assert(cls.dispatch_root + cls.dispatch_internal_compute +
             cls.allocation_helper + cls.scalar_helper +
             cls.tensor_contract_operation + cls.view_operation +
             cls.non_dispatch_metadata + cls.unresolved ==
         cls.total_mlir_operations);
  std::puts("  [PASS] all MLIR ops classified exactly once, reconciled");

  // Backend/kernel truth: configured preference only, never executable
  // without a registered kernel.
  for (const auto &unit : fp.dispatch_units) {
    assert(unit.backend_intent.backend == "coreml");
    assert(unit.backend_intent.intent_basis == "configured_preference");
    assert(unit.kernel_status == "fallback_only");
    assert(unit.fallback_backends.size() == 1 &&
           unit.fallback_backends[0] == "metal");
    assert(!unit.executable);
    assert(unit.non_executable_reason ==
           "no_runtime_adapter_or_registered_kernel");
  }
  std::puts("  [PASS] backend intent + kernel truth per unit");

  // Tensor bindings: roles from arg attrs; outputs identified.
  assert(plan.tensor_bindings.size() == 4 && "3 args + 1 output");
  assert(plan.tensor_bindings[0].role == "model_input");
  assert(plan.tensor_bindings[0].original_name == "images");
  assert(plan.tensor_bindings[0].ownership == "caller");
  assert(plan.tensor_bindings[0].is_mutable);
  assert(plan.tensor_bindings[1].role == "weight");
  assert(plan.tensor_bindings[1].ownership == "model_state");
  assert(!plan.tensor_bindings[1].is_mutable);
  assert(plan.tensor_bindings[1].model_artifact_reference ==
         "models/test-model.onnx");
  assert(plan.tensor_bindings[2].role == "bias");
  assert(plan.tensor_bindings[3].role == "model_output");
  assert(plan.tensor_bindings[3].tensor_id == "result_0");
  std::puts("  [PASS] tensor binding ABI roles");

  // Model artifact reference flows into plan provenance.
  assert(plan.provenance.model_spec_ref == "models/test-model.onnx");
  std::puts("  [PASS] model artifact reference in provenance");

  // Corrected memory metrics collected from attrs (collector: values match
  // the attrs verbatim).
  assert(plan.cv_extension && plan.cv_extension->memory_summary);
  const auto &memory = *plan.cv_extension->memory_summary;
  assert(memory.model_input_bytes == 60);
  assert(memory.initializer_bytes == 40);
  assert(memory.model_output_bytes == 10);
  assert(memory.total_intermediate_tensor_bytes == 490);
  assert(memory.total_intermediate_write_bytes == 500);
  assert(memory.peak_live_temporary_bytes == 123);
  assert(memory.workspace_bytes == 0);
  assert(!memory.planned_slot_bytes && "no slot allocation may be claimed");
  std::puts("  [PASS] corrected memory metrics collected");

  std::puts("DispatchUnitBuilderTest: PASS");
  return 0;
}
