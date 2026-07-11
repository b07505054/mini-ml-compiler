#include "serving/ExecutionPlanBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>

namespace mlir::hir {
namespace {

// Default truth_boundary used when a collected field has no V1 source attr.
static const char kPartialTB[] =
    "decision_collected_from_v1_mlir_attrs_evidence_not_tracked";

// Serving phase detection — mirrors the serving phase analysis pass output.
// Walk nested ops for llm.attention_prefill/decode; fall back to serving.phase attr.
static ServingPhase detectServingPhase(mlir::func::FuncOp funcOp) {
  ServingPhase phase = ServingPhase::Unknown;
  funcOp.walk([&](mlir::Operation *op) -> mlir::WalkResult {
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "llm.attention_prefill") {
      phase = ServingPhase::Prefill;
      return mlir::WalkResult::interrupt();
    }
    if (name == "llm.attention_decode") {
      phase = ServingPhase::Decode;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (phase != ServingPhase::Unknown)
    return phase;
  funcOp.walk([&](mlir::Operation *op) -> mlir::WalkResult {
    if (auto a = op->getAttrOfType<mlir::StringAttr>("serving.phase")) {
      llvm::StringRef v = a.getValue();
      if (v == "prefill")      phase = ServingPhase::Prefill;
      else if (v == "decode")  phase = ServingPhase::Decode;
      if (phase != ServingPhase::Unknown)
        return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return phase;
}

// Attribute helpers — read a specific attr from an op or function.
static std::string strOp(mlir::Operation *op, llvm::StringRef key) {
  if (auto a = op->getAttrOfType<mlir::StringAttr>(key))
    return a.getValue().str();
  return {};
}

static std::string strFn(mlir::func::FuncOp f, llvm::StringRef key) {
  if (auto a = f->getAttrOfType<mlir::StringAttr>(key))
    return a.getValue().str();
  return {};
}

static bool boolFn(mlir::func::FuncOp f, llvm::StringRef key) {
  if (auto a = f->getAttrOfType<mlir::BoolAttr>(key))
    return a.getValue();
  return false;
}

static double floatFn(mlir::func::FuncOp f, llvm::StringRef key) {
  if (auto a = f->getAttrOfType<mlir::FloatAttr>(key))
    return a.getValueAsDouble();
  return 0.0;
}

static bool boolOp(mlir::Operation *op, llvm::StringRef key) {
  if (auto a = op->getAttrOfType<mlir::BoolAttr>(key))
    return a.getValue();
  return false;
}

static std::string dtypeOf(mlir::Type type) {
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
  if (!shaped)
    return {};
  mlir::Type element = shaped.getElementType();
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
  std::string text;
  llvm::raw_string_ostream os(text);
  element.print(os);
  return text;
}

static std::string layoutOf(mlir::Type type) {
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
  if (!shaped || !shaped.hasRank())
    return {};
  if (shaped.getRank() == 4)
    return "nchw";
  if (shaped.getRank() == 3 || shaped.getRank() == 2)
    return "row_major";
  return "ranked_tensor";
}

static int64_t byteSizeOf(mlir::Type type) {
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return 0;
  unsigned bytes = 0;
  mlir::Type element = shaped.getElementType();
  if (element.isF32() || element.isInteger(32))
    bytes = 4;
  else if (element.isF16() || element.isBF16())
    bytes = 2;
  else if (element.isInteger(8))
    bytes = 1;
  else
    return 0;
  int64_t elems = 1;
  for (int64_t dim : shaped.getShape())
    elems *= dim;
  return elems * static_cast<int64_t>(bytes);
}

static std::vector<int64_t> shapeOf(mlir::Type type) {
  std::vector<int64_t> shape;
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
  if (!shaped || !shaped.hasRank())
    return shape;
  for (int64_t dim : shaped.getShape())
    shape.push_back(dim);
  return shape;
}

} // namespace

ExecutionPlan ExecutionPlanBuilder::build(
    mlir::ModuleOp module,
    const CapabilityBundle& capabilities,
    llvm::StringRef plan_id) {

  ExecutionPlan plan;
  plan.plan_id = plan_id.str();

  plan.model_identity = collectModelIdentity(module);

  plan.provenance.compiler_tool   = "compile-for-target";
  plan.provenance.model_spec_ref  = {};  // not in V1 module attrs
  plan.provenance.capability_bundle = collectBundleRefs(module, capabilities);
  plan.provenance.truth_boundary  =
      "execution_planning_declared_profiles_not_measured_runtime";

  plan.global_decisions = collectGlobalDecisions(module, capabilities);
  plan.cv_extension = collectCVPlanExtension(module);

  module.walk([&](mlir::func::FuncOp funcOp) {
    // Same gate attr as V1 builder: only collect annotated functions.
    if (!funcOp->getAttr("serving.policy"))
      return;
    plan.function_plans.push_back(
        collectFunctionDecisions(funcOp, capabilities));
  });

  return plan;
}

ModelIdentity
ExecutionPlanBuilder::collectModelIdentity(mlir::ModuleOp module) {
  ModelIdentity id;
  if (auto a = module->getAttrOfType<mlir::StringAttr>("llm.model"))
    id.model_id = a.getValue().str();
  if (id.model_id.empty())
    if (auto a = module->getAttrOfType<mlir::StringAttr>("cv.model"))
      id.model_id = a.getValue().str();
  if (id.model_id.empty()) {
    module.walk([&](mlir::func::FuncOp funcOp) {
      if (id.model_id.empty())
        if (auto a = funcOp->getAttrOfType<mlir::StringAttr>("cv.model_family"))
          id.model_id = a.getValue().str();
    });
  }
  if (auto a = module->getAttrOfType<mlir::StringAttr>("cv.model_family"))
    id.model_family = a.getValue().str();
  if (id.model_family.empty()) {
    module.walk([&](mlir::func::FuncOp funcOp) {
      if (id.model_family.empty())
        if (auto a = funcOp->getAttrOfType<mlir::StringAttr>("cv.model_family"))
          id.model_family = a.getValue().str();
    });
  }
  if (auto a = module->getAttrOfType<mlir::IntegerAttr>("llm.num_layers"))
    id.num_layers = a.getInt();
  if (auto a = module->getAttrOfType<mlir::IntegerAttr>("llm.hidden_size"))
    id.hidden_size = a.getInt();
  if (auto a = module->getAttrOfType<mlir::IntegerAttr>("llm.num_attention_heads"))
    id.num_attention_heads = a.getInt();
  if (auto a = module->getAttrOfType<mlir::IntegerAttr>("llm.num_key_value_heads"))
    id.num_kv_heads = a.getInt();
  // attention_mechanism and positional_encoding: no V1 module attrs for these.
  id.truth_boundary = !id.model_id.empty() && !id.model_family.empty() &&
      id.model_family == "yoloseg"
      ? "cv_semantic_attrs_from_upstream_mlir_static_contracts"
      : (!id.model_id.empty() ? "declared_model_config_not_full_graph_import"
                              : kPartialTB);
  return id;
}

CapabilityBundleRefs
ExecutionPlanBuilder::collectBundleRefs(
    mlir::ModuleOp module,
    const CapabilityBundle& capabilities) {
  CapabilityBundleRefs refs;

  // hardware_profile_ref: capabilities take precedence; fall back to module attr.
  if (!capabilities.hardware.hardware_id.empty())
    refs.hardware_profile_ref = capabilities.hardware.hardware_id;
  else if (auto a = module->getAttrOfType<mlir::StringAttr>("target.profile_id"))
    refs.hardware_profile_ref = a.getValue().str();

  // backend and kernel refs: from capability bundle fields.
  for (const auto& b : capabilities.backends)
    if (!b.backend_name.empty())
      refs.backend_profile_refs.push_back(b.backend_name);

  for (const auto& k : capabilities.kernels) {
    if (k.kernel_library.empty()) continue;
    bool seen = false;
    for (const auto& r : refs.kernel_profile_refs)
      if (r == k.kernel_library) { seen = true; break; }
    if (!seen)
      refs.kernel_profile_refs.push_back(k.kernel_library);
  }

  return refs;
}

GlobalDecisions
ExecutionPlanBuilder::collectGlobalDecisions(
    mlir::ModuleOp module,
    const CapabilityBundle& capabilities) {
  GlobalDecisions gd;

  // Global quantization: from module-level quantization.plan_dtype attr
  // (set by QuantizationPlanningPass when it ran).
  gd.quantization = attrToGlobalQuantDecision(module);

  // Serving and memory decisions: collected from the first function annotated
  // with serving.policy.  The serving topology and KV layout are effectively
  // compilation-wide decisions expressed per-function in V1 attrs; collecting
  // from the first annotated function gives a valid global view.
  module.walk([&](mlir::func::FuncOp funcOp) -> mlir::WalkResult {
    if (!funcOp->getAttr("serving.policy"))
      return mlir::WalkResult::advance();
    if (!gd.serving)
      gd.serving = attrToServingDecision(funcOp);
    if (!gd.memory)
      gd.memory = attrToMemoryDecision(funcOp, capabilities);
    return mlir::WalkResult::interrupt();
  });

  // CalibrationDecision: no V1 attrs for this; stays nullopt.
  // gd.calibration remains nullopt.

  return gd;
}

std::optional<CVPlanExtension>
ExecutionPlanBuilder::collectCVPlanExtension(mlir::ModuleOp module) {
  CVPlanExtension ext;
  mlir::func::FuncOp selected;
  module.walk([&](mlir::func::FuncOp funcOp) {
    if (!selected &&
        funcOp->getAttrOfType<mlir::StringAttr>("cv.execution_plan.status"))
      selected = funcOp;
  });
  if (!selected)
    return std::nullopt;

  ext.function_name = selected.getName().str();
  ext.model_family = strFn(selected, "cv.model_family");
  if (ext.model_family.empty())
    ext.model_family = strOp(module.getOperation(), "cv.model_family");
  ext.target_profile_id = strOp(module.getOperation(), "target.profile_id");
  ext.truth_boundary = strFn(selected, "cv.execution_plan.truth_boundary");
  ext.postprocess_boundary = "model_output_boundary";

  auto fnType = selected.getFunctionType();
  for (auto [index, type] : llvm::enumerate(fnType.getInputs())) {
    TensorContract tc;
    tc.tensor_id = "arg_" + std::to_string(index);
    tc.shape = shapeOf(type);
    tc.dtype = dtypeOf(type);
    tc.layout = layoutOf(type);
    tc.role = "graph_input";
    ext.estimated_input_bytes += byteSizeOf(type);
    ext.inputs.push_back(std::move(tc));
  }

  mlir::func::ReturnOp returnOp;
  if (!selected.getBody().empty())
    returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
        selected.getBody().front().getTerminator());
  if (returnOp) {
    for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
      TensorContract tc;
      tc.tensor_id = "result_" + std::to_string(index);
      tc.shape = shapeOf(operand.getType());
      tc.dtype = dtypeOf(operand.getType());
      tc.layout = layoutOf(operand.getType());
      tc.role = "graph_output";
      if (mlir::Operation *producer = operand.getDefiningOp()) {
        if (auto role =
                producer->getAttrOfType<mlir::StringAttr>("cv.output_role"))
          tc.role = role.getValue().str();
        if (auto boundary = producer->getAttrOfType<mlir::StringAttr>(
                "cv.postprocess_boundary"))
          ext.postprocess_boundary = boundary.getValue().str();
      }
      ext.estimated_output_bytes += byteSizeOf(operand.getType());
      ext.outputs.push_back(std::move(tc));
    }
  }

  struct RegionAccum {
    std::string role;
    std::string confidence;
    int64_t count = 0;
    std::set<std::string> scales;
  };
  std::map<std::string, RegionAccum> regions;
  selected.walk([&](mlir::Operation *op) {
    auto id = op->getAttrOfType<mlir::StringAttr>("cv.region_id");
    if (!id)
      return;
    RegionAccum &acc = regions[id.getValue().str()];
    ++acc.count;
    if (acc.role.empty())
      if (auto role = op->getAttrOfType<mlir::StringAttr>("cv.semantic_role"))
        acc.role = role.getValue().str();
    if (acc.confidence.empty())
      if (auto conf =
              op->getAttrOfType<mlir::StringAttr>("cv.recognition_confidence"))
        acc.confidence = conf.getValue().str();
    if (auto scale = op->getAttrOfType<mlir::StringAttr>("cv.feature_scale"))
      acc.scales.insert(scale.getValue().str());
  });
  for (const auto &entry : regions) {
    CVSemanticRegion region;
    region.region_id = entry.first;
    region.semantic_role = entry.second.role;
    region.recognition_confidence = entry.second.confidence;
    region.operation_count = entry.second.count;
    for (const auto &scale : entry.second.scales)
      region.feature_scales.push_back(scale);
    ext.semantic_regions.push_back(std::move(region));
  }

  if (auto a = selected->getAttrOfType<mlir::IntegerAttr>(
          "cv.memory.estimated_temporary_bytes"))
    ext.estimated_temporary_bytes = a.getInt();
  if (auto a = selected->getAttrOfType<mlir::IntegerAttr>(
          "cv.memory.estimated_total_tensor_bytes"))
    ext.estimated_total_tensor_bytes = a.getInt();
  if (ext.estimated_total_tensor_bytes == 0) {
    ext.estimated_total_tensor_bytes = ext.estimated_input_bytes +
        ext.estimated_output_bytes + ext.estimated_temporary_bytes;
  }

  return ext;
}

FunctionPlan
ExecutionPlanBuilder::collectFunctionDecisions(
    mlir::func::FuncOp funcOp,
    const CapabilityBundle&) {
  FunctionPlan fp;
  fp.function_name    = funcOp.getName().str();
  fp.serving_phase    = detectServingPhase(funcOp);
  fp.backend          = attrToBackendDecision(funcOp);
  fp.per_op_decisions = collectPerOpDecisionBundles(funcOp);
  return fp;
}

std::vector<PerOpDecisionBundle>
ExecutionPlanBuilder::collectPerOpDecisionBundles(mlir::func::FuncOp funcOp) {
  std::vector<PerOpDecisionBundle> bundles;
  if (funcOp.getBody().empty())
    return bundles;

  mlir::Block& entry = funcOp.getBody().front();
  int opIndex = 0;
  for (mlir::Operation& op : entry.without_terminator()) {
    // Compiler-materialized boundary ops (hir.cast) are not planned ops.
    // Skip them WITHOUT consuming an op index so op_N naming matches the
    // pre-materialization plan exactly; their presence is reported through
    // the anchor op's materialized_boundary_ops instead.
    if (op.getAttr("materialized.by"))
      continue;
    auto quant    = attrToPerOpQuantDecision(&op, opIndex);
    auto kernel   = attrToKernelDecision(&op, opIndex);
    auto fallback = attrToFallbackDecision(&op, opIndex);

    // Boundary ops actually inserted / explicitly deferred by
    // BoundaryMaterializationPass. Absent attrs (materialization did not
    // run) leave both lists empty — the plan then reports planning only.
    std::vector<std::string> materialized;
    std::vector<std::string> deferred;
    if (auto arr =
            op.getAttrOfType<mlir::ArrayAttr>("boundary.materialized_ops"))
      for (mlir::Attribute elem : arr)
        if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
          materialized.push_back(s.getValue().str());
    if (auto arr = op.getAttrOfType<mlir::ArrayAttr>(
            "boundary.materialization.deferred"))
      for (mlir::Attribute elem : arr)
        if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
          deferred.push_back(s.getValue().str());

    // Shape-derived static cost estimate promoted by PlanSelectionPass from
    // the winning candidate. Present only for supported op kinds with static
    // shapes — absence means the op honestly used the V1 fixed model.
    std::optional<ShapeCostEstimate> shapeCost;
    if (auto st = op.getAttrOfType<mlir::StringAttr>(
            "selected_plan.shape_cost.status")) {
      ShapeCostEstimate sc;
      sc.status = st.getValue().str();
      auto rI64 = [&](llvm::StringRef key) -> int64_t {
        if (auto a = op.getAttrOfType<mlir::IntegerAttr>(
                ("selected_plan.shape_cost." + key).str()))
          return a.getInt();
        return 0;
      };
      auto rOptI64 = [&](llvm::StringRef key) -> std::optional<int64_t> {
        if (auto a = op.getAttrOfType<mlir::IntegerAttr>(
                ("selected_plan.shape_cost." + key).str()))
          return a.getInt();
        return std::nullopt;
      };
      sc.flops_estimate              = rI64("flops_estimate");
      sc.input_bytes_estimate        = rI64("input_bytes_estimate");
      sc.output_bytes_estimate       = rI64("output_bytes_estimate");
      sc.weight_bytes_estimate       = rI64("weight_bytes_estimate");
      sc.total_memory_bytes_estimate = rI64("total_memory_bytes_estimate");
      sc.arithmetic_intensity_milli  = rI64("arithmetic_intensity_milli");
      sc.estimated_compute_cost_nanos  = rOptI64("estimated_compute_cost_nanos");
      sc.estimated_memory_cost_nanos   = rOptI64("estimated_memory_cost_nanos");
      sc.estimated_boundary_cost_nanos = rOptI64("estimated_boundary_cost_nanos");
      sc.estimated_total_cost_nanos    = rOptI64("estimated_total_cost_nanos");
      sc.cost_model_version = strOp(&op, "selected_plan.shape_cost.model_version");
      sc.truth_boundary = strOp(&op, "selected_plan.shape_cost.truth_boundary");
      shapeCost = std::move(sc);
    }

    // Static tile plan from TilePlanningPass (matmul-like ops on targets
    // that declare local memory).
    std::optional<TilePlan> tilePlan;
    if (auto st = op.getAttrOfType<mlir::StringAttr>("tile.plan.status")) {
      TilePlan tp;
      tp.status = st.getValue().str();
      auto rI64 = [&](llvm::StringRef key) -> int64_t {
        if (auto a = op.getAttrOfType<mlir::IntegerAttr>(
                ("tile.plan." + key).str()))
          return a.getInt();
        return 0;
      };
      if (auto shape = op.getAttrOfType<mlir::ArrayAttr>("tile.plan.shape");
          shape && shape.size() == 3) {
        auto dim = [&](unsigned i) -> int64_t {
          if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(shape[i]))
            return ia.getInt();
          return 0;
        };
        tp.tile_m = dim(0);
        tp.tile_n = dim(1);
        tp.tile_k = dim(2);
      }
      tp.local_memory_bytes  = rI64("local_memory_bytes");
      tp.rejected_tile_count = rI64("rejected_tile_count");
      tp.estimated_global_traffic_bytes =
          rI64("estimated_global_traffic_bytes");
      if (auto a =
              op.getAttrOfType<mlir::BoolAttr>("tile.plan.double_buffer_fits"))
        tp.double_buffer_fits = a.getValue();
      tp.staging_capability = strOp(&op, "tile.plan.staging_capability");
      tp.rejection_reason   = strOp(&op, "tile.plan.rejection_reason");
      tp.deferred_reason    = strOp(&op, "tile.plan.deferred_reason");
      tp.truth_boundary     = strOp(&op, "tile.plan.truth_boundary");
      tilePlan = std::move(tp);
    }

    // Concrete runtime-kernel contract selection from KernelSelectionPass
    // (kernel_selection_contract_v1). Present on every op when the pass
    // ran — including explicit deferrals when no registry was declared.
    std::optional<KernelSelection> kernelSelection;
    if (auto st =
            op.getAttrOfType<mlir::StringAttr>("kernel_selection.status")) {
      KernelSelection ks;
      ks.status = st.getValue().str();
      ks.selected_kernel_id = strOp(&op, "kernel_selection.selected_id");
      ks.source             = strOp(&op, "kernel_selection.source");
      ks.contract_version   = strOp(&op, "kernel_selection.contract_version");
      ks.truth_boundary     = strOp(&op, "kernel_selection.truth_boundary");
      if (auto arr = op.getAttrOfType<mlir::ArrayAttr>(
              "kernel_selection.rejection_reasons"))
        for (mlir::Attribute elem : arr)
          if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
            ks.rejection_reasons.push_back(s.getValue().str());
      kernelSelection = std::move(ks);
    }

    // Quantization co-design evidence (quantization_codesign_contract_v1).
    // Present only when the co-design pass ran under an explicit policy.
    std::optional<QuantizationCoDesign> quantCoDesign;
    if (auto st =
            op.getAttrOfType<mlir::StringAttr>("quant_codesign.status")) {
      QuantizationCoDesign qc;
      auto rS = [&](llvm::StringRef key) {
        return strOp(&op, ("quant_codesign." + key).str());
      };
      auto rOptI = [&](llvm::StringRef key) -> std::optional<int64_t> {
        if (auto a = op.getAttrOfType<mlir::IntegerAttr>(
                ("quant_codesign." + key).str()))
          return a.getInt();
        return std::nullopt;
      };
      qc.status  = st.getValue().str();
      qc.policy  = rS("policy");
      qc.representation    = rS("candidate.representation");
      qc.weight_dtype      = rS("candidate.weight_dtype");
      qc.activation_dtype  = rS("candidate.activation_dtype");
      qc.accumulator_dtype = rS("candidate.accumulator_dtype");
      qc.algorithm_status  = rS("algorithm.status");
      qc.algorithm_name    = rS("algorithm.name");
      qc.backend_legality  = rS("backend_legality");
      qc.kernel_support_status    = rS("kernel_support.status");
      qc.kernel_support_kernel_id = rS("kernel_support.kernel_id");
      qc.kernel_support_source    = rS("kernel_support.source");
      qc.accuracy_evidence_status = rS("accuracy_evidence.status");
      qc.accuracy_evidence_artifact_ref = rS("accuracy_evidence.artifact_ref");
      qc.scale_source      = rS("scale_source");
      qc.zero_point_source = rS("zero_point_source");
      qc.weight_bytes_before      = rOptI("est.weight_bytes_before");
      qc.weight_bytes_after       = rOptI("est.weight_bytes_after");
      qc.boundary_bytes           = rOptI("est.boundary_bytes");
      qc.total_cost_before_nanos  = rOptI("est.total_cost_before_nanos");
      qc.total_cost_after_nanos   = rOptI("est.total_cost_after_nanos");
      qc.systems_benefit_nanos    = rOptI("est.systems_benefit_nanos");
      if (auto arr = op.getAttrOfType<mlir::ArrayAttr>(
              "quant_codesign.est.excluded_terms"))
        for (mlir::Attribute e : arr)
          if (auto s = mlir::dyn_cast<mlir::StringAttr>(e))
            qc.excluded_cost_terms.push_back(s.getValue().str());
      if (auto a = op.getAttrOfType<mlir::BoolAttr>(
              "quant_codesign.materialization.required"))
        qc.materialization_required = a.getValue();
      qc.materialization_status = rS("materialization.status");
      if (auto arr = op.getAttrOfType<mlir::ArrayAttr>(
              "quant_codesign.rejection_reasons"))
        for (mlir::Attribute e : arr)
          if (auto s = mlir::dyn_cast<mlir::StringAttr>(e))
            qc.rejection_reasons.push_back(s.getValue().str());
      qc.truth_boundary   = rS("truth_boundary");
      qc.contract_version = rS("contract_version");
      quantCoDesign = std::move(qc);
    }

    // Per-op layout decision from LayoutPlanningPass attrs. Collected for
    // every annotated op, but it creates a bundle on its own only when a
    // transform boundary exists — keeps plan size stable for the common
    // no-transition case.
    std::optional<LayoutDecision> layout;
    if (auto eff =
            op.getAttrOfType<mlir::StringAttr>("layout.effective_layout")) {
      LayoutDecision ld;
      ld.meta.decision_id   = "ld_collected_op_" + std::to_string(opIndex);
      ld.meta.decision_type = "LayoutDecision";
      ld.meta.scope         = DecisionScope::PerOp;
      ld.meta.source_pass   = "layout-planning";
      ld.meta.reason        = strOp(&op, "layout.layout_source");
      ld.meta.truth_boundary = strOp(&op, "layout.truth_boundary");
      ld.op_type            = op.getName().getStringRef().str();
      ld.selected_layout    = eff.getValue().str();
      ld.required_input_layout = strOp(&op, "layout.required_input_layout");
      if (auto a =
              op.getAttrOfType<mlir::BoolAttr>("layout.transform_required"))
        ld.requires_layout_transform = a.getValue();
      layout = std::move(ld);
    }
    bool layoutCreatesBundle = layout && layout->requires_layout_transform;

    if (quant || kernel || fallback || !materialized.empty() ||
        !deferred.empty() || shapeCost || tilePlan || layoutCreatesBundle ||
        kernelSelection || quantCoDesign) {
      PerOpDecisionBundle bundle;
      bundle.op_name      = "op_" + std::to_string(opIndex);
      bundle.op_type      = op.getName().getStringRef().str();
      bundle.quantization = std::move(quant);
      bundle.kernel       = std::move(kernel);
      bundle.fallback     = std::move(fallback);
      bundle.materialized_boundary_ops = std::move(materialized);
      bundle.deferred_boundary_ops     = std::move(deferred);
      bundle.shape_cost   = std::move(shapeCost);
      bundle.tile_plan    = std::move(tilePlan);
      bundle.layout       = std::move(layout);
      bundle.kernel_selection = std::move(kernelSelection);
      bundle.quantization_codesign = std::move(quantCoDesign);
      bundles.push_back(std::move(bundle));
    }
    ++opIndex;
  }
  return bundles;
}

std::optional<ServingDecision>
ExecutionPlanBuilder::attrToServingDecision(mlir::func::FuncOp funcOp) {
  // Gate: serving.policy must be present (emitted by ServingPhaseAnalysisPass).
  auto policyAttr = funcOp->getAttrOfType<mlir::StringAttr>("serving.policy");
  if (!policyAttr)
    return std::nullopt;

  ServingDecision d;
  d.meta.decision_id   = "sd_collected_" + funcOp.getName().str();
  d.meta.decision_type = "ServingDecision";
  d.meta.scope         = DecisionScope::Global;
  d.meta.source_pass   = "serving-phase-analysis";

  // serving.policy maps directly to topology vocabulary ("colocated" | "pd_split").
  d.topology = policyAttr.getValue().str();

  // Cost estimates from serving pass attrs; absent → 0.0 (no inference).
  d.colocated_cost_estimate_ms = floatFn(funcOp, "serving.colocated_total_ms");
  d.pd_split_cost_estimate_ms  = floatFn(funcOp, "serving.pd_split_total_ms");

  // replay.eligible from ReplayEligibilityPass (may be absent if that pass didn't run).
  if (auto a = funcOp->getAttrOfType<mlir::BoolAttr>("replay.eligible"))
    d.replay_eligible = a.getValue();

  std::string tb = strFn(funcOp, "serving.truth_boundary");
  d.meta.truth_boundary = tb.empty() ? kPartialTB : tb;

  return d;
}

std::optional<MemoryDecision>
ExecutionPlanBuilder::attrToMemoryDecision(
    mlir::func::FuncOp funcOp,
    const CapabilityBundle& capabilities) {
  // Gate: kv.layout must be present (emitted by KVLayoutPlanningPass).
  auto layoutAttr = funcOp->getAttrOfType<mlir::StringAttr>("kv.layout");
  if (!layoutAttr)
    return std::nullopt;

  MemoryDecision d;
  d.meta.decision_id   = "md_collected_" + funcOp.getName().str();
  d.meta.decision_type = "MemoryDecision";
  d.meta.scope         = DecisionScope::Global;
  d.meta.source_pass   = "kv-layout-planning";
  d.meta.truth_boundary = kPartialTB;

  // kv_cache_layout from kv.layout attr ("paged" | "contiguous").
  d.kv_cache_layout      = layoutAttr.getValue().str();
  d.estimated_kv_peak_mb = floatFn(funcOp, "kv.byte_estimate_mb");

  // memory_budget_fraction: read from CapabilityBundle (no MLIR attr for this).
  d.memory_budget_fraction = capabilities.deployment.memory_budget_fraction;

  // kv_block_size_tokens: no V1 attr for this. Stays 0.

  return d;
}

std::optional<QuantizationDecision>
ExecutionPlanBuilder::attrToGlobalQuantDecision(mlir::ModuleOp module) {
  // Gate: quantization.plan_dtype must be present on module
  // (emitted by QuantizationPlanningPass when it ran).
  auto dtypeAttr = module->getAttrOfType<mlir::StringAttr>("quantization.plan_dtype");
  if (!dtypeAttr)
    return std::nullopt;

  QuantizationDecision d;
  d.meta.decision_id   = "qd_global_collected";
  d.meta.decision_type = "QuantizationDecision";
  d.meta.scope         = DecisionScope::Global;

  std::string src = {};
  if (auto a = module->getAttrOfType<mlir::StringAttr>("quantization.plan_source"))
    src = a.getValue().str();
  d.meta.source_pass = src.empty() ? "quantization-planning" : src;

  std::string tb = {};
  if (auto a = module->getAttrOfType<mlir::StringAttr>("quantization.truth_boundary"))
    tb = a.getValue().str();
  d.meta.truth_boundary = tb.empty() ? kPartialTB : tb;

  // weight_dtype and activation_dtype from plan_dtype.
  d.weight_dtype     = dtypeAttr.getValue().str();
  d.activation_dtype = dtypeAttr.getValue().str();

  // Optional fields only ever set by a forced-quant profile (compile-for-
  // target's forcedQuantization block) today. Absent for every other
  // profile, so d.strategy/d.algorithm/d.quantized_model_artifact_ref stay
  // at their default-constructed "" for the existing no-quant pipeline --
  // this is an additive read, not a behavior change for existing profiles.
  if (auto a = module->getAttrOfType<mlir::StringAttr>("quantization.strategy"))
    d.strategy = a.getValue().str();
  if (auto a = module->getAttrOfType<mlir::StringAttr>("quantization.algorithm"))
    d.algorithm = a.getValue().str();
  if (auto a = module->getAttrOfType<mlir::StringAttr>("quantization.quantized_model_artifact_ref"))
    d.quantized_model_artifact_ref = a.getValue().str();

  return d;
}

BackendDecision
ExecutionPlanBuilder::attrToBackendDecision(mlir::func::FuncOp funcOp) {
  BackendDecision d;
  d.meta.decision_id   = "bd_collected_" + funcOp.getName().str();
  d.meta.decision_type = "BackendDecision";
  d.meta.scope         = DecisionScope::Function;
  d.meta.truth_boundary = kPartialTB;

  // execution_provider.primary maps to selected_backend.
  if (auto a = funcOp->getAttrOfType<mlir::StringAttr>("execution_provider.primary"))
    d.selected_backend = a.getValue().str();

  // execution_provider.fallback_chain maps to fallback_backends.
  if (auto a = funcOp->getAttrOfType<mlir::ArrayAttr>("execution_provider.fallback_chain"))
    for (mlir::Attribute elem : a)
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
        d.fallback_backends.push_back(s.getValue().str());

  // execution_provider.decision_source → source_pass.
  std::string src = strFn(funcOp, "execution_provider.decision_source");
  d.meta.source_pass = src.empty() ? "execution-provider-planning" : src;

  return d;
}

std::optional<KernelDecision>
ExecutionPlanBuilder::attrToKernelDecision(mlir::Operation* op, int opIndex) {
  // Gate: lowering.decision must be present (emitted by LoweringDecisionPlanningPass).
  auto decisionAttr = op->getAttrOfType<mlir::StringAttr>("lowering.decision");
  if (!decisionAttr)
    return std::nullopt;

  KernelDecision d;
  d.meta.decision_id   = "kd_collected_op_" + std::to_string(opIndex);
  d.meta.decision_type = "KernelDecision";
  d.meta.scope         = DecisionScope::PerOp;
  d.meta.source_pass   = "lowering-decision-planning";

  d.op_name        = "op_" + std::to_string(opIndex);
  d.op_type        = op->getName().getStringRef().str();
  d.selected_kernel  = strOp(op, "kernel.name");
  d.kernel_library   = strOp(op, "kernel.library");
  d.lowering_path    = decisionAttr.getValue().str();

  if (auto a = op->getAttrOfType<mlir::BoolAttr>("kernel.exists"))
    d.kernel_exists = a.getValue();

  // Collect required boundary ops from lowering attrs (abstract vocabulary).
  if (boolOp(op, "lowering.requires_dequant"))
    d.required_boundary_ops.push_back("dequant");
  if (boolOp(op, "lowering.requires_layout_transform"))
    d.required_boundary_ops.push_back("layout_transform");
  if (boolOp(op, "lowering.requires_cast"))
    d.required_boundary_ops.push_back("cast");

  d.meta.reason = strOp(op, "lowering.reason");

  std::string tb = strOp(op, "lowering.truth_boundary");
  d.meta.truth_boundary = tb.empty() ? kPartialTB : tb;

  // Cost evidence: present only when ServingCostModelPass and PlanSelectionPass both ran.
  // Gate: selected_plan.cost.model_id non-empty (V1 cost model populated the field).
  std::string costModelId = strOp(op, "selected_plan.cost.model_id");
  if (!costModelId.empty()) {
    auto readI64Cost = [&](llvm::StringRef k) -> int64_t {
      if (auto a = op->getAttrOfType<mlir::IntegerAttr>(k)) return a.getInt();
      return 0;
    };
    DecisionCost cost;
    cost.compute_cost          = readI64Cost("selected_plan.cost.compute");
    cost.memory_cost           = readI64Cost("selected_plan.cost.memory");
    cost.dequant_cost          = readI64Cost("selected_plan.cost.dequant");
    cost.requant_cost          = readI64Cost("selected_plan.cost.requant");
    cost.layout_transform_cost = readI64Cost("selected_plan.cost.layout_transform");
    cost.cast_cost             = readI64Cost("selected_plan.cost.cast");
    cost.backend_switch_cost   = readI64Cost("selected_plan.cost.backend_switch");
    cost.launch_overhead_cost  = readI64Cost("selected_plan.cost.launch_overhead");
    cost.kv_cache_cost         = readI64Cost("selected_plan.cost.kv_cache");
    cost.transfer_cost         = readI64Cost("selected_plan.cost.transfer");
    cost.unsupported_penalty   = readI64Cost("selected_plan.cost.unsupported");
    cost.total_cost            = readI64Cost("selected_plan.cost.total");
    cost.cost_model_id         = costModelId;
    cost.truth_boundary        = strOp(op, "selected_plan.cost.truth_boundary");

    // Debug-assert: the pass must emit total == sum of components.
    assert(cost.total_cost ==
           (cost.compute_cost + cost.memory_cost + cost.dequant_cost
            + cost.requant_cost + cost.layout_transform_cost + cost.cast_cost
            + cost.backend_switch_cost + cost.launch_overhead_cost
            + cost.kv_cache_cost + cost.transfer_cost + cost.unsupported_penalty) &&
           "selected_plan.cost.total is inconsistent with component sum");

    d.meta.evidence.cost = std::move(cost);
  }

  return d;
}

std::optional<QuantizationDecision>
ExecutionPlanBuilder::attrToPerOpQuantDecision(mlir::Operation* op, int opIndex) {
  // Gate: quant.strategy must be present (emitted by QuantizationStrategyPlanningPass).
  auto stratAttr = op->getAttrOfType<mlir::StringAttr>("quant.strategy");
  if (!stratAttr)
    return std::nullopt;

  QuantizationDecision d;
  d.meta.decision_id   = "qd_collected_op_" + std::to_string(opIndex);
  d.meta.decision_type = "QuantizationDecision";
  d.meta.scope         = DecisionScope::PerOp;
  d.meta.source_pass   = "quantization-strategy-planning";

  d.op_type            = op->getName().getStringRef().str();
  d.strategy           = stratAttr.getValue().str();
  d.weight_dtype       = strOp(op, "quant.weight_dtype");
  d.activation_dtype   = strOp(op, "quant.activation_dtype");
  d.accumulation_dtype = strOp(op, "quant.accumulation_dtype");
  d.granularity        = strOp(op, "quant.granularity");
  d.accuracy_risk      = strOp(op, "quant.accuracy_risk");

  d.meta.reason = strOp(op, "quant.decision_reason");

  std::string tb = strOp(op, "quant.truth_boundary");
  d.meta.truth_boundary = tb.empty() ? kPartialTB : tb;

  return d;
}

std::optional<FallbackDecision>
ExecutionPlanBuilder::attrToFallbackDecision(mlir::Operation* op, int opIndex) {
  // Gate: lowering.decision must be exactly "fallback_backend".
  auto decisionAttr = op->getAttrOfType<mlir::StringAttr>("lowering.decision");
  if (!decisionAttr || decisionAttr.getValue() != "fallback_backend")
    return std::nullopt;

  FallbackDecision d;
  d.meta.decision_id   = "fd_collected_op_" + std::to_string(opIndex);
  d.meta.decision_type = "FallbackDecision";
  d.meta.scope         = DecisionScope::PerOp;
  d.meta.source_pass   = "lowering-decision-planning";
  d.meta.truth_boundary = kPartialTB;

  d.op_name          = "op_" + std::to_string(opIndex);
  d.op_type          = op->getName().getStringRef().str();
  d.fallback_kind    = "alternative_backend";
  d.fallback_backend = strOp(op, "lowering.target_backend");
  // tried_paths: no V1 attr tracks previously tried paths. Stays empty.

  return d;
}

void ExecutionPlanBuilder::dumpSummary(const ExecutionPlan& plan,
                                         llvm::raw_ostream& os) {
  os << "ExecutionPlan schema=" << plan.schema_version
     << " id=" << plan.plan_id << "\n";
  os << "  model=" << plan.model_identity.model_id
     << " layers=" << plan.model_identity.num_layers
     << " kv_heads=" << plan.model_identity.num_kv_heads << "\n";

  const auto& gd = plan.global_decisions;
  if (gd.serving)
    os << "  global.serving topology=" << gd.serving->topology
       << " tb=" << gd.serving->meta.truth_boundary << "\n";
  if (gd.memory)
    os << "  global.memory kv_layout=" << gd.memory->kv_cache_layout
       << " budget=" << gd.memory->memory_budget_fraction << "\n";
  if (gd.quantization)
    os << "  global.quant strategy=" << gd.quantization->strategy
       << " weight_dtype=" << gd.quantization->weight_dtype << "\n";
  if (gd.calibration)
    os << "  global.calibration kind=" << gd.calibration->calibration_kind << "\n";

  for (const auto& fp : plan.function_plans) {
    os << "  function=" << fp.function_name
       << " phase="
       << (fp.serving_phase == ServingPhase::Prefill ? "prefill"
           : fp.serving_phase == ServingPhase::Decode ? "decode" : "unknown")
       << " backend=" << fp.backend.selected_backend
       << " tb=" << fp.backend.meta.truth_boundary << "\n";
    for (const auto& pod : fp.per_op_decisions) {
      os << "    op=" << pod.op_name << " type=" << pod.op_type;
      if (pod.kernel)
        os << " lowering_path=" << pod.kernel->lowering_path;
      if (pod.quantization)
        os << " quant_strategy=" << pod.quantization->strategy;
      if (pod.fallback)
        os << " fallback_backend=" << pod.fallback->fallback_backend;
      os << "\n";
    }
  }
}

} // namespace mlir::hir
