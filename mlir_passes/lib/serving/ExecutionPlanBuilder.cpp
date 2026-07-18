#include "serving/ExecutionPlanBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
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

static std::vector<std::string> stringArrayAttr(mlir::ArrayAttr arr) {
  std::vector<std::string> out;
  if (!arr)
    return out;
  for (mlir::Attribute elem : arr)
    if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
      out.push_back(s.getValue().str());
  return out;
}

static std::string dictStr(mlir::DictionaryAttr dict, llvm::StringRef key) {
  if (!dict)
    return {};
  if (auto a = mlir::dyn_cast_if_present<mlir::StringAttr>(dict.get(key)))
    return a.getValue().str();
  return {};
}

static double dictFloat(mlir::DictionaryAttr dict, llvm::StringRef key) {
  if (!dict)
    return 0.0;
  if (auto a = mlir::dyn_cast_if_present<mlir::FloatAttr>(dict.get(key)))
    return a.getValueAsDouble();
  return 0.0;
}

static int64_t dictInt(mlir::DictionaryAttr dict, llvm::StringRef key) {
  if (!dict)
    return 0;
  if (auto a = mlir::dyn_cast_if_present<mlir::IntegerAttr>(dict.get(key)))
    return a.getInt();
  return 0;
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

static int64_t i64InDict(mlir::DictionaryAttr dict, llvm::StringRef key,
                         int64_t fallback = 0) {
  if (auto attr = dict.get(key))
    if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(attr))
      return i.getInt();
  return fallback;
}

static std::string stringInDict(mlir::DictionaryAttr dict, llvm::StringRef key) {
  if (auto attr = dict.get(key))
    if (auto s = mlir::dyn_cast<mlir::StringAttr>(attr))
      return s.getValue().str();
  return {};
}

static std::vector<std::string> stringArrayInDict(mlir::DictionaryAttr dict,
                                                  llvm::StringRef key) {
  std::vector<std::string> values;
  if (auto attr = dict.get(key))
    if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(attr))
      for (mlir::Attribute elem : arr)
        if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
          values.push_back(s.getValue().str());
  return values;
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
  // Phase 26: the emitter stamps the model artifact reference as a module
  // attr so the runtime can locate weight data. Empty for modules emitted
  // before the provenance contract (e.g. the Qwen path).
  plan.provenance.model_spec_ref =
      strOp(module.getOperation(), "source.model_artifact");
  plan.provenance.capability_bundle = collectBundleRefs(module, capabilities);
  plan.provenance.truth_boundary  =
      "execution_planning_declared_profiles_not_measured_runtime";

  plan.global_decisions = collectGlobalDecisions(module, capabilities);
  plan.cv_extension = collectCVPlanExtension(module);
  plan.distributed = collectDistributedPlan(module);

  module.walk([&](mlir::func::FuncOp funcOp) {
    // Same gate attr as V1 builder: only collect annotated functions.
    if (!funcOp->getAttr("serving.policy"))
      return;
    plan.function_plans.push_back(
        collectFunctionDecisions(funcOp, capabilities));
    // Phase 26 tensor ABI: collected only for CV full-graph functions whose
    // arguments carry the emitter's provenance contract.
    if (strFn(funcOp, "serving.policy") == "cv_full_graph" &&
        funcOp.getNumArguments() > 0 &&
        funcOp.getArgAttrOfType<mlir::StringAttr>(0, "source.arg_role")) {
      auto bindings = collectTensorBindings(module, funcOp);
      plan.tensor_bindings.insert(plan.tensor_bindings.end(),
                                  bindings.begin(), bindings.end());
    }
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

  // Phase 26 corrected memory metrics: present only when the attrs pass
  // computed the split/lifetime fields (gate: peak-live attr present).
  if (auto peak = selected->getAttrOfType<mlir::IntegerAttr>(
          "cv.memory.peak_live_temporary_bytes")) {
    CVMemorySummary memory;
    auto readI64 = [&](llvm::StringRef key) -> int64_t {
      if (auto a = selected->getAttrOfType<mlir::IntegerAttr>(key))
        return a.getInt();
      return 0;
    };
    memory.model_input_bytes = readI64("cv.memory.model_input_bytes");
    memory.initializer_bytes = readI64("cv.memory.initializer_bytes");
    memory.model_output_bytes = readI64("cv.memory.model_output_bytes");
    memory.total_intermediate_tensor_bytes =
        readI64("cv.memory.total_intermediate_tensor_bytes");
    memory.total_intermediate_write_bytes =
        readI64("cv.memory.total_intermediate_write_bytes");
    memory.peak_live_temporary_bytes = peak.getInt();
    memory.workspace_bytes = readI64("cv.memory.workspace_bytes");
    memory.planned_slot_bytes = std::nullopt;  // no slot allocator exists
    memory.truth_boundary = strFn(selected, "cv.memory.truth_boundary");
    ext.memory_summary = std::move(memory);
  }

  ext.postprocess_contract = collectPostprocessContract(selected);

  return ext;
}

std::optional<DistributedPlan>
ExecutionPlanBuilder::collectDistributedPlan(mlir::ModuleOp module) {
  mlir::Operation *op = module.getOperation();
  // Absent entirely for TP1/legacy plans -- DistributedStrategyPlanningPass
  // only sets this attr when world_size > 1 was selected.
  auto worldSizeAttr = op->getAttrOfType<mlir::IntegerAttr>("distributed.world_size");
  if (!worldSizeAttr)
    return std::nullopt;

  DistributedPlan plan;
  plan.strategy = strOp(op, "distributed.strategy");
  plan.world_size = worldSizeAttr.getInt();
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("distributed.tensor_parallel_size"))
    plan.tensor_parallel_size = a.getInt();
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("distributed.pipeline_parallel_size"))
    plan.pipeline_parallel_size = a.getInt();
  plan.truth_boundary = strOp(op, "distributed.truth_boundary");

  if (auto ranks = op->getAttrOfType<mlir::ArrayAttr>("distributed.ranks")) {
    for (mlir::Attribute elem : ranks) {
      auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(elem);
      if (!dict)
        continue;
      DistributedRankPlacement r;
      if (auto a = dict.getAs<mlir::IntegerAttr>("rank_id"))
        r.rank_id = a.getInt();
      if (auto a = dict.getAs<mlir::StringAttr>("logical_device"))
        r.logical_device = a.getValue().str();
      plan.ranks.push_back(std::move(r));
    }
  }

  if (auto shards = op->getAttrOfType<mlir::ArrayAttr>("distributed.tensor_shards")) {
    for (mlir::Attribute elem : shards) {
      auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(elem);
      if (!dict)
        continue;
      DistributedTensorShard s;
      if (auto a = dict.getAs<mlir::StringAttr>("tensor_id"))
        s.tensor_id = a.getValue().str();
      if (auto a = dict.getAs<mlir::IntegerAttr>("partition_axis"))
        s.partition_axis = a.getInt();
      if (auto a = dict.getAs<mlir::IntegerAttr>("partition_count"))
        s.partition_count = a.getInt();
      if (auto a = dict.getAs<mlir::IntegerAttr>("shard_index"))
        s.shard_index = a.getInt();
      if (auto a = dict.getAs<mlir::IntegerAttr>("range_start"))
        s.range_start = a.getInt();
      if (auto a = dict.getAs<mlir::IntegerAttr>("range_end"))
        s.range_end = a.getInt();
      plan.tensor_shards.push_back(std::move(s));
    }
  }

  if (auto collectives = op->getAttrOfType<mlir::ArrayAttr>("distributed.collectives")) {
    for (mlir::Attribute elem : collectives) {
      auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(elem);
      if (!dict)
        continue;
      DistributedCollectiveStep c;
      if (auto a = dict.getAs<mlir::StringAttr>("collective_id"))
        c.collective_id = a.getValue().str();
      if (auto a = dict.getAs<mlir::IntegerAttr>("sequence_id"))
        c.sequence_id = a.getInt();
      if (auto a = dict.getAs<mlir::StringAttr>("kind"))
        c.kind = a.getValue().str();
      if (auto a = dict.getAs<mlir::ArrayAttr>("participants"))
        for (mlir::Attribute p : a)
          if (auto pi = mlir::dyn_cast<mlir::IntegerAttr>(p))
            c.participants.push_back(pi.getInt());
      if (auto a = dict.getAs<mlir::StringAttr>("tensor_id"))
        c.tensor_id = a.getValue().str();
      if (auto a = dict.getAs<mlir::StringAttr>("reduction"))
        c.reduction = a.getValue().str();
      plan.collectives.push_back(std::move(c));
    }
  }

  return plan;
}

FunctionPlan
ExecutionPlanBuilder::collectFunctionDecisions(
    mlir::func::FuncOp funcOp,
    const CapabilityBundle&) {
  FunctionPlan fp;
  fp.function_name    = funcOp.getName().str();
  fp.serving_phase    = detectServingPhase(funcOp);
  fp.backend          = attrToBackendDecision(funcOp);
  // Phase 26: CV full-graph functions are collected as dispatch units built
  // from the emitter's source provenance; internal MLIR ops stay out of the
  // runtime decision list. LLM functions keep the per-op decision path
  // unchanged (Qwen plans stay byte-identical).
  bool hasDispatchProvenance = false;
  if (strFn(funcOp, "serving.policy") == "cv_full_graph" &&
      !funcOp.getBody().empty()) {
    for (mlir::Operation &op :
         funcOp.getBody().front().without_terminator()) {
      if (op.getAttr("source.dispatch_group")) {
        hasDispatchProvenance = true;
        break;
      }
    }
  }
  if (hasDispatchProvenance) {
    collectDispatchUnits(funcOp, fp);
  } else {
    fp.per_op_decisions = collectPerOpDecisionBundles(funcOp);
  }
  return fp;
}

namespace {

// Weakest-wins ordering for aggregating member kernel truth into one unit
// status. Lower rank = weaker claim.
static int kernelStatusRank(llvm::StringRef status) {
  if (status == "unavailable")        return 0;
  if (status == "fallback_only")      return 1;
  if (status == "deferred")           return 2;
  if (status == "lowering_only")      return 3;
  if (status == "library_available")  return 4;
  if (status == "runtime_registered") return 5;
  return 0;
}

static std::string memberKernelStatus(mlir::Operation *op) {
  // Concrete runtime-kernel selection is the only source of a dispatchable
  // claim (kernel_selection_contract_v1).
  if (!strOp(op, "kernel_selection.selected_id").empty())
    return "runtime_registered";
  std::string ksStatus = strOp(op, "kernel_selection.status");
  std::string lowering = strOp(op, "lowering.decision");
  if (lowering == "direct_lower")
    return boolOp(op, "kernel.exists") ? "library_available" : "lowering_only";
  if (lowering == "rewrite_then_lower" || lowering == "dequant_then_lower")
    return "lowering_only";
  if (lowering == "fallback_backend")
    return "fallback_only";
  if (llvm::StringRef(ksStatus).starts_with("deferred"))
    return "deferred";
  return "unavailable";
}

} // namespace

void ExecutionPlanBuilder::collectDispatchUnits(mlir::func::FuncOp funcOp,
                                                FunctionPlan &fp) {
  if (funcOp.getBody().empty())
    return;
  mlir::Block &entry = funcOp.getBody().front();

  DispatchOpClassification cls;
  cls.source_graph_node_count = 0;

  struct UnitAccum {
    DispatchUnit unit;
    llvm::SmallPtrSet<mlir::Operation *, 16> members;
    std::vector<mlir::Operation *> roots;
    std::set<int64_t> nodeIds;
    std::set<int64_t> importedIds;
    std::set<std::string> onnxNames;
    std::set<std::string> fallbacks;
    int weakestKernelRank = 6;  // above strongest; lowered by members
    std::string weakestKernelStatus = "unavailable";
    bool sawKernelDecision = false;
    std::string selectedKernelId;
    int64_t flops = 0;
  };
  std::map<std::string, UnitAccum> accums;   // keyed by dispatch group id
  std::vector<std::string> unitOrder;        // first-occurrence order
  std::set<int64_t> allNodeIds;

  auto readI64 = [](mlir::Operation *op, llvm::StringRef key,
                    int64_t fallback) -> int64_t {
    if (auto a = op->getAttrOfType<mlir::IntegerAttr>(key))
      return a.getInt();
    return fallback;
  };

  int opIndex = 0;
  for (mlir::Operation &op : entry.without_terminator()) {
    // Compiler-materialized boundary ops are plan metadata, not source ops.
    if (op.getAttr("materialized.by")) {
      ++cls.total_mlir_operations;
      ++cls.non_dispatch_metadata;
      continue;
    }
    ++cls.total_mlir_operations;
    std::string role = strOp(&op, "source.op_role");
    std::string group = strOp(&op, "source.dispatch_group");
    if (role == "dispatch_root")                  ++cls.dispatch_root;
    else if (role == "dispatch_internal_compute") ++cls.dispatch_internal_compute;
    else if (role == "tensor_contract_operation") ++cls.tensor_contract_operation;
    else if (role == "allocation_helper")         ++cls.allocation_helper;
    else if (role == "scalar_helper")             ++cls.scalar_helper;
    else if (role == "view_operation")            ++cls.view_operation;
    else                                          ++cls.unresolved;

    if (group.empty()) {
      ++opIndex;
      continue;
    }
    ++cls.operations_assigned_to_units;

    auto inserted = accums.emplace(group, UnitAccum{});
    UnitAccum &acc = inserted.first->second;
    if (inserted.second)
      unitOrder.push_back(group);
    acc.members.insert(&op);
    acc.unit.mlir_operation_refs.push_back(
        "op_" + std::to_string(opIndex) + ":" +
        op.getName().getStringRef().str());
    if (int64_t nodeId = readI64(&op, "source.graph_node_id", -1); nodeId >= 0) {
      acc.nodeIds.insert(nodeId);
      allNodeIds.insert(nodeId);
    }
    if (int64_t impId = readI64(&op, "source.imported_node_id", -1); impId >= 0)
      acc.importedIds.insert(impId);
    if (std::string name = strOp(&op, "source.onnx_name"); !name.empty())
      acc.onnxNames.insert(name);
    if (acc.unit.source_op_type.empty())
      acc.unit.source_op_type = strOp(&op, "source.op_type");
    if (acc.unit.operation_family.empty())
      acc.unit.operation_family = strOp(&op, "source.generic_op");
    if (acc.unit.semantic_region_id.empty())
      acc.unit.semantic_region_id = strOp(&op, "cv.region_id");
    if (role == "dispatch_root")
      acc.roots.push_back(&op);
    // Kernel truth aggregation over members carrying a lowering decision or
    // a concrete kernel selection. Helper ops that carry only a blanket
    // kernel_selection rejection (e.g. scalar constants) do not participate —
    // they are unit implementation detail, not dispatch candidates.
    if (!strOp(&op, "lowering.decision").empty() ||
        !strOp(&op, "kernel_selection.selected_id").empty()) {
      std::string status = memberKernelStatus(&op);
      acc.sawKernelDecision = true;
      int rank = kernelStatusRank(status);
      if (rank < acc.weakestKernelRank) {
        acc.weakestKernelRank = rank;
        acc.weakestKernelStatus = status;
      }
      if (status == "runtime_registered" && acc.selectedKernelId.empty())
        acc.selectedKernelId = strOp(&op, "kernel_selection.selected_id");
      if (std::string fb = strOp(&op, "lowering.target_backend");
          !fb.empty() && strOp(&op, "lowering.decision") == "fallback_backend")
        acc.fallbacks.insert(fb);
    }
    acc.flops += readI64(&op, "selected_plan.shape_cost.flops_estimate", 0);
    ++opIndex;
  }
  cls.source_graph_node_count = static_cast<int64_t>(allNodeIds.size());

  // Second pass: tensor identity. Function arguments are "arg_<i>"; each
  // unit's escaping results become "<unit_id>:o<k>".
  llvm::DenseMap<mlir::Value, std::string> valueId;
  llvm::DenseMap<mlir::Value, std::string> argRole;
  for (auto [index, arg] : llvm::enumerate(funcOp.getArguments())) {
    valueId[arg] = "arg_" + std::to_string(index);
    if (auto role =
            funcOp.getArgAttrOfType<mlir::StringAttr>(index, "source.arg_role"))
      argRole[arg] = role.getValue().str();
  }

  // Unit ids come from the dispatch group ("dg_42" -> "du_42").
  auto unitIdFor = [](const std::string &group) {
    llvm::StringRef ref(group);
    if (ref.starts_with("dg_"))
      return ("du_" + ref.drop_front(3)).str();
    return ("du_" + ref).str();
  };

  // Map any op (possibly nested inside a member's region) to its top-level
  // ancestor in the entry block, so region-internal uses (tensor.pad yields,
  // tensor.generate bodies, linalg.generic bodies) resolve to their unit.
  auto topLevelAncestor = [&entry](mlir::Operation *op) -> mlir::Operation * {
    while (op && op->getBlock() != &entry)
      op = op->getParentOp();
    return op;
  };

  // Assign output ids first so later units can reference earlier outputs.
  for (const std::string &group : unitOrder) {
    UnitAccum &acc = accums[group];
    acc.unit.dispatch_unit_id = unitIdFor(group);
    int outIndex = 0;
    for (mlir::Operation &op : entry.without_terminator()) {
      if (!acc.members.contains(&op))
        continue;
      for (mlir::OpResult result : op.getResults()) {
        bool escapes = false;
        for (mlir::Operation *user : result.getUsers()) {
          mlir::Operation *top = topLevelAncestor(user);
          if (!top || !acc.members.contains(top)) {
            escapes = true;
            break;
          }
        }
        if (!escapes)
          continue;
        std::string id =
            acc.unit.dispatch_unit_id + ":o" + std::to_string(outIndex++);
        valueId[result] = id;
        acc.unit.output_tensor_ids.push_back(id);
        acc.unit.estimated_write_bytes += byteSizeOf(result.getType());
        if (acc.unit.dtype.empty())
          acc.unit.dtype = dtypeOf(result.getType());
        if (acc.unit.layout.empty())
          acc.unit.layout = layoutOf(result.getType());
      }
    }
  }

  BackendDecision backend = attrToBackendDecision(funcOp);

  for (const std::string &group : unitOrder) {
    UnitAccum &acc = accums[group];
    DispatchUnit &unit = acc.unit;
    unit.source_graph_node_ids.assign(acc.nodeIds.begin(), acc.nodeIds.end());
    unit.source_imported_node_ids.assign(acc.importedIds.begin(),
                                         acc.importedIds.end());
    unit.source_onnx_node_names.assign(acc.onnxNames.begin(),
                                       acc.onnxNames.end());

    // Inputs: values flowing into the unit from outside it — function
    // arguments or other units' outputs — deduplicated in first-use order.
    // Member regions are walked so region-internal references (e.g. the
    // tensor.extract inside a resize's tensor.generate body) are captured.
    // Initializer-role function arguments are listed separately.
    llvm::SmallPtrSet<mlir::Value, 16> seen;
    for (mlir::Operation &op : entry.without_terminator()) {
      if (!acc.members.contains(&op))
        continue;
      op.walk([&](mlir::Operation *inner) {
        for (mlir::Value operand : inner->getOperands()) {
          if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
            // Region block arguments (pad indices, linalg body scalars) are
            // unit-internal; only entry-block arguments are unit inputs.
            if (blockArg.getOwner() != &entry)
              continue;
          } else {
            mlir::Operation *top = topLevelAncestor(operand.getDefiningOp());
            if (!top || acc.members.contains(top))
              continue;
          }
          if (!seen.insert(operand).second)
            continue;
          auto idIt = valueId.find(operand);
          std::string id = idIt != valueId.end()
                               ? idIt->second
                               : std::string("unmapped_value");
          auto roleIt = argRole.find(operand);
          bool isInitializer =
              roleIt != argRole.end() && roleIt->second != "model_input";
          if (isInitializer)
            unit.initializer_tensor_ids.push_back(id);
          else
            unit.input_tensor_ids.push_back(id);
          unit.estimated_read_bytes += byteSizeOf(operand.getType());
        }
      });
    }

    unit.backend_intent.backend = backend.selected_backend;
    unit.backend_intent.intent_basis =
        backend.selected_backend.empty() ? "unavailable"
                                         : "configured_preference";
    unit.execution_domain = "unassigned";
    unit.kernel_status =
        acc.sawKernelDecision ? acc.weakestKernelStatus : "unavailable";
    unit.selected_kernel_id = acc.selectedKernelId;
    unit.fallback_backends.assign(acc.fallbacks.begin(), acc.fallbacks.end());
    unit.estimated_compute_flops = acc.flops;
    unit.workspace_bytes = 0;
    unit.decision_provenance =
        "source_attrs:generic_emitter_source_attrs_v1;"
        "passes:cv-execution-plan-attrs,kernel-availability-planning,"
        "lowering-decision-planning,kernel-selection";
    unit.executable = unit.kernel_status == "runtime_registered";
    if (!unit.executable)
      unit.non_executable_reason = "no_runtime_adapter_or_registered_kernel";
    fp.dispatch_units.push_back(std::move(unit));
  }

  fp.op_classification = cls;
}

std::vector<TensorBinding>
ExecutionPlanBuilder::collectTensorBindings(mlir::ModuleOp module,
                                            mlir::func::FuncOp funcOp) {
  std::vector<TensorBinding> bindings;
  std::string modelArtifact =
      strOp(module.getOperation(), "source.model_artifact");

  for (auto [index, arg] : llvm::enumerate(funcOp.getArguments())) {
    TensorBinding binding;
    binding.tensor_id = "arg_" + std::to_string(index);
    if (auto name =
            funcOp.getArgAttrOfType<mlir::StringAttr>(index, "source.name")) {
      binding.original_name = name.getValue().str();
      binding.source_value_id = binding.original_name;
    }
    std::string role = "initializer";
    if (auto roleAttr = funcOp.getArgAttrOfType<mlir::StringAttr>(
            index, "source.arg_role"))
      role = roleAttr.getValue().str();
    binding.role = role;
    binding.argument_index = static_cast<int64_t>(index);
    binding.shape = shapeOf(arg.getType());
    binding.dtype = dtypeOf(arg.getType());
    binding.layout = layoutOf(arg.getType());
    binding.byte_size = byteSizeOf(arg.getType());
    binding.ownership = role == "model_input" ? "caller" : "model_state";
    binding.is_mutable = role == "model_input";
    binding.external_data_reference = "";
    binding.model_artifact_reference = modelArtifact;
    bindings.push_back(std::move(binding));
  }

  if (!funcOp.getBody().empty()) {
    if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(
            funcOp.getBody().front().getTerminator())) {
      for (auto [index, operand] : llvm::enumerate(ret.getOperands())) {
        TensorBinding binding;
        binding.tensor_id = "result_" + std::to_string(index);
        if (mlir::Operation *producer = operand.getDefiningOp()) {
          binding.original_name = strOp(producer, "source.onnx_name");
          binding.source_value_id = strOp(producer, "source.dispatch_group");
        }
        binding.role = "model_output";
        binding.argument_index = -1;
        binding.shape = shapeOf(operand.getType());
        binding.dtype = dtypeOf(operand.getType());
        binding.layout = layoutOf(operand.getType());
        binding.byte_size = byteSizeOf(operand.getType());
        binding.ownership = "caller";
        binding.is_mutable = true;
        binding.model_artifact_reference = modelArtifact;
        bindings.push_back(std::move(binding));
      }
    }
  }
  return bindings;
}

std::optional<CVPostprocessContract>
ExecutionPlanBuilder::collectPostprocessContract(mlir::func::FuncOp funcOp) {
  if (funcOp.getBody().empty())
    return std::nullopt;
  auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(
      funcOp.getBody().front().getTerminator());
  if (!ret)
    return std::nullopt;

  CVPostprocessContract contract;
  mlir::Value detection;
  for (auto [index, operand] : llvm::enumerate(ret.getOperands())) {
    mlir::Operation *producer = operand.getDefiningOp();
    std::string role =
        producer ? strOp(producer, "cv.output_role") : std::string();
    if (role == "detection") {
      contract.detection_tensor_id = "result_" + std::to_string(index);
      contract.detection_shape = shapeOf(operand.getType());
      detection = operand;
    } else if (role == "segmentation_prototype") {
      contract.prototype_tensor_id = "result_" + std::to_string(index);
      contract.prototype_shape = shapeOf(operand.getType());
    }
  }
  if (contract.detection_tensor_id.empty() ||
      contract.prototype_tensor_id.empty())
    return std::nullopt;

  // Trace the detection tensor's channel composition from the static
  // insert_slice chain that assembles it (channel axis = 1 for [N, C, A]).
  mlir::Value cursor = detection;
  while (auto insert = cursor.getDefiningOp<mlir::tensor::InsertSliceOp>()) {
    auto offsets = insert.getStaticOffsets();
    auto sizes = insert.getStaticSizes();
    if (offsets.size() < 2 || sizes.size() < 2)
      break;
    CVPostprocessContract::ChannelGroup groupInfo;
    groupInfo.channel_start = offsets[1];
    groupInfo.channel_count = sizes[1];
    mlir::Operation *sourceOp = insert.getSource().getDefiningOp();
    if (sourceOp) {
      groupInfo.source_region_id = strOp(sourceOp, "cv.region_id");
      std::string genericOp = strOp(sourceOp, "source.generic_op");
      if (groupInfo.source_region_id.find("mask_coefficient") !=
          std::string::npos) {
        groupInfo.semantic = "mask_coefficients";
        contract.mask_coefficient_channel_start = groupInfo.channel_start;
        contract.mask_coefficient_channel_end =
            groupInfo.channel_start + groupInfo.channel_count;
      } else if (genericOp == "nn.sigmoid") {
        groupInfo.semantic = "class_scores";
      } else if (groupInfo.source_region_id.find("detection_head") !=
                 std::string::npos ||
                 groupInfo.channel_start == 0) {
        groupInfo.semantic = "box_regression";
      } else {
        groupInfo.semantic = "unclassified";
      }
    } else {
      groupInfo.semantic = "unclassified";
    }
    contract.detection_channel_groups.push_back(groupInfo);
    cursor = insert.getDest();
  }
  std::sort(contract.detection_channel_groups.begin(),
            contract.detection_channel_groups.end(),
            [](const CVPostprocessContract::ChannelGroup &a,
               const CVPostprocessContract::ChannelGroup &b) {
              return a.channel_start < b.channel_start;
            });

  // The compiled graph contains no NMS or mask-decode operations (the emitted
  // op set is linalg/tensor/arith only), so both remain runtime obligations.
  // No thresholds or algorithms are invented here.
  contract.nms_required = "true";
  contract.mask_decode_required = true;
  contract.implementation_status = "runtime_required";
  contract.expected_output_semantics =
      "detection tensor holds per-anchor channel vectors (box regression + "
      "class scores + mask coefficients, see detection_channel_groups); "
      "instance masks are decoded from mask coefficients combined with the "
      "prototype tensor after score filtering and NMS, outside the compiled "
      "graph";
  bool maskProven = contract.mask_coefficient_channel_start >= 0;
  contract.confidence = maskProven ? "high" : "medium";
  contract.provenance =
      "traced_static_insert_slice_offsets_and_cv_region_attrs_"
      "no_nms_or_mask_ops_in_compiled_graph";
  return contract;
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

    // Slice 2: compiler-owned memory placement and transfer contract. The
    // Runtime must consume this block exactly; it is not diagnostic-only.
    std::optional<MemoryPlacementPlan> memoryPlacement;
    if (auto st =
            op.getAttrOfType<mlir::StringAttr>("memory_placement.status")) {
      MemoryPlacementPlan mp;
      mp.status = st.getValue().str();
      auto rI64 = [&](llvm::StringRef key) -> int64_t {
        if (auto a = op.getAttrOfType<mlir::IntegerAttr>(
                ("memory_placement." + key).str()))
          return a.getInt();
        return 0;
      };
      mp.compute_unit = strOp(&op, "memory_placement.compute_unit");
      mp.selected_memory_space =
          strOp(&op, "memory_placement.selected_memory_space");
      mp.input_tile_bytes = rI64("input_tile_bytes");
      mp.weight_tile_bytes = rI64("weight_tile_bytes");
      mp.output_tile_bytes = rI64("output_tile_bytes");
      mp.scratch_bytes = rI64("scratch_bytes");
      mp.padding_bytes = rI64("padding_bytes");
      mp.single_buffer_bytes = rI64("single_buffer_bytes");
      mp.additional_double_buffer_bytes =
          rI64("additional_double_buffer_bytes");
      mp.total_required_local_memory_bytes =
          rI64("total_required_local_memory_bytes");
      mp.rejection_reason =
          strOp(&op, "memory_placement.rejection_reason");
      mp.truth_boundary =
          strOp(&op, "memory_placement.truth_boundary");

      if (auto arr = op.getAttrOfType<mlir::ArrayAttr>(
              "memory_placement.buffer_placements")) {
        for (mlir::Attribute elem : arr) {
          auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(elem);
          if (!dict)
            continue;
          BufferPlacement bp;
          bp.buffer_id = stringInDict(dict, "buffer_id");
          bp.role = stringInDict(dict, "role");
          bp.memory_space = stringInDict(dict, "memory_space");
          bp.byte_count = i64InDict(dict, "byte_count");
          bp.alignment = i64InDict(dict, "alignment", 1);
          mp.buffer_placements.push_back(std::move(bp));
        }
      }
      if (auto arr = op.getAttrOfType<mlir::ArrayAttr>(
              "memory_placement.transfer_operations")) {
        for (mlir::Attribute elem : arr) {
          auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(elem);
          if (!dict)
            continue;
          TransferOperation transfer;
          transfer.transfer_id = stringInDict(dict, "transfer_id");
          transfer.source_buffer = stringInDict(dict, "source_buffer");
          transfer.destination_buffer =
              stringInDict(dict, "destination_buffer");
          transfer.source_memory_space =
              stringInDict(dict, "source_memory_space");
          transfer.destination_memory_space =
              stringInDict(dict, "destination_memory_space");
          transfer.byte_count = i64InDict(dict, "byte_count");
          transfer.alignment = i64InDict(dict, "alignment", 1);
          transfer.mode = stringInDict(dict, "mode");
          transfer.dependency_ids =
              stringArrayInDict(dict, "dependency_ids");
          transfer.completion_token =
              stringInDict(dict, "completion_token");
          mp.transfer_operations.push_back(std::move(transfer));
        }
      }
      if (auto arr = op.getAttrOfType<mlir::ArrayAttr>(
              "memory_placement.compute_dependency_ids"))
        for (mlir::Attribute elem : arr)
          if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
            mp.compute_dependency_ids.push_back(s.getValue().str());
      memoryPlacement = std::move(mp);
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

    // Thread-decomposition schedule from KernelSelectionPass (Phase P1D,
    // thread_schedule_contract_v1). Absent when the selected kernel (or no
    // kernel) declares no thread schedules -- existing P1B/P1C plans stay
    // byte-identical (no thread_schedule.* attrs are ever set for them).
    std::optional<ThreadSchedule> threadSchedule;
    if (auto st =
            op.getAttrOfType<mlir::StringAttr>("thread_schedule.status")) {
      ThreadSchedule tsched;
      tsched.status = st.getValue().str();
      if (auto tc = op.getAttrOfType<mlir::IntegerAttr>(
              "thread_schedule.thread_count"))
        tsched.thread_count = tc.getInt();
      tsched.partition_axis     = strOp(&op, "thread_schedule.partition_axis");
      tsched.partition_strategy = strOp(&op, "thread_schedule.partition_strategy");
      tsched.source             = strOp(&op, "thread_schedule.source");
      tsched.policy_id          = strOp(&op, "thread_schedule.policy_id");
      tsched.policy_version     = strOp(&op, "thread_schedule.policy_version");
      tsched.policy_metric      = strOp(&op, "thread_schedule.policy_metric");
      if (auto v = op.getAttrOfType<mlir::IntegerAttr>(
              "thread_schedule.policy_metric_value"))
        tsched.policy_metric_value = v.getInt();
      if (auto v = op.getAttrOfType<mlir::IntegerAttr>(
              "thread_schedule.policy_threshold"))
        tsched.policy_threshold = v.getInt();
      tsched.policy_boundary_rule =
          strOp(&op, "thread_schedule.policy_boundary_rule");
      tsched.policy_selection_reason =
          strOp(&op, "thread_schedule.policy_selection_reason");
      tsched.policy_evidence_ref =
          strOp(&op, "thread_schedule.policy_evidence_ref");
      tsched.policy_evidence_sha256 =
          strOp(&op, "thread_schedule.policy_evidence_sha256");
      tsched.policy_truth_boundary =
          strOp(&op, "thread_schedule.policy_truth_boundary");
      tsched.contract_version   = strOp(&op, "thread_schedule.contract_version");
      tsched.truth_boundary     = strOp(&op, "thread_schedule.truth_boundary");
      if (auto arr = op.getAttrOfType<mlir::ArrayAttr>(
              "thread_schedule.rejection_reasons"))
        for (mlir::Attribute elem : arr)
          if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
            tsched.rejection_reasons.push_back(s.getValue().str());
      threadSchedule = std::move(tsched);
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

    std::optional<AttentionExecutionContract> attentionExecution;
    std::optional<KVCacheExecutionContract> kvCacheExecution;
    if (op.getName().getStringRef() == "hir.cpu_attention") {
      AttentionExecutionContract a;
      a.execution_unit = strOp(&op, "runtime_execution_unit");
      a.backend = "portable_cpu"; a.phase = strOp(&op, "phase");
      a.candidate_id = strOp(&op, "candidate_id");
      a.kernel_id = strOp(&op, "kernel_id"); a.entry_point = strOp(&op, "entry_point");
      a.artifact_ref = strOp(&op, "artifact_ref"); a.artifact_sha256 = strOp(&op, "artifact_sha256");
      a.artifact_version = strOp(&op, "artifact_version"); a.dtype = strOp(&op, "dtype");
      a.input_layout = strOp(&op, "input_layout"); a.output_layout = strOp(&op, "output_layout");
      a.required_isa = strOp(&op, "required_isa"); a.fallback_identity = strOp(&op, "fallback_identity");
      a.implementation_strategy = strOp(&op, "attention_implementation_strategy");
      a.truth_boundary = strOp(&op, "truth_boundary");
      a.batch = op.getAttrOfType<IntegerAttr>("batch").getInt();
      a.query_length = op.getAttrOfType<IntegerAttr>("query_length").getInt();
      a.context_length = op.getAttrOfType<IntegerAttr>("context_length").getInt();
      a.num_query_heads = op.getAttrOfType<IntegerAttr>("num_query_heads").getInt();
      a.num_kv_heads = op.getAttrOfType<IntegerAttr>("num_kv_heads").getInt();
      a.head_dim = op.getAttrOfType<IntegerAttr>("head_dim").getInt();
      a.workspace_bytes = op.getAttrOfType<IntegerAttr>("workspace_bytes").getInt();
      a.alignment_bytes = op.getAttrOfType<IntegerAttr>("alignment_bytes").getInt();
      a.causal = op.getAttrOfType<BoolAttr>("causal").getValue();
      a.runtime_no_redecision = op.getAttrOfType<BoolAttr>("runtime_no_redecision").getValue();
      attentionExecution = std::move(a);
      KVCacheExecutionContract kv;
      if (op.hasAttr("kv_layout_kind")) {
        kv.pool_create_entry_point=strOp(&op,"pool_create_entry_point");kv.paged_prefill_write_entry_point=strOp(&op,"prefill_write_entry_point");kv.paged_append_entry_point=strOp(&op,"append_entry_point");kv.paged_view_binding=strOp(&op,"view_binding");kv.paged_reset_entry_point=strOp(&op,"reset_entry_point");kv.paged_release_entry_point=strOp(&op,"release_entry_point");
        kv.execution_unit="portable_cpu_paged_kv";kv.candidate_id=strOp(&op,"kv_candidate_id");kv.layout_kind=strOp(&op,"kv_layout_kind");kv.dtype=strOp(&op,"kv_dtype");kv.artifact_ref=strOp(&op,"pool_artifact_ref");kv.artifact_sha256=strOp(&op,"pool_artifact_sha256");kv.artifact_version=strOp(&op,"pool_artifact_version");kv.block_table_element_type=strOp(&op,"block_table_element_type");kv.paged_attention_kernel_id=strOp(&op,"paged_attention_kernel_id");kv.paged_attention_entry_point=strOp(&op,"paged_attention_entry_point");kv.implementation_strategy=strOp(&op,"implementation_strategy");kv.measurement_provenance=strOp(&op,"measurement_provenance");kv.contiguous_fallback_identity=strOp(&op,"contiguous_fallback_identity");kv.truth_boundary=strOp(&op,"truth_boundary");kv.operation_order=strOp(&op,"paged_operation_order");auto i=[&](StringRef n){return op.getAttrOfType<IntegerAttr>(n).getInt();};kv.batch=i("batch");kv.num_kv_heads=i("num_kv_heads");kv.head_dim=i("head_dim");kv.page_tokens=i("page_tokens");kv.num_physical_pages=i("num_physical_pages");kv.maximum_logical_tokens=i("maximum_logical_tokens");kv.maximum_logical_blocks=i("maximum_logical_blocks");kv.block_table_length=i("block_table_length");kv.invalid_page_sentinel=i("invalid_page_sentinel");kv.bytes_per_token=i("bytes_per_token");kv.bytes_per_k_page=i("bytes_per_k_page");kv.bytes_per_v_page=i("bytes_per_v_page");kv.bytes_per_combined_page=i("bytes_per_combined_page");kv.total_pool_bytes=i("total_pool_bytes");kv.alignment_bytes=i("alignment_bytes");for(int64_t x:op.getAttrOfType<DenseI64ArrayAttr>("k_page_strides").asArrayRef())kv.k_page_strides.push_back(x);for(int64_t x:op.getAttrOfType<DenseI64ArrayAttr>("v_page_strides").asArrayRef())kv.v_page_strides.push_back(x);kv.runtime_no_layout_redecision=op.getAttrOfType<BoolAttr>("runtime_no_layout_redecision").getValue();kv.runtime_no_kernel_redecision=op.getAttrOfType<BoolAttr>("runtime_no_kernel_redecision").getValue();kvCacheExecution=std::move(kv);
      } else {
      kv.execution_unit = "portable_cpu_contiguous_kv";
      kv.candidate_id = strOp(&op, "kv_candidate_id");
      kv.cache_id = strOp(&op, "kv_cache_id");
      kv.artifact_ref = strOp(&op, "kv_artifact_ref");
      kv.artifact_sha256 = strOp(&op, "kv_artifact_sha256");
      kv.artifact_version = strOp(&op, "kv_artifact_version");
      kv.dtype = strOp(&op, "kv_dtype"); kv.layout = strOp(&op, "kv_layout");
      kv.create_entry_point = strOp(&op, "kv_create_entry_point");
      kv.prefill_write_entry_point = strOp(&op, "kv_prefill_write_entry_point");
      kv.decode_append_entry_point = strOp(&op, "kv_decode_append_entry_point");
      kv.view_binding = strOp(&op, "kv_view_binding");
      kv.reset_entry_point = strOp(&op, "kv_reset_entry_point");
      kv.compatible_prefill_kernel_id = strOp(&op, "compatible_prefill_kernel_id");
      kv.compatible_decode_kernel_id = strOp(&op, "compatible_decode_kernel_id");
      kv.attention_entry_point = strOp(&op, "attention_entry_point");
      kv.implementation_strategy = strOp(&op, "implementation_strategy");
      kv.measurement_provenance = strOp(&op, "measurement_provenance");
      kv.fallback_identity = strOp(&op, "fallback_identity");
      kv.operation_order = strOp(&op, "kv_operation_order");
      kv.truth_boundary = strOp(&op, "truth_boundary");
      auto i = [&](StringRef n) { return op.getAttrOfType<IntegerAttr>(n).getInt(); };
      kv.batch=i("batch"); kv.num_kv_heads=i("num_kv_heads"); kv.head_dim=i("head_dim");
      kv.capacity_tokens=i("capacity_tokens"); kv.initial_valid_tokens=i("initial_valid_tokens");
      kv.bytes_per_token=i("bytes_per_token"); kv.k_cache_bytes=i("k_cache_bytes");
      kv.v_cache_bytes=i("v_cache_bytes"); kv.total_cache_bytes=i("total_cache_bytes");
      kv.alignment_bytes=i("alignment_bytes");
      for (int64_t x : op.getAttrOfType<DenseI64ArrayAttr>("k_strides").asArrayRef()) kv.k_strides.push_back(x);
      for (int64_t x : op.getAttrOfType<DenseI64ArrayAttr>("v_strides").asArrayRef()) kv.v_strides.push_back(x);
      kv.runtime_no_layout_redecision = op.getAttrOfType<BoolAttr>("runtime_no_layout_redecision").getValue();
      kvCacheExecution = std::move(kv);
      }
    }

    if (quant || kernel || fallback || !materialized.empty() ||
        !deferred.empty() || shapeCost || tilePlan || layoutCreatesBundle ||
        kernelSelection || quantCoDesign || threadSchedule ||
        memoryPlacement || attentionExecution || kvCacheExecution) {
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
      bundle.thread_schedule = std::move(threadSchedule);
      bundle.memory_placement = std::move(memoryPlacement);
      bundle.attention_execution = std::move(attentionExecution);
      bundle.kv_cache_execution = std::move(kvCacheExecution);
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
  d.output_dtype       = strOp(op, "quant.output_dtype");
  d.granularity        = strOp(op, "quant.granularity");
  d.accuracy_risk      = strOp(op, "quant.accuracy_risk");
  d.selected_candidate_id = strOp(op, "quant.selected_candidate_id");
  d.scheme                = strOp(op, "quant.scheme");
  d.backend               = strOp(op, "quant.backend");
  d.kernel_id             = strOp(op, "quant.kernel_id");
  d.required_backend_capability =
      strOp(op, "quant.required_backend_capability");
  d.required_kernel_capability =
      strOp(op, "quant.required_kernel_capability");
  d.calibration_artifact_ref = strOp(op, "quant.calibration_artifact_ref");
  d.calibration_artifact_id = strOp(op, "quant.calibration_artifact_id");
  d.calibration_artifact_sha256 = strOp(op, "quant.calibration_artifact_sha256");
  d.packed_weight_artifact_ref = strOp(op, "quant.packed_weight_artifact_ref");
  d.packed_weight_artifact_id = strOp(op, "quant.packed_weight_artifact_id");
  d.packed_weight_sha256 = strOp(op, "quant.packed_weight_sha256");
  d.source_weight_sha256 = strOp(op, "quant.source_weight_sha256");
  d.packed_layout = strOp(op, "quant.packed_layout");
  d.packing_scheme = strOp(op, "quant.packing_scheme");
  if (auto a = op->getAttrOfType<mlir::BoolAttr>("quant.kernel_requires_packed_weight"))
    d.kernel_requires_packed_weight = a.getValue();
  d.selected_complete_candidate_id =
      strOp(op, "quant.selected_complete_candidate_id");
  d.codegen_target_id = strOp(op, "quant.codegen_target_id");
  d.target_architecture = strOp(op, "quant.target_architecture");
  d.target_microarchitecture = strOp(op, "quant.target_microarchitecture");
  if (auto arr = op->getAttrOfType<mlir::ArrayAttr>("quant.required_isa_features"))
    d.required_isa_features = stringArrayAttr(arr);
  if (auto arr = op->getAttrOfType<mlir::ArrayAttr>("quant.compiler_flags"))
    d.compiler_flags = stringArrayAttr(arr);
  d.binary_sha256 = strOp(op, "quant.binary_sha256");
  d.measurement_artifact_ref = strOp(op, "quant.measurement_artifact_ref");
  d.build_manifest_ref = strOp(op, "quant.build_manifest_ref");
  d.workload_id = strOp(op, "quant.workload_id");
  d.activation_granularity = strOp(op, "quant.activation_granularity");
  d.weight_granularity = strOp(op, "quant.weight_granularity");
  if (auto a = op->getAttrOfType<mlir::FloatAttr>("quant.activation_scale"))
    d.activation_scale = a.getValueAsDouble();
  if (auto a = op->getAttrOfType<mlir::FloatAttr>("quant.weight_scale"))
    d.weight_scale = a.getValueAsDouble();
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("quant.activation_zero_point"))
    d.activation_zero_point = a.getInt();
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("quant.weight_zero_point"))
    d.weight_zero_point = a.getInt();
  d.policy_id = strOp(op, "quant.policy_id");
  d.selection_reason = strOp(op, "quant.selection_reason");
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("quant.group_size"))
    d.group_size = a.getInt();
  if (auto a = op->getAttrOfType<mlir::BoolAttr>("quant.requires_calibration"))
    d.requires_calibration = a.getValue();
  if (auto a = op->getAttrOfType<mlir::BoolAttr>("quant.calibration_available"))
    d.calibration_available = a.getValue();
  if (auto arr = op->getAttrOfType<mlir::ArrayAttr>("quant.considered_candidate_ids"))
    for (mlir::Attribute elem : arr)
      if (auto st = mlir::dyn_cast<mlir::StringAttr>(elem))
        d.considered_candidate_ids.push_back(st.getValue().str());
  if (auto arr = op->getAttrOfType<mlir::ArrayAttr>("quant.rejected_candidate_ids"))
    for (mlir::Attribute elem : arr)
      if (auto st = mlir::dyn_cast<mlir::StringAttr>(elem))
        d.rejected_candidate_ids.push_back(st.getValue().str());
  if (auto arr = op->getAttrOfType<mlir::ArrayAttr>("quant.rejected_candidate_reasons"))
    for (mlir::Attribute elem : arr)
      if (auto st = mlir::dyn_cast<mlir::StringAttr>(elem))
        d.rejected_candidate_reasons.push_back(st.getValue().str());
  if (auto arr = op->getAttrOfType<mlir::ArrayAttr>("quant.execution_stages")) {
    for (mlir::Attribute elem : arr) {
      auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(elem);
      if (!dict)
        continue;
      QuantizedExecutionStage stage;
      stage.stage_id = dictStr(dict, "stage_id");
      stage.op = dictStr(dict, "op");
      if (auto deps =
              mlir::dyn_cast_if_present<mlir::ArrayAttr>(dict.get("dependency_ids")))
        stage.dependency_ids = stringArrayAttr(deps);
      stage.produces = dictStr(dict, "produces");
      stage.kernel_id = dictStr(dict, "kernel_id");
      stage.artifact_ref = dictStr(dict, "artifact_ref");
      stage.artifact_sha256 = dictStr(dict, "artifact_sha256");
      stage.packed_layout = dictStr(dict, "packed_layout");
      stage.fused_postprocess = dictStr(dict, "fused_postprocess");
      stage.scale = dictFloat(dict, "scale");
      stage.zero_point = dictInt(dict, "zero_point");
      stage.clamp_min = dictInt(dict, "clamp_min");
      stage.clamp_max = dictInt(dict, "clamp_max");
      stage.rounding_mode = dictStr(dict, "rounding_mode");
      stage.source_dtype = dictStr(dict, "source_dtype");
      stage.destination_dtype = dictStr(dict, "destination_dtype");
      stage.binary_sha256 = dictStr(dict, "binary_sha256");
      if (!stage.stage_id.empty())
        d.execution_stages.push_back(std::move(stage));
    }
  }

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
