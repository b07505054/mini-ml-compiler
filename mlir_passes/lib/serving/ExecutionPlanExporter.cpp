#include "serving/ExecutionPlanExporter.h"
#include "serving/ServingEnums.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::hir {
namespace {

// ---------------------------------------------------------------------------
// Enum → string helpers
// ---------------------------------------------------------------------------

static llvm::StringRef servingPhaseStr(ServingPhase p) {
  switch (p) {
  case ServingPhase::Prefill: return "prefill";
  case ServingPhase::Decode:  return "decode";
  default:                    return "unknown";
  }
}

static llvm::StringRef scopeStr(DecisionScope s) {
  switch (s) {
  case DecisionScope::Global:   return "Global";
  case DecisionScope::Function: return "Function";
  case DecisionScope::PerOp:    return "PerOp";
  default:                      return "Unknown";
  }
}

// ---------------------------------------------------------------------------
// Write JSON value to file (creates parent dirs as needed)
// ---------------------------------------------------------------------------

static llvm::Error writeJSON(const llvm::json::Value &val,
                              llvm::StringRef path) {
  llvm::SmallString<256> parent(path);
  llvm::sys::path::remove_filename(parent);
  if (!parent.empty()) {
    if (std::error_code ec =
            llvm::sys::fs::create_directories(parent, /*IgnoreExisting=*/true))
      return llvm::make_error<llvm::StringError>(
          "failed to create output directory: " + ec.message(),
          llvm::inconvertibleErrorCode());
  }
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return llvm::make_error<llvm::StringError>(
        "cannot open output file '" + path.str() + "': " + ec.message(),
        llvm::inconvertibleErrorCode());
  os << llvm::formatv("{0:2}", val);
  os << "\n";
  return llvm::Error::success();
}

// ---------------------------------------------------------------------------
// DecisionMetadata serializer
// Emits the shared header fields present on every Decision type.
// ---------------------------------------------------------------------------

static llvm::json::Object serializeMeta(const DecisionMetadata &meta) {
  llvm::json::Object obj;
  obj["decision_id"]    = meta.decision_id;
  obj["decision_type"]  = meta.decision_type;
  obj["scope"]          = scopeStr(meta.scope);
  obj["source_pass"]    = meta.source_pass;
  obj["truth_boundary"] = meta.truth_boundary;
  if (!meta.reason.empty())
    obj["reason"] = meta.reason;

  // evidence.cost — present when ServingCostModelPass and PlanSelectionPass ran.
  if (meta.evidence.cost) {
    const auto &c = *meta.evidence.cost;
    llvm::json::Object cost;
    cost["compute_cost"]          = c.compute_cost;
    cost["memory_cost"]           = c.memory_cost;
    cost["dequant_cost"]          = c.dequant_cost;
    cost["requant_cost"]          = c.requant_cost;
    cost["layout_transform_cost"] = c.layout_transform_cost;
    cost["cast_cost"]             = c.cast_cost;
    cost["backend_switch_cost"]   = c.backend_switch_cost;
    cost["launch_overhead_cost"]  = c.launch_overhead_cost;
    cost["kv_cache_cost"]         = c.kv_cache_cost;
    cost["transfer_cost"]         = c.transfer_cost;
    cost["unsupported_penalty"]   = c.unsupported_penalty;
    cost["total_cost"]            = c.total_cost;
    cost["cost_model_id"]         = c.cost_model_id;
    cost["truth_boundary"]        = c.truth_boundary;
    llvm::json::Object evidence;
    evidence["cost"] = std::move(cost);
    obj["evidence"] = std::move(evidence);
  }

  return obj;
}

// ---------------------------------------------------------------------------
// Per-Decision-type serializers
// ---------------------------------------------------------------------------

static llvm::json::Object serializeServingDecision(const ServingDecision &d) {
  auto obj = serializeMeta(d.meta);
  obj["topology"]                   = d.topology;
  obj["colocated_cost_estimate_ms"] = d.colocated_cost_estimate_ms;
  obj["pd_split_cost_estimate_ms"]  = d.pd_split_cost_estimate_ms;
  obj["replay_eligible"]            = d.replay_eligible;
  obj["token_budget_per_step"]      = d.token_budget_per_step;
  obj["prefix_reuse_eligible"]      = d.prefix_reuse_eligible;
  obj["chunked_prefill_eligible"]   = d.chunked_prefill_eligible;
  obj["parallelism_degree"]         = d.parallelism_degree;
  if (!d.parallelism_kind.empty())
    obj["parallelism_kind"] = d.parallelism_kind;
  if (!d.attention_algorithm.empty())
    obj["attention_algorithm"] = d.attention_algorithm;
  if (!d.speculative_decoding.empty())
    obj["speculative_decoding"] = d.speculative_decoding;
  return obj;
}

static llvm::json::Object serializeMemoryDecision(const MemoryDecision &d) {
  auto obj = serializeMeta(d.meta);
  obj["kv_cache_layout"]        = d.kv_cache_layout;
  obj["kv_block_size_tokens"]   = d.kv_block_size_tokens;
  obj["estimated_kv_peak_mb"]   = d.estimated_kv_peak_mb;
  obj["memory_budget_fraction"] = d.memory_budget_fraction;
  return obj;
}

static llvm::json::Object serializeQuantizationDecision(const QuantizationDecision &d) {
  auto obj = serializeMeta(d.meta);
  obj["strategy"]           = d.strategy;
  obj["weight_dtype"]       = d.weight_dtype;
  obj["activation_dtype"]   = d.activation_dtype;
  obj["accumulation_dtype"] = d.accumulation_dtype;
  obj["granularity"]        = d.granularity;
  obj["accuracy_risk"]      = d.accuracy_risk;
  obj["algorithm"]          = d.algorithm;
  if (!d.op_type.empty())
    obj["op_type"] = d.op_type;
  if (!d.quantized_model_artifact_ref.empty())
    obj["quantized_model_artifact_ref"] = d.quantized_model_artifact_ref;
  return obj;
}

static llvm::json::Object serializeBackendDecision(const BackendDecision &d) {
  auto obj = serializeMeta(d.meta);
  obj["selected_backend"] = d.selected_backend;
  llvm::json::Array fallback;
  for (const auto &b : d.fallback_backends)
    fallback.push_back(b);
  obj["fallback_backends"] = std::move(fallback);
  return obj;
}

static llvm::json::Object serializeKernelDecision(const KernelDecision &d) {
  auto obj = serializeMeta(d.meta);
  obj["op_name"]         = d.op_name;
  obj["op_type"]         = d.op_type;
  obj["selected_kernel"] = d.selected_kernel;
  obj["kernel_library"]  = d.kernel_library;
  obj["lowering_path"]   = d.lowering_path;
  obj["kernel_exists"]   = d.kernel_exists;
  llvm::json::Array boundary;
  for (const auto &b : d.required_boundary_ops)
    boundary.push_back(b);
  obj["required_boundary_ops"] = std::move(boundary);
  return obj;
}

static llvm::json::Object serializeLayoutDecision(const LayoutDecision &d) {
  auto obj = serializeMeta(d.meta);
  obj["op_type"]                   = d.op_type;
  obj["selected_layout"]           = d.selected_layout;
  obj["requires_layout_transform"] = d.requires_layout_transform;
  return obj;
}

static llvm::json::Object serializeFallbackDecision(const FallbackDecision &d) {
  auto obj = serializeMeta(d.meta);
  obj["op_name"]          = d.op_name;
  obj["op_type"]          = d.op_type;
  obj["fallback_kind"]    = d.fallback_kind;
  obj["fallback_backend"] = d.fallback_backend;
  llvm::json::Array tried;
  for (const auto &p : d.tried_paths)
    tried.push_back(p);
  obj["tried_paths"] = std::move(tried);
  return obj;
}

static llvm::json::Object serializePerOpBundle(const PerOpDecisionBundle &b) {
  llvm::json::Object obj;
  obj["op_name"] = b.op_name;
  obj["op_type"] = b.op_type;
  if (b.quantization)
    obj["quantization"] = serializeQuantizationDecision(*b.quantization);
  if (b.layout)
    obj["layout"] = serializeLayoutDecision(*b.layout);
  if (b.kernel)
    obj["kernel"] = serializeKernelDecision(*b.kernel);
  if (b.fallback)
    obj["fallback"] = serializeFallbackDecision(*b.fallback);
  return obj;
}

static llvm::json::Object serializeFunctionPlan(const FunctionPlan &fp) {
  llvm::json::Object obj;
  obj["function_name"] = fp.function_name;
  obj["serving_phase"] = servingPhaseStr(fp.serving_phase);
  obj["backend"]       = serializeBackendDecision(fp.backend);

  llvm::json::Array per_op;
  for (const auto &bundle : fp.per_op_decisions)
    per_op.push_back(serializePerOpBundle(bundle));
  obj["per_op_decisions"] = std::move(per_op);

  return obj;
}

} // namespace

// ---------------------------------------------------------------------------
// exportToFile
// ---------------------------------------------------------------------------

llvm::Error ExecutionPlanExporter::exportToFile(const ExecutionPlan &plan,
                                                   llvm::StringRef outPath) {
  // capability_bundle
  llvm::json::Array backendRefs;
  for (const auto &r : plan.provenance.capability_bundle.backend_profile_refs)
    backendRefs.push_back(r);
  llvm::json::Array kernelRefs;
  for (const auto &r : plan.provenance.capability_bundle.kernel_profile_refs)
    kernelRefs.push_back(r);

  llvm::json::Object capBundle;
  capBundle["hardware_profile_ref"] =
      plan.provenance.capability_bundle.hardware_profile_ref;
  capBundle["backend_profile_refs"] = std::move(backendRefs);
  capBundle["kernel_profile_refs"]  = std::move(kernelRefs);
  if (!plan.provenance.capability_bundle.workload_ref.empty())
    capBundle["workload_ref"] = plan.provenance.capability_bundle.workload_ref;
  if (!plan.provenance.capability_bundle.deployment_profile_ref.empty())
    capBundle["deployment_profile_ref"] =
        plan.provenance.capability_bundle.deployment_profile_ref;

  // provenance
  llvm::json::Object provenance;
  provenance["compiler_tool"]     = plan.provenance.compiler_tool;
  provenance["model_spec_ref"]    = plan.provenance.model_spec_ref;
  provenance["capability_bundle"] = std::move(capBundle);
  provenance["truth_boundary"]    = plan.provenance.truth_boundary;

  // model_identity
  llvm::json::Object modelId;
  modelId["model_id"]            = plan.model_identity.model_id;
  modelId["model_family"]        = plan.model_identity.model_family;
  modelId["num_layers"]          = plan.model_identity.num_layers;
  modelId["hidden_size"]         = plan.model_identity.hidden_size;
  modelId["num_attention_heads"] = plan.model_identity.num_attention_heads;
  modelId["num_kv_heads"]        = plan.model_identity.num_kv_heads;
  modelId["truth_boundary"]      = plan.model_identity.truth_boundary;
  if (!plan.model_identity.attention_mechanism.empty())
    modelId["attention_mechanism"] = plan.model_identity.attention_mechanism;
  if (!plan.model_identity.positional_encoding.empty())
    modelId["positional_encoding"] = plan.model_identity.positional_encoding;

  // global_decisions — emit only the optional fields that are present.
  llvm::json::Object globalDecisions;
  if (plan.global_decisions.serving)
    globalDecisions["serving"] =
        serializeServingDecision(*plan.global_decisions.serving);
  if (plan.global_decisions.memory)
    globalDecisions["memory"] =
        serializeMemoryDecision(*plan.global_decisions.memory);
  if (plan.global_decisions.quantization)
    globalDecisions["quantization"] =
        serializeQuantizationDecision(*plan.global_decisions.quantization);
  if (plan.global_decisions.calibration) {
    const auto &cal = *plan.global_decisions.calibration;
    auto calObj = serializeMeta(cal.meta);
    calObj["calibration_kind"]          = cal.calibration_kind;
    calObj["calibration_dataset_hint"]  = cal.calibration_dataset_hint;
    calObj["num_calibration_samples"]   = cal.num_calibration_samples;
    calObj["weight_group_size"]         = cal.weight_group_size;
    calObj["zero_point_required"]       = cal.zero_point_required;
    llvm::json::Array targetPats;
    for (const auto &p : cal.target_layer_patterns) targetPats.push_back(p);
    llvm::json::Array skipPats;
    for (const auto &p : cal.skip_layer_patterns) skipPats.push_back(p);
    calObj["target_layer_patterns"] = std::move(targetPats);
    calObj["skip_layer_patterns"]   = std::move(skipPats);
    globalDecisions["calibration"]  = std::move(calObj);
  }

  // function_plans
  llvm::json::Array functionPlans;
  for (const auto &fp : plan.function_plans)
    functionPlans.push_back(serializeFunctionPlan(fp));

  // root
  llvm::json::Object root;
  root["schema"]           = plan.schema;
  root["schema_version"]   = plan.schema_version;
  root["plan_id"]          = plan.plan_id;
  root["provenance"]       = std::move(provenance);
  root["model_identity"]   = std::move(modelId);
  root["global_decisions"] = std::move(globalDecisions);
  root["function_plans"]   = std::move(functionPlans);

  return writeJSON(llvm::json::Value(std::move(root)), outPath);
}

} // namespace mlir::hir
