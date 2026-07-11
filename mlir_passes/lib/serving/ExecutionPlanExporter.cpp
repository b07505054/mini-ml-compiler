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
  if (!d.required_input_layout.empty())
    obj["required_input_layout"]   = d.required_input_layout;
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
  // Boundary materialization record: emitted only when
  // BoundaryMaterializationPass recorded it, so plans from planning-only
  // pipelines are unchanged. materialized = inserted into IR;
  // deferred = planned but not yet materializable without inventing
  // metadata the planner does not produce.
  if (!b.materialized_boundary_ops.empty()) {
    llvm::json::Array materialized;
    for (const auto &m : b.materialized_boundary_ops)
      materialized.push_back(m);
    obj["materialized_boundary_ops"] = std::move(materialized);
  }
  if (!b.deferred_boundary_ops.empty()) {
    llvm::json::Array deferred;
    for (const auto &m : b.deferred_boundary_ops)
      deferred.push_back(m);
    obj["deferred_boundary_ops"] = std::move(deferred);
  }
  // Shape-derived static cost estimate (shape_cost_model_v2). A static
  // compiler estimate from tensor shapes and declared profile numbers —
  // never a measured benchmark. Time fields absent when the profile
  // declared no peak numbers.
  if (b.shape_cost) {
    const ShapeCostEstimate &sc = *b.shape_cost;
    llvm::json::Object scObj;
    scObj["flops_estimate"]              = sc.flops_estimate;
    scObj["input_bytes_estimate"]        = sc.input_bytes_estimate;
    scObj["output_bytes_estimate"]       = sc.output_bytes_estimate;
    scObj["weight_bytes_estimate"]       = sc.weight_bytes_estimate;
    scObj["total_memory_bytes_estimate"] = sc.total_memory_bytes_estimate;
    scObj["arithmetic_intensity_milli"]  = sc.arithmetic_intensity_milli;
    if (sc.estimated_compute_cost_nanos)
      scObj["estimated_compute_cost_nanos"] = *sc.estimated_compute_cost_nanos;
    if (sc.estimated_memory_cost_nanos)
      scObj["estimated_memory_cost_nanos"] = *sc.estimated_memory_cost_nanos;
    if (sc.estimated_boundary_cost_nanos)
      scObj["estimated_boundary_cost_nanos"] =
          *sc.estimated_boundary_cost_nanos;
    if (sc.estimated_total_cost_nanos)
      scObj["estimated_total_cost_nanos"] = *sc.estimated_total_cost_nanos;
    scObj["status"]             = sc.status;
    scObj["cost_model_version"] = sc.cost_model_version;
    scObj["cost_truth_boundary"] = sc.truth_boundary;
    obj["shape_cost"] = std::move(scObj);
  }
  // Static local-memory tile plan (tile_planning_v1): feasibility against
  // the declared capacity plus a reuse-limited traffic estimate. Not
  // measured performance, not DMA execution, not codegen.
  if (b.tile_plan) {
    const TilePlan &tp = *b.tile_plan;
    llvm::json::Object tpObj;
    tpObj["status"] = tp.status;
    if (tp.status == "planned") {
      llvm::json::Array shape;
      shape.push_back(tp.tile_m);
      shape.push_back(tp.tile_n);
      shape.push_back(tp.tile_k);
      tpObj["tile_shape_mnk"]      = std::move(shape);
      tpObj["local_memory_bytes"]  = tp.local_memory_bytes;
      tpObj["estimated_global_traffic_bytes"] =
          tp.estimated_global_traffic_bytes;
      tpObj["double_buffer_fits"]  = tp.double_buffer_fits;
      tpObj["staging_capability"]  = tp.staging_capability;
    }
    tpObj["rejected_tile_count"] = tp.rejected_tile_count;
    if (!tp.rejection_reason.empty())
      tpObj["rejection_reason"] = tp.rejection_reason;
    if (!tp.deferred_reason.empty())
      tpObj["deferred_reason"] = tp.deferred_reason;
    tpObj["truth_boundary"] = tp.truth_boundary;
    obj["tile_plan"] = std::move(tpObj);
  }
  // Concrete runtime-kernel contract selection
  // (kernel_selection_contract_v1). A selected kernel is a contract handed
  // to the runtime — not a claim of runtime execution or measured
  // performance. Rejections and deferrals carry their explicit reasons.
  if (b.kernel_selection) {
    const KernelSelection &ks = *b.kernel_selection;
    llvm::json::Object ksObj;
    ksObj["status"] = ks.status;
    if (!ks.selected_kernel_id.empty()) {
      ksObj["selected_kernel"] = ks.selected_kernel_id;
      ksObj["source"]          = ks.source;
    }
    if (!ks.rejection_reasons.empty()) {
      llvm::json::Array reasons;
      for (const auto &r : ks.rejection_reasons)
        reasons.push_back(r);
      ksObj["rejection_reasons"] = std::move(reasons);
    }
    ksObj["contract_version"] = ks.contract_version;
    ksObj["truth_boundary"]   = ks.truth_boundary;
    obj["kernel_selection"] = std::move(ksObj);
  }
  // Quantization co-design evidence (quantization_codesign_contract_v1).
  // Named "quantization_codesign" to stay distinct from the existing
  // "quantization" object (QuantizationStrategyPlanningPass output).
  // Unknown fields are omitted, never defaulted. Static planning evidence
  // only — no calibration, no measured accuracy, no quantized execution.
  if (b.quantization_codesign) {
    const QuantizationCoDesign &qc = *b.quantization_codesign;
    llvm::json::Object qcObj;
    auto addIf = [&](llvm::StringRef key, const std::string &v) {
      if (!v.empty()) qcObj[key] = v;
    };
    qcObj["status"] = qc.status;
    qcObj["policy"] = qc.policy;
    addIf("representation",    qc.representation);
    addIf("weight_dtype",      qc.weight_dtype);
    addIf("activation_dtype",  qc.activation_dtype);
    addIf("accumulator_dtype", qc.accumulator_dtype);
    addIf("algorithm_status",  qc.algorithm_status);
    addIf("algorithm_name",    qc.algorithm_name);
    addIf("backend_legality",  qc.backend_legality);
    if (!qc.kernel_support_status.empty()) {
      llvm::json::Object ksup;
      ksup["status"] = qc.kernel_support_status;
      if (!qc.kernel_support_kernel_id.empty()) {
        ksup["kernel_id"] = qc.kernel_support_kernel_id;
        ksup["source"]    = qc.kernel_support_source;
      }
      qcObj["kernel_support"] = std::move(ksup);
    }
    if (!qc.accuracy_evidence_status.empty()) {
      llvm::json::Object acc;
      acc["status"] = qc.accuracy_evidence_status;
      if (!qc.accuracy_evidence_artifact_ref.empty())
        acc["artifact_ref"] = qc.accuracy_evidence_artifact_ref;
      qcObj["accuracy_evidence"] = std::move(acc);
    }
    addIf("scale_source",      qc.scale_source);
    addIf("zero_point_source", qc.zero_point_source);
    if (qc.weight_bytes_before || qc.total_cost_before_nanos) {
      llvm::json::Object est;
      auto addOpt = [&](llvm::StringRef key,
                        const std::optional<int64_t> &v) {
        if (v) est[key] = *v;
      };
      addOpt("weight_bytes_before",     qc.weight_bytes_before);
      addOpt("weight_bytes_after",      qc.weight_bytes_after);
      addOpt("boundary_bytes",          qc.boundary_bytes);
      addOpt("total_cost_before_nanos", qc.total_cost_before_nanos);
      addOpt("total_cost_after_nanos",  qc.total_cost_after_nanos);
      addOpt("systems_benefit_nanos",   qc.systems_benefit_nanos);
      if (!qc.excluded_cost_terms.empty()) {
        llvm::json::Array excluded;
        for (const auto &t : qc.excluded_cost_terms) excluded.push_back(t);
        est["excluded_terms"] = std::move(excluded);
      }
      qcObj["systems_cost_estimate"] = std::move(est);
    }
    qcObj["materialization_required"] = qc.materialization_required;
    addIf("materialization_status", qc.materialization_status);
    if (!qc.rejection_reasons.empty()) {
      llvm::json::Array reasons;
      for (const auto &r : qc.rejection_reasons) reasons.push_back(r);
      qcObj["rejection_reasons"] = std::move(reasons);
    }
    qcObj["truth_boundary"]   = qc.truth_boundary;
    qcObj["contract_version"] = qc.contract_version;
    obj["quantization_codesign"] = std::move(qcObj);
  }
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

static llvm::json::Array serializeShape(const std::vector<int64_t> &shape) {
  llvm::json::Array arr;
  for (int64_t dim : shape)
    arr.push_back(dim);
  return arr;
}

static llvm::json::Object serializeTensorContract(const TensorContract &t) {
  llvm::json::Object obj;
  obj["tensor_id"] = t.tensor_id;
  obj["shape"] = serializeShape(t.shape);
  obj["dtype"] = t.dtype;
  obj["layout"] = t.layout;
  obj["role"] = t.role;
  return obj;
}

static llvm::json::Object serializeCVExtension(const CVPlanExtension &cv) {
  llvm::json::Object obj;
  obj["model_family"] = cv.model_family;
  obj["function_name"] = cv.function_name;
  obj["target_profile_id"] = cv.target_profile_id;

  llvm::json::Array inputs;
  for (const auto &input : cv.inputs)
    inputs.push_back(serializeTensorContract(input));
  obj["inputs"] = std::move(inputs);

  llvm::json::Array outputs;
  for (const auto &output : cv.outputs)
    outputs.push_back(serializeTensorContract(output));
  obj["outputs"] = std::move(outputs);

  llvm::json::Array regions;
  for (const auto &region : cv.semantic_regions) {
    llvm::json::Object r;
    r["region_id"] = region.region_id;
    r["semantic_role"] = region.semantic_role;
    r["recognition_confidence"] = region.recognition_confidence;
    r["operation_count"] = region.operation_count;
    llvm::json::Array scales;
    for (const auto &scale : region.feature_scales)
      scales.push_back(scale);
    r["feature_scales"] = std::move(scales);
    regions.push_back(std::move(r));
  }
  obj["semantic_regions"] = std::move(regions);

  llvm::json::Object memory;
  memory["estimated_input_bytes"] = cv.estimated_input_bytes;
  memory["estimated_output_bytes"] = cv.estimated_output_bytes;
  memory["estimated_temporary_bytes"] = cv.estimated_temporary_bytes;
  memory["estimated_total_tensor_bytes"] = cv.estimated_total_tensor_bytes;
  memory["scope"] = "static_tensor_byte_estimates_no_slot_allocation";
  obj["memory_estimates"] = std::move(memory);

  obj["postprocess_boundary"] = cv.postprocess_boundary;
  obj["truth_boundary"] = cv.truth_boundary;
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
  if (plan.cv_extension)
    root["cv_extension"] = serializeCVExtension(*plan.cv_extension);

  return writeJSON(llvm::json::Value(std::move(root)), outPath);
}

} // namespace mlir::hir
