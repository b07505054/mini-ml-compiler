#include "serving/ExecutionPlanExporter.h"
#include "serving/ServingEnums.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

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
  if (!d.output_dtype.empty())
    obj["output_dtype"] = d.output_dtype;
  obj["granularity"]        = d.granularity;
  if (d.group_size > 0)
    obj["group_size"] = d.group_size;
  obj["requires_calibration"] = d.requires_calibration;
  obj["calibration_available"] = d.calibration_available;
  obj["accuracy_risk"]      = d.accuracy_risk;
  obj["algorithm"]          = d.algorithm;
  if (!d.selected_candidate_id.empty())
    obj["selected_candidate_id"] = d.selected_candidate_id;
  if (!d.scheme.empty())
    obj["scheme"] = d.scheme;
  if (!d.backend.empty())
    obj["backend"] = d.backend;
  if (!d.kernel_id.empty())
    obj["kernel_id"] = d.kernel_id;
  if (!d.required_backend_capability.empty())
    obj["required_backend_capability"] = d.required_backend_capability;
  if (!d.required_kernel_capability.empty())
    obj["required_kernel_capability"] = d.required_kernel_capability;
  if (!d.calibration_artifact_ref.empty())
    obj["calibration_artifact_ref"] = d.calibration_artifact_ref;
  if (!d.calibration_artifact_id.empty())
    obj["calibration_artifact_id"] = d.calibration_artifact_id;
  if (!d.calibration_artifact_sha256.empty())
    obj["calibration_artifact_sha256"] = d.calibration_artifact_sha256;
  if (!d.packed_weight_artifact_ref.empty())
    obj["packed_weight_artifact_ref"] = d.packed_weight_artifact_ref;
  if (!d.packed_weight_artifact_id.empty())
    obj["packed_weight_artifact_id"] = d.packed_weight_artifact_id;
  if (!d.packed_weight_sha256.empty())
    obj["packed_weight_sha256"] = d.packed_weight_sha256;
  if (!d.source_weight_sha256.empty())
    obj["source_weight_sha256"] = d.source_weight_sha256;
  if (!d.packed_layout.empty())
    obj["packed_layout"] = d.packed_layout;
  if (!d.packing_scheme.empty())
    obj["packing_scheme"] = d.packing_scheme;
  if (d.kernel_requires_packed_weight)
    obj["kernel_requires_packed_weight"] = d.kernel_requires_packed_weight;
  if (!d.selected_complete_candidate_id.empty())
    obj["selected_complete_candidate_id"] = d.selected_complete_candidate_id;
  if (!d.codegen_target_id.empty())
    obj["codegen_target_id"] = d.codegen_target_id;
  if (!d.target_architecture.empty())
    obj["target_architecture"] = d.target_architecture;
  if (!d.target_microarchitecture.empty())
    obj["target_microarchitecture"] = d.target_microarchitecture;
  if (!d.required_isa_features.empty()) {
    llvm::json::Array arr;
    for (const auto &v : d.required_isa_features) arr.push_back(v);
    obj["required_isa_features"] = std::move(arr);
  }
  if (!d.compiler_flags.empty()) {
    llvm::json::Array arr;
    for (const auto &v : d.compiler_flags) arr.push_back(v);
    obj["compiler_flags"] = std::move(arr);
  }
  if (!d.binary_sha256.empty())
    obj["binary_sha256"] = d.binary_sha256;
  if (!d.measurement_artifact_ref.empty())
    obj["measurement_artifact_ref"] = d.measurement_artifact_ref;
  if (!d.build_manifest_ref.empty())
    obj["build_manifest_ref"] = d.build_manifest_ref;
  if (!d.workload_id.empty())
    obj["workload_id"] = d.workload_id;
  if (!d.activation_granularity.empty())
    obj["activation_granularity"] = d.activation_granularity;
  if (!d.weight_granularity.empty())
    obj["weight_granularity"] = d.weight_granularity;
  if (d.activation_scale > 0.0)
    obj["activation_scale"] = d.activation_scale;
  if (d.weight_scale > 0.0)
    obj["weight_scale"] = d.weight_scale;
  obj["activation_zero_point"] = d.activation_zero_point;
  obj["weight_zero_point"] = d.weight_zero_point;
  if (!d.policy_id.empty())
    obj["policy_id"] = d.policy_id;
  if (!d.selection_reason.empty())
    obj["selection_reason"] = d.selection_reason;
  if (!d.considered_candidate_ids.empty()) {
    llvm::json::Array arr;
    for (const auto &v : d.considered_candidate_ids) arr.push_back(v);
    obj["considered_candidate_ids"] = std::move(arr);
  }
  if (!d.rejected_candidate_ids.empty()) {
    llvm::json::Array arr;
    for (const auto &v : d.rejected_candidate_ids) arr.push_back(v);
    obj["rejected_candidate_ids"] = std::move(arr);
  }
  if (!d.rejected_candidate_reasons.empty()) {
    llvm::json::Array arr;
    for (const auto &v : d.rejected_candidate_reasons) arr.push_back(v);
    obj["rejected_candidate_reasons"] = std::move(arr);
  }
  if (!d.op_type.empty())
    obj["op_type"] = d.op_type;
  if (!d.quantized_model_artifact_ref.empty())
    obj["quantized_model_artifact_ref"] = d.quantized_model_artifact_ref;
  if (!d.execution_stages.empty()) {
    llvm::json::Array stages;
    for (const auto &stage : d.execution_stages) {
      llvm::json::Object s;
      s["stage_id"] = stage.stage_id;
      s["op"] = stage.op;
      llvm::json::Array deps;
      for (const auto &dep : stage.dependency_ids)
        deps.push_back(dep);
      s["dependency_ids"] = std::move(deps);
      if (!stage.produces.empty())
        s["produces"] = stage.produces;
      if (!stage.kernel_id.empty())
        s["kernel_id"] = stage.kernel_id;
      if (!stage.artifact_ref.empty())
        s["artifact_ref"] = stage.artifact_ref;
      if (!stage.artifact_sha256.empty())
        s["artifact_sha256"] = stage.artifact_sha256;
      if (!stage.packed_layout.empty())
        s["packed_layout"] = stage.packed_layout;
      if (!stage.fused_postprocess.empty())
        s["fused_postprocess"] = stage.fused_postprocess;
      if (stage.scale > 0.0)
        s["scale"] = stage.scale;
      s["zero_point"] = stage.zero_point;
      if (stage.clamp_min || stage.clamp_max) {
        s["clamp_min"] = stage.clamp_min;
        s["clamp_max"] = stage.clamp_max;
      }
      if (!stage.rounding_mode.empty())
        s["rounding_mode"] = stage.rounding_mode;
      if (!stage.source_dtype.empty())
        s["source_dtype"] = stage.source_dtype;
      if (!stage.destination_dtype.empty())
        s["destination_dtype"] = stage.destination_dtype;
      if (!stage.binary_sha256.empty())
        s["binary_sha256"] = stage.binary_sha256;
      stages.push_back(std::move(s));
    }
    obj["execution_stages"] = std::move(stages);
  }
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

static llvm::json::Object
serializeMemoryPlacementPlan(const MemoryPlacementPlan &mp) {
  llvm::json::Object obj;
  obj["status"] = mp.status;
  obj["compute_unit"] = mp.compute_unit;
  obj["selected_memory_space"] = mp.selected_memory_space;
  obj["input_tile_bytes"] = mp.input_tile_bytes;
  obj["weight_tile_bytes"] = mp.weight_tile_bytes;
  obj["output_tile_bytes"] = mp.output_tile_bytes;
  obj["scratch_bytes"] = mp.scratch_bytes;
  obj["padding_bytes"] = mp.padding_bytes;
  obj["single_buffer_bytes"] = mp.single_buffer_bytes;
  obj["additional_double_buffer_bytes"] =
      mp.additional_double_buffer_bytes;
  obj["total_required_local_memory_bytes"] =
      mp.total_required_local_memory_bytes;
  if (!mp.rejection_reason.empty())
    obj["rejection_reason"] = mp.rejection_reason;
  obj["truth_boundary"] = mp.truth_boundary;

  llvm::json::Array placements;
  for (const auto &bp : mp.buffer_placements) {
    llvm::json::Object b;
    b["buffer_id"] = bp.buffer_id;
    b["role"] = bp.role;
    b["memory_space"] = bp.memory_space;
    b["byte_count"] = bp.byte_count;
    b["alignment"] = bp.alignment;
    placements.push_back(std::move(b));
  }
  obj["buffer_placements"] = std::move(placements);

  llvm::json::Array transfers;
  for (const auto &transfer : mp.transfer_operations) {
    llvm::json::Object t;
    t["transfer_id"] = transfer.transfer_id;
    t["source_buffer"] = transfer.source_buffer;
    t["destination_buffer"] = transfer.destination_buffer;
    t["source_memory_space"] = transfer.source_memory_space;
    t["destination_memory_space"] = transfer.destination_memory_space;
    t["byte_count"] = transfer.byte_count;
    t["alignment"] = transfer.alignment;
    t["mode"] = transfer.mode;
    llvm::json::Array deps;
    for (const auto &dep : transfer.dependency_ids)
      deps.push_back(dep);
    t["dependency_ids"] = std::move(deps);
    t["completion_token"] = transfer.completion_token;
    transfers.push_back(std::move(t));
  }
  obj["transfer_operations"] = std::move(transfers);

  llvm::json::Array computeDeps;
  for (const auto &dep : mp.compute_dependency_ids)
    computeDeps.push_back(dep);
  obj["compute_dependency_ids"] = std::move(computeDeps);
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
  if (b.attention_execution) {
    const auto &a = *b.attention_execution; llvm::json::Object x;
    x["execution_unit"] = a.execution_unit; x["backend"] = a.backend;
    x["phase"] = a.phase; x["candidate_id"] = a.candidate_id;
    x["kernel_id"] = a.kernel_id; x["entry_point"] = a.entry_point;
    x["artifact_ref"] = a.artifact_ref; x["artifact_sha256"] = a.artifact_sha256;
    x["artifact_version"] = a.artifact_version; x["dtype"] = a.dtype;
    x["input_layout"] = a.input_layout; x["output_layout"] = a.output_layout;
    x["batch"] = a.batch; x["query_length"] = a.query_length; x["context_length"] = a.context_length;
    x["num_query_heads"] = a.num_query_heads; x["num_kv_heads"] = a.num_kv_heads;
    x["head_dim"] = a.head_dim; x["causal"] = a.causal; x["workspace_bytes"] = a.workspace_bytes;
    x["alignment_bytes"] = a.alignment_bytes;
    x["required_isa"] = a.required_isa; x["fallback_identity"] = a.fallback_identity;
    x["runtime_no_redecision"] = a.runtime_no_redecision; x["truth_boundary"] = a.truth_boundary;
    obj["attention_execution"] = std::move(x);
  }
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
  // Slice 2 memory placement and transfer contract. This block is a
  // runtime-consumable contract: consumers validate it instead of inventing
  // buffer placement or transfer ordering.
  if (b.memory_placement)
    obj["memory_placement"] =
        serializeMemoryPlacementPlan(*b.memory_placement);
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
  // Thread-decomposition schedule (Phase P1D, thread_schedule_contract_v1).
  // A decision SEPARATE from kernel_selection: which kernel/tile runs vs.
  // how many threads and what partitioning it uses. Absent when the
  // selected kernel (or no kernel) declares no thread schedules -- never
  // a claim of runtime execution or measured performance.
  if (b.thread_schedule) {
    const ThreadSchedule &tsched = *b.thread_schedule;
    llvm::json::Object tsObj;
    tsObj["status"] = tsched.status;
    if (tsched.status == "selected") {
      tsObj["thread_count"]       = tsched.thread_count;
      tsObj["partition_axis"]     = tsched.partition_axis;
      tsObj["partition_strategy"] = tsched.partition_strategy;
      tsObj["source"]             = tsched.source;
    }
    if (!tsched.policy_id.empty()) {
      tsObj["policy_id"] = tsched.policy_id;
      tsObj["policy_version"] = tsched.policy_version;
      tsObj["policy_metric"] = tsched.policy_metric;
      tsObj["policy_metric_value"] = tsched.policy_metric_value;
      tsObj["policy_threshold"] = tsched.policy_threshold;
      tsObj["policy_boundary_rule"] = tsched.policy_boundary_rule;
      tsObj["policy_selection_reason"] = tsched.policy_selection_reason;
      tsObj["policy_evidence_ref"] = tsched.policy_evidence_ref;
      tsObj["policy_evidence_sha256"] = tsched.policy_evidence_sha256;
      tsObj["policy_truth_boundary"] = tsched.policy_truth_boundary;
    }
    if (!tsched.rejection_reasons.empty()) {
      llvm::json::Array reasons;
      for (const auto &r : tsched.rejection_reasons)
        reasons.push_back(r);
      tsObj["rejection_reasons"] = std::move(reasons);
    }
    tsObj["contract_version"] = tsched.contract_version;
    tsObj["truth_boundary"]   = tsched.truth_boundary;
    obj["thread_schedule"] = std::move(tsObj);
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

static llvm::json::Array serializeShape(const std::vector<int64_t> &shape);

static llvm::json::Array
serializeStringArray(const std::vector<std::string> &values) {
  llvm::json::Array arr;
  for (const auto &value : values)
    arr.push_back(value);
  return arr;
}

static llvm::json::Array serializeI64Array(const std::vector<int64_t> &values) {
  llvm::json::Array arr;
  for (int64_t value : values)
    arr.push_back(value);
  return arr;
}

// Phase 26: runtime dispatch unit. One GenericGraphIR source node (or a
// materialized fusion) with helper MLIR ops folded inside; the runtime-facing
// execution granule. A unit is executable only when a concrete runtime kernel
// was registered and selected — never from configured backend policy alone.
static llvm::json::Object serializeDispatchUnit(const DispatchUnit &unit) {
  llvm::json::Object obj;
  obj["dispatch_unit_id"] = unit.dispatch_unit_id;
  obj["source_graph_node_ids"] = serializeI64Array(unit.source_graph_node_ids);
  obj["source_imported_node_ids"] =
      serializeI64Array(unit.source_imported_node_ids);
  obj["source_onnx_node_names"] =
      serializeStringArray(unit.source_onnx_node_names);
  obj["source_op_type"] = unit.source_op_type;
  obj["operation_family"] = unit.operation_family;
  if (!unit.semantic_region_id.empty())
    obj["semantic_region_id"] = unit.semantic_region_id;
  obj["mlir_operation_refs"] = serializeStringArray(unit.mlir_operation_refs);
  obj["input_tensor_ids"] = serializeStringArray(unit.input_tensor_ids);
  obj["output_tensor_ids"] = serializeStringArray(unit.output_tensor_ids);
  obj["initializer_tensor_ids"] =
      serializeStringArray(unit.initializer_tensor_ids);
  llvm::json::Object intent;
  intent["backend"] = unit.backend_intent.backend;
  intent["intent_basis"] = unit.backend_intent.intent_basis;
  obj["backend_intent"] = std::move(intent);
  obj["execution_domain"] = unit.execution_domain;
  obj["kernel_status"] = unit.kernel_status;
  if (!unit.selected_kernel_id.empty())
    obj["selected_kernel_id"] = unit.selected_kernel_id;
  obj["fallback_backends"] = serializeStringArray(unit.fallback_backends);
  obj["dtype"] = unit.dtype;
  obj["layout"] = unit.layout;
  obj["estimated_compute_flops"] = unit.estimated_compute_flops;
  obj["estimated_read_bytes"] = unit.estimated_read_bytes;
  obj["estimated_write_bytes"] = unit.estimated_write_bytes;
  obj["workspace_bytes"] = unit.workspace_bytes;
  obj["decision_provenance"] = unit.decision_provenance;
  obj["executable"] = unit.executable;
  if (!unit.non_executable_reason.empty())
    obj["non_executable_reason"] = unit.non_executable_reason;
  return obj;
}

static llvm::json::Object
serializeOpClassification(const DispatchOpClassification &cls) {
  llvm::json::Object obj;
  obj["total_mlir_operations"] = cls.total_mlir_operations;
  obj["dispatch_root"] = cls.dispatch_root;
  obj["dispatch_internal_compute"] = cls.dispatch_internal_compute;
  obj["tensor_contract_operation"] = cls.tensor_contract_operation;
  obj["allocation_helper"] = cls.allocation_helper;
  obj["scalar_helper"] = cls.scalar_helper;
  obj["view_operation"] = cls.view_operation;
  obj["non_dispatch_metadata"] = cls.non_dispatch_metadata;
  obj["unresolved"] = cls.unresolved;
  obj["operations_assigned_to_units"] = cls.operations_assigned_to_units;
  obj["source_graph_node_count"] = cls.source_graph_node_count;
  return obj;
}

static llvm::json::Object serializeTensorBinding(const TensorBinding &b) {
  llvm::json::Object obj;
  obj["tensor_id"] = b.tensor_id;
  obj["original_name"] = b.original_name;
  obj["source_value_id"] = b.source_value_id;
  obj["role"] = b.role;
  if (b.argument_index >= 0)
    obj["argument_index"] = b.argument_index;
  obj["shape"] = serializeShape(b.shape);
  obj["dtype"] = b.dtype;
  obj["layout"] = b.layout;
  obj["byte_size"] = b.byte_size;
  obj["ownership"] = b.ownership;
  obj["mutable"] = b.is_mutable;
  if (!b.external_data_reference.empty())
    obj["external_data_reference"] = b.external_data_reference;
  obj["model_artifact_reference"] = b.model_artifact_reference;
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

  // Phase 26 (CV full-graph only): dispatch units replace per-op decisions
  // as the runtime-facing list. Absent for LLM plans — Qwen output is
  // byte-identical.
  if (!fp.dispatch_units.empty()) {
    llvm::json::Array units;
    for (const auto &unit : fp.dispatch_units)
      units.push_back(serializeDispatchUnit(unit));
    obj["dispatch_units"] = std::move(units);
  }
  if (fp.op_classification)
    obj["op_classification"] = serializeOpClassification(*fp.op_classification);

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
  if (cv.memory_summary) {
    // Legacy field kept for compatibility; its cumulative semantics are now
    // explicit and deprecated in favor of memory_summary.
    memory["estimated_temporary_bytes_definition"] =
        "cumulative_ssa_result_write_volume_not_peak_live_deprecated";
  }
  obj["memory_estimates"] = std::move(memory);

  // Phase 26 corrected memory metrics.
  if (cv.memory_summary) {
    const CVMemorySummary &ms = *cv.memory_summary;
    llvm::json::Object summary;
    summary["model_input_bytes"] = ms.model_input_bytes;
    summary["initializer_bytes"] = ms.initializer_bytes;
    summary["model_output_bytes"] = ms.model_output_bytes;
    summary["total_intermediate_tensor_bytes"] =
        ms.total_intermediate_tensor_bytes;
    summary["total_intermediate_write_bytes"] =
        ms.total_intermediate_write_bytes;
    summary["peak_live_temporary_bytes"] = ms.peak_live_temporary_bytes;
    summary["workspace_bytes"] = ms.workspace_bytes;
    if (ms.planned_slot_bytes)
      summary["planned_slot_bytes"] = *ms.planned_slot_bytes;
    else
      summary["planned_slot_bytes"] = nullptr;  // no slot allocator exists
    summary["truth_boundary"] = ms.truth_boundary;
    obj["memory_summary"] = std::move(summary);
  }

  // Phase 26 runtime-facing postprocess contract.
  if (cv.postprocess_contract) {
    const CVPostprocessContract &pc = *cv.postprocess_contract;
    llvm::json::Object contract;
    contract["detection_tensor_id"] = pc.detection_tensor_id;
    contract["detection_shape"] = serializeShape(pc.detection_shape);
    contract["prototype_tensor_id"] = pc.prototype_tensor_id;
    contract["prototype_shape"] = serializeShape(pc.prototype_shape);
    llvm::json::Array groups;
    for (const auto &group : pc.detection_channel_groups) {
      llvm::json::Object g;
      g["channel_start"] = group.channel_start;
      g["channel_count"] = group.channel_count;
      g["semantic"] = group.semantic;
      if (!group.source_region_id.empty())
        g["source_region_id"] = group.source_region_id;
      groups.push_back(std::move(g));
    }
    contract["detection_channel_groups"] = std::move(groups);
    if (pc.mask_coefficient_channel_start >= 0) {
      contract["mask_coefficient_channel_start"] =
          pc.mask_coefficient_channel_start;
      contract["mask_coefficient_channel_end"] =
          pc.mask_coefficient_channel_end;
    }
    contract["nms_required"] = pc.nms_required;
    contract["mask_decode_required"] = pc.mask_decode_required;
    contract["implementation_status"] = pc.implementation_status;
    contract["expected_output_semantics"] = pc.expected_output_semantics;
    contract["confidence"] = pc.confidence;
    contract["provenance"] = pc.provenance;
    obj["postprocess_contract"] = std::move(contract);
  }

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
  // Phase 26 typed tensor ABI — emitted only when collected (CV path), so
  // existing LLM plans stay byte-identical.
  if (!plan.tensor_bindings.empty()) {
    llvm::json::Array bindings;
    for (const auto &binding : plan.tensor_bindings)
      bindings.push_back(serializeTensorBinding(binding));
    root["tensor_bindings"] = std::move(bindings);
  }

  return writeJSON(llvm::json::Value(std::move(root)), outPath);
}

llvm::Error
ExecutionPlanExporter::exportDispatchUnitReport(const ExecutionPlan &plan,
                                                llvm::StringRef outPath) {
  llvm::json::Object root;
  root["schema"] = "dispatch_unit_report";
  root["schema_version"] = "1.0.0";
  root["plan_id"] = plan.plan_id;
  root["truth_boundary"] =
      "real_yoloseg_dispatch_units_materialized_no_runtime_execution_"
      "no_registered_cv_kernels_no_measured_performance_"
      "no_memory_slot_assignment";

  llvm::json::Array functions;
  for (const auto &fp : plan.function_plans) {
    if (fp.dispatch_units.empty() && !fp.op_classification)
      continue;
    llvm::json::Object fn;
    fn["function_name"] = fp.function_name;
    fn["dispatch_unit_count"] =
        static_cast<int64_t>(fp.dispatch_units.size());
    int64_t executable = 0;
    llvm::json::Object kernelStatusCounts;
    std::map<std::string, int64_t> statusCounts;
    for (const auto &unit : fp.dispatch_units) {
      if (unit.executable)
        ++executable;
      ++statusCounts[unit.kernel_status];
    }
    for (const auto &entry : statusCounts)
      kernelStatusCounts[entry.first] = entry.second;
    fn["executable_dispatch_unit_count"] = executable;
    fn["non_executable_dispatch_unit_count"] =
        static_cast<int64_t>(fp.dispatch_units.size()) - executable;
    fn["kernel_status_counts"] = std::move(kernelStatusCounts);
    if (fp.op_classification)
      fn["op_classification"] =
          serializeOpClassification(*fp.op_classification);
    functions.push_back(std::move(fn));
  }
  root["function_reports"] = std::move(functions);

  llvm::json::Object bindingCounts;
  std::map<std::string, int64_t> roleCounts;
  for (const auto &binding : plan.tensor_bindings)
    ++roleCounts[binding.role];
  for (const auto &entry : roleCounts)
    bindingCounts[entry.first] = entry.second;
  root["tensor_binding_counts_by_role"] = std::move(bindingCounts);
  root["tensor_binding_count"] =
      static_cast<int64_t>(plan.tensor_bindings.size());

  if (plan.cv_extension && plan.cv_extension->memory_summary) {
    const CVMemorySummary &ms = *plan.cv_extension->memory_summary;
    llvm::json::Object memory;
    memory["model_input_bytes"] = ms.model_input_bytes;
    memory["initializer_bytes"] = ms.initializer_bytes;
    memory["model_output_bytes"] = ms.model_output_bytes;
    memory["total_intermediate_tensor_bytes"] =
        ms.total_intermediate_tensor_bytes;
    memory["total_intermediate_write_bytes"] =
        ms.total_intermediate_write_bytes;
    memory["peak_live_temporary_bytes"] = ms.peak_live_temporary_bytes;
    memory["legacy_estimated_temporary_bytes"] =
        plan.cv_extension->estimated_temporary_bytes;
    memory["legacy_definition"] =
        "cumulative_ssa_result_write_volume_not_peak_live_deprecated";
    root["memory_metric_reconciliation"] = std::move(memory);
  }

  return writeJSON(llvm::json::Value(std::move(root)), outPath);
}

} // namespace mlir::hir
