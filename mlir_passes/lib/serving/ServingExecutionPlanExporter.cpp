#include "serving/ServingExecutionPlanExporter.h"
#include "serving/ServingEnums.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <ctime>
#include <set>

namespace mlir::hir {
namespace {

// ---------------------------------------------------------------------------
// Enum → string helpers
// ---------------------------------------------------------------------------

static llvm::StringRef toString(ServingPhase p) {
  switch (p) {
  case ServingPhase::Prefill: return "prefill";
  case ServingPhase::Decode:  return "decode";
  default:                    return "unknown";
  }
}

static llvm::StringRef toString(ExecutionMode m) {
  switch (m) {
  case ExecutionMode::Colocated: return "colocated";
  case ExecutionMode::PDSplit:   return "pd_split";
  default:                       return "unknown";
  }
}

static llvm::StringRef toString(KVLayout l) {
  switch (l) {
  case KVLayout::Paged:      return "paged";
  case KVLayout::Contiguous: return "contiguous";
  default:                   return "unknown";
  }
}

static llvm::StringRef toString(Confidence c) {
  switch (c) {
  case Confidence::Medium: return "medium";
  case Confidence::High:   return "high";
  default:                 return "low";
  }
}

// ---------------------------------------------------------------------------
// ISO 8601 UTC timestamp
// ---------------------------------------------------------------------------

static std::string utcTimestamp() {
  std::time_t t = std::time(nullptr);
  std::tm tm_utc{};
#if defined(_WIN32)
  gmtime_s(&tm_utc, &t);
#else
  gmtime_r(&t, &tm_utc);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
  return buf;
}

// ---------------------------------------------------------------------------
// Write JSON value to file (creates parent dirs as needed)
// ---------------------------------------------------------------------------

static llvm::Error writeJSON(const llvm::json::Value &val,
                              llvm::StringRef path) {
  // Create parent directories.
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
// Build canonical function plan object
// ---------------------------------------------------------------------------

static llvm::json::Object buildFunctionPlanObject(
    const FunctionExecutionPlan &fp) {

  // cost_summary
  llvm::json::Object cost;
  cost["colocated_total_ms"]   = fp.cost_summary.colocated_total_ms;
  cost["pd_split_total_ms"]    = fp.cost_summary.pd_split_total_ms;
  cost["decision_margin_ms"]   = fp.cost_summary.decision_margin_ms;
  cost["decision_margin_pct"]  = fp.cost_summary.decision_margin_pct;
  cost["confidence"]           = toString(fp.cost_summary.confidence);
  cost["policy"]               = toString(fp.execution_mode);
  cost["cost_source"]          = fp.provenance.cost_source;

  // kv_plan
  llvm::json::Object kv;
  kv["layout"]             = toString(fp.kv_plan.layout);
  kv["kv_byte_estimate_mb"] = fp.kv_plan.kv_byte_estimate_mb;
  kv["layout_reason"]      = "";
  kv["truth_boundary"]     = "static_formula_estimate_not_measured_memory";

  // replay_plan
  llvm::json::Object replay;
  replay["replay_eligible"]   = fp.replay_plan.replay_eligible;
  replay["cuda_graph_bucket"] = fp.replay_plan.cuda_graph_bucket;
  replay["override_reason"]   = "";
  replay["truth_boundary"]    =
      "static_shape_replay_eligibility_not_cuda_graph_capture";

  // backend_execution_plan
  llvm::json::Array fallback;
  for (const auto &b : fp.backend_execution_plan.fallback_chain)
    fallback.push_back(b);

  llvm::json::Object bep;
  bep["primary_backend"]  = fp.backend_execution_plan.primary_backend;
  bep["fallback_chain"]   = std::move(fallback);
  bep["decision_source"]  = fp.backend_execution_plan.decision_source;
  bep["required_precision"]   = fp.backend_execution_plan.required_precision;
  bep["required_kv_layout"]   = fp.backend_execution_plan.required_kv_layout;
  bep["requires_replay"]      = fp.backend_execution_plan.requires_replay;

  // provenance
  llvm::json::Object prov;
  prov["truth_boundary"] = fp.provenance.truth_boundary;
  prov["cost_source"]    = fp.provenance.cost_source;

  // source_passes
  llvm::json::Array passes;
  for (const auto &p : fp.source_passes)
    passes.push_back(p);

  llvm::json::Object obj;
  obj["function_name"]          = fp.function_name;
  obj["serving_phase"]          = toString(fp.serving_phase);
  obj["execution_mode"]         = toString(fp.execution_mode);
  obj["cost_summary"]           = std::move(cost);
  obj["kv_plan"]                = std::move(kv);
  obj["replay_plan"]            = std::move(replay);
  obj["backend_execution_plan"] = std::move(bep);
  obj["provenance"]             = std::move(prov);
  obj["source_passes"]          = std::move(passes);

  return obj;
}

} // namespace

// ---------------------------------------------------------------------------
// exportToFile — canonical runtime artifact
// ---------------------------------------------------------------------------

llvm::Error ServingExecutionPlanExporter::exportToFile(
    const ServingExecutionPlan &plan, llvm::StringRef outPath) {

  llvm::json::Array functionPlans;
  for (const auto &fp : plan.function_plans)
    functionPlans.push_back(buildFunctionPlanObject(fp));

  llvm::json::Object root;
  root["schema_version"]    = "1.0.0";
  root["artifact_type"]     = "serving_execution_plan";
  root["truth_boundary"]    =
      "declared_device_profile_not_measured_silicon_performance";
  root["target_profile_id"] = plan.target_profile_id;
  root["model_name"]        = plan.model_name;
  root["num_layers"]        = plan.num_layers;
  root["hidden_size"]       = plan.hidden_size;
  if (plan.num_attention_heads > 0)
    root["num_attention_heads"] = plan.num_attention_heads;
  if (plan.num_kv_heads > 0)
    root["num_kv_heads"] = plan.num_kv_heads;
  root["function_plans"]    = std::move(functionPlans);

  return writeJSON(llvm::json::Value(std::move(root)), outPath);
}

// ---------------------------------------------------------------------------
// exportSummaryToFile — human/demo artifact (NOT a runtime input)
// ---------------------------------------------------------------------------

llvm::Error ServingExecutionPlanExporter::exportSummaryToFile(
    const ServingExecutionPlan &plan, llvm::StringRef summaryPath) {

  // Collect ordered unique pass names across all function plans.
  std::set<std::string> seen;
  llvm::json::Array passes;
  for (const auto &fp : plan.function_plans) {
    for (const auto &p : fp.source_passes) {
      if (seen.insert(p).second)
        passes.push_back(p);
    }
  }

  // Per-function summaries keyed by function_name.
  llvm::json::Object backendSummary;
  llvm::json::Object kvLayoutSummary;
  llvm::json::Object replaySummary;

  for (const auto &fp : plan.function_plans) {
    llvm::json::Array fallback;
    for (const auto &b : fp.backend_execution_plan.fallback_chain)
      fallback.push_back(b);

    llvm::json::Object bep;
    bep["primary_backend"]  = fp.backend_execution_plan.primary_backend;
    bep["fallback_chain"]   = std::move(fallback);
    bep["decision_source"]  = fp.backend_execution_plan.decision_source;

    backendSummary[fp.function_name]   = std::move(bep);
    kvLayoutSummary[fp.function_name]  = toString(fp.kv_plan.layout);
    replaySummary[fp.function_name]    = fp.replay_plan.replay_eligible;
  }

  // Truth boundaries block.
  llvm::json::Object truthBoundaries;
  truthBoundaries["device_profile"] =
      "declared_device_profile_not_measured_silicon_performance";
  truthBoundaries["kv_estimates"] =
      "static_formula_estimate_not_measured_memory";
  truthBoundaries["replay_eligibility"] =
      "static_shape_replay_eligibility_not_cuda_graph_capture";
  truthBoundaries["backend_decisions"] =
      "compiler_execution_provider_plan_not_runtime_dispatch";

  llvm::json::Object root;
  root["schema_version"]      = "1.0.0";
  root["artifact_type"]       = "serving_execution_plan_summary";
  root["NOT_A_RUNTIME_INPUT"] =
      "Runtime consumes serving_execution_plan_iphone.json. "
      "This file is documentation only.";
  root["target_profile_id"]   = plan.target_profile_id;
  root["model_name"]          = plan.model_name;
  root["function_count"]      = static_cast<int64_t>(plan.function_plans.size());
  root["generation_timestamp"] = utcTimestamp();
  root["compiler_version"]    = "hir-compiler-0.1";
  root["compiler_passes"]     = std::move(passes);
  root["backend_summary"]     = std::move(backendSummary);
  root["kv_layout_summary"]   = std::move(kvLayoutSummary);
  root["replay_summary"]      = std::move(replaySummary);
  root["truth_boundaries"]    = std::move(truthBoundaries);

  return writeJSON(llvm::json::Value(std::move(root)), summaryPath);
}

} // namespace mlir::hir
