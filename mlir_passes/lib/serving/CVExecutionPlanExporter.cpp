#include "serving/CVExecutionPlanExporter.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <ctime>
#include <system_error>

namespace mlir::hir {
namespace {

std::string utcTimestamp() {
  std::time_t t = std::time(nullptr);
  std::tm tmUtc{};
#if defined(_WIN32)
  gmtime_s(&tmUtc, &t);
#else
  gmtime_r(&t, &tmUtc);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmUtc);
  return buf;
}

llvm::Error writeJSON(const llvm::json::Value &val, llvm::StringRef path) {
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

llvm::json::Object buildTruthBoundariesObject(const CVTruthBoundaries &tb) {
  llvm::json::Object obj;
  obj["frontend"] = tb.frontend;
  obj["shape_inference"] = tb.shape_inference;
  obj["memory_plan"] = tb.memory_plan;
  obj["execution_domain"] = tb.execution_domain;
  return obj;
}

llvm::json::Object buildMemorySummaryObject(const CVMemoryPlanSummary &summary) {
  llvm::json::Object obj;
  obj["status"] = summary.status;
  obj["buffer_count"] = summary.buffer_count;
  obj["peak_memory_bytes"] = summary.peak_memory_bytes;
  obj["planned_ops"] = summary.planned_ops;
  obj["reused_buffer_count"] = summary.reused_buffer_count;
  obj["skipped_ops"] = summary.skipped_ops;
  obj["total_allocated_bytes"] = summary.total_allocated_bytes;
  obj["truth_boundary"] = summary.truth_boundary;
  return obj;
}

llvm::json::Object
buildExecutionDomainSummaryObject(const CVExecutionDomainSummary &summary) {
  llvm::json::Object obj;
  obj["status"] = summary.status;
  obj["accelerated_ops"] = summary.accelerated_ops;
  obj["host_ops"] = summary.host_ops;
  obj["fallback_ops"] = summary.fallback_ops;
  obj["planned_ops"] = summary.planned_ops;
  obj["truth_boundary"] = summary.truth_boundary;
  return obj;
}

llvm::json::Object buildStepObject(const CVExecutionStep &step,
                                   int64_t stepId) {
  llvm::json::Object obj;
  obj["step_id"] = stepId;
  obj["op_name"] = step.op_name;
  obj["source_op"] = step.source_op;
  obj["execution_domain"] = step.execution_domain;
  obj["execution_domain_reason"] = step.execution_domain_reason;
  obj["buffer_id"] = step.buffer_id;
  obj["buffer_offset"] = step.buffer_offset;
  obj["lifetime_begin"] = step.lifetime_begin;
  obj["lifetime_end"] = step.lifetime_end;
  obj["bytes_estimate"] = step.bytes_estimate;
  obj["num_elements"] = step.num_elements;
  obj["memory_plan_reason"] = step.memory_plan_reason;
  return obj;
}

llvm::json::Object buildGraphPlanObject(const CVGraphPlan &graph) {
  llvm::json::Array steps;
  for (size_t i = 0; i < graph.steps.size(); ++i)
    steps.push_back(buildStepObject(graph.steps[i], static_cast<int64_t>(i)));

  llvm::json::Object obj;
  obj["function_name"] = graph.function_name;
  obj["total_cv_ops"] = static_cast<int64_t>(graph.steps.size());
  obj["truth_boundaries"] =
      buildTruthBoundariesObject(graph.truth_boundaries);
  obj["memory_summary"] = buildMemorySummaryObject(graph.memory_plan);
  obj["execution_domain_summary"] =
      buildExecutionDomainSummaryObject(graph.execution_domain_plan);
  obj["steps"] = std::move(steps);
  return obj;
}

} // namespace

llvm::Error CVExecutionPlanExporter::exportToFile(const CVExecutionPlan &plan,
                                                  llvm::StringRef outPath) {
  llvm::json::Array graphPlans;
  for (const CVGraphPlan &graph : plan.graph_plans)
    graphPlans.push_back(buildGraphPlanObject(graph));

  llvm::json::Object root;
  root["artifact_type"] = "cv_execution_plan";
  root["schema_version"] = "1.0.0";
  root["model_name"] = plan.model_name;
  root["target_profile_id"] = plan.target_profile_id;
  root["generated_at"] = utcTimestamp();
  root["graph_plans"] = std::move(graphPlans);

  return writeJSON(llvm::json::Value(std::move(root)), outPath);
}

} // namespace mlir::hir
