// compile-for-target: compiler driver for iPhone serving artifact generation.
//
// Pipeline:
//   TargetDeviceProfile JSON
//     → TargetDeviceProfile (parsed here, tool boundary)
//     → TargetProfileLowering (tool boundary)
//     → TargetConstraints
//     → TargetConstraints::attachToModule()
//     → serving passes (ServingPhaseAnalysisPass, KVLayoutPlanningPass,
//                       ReplayEligibilityPass, ExecutionProviderPlanningPass)
//     → ServingExecutionPlanBuilder
//     → ServingExecutionPlan
//     → ServingExecutionPlanExporter
//         → canonical artifact  (serving_execution_plan_iphone.json)
//         → summary artifact    (serving_execution_plan_summary.json)
//
// JSON construction is fully owned by ServingExecutionPlanExporter.
// This file never touches llvm::json types directly.

#include "FusionPasses.h"
#include "serving/ServingExecutionPlan.h"
#include "serving/ServingExecutionPlanBuilder.h"
#include "serving/ServingExecutionPlanExporter.h"
#include "serving/TargetConstraints.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// CLI flags
// ---------------------------------------------------------------------------

static llvm::cl::opt<std::string> DeviceProfilePath(
    "device-profile",
    llvm::cl::desc("Path to TargetDeviceProfile JSON (e.g. configs/target_profiles/apple_a17pro_mobile.json)"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> MlirPath(
    "mlir",
    llvm::cl::desc("Path to input MLIR file"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> OutPath(
    "out",
    llvm::cl::desc("Output path for canonical serving_execution_plan artifact"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> DumpAnnotatedMlir(
    "dump-annotated-mlir",
    llvm::cl::desc("(Optional) write pass-annotated MLIR to this path"),
    llvm::cl::init(""));

// ---------------------------------------------------------------------------
// TargetDeviceProfile — tool-boundary struct
// Only the fields consumed by TargetProfileLowering are populated.
// chipName, totalRAMBytes, thermalState, lowPowerMode, modelIdentifier,
// and CPU count are parsed for provenance only and not forwarded to the compiler.
// ---------------------------------------------------------------------------

struct TargetDeviceProfile {
  std::string profileId;
  double      metalMaxWorkingSetMB = 0.0;
  std::string configuredComputeUnits;
  bool        staticShapeSupport = true;
  std::vector<std::string> supportedPrecisions;
  std::string truthBoundary;
};

// ---------------------------------------------------------------------------
// TargetProfileLowering — tool-boundary mapping
// ---------------------------------------------------------------------------

static mlir::hir::TargetConstraints
lowerToTargetConstraints(const TargetDeviceProfile &prof) {
  mlir::hir::TargetConstraints tc;

  tc.profile_id = prof.profileId;

  if (prof.metalMaxWorkingSetMB > 0.0) {
    tc.memory_budget_mb  = prof.metalMaxWorkingSetMB;
    tc.has_memory_budget = true;
  }

  tc.static_shape_support     = prof.staticShapeSupport;
  tc.has_static_shape_support = true;

  const auto &cu = prof.configuredComputeUnits;
  if (cu == "CPU+GPU+ANE") {
    tc.preferred_backend = "coreml";
    tc.allowed_backends  = {"coreml", "metal", "cpu"};
  } else if (cu == "CPU+GPU") {
    tc.preferred_backend = "metal";
    tc.allowed_backends  = {"metal", "cpu"};
  } else if (cu == "CPU") {
    tc.preferred_backend = "cpu";
    tc.allowed_backends  = {"cpu"};
  }
  // Unknown/absent configuredComputeUnits: leave unconstrained.

  if (!prof.supportedPrecisions.empty()) {
    tc.supported_precisions = prof.supportedPrecisions;
  } else {
    // Default: fp16 for GPU/ANE targets, fp32+fp16 for CPU-only.
    if (cu == "CPU") {
      tc.supported_precisions = {"fp32", "fp16"};
    } else {
      tc.supported_precisions = {"fp16"};
    }
  }

  return tc;
}

// ---------------------------------------------------------------------------
// parseDeviceProfile — reads and validates the JSON device profile
// ---------------------------------------------------------------------------

static llvm::Expected<TargetDeviceProfile>
parseDeviceProfile(llvm::StringRef path) {
  auto buf = llvm::MemoryBuffer::getFile(path);
  if (!buf)
    return llvm::make_error<llvm::StringError>(
        "cannot read device profile '" + path.str() + "': " +
            buf.getError().message(),
        llvm::inconvertibleErrorCode());

  auto json = llvm::json::parse((*buf)->getBuffer());
  if (!json)
    return llvm::make_error<llvm::StringError>(
        "JSON parse error in device profile '" + path.str() + "': " +
            llvm::toString(json.takeError()),
        llvm::inconvertibleErrorCode());

  const llvm::json::Object *obj = json->getAsObject();
  if (!obj)
    return llvm::make_error<llvm::StringError>(
        "device profile must be a JSON object",
        llvm::inconvertibleErrorCode());

  TargetDeviceProfile prof;

  if (auto v = obj->getString("profileId"))
    prof.profileId = v->str();
  else
    return llvm::make_error<llvm::StringError>(
        "device profile missing required field 'profileId'",
        llvm::inconvertibleErrorCode());

  if (auto v = obj->getNumber("metalMaxWorkingSetMB"))
    prof.metalMaxWorkingSetMB = *v;

  if (auto v = obj->getString("configuredComputeUnits"))
    prof.configuredComputeUnits = v->str();

  if (auto v = obj->getBoolean("staticShapeSupport"))
    prof.staticShapeSupport = *v;

  if (auto *arr = obj->getArray("supportedPrecisions")) {
    for (const auto &elem : *arr) {
      if (auto s = elem.getAsString())
        prof.supportedPrecisions.push_back(s->str());
    }
  }

  if (auto v = obj->getString("truthBoundary"))
    prof.truthBoundary = v->str();

  return prof;
}

// ---------------------------------------------------------------------------
// deriveSummaryPath: same directory as outPath, fixed filename
// ---------------------------------------------------------------------------

static std::string deriveSummaryPath(llvm::StringRef outPath) {
  llvm::SmallString<256> dir(outPath);
  llvm::sys::path::remove_filename(dir);
  llvm::SmallString<256> summary(dir);
  llvm::sys::path::append(summary, "serving_execution_plan_summary.json");
  return summary.str().str();
}

// ---------------------------------------------------------------------------
// printTerminalSummary
// ---------------------------------------------------------------------------

static void printTerminalSummary(const TargetDeviceProfile &prof,
                                  const mlir::hir::ServingExecutionPlan &plan,
                                  llvm::StringRef canonicalPath,
                                  llvm::StringRef summaryPath) {
  llvm::outs() << "\n";
  llvm::outs() << "compile-for-target: " << prof.profileId
               << " \xe2\x86\x92 " << plan.model_name << "\n";
  llvm::outs() << "  canonical:      " << canonicalPath << "\n";
  llvm::outs() << "  summary:        " << summaryPath
               << " (documentation only)\n";
  llvm::outs() << "  function plans: "
               << plan.function_plans.size() << "\n";

  auto kvStr = [](mlir::hir::KVLayout l) -> llvm::StringRef {
    switch (l) {
    case mlir::hir::KVLayout::Paged:      return "paged";
    case mlir::hir::KVLayout::Contiguous: return "contiguous";
    default:                               return "unknown";
    }
  };

  for (const auto &fp : plan.function_plans) {
    llvm::outs() << "\n";
    llvm::outs() << "  " << fp.function_name << ":\n";
    llvm::outs() << "    serving_phase:   "
                 << (fp.serving_phase == mlir::hir::ServingPhase::Prefill
                         ? "prefill"
                         : fp.serving_phase == mlir::hir::ServingPhase::Decode
                               ? "decode"
                               : "unknown")
                 << "\n";
    llvm::outs() << "    primary_backend: "
                 << fp.backend_execution_plan.primary_backend << "\n";
    llvm::outs() << "    kv_layout:       "
                 << kvStr(fp.kv_plan.layout) << "\n";
    llvm::outs() << "    replay_eligible: "
                 << (fp.replay_plan.replay_eligible ? "true" : "false") << "\n";
    llvm::outs() << "    decision_source: "
                 << fp.backend_execution_plan.decision_source << "\n";
  }
  llvm::outs() << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "compile-for-target: compile MLIR for a target device profile\n");

  // 1. Parse device profile.
  auto profOrErr = parseDeviceProfile(DeviceProfilePath);
  if (!profOrErr) {
    llvm::errs() << "error: " << llvm::toString(profOrErr.takeError()) << "\n";
    return 1;
  }
  TargetDeviceProfile prof = std::move(*profOrErr);

  // 2. Lower profile → TargetConstraints.
  mlir::hir::TargetConstraints constraints = lowerToTargetConstraints(prof);

  // 3. Parse MLIR module.
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);
  ctx.loadDialect<mlir::func::FuncDialect>();

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(MlirPath, &ctx);
  if (!module) {
    llvm::errs() << "error: failed to parse MLIR file: " << MlirPath << "\n";
    return 1;
  }

  // 4. Attach TargetConstraints as module attrs.
  constraints.attachToModule(module.get(), &ctx);

  // 5. Run the serving-optimization-pipeline (4 passes).
  mlir::PassManager pm(&ctx);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createServingPhaseAnalysisPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createKVLayoutPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createReplayEligibilityPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createExecutionProviderPlanningPass());

  if (pm.run(module.get()).failed()) {
    llvm::errs() << "error: serving pass pipeline failed\n";
    return 1;
  }

  // 6. Optionally dump annotated MLIR.
  if (!DumpAnnotatedMlir.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream os(DumpAnnotatedMlir, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "warning: cannot write annotated MLIR to '"
                   << DumpAnnotatedMlir << "': " << ec.message() << "\n";
    } else {
      module->print(os);
    }
  }

  // 7. Build ServingExecutionPlan from annotated module.
  mlir::hir::ServingExecutionPlan plan =
      mlir::hir::ServingExecutionPlanBuilder::build(module.get());

  // 8. Export canonical artifact.
  if (auto err = mlir::hir::ServingExecutionPlanExporter::exportToFile(
          plan, OutPath)) {
    llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }

  // 9. Export summary artifact (same plan instance, derived path).
  std::string summaryPath = deriveSummaryPath(OutPath);
  if (auto err = mlir::hir::ServingExecutionPlanExporter::exportSummaryToFile(
          plan, summaryPath)) {
    llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }

  // 10. Print terminal summary.
  printTerminalSummary(prof, plan, OutPath, summaryPath);

  return 0;
}
