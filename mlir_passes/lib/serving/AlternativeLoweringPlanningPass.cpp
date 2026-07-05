#include "FusionPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_ALTERNATIVELOWERINGPLANNING
#include "FusionPasses.h.inc"

static constexpr StringLiteral kTruth =
    "alternative_lowering_static_not_materialized_not_cost_evaluated";

// ---------------------------------------------------------------------------
// Kernel library entry (minimal subset needed for alternative matching).
// ---------------------------------------------------------------------------

struct AltKernelEntry {
  std::string op_type;
  std::vector<std::string> supported_dtypes;
  std::vector<std::string> supported_layouts;
  std::vector<std::string> supported_quant_modes;
  std::string fallback_backend;
};

static bool inList(const std::vector<std::string> &list, const std::string &val) {
  return std::find(list.begin(), list.end(), val) != list.end();
}

static std::vector<AltKernelEntry> loadKernelLibrary(mlir::Operation *module,
                                                      const std::string &backend) {
  std::vector<AltKernelEntry> out;
  if (!module || backend.empty()) return out;
  auto arr = module->getAttrOfType<mlir::ArrayAttr>(
      "target.kernel_libraries." + backend);
  if (!arr) return out;
  for (auto elem : arr) {
    auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(elem);
    if (!dict) continue;
    auto rStr = [&](llvm::StringRef k) -> std::string {
      if (auto a = dict.get(k))
        if (auto s = mlir::dyn_cast<mlir::StringAttr>(a)) return s.getValue().str();
      return {};
    };
    auto rStrs = [&](llvm::StringRef k) -> std::vector<std::string> {
      std::vector<std::string> r;
      if (auto a = dict.get(k))
        if (auto a2 = mlir::dyn_cast<mlir::ArrayAttr>(a))
          for (auto e : a2)
            if (auto s = mlir::dyn_cast<mlir::StringAttr>(e))
              r.push_back(s.getValue().str());
      return r;
    };
    AltKernelEntry ke;
    ke.op_type              = rStr("op_type");
    ke.supported_dtypes     = rStrs("supported_dtypes");
    ke.supported_layouts    = rStrs("supported_layouts");
    ke.supported_quant_modes = rStrs("supported_quant_modes");
    ke.fallback_backend     = rStr("fallback_backend");
    out.push_back(std::move(ke));
  }
  return out;
}

static bool hasKernel(const std::vector<AltKernelEntry> &lib,
                      const std::string &op_type,
                      const std::string &dtype,
                      const std::string &layout = "") {
  for (const auto &ke : lib) {
    if (ke.op_type != op_type) continue;
    bool dtypeOk = ke.supported_dtypes.empty() || inList(ke.supported_dtypes, dtype);
    bool layoutOk = layout.empty() || ke.supported_layouts.empty()
                    || inList(ke.supported_layouts, layout);
    if (dtypeOk && layoutOk) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Candidate building.
// ---------------------------------------------------------------------------

static mlir::DictionaryAttr buildCandidate(
    mlir::MLIRContext *ctx,
    const std::string &id,
    const std::string &alt_type,
    const std::string &source_op,
    const std::vector<std::string> &target_ops,
    const std::vector<std::string> &boundary_ops,
    const std::vector<std::string> &kernel_checks,
    const std::string &status,
    const std::string &failures,
    const std::string &risk,
    const std::string &source_level = "pass_generated") {
  auto S = [&](llvm::StringRef s) -> mlir::Attribute {
    return mlir::StringAttr::get(ctx, s);
  };
  auto SArr = [&](const std::vector<std::string> &strs) -> mlir::Attribute {
    llvm::SmallVector<mlir::Attribute> attrs;
    for (const auto &s : strs) attrs.push_back(mlir::StringAttr::get(ctx, s));
    return mlir::ArrayAttr::get(ctx, attrs);
  };
  // DictionaryAttr::get sorts keys alphabetically.
  llvm::SmallVector<mlir::NamedAttribute> entries = {
    {mlir::StringAttr::get(ctx, "alternative_id"),         S(id)},
    {mlir::StringAttr::get(ctx, "alternative_type"),       S(alt_type)},
    {mlir::StringAttr::get(ctx, "required_boundary_ops"),  SArr(boundary_ops)},
    {mlir::StringAttr::get(ctx, "required_kernel_checks"), SArr(kernel_checks)},
    {mlir::StringAttr::get(ctx, "risk"),                   S(risk)},
    {mlir::StringAttr::get(ctx, "source_level"),           S(source_level)},
    {mlir::StringAttr::get(ctx, "source_op"),              S(source_op)},
    {mlir::StringAttr::get(ctx, "target_ops"),             SArr(target_ops)},
    {mlir::StringAttr::get(ctx, "truth_boundary"),         S(kTruth)},
    {mlir::StringAttr::get(ctx, "validation_failures"),    S(failures)},
    {mlir::StringAttr::get(ctx, "validation_status"),      S(status)},
  };
  return mlir::DictionaryAttr::get(ctx, entries);
}

// ---------------------------------------------------------------------------
// Algebraic decomposition rules.
// ---------------------------------------------------------------------------

struct DecompRule {
  const char *source_op;
  std::vector<std::string> target_ops;
  const char *risk;
};

static const DecompRule kDecompRules[] = {
  {"gelu",      {"mul", "sigmoid", "mul"},             "medium"},
  {"attention", {"matmul", "softmax", "matmul"},       "high"},
};

// ---------------------------------------------------------------------------
// Pass implementation.
// ---------------------------------------------------------------------------

struct AlternativeLoweringPlanningPass
    : impl::AlternativeLoweringPlanningBase<AlternativeLoweringPlanningPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    mlir::MLIRContext *ctx = funcOp.getContext();
    mlir::Operation *module = funcOp->getParentOp();

    if (funcOp.getBody().empty()) return;

    std::string backend;
    if (auto a = funcOp->getAttrOfType<mlir::StringAttr>("representation.source_backend"))
      backend = a.getValue().str();

    auto lib = loadKernelLibrary(module, backend);

    auto readModuleBool = [&](llvm::StringRef key) -> bool {
      if (module)
        if (auto a = module->getAttrOfType<mlir::BoolAttr>(key))
          return a.getValue();
      return false;
    };

    std::string capPfx = backend.empty() ? "" : "target.backend_capabilities." + backend + ".";
    bool supportsCast    = capPfx.empty() ? false : readModuleBool(capPfx + "supports_cast");
    bool supportsDequant = capPfx.empty() ? false : readModuleBool(capPfx + "supports_dequant_boundary");
    bool supportsLayout  = capPfx.empty() ? false : readModuleBool(capPfx + "supports_layout_transform");

    mlir::Block &entry = funcOp.getBody().front();
    int altCounter = 0;

    for (mlir::Operation &op : entry.without_terminator()) {
      auto lsAttr = op.getAttrOfType<mlir::StringAttr>("kernel.lowering_status");
      if (!lsAttr) continue;
      std::string ls = lsAttr.getValue().str();

      bool kernelExists = false;
      if (auto a = op.getAttrOfType<mlir::BoolAttr>("kernel.exists"))
        kernelExists = a.getValue();

      if (kernelExists && ls == "lowerable") {
        op.setAttr("alternative.available",      mlir::BoolAttr::get(ctx, false));
        op.setAttr("alternative.best_candidate", mlir::StringAttr::get(ctx, ""));
        op.setAttr("alternative.candidates",     mlir::ArrayAttr::get(ctx, {}));
        op.setAttr("alternative.reason",         mlir::StringAttr::get(ctx, "exact_kernel_available"));
        op.setAttr("alternative.truth_boundary", mlir::StringAttr::get(ctx, kTruth));
        continue;
      }

      // Determine op short name (everything after last '.').
      llvm::StringRef fullName = op.getName().getStringRef();
      std::string opType = fullName.str();
      if (auto dot = fullName.rfind('.'); dot != llvm::StringRef::npos)
        opType = fullName.substr(dot + 1).str();

      // Effective dtype: prefer per-op quant.activation_dtype, then func-level.
      std::string effectiveDtype;
      if (auto a = op.getAttrOfType<mlir::StringAttr>("quant.activation_dtype"))
        effectiveDtype = a.getValue().str();
      if (effectiveDtype.empty())
        if (auto a = funcOp->getAttrOfType<mlir::StringAttr>("representation.effective_dtype"))
          effectiveDtype = a.getValue().str();

      std::string effectiveLayout;
      if (auto a = op.getAttrOfType<mlir::StringAttr>("layout.effective_layout"))
        effectiveLayout = a.getValue().str();

      std::string strategyStr;
      if (auto a = op.getAttrOfType<mlir::StringAttr>("quant.strategy"))
        strategyStr = a.getValue().str();

      std::string fallbackBackend;
      if (auto a = op.getAttrOfType<mlir::StringAttr>("kernel.fallback_backend"))
        fallbackBackend = a.getValue().str();

      bool castRequired = false;
      if (auto a = op.getAttrOfType<mlir::BoolAttr>("boundary.cast_required"))
        castRequired = a.getValue();

      bool weightDequantRequired = false;
      if (auto a = op.getAttrOfType<mlir::BoolAttr>("boundary.weight_dequant_required"))
        weightDequantRequired = a.getValue();

      llvm::SmallVector<mlir::Attribute> candidates;

      // 1. Algebraic decomposition.
      for (const auto &rule : kDecompRules) {
        if (opType != rule.source_op) continue;
        std::vector<std::string> unique_targets;
        for (const auto &tgt : rule.target_ops)
          if (!inList(unique_targets, tgt)) unique_targets.push_back(tgt);
        std::vector<std::string> missing;
        for (const auto &tgt : unique_targets)
          if (!hasKernel(lib, tgt, effectiveDtype)) missing.push_back(tgt);
        std::string status = missing.empty() ? "valid" : "invalid_missing_kernel";
        std::string failures;
        if (!missing.empty()) {
          failures = "missing_kernels:";
          for (size_t i = 0; i < missing.size(); ++i)
            failures += (i ? "," : "") + missing[i];
        }
        candidates.push_back(buildCandidate(
            ctx, "alt_" + std::to_string(altCounter++),
            "algebraic_decomposition", opType,
            rule.target_ops, {}, unique_targets,
            status, failures, rule.risk));
      }

      // 2. Representation conversion: weight_only_int8 → dequant_weight + fp16 op.
      if (strategyStr == "weight_only_int8") {
        bool fp16KernelExists = hasKernel(lib, opType, "fp16");
        bool dequantOk = supportsDequant || weightDequantRequired;
        std::string status, failures;
        if (fp16KernelExists && dequantOk) {
          status = "valid";
        } else {
          status = fp16KernelExists ? "invalid_missing_boundary_support"
                                    : "invalid_missing_kernel";
          if (!fp16KernelExists)
            failures = "missing_fp16_" + opType + "_kernel";
          if (!dequantOk)
            failures += (failures.empty() ? "" : ",") + std::string("dequant_boundary_not_supported");
        }
        candidates.push_back(buildCandidate(
            ctx, "alt_" + std::to_string(altCounter++),
            "representation_conversion", opType,
            {"dequant_weight", opType}, {"dequant_weight"}, {opType},
            status, failures, "medium"));
      }

      // 3. Layout conversion: layout_transform + same op, when backend supports
      //    layout transforms and there's a kernel for this op with a different
      //    specific layout than the effective layout.
      if (supportsLayout) {
        bool altLayoutKernelExists = false;
        for (const auto &ke : lib) {
          if (ke.op_type != opType) continue;
          if (ke.supported_layouts.empty()) continue;
          bool dtypeOk = ke.supported_dtypes.empty()
                         || inList(ke.supported_dtypes, effectiveDtype);
          if (!dtypeOk) continue;
          if (!inList(ke.supported_layouts, effectiveLayout)) {
            altLayoutKernelExists = true;
            break;
          }
        }
        if (altLayoutKernelExists) {
          candidates.push_back(buildCandidate(
              ctx, "alt_" + std::to_string(altCounter++),
              "layout_conversion", opType,
              {"layout_transform", opType}, {"layout_transform"}, {opType},
              "valid", "", "low"));
        }
      }

      // 4. Cast conversion: cast + same op, when backend supports cast and a
      //    cast boundary is required.
      if (supportsCast && castRequired) {
        bool castKernelExists = hasKernel(lib, opType, effectiveDtype);
        std::string status = castKernelExists ? "valid" : "invalid_missing_kernel";
        std::string failures = castKernelExists
                               ? ""
                               : "missing_" + effectiveDtype + "_" + opType + "_kernel";
        candidates.push_back(buildCandidate(
            ctx, "alt_" + std::to_string(altCounter++),
            "cast_conversion", opType,
            {"cast", opType}, {"cast"}, {opType},
            status, failures, "low"));
      }

      // 5. Backend fallback: always a last-resort when a fallback backend is declared.
      if (!fallbackBackend.empty()) {
        candidates.push_back(buildCandidate(
            ctx, "alt_" + std::to_string(altCounter++),
            "backend_fallback", opType,
            {opType}, {}, {},
            "valid", "", "last_resort",
            "fallback_backend_" + fallbackBackend));
      }

      // available = true iff at least one candidate is valid.
      bool available = false;
      for (auto &c : candidates) {
        if (auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(c))
          if (auto vs = dict.get("validation_status"))
            if (auto s = mlir::dyn_cast<mlir::StringAttr>(vs))
              if (s.getValue() == "valid") { available = true; break; }
      }

      std::string reason = candidates.empty()
                           ? "no_alternatives_found"
                           : (available ? "alternatives_generated" : "all_alternatives_invalid");

      op.setAttr("alternative.available",      mlir::BoolAttr::get(ctx, available));
      op.setAttr("alternative.best_candidate", mlir::StringAttr::get(ctx, ""));
      op.setAttr("alternative.candidates",
                 mlir::ArrayAttr::get(ctx, candidates));
      op.setAttr("alternative.reason",         mlir::StringAttr::get(ctx, reason));
      op.setAttr("alternative.truth_boundary", mlir::StringAttr::get(ctx, kTruth));
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createAlternativeLoweringPlanningPass() {
  return std::make_unique<AlternativeLoweringPlanningPass>();
}

} // namespace mlir::hir
