#include "FusionPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#define GEN_PASS_DEF_KERNELAVAILABILITYPLANNING
#include "FusionPasses.h.inc"

namespace mlir::hir {
namespace {

static constexpr StringLiteral kTruth =
    "kernel_availability_static_library_model_not_runtime_verified";

// Minimal in-pass view of one kernel library entry.
struct KernelEntry {
  std::string op_type;
  std::string kernel_name;
  std::string kernel_library;
  std::vector<std::string> supported_dtypes;
  std::vector<std::string> supported_layouts;
  std::vector<std::string> supported_quant_modes;
  std::vector<std::string> rewrite_patterns;
  std::string fallback_kernel;
  std::string fallback_backend;
  std::string truth_boundary;
};

static KernelEntry parseDictEntry(mlir::DictionaryAttr dict) {
  KernelEntry ke;
  auto rStr = [&](llvm::StringRef k) -> std::string {
    if (auto a = dict.get(k))
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(a)) return s.getValue().str();
    return {};
  };
  auto rStrs = [&](llvm::StringRef k) -> std::vector<std::string> {
    std::vector<std::string> r;
    if (auto a = dict.get(k))
      if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(a))
        for (auto e : arr)
          if (auto s = mlir::dyn_cast<mlir::StringAttr>(e))
            r.push_back(s.getValue().str());
    return r;
  };
  ke.op_type              = rStr("op_type");
  ke.kernel_name          = rStr("kernel_name");
  ke.kernel_library       = rStr("kernel_library");
  ke.supported_dtypes     = rStrs("supported_dtypes");
  ke.supported_layouts    = rStrs("supported_layouts");
  ke.supported_quant_modes = rStrs("supported_quant_modes");
  ke.rewrite_patterns     = rStrs("rewrite_patterns");
  ke.fallback_kernel      = rStr("fallback_kernel");
  ke.fallback_backend     = rStr("fallback_backend");
  ke.truth_boundary       = rStr("truth_boundary");
  return ke;
}

static bool inList(const std::vector<std::string> &list, const std::string &val) {
  return std::find(list.begin(), list.end(), val) != list.end();
}

static void annotateOp(mlir::Operation &op, llvm::StringRef backend,
                       llvm::StringRef library, llvm::StringRef name,
                       bool exists, llvm::StringRef loweringStatus,
                       llvm::StringRef requiredRewrite,
                       llvm::StringRef fallbackBackend,
                       llvm::StringRef decisionReason,
                       llvm::StringRef truthBoundary,
                       mlir::MLIRContext *ctx) {
  auto S = [&](llvm::StringRef s) { return mlir::StringAttr::get(ctx, s); };
  auto B = [&](bool b)            { return mlir::BoolAttr::get(ctx, b); };
  llvm::StringRef tb = truthBoundary.empty() ? llvm::StringRef(kTruth) : truthBoundary;
  op.setAttr("kernel.backend",          S(backend));
  op.setAttr("kernel.decision_reason",  S(decisionReason));
  op.setAttr("kernel.exists",           B(exists));
  op.setAttr("kernel.fallback_backend", S(fallbackBackend));
  op.setAttr("kernel.library",          S(library));
  op.setAttr("kernel.lowering_status",  S(loweringStatus));
  op.setAttr("kernel.name",             S(name));
  op.setAttr("kernel.required_rewrite", S(requiredRewrite));
  op.setAttr("kernel.truth_boundary",   S(tb));
}

struct KernelAvailabilityPlanningPass
    : ::impl::KernelAvailabilityPlanningBase<KernelAvailabilityPlanningPass> {
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::ModuleOp module = func->getParentOfType<mlir::ModuleOp>();
    if (!module) return;

    // Get primary backend from representation or execution_provider pass attrs.
    std::string backend;
    if (auto a = func->getAttrOfType<mlir::StringAttr>("representation.source_backend"))
      backend = a.getValue().str();
    else if (auto a = func->getAttrOfType<mlir::StringAttr>("execution_provider.primary"))
      backend = a.getValue().str();
    if (backend.empty()) return;

    // Load kernel library entries for this backend.
    std::vector<KernelEntry> kernels;
    if (auto arr = module->getAttrOfType<mlir::ArrayAttr>(
            "target.kernel_libraries." + backend)) {
      for (auto elem : arr)
        if (auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(elem))
          kernels.push_back(parseDictEntry(dict));
    }
    if (kernels.empty()) return; // No kernel library declared for this backend.

    mlir::MLIRContext *ctx = &getContext();
    if (func.getBody().empty()) return;
    mlir::Block &entry = func.getBody().front();

    for (mlir::Operation &op : entry.without_terminator()) {
      // Short op name: "hir.matmul" → "matmul".
      llvm::StringRef fullName = op.getName().getStringRef();
      std::string shortName = fullName.str();
      if (auto dot = fullName.find('.'); dot != std::string::npos)
        shortName = fullName.substr(dot + 1).str();

      // Effective dtype: quant.activation_dtype → representation.effective_dtype → "".
      std::string effectiveDtype;
      if (auto a = op.getAttrOfType<mlir::StringAttr>("quant.activation_dtype"))
        effectiveDtype = a.getValue().str();
      if (effectiveDtype.empty())
        if (auto a = func->getAttrOfType<mlir::StringAttr>("representation.effective_dtype"))
          effectiveDtype = a.getValue().str();

      // Effective layout: layout.effective_layout → "".
      std::string effectiveLayout;
      if (auto a = op.getAttrOfType<mlir::StringAttr>("layout.effective_layout"))
        effectiveLayout = a.getValue().str();

      // Quant mode: quant.activation_quant_mode → "none".
      std::string quantMode = "none";
      if (auto a = op.getAttrOfType<mlir::StringAttr>("quant.activation_quant_mode"))
        quantMode = a.getValue().str();

      // Filter kernel entries to those matching this op type.
      std::vector<const KernelEntry *> forThisOp;
      for (const auto &ke : kernels)
        if (ke.op_type == shortName) forThisOp.push_back(&ke);

      if (forThisOp.empty()) {
        // Rule 5: op type not in library → unsupported.
        annotateOp(op, backend, "", "", false, "unsupported", "", "",
                   "no_kernel_for_op_type", kTruth, ctx);
        continue;
      }

      // Rule 1: exact match (empty list matches any value).
      const KernelEntry *exact = nullptr;
      for (const auto *ke : forThisOp) {
        bool dtypeOk  = ke->supported_dtypes.empty()      || inList(ke->supported_dtypes,      effectiveDtype);
        bool layoutOk = ke->supported_layouts.empty()     || inList(ke->supported_layouts,     effectiveLayout);
        bool quantOk  = ke->supported_quant_modes.empty() || inList(ke->supported_quant_modes, quantMode);
        if (dtypeOk && layoutOk && quantOk) { exact = ke; break; }
      }
      if (exact) {
        annotateOp(op, backend, exact->kernel_library, exact->kernel_name,
                   true, "lowerable", "", exact->fallback_backend,
                   "exact_kernel_match", exact->truth_boundary, ctx);
        continue;
      }

      // Rule 2: rewrite_patterns declared → rewrite_candidate.
      const KernelEntry *rewriteKe = nullptr;
      for (const auto *ke : forThisOp)
        if (!ke->rewrite_patterns.empty()) { rewriteKe = ke; break; }
      if (rewriteKe) {
        annotateOp(op, backend, rewriteKe->kernel_library, rewriteKe->kernel_name,
                   false, "rewrite_candidate", rewriteKe->rewrite_patterns[0],
                   rewriteKe->fallback_backend,
                   "rewrite_pattern_available", rewriteKe->truth_boundary, ctx);
        continue;
      }

      // Rule 3: fallback_backend declared → fallback_required.
      const KernelEntry *fallbackKe = nullptr;
      for (const auto *ke : forThisOp)
        if (!ke->fallback_backend.empty()) { fallbackKe = ke; break; }
      if (fallbackKe) {
        annotateOp(op, backend, "", fallbackKe->kernel_name, false,
                   "fallback_required", "", fallbackKe->fallback_backend,
                   "no_exact_kernel_fallback_to_" + fallbackKe->fallback_backend,
                   fallbackKe->truth_boundary, ctx);
        continue;
      }

      // Rule 4: op type found, no path → unsupported.
      annotateOp(op, backend, "", forThisOp[0]->kernel_name, false,
                 "unsupported", "", "",
                 "kernel_exists_wrong_dtype_or_layout",
                 forThisOp[0]->truth_boundary, ctx);
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createKernelAvailabilityPlanningPass() {
  return std::make_unique<KernelAvailabilityPlanningPass>();
}

} // namespace mlir::hir
