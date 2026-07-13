// KernelSelectionPass — compiler-side kernel selection against concrete
// runtime kernel descriptors (kernel_selection_contract_v1).
//
// This is a different question from KernelAvailabilityPlanningPass: that
// pass answers "does the declared THIRD-PARTY LIBRARY have coverage for
// this (op, dtype, layout, quant) tuple" (cuBLAS/Triton/CoreML public-API
// coverage); this pass answers "which CONCRETE runtime kernel descriptor —
// a kernel someone can actually dispatch — is the contract for this op".
// A kernel is selected ONLY when a RuntimeKernelDescriptor exists in
// target.runtime_kernels and matches the planned op name, backend, dtype,
// quant mode, layout, tile plan, and memory hierarchy. Everything else is
// rejected or deferred with an explicit reason. The registry is expected
// to be small and honest — no coverage is inferred or invented.
//
// Emitted per non-terminator op (kernel_selection.* namespace — distinct
// from the availability pass's kernel.* attrs and from the literal
// "kernel.selection" string attr on hir fused ops):
//   kernel_selection.status           -- see FusionPasses.td for the values
//   kernel_selection.selected_id      -- when status == "selected"
//   kernel_selection.source           -- descriptor source, when selected
//   kernel_selection.rejection_reasons-- per-descriptor reasons, when not
//   kernel_selection.contract_version -- "kernel_selection_contract_v1"
//   kernel_selection.truth_boundary
//
// Truth boundary: static descriptor matching only. The compiler does not
// execute, dispatch, or benchmark kernels; a selection is a contract for
// the runtime, not an execution claim.

#include "serving/OpShapeFacts.h"
#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_KERNELSELECTION
#include "FusionPasses.h.inc"

static constexpr StringLiteral kContractVersion = "kernel_selection_contract_v1";
static constexpr StringLiteral kTruth =
    "kernel_selection_static_descriptor_match_not_runtime_execution";

// One declared thread-decomposition option (Phase P1D,
// thread_schedule_contract_v1).
struct ThreadScheduleOption {
  int64_t thread_count = 1;
  std::string partition_axis;
  std::string partition_strategy;
};

// Minimal in-pass view of one RuntimeKernelDescriptor dict.
struct DescriptorView {
  std::string kernel_id;
  std::string op_name;
  std::string backend;
  std::vector<std::string> supported_dtypes;
  std::vector<std::string> supported_quant_modes;
  std::vector<std::string> supported_layouts;
  std::vector<std::string> supported_tile_shapes;
  std::vector<ThreadScheduleOption> supported_thread_schedules;
  bool requires_static_shape = true;
  int64_t requires_local_memory_bytes = 0;
  std::string source;
  std::string truth_boundary;
};

static DescriptorView parseDescriptor(DictionaryAttr dict) {
  DescriptorView d;
  auto rStr = [&](StringRef k) -> std::string {
    if (auto a = dict.get(k))
      if (auto s = dyn_cast<StringAttr>(a)) return s.getValue().str();
    return {};
  };
  auto rStrs = [&](StringRef k) -> std::vector<std::string> {
    std::vector<std::string> r;
    if (auto a = dict.get(k))
      if (auto arr = dyn_cast<ArrayAttr>(a))
        for (auto e : arr)
          if (auto s = dyn_cast<StringAttr>(e)) r.push_back(s.getValue().str());
    return r;
  };
  d.kernel_id             = rStr("kernel_id");
  d.op_name               = rStr("op_name");
  d.backend               = rStr("backend");
  d.supported_dtypes      = rStrs("supported_dtypes");
  d.supported_quant_modes = rStrs("supported_quant_modes");
  d.supported_layouts     = rStrs("supported_layouts");
  d.supported_tile_shapes = rStrs("supported_tile_shapes");
  if (auto a = dict.get("supported_thread_schedules"))
    if (auto arr = dyn_cast<ArrayAttr>(a))
      for (auto e : arr)
        if (auto tsDict = dyn_cast<DictionaryAttr>(e)) {
          ThreadScheduleOption ts;
          if (auto tc = tsDict.get("thread_count"))
            if (auto ia = dyn_cast<IntegerAttr>(tc)) ts.thread_count = ia.getInt();
          if (auto pa = tsDict.get("partition_axis"))
            if (auto s = dyn_cast<StringAttr>(pa)) ts.partition_axis = s.getValue().str();
          if (auto ps = tsDict.get("partition_strategy"))
            if (auto s = dyn_cast<StringAttr>(ps)) ts.partition_strategy = s.getValue().str();
          d.supported_thread_schedules.push_back(std::move(ts));
        }
  if (auto a = dict.get("requires_static_shape"))
    if (auto b = dyn_cast<BoolAttr>(a)) d.requires_static_shape = b.getValue();
  if (auto a = dict.get("requires_local_memory_bytes"))
    if (auto ia = dyn_cast<IntegerAttr>(a))
      d.requires_local_memory_bytes = ia.getInt();
  d.source         = rStr("source");
  d.truth_boundary = rStr("truth_boundary");
  return d;
}

static bool inList(const std::vector<std::string>& list,
                   const std::string& val) {
  for (const auto& e : list)
    if (e == val) return true;
  return false;
}

// Planned context of one op, gathered from existing planning attrs.
struct OpContext {
  std::string op_name;   // short name, e.g. "rmsnorm"
  std::string backend;   // func-level selected backend
  std::string dtype;     // resolved activation dtype
  std::string quant_mode; // "none" | "weight_only" | "static_int8"
  std::string layout;    // layout.effective_layout, "" when absent
  bool has_dynamic_shape = false;
  bool has_tile_plan = false;      // tile.plan.status == "planned"
  std::string tile_shape;          // "MxNxK" when planned
  int64_t declared_local_memory = 0; // 0 = not declared
};

// Result of matching one descriptor against one op context.
// ok == true means full match; otherwise `reason` names the first failed
// check and `deferred` says whether it is a deferral (missing information)
// rather than a rejection (contradicting information).
struct MatchResult {
  bool ok = false;
  bool deferred = false;
  std::string reason;
};

static MatchResult matchDescriptor(const DescriptorView& d,
                                   const OpContext& op) {
  MatchResult r;
  if (d.backend != op.backend) {
    r.reason = "backend_mismatch";
    return r;
  }
  if (!d.supported_dtypes.empty() && !inList(d.supported_dtypes, op.dtype)) {
    r.reason = "dtype_unsupported";
    return r;
  }
  if (!d.supported_quant_modes.empty() &&
      !inList(d.supported_quant_modes, op.quant_mode)) {
    r.reason = "quant_mode_unsupported";
    return r;
  }
  // Layout: checked only when both sides state one. An op without a layout
  // decision is unconstrained; a layout-agnostic descriptor accepts any.
  if (!d.supported_layouts.empty() && !op.layout.empty() &&
      !inList(d.supported_layouts, op.layout)) {
    r.reason = "layout_unsupported";
    return r;
  }
  if (d.requires_static_shape && op.has_dynamic_shape) {
    r.deferred = true;
    r.reason = "dynamic_shape";
    return r;
  }
  if (!d.supported_tile_shapes.empty()) {
    if (!op.has_tile_plan) {
      r.deferred = true;
      r.reason = "missing_tile_plan";
      return r;
    }
    if (!inList(d.supported_tile_shapes, op.tile_shape)) {
      r.reason = "tile_shape_unsupported";
      return r;
    }
  }
  if (d.requires_local_memory_bytes > 0) {
    if (op.declared_local_memory <= 0) {
      r.deferred = true;
      r.reason = "missing_memory_hierarchy";
      return r;
    }
    if (op.declared_local_memory < d.requires_local_memory_bytes) {
      r.reason = "local_memory_insufficient";
      return r;
    }
  }
  r.ok = true;
  return r;
}

// Map a per-descriptor failure reason to the op-level status string.
static std::string statusFor(const MatchResult& m) {
  if (m.deferred) return "deferred_" + m.reason;
  return "rejected_" + m.reason;
}

// Thread-schedule resolution (Phase P1D, thread_schedule_contract_v1),
// resolved AFTER kernel selection, for the already-selected kernel only.
// This is a SEPARATE decision from kernel_selection: which kernel/tile
// runs vs. how many threads and what partitioning it uses. Declaration
// order is preference order (same convention as kernel selection itself):
// the first declared schedule whose thread_count fits the profile's
// verified available compute units wins. physicalComputeUnits absent
// means one-thread-only eligibility -- never invented parallel capacity.
struct ThreadScheduleResult {
  std::string status;   // "selected" | "rejected_*" | "deferred_*"
  const ThreadScheduleOption* option = nullptr;
  std::vector<std::string> rejection_reasons;
};

static ThreadScheduleResult resolveThreadSchedule(
    const DescriptorView& selected,
    std::optional<int64_t> physicalComputeUnits) {
  ThreadScheduleResult r;
  if (selected.supported_thread_schedules.empty()) {
    // No thread schedules declared for this kernel at all -- absence is
    // never invented; leave completely absent from the plan (handled by
    // the caller, which skips emitting thread_schedule.* attrs entirely).
    r.status = "";
    return r;
  }
  for (const ThreadScheduleOption& ts : selected.supported_thread_schedules) {
    if (ts.thread_count <= 1) {
      r.status = "selected";
      r.option = &ts;
      return r;
    }
    if (!physicalComputeUnits.has_value()) {
      r.rejection_reasons.push_back(
          ts.partition_axis + "_threads_" + std::to_string(ts.thread_count) +
          ":deferred_missing_compute_units");
      continue;
    }
    if (ts.thread_count > *physicalComputeUnits) {
      r.rejection_reasons.push_back(
          ts.partition_axis + "_threads_" + std::to_string(ts.thread_count) +
          ":rejected_exceeds_compute_units");
      continue;
    }
    r.status = "selected";
    r.option = &ts;
    return r;
  }
  // Nothing fit. Honest deferral if it was purely a missing-profile-fact
  // issue for every candidate; otherwise an explicit rejection.
  bool anyDeferred = false;
  for (const auto& reason : r.rejection_reasons)
    if (reason.find("deferred_missing_compute_units") != std::string::npos)
      anyDeferred = true;
  r.status = anyDeferred ? "deferred_missing_compute_units"
                         : "rejected_exceeds_compute_units";
  return r;
}

struct KernelSelectionPass : impl::KernelSelectionBase<KernelSelectionPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext* ctx = funcOp.getContext();
    if (funcOp.getBody().empty()) return;
    Operation* module = funcOp->getParentOp();

    auto S = [&](StringRef s) { return StringAttr::get(ctx, s); };

    // Load the runtime kernel registry. An absent/empty registry is a
    // recorded deferral on every op — never a silent no-op.
    std::vector<DescriptorView> registry;
    bool registryDeclared = false;
    if (module)
      if (auto arr = module->getAttrOfType<ArrayAttr>("target.runtime_kernels")) {
        registryDeclared = true;
        for (auto e : arr)
          if (auto dict = dyn_cast<DictionaryAttr>(e))
            registry.push_back(parseDescriptor(dict));
      }

    // Func-level planned backend (same chain KernelAvailability uses).
    std::string backend;
    if (auto a = funcOp->getAttrOfType<StringAttr>("representation.source_backend"))
      backend = a.getValue().str();
    else if (auto a = funcOp->getAttrOfType<StringAttr>("execution_provider.primary"))
      backend = a.getValue().str();

    std::string effectiveDtype;
    if (auto a =
            funcOp->getAttrOfType<StringAttr>("representation.effective_dtype"))
      effectiveDtype = a.getValue().str();

    int64_t declaredLocalMemory =
        readProfileNums(module).memory_hierarchy.local_memory_bytes;

    // Phase P1D: the profile's verified available compute units, read
    // directly from the module attr TargetConstraints::attachToModule
    // already lowers (target.hardware.physical_compute_units). This is
    // the first pass to actually CONSUME that attr for a decision --
    // previously round-tripped but unused by any pass (confirmed by the
    // P1B/P1C audits). Absence means one-thread-only eligibility below,
    // never invented parallel capacity.
    std::optional<int64_t> physicalComputeUnits;
    if (module)
      if (auto a = module->getAttrOfType<IntegerAttr>(
              "target.hardware.physical_compute_units"))
        physicalComputeUnits = a.getInt();

    for (Operation& op : funcOp.getBody().front().without_terminator()) {
      op.setAttr("kernel_selection.contract_version", S(kContractVersion));
      op.setAttr("kernel_selection.truth_boundary", S(kTruth));

      if (!registryDeclared || registry.empty()) {
        op.setAttr("kernel_selection.status",
                   S("deferred_no_kernel_library_declared"));
        continue;
      }

      // Gather this op's planned context from existing planning attrs.
      OpContext opCtx;
      StringRef fullName = op.getName().getStringRef();
      opCtx.op_name = fullName.str();
      if (auto dot = fullName.find('.'); dot != StringRef::npos)
        opCtx.op_name = fullName.substr(dot + 1).str();
      opCtx.backend = backend;
      opCtx.dtype = resolveOpActivationDtype(op, effectiveDtype);
      opCtx.quant_mode = "none";
      if (auto a = op.getAttrOfType<StringAttr>("quant.strategy")) {
        StringRef strat = a.getValue();
        if (strat == "weight_only_int8")     opCtx.quant_mode = "weight_only";
        else if (strat == "static_int8")     opCtx.quant_mode = "static_int8";
      }
      if (auto a = op.getAttrOfType<StringAttr>("layout.effective_layout"))
        opCtx.layout = a.getValue().str();
      if (auto st = op.getAttrOfType<StringAttr>("tile.plan.status");
          st && st.getValue() == "planned") {
        if (auto shape = op.getAttrOfType<ArrayAttr>("tile.plan.shape");
            shape && shape.size() == 3) {
          opCtx.has_tile_plan = true;
          std::string s;
          for (unsigned i = 0; i < 3; ++i) {
            if (auto ia = dyn_cast<IntegerAttr>(shape[i]))
              s += std::to_string(ia.getInt());
            if (i < 2) s += "x";
          }
          opCtx.tile_shape = s;
        }
      }
      opCtx.declared_local_memory = declaredLocalMemory;
      for (Type t : op.getResultTypes())
        if (auto stTy = dyn_cast<ShapedType>(t))
          if (!stTy.hasStaticShape()) { opCtx.has_dynamic_shape = true; break; }
      if (!opCtx.has_dynamic_shape)
        for (Type t : op.getOperandTypes())
          if (auto stTy = dyn_cast<ShapedType>(t))
            if (!stTy.hasStaticShape()) { opCtx.has_dynamic_shape = true; break; }

      // Match against every descriptor for this op name; first full match
      // wins (registry order is declaration order). All failures are
      // collected as per-descriptor reasons.
      const DescriptorView* selected = nullptr;
      MatchResult firstFailure;
      bool sawOpNameMatch = false;
      SmallVector<Attribute> reasons;
      for (const DescriptorView& d : registry) {
        if (d.op_name != opCtx.op_name) continue;
        sawOpNameMatch = true;
        MatchResult m = matchDescriptor(d, opCtx);
        if (m.ok) { selected = &d; break; }
        if (firstFailure.reason.empty()) firstFailure = m;
        reasons.push_back(S(d.kernel_id + ":" + m.reason));
      }

      if (selected) {
        op.setAttr("kernel_selection.status", S("selected"));
        op.setAttr("kernel_selection.selected_id", S(selected->kernel_id));
        op.setAttr("kernel_selection.source", S(selected->source));
        if (!selected->truth_boundary.empty())
          op.setAttr("kernel_selection.truth_boundary",
                     S(selected->truth_boundary));

        // Phase P1D: resolve a thread-decomposition schedule for the
        // already-selected kernel, a SEPARATE decision from which
        // kernel/tile runs. Absent entirely (no attrs set) when this
        // kernel declares no thread schedules at all -- old P1B/P1C
        // profiles stay byte-identical.
        ThreadScheduleResult tsResult =
            resolveThreadSchedule(*selected, physicalComputeUnits);
        if (!tsResult.status.empty()) {
          op.setAttr("thread_schedule.contract_version",
                     S("thread_schedule_contract_v1"));
          op.setAttr("thread_schedule.truth_boundary",
                     S("thread_schedule_static_descriptor_match_not_runtime_execution"));
          op.setAttr("thread_schedule.status", S(tsResult.status));
          if (tsResult.option) {
            op.setAttr("thread_schedule.thread_count",
                       IntegerAttr::get(IntegerType::get(ctx, 64),
                                        tsResult.option->thread_count));
            op.setAttr("thread_schedule.partition_axis",
                       S(tsResult.option->partition_axis));
            op.setAttr("thread_schedule.partition_strategy",
                       S(tsResult.option->partition_strategy));
            op.setAttr("thread_schedule.source", S(selected->source));
          }
          if (!tsResult.rejection_reasons.empty()) {
            SmallVector<Attribute> tsReasons;
            for (const auto& reason : tsResult.rejection_reasons)
              tsReasons.push_back(S(reason));
            op.setAttr("thread_schedule.rejection_reasons",
                       ArrayAttr::get(ctx, tsReasons));
          }
        }
        continue;
      }

      if (!sawOpNameMatch) {
        op.setAttr("kernel_selection.status", S("rejected_no_kernel_for_op"));
        continue;
      }
      // Status reflects the first op-name-matching descriptor's first failed
      // check (deterministic); all per-descriptor reasons are recorded.
      op.setAttr("kernel_selection.status", S(statusFor(firstFailure)));
      op.setAttr("kernel_selection.rejection_reasons",
                 ArrayAttr::get(ctx, reasons));
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createKernelSelectionPass() {
  return std::make_unique<KernelSelectionPass>();
}

} // namespace mlir::hir
