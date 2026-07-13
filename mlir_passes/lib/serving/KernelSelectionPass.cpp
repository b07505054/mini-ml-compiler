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
#include "serving/ImplementationCandidate.h"
#include "serving/PortableCPUProvider.h"
#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
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

// Thread-schedule resolution (Phase P1D/P1D.1, thread_schedule_contract_v1),
// resolved AFTER kernel selection, for the already-selected kernel only.
// P1D default behavior is preserved when no validated offline policy exists:
// declaration order remains preference order. P1D.1 policy may only choose
// between already-declared legal schedules; it never creates a schedule and
// never uses measurement to override kernel/op/dtype/shape legality.
struct ThreadScheduleResult {
  std::string status;   // "selected" | "rejected_*" | "deferred_*"
  const ThreadScheduleOption* option = nullptr;
  std::vector<std::string> rejection_reasons;
  std::string policy_id;
  std::string policy_version;
  std::string policy_metric;
  int64_t policy_metric_value = 0;
  int64_t policy_threshold = 0;
  std::string policy_boundary_rule;
  std::string policy_selection_reason;
  std::string policy_evidence_ref;
  std::string policy_evidence_sha256;
  std::string policy_truth_boundary;
  std::string selected_candidate_id;
  ImplementationCandidate selected_candidate;
  bool has_selected_candidate = false;
  std::vector<std::string> considered_candidate_ids;
  std::vector<PolicyResultCandidateRejection> candidate_rejections;
};

struct OfflineThreadPolicyView {
  bool active = false;
  std::string policy_id;
  std::string policy_version;
  std::string target_profile_id;
  std::string fused_region_identity;
  std::string dtype;
  std::string kernel_id;
  std::string metric;
  int64_t threshold = 0;
  std::string boundary_rule;
  ThreadScheduleOption below_threshold;
  ThreadScheduleOption at_or_above_threshold;
  std::string calibration_evidence_ref;
  std::string evidence_sha256;
  std::string truth_boundary;
};

static std::string moduleStringAttr(Operation* module, StringRef key) {
  if (!module) return {};
  if (auto a = module->getAttrOfType<StringAttr>(key))
    return a.getValue().str();
  return {};
}

static std::optional<int64_t> moduleI64Attr(Operation* module, StringRef key) {
  if (!module) return std::nullopt;
  if (auto a = module->getAttrOfType<IntegerAttr>(key))
    return a.getInt();
  return std::nullopt;
}

static ThreadScheduleOption readPolicySchedule(Operation* module,
                                               StringRef prefix) {
  ThreadScheduleOption ts;
  if (auto v = moduleI64Attr(module, (prefix + ".thread_count").str()))
    ts.thread_count = *v;
  ts.partition_axis = moduleStringAttr(module, (prefix + ".partition_axis").str());
  ts.partition_strategy = moduleStringAttr(module, (prefix + ".partition_strategy").str());
  return ts;
}

static OfflineThreadPolicyView readOfflineThreadPolicy(Operation* module) {
  OfflineThreadPolicyView p;
  p.policy_id = moduleStringAttr(module, "target.thread_schedule_policy.policy_id");
  if (p.policy_id.empty()) return p;
  p.active = true;
  p.policy_version = moduleStringAttr(module, "target.thread_schedule_policy.policy_version");
  p.target_profile_id = moduleStringAttr(module, "target.thread_schedule_policy.target_profile_id");
  p.fused_region_identity = moduleStringAttr(module, "target.thread_schedule_policy.fused_region_identity");
  p.dtype = moduleStringAttr(module, "target.thread_schedule_policy.dtype");
  p.kernel_id = moduleStringAttr(module, "target.thread_schedule_policy.kernel_id");
  p.metric = moduleStringAttr(module, "target.thread_schedule_policy.metric");
  p.threshold = moduleI64Attr(module, "target.thread_schedule_policy.threshold").value_or(0);
  p.boundary_rule = moduleStringAttr(module, "target.thread_schedule_policy.boundary_rule");
  p.below_threshold = readPolicySchedule(module, "target.thread_schedule_policy.below_threshold");
  p.at_or_above_threshold = readPolicySchedule(module, "target.thread_schedule_policy.at_or_above_threshold");
  p.calibration_evidence_ref = moduleStringAttr(module, "target.thread_schedule_policy.calibration_evidence_ref");
  p.evidence_sha256 = moduleStringAttr(module, "target.thread_schedule_policy.evidence_sha256");
  p.truth_boundary = moduleStringAttr(module, "target.thread_schedule_policy.truth_boundary");
  return p;
}

static bool sameSchedule(const ThreadScheduleOption& a,
                         const ThreadScheduleOption& b) {
  return a.thread_count == b.thread_count &&
         a.partition_axis == b.partition_axis &&
         a.partition_strategy == b.partition_strategy;
}

static const ThreadScheduleOption*
findDeclaredSchedule(const DescriptorView& selected,
                     const ThreadScheduleOption& wanted) {
  for (const ThreadScheduleOption& ts : selected.supported_thread_schedules)
    if (sameSchedule(ts, wanted)) return &ts;
  return nullptr;
}

static bool eligibleForComputeUnits(const ThreadScheduleOption& ts,
                                    std::optional<int64_t> physicalComputeUnits) {
  if (ts.thread_count <= 1) return true;
  return physicalComputeUnits.has_value() && ts.thread_count <= *physicalComputeUnits;
}

static bool safeMul3(int64_t a, int64_t b, int64_t c, int64_t& out) {
  if (a <= 0 || b <= 0 || c <= 0) return false;
  __int128 product = static_cast<__int128>(a) * b * c;
  if (product > std::numeric_limits<int64_t>::max()) return false;
  out = static_cast<int64_t>(product);
  return true;
}

static bool hasCandidateIdCollision(
    const std::vector<PortableCpuCandidateView>& candidates) {
  for (size_t i = 0; i < candidates.size(); ++i)
    for (size_t j = i + 1; j < candidates.size(); ++j)
      if (candidates[i].candidate.candidateId ==
          candidates[j].candidate.candidateId)
        return true;
  return false;
}

static PortableCpuRuntimeKernelDescriptor
toPortableCpuDescriptor(const DescriptorView& selected) {
  PortableCpuRuntimeKernelDescriptor descriptor;
  descriptor.kernelId = selected.kernel_id;
  descriptor.opName = selected.op_name;
  descriptor.backend = selected.backend;
  descriptor.supportedDtypes = selected.supported_dtypes;
  descriptor.supportedTileShapes = selected.supported_tile_shapes;
  descriptor.truthBoundary = selected.truth_boundary;
  for (const auto& schedule : selected.supported_thread_schedules) {
    PortableCpuThreadSchedule providerSchedule;
    providerSchedule.threadCount = schedule.thread_count;
    providerSchedule.partitionAxis = schedule.partition_axis;
    providerSchedule.partitionStrategy = schedule.partition_strategy;
    descriptor.supportedThreadSchedules.push_back(
        std::move(providerSchedule));
  }
  return descriptor;
}

static PortableCpuProviderContext
toPortableCpuProviderContext(const OpContext& opCtx,
                             StringRef targetProfileId) {
  PortableCpuProviderContext ctx;
  ctx.semanticTargetRef = opCtx.op_name;
  ctx.scopeKind = CandidateScopeKind::FusedRegion;
  ctx.targetProfileId = targetProfileId.str();
  ctx.backend = opCtx.backend;
  ctx.dtype = opCtx.dtype;
  ctx.truthBoundary = kTruth.str();
  return ctx;
}

static ThreadScheduleResult resolveThreadScheduleStatic(
    const DescriptorView& selected,
    std::optional<int64_t> physicalComputeUnits) {
  ThreadScheduleResult r;
  if (selected.supported_thread_schedules.empty()) {
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
  bool anyDeferred = false;
  for (const auto& reason : r.rejection_reasons)
    if (reason.find("deferred_missing_compute_units") != std::string::npos)
      anyDeferred = true;
  r.status = anyDeferred ? "deferred_missing_compute_units"
                         : "rejected_exceeds_compute_units";
  return r;
}

static ThreadScheduleResult resolveThreadSchedule(
    const DescriptorView& selected,
    std::optional<int64_t> physicalComputeUnits,
    const OfflineThreadPolicyView& policy,
    const OpContext& opCtx,
    Operation& op,
    StringRef targetProfileId) {
  if (selected.supported_thread_schedules.empty()) {
    ThreadScheduleResult r;
    r.status = "";
    return r;
  }
  if (!policy.active)
    return resolveThreadScheduleStatic(selected, physicalComputeUnits);

  ThreadScheduleResult r;
  r.policy_id = policy.policy_id;
  r.policy_version = policy.policy_version;
  r.policy_metric = policy.metric;
  r.policy_threshold = policy.threshold;
  r.policy_boundary_rule = policy.boundary_rule;
  r.policy_evidence_ref = policy.calibration_evidence_ref;
  r.policy_evidence_sha256 = policy.evidence_sha256;
  r.policy_truth_boundary = policy.truth_boundary;

  const ThreadScheduleOption* serial =
      findDeclaredSchedule(selected, policy.below_threshold);
  const ThreadScheduleOption* parallel =
      findDeclaredSchedule(selected, policy.at_or_above_threshold);

  if (serial && parallel && sameSchedule(*serial, *parallel)) {
    ThreadScheduleResult collision;
    collision.policy_id = policy.policy_id;
    collision.policy_version = policy.policy_version;
    collision.policy_metric = policy.metric;
    collision.policy_threshold = policy.threshold;
    collision.policy_boundary_rule = policy.boundary_rule;
    collision.policy_evidence_ref = policy.calibration_evidence_ref;
    collision.policy_evidence_sha256 = policy.evidence_sha256;
    collision.policy_truth_boundary = policy.truth_boundary;
    collision.status = "rejected_thread_schedule_candidate_id_collision";
    collision.rejection_reasons.push_back(
        "thread_schedule_candidate_id_collision");
    collision.policy_selection_reason = "candidate_identity_collision";
    return collision;
  }

  PortableCPUProvider provider;
  PortableCpuProviderResult providerResult = provider.enumerateCandidates(
      toPortableCpuProviderContext(opCtx, targetProfileId),
      toPortableCpuDescriptor(selected));
  std::vector<PortableCpuCandidateView> candidates =
      std::move(providerResult.candidates);
  for (const auto& candidate : candidates)
    r.considered_candidate_ids.push_back(candidate.candidate.candidateId);
  if (hasCandidateIdCollision(candidates)) {
    r.status = "rejected_thread_schedule_candidate_id_collision";
    r.rejection_reasons.push_back("thread_schedule_candidate_id_collision");
    r.policy_selection_reason = "candidate_identity_collision";
    return r;
  }

  auto findCandidateFor = [&](const ThreadScheduleOption* option)
      -> const PortableCpuCandidateView* {
    if (!option) return nullptr;
    for (const auto& candidate : candidates)
      if (candidate.schedule.threadCount == option->thread_count &&
          candidate.schedule.partitionAxis == option->partition_axis &&
          candidate.schedule.partitionStrategy == option->partition_strategy &&
          candidate.candidate.feasibility.status ==
              CandidateFeasibilityStatus::Feasible)
        return &candidate;
    return nullptr;
  };

  auto rejectCandidate = [&](const ThreadScheduleOption* option,
                             StringRef reason) {
    if (auto candidate = findCandidateFor(option)) {
      r.candidate_rejections.push_back(
          {candidate->candidate.candidateId, reason.str()});
    }
  };

  auto chooseSerial = [&](StringRef reason) {
    if (serial) {
      const PortableCpuCandidateView* selectedCandidate =
          findCandidateFor(serial);
      r.status = "selected";
      r.option = serial;
      if (selectedCandidate) {
        r.selected_candidate_id = selectedCandidate->candidate.candidateId;
        r.selected_candidate = selectedCandidate->candidate;
        r.has_selected_candidate = true;
      }
      r.policy_selection_reason = reason.str();
    } else {
      r.status = "rejected_missing_serial_thread_schedule";
      r.rejection_reasons.push_back("policy_serial_schedule_not_declared");
      r.policy_selection_reason = reason.str();
    }
  };

  if (!serial) {
    r.status = "rejected_missing_serial_thread_schedule";
    r.rejection_reasons.push_back("policy_serial_schedule_not_declared");
    r.policy_selection_reason = "policy_cannot_apply_without_declared_serial_fallback";
    return r;
  }

  if (policy.target_profile_id != targetProfileId) {
    rejectCandidate(parallel, "policy_target_profile_mismatch");
    chooseSerial("policy_target_profile_mismatch_serial_fallback");
    return r;
  }
  if (policy.fused_region_identity != opCtx.op_name) {
    rejectCandidate(parallel, "policy_fused_region_mismatch");
    chooseSerial("policy_fused_region_mismatch_serial_fallback");
    return r;
  }
  if (policy.dtype != opCtx.dtype) {
    rejectCandidate(parallel, "policy_dtype_mismatch");
    chooseSerial("policy_dtype_mismatch_serial_fallback");
    return r;
  }
  if (policy.kernel_id != selected.kernel_id) {
    rejectCandidate(parallel, "policy_kernel_mismatch");
    chooseSerial("policy_kernel_mismatch_serial_fallback");
    return r;
  }
  if (policy.metric != "matmul_mnk") {
    rejectCandidate(parallel, "policy_metric_unsupported");
    chooseSerial("policy_metric_unsupported_serial_fallback");
    return r;
  }
  if (policy.boundary_rule != "at_or_above_threshold_selects_parallel") {
    rejectCandidate(parallel, "policy_boundary_rule_unsupported");
    chooseSerial("policy_boundary_rule_unsupported_serial_fallback");
    return r;
  }
  ShapeFacts facts = computeShapeFacts(op);
  int64_t metric = 0;
  if (facts.status != "static_shapes" ||
      !safeMul3(facts.m, facts.n, facts.k, metric)) {
    rejectCandidate(parallel, "policy_missing_static_mnk");
    chooseSerial("policy_missing_static_mnk_serial_fallback");
    return r;
  }
  r.policy_metric_value = metric;
  if (metric < policy.threshold) {
    rejectCandidate(parallel, "metric_below_threshold");
    chooseSerial("metric_below_threshold_select_serial");
    return r;
  }
  if (!parallel) {
    r.rejection_reasons.push_back("policy_parallel_schedule_not_declared");
    chooseSerial("metric_at_or_above_threshold_but_parallel_not_declared_serial_fallback");
    return r;
  }
  if (!physicalComputeUnits.has_value()) {
    r.rejection_reasons.push_back("policy_parallel_deferred_missing_compute_units");
    rejectCandidate(parallel, "deferred_missing_compute_units");
    chooseSerial("metric_at_or_above_threshold_but_missing_compute_units_serial_fallback");
    return r;
  }
  if (!eligibleForComputeUnits(*parallel, physicalComputeUnits)) {
    r.rejection_reasons.push_back("policy_parallel_rejected_exceeds_compute_units");
    rejectCandidate(parallel, "rejected_exceeds_compute_units");
    chooseSerial("metric_at_or_above_threshold_but_compute_units_insufficient_serial_fallback");
    return r;
  }
  const PortableCpuCandidateView* selectedCandidate =
      findCandidateFor(parallel);
  r.status = "selected";
  r.option = parallel;
  if (selectedCandidate) {
    r.selected_candidate_id = selectedCandidate->candidate.candidateId;
    r.selected_candidate = selectedCandidate->candidate;
    r.has_selected_candidate = true;
  }
  rejectCandidate(serial, "metric_at_or_above_threshold");
  r.policy_selection_reason = "metric_at_or_above_threshold_select_parallel";
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

    OfflineThreadPolicyView offlineThreadPolicy =
        readOfflineThreadPolicy(module);
    std::string targetProfileId =
        moduleStringAttr(module, "target.profile_id");

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
        if (!selected->truth_boundary.empty())
          op.setAttr("kernel_selection.truth_boundary",
                     S(selected->truth_boundary));

        // Phase P1D: resolve a thread-decomposition schedule for the
        // already-selected kernel, a SEPARATE decision from which
        // kernel/tile runs. Absent entirely (no attrs set) when this
        // kernel declares no thread schedules at all -- old P1B/P1C
        // profiles stay byte-identical.
        ThreadScheduleResult tsResult =
            resolveThreadSchedule(*selected, physicalComputeUnits,
                                  offlineThreadPolicy, opCtx, op,
                                  targetProfileId);
        const ImplementationCandidate* selectedImplementation =
            tsResult.has_selected_candidate ? &tsResult.selected_candidate
                                            : nullptr;
        op.setAttr("kernel_selection.selected_id",
                   S(selectedImplementation
                         ? selectedImplementation->kernelId
                         : selected->kernel_id));
        op.setAttr("kernel_selection.source", S(selected->source));
        if (selectedImplementation) {
          op.setAttr("implementation_candidate.selected_id",
                     S(selectedImplementation->candidateId));
          op.setAttr("implementation_candidate.provider_id",
                     S(selectedImplementation->providerId));
          op.setAttr("implementation_candidate.backend",
                     S(selectedImplementation->backend));
          op.setAttr("implementation_candidate.implementation_kind",
                     S(selectedImplementation->implementationKind));
          op.setAttr("implementation_candidate.runtime_contract_kind",
                     S(selectedImplementation->runtimeContractKind));
          op.setAttr("implementation_candidate.kernel_id",
                     S(selectedImplementation->kernelId));
          op.setAttr("implementation_candidate.dtype",
                     S(selectedImplementation->dtype));
          if (selectedImplementation->tile.present) {
            op.setAttr("implementation_candidate.tile_identity",
                       S(PortableCPUProvider::tileIdentity(
                           selectedImplementation->tile)));
            op.setAttr("implementation_candidate.tile_block_m",
                       IntegerAttr::get(IntegerType::get(ctx, 64),
                                        selectedImplementation->tile.blockM));
            op.setAttr("implementation_candidate.tile_block_n",
                       IntegerAttr::get(IntegerType::get(ctx, 64),
                                        selectedImplementation->tile.blockN));
            op.setAttr("implementation_candidate.tile_block_k",
                       IntegerAttr::get(IntegerType::get(ctx, 64),
                                        selectedImplementation->tile.blockK));
          }
        }
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
          if (!tsResult.policy_id.empty()) {
            op.setAttr("thread_schedule.policy_id", S(tsResult.policy_id));
            op.setAttr("thread_schedule.policy_version", S(tsResult.policy_version));
            op.setAttr("thread_schedule.policy_metric", S(tsResult.policy_metric));
            op.setAttr("thread_schedule.policy_metric_value",
                       IntegerAttr::get(IntegerType::get(ctx, 64),
                                        tsResult.policy_metric_value));
            op.setAttr("thread_schedule.policy_threshold",
                       IntegerAttr::get(IntegerType::get(ctx, 64),
                                        tsResult.policy_threshold));
            op.setAttr("thread_schedule.policy_boundary_rule",
                       S(tsResult.policy_boundary_rule));
            op.setAttr("thread_schedule.policy_selection_reason",
                       S(tsResult.policy_selection_reason));
            op.setAttr("thread_schedule.policy_evidence_ref",
                       S(tsResult.policy_evidence_ref));
            op.setAttr("thread_schedule.policy_evidence_sha256",
                       S(tsResult.policy_evidence_sha256));
            op.setAttr("thread_schedule.policy_truth_boundary",
                       S(tsResult.policy_truth_boundary));
          }
          if (!tsResult.selected_candidate_id.empty()) {
            op.setAttr("thread_schedule.selected_candidate_id",
                       S(tsResult.selected_candidate_id));
          }
          if (!tsResult.considered_candidate_ids.empty()) {
            SmallVector<Attribute> candidateIds;
            for (const auto& id : tsResult.considered_candidate_ids)
              candidateIds.push_back(S(id));
            op.setAttr("thread_schedule.considered_candidate_ids",
                       ArrayAttr::get(ctx, candidateIds));
          }
          if (!tsResult.candidate_rejections.empty()) {
            SmallVector<Attribute> rejectedIds;
            SmallVector<Attribute> rejectedReasons;
            for (const auto& rejection : tsResult.candidate_rejections) {
              rejectedIds.push_back(S(rejection.candidateId));
              rejectedReasons.push_back(S(rejection.reason));
            }
            op.setAttr("thread_schedule.rejected_candidate_ids",
                       ArrayAttr::get(ctx, rejectedIds));
            op.setAttr("thread_schedule.rejected_candidate_reasons",
                       ArrayAttr::get(ctx, rejectedReasons));
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
