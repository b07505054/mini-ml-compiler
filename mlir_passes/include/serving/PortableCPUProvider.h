#pragma once

#include "serving/ImplementationCandidate.h"

#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>
#include <vector>

namespace mlir::hir {

struct PortableCpuThreadSchedule {
  int64_t threadCount = 1;
  std::string partitionAxis;
  std::string partitionStrategy;
};

struct PortableCpuRuntimeKernelDescriptor {
  std::string kernelId;
  std::string opName;
  std::string backend;
  std::vector<std::string> supportedDtypes;
  std::vector<std::string> supportedTileShapes;
  std::vector<PortableCpuThreadSchedule> supportedThreadSchedules;
  std::string truthBoundary;
};

struct PortableCpuProviderContext {
  std::string semanticTargetRef;
  CandidateScopeKind scopeKind = CandidateScopeKind::Unknown;
  bool hasStaticShape = false;
  std::string targetProfileId;
  std::string backend;
  std::string dtype;
  std::string truthBoundary;
};

struct PortableCpuProviderDiagnostic {
  std::string reason;
};

struct PortableCpuCandidateView {
  ImplementationCandidate candidate;
  PortableCpuThreadSchedule schedule;
};

struct PortableCpuProviderResult {
  std::vector<PortableCpuCandidateView> candidates;
  std::vector<PortableCpuProviderDiagnostic> diagnostics;
};

struct PortableCpuFeasibilityContext {
  std::string semanticTargetRef;
  CandidateScopeKind scopeKind = CandidateScopeKind::Unknown;
  bool hasStaticShape = false;
  std::string targetProfileId;
  std::string backend;
  std::string dtype;
  std::optional<int64_t> physicalComputeUnits;
};

class PortableCPUFeasibilityEvaluator {
public:
  CandidateFeasibilitySummary evaluate(
      const ImplementationCandidate& candidate,
      const PortableCpuFeasibilityContext& ctx) const {
    CandidateFeasibilitySummary summary;
    if (candidate.semanticTargetRef != ctx.semanticTargetRef ||
        candidate.scopeKind != ctx.scopeKind) {
      summary.status = CandidateFeasibilityStatus::Unsupported;
      summary.reason = "wrong_semantic_scope";
      return summary;
    }
    if (candidate.targetProfileId != ctx.targetProfileId) {
      summary.status = CandidateFeasibilityStatus::Rejected;
      summary.reason = "target_profile_mismatch";
      return summary;
    }
    if (candidate.backend != ctx.backend || candidate.backend != "cpu") {
      summary.status = CandidateFeasibilityStatus::Rejected;
      summary.reason = "backend_unavailable";
      return summary;
    }
    if (candidate.dtype != ctx.dtype || candidate.dtype != "fp32") {
      summary.status = CandidateFeasibilityStatus::Rejected;
      summary.reason = "wrong_dtype";
      return summary;
    }
    if (candidate.kernelId.empty()) {
      summary.status = CandidateFeasibilityStatus::Rejected;
      summary.reason = "no_matching_kernel_descriptor";
      return summary;
    }
    if (!candidate.tile.present || candidate.tile.blockM != 32 ||
        candidate.tile.blockN != 128 || candidate.tile.blockK != 32) {
      summary.status = CandidateFeasibilityStatus::Rejected;
      summary.reason = "kernel_tile_identity_mismatch";
      return summary;
    }
    if (!candidate.threadSchedule.present) {
      summary.status = CandidateFeasibilityStatus::Rejected;
      summary.reason = "missing_thread_schedule";
      return summary;
    }
    const auto& ts = candidate.threadSchedule;
    bool serial = ts.threadCount == 1 && ts.partitionAxis == "none" &&
                  ts.partitionStrategy == "serial";
    bool splitM4 = ts.threadCount == 4 && ts.partitionAxis == "m" &&
                   ts.partitionStrategy == "contiguous_chunks";
    if (!serial && !splitM4) {
      summary.status = CandidateFeasibilityStatus::Rejected;
      summary.reason = "invalid_schedule_tuple";
      return summary;
    }
    if (!ctx.hasStaticShape) {
      summary.status = CandidateFeasibilityStatus::Deferred;
      summary.reason = "missing_static_shape";
      return summary;
    }
    if (splitM4) {
      if (!ctx.physicalComputeUnits.has_value()) {
        summary.status = CandidateFeasibilityStatus::Deferred;
        summary.reason = "deferred_missing_compute_units";
        return summary;
      }
      if (*ctx.physicalComputeUnits < ts.threadCount) {
        summary.status = CandidateFeasibilityStatus::Rejected;
        summary.reason = "rejected_exceeds_compute_units";
        return summary;
      }
    }
    summary.status = CandidateFeasibilityStatus::Feasible;
    summary.reason = serial ? "serial_schedule_declared_legal"
                            : "parallel_schedule_declared_legal";
    return summary;
  }
};

class PortableCPUProvider {
public:
  static constexpr llvm::StringLiteral kProviderId =
      "portable_cpu_provider";
  static constexpr llvm::StringLiteral kProviderVersion = "a4.v1";
  static constexpr llvm::StringLiteral kImplementationKind =
      "opaque_portable_cpu_native_kernel";
  static constexpr llvm::StringLiteral kRuntimeContractKind =
      "portable_cpu_kernel_adapter_contract";

  llvm::StringRef providerId() const { return kProviderId; }
  llvm::StringRef providerVersion() const { return kProviderVersion; }

  bool supportsScope(const PortableCpuProviderContext& ctx) const {
    return ctx.scopeKind == CandidateScopeKind::FusedRegion &&
           ctx.semanticTargetRef == "fused_matmul_bias_relu";
  }

  PortableCpuProviderResult enumerateCandidates(
      const PortableCpuProviderContext& ctx,
      const PortableCpuRuntimeKernelDescriptor& descriptor) const {
    PortableCpuProviderResult result;
    auto diag = [&](llvm::StringRef reason) {
      result.diagnostics.push_back({reason.str()});
    };

    if (!supportsScope(ctx)) {
      diag("unsupported_semantic_scope");
      return result;
    }
    if (ctx.backend != "cpu" || descriptor.backend != "cpu") {
      diag("no_cpu_backend_declared");
      return result;
    }
    if (descriptor.opName != ctx.semanticTargetRef) {
      diag("no_matching_runtime_kernel_descriptor");
      return result;
    }
    if (descriptor.kernelId.empty()) {
      diag("malformed_descriptor_missing_kernel_id");
      return result;
    }
    if (!descriptor.supportedDtypes.empty() &&
        !contains(descriptor.supportedDtypes, ctx.dtype)) {
      diag("wrong_dtype");
      return result;
    }

    std::optional<TileCandidateSpec> tile =
        parseTileIdentityFromKernelId(descriptor.kernelId);
    if (!tile) {
      diag("kernel_tile_identity_unresolved");
      return result;
    }
    if (!descriptorAcceptsCandidateTile(descriptor, *tile)) {
      diag("kernel_tile_identity_mismatch");
      return result;
    }

    const PortableCpuThreadSchedule* serial =
        findSchedule(descriptor, 1, "none", "serial");
    const PortableCpuThreadSchedule* parallel =
        findSchedule(descriptor, 4, "m", "contiguous_chunks");
    if (!serial)
      diag("missing_serial_schedule");
    if (!parallel)
      diag("missing_parallel_schedule");

    if (serial)
      result.candidates.push_back(
          buildCandidate(ctx, descriptor, *tile, *serial,
                         "p1d1_below_threshold_serial_candidate",
                         "serial_schedule_declared_legal"));
    if (parallel)
      result.candidates.push_back(
          buildCandidate(ctx, descriptor, *tile, *parallel,
                         "p1d1_at_or_above_threshold_parallel_candidate",
                         "parallel_schedule_declared_structurally_legal"));

    if (hasCandidateIdCollision(result.candidates))
      diag("duplicate_candidate_identity");
    return result;
  }

  static std::optional<TileCandidateSpec> parseTileIdentityFromKernelId(
      llvm::StringRef kernelId) {
    auto readPart = [&](llvm::StringRef marker) -> std::optional<int64_t> {
      size_t pos = kernelId.find(marker);
      if (pos == llvm::StringRef::npos)
        return std::nullopt;
      pos += marker.size();
      int64_t value = 0;
      bool sawDigit = false;
      while (pos < kernelId.size() && kernelId[pos] >= '0' &&
             kernelId[pos] <= '9') {
        sawDigit = true;
        value = value * 10 + (kernelId[pos] - '0');
        ++pos;
      }
      if (!sawDigit)
        return std::nullopt;
      return value;
    };
    auto bm = readPart("bm");
    auto bn = readPart("bn");
    auto bk = readPart("bk");
    if (!bm || !bn || !bk)
      return std::nullopt;
    TileCandidateSpec tile;
    tile.present = true;
    tile.blockM = *bm;
    tile.blockN = *bn;
    tile.blockK = *bk;
    return tile;
  }

  static std::string tileIdentity(const TileCandidateSpec& tile) {
    if (!tile.present)
      return "";
    return "bm" + std::to_string(tile.blockM) + "_bn" +
           std::to_string(tile.blockN) + "_bk" +
           std::to_string(tile.blockK);
  }

private:
  static bool contains(const std::vector<std::string>& values,
                       const std::string& value) {
    for (const auto& candidate : values)
      if (candidate == value)
        return true;
    return false;
  }

  static bool descriptorAcceptsCandidateTile(
      const PortableCpuRuntimeKernelDescriptor& descriptor,
      const TileCandidateSpec& tile) {
    if (!tile.present)
      return false;
    if (descriptor.supportedTileShapes.empty())
      return true;
    std::string shape = std::to_string(tile.blockM) + "x" +
                        std::to_string(tile.blockN) + "x" +
                        std::to_string(tile.blockK);
    return contains(descriptor.supportedTileShapes, shape) ||
           contains(descriptor.supportedTileShapes, tileIdentity(tile));
  }

  static const PortableCpuThreadSchedule* findSchedule(
      const PortableCpuRuntimeKernelDescriptor& descriptor,
      int64_t threadCount,
      llvm::StringRef partitionAxis,
      llvm::StringRef partitionStrategy) {
    for (const auto& schedule : descriptor.supportedThreadSchedules)
      if (schedule.threadCount == threadCount &&
          schedule.partitionAxis == partitionAxis &&
          schedule.partitionStrategy == partitionStrategy)
        return &schedule;
    return nullptr;
  }

  PortableCpuCandidateView buildCandidate(
      const PortableCpuProviderContext& ctx,
      const PortableCpuRuntimeKernelDescriptor& descriptor,
      const TileCandidateSpec& tile,
      const PortableCpuThreadSchedule& schedule,
      llvm::StringRef reason,
      llvm::StringRef feasibilityReason) const {
    ImplementationCandidate candidate;
    candidate.providerId = kProviderId.str();
    candidate.targetProfileId = ctx.targetProfileId;
    candidate.scopeKind = ctx.scopeKind;
    candidate.semanticTargetRef = ctx.semanticTargetRef;
    candidate.backend = "cpu";
    candidate.implementationKind = kImplementationKind.str();
    candidate.runtimeContractKind = kRuntimeContractKind.str();
    candidate.kernelId = descriptor.kernelId;
    candidate.dtype = ctx.dtype;
    candidate.tile = tile;
    candidate.threadSchedule.present = true;
    candidate.threadSchedule.threadCount = schedule.threadCount;
    candidate.threadSchedule.partitionAxis = schedule.partitionAxis;
    candidate.threadSchedule.partitionStrategy = schedule.partitionStrategy;
    candidate.candidateReason = reason.str();
    candidate.truthBoundary = ctx.truthBoundary.empty()
                                  ? descriptor.truthBoundary
                                  : ctx.truthBoundary;
    candidate.feasibility.status = CandidateFeasibilityStatus::Unknown;
    candidate.feasibility.reason =
        std::string("provider_enumerated_requires_feasibility:") +
        feasibilityReason.str();
    candidate.candidateId = makeFallbackCandidateId(candidate);
    return {candidate, schedule};
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
};

} // namespace mlir::hir
