#pragma once

#include "serving/ImplementationCandidate.h"

#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>
#include <vector>

namespace mlir::hir {

struct XNNPACKProviderContext {
  std::string semanticTargetRef;
  CandidateScopeKind scopeKind = CandidateScopeKind::Unknown;
  bool hasStaticShape = false;
  std::string targetProfileId;
  std::string backend;
  std::string dtype;
  std::string pteArtifactRef;
  std::string pteSha256;
  std::string runnerSha256;
  std::string executorchTag;
  std::string executorchCommit;
  std::string xnnpackCommit;
  bool xnnpackDelegationProven = false;
  bool inputBindingCompatible = false;
  std::string truthBoundary;
};

struct XNNPACKProviderDiagnostic {
  std::string reason;
};

struct XNNPACKCandidateView {
  ImplementationCandidate candidate;
  int64_t requestedThreadCount = 0;
};

struct XNNPACKProviderResult {
  std::vector<XNNPACKCandidateView> candidates;
  std::vector<XNNPACKProviderDiagnostic> diagnostics;
};

struct XNNPACKFeasibilityContext {
  std::string semanticTargetRef;
  CandidateScopeKind scopeKind = CandidateScopeKind::Unknown;
  bool hasStaticShape = false;
  std::string targetProfileId;
  std::string backend;
  std::string dtype;
  std::string expectedPteSha256;
  std::string expectedRunnerSha256;
  std::string expectedExecutorchCommit;
  std::string expectedXNNPACKCommit;
  bool xnnpackDelegationProven = false;
  bool inputBindingCompatible = false;
  std::optional<int64_t> physicalComputeUnits;
};

class XNNPACKFeasibilityEvaluator {
public:
  CandidateFeasibilitySummary evaluate(
      const ImplementationCandidate& candidate,
      const XNNPACKFeasibilityContext& ctx) const {
    CandidateFeasibilitySummary summary;
    auto reject = [&](CandidateFeasibilityStatus status,
                      llvm::StringRef reason) {
      summary.status = status;
      summary.reason = reason.str();
      return summary;
    };
    if (candidate.semanticTargetRef != ctx.semanticTargetRef ||
        candidate.scopeKind != ctx.scopeKind)
      return reject(CandidateFeasibilityStatus::Unsupported,
                    "wrong_semantic_scope");
    if (candidate.targetProfileId != ctx.targetProfileId)
      return reject(CandidateFeasibilityStatus::Rejected,
                    "target_profile_mismatch");
    if (candidate.backend != ctx.backend || candidate.backend != "cpu" ||
        candidate.library != "xnnpack")
      return reject(CandidateFeasibilityStatus::Rejected,
                    "backend_library_unavailable");
    if (candidate.dtype != ctx.dtype || candidate.dtype != "fp32")
      return reject(CandidateFeasibilityStatus::Rejected, "wrong_dtype");
    if (!ctx.hasStaticShape)
      return reject(CandidateFeasibilityStatus::Deferred,
                    "missing_static_shape");
    if (candidate.pteSha256.empty())
      return reject(CandidateFeasibilityStatus::Rejected, "missing_pte_hash");
    if (candidate.pteSha256 != ctx.expectedPteSha256)
      return reject(CandidateFeasibilityStatus::Rejected, "pte_hash_mismatch");
    if (candidate.runnerSha256.empty())
      return reject(CandidateFeasibilityStatus::Rejected, "missing_runner_hash");
    if (candidate.runnerSha256 != ctx.expectedRunnerSha256)
      return reject(CandidateFeasibilityStatus::Rejected, "runner_hash_mismatch");
    if (candidate.executorchCommit != ctx.expectedExecutorchCommit)
      return reject(CandidateFeasibilityStatus::Rejected,
                    "executorch_commit_mismatch");
    if (candidate.xnnpackCommit != ctx.expectedXNNPACKCommit)
      return reject(CandidateFeasibilityStatus::Rejected,
                    "xnnpack_commit_mismatch");
    if (!ctx.xnnpackDelegationProven)
      return reject(CandidateFeasibilityStatus::Rejected,
                    "xnnpack_delegation_unproven");
    if (!ctx.inputBindingCompatible)
      return reject(CandidateFeasibilityStatus::Rejected,
                    "input_binding_incompatible");
    if (!candidate.threadSchedule.present)
      return reject(CandidateFeasibilityStatus::Rejected,
                    "missing_requested_thread_mode");
    int64_t threads = candidate.threadSchedule.threadCount;
    if (threads != 1 && threads != 4)
      return reject(CandidateFeasibilityStatus::Rejected,
                    "unsupported_requested_thread_mode");
    if (threads > 1) {
      if (!ctx.physicalComputeUnits.has_value())
        return reject(CandidateFeasibilityStatus::Deferred,
                      "deferred_missing_compute_units");
      if (*ctx.physicalComputeUnits < threads)
        return reject(CandidateFeasibilityStatus::Rejected,
                      "rejected_exceeds_compute_units");
    }
    summary.status = CandidateFeasibilityStatus::Feasible;
    summary.reason = "executorch_xnnpack_artifacts_validated";
    return summary;
  }
};

class XNNPACKCandidateProvider {
public:
  static constexpr llvm::StringLiteral kProviderId =
      "executorch_xnnpack_candidate_provider";
  static constexpr llvm::StringLiteral kProviderVersion = "e3a.v1";
  static constexpr llvm::StringLiteral kImplementationKind =
      "external_library_delegate";
  static constexpr llvm::StringLiteral kRuntimeContractKind =
      "executorch_xnnpack_runner_contract";

  llvm::StringRef providerId() const { return kProviderId; }
  llvm::StringRef providerVersion() const { return kProviderVersion; }

  bool supportsScope(const XNNPACKProviderContext& ctx) const {
    return ctx.scopeKind == CandidateScopeKind::FusedRegion &&
           ctx.semanticTargetRef == "fused_matmul_bias_relu";
  }

  XNNPACKProviderResult enumerateCandidates(
      const XNNPACKProviderContext& ctx) const {
    XNNPACKProviderResult result;
    auto diag = [&](llvm::StringRef reason) {
      result.diagnostics.push_back({reason.str()});
    };
    if (!supportsScope(ctx)) {
      diag("unsupported_semantic_scope");
      return result;
    }
    if (ctx.backend != "cpu") {
      diag("no_cpu_backend_declared");
      return result;
    }
    if (ctx.dtype != "fp32") {
      diag("wrong_dtype");
      return result;
    }
    if (ctx.pteSha256.empty())
      diag("missing_pte_hash");
    if (ctx.runnerSha256.empty())
      diag("missing_runner_hash");
    if (ctx.executorchCommit.empty())
      diag("missing_executorch_commit");
    if (ctx.xnnpackCommit.empty())
      diag("missing_xnnpack_commit");
    if (!result.diagnostics.empty())
      return result;

    result.candidates.push_back(buildCandidate(ctx, 1, "xnnpack_requested_1_thread"));
    result.candidates.push_back(buildCandidate(ctx, 4, "xnnpack_requested_4_threads"));
    return result;
  }

private:
  XNNPACKCandidateView buildCandidate(const XNNPACKProviderContext& ctx,
                                      int64_t requestedThreads,
                                      llvm::StringRef reason) const {
    ImplementationCandidate candidate;
    candidate.providerId = kProviderId.str();
    candidate.targetProfileId = ctx.targetProfileId;
    candidate.scopeKind = ctx.scopeKind;
    candidate.semanticTargetRef = ctx.semanticTargetRef;
    candidate.backend = "cpu";
    candidate.library = "xnnpack";
    candidate.implementationKind = kImplementationKind.str();
    candidate.runtimeContractKind = kRuntimeContractKind.str();
    candidate.dtype = ctx.dtype;
    candidate.pteArtifactRef = ctx.pteArtifactRef;
    candidate.pteSha256 = ctx.pteSha256;
    candidate.runnerSha256 = ctx.runnerSha256;
    candidate.executorchTag = ctx.executorchTag;
    candidate.executorchCommit = ctx.executorchCommit;
    candidate.xnnpackCommit = ctx.xnnpackCommit;
    candidate.threadSchedule.present = true;
    candidate.threadSchedule.threadCount = requestedThreads;
    candidate.threadSchedule.partitionAxis = "runtime_threadpool";
    candidate.threadSchedule.partitionStrategy = "xnnpack_requested_threads";
    candidate.candidateReason = reason.str();
    candidate.truthBoundary = ctx.truthBoundary.empty()
                                  ? "xnnpack_candidate_provider_enumerates_only"
                                  : ctx.truthBoundary;
    candidate.feasibility.status = CandidateFeasibilityStatus::Unknown;
    candidate.feasibility.reason =
        "provider_enumerated_requires_artifact_feasibility";
    candidate.candidateId = makeFallbackCandidateId(candidate);
    return {candidate, requestedThreads};
  }
};

} // namespace mlir::hir
