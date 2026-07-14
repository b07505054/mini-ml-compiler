#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::hir {

enum class CandidateScopeKind {
  Operator,
  FusedRegion,
  Function,
  Unknown,
};

enum class CandidateFeasibilityStatus {
  Feasible,
  Deferred,
  Rejected,
  Unsupported,
  Unknown,
};

struct CandidateFeasibilitySummary {
  CandidateFeasibilityStatus status = CandidateFeasibilityStatus::Unknown;
  std::string reason;
  std::vector<std::string> requiredCapabilityRefs;
};

struct ImplementationCandidateCostSummary {
  bool hasPenaltyScore = false;
  int64_t penaltyScore = 0;
  std::string evaluationStatus;
  std::string evaluationReason;
  std::string costModelId;
  std::string truthBoundary;
};

struct ThreadScheduleCandidateSpec {
  bool present = false;
  int64_t threadCount = 1;
  std::string partitionAxis;
  std::string partitionStrategy;
};

struct QuantizationCandidateSpec {
  bool present = false;
  std::string scheme;
  std::string activationDtype;
  std::string weightDtype;
  std::string accumulatorDtype;
  std::string outputDtype;
  std::string granularity;
  int64_t groupSize = 0;
  bool calibrationRequired = false;
  bool calibrationAvailable = false;
  std::string requiredBackendCapability;
  std::string requiredKernelCapability;
};

struct TileCandidateSpec {
  bool present = false;
  int64_t blockM = 0;
  int64_t blockN = 0;
  int64_t blockK = 0;
};

struct ImplementationCandidate {
  std::string candidateId;
  std::string providerId;
  std::string targetProfileId;
  CandidateScopeKind scopeKind = CandidateScopeKind::Unknown;
  std::string semanticTargetRef;
  std::string functionRef;
  std::string backend;
  std::string library;
  std::string implementationKind;
  std::string runtimeContractKind;
  std::string kernelId;
  std::string dtype;
  std::string pteArtifactRef;
  std::string pteSha256;
  std::string runnerSha256;
  std::string executorchTag;
  std::string executorchCommit;
  std::string xnnpackCommit;
  TileCandidateSpec tile;
  ThreadScheduleCandidateSpec threadSchedule;
  QuantizationCandidateSpec quantization;
  std::string candidateReason;
  std::vector<std::string> requiredBoundaryOps;
  std::string fallbackBackend;
  CandidateFeasibilitySummary feasibility;
  ImplementationCandidateCostSummary cost;
  std::string truthBoundary;
};

struct PolicyResultCandidateRejection {
  std::string candidateId;
  std::string reason;
};

struct PolicyResult {
  std::string selectedCandidateId;
  std::vector<std::string> consideredCandidateIds;
  std::vector<PolicyResultCandidateRejection> rejectedCandidates;
  std::string policyId;
  std::string selectionReason;
  std::string objectiveSummary;
  std::string truthBoundary;
};

inline llvm::StringRef stringifyScopeKind(CandidateScopeKind kind) {
  switch (kind) {
  case CandidateScopeKind::Operator:
    return "operator";
  case CandidateScopeKind::FusedRegion:
    return "fused_region";
  case CandidateScopeKind::Function:
    return "function";
  case CandidateScopeKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

inline CandidateScopeKind parseScopeKind(llvm::StringRef value) {
  if (value == "operator")
    return CandidateScopeKind::Operator;
  if (value == "fused_region")
    return CandidateScopeKind::FusedRegion;
  if (value == "function")
    return CandidateScopeKind::Function;
  return CandidateScopeKind::Unknown;
}

inline llvm::StringRef stringifyFeasibilityStatus(
    CandidateFeasibilityStatus status) {
  switch (status) {
  case CandidateFeasibilityStatus::Feasible:
    return "feasible";
  case CandidateFeasibilityStatus::Deferred:
    return "deferred";
  case CandidateFeasibilityStatus::Rejected:
    return "rejected";
  case CandidateFeasibilityStatus::Unsupported:
    return "unsupported";
  case CandidateFeasibilityStatus::Unknown:
    return "unknown";
  }
  return "unknown";
}

inline CandidateFeasibilityStatus parseFeasibilityStatus(
    llvm::StringRef value) {
  if (value == "feasible")
    return CandidateFeasibilityStatus::Feasible;
  if (value == "deferred")
    return CandidateFeasibilityStatus::Deferred;
  if (value == "rejected")
    return CandidateFeasibilityStatus::Rejected;
  if (value == "unsupported")
    return CandidateFeasibilityStatus::Unsupported;
  return CandidateFeasibilityStatus::Unknown;
}

inline std::string getCandidateString(DictionaryAttr dict, llvm::StringRef key) {
  if (!dict)
    return {};
  if (auto attr = dict.get(key))
    if (auto stringAttr = dyn_cast<StringAttr>(attr))
      return stringAttr.getValue().str();
  return {};
}

inline int64_t getCandidateI64(DictionaryAttr dict, llvm::StringRef key,
                               int64_t defaultValue = 0) {
  if (!dict)
    return defaultValue;
  if (auto attr = dict.get(key))
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getInt();
  return defaultValue;
}

inline bool hasCandidateI64(DictionaryAttr dict, llvm::StringRef key) {
  return dict && dict.get(key) && isa<IntegerAttr>(dict.get(key));
}

inline std::vector<std::string> getCandidateStringArray(DictionaryAttr dict,
                                                        llvm::StringRef key) {
  std::vector<std::string> out;
  if (!dict)
    return out;
  if (auto attr = dict.get(key))
    if (auto arr = dyn_cast<ArrayAttr>(attr))
      for (auto elem : arr)
        if (auto stringAttr = dyn_cast<StringAttr>(elem))
          out.push_back(stringAttr.getValue().str());
  return out;
}

inline void upsertCandidateAttr(SmallVectorImpl<NamedAttribute> &attrs,
                                MLIRContext *ctx, llvm::StringRef key,
                                Attribute value) {
  for (NamedAttribute &attr : attrs) {
    if (attr.getName() == key) {
      attr = NamedAttribute(StringAttr::get(ctx, key), value);
      return;
    }
  }
  attrs.push_back(NamedAttribute(StringAttr::get(ctx, key), value));
}

inline ArrayAttr makeCandidateStringArray(MLIRContext *ctx,
                                          const std::vector<std::string> &values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (const std::string &value : values)
    attrs.push_back(StringAttr::get(ctx, value));
  return ArrayAttr::get(ctx, attrs);
}

inline std::string makeFallbackCandidateId(
    const ImplementationCandidate &candidate) {
  std::string id;
  if (!candidate.semanticTargetRef.empty())
    id += candidate.semanticTargetRef;
  else
    id += "unknown_scope";
  id += ":scope=";
  id += stringifyScopeKind(candidate.scopeKind).str();
  if (!candidate.backend.empty()) {
    id += ":backend=";
    id += candidate.backend;
  }
  if (!candidate.library.empty()) {
    id += ":library=";
    id += candidate.library;
  }
  id += ":";
  if (!candidate.implementationKind.empty())
    id += candidate.implementationKind;
  else
    id += "unknown_implementation";
  if (!candidate.runtimeContractKind.empty()) {
    id += ":contract=";
    id += candidate.runtimeContractKind;
  }
  if (!candidate.kernelId.empty()) {
    id += ":kernel=";
    id += candidate.kernelId;
  }
  if (candidate.tile.present) {
    id += ":tile=bm";
    id += std::to_string(candidate.tile.blockM);
    id += "_bn";
    id += std::to_string(candidate.tile.blockN);
    id += "_bk";
    id += std::to_string(candidate.tile.blockK);
  }
  if (!candidate.dtype.empty()) {
    id += ":dtype=";
    id += candidate.dtype;
  }
  if (candidate.quantization.present) {
    id += ":quant=";
    id += candidate.quantization.scheme.empty()
              ? "unknown"
              : candidate.quantization.scheme;
    id += ":act=";
    id += candidate.quantization.activationDtype.empty()
              ? "unknown"
              : candidate.quantization.activationDtype;
    id += ":weight=";
    id += candidate.quantization.weightDtype.empty()
              ? "unknown"
              : candidate.quantization.weightDtype;
    id += ":acc=";
    id += candidate.quantization.accumulatorDtype.empty()
              ? "unknown"
              : candidate.quantization.accumulatorDtype;
    id += ":out=";
    id += candidate.quantization.outputDtype.empty()
              ? "unknown"
              : candidate.quantization.outputDtype;
    if (!candidate.quantization.granularity.empty()) {
      id += ":gran=";
      id += candidate.quantization.granularity;
    }
    if (candidate.quantization.groupSize > 0) {
      id += ":group=";
      id += std::to_string(candidate.quantization.groupSize);
    }
  }
  if (!candidate.pteSha256.empty()) {
    id += ":pte=";
    id += candidate.pteSha256.substr(0, 12);
  }
  if (candidate.threadSchedule.present) {
    id += ":threads=";
    id += std::to_string(candidate.threadSchedule.threadCount);
    id += ":axis=";
    id += candidate.threadSchedule.partitionAxis.empty()
              ? "unknown"
              : candidate.threadSchedule.partitionAxis;
    id += ":strategy=";
    id += candidate.threadSchedule.partitionStrategy.empty()
              ? "unknown"
              : candidate.threadSchedule.partitionStrategy;
  }
  if (!candidate.targetProfileId.empty()) {
    id += ":target=";
    id += candidate.targetProfileId;
  }
  return id;
}

inline CandidateFeasibilitySummary inferCandidateFeasibility(
    DictionaryAttr dict, const ImplementationCandidate &candidate) {
  CandidateFeasibilitySummary feasibility;

  std::string explicitStatus = getCandidateString(dict, "feasibility.status");
  if (!explicitStatus.empty()) {
    feasibility.status = parseFeasibilityStatus(explicitStatus);
    feasibility.reason = getCandidateString(dict, "feasibility.reason");
    feasibility.requiredCapabilityRefs =
        getCandidateStringArray(dict, "feasibility.required_capability_refs");
    return feasibility;
  }

  std::string rejectionReason = getCandidateString(dict, "rejection_reason");
  if (!rejectionReason.empty()) {
    feasibility.status = CandidateFeasibilityStatus::Rejected;
    feasibility.reason = rejectionReason;
    return feasibility;
  }

  std::string evalStatus = getCandidateString(dict, "evaluation.status");
  if (evalStatus == "evaluated") {
    feasibility.status = CandidateFeasibilityStatus::Feasible;
    feasibility.reason = getCandidateString(dict, "evaluation.reason");
    return feasibility;
  }
  if (evalStatus == "partially_evaluated") {
    feasibility.status = CandidateFeasibilityStatus::Deferred;
    feasibility.reason = getCandidateString(dict, "evaluation.reason");
    return feasibility;
  }
  if (evalStatus == "rejected") {
    feasibility.status = candidate.implementationKind == "unsupported"
                             ? CandidateFeasibilityStatus::Unsupported
                             : CandidateFeasibilityStatus::Rejected;
    feasibility.reason = getCandidateString(dict, "evaluation.reason");
    return feasibility;
  }

  std::string constraintStatus = getCandidateString(dict, "constraint_status");
  if (constraintStatus == "pass") {
    feasibility.status = CandidateFeasibilityStatus::Feasible;
    feasibility.reason = "constraints_passed";
    return feasibility;
  }
  if (constraintStatus == "fail") {
    feasibility.status = CandidateFeasibilityStatus::Rejected;
    feasibility.reason = getCandidateString(dict, "failed_constraints");
    return feasibility;
  }

  if (candidate.implementationKind == "unsupported") {
    feasibility.status = CandidateFeasibilityStatus::Unsupported;
    feasibility.reason = candidate.candidateReason.empty()
                             ? "no_viable_lowering_path"
                             : candidate.candidateReason;
    return feasibility;
  }

  if (!candidate.implementationKind.empty()) {
    feasibility.status = CandidateFeasibilityStatus::Feasible;
    feasibility.reason = candidate.candidateReason;
    return feasibility;
  }

  feasibility.status = CandidateFeasibilityStatus::Unknown;
  feasibility.reason = "candidate_status_not_declared";
  return feasibility;
}

inline ImplementationCandidate decodeImplementationCandidate(
    DictionaryAttr dict, llvm::StringRef providerId = "compiler.internal") {
  ImplementationCandidate candidate;
  candidate.providerId = getCandidateString(dict, "provider_id");
  if (candidate.providerId.empty())
    candidate.providerId = providerId.str();
  candidate.targetProfileId = getCandidateString(dict, "target_profile_id");
  candidate.scopeKind = parseScopeKind(getCandidateString(dict, "scope_kind"));
  if (candidate.scopeKind == CandidateScopeKind::Unknown && dict)
    candidate.scopeKind = CandidateScopeKind::Operator;
  candidate.semanticTargetRef = getCandidateString(dict, "semantic_target_ref");
  if (candidate.semanticTargetRef.empty())
    candidate.semanticTargetRef = getCandidateString(dict, "source_op");
  candidate.functionRef = getCandidateString(dict, "function_ref");
  candidate.backend = getCandidateString(dict, "backend");
  candidate.library = getCandidateString(dict, "library");
  candidate.implementationKind = getCandidateString(dict, "implementation_kind");
  if (candidate.implementationKind.empty())
    candidate.implementationKind = getCandidateString(dict, "candidate_type");
  candidate.runtimeContractKind =
      getCandidateString(dict, "runtime_contract_kind");
  candidate.kernelId = getCandidateString(dict, "kernel_id");
  candidate.dtype = getCandidateString(dict, "dtype");
  candidate.pteArtifactRef = getCandidateString(dict, "artifact.pte_ref");
  candidate.pteSha256 = getCandidateString(dict, "artifact.pte_sha256");
  candidate.runnerSha256 = getCandidateString(dict, "artifact.runner_sha256");
  candidate.executorchTag = getCandidateString(dict, "provenance.executorch_tag");
  candidate.executorchCommit = getCandidateString(dict, "provenance.executorch_commit");
  candidate.xnnpackCommit = getCandidateString(dict, "provenance.xnnpack_commit");
  if (hasCandidateI64(dict, "tile.block_m") &&
      hasCandidateI64(dict, "tile.block_n") &&
      hasCandidateI64(dict, "tile.block_k")) {
    candidate.tile.present = true;
    candidate.tile.blockM = getCandidateI64(dict, "tile.block_m");
    candidate.tile.blockN = getCandidateI64(dict, "tile.block_n");
    candidate.tile.blockK = getCandidateI64(dict, "tile.block_k");
  }
  if (hasCandidateI64(dict, "thread_schedule.thread_count")) {
    candidate.threadSchedule.present = true;
    candidate.threadSchedule.threadCount =
        getCandidateI64(dict, "thread_schedule.thread_count", 1);
    candidate.threadSchedule.partitionAxis =
        getCandidateString(dict, "thread_schedule.partition_axis");
    candidate.threadSchedule.partitionStrategy =
        getCandidateString(dict, "thread_schedule.partition_strategy");
  }
  std::string quantScheme = getCandidateString(dict, "quantization.scheme");
  if (!quantScheme.empty()) {
    candidate.quantization.present = true;
    candidate.quantization.scheme = quantScheme;
    candidate.quantization.activationDtype =
        getCandidateString(dict, "quantization.activation_dtype");
    candidate.quantization.weightDtype =
        getCandidateString(dict, "quantization.weight_dtype");
    candidate.quantization.accumulatorDtype =
        getCandidateString(dict, "quantization.accumulator_dtype");
    candidate.quantization.outputDtype =
        getCandidateString(dict, "quantization.output_dtype");
    candidate.quantization.granularity =
        getCandidateString(dict, "quantization.granularity");
    candidate.quantization.groupSize =
        getCandidateI64(dict, "quantization.group_size", 0);
    candidate.quantization.requiredBackendCapability =
        getCandidateString(dict, "quantization.required_backend_capability");
    candidate.quantization.requiredKernelCapability =
        getCandidateString(dict, "quantization.required_kernel_capability");
    candidate.quantization.calibrationRequired =
        getCandidateString(dict, "quantization.calibration_required") == "true";
    candidate.quantization.calibrationAvailable =
        getCandidateString(dict, "quantization.calibration_available") == "true";
  }
  candidate.candidateReason = getCandidateString(dict, "candidate_reason");
  candidate.requiredBoundaryOps =
      getCandidateStringArray(dict, "required_boundary_ops");
  candidate.fallbackBackend = getCandidateString(dict, "fallback_backend");
  candidate.truthBoundary = getCandidateString(dict, "truth_boundary");

  candidate.candidateId = getCandidateString(dict, "candidate_id");
  if (candidate.candidateId.empty())
    candidate.candidateId = getCandidateString(dict, "id");
  if (candidate.candidateId.empty())
    candidate.candidateId = makeFallbackCandidateId(candidate);

  candidate.cost.evaluationStatus =
      getCandidateString(dict, "evaluation.status");
  candidate.cost.evaluationReason =
      getCandidateString(dict, "evaluation.reason");
  candidate.cost.costModelId =
      getCandidateString(dict, "evaluation.cost.model_id");
  candidate.cost.truthBoundary =
      getCandidateString(dict, "evaluation.cost.truth_boundary");
  if (hasCandidateI64(dict, "evaluation.penalty_score")) {
    candidate.cost.hasPenaltyScore = true;
    candidate.cost.penaltyScore =
        getCandidateI64(dict, "evaluation.penalty_score");
  }

  candidate.feasibility = inferCandidateFeasibility(dict, candidate);
  return candidate;
}

inline DictionaryAttr encodeImplementationCandidate(
    MLIRContext *ctx, const ImplementationCandidate &candidate,
    DictionaryAttr base = {}) {
  SmallVector<NamedAttribute> attrs;
  if (base)
    attrs.append(base.begin(), base.end());

  auto stringAttr = [&](llvm::StringRef value) -> Attribute {
    return StringAttr::get(ctx, value);
  };

  std::string candidateId = candidate.candidateId.empty()
                                ? makeFallbackCandidateId(candidate)
                                : candidate.candidateId;
  upsertCandidateAttr(attrs, ctx, "candidate_id", stringAttr(candidateId));
  upsertCandidateAttr(attrs, ctx, "provider_id",
                      stringAttr(candidate.providerId));
  if (!candidate.targetProfileId.empty())
    upsertCandidateAttr(attrs, ctx, "target_profile_id",
                        stringAttr(candidate.targetProfileId));
  upsertCandidateAttr(attrs, ctx, "scope_kind",
                      stringAttr(stringifyScopeKind(candidate.scopeKind)));
  upsertCandidateAttr(attrs, ctx, "semantic_target_ref",
                      stringAttr(candidate.semanticTargetRef));
  if (!candidate.functionRef.empty())
    upsertCandidateAttr(attrs, ctx, "function_ref",
                        stringAttr(candidate.functionRef));
  if (!candidate.backend.empty())
    upsertCandidateAttr(attrs, ctx, "backend",
                        stringAttr(candidate.backend));
  if (!candidate.library.empty())
    upsertCandidateAttr(attrs, ctx, "library",
                        stringAttr(candidate.library));
  upsertCandidateAttr(attrs, ctx, "implementation_kind",
                      stringAttr(candidate.implementationKind));
  upsertCandidateAttr(attrs, ctx, "candidate_type",
                      stringAttr(candidate.implementationKind));
  upsertCandidateAttr(attrs, ctx, "source_op",
                      stringAttr(candidate.semanticTargetRef));
  if (!candidate.runtimeContractKind.empty())
    upsertCandidateAttr(attrs, ctx, "runtime_contract_kind",
                        stringAttr(candidate.runtimeContractKind));
  if (!candidate.kernelId.empty())
    upsertCandidateAttr(attrs, ctx, "kernel_id",
                        stringAttr(candidate.kernelId));
  if (!candidate.dtype.empty())
    upsertCandidateAttr(attrs, ctx, "dtype",
                        stringAttr(candidate.dtype));
  if (!candidate.pteArtifactRef.empty())
    upsertCandidateAttr(attrs, ctx, "artifact.pte_ref",
                        stringAttr(candidate.pteArtifactRef));
  if (!candidate.pteSha256.empty())
    upsertCandidateAttr(attrs, ctx, "artifact.pte_sha256",
                        stringAttr(candidate.pteSha256));
  if (!candidate.runnerSha256.empty())
    upsertCandidateAttr(attrs, ctx, "artifact.runner_sha256",
                        stringAttr(candidate.runnerSha256));
  if (!candidate.executorchTag.empty())
    upsertCandidateAttr(attrs, ctx, "provenance.executorch_tag",
                        stringAttr(candidate.executorchTag));
  if (!candidate.executorchCommit.empty())
    upsertCandidateAttr(attrs, ctx, "provenance.executorch_commit",
                        stringAttr(candidate.executorchCommit));
  if (!candidate.xnnpackCommit.empty())
    upsertCandidateAttr(attrs, ctx, "provenance.xnnpack_commit",
                        stringAttr(candidate.xnnpackCommit));
  if (candidate.tile.present) {
    upsertCandidateAttr(attrs, ctx, "tile.block_m",
                        IntegerAttr::get(IntegerType::get(ctx, 64),
                                         candidate.tile.blockM));
    upsertCandidateAttr(attrs, ctx, "tile.block_n",
                        IntegerAttr::get(IntegerType::get(ctx, 64),
                                         candidate.tile.blockN));
    upsertCandidateAttr(attrs, ctx, "tile.block_k",
                        IntegerAttr::get(IntegerType::get(ctx, 64),
                                         candidate.tile.blockK));
  }
  if (candidate.threadSchedule.present) {
    upsertCandidateAttr(
        attrs, ctx, "thread_schedule.thread_count",
        IntegerAttr::get(IntegerType::get(ctx, 64),
                         candidate.threadSchedule.threadCount));
    upsertCandidateAttr(attrs, ctx, "thread_schedule.partition_axis",
                        stringAttr(candidate.threadSchedule.partitionAxis));
    upsertCandidateAttr(attrs, ctx, "thread_schedule.partition_strategy",
                        stringAttr(candidate.threadSchedule.partitionStrategy));
  }
  if (candidate.quantization.present) {
    upsertCandidateAttr(attrs, ctx, "quantization.scheme",
                        stringAttr(candidate.quantization.scheme));
    upsertCandidateAttr(attrs, ctx, "quantization.activation_dtype",
                        stringAttr(candidate.quantization.activationDtype));
    upsertCandidateAttr(attrs, ctx, "quantization.weight_dtype",
                        stringAttr(candidate.quantization.weightDtype));
    upsertCandidateAttr(attrs, ctx, "quantization.accumulator_dtype",
                        stringAttr(candidate.quantization.accumulatorDtype));
    upsertCandidateAttr(attrs, ctx, "quantization.output_dtype",
                        stringAttr(candidate.quantization.outputDtype));
    if (!candidate.quantization.granularity.empty())
      upsertCandidateAttr(attrs, ctx, "quantization.granularity",
                          stringAttr(candidate.quantization.granularity));
    if (candidate.quantization.groupSize > 0)
      upsertCandidateAttr(attrs, ctx, "quantization.group_size",
                          IntegerAttr::get(IntegerType::get(ctx, 64),
                                           candidate.quantization.groupSize));
    upsertCandidateAttr(attrs, ctx, "quantization.calibration_required",
                        stringAttr(candidate.quantization.calibrationRequired
                                       ? "true"
                                       : "false"));
    upsertCandidateAttr(attrs, ctx, "quantization.calibration_available",
                        stringAttr(candidate.quantization.calibrationAvailable
                                       ? "true"
                                       : "false"));
    if (!candidate.quantization.requiredBackendCapability.empty())
      upsertCandidateAttr(
          attrs, ctx, "quantization.required_backend_capability",
          stringAttr(candidate.quantization.requiredBackendCapability));
    if (!candidate.quantization.requiredKernelCapability.empty())
      upsertCandidateAttr(
          attrs, ctx, "quantization.required_kernel_capability",
          stringAttr(candidate.quantization.requiredKernelCapability));
  }
  if (!candidate.candidateReason.empty())
    upsertCandidateAttr(attrs, ctx, "candidate_reason",
                        stringAttr(candidate.candidateReason));
  upsertCandidateAttr(attrs, ctx, "required_boundary_ops",
                      makeCandidateStringArray(ctx,
                                               candidate.requiredBoundaryOps));
  if (!candidate.fallbackBackend.empty())
    upsertCandidateAttr(attrs, ctx, "fallback_backend",
                        stringAttr(candidate.fallbackBackend));
  if (!candidate.truthBoundary.empty())
    upsertCandidateAttr(attrs, ctx, "truth_boundary",
                        stringAttr(candidate.truthBoundary));

  upsertCandidateAttr(attrs, ctx, "feasibility.status",
                      stringAttr(stringifyFeasibilityStatus(
                          candidate.feasibility.status)));
  if (!candidate.feasibility.reason.empty())
    upsertCandidateAttr(attrs, ctx, "feasibility.reason",
                        stringAttr(candidate.feasibility.reason));
  if (!candidate.feasibility.requiredCapabilityRefs.empty())
    upsertCandidateAttr(
        attrs, ctx, "feasibility.required_capability_refs",
        makeCandidateStringArray(ctx,
                                 candidate.feasibility.requiredCapabilityRefs));

  return DictionaryAttr::get(ctx, attrs);
}

} // namespace mlir::hir
