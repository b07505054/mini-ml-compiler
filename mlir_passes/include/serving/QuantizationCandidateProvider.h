#pragma once

#include "serving/ImplementationCandidate.h"

#include <algorithm>
#include <string>
#include <vector>

namespace mlir::hir {

struct QuantizationCapabilityContext {
  std::string semanticTargetRef;
  CandidateScopeKind scopeKind = CandidateScopeKind::Operator;
  std::string targetProfileId;
  std::string backend;
  std::string kernelId;
  std::string activationDtype = "fp32";
  std::string outputDtype = "fp32";
  bool fp32FallbackAllowed = true;
  std::vector<std::string> supportedQuantizationSchemes;
  std::vector<std::string> supportedActivationDtypes;
  std::vector<std::string> supportedWeightDtypes;
  std::vector<std::string> supportedAccumulatorDtypes;
  std::vector<std::string> supportedOutputDtypes;
  std::vector<std::string> supportedGranularities;
  std::vector<int64_t> supportedGroupSizes;
  bool supportsPerChannel = false;
  std::vector<std::string> calibrationAvailableSchemes;
  std::vector<std::string> requiredKernelCapabilities;
  std::vector<std::string> runtimeKernelQuantModes;
  std::vector<std::string> runtimeKernelDtypes;
  std::string truthBoundary = "quantization_candidate_policy_declared_capabilities_not_accuracy_calibrated";
};

struct QuantizationProviderResult {
  std::vector<ImplementationCandidate> candidates;
  PolicyResult policy;
};

inline bool quantListContains(const std::vector<std::string> &values,
                              const std::string &needle) {
  return std::find(values.begin(), values.end(), needle) != values.end();
}

inline bool quantDtypeSupported(const std::vector<std::string> &values,
                                const std::string &dtype) {
  if (values.empty()) return false;
  if (quantListContains(values, dtype)) return true;
  if (dtype == "fp32" && quantListContains(values, "f32")) return true;
  if (dtype == "f32" && quantListContains(values, "fp32")) return true;
  if (dtype == "fp16" && quantListContains(values, "f16")) return true;
  if (dtype == "f16" && quantListContains(values, "fp16")) return true;
  if (dtype == "int8" && quantListContains(values, "i8")) return true;
  if (dtype == "int4" && quantListContains(values, "i4")) return true;
  return false;
}

class QuantizationCandidateProvider {
public:
  std::string providerId() const { return "quantization_candidate_provider"; }
  std::string providerVersion() const { return "slice1.v1"; }

  QuantizationProviderResult enumerateAndSelect(
      const QuantizationCapabilityContext &ctx) const {
    QuantizationProviderResult result;
    result.candidates = enumerateCandidates(ctx);
    result.policy = select(result.candidates);
    return result;
  }

  std::vector<ImplementationCandidate> enumerateCandidates(
      const QuantizationCapabilityContext &ctx) const {
    std::vector<ImplementationCandidate> out;
    out.push_back(makeCandidate(ctx, "fp32_baseline", "fp32", "fp32",
                                "fp32", "fp32", "per_tensor", 0,
                                false, "quantization.scheme.fp32_baseline",
                                "quant_kernel.none"));
    out.push_back(makeCandidate(ctx, "fp16", "fp16", "fp16", "fp32",
                                "fp16", "per_tensor", 0, false,
                                "quantization.scheme.fp16",
                                "quant_kernel.fp16"));
    out.push_back(makeCandidate(ctx, "int8_static_symmetric", "int8", "int8",
                                "int32", "fp32", "per_tensor", 0, true,
                                "quantization.scheme.int8_static_symmetric",
                                "quant_kernel.int8_static_symmetric"));
    out.push_back(makeCandidate(ctx, "int8_static", "int8", "int8",
                                "int32", "fp32", "per_channel", 0, true,
                                "quantization.scheme.int8_static",
                                "quant_kernel.int8_static"));
    out.push_back(makeCandidate(ctx, "int4_weight_only", "fp16", "int4",
                                "fp32", "fp16", "per_group", 128, true,
                                "quantization.scheme.int4_weight_only",
                                "quant_kernel.int4_weight_only"));
    for (auto &candidate : out)
      candidate.feasibility = evaluate(candidate, ctx);
    return out;
  }

  CandidateFeasibilitySummary evaluate(
      const ImplementationCandidate &candidate,
      const QuantizationCapabilityContext &ctx) const {
    CandidateFeasibilitySummary summary;
    const auto &q = candidate.quantization;
    summary.requiredCapabilityRefs = {q.requiredBackendCapability,
                                      q.requiredKernelCapability};
    if (!q.present) {
      summary.status = CandidateFeasibilityStatus::Rejected;
      summary.reason = "missing_quantization_candidate";
      return summary;
    }

    if (q.scheme == "fp32_baseline") {
      bool dtypeOk = quantDtypeSupported(ctx.runtimeKernelDtypes, "fp32") ||
                     quantDtypeSupported(ctx.supportedActivationDtypes, "fp32") ||
                     quantDtypeSupported(ctx.supportedWeightDtypes, "fp32") ||
                     ctx.fp32FallbackAllowed;
      bool modeOk = ctx.runtimeKernelQuantModes.empty() ||
                    quantListContains(ctx.runtimeKernelQuantModes, "none");
      summary.status = (dtypeOk && modeOk) ? CandidateFeasibilityStatus::Feasible
                                           : CandidateFeasibilityStatus::Unsupported;
      summary.reason = (dtypeOk && modeOk) ? "supported" : "unsupported";
      return summary;
    }

    if (!quantListContains(ctx.supportedQuantizationSchemes, q.scheme)) {
      summary.status = CandidateFeasibilityStatus::Unsupported;
      summary.reason = "unsupported";
      return summary;
    }
    if (!quantDtypeSupported(ctx.supportedActivationDtypes, q.activationDtype) ||
        !quantDtypeSupported(ctx.supportedWeightDtypes, q.weightDtype) ||
        !quantDtypeSupported(ctx.supportedAccumulatorDtypes, q.accumulatorDtype) ||
        !quantDtypeSupported(ctx.supportedOutputDtypes, q.outputDtype)) {
      summary.status = CandidateFeasibilityStatus::Unsupported;
      summary.reason = "unsupported_dtype";
      return summary;
    }
    if ((q.granularity == "per_channel" && !ctx.supportsPerChannel) ||
        (!q.granularity.empty() && !ctx.supportedGranularities.empty() &&
         !quantListContains(ctx.supportedGranularities, q.granularity))) {
      summary.status = CandidateFeasibilityStatus::Unsupported;
      summary.reason = "unsupported_granularity";
      return summary;
    }
    if (q.groupSize > 0 &&
        std::find(ctx.supportedGroupSizes.begin(), ctx.supportedGroupSizes.end(),
                  q.groupSize) == ctx.supportedGroupSizes.end()) {
      summary.status = CandidateFeasibilityStatus::Rejected;
      summary.reason = "invalid_group_size";
      return summary;
    }
    if (!quantListContains(ctx.requiredKernelCapabilities,
                           q.requiredKernelCapability)) {
      summary.status = CandidateFeasibilityStatus::Rejected;
      summary.reason = "missing_kernel_capability";
      return summary;
    }
    if (q.calibrationRequired &&
        !quantListContains(ctx.calibrationAvailableSchemes, q.scheme)) {
      summary.status = CandidateFeasibilityStatus::Rejected;
      summary.reason = "requires_unavailable_calibration";
      return summary;
    }
    summary.status = CandidateFeasibilityStatus::Feasible;
    summary.reason = "supported";
    return summary;
  }

private:
  ImplementationCandidate makeCandidate(
      const QuantizationCapabilityContext &ctx, const std::string &scheme,
      const std::string &activationDtype, const std::string &weightDtype,
      const std::string &accumulatorDtype, const std::string &outputDtype,
      const std::string &granularity, int64_t groupSize,
      bool calibrationRequired, const std::string &backendCapability,
      const std::string &kernelCapability) const {
    ImplementationCandidate candidate;
    candidate.providerId = providerId();
    candidate.targetProfileId = ctx.targetProfileId;
    candidate.scopeKind = ctx.scopeKind;
    candidate.semanticTargetRef = ctx.semanticTargetRef;
    candidate.backend = ctx.backend;
    candidate.implementationKind = "quantization_configuration";
    candidate.runtimeContractKind = "execution_plan_quantization_contract_v1";
    candidate.kernelId = ctx.kernelId;
    candidate.dtype = outputDtype;
    candidate.truthBoundary = ctx.truthBoundary;
    candidate.quantization.present = true;
    candidate.quantization.scheme = scheme;
    candidate.quantization.activationDtype = activationDtype;
    candidate.quantization.weightDtype = weightDtype;
    candidate.quantization.accumulatorDtype = accumulatorDtype;
    candidate.quantization.outputDtype = outputDtype;
    candidate.quantization.granularity = granularity;
    candidate.quantization.groupSize = groupSize;
    candidate.quantization.calibrationRequired = calibrationRequired;
    candidate.quantization.calibrationAvailable =
        quantListContains(ctx.calibrationAvailableSchemes, scheme);
    candidate.quantization.requiredBackendCapability = backendCapability;
    candidate.quantization.requiredKernelCapability = kernelCapability;
    candidate.candidateReason = "generated_quantization_candidate";
    candidate.candidateId = makeFallbackCandidateId(candidate);
    return candidate;
  }

  PolicyResult select(const std::vector<ImplementationCandidate> &candidates) const {
    PolicyResult policy;
    policy.policyId = "quantization_rule_policy_slice1_v1";
    policy.objectiveSummary = "prefer highest-priority legal quantization candidate; preserve fp32 fallback";
    policy.truthBoundary = "deterministic_rule_policy_no_accuracy_or_performance_model";
    for (const auto &candidate : candidates) {
      policy.consideredCandidateIds.push_back(candidate.candidateId);
      if (candidate.feasibility.status != CandidateFeasibilityStatus::Feasible)
        policy.rejectedCandidates.push_back({candidate.candidateId,
                                             candidate.feasibility.reason});
    }
    static const char *priority[] = {"int4_weight_only", "int8_static_symmetric",
                                     "int8_static", "fp16", "fp32_baseline"};
    for (const char *scheme : priority) {
      for (const auto &candidate : candidates) {
        if (candidate.quantization.present && candidate.quantization.scheme == scheme &&
            candidate.feasibility.status == CandidateFeasibilityStatus::Feasible) {
          policy.selectedCandidateId = candidate.candidateId;
          policy.selectionReason = candidate.quantization.scheme == "fp32_baseline"
                                       ? "fallback_fp32_selected"
                                       : "selected_highest_priority_legal_candidate";
          return policy;
        }
      }
    }
    policy.selectionReason = "no_legal_quantization_candidate";
    return policy;
  }
};

} // namespace mlir::hir
