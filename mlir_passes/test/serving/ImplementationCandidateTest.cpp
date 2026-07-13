#include "serving/ImplementationCandidate.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"

#include <cassert>
#include <cstdio>
#include <string>

using namespace mlir;
using namespace mlir::hir;

static std::string str(DictionaryAttr dict, llvm::StringRef key) {
  return getCandidateString(dict, key);
}

int main() {
  std::puts("=== ImplementationCandidateTest ===");

  MLIRContext ctx;

  ImplementationCandidate candidate;
  candidate.candidateId = "matmul:direct_lower";
  candidate.providerId = "candidate_generation_pass";
  candidate.scopeKind = CandidateScopeKind::Operator;
  candidate.semanticTargetRef = "matmul";
  candidate.implementationKind = "direct_lower";
  candidate.candidateReason = "kernel_exact_match";
  candidate.requiredBoundaryOps = {"cast"};
  candidate.truthBoundary =
      "candidate_generation_static_constraints_not_cost_evaluated";
  candidate.feasibility.status = CandidateFeasibilityStatus::Feasible;
  candidate.feasibility.reason = "kernel_exact_match";

  DictionaryAttr encoded = encodeImplementationCandidate(&ctx, candidate);
  assert(str(encoded, "candidate_id") == "matmul:direct_lower");
  assert(str(encoded, "scope_kind") == "operator");
  assert(str(encoded, "semantic_target_ref") == "matmul");
  assert(str(encoded, "candidate_type") == "direct_lower");
  assert(str(encoded, "implementation_kind") == "direct_lower");
  assert(str(encoded, "feasibility.status") == "feasible");

  ImplementationCandidate decoded =
      decodeImplementationCandidate(encoded, "test_provider");
  assert(decoded.candidateId == candidate.candidateId);
  assert(decoded.providerId == candidate.providerId);
  assert(decoded.scopeKind == CandidateScopeKind::Operator);
  assert(decoded.semanticTargetRef == "matmul");
  assert(decoded.implementationKind == "direct_lower");
  assert(decoded.requiredBoundaryOps.size() == 1);
  assert(decoded.requiredBoundaryOps[0] == "cast");
  assert(decoded.feasibility.status == CandidateFeasibilityStatus::Feasible);

  SmallVector<NamedAttribute> legacyAttrs;
  upsertCandidateAttr(legacyAttrs, &ctx, "candidate_type",
                      StringAttr::get(&ctx, "backend_fallback"));
  upsertCandidateAttr(legacyAttrs, &ctx, "source_op",
                      StringAttr::get(&ctx, "gelu"));
  upsertCandidateAttr(legacyAttrs, &ctx, "fallback_backend",
                      StringAttr::get(&ctx, "cpu_reference"));
  upsertCandidateAttr(legacyAttrs, &ctx, "required_boundary_ops",
                      makeCandidateStringArray(&ctx, {}));
  upsertCandidateAttr(legacyAttrs, &ctx, "evaluation.status",
                      StringAttr::get(&ctx, "evaluated"));
  upsertCandidateAttr(legacyAttrs, &ctx, "evaluation.reason",
                      StringAttr::get(&ctx, "backend_fallback_high_penalty"));
  upsertCandidateAttr(
      legacyAttrs, &ctx, "evaluation.penalty_score",
      IntegerAttr::get(IntegerType::get(&ctx, 64), 20));

  DictionaryAttr legacy = DictionaryAttr::get(&ctx, legacyAttrs);
  ImplementationCandidate legacyDecoded =
      decodeImplementationCandidate(legacy, "plan_selection_pass");
  assert(legacyDecoded.candidateId == "gelu:backend_fallback");
  assert(legacyDecoded.providerId == "plan_selection_pass");
  assert(legacyDecoded.scopeKind == CandidateScopeKind::Operator);
  assert(legacyDecoded.fallbackBackend == "cpu_reference");
  assert(legacyDecoded.cost.hasPenaltyScore);
  assert(legacyDecoded.cost.penaltyScore == 20);
  assert(legacyDecoded.feasibility.status ==
         CandidateFeasibilityStatus::Feasible);

  DictionaryAttr normalized =
      encodeImplementationCandidate(&ctx, legacyDecoded, legacy);
  assert(str(normalized, "candidate_id") == "gelu:backend_fallback");
  assert(str(normalized, "provider_id") == "plan_selection_pass");
  assert(str(normalized, "feasibility.status") == "feasible");
  assert(getCandidateI64(normalized, "evaluation.penalty_score") == 20);

  SmallVector<NamedAttribute> rejectedAttrs;
  upsertCandidateAttr(rejectedAttrs, &ctx, "candidate_type",
                      StringAttr::get(&ctx, "algebraic_decomposition"));
  upsertCandidateAttr(rejectedAttrs, &ctx, "source_op",
                      StringAttr::get(&ctx, "gelu"));
  upsertCandidateAttr(rejectedAttrs, &ctx, "rejection_reason",
                      StringAttr::get(&ctx, "missing_kernels:sigmoid"));
  DictionaryAttr rejected = DictionaryAttr::get(&ctx, rejectedAttrs);
  ImplementationCandidate rejectedDecoded =
      decodeImplementationCandidate(rejected, "candidate_generation_pass");
  assert(rejectedDecoded.feasibility.status ==
         CandidateFeasibilityStatus::Rejected);
  assert(rejectedDecoded.feasibility.reason == "missing_kernels:sigmoid");

  SmallVector<NamedAttribute> unsupportedAttrs;
  upsertCandidateAttr(unsupportedAttrs, &ctx, "candidate_type",
                      StringAttr::get(&ctx, "unsupported"));
  upsertCandidateAttr(unsupportedAttrs, &ctx, "source_op",
                      StringAttr::get(&ctx, "gelu"));
  upsertCandidateAttr(unsupportedAttrs, &ctx, "evaluation.status",
                      StringAttr::get(&ctx, "rejected"));
  upsertCandidateAttr(unsupportedAttrs, &ctx, "evaluation.reason",
                      StringAttr::get(&ctx, "no_viable_lowering_path"));
  DictionaryAttr unsupported = DictionaryAttr::get(&ctx, unsupportedAttrs);
  ImplementationCandidate unsupportedDecoded =
      decodeImplementationCandidate(unsupported, "plan_selection_pass");
  assert(unsupportedDecoded.feasibility.status ==
         CandidateFeasibilityStatus::Unsupported);

  PolicyResult result;
  result.selectedCandidateId = decoded.candidateId;
  result.consideredCandidateIds = {decoded.candidateId,
                                   legacyDecoded.candidateId};
  result.rejectedCandidates = {
      {legacyDecoded.candidateId, "not_lowest_ranked"}};
  result.policyId = "plan_selection_static_penalty_v1";
  result.selectionReason = "lowest_penalty_evaluated";
  result.truthBoundary = "plan_selection_static_penalty_not_measured_runtime";

  assert(result.selectedCandidateId == "matmul:direct_lower");
  assert(result.consideredCandidateIds.size() == 2);
  assert(result.rejectedCandidates[0].candidateId == "gelu:backend_fallback");

  ImplementationCandidate serialSchedule;
  serialSchedule.providerId = "kernel_selection_thread_schedule_candidates";
  serialSchedule.targetProfileId = "raspberry-pi5-cortex-a76-cpu";
  serialSchedule.scopeKind = CandidateScopeKind::FusedRegion;
  serialSchedule.semanticTargetRef = "fused_matmul_bias_relu";
  serialSchedule.implementationKind = "portable_cpu_opaque_kernel_thread_schedule";
  serialSchedule.kernelId =
      "portable_fused_matmul_bias_relu_bm32_bn128_bk32";
  serialSchedule.threadSchedule.present = true;
  serialSchedule.threadSchedule.threadCount = 1;
  serialSchedule.threadSchedule.partitionAxis = "none";
  serialSchedule.threadSchedule.partitionStrategy = "serial";
  serialSchedule.candidateId = makeFallbackCandidateId(serialSchedule);

  ImplementationCandidate parallelSchedule = serialSchedule;
  parallelSchedule.threadSchedule.threadCount = 4;
  parallelSchedule.threadSchedule.partitionAxis = "m";
  parallelSchedule.threadSchedule.partitionStrategy = "contiguous_chunks";
  parallelSchedule.candidateId = makeFallbackCandidateId(parallelSchedule);

  assert(serialSchedule.candidateId != parallelSchedule.candidateId);
  assert(serialSchedule.candidateId.find("threads=1") != std::string::npos);
  assert(parallelSchedule.candidateId.find("threads=4") != std::string::npos);
  assert(parallelSchedule.candidateId.find("axis=m") != std::string::npos);
  assert(parallelSchedule.candidateId.find(
             "portable_fused_matmul_bias_relu_bm32_bn128_bk32") !=
         std::string::npos);

  DictionaryAttr encodedSchedule =
      encodeImplementationCandidate(&ctx, parallelSchedule);
  ImplementationCandidate decodedSchedule =
      decodeImplementationCandidate(encodedSchedule, "test_provider");
  assert(decodedSchedule.candidateId == parallelSchedule.candidateId);
  assert(decodedSchedule.kernelId == parallelSchedule.kernelId);
  assert(decodedSchedule.threadSchedule.present);
  assert(decodedSchedule.threadSchedule.threadCount == 4);
  assert(decodedSchedule.threadSchedule.partitionAxis == "m");
  assert(decodedSchedule.threadSchedule.partitionStrategy ==
         "contiguous_chunks");

  ImplementationCandidate duplicateSchedule = parallelSchedule;
  duplicateSchedule.feasibility.reason = "different_state_must_not_change_id";
  assert(duplicateSchedule.candidateId == parallelSchedule.candidateId);

  std::puts("ImplementationCandidateTest passed");
  return 0;
}
