#include "serving/ImplementationCandidate.h"
#include "serving/MemoryPlanning.h"
#include "serving/PortableCPUProvider.h"
#include "serving/QuantizationCandidateProvider.h"
#include "serving/XNNPACKCandidateProvider.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"

#include <cassert>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

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
  serialSchedule.backend = "cpu";
  serialSchedule.implementationKind = "opaque_portable_cpu_native_kernel";
  serialSchedule.runtimeContractKind = "portable_cpu_kernel_adapter_contract";
  serialSchedule.kernelId =
      "portable_fused_matmul_bias_relu_bm32_bn128_bk32";
  serialSchedule.dtype = "fp32";
  serialSchedule.tile.present = true;
  serialSchedule.tile.blockM = 32;
  serialSchedule.tile.blockN = 128;
  serialSchedule.tile.blockK = 32;
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
  assert(serialSchedule.candidateId.find("scope=fused_region") !=
         std::string::npos);
  assert(serialSchedule.candidateId.find("backend=cpu") !=
         std::string::npos);
  assert(serialSchedule.candidateId.find(
             "opaque_portable_cpu_native_kernel") != std::string::npos);
  assert(serialSchedule.candidateId.find(
             "contract=portable_cpu_kernel_adapter_contract") !=
         std::string::npos);
  assert(serialSchedule.candidateId.find("tile=bm32_bn128_bk32") !=
         std::string::npos);
  assert(serialSchedule.candidateId.find("dtype=fp32") !=
         std::string::npos);
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
  assert(decodedSchedule.backend == parallelSchedule.backend);
  assert(decodedSchedule.implementationKind ==
         parallelSchedule.implementationKind);
  assert(decodedSchedule.runtimeContractKind ==
         parallelSchedule.runtimeContractKind);
  assert(decodedSchedule.kernelId == parallelSchedule.kernelId);
  assert(decodedSchedule.dtype == parallelSchedule.dtype);
  assert(decodedSchedule.tile.present);
  assert(decodedSchedule.tile.blockM == 32);
  assert(decodedSchedule.tile.blockN == 128);
  assert(decodedSchedule.tile.blockK == 32);
  assert(decodedSchedule.threadSchedule.present);
  assert(decodedSchedule.threadSchedule.threadCount == 4);
  assert(decodedSchedule.threadSchedule.partitionAxis == "m");
  assert(decodedSchedule.threadSchedule.partitionStrategy ==
         "contiguous_chunks");

  ImplementationCandidate duplicateSchedule = parallelSchedule;
  duplicateSchedule.feasibility.reason = "different_state_must_not_change_id";
  assert(duplicateSchedule.candidateId == parallelSchedule.candidateId);

  ImplementationCandidate differentKernel = parallelSchedule;
  differentKernel.kernelId =
      "portable_fused_matmul_bias_relu_bm32_bn32_bk32";
  differentKernel.candidateId = makeFallbackCandidateId(differentKernel);
  assert(differentKernel.candidateId != parallelSchedule.candidateId);

  ImplementationCandidate differentTile = parallelSchedule;
  differentTile.tile.blockN = 32;
  differentTile.candidateId = makeFallbackCandidateId(differentTile);
  assert(differentTile.candidateId != parallelSchedule.candidateId);

  ImplementationCandidate differentDtype = parallelSchedule;
  differentDtype.dtype = "fp16";
  differentDtype.candidateId = makeFallbackCandidateId(differentDtype);
  assert(differentDtype.candidateId != parallelSchedule.candidateId);

  ImplementationCandidate sameCompleteIdentity = parallelSchedule;
  sameCompleteIdentity.cost.hasPenaltyScore = true;
  sameCompleteIdentity.cost.penaltyScore = 999;
  assert(makeFallbackCandidateId(sameCompleteIdentity) ==
         parallelSchedule.candidateId);

  PortableCPUProvider provider;
  assert(provider.providerId() == "portable_cpu_provider");
  assert(provider.providerVersion() == "a4.v1");

  PortableCpuProviderContext providerCtx;
  providerCtx.semanticTargetRef = "fused_matmul_bias_relu";
  providerCtx.scopeKind = CandidateScopeKind::FusedRegion;
  providerCtx.hasStaticShape = true;
  providerCtx.targetProfileId = "raspberry-pi5-cortex-a76-cpu";
  providerCtx.backend = "cpu";
  providerCtx.dtype = "fp32";
  providerCtx.truthBoundary = "kernel_selection_static_descriptor_match_not_runtime_execution";

  PortableCpuRuntimeKernelDescriptor descriptor;
  descriptor.kernelId =
      "portable_fused_matmul_bias_relu_bm32_bn128_bk32";
  descriptor.opName = "fused_matmul_bias_relu";
  descriptor.backend = "cpu";
  descriptor.supportedDtypes = {"fp32"};
  descriptor.supportedThreadSchedules = {
      {1, "none", "serial"},
      {4, "m", "contiguous_chunks"},
      {2, "m", "contiguous_chunks"},
      {4, "n", "contiguous_chunks"}};
  descriptor.truthBoundary = "test_descriptor";

  assert(provider.supportsScope(providerCtx));
  PortableCpuProviderResult providerResult =
      provider.enumerateCandidates(providerCtx, descriptor);
  assert(providerResult.candidates.size() == 2);
  assert(providerResult.candidates[0].candidate.providerId ==
         "portable_cpu_provider");
  assert(providerResult.candidates[1].candidate.providerId ==
         "portable_cpu_provider");
  assert(providerResult.candidates[0].candidate.semanticTargetRef ==
         providerResult.candidates[1].candidate.semanticTargetRef);
  assert(providerResult.candidates[0].candidate.kernelId ==
         providerResult.candidates[1].candidate.kernelId);
  assert(providerResult.candidates[0].candidate.tile.blockM == 32);
  assert(providerResult.candidates[0].candidate.tile.blockN == 128);
  assert(providerResult.candidates[0].candidate.tile.blockK == 32);
  assert(providerResult.candidates[0].candidate.dtype == "fp32");
  assert(providerResult.candidates[1].candidate.dtype == "fp32");
  assert(providerResult.candidates[0].candidate.candidateId !=
         providerResult.candidates[1].candidate.candidateId);
  assert(providerResult.candidates[0].schedule.threadCount == 1);
  assert(providerResult.candidates[1].schedule.threadCount == 4);
  assert(providerResult.candidates[0].candidate.feasibility.status ==
         CandidateFeasibilityStatus::Unknown);
  assert(providerResult.candidates[1].candidate.feasibility.status ==
         CandidateFeasibilityStatus::Unknown);

  for (const auto& emitted : providerResult.candidates) {
    assert(emitted.schedule.threadCount == 1 ||
           emitted.schedule.threadCount == 4);
    assert(emitted.schedule.partitionAxis != "n");
    assert(emitted.candidate.kernelId.find("bm32_bn128_bk32") !=
           std::string::npos);
  }

  PortableCpuProviderContext largeShapeSameContext = providerCtx;
  PortableCpuProviderResult providerResultLarge =
      provider.enumerateCandidates(largeShapeSameContext, descriptor);
  assert(providerResultLarge.candidates.size() == providerResult.candidates.size());
  assert(providerResultLarge.candidates[0].candidate.candidateId ==
         providerResult.candidates[0].candidate.candidateId);
  assert(providerResultLarge.candidates[1].candidate.candidateId ==
         providerResult.candidates[1].candidate.candidateId);

  PortableCPUFeasibilityEvaluator feasibilityEvaluator;
  PortableCpuFeasibilityContext feasibilityCtx;
  feasibilityCtx.semanticTargetRef = providerCtx.semanticTargetRef;
  feasibilityCtx.scopeKind = providerCtx.scopeKind;
  feasibilityCtx.hasStaticShape = true;
  feasibilityCtx.targetProfileId = providerCtx.targetProfileId;
  feasibilityCtx.backend = providerCtx.backend;
  feasibilityCtx.dtype = providerCtx.dtype;
  feasibilityCtx.physicalComputeUnits = 4;
  CandidateFeasibilitySummary serialFeasibility =
      feasibilityEvaluator.evaluate(providerResult.candidates[0].candidate,
                                    feasibilityCtx);
  CandidateFeasibilitySummary parallelFeasibility =
      feasibilityEvaluator.evaluate(providerResult.candidates[1].candidate,
                                    feasibilityCtx);
  assert(serialFeasibility.status == CandidateFeasibilityStatus::Feasible);
  assert(parallelFeasibility.status == CandidateFeasibilityStatus::Feasible);

  PortableCpuFeasibilityContext insufficientCompute = feasibilityCtx;
  insufficientCompute.physicalComputeUnits = 2;
  CandidateFeasibilitySummary rejectedParallel =
      feasibilityEvaluator.evaluate(providerResult.candidates[1].candidate,
                                    insufficientCompute);
  assert(rejectedParallel.status == CandidateFeasibilityStatus::Rejected);
  assert(rejectedParallel.reason == "rejected_exceeds_compute_units");

  PortableCpuFeasibilityContext missingShape = feasibilityCtx;
  missingShape.hasStaticShape = false;
  CandidateFeasibilitySummary deferredSerial =
      feasibilityEvaluator.evaluate(providerResult.candidates[0].candidate,
                                    missingShape);
  assert(deferredSerial.status == CandidateFeasibilityStatus::Deferred);
  assert(deferredSerial.reason == "missing_static_shape");

  PortableCpuProviderContext wrongOpCtx = providerCtx;
  wrongOpCtx.semanticTargetRef = "rmsnorm";
  PortableCpuProviderResult wrongOp =
      provider.enumerateCandidates(wrongOpCtx, descriptor);
  assert(wrongOp.candidates.empty());
  assert(!wrongOp.diagnostics.empty());
  assert(wrongOp.diagnostics[0].reason == "unsupported_semantic_scope");

  PortableCpuRuntimeKernelDescriptor wrongDtype = descriptor;
  wrongDtype.supportedDtypes = {"fp16"};
  PortableCpuProviderResult wrongDtypeResult =
      provider.enumerateCandidates(providerCtx, wrongDtype);
  assert(wrongDtypeResult.candidates.empty());
  assert(!wrongDtypeResult.diagnostics.empty());
  assert(wrongDtypeResult.diagnostics[0].reason == "wrong_dtype");

  PortableCpuRuntimeKernelDescriptor missingParallel = descriptor;
  missingParallel.supportedThreadSchedules = {{1, "none", "serial"}};
  PortableCpuProviderResult missingParallelResult =
      provider.enumerateCandidates(providerCtx, missingParallel);
  assert(missingParallelResult.candidates.size() == 1);
  bool sawMissingParallel = false;
  for (const auto& diagnostic : missingParallelResult.diagnostics)
    if (diagnostic.reason == "missing_parallel_schedule")
      sawMissingParallel = true;
  assert(sawMissingParallel);

  PortableCpuRuntimeKernelDescriptor tileMismatch = descriptor;
  tileMismatch.supportedTileShapes = {"64x64x64"};
  PortableCpuProviderResult tileMismatchResult =
      provider.enumerateCandidates(providerCtx, tileMismatch);
  assert(tileMismatchResult.candidates.empty());
  assert(!tileMismatchResult.diagnostics.empty());
  assert(tileMismatchResult.diagnostics[0].reason ==
         "kernel_tile_identity_mismatch");


  XNNPACKCandidateProvider xnnProvider;
  assert(xnnProvider.providerId() ==
         "executorch_xnnpack_candidate_provider");
  assert(xnnProvider.providerVersion() == "e3a.v1");

  XNNPACKProviderContext xctx;
  xctx.semanticTargetRef = "fused_matmul_bias_relu";
  xctx.scopeKind = CandidateScopeKind::FusedRegion;
  xctx.hasStaticShape = true;
  xctx.targetProfileId = "raspberry-pi5-cortex-a76-cpu";
  xctx.backend = "cpu";
  xctx.dtype = "fp32";
  xctx.pteArtifactRef = "fused_matmul_bias_relu_64x64x64_xnnpack.pte";
  xctx.pteSha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  xctx.runnerSha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
  xctx.executorchTag = "v1.3.1";
  xctx.executorchCommit = "e2f18eb23c45bd22ca332b0b8b49a81de304b472";
  xctx.xnnpackCommit = "1adaa7c709d4839d29e1f219cb962b01c9e6a905";
  xctx.xnnpackDelegationProven = true;
  xctx.inputBindingCompatible = true;
  xctx.truthBoundary = "test_xnnpack_provider_enumerates_only";

  XNNPACKProviderResult xres = xnnProvider.enumerateCandidates(xctx);
  assert(xres.diagnostics.empty());
  assert(xres.candidates.size() == 2);
  assert(xres.candidates[0].requestedThreadCount == 1);
  assert(xres.candidates[1].requestedThreadCount == 4);
  assert(xres.candidates[0].candidate.candidateId !=
         xres.candidates[1].candidate.candidateId);
  assert(xres.candidates[0].candidate.library == "xnnpack");
  assert(xres.candidates[0].candidate.runtimeContractKind ==
         "executorch_xnnpack_runner_contract");
  assert(xres.candidates[0].candidate.kernelId.empty());
  assert(xres.candidates[0].candidate.threadSchedule.partitionAxis ==
         "runtime_threadpool");

  DictionaryAttr encodedXnn =
      encodeImplementationCandidate(&ctx, xres.candidates[1].candidate);
  ImplementationCandidate decodedXnn =
      decodeImplementationCandidate(encodedXnn, "test_provider");
  assert(decodedXnn.library == "xnnpack");
  assert(decodedXnn.pteSha256 == xctx.pteSha256);
  assert(decodedXnn.runnerSha256 == xctx.runnerSha256);
  assert(decodedXnn.executorchCommit == xctx.executorchCommit);
  assert(decodedXnn.xnnpackCommit == xctx.xnnpackCommit);
  assert(decodedXnn.threadSchedule.threadCount == 4);

  XNNPACKFeasibilityContext xfctx;
  xfctx.semanticTargetRef = xctx.semanticTargetRef;
  xfctx.scopeKind = xctx.scopeKind;
  xfctx.hasStaticShape = true;
  xfctx.targetProfileId = xctx.targetProfileId;
  xfctx.backend = xctx.backend;
  xfctx.dtype = xctx.dtype;
  xfctx.expectedPteSha256 = xctx.pteSha256;
  xfctx.expectedRunnerSha256 = xctx.runnerSha256;
  xfctx.expectedExecutorchCommit = xctx.executorchCommit;
  xfctx.expectedXNNPACKCommit = xctx.xnnpackCommit;
  xfctx.xnnpackDelegationProven = true;
  xfctx.inputBindingCompatible = true;
  xfctx.physicalComputeUnits = 4;
  XNNPACKFeasibilityEvaluator xevaluator;
  CandidateFeasibilitySummary x1 =
      xevaluator.evaluate(xres.candidates[0].candidate, xfctx);
  CandidateFeasibilitySummary x4 =
      xevaluator.evaluate(xres.candidates[1].candidate, xfctx);
  assert(x1.status == CandidateFeasibilityStatus::Feasible);
  assert(x4.status == CandidateFeasibilityStatus::Feasible);

  XNNPACKFeasibilityContext wrongPte = xfctx;
  wrongPte.expectedPteSha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
  CandidateFeasibilitySummary wrongPteSummary =
      xevaluator.evaluate(xres.candidates[0].candidate, wrongPte);
  assert(wrongPteSummary.status == CandidateFeasibilityStatus::Rejected);
  assert(wrongPteSummary.reason == "pte_hash_mismatch");

  XNNPACKProviderContext wrongXnnOp = xctx;
  wrongXnnOp.semanticTargetRef = "rmsnorm";
  XNNPACKProviderResult wrongXnnOpResult =
      xnnProvider.enumerateCandidates(wrongXnnOp);
  assert(wrongXnnOpResult.candidates.empty());
  assert(wrongXnnOpResult.diagnostics[0].reason ==
         "unsupported_semantic_scope");

  XNNPACKProviderContext wrongXnnDtype = xctx;
  wrongXnnDtype.dtype = "fp16";
  XNNPACKProviderResult wrongXnnDtypeResult =
      xnnProvider.enumerateCandidates(wrongXnnDtype);
  assert(wrongXnnDtypeResult.candidates.empty());
  assert(wrongXnnDtypeResult.diagnostics[0].reason == "wrong_dtype");



  QuantizationCandidateProvider qprovider;
  assert(qprovider.providerId() == "quantization_candidate_provider");
  QuantizationCapabilityContext qctx;
  qctx.semanticTargetRef = "matmul";
  qctx.scopeKind = CandidateScopeKind::Operator;
  qctx.targetProfileId = "synthetic-quant-slice1";
  qctx.backend = "synthetic_cpu";
  qctx.kernelId = "synthetic_matmul_fp32";
  qctx.supportedQuantizationSchemes = {"fp16", "int8_static", "int4_weight_only"};
  qctx.supportedActivationDtypes = {"fp32", "fp16", "int8"};
  qctx.supportedWeightDtypes = {"fp32", "fp16", "int8", "int4"};
  qctx.supportedAccumulatorDtypes = {"fp32", "int32"};
  qctx.supportedOutputDtypes = {"fp32", "fp16"};
  qctx.supportedGranularities = {"per_tensor", "per_channel", "per_group"};
  qctx.supportsPerChannel = true;
  qctx.supportedGroupSizes = {64};
  qctx.requiredKernelCapabilities = {"quant_kernel.none", "quant_kernel.fp16", "quant_kernel.int8_static"};
  qctx.runtimeKernelQuantModes = {"none"};
  qctx.runtimeKernelDtypes = {"f32"};

  QuantizationProviderResult qres = qprovider.enumerateAndSelect(qctx);
  assert(qres.candidates.size() == 4);
  bool sawFp32 = false, sawFp16 = false, sawInt8RejectCalibration = false;
  bool sawInt4RejectGroup = false;
  for (const auto &qcand : qres.candidates) {
    if (qcand.quantization.scheme == "fp32_baseline") {
      sawFp32 = true;
      assert(qcand.feasibility.status == CandidateFeasibilityStatus::Feasible);
    }
    if (qcand.quantization.scheme == "fp16") {
      sawFp16 = true;
      assert(qcand.feasibility.status == CandidateFeasibilityStatus::Feasible);
    }
    if (qcand.quantization.scheme == "int8_static") {
      assert(qcand.quantization.calibrationRequired);
      sawInt8RejectCalibration =
          qcand.feasibility.reason == "requires_unavailable_calibration";
    }
    if (qcand.quantization.scheme == "int4_weight_only")
      sawInt4RejectGroup = qcand.feasibility.reason == "invalid_group_size";
  }
  assert(sawFp32 && sawFp16 && sawInt8RejectCalibration && sawInt4RejectGroup);
  assert(!qres.policy.selectedCandidateId.empty());
  const ImplementationCandidate *qselected = nullptr;
  for (const auto &qcand : qres.candidates)
    if (qcand.candidateId == qres.policy.selectedCandidateId)
      qselected = &qcand;
  assert(qselected && qselected->quantization.scheme == "fp16");
  assert(qres.policy.rejectedCandidates.size() == 2);

  QuantizationCapabilityContext qctxCalibrated = qctx;
  qctxCalibrated.calibrationAvailableSchemes = {"int8_static"};
  QuantizationProviderResult qresCalibrated =
      qprovider.enumerateAndSelect(qctxCalibrated);
  const ImplementationCandidate *qselectedCalibrated = nullptr;
  for (const auto &qcand : qresCalibrated.candidates)
    if (qcand.candidateId == qresCalibrated.policy.selectedCandidateId)
      qselectedCalibrated = &qcand;
  assert(qselectedCalibrated &&
         qselectedCalibrated->quantization.scheme == "int8_static");

  QuantizationCapabilityContext qctxInt4 = qctxCalibrated;
  qctxInt4.supportedGroupSizes = {128};
  qctxInt4.requiredKernelCapabilities.push_back("quant_kernel.int4_weight_only");
  qctxInt4.calibrationAvailableSchemes.push_back("int4_weight_only");
  QuantizationProviderResult qresInt4 = qprovider.enumerateAndSelect(qctxInt4);
  const ImplementationCandidate *qselectedInt4 = nullptr;
  for (const auto &qcand : qresInt4.candidates)
    if (qcand.candidateId == qresInt4.policy.selectedCandidateId)
      qselectedInt4 = &qcand;
  assert(qselectedInt4 &&
         qselectedInt4->quantization.scheme == "int4_weight_only");

  DictionaryAttr encodedQuant =
      encodeImplementationCandidate(&ctx, *qselectedInt4);
  ImplementationCandidate decodedQuant =
      decodeImplementationCandidate(encodedQuant, "test_provider");
  assert(decodedQuant.quantization.present);
  assert(decodedQuant.quantization.scheme == "int4_weight_only");
  assert(decodedQuant.quantization.groupSize == 128);
  assert(decodedQuant.quantization.requiredKernelCapability ==
         "quant_kernel.int4_weight_only");

  assert(dtypeByteWidth("fp32") == 4);
  assert(dtypeByteWidth("f32") == 4);
  assert(dtypeByteWidth("fp16") == 2);
  assert(dtypeByteWidth("bf16") == 2);
  assert(dtypeByteWidth("int8") == 1);
  assert(dtypeByteWidth("int4") == 1);
  assert(dtypeByteWidth("unknown_dtype") == 0);

  TileMemoryRequest memReq;
  memReq.tileM = 32;
  memReq.tileN = 64;
  memReq.tileK = 16;
  memReq.activationDtype = "fp32";
  memReq.weightDtype = "fp32";
  memReq.outputDtype = "fp32";
  memReq.alignment = 64;
  TileMemoryFootprint fp = calculateTileMemoryFootprint(memReq);
  assert(fp.ok);
  assert(fp.inputTileBytes == 32 * 16 * 4);
  assert(fp.weightTileBytes == 16 * 64 * 4);
  assert(fp.outputTileBytes == 32 * 64 * 4);
  assert(fp.scratchBytes == 0);
  assert(fp.paddingBytes == 0);
  assert(fp.singleBufferBytes ==
         fp.inputTileBytes + fp.weightTileBytes + fp.outputTileBytes);
  assert(fp.additionalDoubleBufferBytes == 0);
  assert(fp.totalRequiredLocalMemoryBytes == fp.singleBufferBytes);

  TileMemoryRequest paddedReq;
  paddedReq.tileM = 3;
  paddedReq.tileN = 5;
  paddedReq.tileK = 7;
  paddedReq.activationDtype = "fp16";
  paddedReq.weightDtype = "int8";
  paddedReq.outputDtype = "fp32";
  paddedReq.scratchBytes = 13;
  paddedReq.alignment = 16;
  paddedReq.doubleBufferInputAndWeight = true;
  TileMemoryFootprint padded = calculateTileMemoryFootprint(paddedReq);
  assert(padded.ok);
  assert(padded.inputTileBytes == 3 * 7 * 2);
  assert(padded.weightTileBytes == 7 * 5);
  assert(padded.outputTileBytes == 3 * 5 * 4);
  assert(padded.scratchBytes == 13);
  assert(padded.singleBufferBytes == 48 + 48 + 64 + 16);
  assert(padded.additionalDoubleBufferBytes == 48 + 48);
  assert(padded.totalRequiredLocalMemoryBytes == 272);

  TileMemoryRequest overflowReq = memReq;
  overflowReq.tileM = std::numeric_limits<int64_t>::max();
  overflowReq.tileN = std::numeric_limits<int64_t>::max();
  overflowReq.tileK = std::numeric_limits<int64_t>::max();
  TileMemoryFootprint overflowFp =
      calculateTileMemoryFootprint(overflowReq);
  assert(!overflowFp.ok);
  assert(overflowFp.reason == "byte_size_overflow");

  MemoryFeasibilityContext memCtx;
  memCtx.computeUnit.identifier = "synthetic_accelerator";
  memCtx.computeUnit.kind = "accelerator";
  memCtx.computeUnit.accessibleMemorySpaces = {"local_sram"};
  memCtx.requiredMemorySpace.identifier = "local_sram";
  memCtx.requiredMemorySpace.kind = "local_sram";
  memCtx.requiredMemorySpace.accessibleComputeUnits =
      {"synthetic_accelerator"};
  memCtx.requiredMemorySpace.requiredAlignment = 1;
  memCtx.requiredMemorySpace.capacityKnown = true;
  memCtx.requiredMemorySpace.compilerManagedCapacity = true;

  MemoryFeasibilityContext smallMemCtx = memCtx;
  smallMemCtx.requiredMemorySpace.capacityBytes =
      fp.totalRequiredLocalMemoryBytes - 1;
  CandidateFeasibilitySummary smallMem =
      evaluateMemoryFeasibility(fp, smallMemCtx);
  assert(smallMem.status == CandidateFeasibilityStatus::Rejected);
  assert(smallMem.reason == "insufficient_memory_capacity");

  MemoryFeasibilityContext exactMemCtx = memCtx;
  exactMemCtx.requiredMemorySpace.capacityBytes =
      fp.totalRequiredLocalMemoryBytes;
  CandidateFeasibilitySummary exactMem =
      evaluateMemoryFeasibility(fp, exactMemCtx);
  assert(exactMem.status == CandidateFeasibilityStatus::Feasible);

  MemoryFeasibilityContext largeMemCtx = memCtx;
  largeMemCtx.requiredMemorySpace.capacityBytes =
      fp.totalRequiredLocalMemoryBytes + 1;
  CandidateFeasibilitySummary largeMem =
      evaluateMemoryFeasibility(fp, largeMemCtx);
  assert(largeMem.status == CandidateFeasibilityStatus::Feasible);

  MemoryFeasibilityContext missingSpaceCtx = memCtx;
  missingSpaceCtx.requiredMemorySpace.identifier.clear();
  CandidateFeasibilitySummary missingSpace =
      evaluateMemoryFeasibility(fp, missingSpaceCtx);
  assert(missingSpace.status == CandidateFeasibilityStatus::Rejected);
  assert(missingSpace.reason == "missing_memory_space");

  MemoryFeasibilityContext inaccessibleCtx = memCtx;
  inaccessibleCtx.computeUnit.accessibleMemorySpaces = {"host"};
  CandidateFeasibilitySummary inaccessible =
      evaluateMemoryFeasibility(fp, inaccessibleCtx);
  assert(inaccessible.status == CandidateFeasibilityStatus::Rejected);
  assert(inaccessible.reason == "inaccessible_memory_space");

  MemoryFeasibilityContext invalidMultipleCtx = largeMemCtx;
  invalidMultipleCtx.requiredTileMultipleM = 64;
  CandidateFeasibilitySummary invalidMultiple =
      evaluateMemoryFeasibility(fp, invalidMultipleCtx);
  assert(invalidMultiple.status == CandidateFeasibilityStatus::Rejected);
  assert(invalidMultiple.reason == "invalid_tile_dimension_multiple");

  MemoryFeasibilityContext missingTransferCtx = largeMemCtx;
  missingTransferCtx.requiresHostToLocalTransfers = true;
  missingTransferCtx.hostMemorySpace = "host";
  CandidateFeasibilitySummary missingTransfer =
      evaluateMemoryFeasibility(fp, missingTransferCtx);
  assert(missingTransfer.status == CandidateFeasibilityStatus::Rejected);
  assert(missingTransfer.reason == "missing_transfer_path");

  MemoryFeasibilityContext transferCtx = missingTransferCtx;
  transferCtx.transferPaths = {
      {"synthetic_h2l", "host", "local_sram", "synthetic_copy", 64,
       true, false, false, 1},
      {"synthetic_l2h", "local_sram", "host", "synthetic_copy", 64,
       true, false, false, 1},
  };
  CandidateFeasibilitySummary transferOk =
      evaluateMemoryFeasibility(fp, transferCtx);
  assert(transferOk.status == CandidateFeasibilityStatus::Feasible);

  std::puts("ImplementationCandidateTest passed");
  return 0;
}
