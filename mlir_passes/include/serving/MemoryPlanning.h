#pragma once

#include "serving/ImplementationCandidate.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace mlir::hir {

struct ComputeUnitCapability {
  std::string identifier;
  std::string kind;
  std::vector<std::string> supportedCapabilityRefs;
  std::vector<std::string> supportedDtypes;
  std::vector<std::string> accessibleMemorySpaces;
};

struct MemorySpaceCapability {
  std::string identifier;
  std::string kind;
  int64_t capacityBytes = 0;
  bool capacityKnown = false;
  std::string addressSpaceCategory;
  std::vector<std::string> accessibleComputeUnits;
  int64_t requiredAlignment = 1;
  bool compilerManagedCapacity = false;
};

struct TransferPathCapability {
  std::string identifier;
  std::string sourceMemorySpace;
  std::string destinationMemorySpace;
  std::string mechanism;
  int64_t requiredAlignment = 1;
  bool synchronous = true;
  bool asynchronous = false;
  bool overlapWithCompute = false;
  int64_t maximumInFlight = 1;
};

struct TileMemoryRequest {
  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t tileK = 0;
  std::string activationDtype = "fp32";
  std::string weightDtype = "fp32";
  std::string outputDtype = "fp32";
  int64_t scratchBytes = 0;
  int64_t alignment = 1;
  bool doubleBufferInputAndWeight = false;
};

struct TileMemoryFootprint {
  bool ok = false;
  std::string reason;
  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t tileK = 0;
  int64_t inputTileBytes = 0;
  int64_t weightTileBytes = 0;
  int64_t outputTileBytes = 0;
  int64_t scratchBytes = 0;
  int64_t paddingBytes = 0;
  int64_t singleBufferBytes = 0;
  int64_t additionalDoubleBufferBytes = 0;
  int64_t totalRequiredLocalMemoryBytes = 0;
};

struct MemoryFeasibilityContext {
  ComputeUnitCapability computeUnit;
  MemorySpaceCapability requiredMemorySpace;
  std::vector<TransferPathCapability> transferPaths;
  int64_t requiredTileMultipleM = 1;
  int64_t requiredTileMultipleN = 1;
  int64_t requiredTileMultipleK = 1;
  bool requiresHostToLocalTransfers = false;
  std::string hostMemorySpace = "host";
};

inline bool memoryListContains(const std::vector<std::string> &values,
                               const std::string &needle) {
  return std::find(values.begin(), values.end(), needle) != values.end();
}

inline int64_t dtypeByteWidth(const std::string &dtype) {
  if (dtype == "fp32" || dtype == "f32" || dtype == "int32" || dtype == "i32") return 4;
  if (dtype == "fp16" || dtype == "f16" || dtype == "bf16") return 2;
  if (dtype == "int8" || dtype == "i8") return 1;
  if (dtype == "int4" || dtype == "i4") return 1; // storage is byte-packed at buffer granularity in Slice 2.
  return 0;
}

inline bool checkedMulI64(int64_t a, int64_t b, int64_t &out) {
  if (a < 0 || b < 0) return false;
  __int128 v = static_cast<__int128>(a) * static_cast<__int128>(b);
  if (v > std::numeric_limits<int64_t>::max()) return false;
  out = static_cast<int64_t>(v);
  return true;
}

inline bool checkedAddI64(int64_t a, int64_t b, int64_t &out) {
  if (a < 0 || b < 0) return false;
  __int128 v = static_cast<__int128>(a) + static_cast<__int128>(b);
  if (v > std::numeric_limits<int64_t>::max()) return false;
  out = static_cast<int64_t>(v);
  return true;
}

inline bool checkedAlignUp(int64_t value, int64_t alignment, int64_t &out) {
  if (value < 0 || alignment <= 0) return false;
  int64_t rem = value % alignment;
  if (rem == 0) { out = value; return true; }
  return checkedAddI64(value, alignment - rem, out);
}

inline bool checkedProduct3(int64_t a, int64_t b, int64_t c, int64_t &out) {
  int64_t t = 0;
  return checkedMulI64(a, b, t) && checkedMulI64(t, c, out);
}

inline TileMemoryFootprint calculateTileMemoryFootprint(
    const TileMemoryRequest &req) {
  TileMemoryFootprint fp;
  if (req.tileM <= 0 || req.tileN <= 0 || req.tileK <= 0) {
    fp.reason = "invalid_tile_shape";
    return fp;
  }
  fp.tileM = req.tileM;
  fp.tileN = req.tileN;
  fp.tileK = req.tileK;
  int64_t actBytes = dtypeByteWidth(req.activationDtype);
  int64_t weightBytes = dtypeByteWidth(req.weightDtype);
  int64_t outBytes = dtypeByteWidth(req.outputDtype);
  if (actBytes <= 0 || weightBytes <= 0 || outBytes <= 0) {
    fp.reason = "unsupported_dtype";
    return fp;
  }
  if (!checkedProduct3(req.tileM, req.tileK, actBytes, fp.inputTileBytes) ||
      !checkedProduct3(req.tileK, req.tileN, weightBytes, fp.weightTileBytes) ||
      !checkedProduct3(req.tileM, req.tileN, outBytes, fp.outputTileBytes)) {
    fp.reason = "byte_size_overflow";
    return fp;
  }
  fp.scratchBytes = req.scratchBytes;
  int64_t alignedInput = 0, alignedWeight = 0, alignedOutput = 0, alignedScratch = 0;
  if (!checkedAlignUp(fp.inputTileBytes, req.alignment, alignedInput) ||
      !checkedAlignUp(fp.weightTileBytes, req.alignment, alignedWeight) ||
      !checkedAlignUp(fp.outputTileBytes, req.alignment, alignedOutput) ||
      !checkedAlignUp(fp.scratchBytes, req.alignment, alignedScratch)) {
    fp.reason = "alignment_overflow";
    return fp;
  }
  fp.paddingBytes = (alignedInput - fp.inputTileBytes) +
                    (alignedWeight - fp.weightTileBytes) +
                    (alignedOutput - fp.outputTileBytes) +
                    (alignedScratch - fp.scratchBytes);
  int64_t sum1 = 0, sum2 = 0, sum3 = 0;
  if (!checkedAddI64(alignedInput, alignedWeight, sum1) ||
      !checkedAddI64(sum1, alignedOutput, sum2) ||
      !checkedAddI64(sum2, alignedScratch, fp.singleBufferBytes)) {
    fp.reason = "byte_size_overflow";
    return fp;
  }
  if (req.doubleBufferInputAndWeight) {
    if (!checkedAddI64(alignedInput, alignedWeight, fp.additionalDoubleBufferBytes)) {
      fp.reason = "byte_size_overflow";
      return fp;
    }
  }
  if (!checkedAddI64(fp.singleBufferBytes, fp.additionalDoubleBufferBytes, sum3)) {
    fp.reason = "byte_size_overflow";
    return fp;
  }
  fp.totalRequiredLocalMemoryBytes = sum3;
  fp.ok = true;
  fp.reason = "calculated";
  return fp;
}

inline CandidateFeasibilitySummary evaluateMemoryFeasibility(
    const TileMemoryFootprint &fp,
    const MemoryFeasibilityContext &ctx) {
  CandidateFeasibilitySummary summary;
  if (!fp.ok) {
    summary.status = CandidateFeasibilityStatus::Rejected;
    summary.reason = fp.reason;
    return summary;
  }
  if (ctx.requiredMemorySpace.identifier.empty()) {
    summary.status = CandidateFeasibilityStatus::Rejected;
    summary.reason = "missing_memory_space";
    return summary;
  }
  if (!memoryListContains(ctx.computeUnit.accessibleMemorySpaces,
                          ctx.requiredMemorySpace.identifier) ||
      !memoryListContains(ctx.requiredMemorySpace.accessibleComputeUnits,
                          ctx.computeUnit.identifier)) {
    summary.status = CandidateFeasibilityStatus::Rejected;
    summary.reason = "inaccessible_memory_space";
    return summary;
  }
  if (ctx.requiredMemorySpace.requiredAlignment > 1 &&
      fp.totalRequiredLocalMemoryBytes % ctx.requiredMemorySpace.requiredAlignment != 0) {
    summary.status = CandidateFeasibilityStatus::Rejected;
    summary.reason = "invalid_alignment";
    return summary;
  }
  if (ctx.requiredMemorySpace.capacityKnown &&
      fp.totalRequiredLocalMemoryBytes > ctx.requiredMemorySpace.capacityBytes) {
    summary.status = CandidateFeasibilityStatus::Rejected;
    summary.reason = "insufficient_memory_capacity";
    return summary;
  }
  if ((ctx.requiredTileMultipleM > 1 && fp.tileM % ctx.requiredTileMultipleM != 0) ||
      (ctx.requiredTileMultipleN > 1 && fp.tileN % ctx.requiredTileMultipleN != 0) ||
      (ctx.requiredTileMultipleK > 1 && fp.tileK % ctx.requiredTileMultipleK != 0)) {
    summary.status = CandidateFeasibilityStatus::Rejected;
    summary.reason = "invalid_tile_dimension_multiple";
    return summary;
  }
  if (ctx.requiresHostToLocalTransfers) {
    bool h2d = false, d2h = false;
    for (const auto &path : ctx.transferPaths) {
      if (path.sourceMemorySpace == ctx.hostMemorySpace &&
          path.destinationMemorySpace == ctx.requiredMemorySpace.identifier)
        h2d = true;
      if (path.sourceMemorySpace == ctx.requiredMemorySpace.identifier &&
          path.destinationMemorySpace == ctx.hostMemorySpace)
        d2h = true;
    }
    if (!h2d || !d2h) {
      summary.status = CandidateFeasibilityStatus::Rejected;
      summary.reason = "missing_transfer_path";
      return summary;
    }
  }
  summary.status = CandidateFeasibilityStatus::Feasible;
  summary.reason = "memory_feasible";
  return summary;
}

} // namespace mlir::hir
