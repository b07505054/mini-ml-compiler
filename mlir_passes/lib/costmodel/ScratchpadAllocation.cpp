#include "costmodel/ScratchpadAllocation.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <map>
#include <optional>

namespace mlir::costmodel {

llvm::StringRef toString(MemoryRegionKind kind) {
  switch (kind) {
  case MemoryRegionKind::InputBuffer:
    return "InputBuffer";
  case MemoryRegionKind::WeightBuffer:
    return "WeightBuffer";
  case MemoryRegionKind::OutputBuffer:
    return "OutputBuffer";
  case MemoryRegionKind::Scratchpad:
    return "Scratchpad";
  }
  llvm_unreachable("unhandled MemoryRegionKind");
}

llvm::StringRef toString(BufferRole role) {
  switch (role) {
  case BufferRole::InputTile:
    return "InputTile";
  case BufferRole::WeightTile:
    return "WeightTile";
  case BufferRole::OutputAccumulator:
    return "OutputAccumulator";
  }
  llvm_unreachable("unhandled BufferRole");
}

bool BufferLifetime::operator==(const BufferLifetime &other) const {
  return bufferId == other.bufferId && role == other.role &&
         coordinate.mTile == other.coordinate.mTile && coordinate.nTile == other.coordinate.nTile &&
         coordinate.kTile == other.coordinate.kTile && sizeBytes == other.sizeBytes &&
         alignmentBytes == other.alignmentBytes &&
         firstOperationIndex == other.firstOperationIndex &&
         lastOperationIndex == other.lastOperationIndex && region == other.region;
}

bool BufferAllocation::operator==(const BufferAllocation &other) const {
  return bufferId == other.bufferId && region == other.region && offsetBytes == other.offsetBytes &&
         sizeBytes == other.sizeBytes && alignmentBytes == other.alignmentBytes &&
         firstOperationIndex == other.firstOperationIndex &&
         lastOperationIndex == other.lastOperationIndex;
}

bool ScratchpadAllocationPlan::operator==(const ScratchpadAllocationPlan &other) const {
  if (candidateId != other.candidateId || peakInputBufferBytes != other.peakInputBufferBytes ||
      peakWeightBufferBytes != other.peakWeightBufferBytes ||
      peakOutputBufferBytes != other.peakOutputBufferBytes ||
      peakAggregateScratchpadBytes != other.peakAggregateScratchpadBytes)
    return false;
  if (lifetimes.size() != other.lifetimes.size() || allocations.size() != other.allocations.size())
    return false;
  for (std::size_t i = 0; i < lifetimes.size(); ++i)
    if (!(lifetimes[i] == other.lifetimes[i]))
      return false;
  for (std::size_t i = 0; i < allocations.size(); ++i)
    if (!(allocations[i] == other.allocations[i]))
      return false;
  return true;
}

namespace {

llvm::Error makeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

MemoryRegionKind regionForRole(BufferRole role) {
  switch (role) {
  case BufferRole::InputTile:
    return MemoryRegionKind::InputBuffer;
  case BufferRole::WeightTile:
    return MemoryRegionKind::WeightBuffer;
  case BufferRole::OutputAccumulator:
    return MemoryRegionKind::OutputBuffer;
  }
  llvm_unreachable("unhandled BufferRole");
}

std::uint64_t countComputesInRange(const SchedulePlan &schedule, std::uint64_t first,
                                    std::uint64_t last) {
  std::uint64_t n = 0;
  for (const auto &op : schedule.operations)
    if (op.operationIndex >= first && op.operationIndex <= last &&
        op.kind == ScheduleOperationKind::ComputeTile)
      ++n;
  return n;
}

struct SimpleBuf {
  LogicalBufferId id;
  TileCoordinate coord;
  std::uint64_t sizeBytes = 0;
  std::uint64_t firstIndex = 0;
};

} // namespace

llvm::Expected<std::vector<BufferLifetime>> analyzeBufferLifetimes(const SchedulePlan &schedule,
                                                                    const NPUTargetConfig &target) {
  // ---- Section 6: dense/monotonic operation indices ----
  for (std::size_t i = 0; i < schedule.operations.size(); ++i)
    if (schedule.operations[i].operationIndex != i)
      return makeError("analyzeBufferLifetimes: operation indices are not dense and monotonic");

  if (target.dmaAlignmentBytes <= 0)
    return makeError("analyzeBufferLifetimes: target.dmaAlignmentBytes must be positive");
  std::uint64_t alignment = static_cast<std::uint64_t>(target.dmaAlignmentBytes);

  std::uint64_t nextBufferId = 0;
  std::vector<BufferLifetime> lifetimes;

  std::optional<SimpleBuf> openInput, openWeight, openAccum;
  std::uint64_t lastComputeIndexSeen = 0;
  bool anyComputeSeen = false;

  auto finalizeSimple = [&](std::optional<SimpleBuf> &open, BufferRole role) -> llvm::Error {
    if (!anyComputeSeen || lastComputeIndexSeen < open->firstIndex)
      return makeError("analyzeBufferLifetimes: a " + toString(role).str() +
                        " buffer was defined but never used by any ComputeTile");
    BufferLifetime bl;
    bl.bufferId = open->id;
    bl.role = role;
    bl.coordinate = open->coord;
    bl.sizeBytes = open->sizeBytes;
    bl.alignmentBytes = alignment;
    bl.firstOperationIndex = open->firstIndex;
    bl.lastOperationIndex = lastComputeIndexSeen;
    bl.region = regionForRole(role);
    lifetimes.push_back(bl);
    open.reset();
    return llvm::Error::success();
  };

  for (const ScheduleOperation &op : schedule.operations) {
    switch (op.kind) {
    case ScheduleOperationKind::LoadInputTile: {
      if (openInput) {
        if (llvm::Error err = finalizeSimple(openInput, BufferRole::InputTile))
          return std::move(err);
      }
      openInput = SimpleBuf{LogicalBufferId{nextBufferId++}, op.coordinate, op.bytes,
                            op.operationIndex};
      break;
    }
    case ScheduleOperationKind::LoadWeightTile: {
      if (openWeight) {
        if (llvm::Error err = finalizeSimple(openWeight, BufferRole::WeightTile))
          return std::move(err);
      }
      openWeight = SimpleBuf{LogicalBufferId{nextBufferId++}, op.coordinate, op.bytes,
                            op.operationIndex};
      break;
    }
    case ScheduleOperationKind::ReloadPartialOutput: {
      if (openAccum)
        return makeError(
            "analyzeBufferLifetimes: ReloadPartialOutput occurred while an accumulator was "
            "already live (missing prior store)");
      openAccum = SimpleBuf{LogicalBufferId{nextBufferId++}, op.coordinate, op.bytes,
                           op.operationIndex};
      break;
    }
    case ScheduleOperationKind::ComputeTile: {
      if (!openInput)
        return makeError("analyzeBufferLifetimes: ComputeTile with no live input buffer "
                          "(missing load before use)");
      if (!openWeight)
        return makeError("analyzeBufferLifetimes: ComputeTile with no live weight buffer "
                          "(missing load before use)");
      if (op.coordinate.kTile == 0) {
        if (!openAccum)
          openAccum = SimpleBuf{LogicalBufferId{nextBufferId++}, op.coordinate, op.bytes,
                               op.operationIndex};
      } else if (!openAccum) {
        return makeError(
            "analyzeBufferLifetimes: ComputeTile at kTile > 0 with no live accumulator "
            "(missing ReloadPartialOutput before use)");
      }
      lastComputeIndexSeen = op.operationIndex;
      anyComputeSeen = true;
      break;
    }
    case ScheduleOperationKind::StorePartialOutput:
    case ScheduleOperationKind::StoreFinalOutput: {
      if (!openAccum)
        return makeError("analyzeBufferLifetimes: store operation with no live accumulator "
                          "buffer");
      BufferLifetime bl;
      bl.bufferId = openAccum->id;
      bl.role = BufferRole::OutputAccumulator;
      bl.coordinate = openAccum->coord;
      // Always size the accumulator from the CLOSING store's own bytes
      // (always the correct O value per Slice 6). The value captured at
      // open time is only reliable when opened via ReloadPartialOutput
      // (bytes=O); when opened implicitly at a k==0 ComputeTile, the
      // captured value would be 0 (ComputeTile.bytes is always 0), so it
      // must never be used here.
      bl.sizeBytes = op.bytes;
      bl.alignmentBytes = alignment;
      bl.firstOperationIndex = openAccum->firstIndex;
      bl.lastOperationIndex = op.operationIndex;
      bl.region = MemoryRegionKind::OutputBuffer;
      lifetimes.push_back(bl);
      openAccum.reset();
      break;
    }
    case ScheduleOperationKind::Synchronize:
      break;
    }
  }

  // ---- finalize any trailing input/weight buffer ----
  // Input/weight buffers close on "the next load of the same role, or
  // end of schedule" (see the header doc comment) -- the very LAST such
  // buffer never sees a "next load", so finalizing it here at the end of
  // the scan is the NORMAL, expected closure path, not an error.
  if (openInput) {
    if (llvm::Error err = finalizeSimple(openInput, BufferRole::InputTile))
      return std::move(err);
  }
  if (openWeight) {
    if (llvm::Error err = finalizeSimple(openWeight, BufferRole::WeightTile))
      return std::move(err);
  }
  // The accumulator, in contrast, has an explicit REQUIRED closing event
  // (StorePartialOutput/StoreFinalOutput) -- if one is still open here,
  // the schedule is genuinely malformed (Section 6: "no buffer remains
  // live beyond the final operation" / missing final store).
  if (openAccum)
    return makeError("analyzeBufferLifetimes: accumulator buffer remains live past the final "
                      "operation (missing final store?)");

  // ---- residency flags agree with observed lifetime spans ----
  for (const BufferLifetime &bl : lifetimes) {
    std::uint64_t computes = countComputesInRange(schedule, bl.firstOperationIndex,
                                                  bl.lastOperationIndex);
    std::uint64_t expected = 1;
    bool residentRole = (bl.role == BufferRole::WeightTile &&
                          schedule.dataflow == Dataflow::WeightStationary) ||
                        (bl.role == BufferRole::InputTile &&
                          schedule.dataflow == Dataflow::InputStationary) ||
                        (bl.role == BufferRole::OutputAccumulator &&
                          schedule.dataflow == Dataflow::OutputStationary);
    if (residentRole) {
      switch (bl.role) {
      case BufferRole::WeightTile:
        expected = schedule.numMTiles;
        break;
      case BufferRole::InputTile:
        expected = schedule.numNTiles;
        break;
      case BufferRole::OutputAccumulator:
        expected = schedule.numKTiles;
        break;
      }
    }
    if (computes != expected)
      return makeError("analyzeBufferLifetimes: a " + toString(bl.role).str() +
                        " lifetime spans " + llvm::Twine(computes) + " compute(s), expected " +
                        llvm::Twine(expected) +
                        " (schedule residency flags disagree with observed lifetime span)");
  }

  std::sort(lifetimes.begin(), lifetimes.end(),
            [](const BufferLifetime &a, const BufferLifetime &b) {
              return a.bufferId.value < b.bufferId.value;
            });
  return lifetimes;
}

namespace {

bool interferes(const BufferLifetime &a, const BufferLifetime &b) {
  return !(a.lastOperationIndex < b.firstOperationIndex ||
           b.lastOperationIndex < a.firstOperationIndex);
}

// Returns false (leaving `result` untouched) on overflow -- never wraps
// silently.
bool alignUpChecked(std::uint64_t value, std::uint64_t alignment, std::uint64_t &result) {
  if (alignment <= 1) {
    result = value;
    return true;
  }
  std::uint64_t remainder = value % alignment;
  if (remainder == 0) {
    result = value;
    return true;
  }
  return !__builtin_add_overflow(value, alignment - remainder, &result);
}

// Deterministic aligned first-fit over one region's buffers (Section 10).
llvm::Expected<std::vector<BufferAllocation>>
allocateRegion(std::vector<BufferLifetime> region) {
  std::sort(region.begin(), region.end(), [](const BufferLifetime &a, const BufferLifetime &b) {
    if (a.firstOperationIndex != b.firstOperationIndex)
      return a.firstOperationIndex < b.firstOperationIndex;
    if (a.coordinate.mTile != b.coordinate.mTile)
      return a.coordinate.mTile < b.coordinate.mTile;
    if (a.coordinate.nTile != b.coordinate.nTile)
      return a.coordinate.nTile < b.coordinate.nTile;
    if (a.coordinate.kTile != b.coordinate.kTile)
      return a.coordinate.kTile < b.coordinate.kTile;
    return a.bufferId.value < b.bufferId.value;
  });

  std::vector<BufferAllocation> live; // currently-live placed allocations
  std::vector<BufferAllocation> all;

  for (const BufferLifetime &bl : region) {
    if (bl.sizeBytes == 0)
      return makeError("allocateRegion: buffer '" + llvm::Twine(bl.bufferId.value) +
                        "' has zero size");
    if (bl.alignmentBytes == 0)
      return makeError("allocateRegion: buffer '" + llvm::Twine(bl.bufferId.value) +
                        "' has zero alignment");

    // Expire allocations that cannot interfere with this buffer.
    std::vector<BufferAllocation> stillLive;
    for (const BufferAllocation &placed : live)
      if (!(placed.lastOperationIndex < bl.firstOperationIndex))
        stillLive.push_back(placed);
    live = std::move(stillLive);

    std::sort(live.begin(), live.end(), [](const BufferAllocation &a, const BufferAllocation &b) {
      return a.offsetBytes < b.offsetBytes;
    });

    std::uint64_t candidate = 0;
    if (!alignUpChecked(0, bl.alignmentBytes, candidate))
      return makeError("allocateRegion: address arithmetic overflow aligning the initial offset");
    for (const BufferAllocation &placed : live) {
      std::uint64_t candidateEnd = 0;
      if (__builtin_add_overflow(candidate, bl.sizeBytes, &candidateEnd))
        return makeError("allocateRegion: address arithmetic overflow computing candidate end");
      if (candidateEnd <= placed.offsetBytes)
        break; // fits before this placed allocation
      std::uint64_t placedEnd = 0;
      if (__builtin_add_overflow(placed.offsetBytes, placed.sizeBytes, &placedEnd))
        return makeError("allocateRegion: address arithmetic overflow computing placed end");
      if (!alignUpChecked(placedEnd, bl.alignmentBytes, candidate))
        return makeError("allocateRegion: address arithmetic overflow aligning the next "
                          "candidate offset");
    }

    std::uint64_t checkEnd = 0;
    if (__builtin_add_overflow(candidate, bl.sizeBytes, &checkEnd))
      return makeError("allocateRegion: address arithmetic overflow placing buffer");

    BufferAllocation alloc;
    alloc.bufferId = bl.bufferId;
    alloc.region = bl.region;
    alloc.offsetBytes = candidate;
    alloc.sizeBytes = bl.sizeBytes;
    alloc.alignmentBytes = bl.alignmentBytes;
    alloc.firstOperationIndex = bl.firstOperationIndex;
    alloc.lastOperationIndex = bl.lastOperationIndex;

    live.push_back(alloc);
    all.push_back(alloc);
  }

  std::sort(all.begin(), all.end(), [](const BufferAllocation &a, const BufferAllocation &b) {
    return a.bufferId.value < b.bufferId.value;
  });
  return all;
}

} // namespace

namespace {

llvm::Expected<ScratchpadAllocationPlan>
buildAllocationPlan(const std::string &candidateId, std::vector<BufferLifetime> lifetimes) {
  std::vector<BufferLifetime> inputRegion, weightRegion, outputRegion;
  for (const BufferLifetime &bl : lifetimes) {
    switch (bl.region) {
    case MemoryRegionKind::InputBuffer:
      inputRegion.push_back(bl);
      break;
    case MemoryRegionKind::WeightBuffer:
      weightRegion.push_back(bl);
      break;
    case MemoryRegionKind::OutputBuffer:
      outputRegion.push_back(bl);
      break;
    case MemoryRegionKind::Scratchpad:
      return makeError("allocateScratchpad: a buffer lifetime was assigned directly to the "
                        "aggregate Scratchpad region, which is never a placement target");
    }
  }

  llvm::Expected<std::vector<BufferAllocation>> inputAllocOrErr = allocateRegion(inputRegion);
  if (!inputAllocOrErr)
    return inputAllocOrErr.takeError();
  llvm::Expected<std::vector<BufferAllocation>> weightAllocOrErr = allocateRegion(weightRegion);
  if (!weightAllocOrErr)
    return weightAllocOrErr.takeError();
  llvm::Expected<std::vector<BufferAllocation>> outputAllocOrErr = allocateRegion(outputRegion);
  if (!outputAllocOrErr)
    return outputAllocOrErr.takeError();

  ScratchpadAllocationPlan plan;
  plan.candidateId = candidateId;
  plan.lifetimes = std::move(lifetimes);
  plan.allocations.reserve(inputAllocOrErr->size() + weightAllocOrErr->size() +
                          outputAllocOrErr->size());
  plan.allocations.insert(plan.allocations.end(), inputAllocOrErr->begin(), inputAllocOrErr->end());
  plan.allocations.insert(plan.allocations.end(), weightAllocOrErr->begin(),
                          weightAllocOrErr->end());
  plan.allocations.insert(plan.allocations.end(), outputAllocOrErr->begin(),
                          outputAllocOrErr->end());
  std::sort(plan.allocations.begin(), plan.allocations.end(),
            [](const BufferAllocation &a, const BufferAllocation &b) {
              return a.bufferId.value < b.bufferId.value;
            });

  // ---- Section 11: peak bytes via a genuine timeline sweep over
  // concurrently-live sizes (never allocator-offset-dependent). ----
  std::map<std::uint64_t, std::array<std::int64_t, 3>> events; // index -> per-region delta
  auto addEvents = [&](const std::vector<BufferLifetime> &region, int idx) {
    for (const BufferLifetime &bl : region) {
      events[bl.firstOperationIndex][idx] += static_cast<std::int64_t>(bl.sizeBytes);
      std::uint64_t deathIndex = bl.lastOperationIndex + 1; // one past the last live index
      events[deathIndex][idx] -= static_cast<std::int64_t>(bl.sizeBytes);
    }
  };
  addEvents(inputRegion, 0);
  addEvents(weightRegion, 1);
  addEvents(outputRegion, 2);

  std::array<std::int64_t, 3> running = {0, 0, 0};
  std::uint64_t peakInput = 0, peakWeight = 0, peakOutput = 0, peakAggregate = 0;
  for (const auto &[index, deltas] : events) {
    (void)index;
    for (int i = 0; i < 3; ++i)
      running[i] += deltas[i];
    peakInput = std::max<std::uint64_t>(peakInput, static_cast<std::uint64_t>(running[0]));
    peakWeight = std::max<std::uint64_t>(peakWeight, static_cast<std::uint64_t>(running[1]));
    peakOutput = std::max<std::uint64_t>(peakOutput, static_cast<std::uint64_t>(running[2]));
    std::uint64_t aggregate = static_cast<std::uint64_t>(running[0] + running[1] + running[2]);
    peakAggregate = std::max(peakAggregate, aggregate);
  }

  plan.peakInputBufferBytes = peakInput;
  plan.peakWeightBufferBytes = peakWeight;
  plan.peakOutputBufferBytes = peakOutput;
  plan.peakAggregateScratchpadBytes = peakAggregate;
  return plan;
}

} // namespace

llvm::Expected<ScratchpadAllocationPlan> allocateScratchpad(const SchedulePlan &schedule,
                                                            const NPUTargetConfig &target) {
  llvm::Expected<std::vector<BufferLifetime>> lifetimesOrErr =
      analyzeBufferLifetimes(schedule, target);
  if (!lifetimesOrErr)
    return lifetimesOrErr.takeError();
  return buildAllocationPlan(schedule.candidateId, std::move(*lifetimesOrErr));
}

llvm::Expected<ScratchpadAllocationPlan>
allocateScratchpadFromLifetimes(const std::string &candidateId,
                                std::vector<BufferLifetime> lifetimes) {
  return buildAllocationPlan(candidateId, std::move(lifetimes));
}

llvm::Error validateAllocationAgainstTarget(const ScratchpadAllocationPlan &plan,
                                            const NPUTargetConfig &target) {
  if (plan.candidateId.empty())
    return makeError("validateAllocationAgainstTarget: plan.candidateId is empty");

  if (plan.peakInputBufferBytes > target.inputBufferBytes)
    return makeError("validateAllocationAgainstTarget: peakInputBufferBytes exceeds "
                      "target.inputBufferBytes");
  if (plan.peakWeightBufferBytes > target.weightBufferBytes)
    return makeError("validateAllocationAgainstTarget: peakWeightBufferBytes exceeds "
                      "target.weightBufferBytes");
  if (plan.peakOutputBufferBytes > target.outputBufferBytes)
    return makeError("validateAllocationAgainstTarget: peakOutputBufferBytes exceeds "
                      "target.outputBufferBytes");
  if (plan.peakAggregateScratchpadBytes > target.scratchpadBytes)
    return makeError("validateAllocationAgainstTarget: peakAggregateScratchpadBytes exceeds "
                      "target.scratchpadBytes");

  auto capacityFor = [&](MemoryRegionKind region) -> std::uint64_t {
    switch (region) {
    case MemoryRegionKind::InputBuffer:
      return target.inputBufferBytes;
    case MemoryRegionKind::WeightBuffer:
      return target.weightBufferBytes;
    case MemoryRegionKind::OutputBuffer:
      return target.outputBufferBytes;
    case MemoryRegionKind::Scratchpad:
      return target.scratchpadBytes;
    }
    llvm_unreachable("unhandled MemoryRegionKind");
  };

  std::map<std::uint64_t, const BufferAllocation *> byId;
  for (const BufferAllocation &alloc : plan.allocations) {
    if (byId.count(alloc.bufferId.value))
      return makeError("validateAllocationAgainstTarget: duplicate allocation for buffer id " +
                        llvm::Twine(alloc.bufferId.value));
    byId[alloc.bufferId.value] = &alloc;

    if (alloc.alignmentBytes == 0 || alloc.offsetBytes % alloc.alignmentBytes != 0)
      return makeError("validateAllocationAgainstTarget: buffer id " +
                        llvm::Twine(alloc.bufferId.value) + " has a misaligned offset");

    std::uint64_t end = 0;
    if (__builtin_add_overflow(alloc.offsetBytes, alloc.sizeBytes, &end))
      return makeError("validateAllocationAgainstTarget: address overflow for buffer id " +
                        llvm::Twine(alloc.bufferId.value));
    if (end > capacityFor(alloc.region))
      return makeError("validateAllocationAgainstTarget: buffer id " +
                        llvm::Twine(alloc.bufferId.value) +
                        " ends outside its region's declared capacity");
  }

  // Every lifetime has exactly one allocation; no unknown allocations.
  std::map<std::uint64_t, const BufferLifetime *> lifetimeById;
  for (const BufferLifetime &bl : plan.lifetimes) {
    if (lifetimeById.count(bl.bufferId.value))
      return makeError("validateAllocationAgainstTarget: duplicate lifetime for buffer id " +
                        llvm::Twine(bl.bufferId.value));
    lifetimeById[bl.bufferId.value] = &bl;
  }
  for (const auto &[id, lifetime] : lifetimeById) {
    (void)lifetime;
    if (!byId.count(id))
      return makeError("validateAllocationAgainstTarget: buffer id " + llvm::Twine(id) +
                        " has a lifetime but no allocation");
  }
  for (const auto &[id, alloc] : byId) {
    (void)alloc;
    if (!lifetimeById.count(id))
      return makeError("validateAllocationAgainstTarget: buffer id " + llvm::Twine(id) +
                        " has an allocation but no corresponding lifetime (unknown allocation)");
  }

  // No interfering same-region allocations overlap.
  for (std::size_t i = 0; i < plan.allocations.size(); ++i) {
    for (std::size_t j = i + 1; j < plan.allocations.size(); ++j) {
      const BufferAllocation &a = plan.allocations[i];
      const BufferAllocation &b = plan.allocations[j];
      if (a.region != b.region)
        continue;
      bool timesOverlap =
          !(a.lastOperationIndex < b.firstOperationIndex || b.lastOperationIndex < a.firstOperationIndex);
      if (!timesOverlap)
        continue;
      bool addressesOverlap = a.offsetBytes < b.offsetBytes + b.sizeBytes &&
                              b.offsetBytes < a.offsetBytes + a.sizeBytes;
      if (addressesOverlap)
        return makeError("validateAllocationAgainstTarget: interfering allocations for buffer "
                          "ids " +
                          llvm::Twine(a.bufferId.value) + " and " +
                          llvm::Twine(b.bufferId.value) + " overlap in address range");
    }
  }

  return llvm::Error::success();
}

llvm::Error validateAllocationAgainstLegality(const ScratchpadAllocationPlan &allocation,
                                              const CandidateLegalityResult &legality) {
  if (allocation.candidateId != legality.candidateId)
    return makeError("validateAllocationAgainstLegality: allocation.candidateId does not match "
                      "legality.candidateId");
  if (allocation.peakInputBufferBytes > static_cast<std::uint64_t>(legality.inputTileBytes))
    return makeError("validateAllocationAgainstLegality: peakInputBufferBytes exceeds the "
                      "Slice 2 legality input-tile footprint");
  if (allocation.peakWeightBufferBytes > static_cast<std::uint64_t>(legality.weightTileBytes))
    return makeError("validateAllocationAgainstLegality: peakWeightBufferBytes exceeds the "
                      "Slice 2 legality weight-tile footprint");
  if (allocation.peakOutputBufferBytes > static_cast<std::uint64_t>(legality.outputTileBytes))
    return makeError("validateAllocationAgainstLegality: peakOutputBufferBytes exceeds the "
                      "Slice 2 legality output-tile footprint");
  if (allocation.peakAggregateScratchpadBytes >
      static_cast<std::uint64_t>(legality.totalScratchpadBytes))
    return makeError("validateAllocationAgainstLegality: peakAggregateScratchpadBytes exceeds "
                      "the Slice 2 legality total-scratchpad footprint");
  return llvm::Error::success();
}

std::vector<AddressedScheduleOperation>
buildAddressedSchedule(const SchedulePlan &schedule, const ScratchpadAllocationPlan &allocation) {
  std::vector<AddressedScheduleOperation> result;
  result.reserve(schedule.operations.size());

  auto findLive = [&](MemoryRegionKind region,
                      std::uint64_t opIndex) -> std::optional<BufferAllocation> {
    for (const BufferAllocation &alloc : allocation.allocations)
      if (alloc.region == region && alloc.firstOperationIndex <= opIndex &&
          opIndex <= alloc.lastOperationIndex)
        return alloc;
    return std::nullopt;
  };

  for (const ScheduleOperation &op : schedule.operations) {
    AddressedScheduleOperation aop;
    aop.operation = op;
    aop.input = findLive(MemoryRegionKind::InputBuffer, op.operationIndex);
    aop.weight = findLive(MemoryRegionKind::WeightBuffer, op.operationIndex);
    aop.output = findLive(MemoryRegionKind::OutputBuffer, op.operationIndex);
    result.push_back(std::move(aop));
  }
  return result;
}

std::string serializeScratchpadAllocationPlanToJson(const ScratchpadAllocationPlan &plan) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "{\n";
  os << "  \"candidate_id\": \"" << plan.candidateId << "\",\n";
  os << "  \"peak_bytes\": {\"input\": " << plan.peakInputBufferBytes
     << ", \"weight\": " << plan.peakWeightBufferBytes
     << ", \"output\": " << plan.peakOutputBufferBytes
     << ", \"aggregate\": " << plan.peakAggregateScratchpadBytes << "},\n";
  os << "  \"allocations\": [\n";
  for (std::size_t i = 0; i < plan.allocations.size(); ++i) {
    const BufferAllocation &a = plan.allocations[i];
    // find matching role from lifetimes for diagnostic purposes
    llvm::StringRef role = "Unknown";
    for (const BufferLifetime &bl : plan.lifetimes)
      if (bl.bufferId.value == a.bufferId.value) {
        role = toString(bl.role);
        break;
      }
    os << "    {\"buffer_id\": " << a.bufferId.value << ", \"role\": \"" << role
       << "\", \"region\": \"" << toString(a.region) << "\", \"offset\": " << a.offsetBytes
       << ", \"size\": " << a.sizeBytes << ", \"first_op\": " << a.firstOperationIndex
       << ", \"last_op\": " << a.lastOperationIndex << "}";
    if (i + 1 < plan.allocations.size())
      os << ",";
    os << "\n";
  }
  os << "  ]\n";
  os << "}";
  os.flush();
  return out;
}

} // namespace mlir::costmodel
