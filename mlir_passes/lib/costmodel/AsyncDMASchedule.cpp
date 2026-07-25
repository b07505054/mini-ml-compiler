#include "costmodel/AsyncDMASchedule.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <set>

namespace mlir::costmodel {

// ===========================================================================
// toString / small helpers
// ===========================================================================

llvm::StringRef toString(DMASchedulingPolicy policy) {
  switch (policy) {
  case DMASchedulingPolicy::Synchronous:
    return "synchronous";
  case DMASchedulingPolicy::PrefetchNext:
    return "prefetch_next";
  }
  llvm_unreachable("unhandled DMASchedulingPolicy");
}

const std::vector<DMASchedulingPolicy> &allDMASchedulingPolicies() {
  static const std::vector<DMASchedulingPolicy> kPolicies = {
      DMASchedulingPolicy::Synchronous,
      DMASchedulingPolicy::PrefetchNext,
  };
  return kPolicies;
}

llvm::StringRef toString(BufferSlot slot) {
  switch (slot) {
  case BufferSlot::Single:
    return "single";
  case BufferSlot::Ping:
    return "ping";
  case BufferSlot::Pong:
    return "pong";
  }
  llvm_unreachable("unhandled BufferSlot");
}

llvm::StringRef toString(AsyncOperationKind kind) {
  switch (kind) {
  case AsyncOperationKind::DMAIssue:
    return "dma_issue";
  case AsyncOperationKind::DMAWait:
    return "dma_wait";
  case AsyncOperationKind::Compute:
    return "compute";
  case AsyncOperationKind::ComputeBarrier:
    return "compute_barrier";
  }
  llvm_unreachable("unhandled AsyncOperationKind");
}

llvm::StringRef toString(DMATransferKind kind) {
  switch (kind) {
  case DMATransferKind::InputLoad:
    return "input_load";
  case DMATransferKind::WeightLoad:
    return "weight_load";
  case DMATransferKind::PartialOutputReload:
    return "partial_output_reload";
  case DMATransferKind::PartialOutputStore:
    return "partial_output_store";
  case DMATransferKind::FinalOutputStore:
    return "final_output_store";
  }
  llvm_unreachable("unhandled DMATransferKind");
}

std::string extendedCandidateIdSuffix(const BufferingPolicy &buffering,
                                      DMASchedulingPolicy dmaPolicy) {
  std::string s;
  s += ":buf_i=";
  s += toString(buffering.input).str();
  s += ":buf_w=";
  s += toString(buffering.weight).str();
  s += ":buf_o=";
  s += toString(buffering.output).str();
  s += ":dma=";
  s += toString(dmaPolicy).str();
  return s;
}

std::vector<ExtendedKernelCandidate>
generateExtendedCandidates(llvm::ArrayRef<KernelCandidate> candidates) {
  std::vector<ExtendedKernelCandidate> result;
  result.reserve(candidates.size() * allBufferingPolicies().size() *
                allDMASchedulingPolicies().size());
  for (const KernelCandidate &c : candidates) {
    for (const BufferingPolicy &b : allBufferingPolicies()) {
      for (DMASchedulingPolicy dma : allDMASchedulingPolicies()) {
        ExtendedKernelCandidate ext;
        ext.candidate = c;
        ext.buffering = b;
        ext.dmaPolicy = dma;
        ext.candidateId = c.candidateId + extendedCandidateIdSuffix(b, dma);
        result.push_back(std::move(ext));
      }
    }
  }
  return result;
}

namespace {

llvm::Error makeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

struct CheckedArith {
  bool overflowed = false;
  std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    std::uint64_t r = 0;
    if (__builtin_mul_overflow(a, b, &r))
      overflowed = true;
    return r;
  }
  std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    std::uint64_t r = 0;
    if (__builtin_add_overflow(a, b, &r))
      overflowed = true;
    return r;
  }
  std::uint64_t ceilDiv(std::uint64_t a, std::uint64_t b) {
    if (b == 0) {
      overflowed = true;
      return 0;
    }
    return a / b + (a % b != 0 ? 1 : 0);
  }
};

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

// ceil(bytes * clockHz / bandwidth), 128-bit intermediate (same philosophy
// as Slice 3's dmaTransferCycles -- reused formula, not reinvented).
bool transferCyclesChecked(std::uint64_t bytes, std::uint64_t clockHz, std::uint64_t bandwidth,
                            std::uint64_t &result) {
  if (bandwidth == 0)
    return false;
  __uint128_t num = static_cast<__uint128_t>(bytes) * static_cast<__uint128_t>(clockHz);
  __uint128_t cycles = num / bandwidth + (num % bandwidth != 0 ? 1 : 0);
  if (cycles > std::numeric_limits<std::uint64_t>::max())
    return false;
  result = static_cast<std::uint64_t>(cycles);
  return true;
}

} // namespace

// ===========================================================================
// Section 14: buffering-policy legality.
// ===========================================================================
BufferingLegalityResult checkBufferingLegality(const KernelCandidate &candidate,
                                                const CandidateLegalityResult &baseLegality,
                                                const BufferingPolicy &buffering,
                                                DMASchedulingPolicy dmaPolicy,
                                                const NPUTargetConfig &target) {
  BufferingLegalityResult r;
  r.candidateId = candidate.candidateId + extendedCandidateIdSuffix(buffering, dmaPolicy);

  if (baseLegality.candidateId != candidate.candidateId || !baseLegality.isLegal) {
    r.reasons.push_back(LegalityReason::UnsupportedBufferingPolicy);
    r.isLegal = false;
    return r;
  }

  if (target.dmaEngineCount == 0)
    r.reasons.push_back(LegalityReason::UnsupportedAsyncSchedule);

  std::uint64_t I = static_cast<std::uint64_t>(baseLegality.inputTileBytes);
  std::uint64_t W = static_cast<std::uint64_t>(baseLegality.weightTileBytes);
  std::uint64_t O = static_cast<std::uint64_t>(baseLegality.outputTileBytes);

  CheckedArith ck;
  r.inputRequiredBytes = ck.mul(I, slotCount(buffering.input));
  r.weightRequiredBytes = ck.mul(W, slotCount(buffering.weight));
  r.outputRequiredBytes = ck.mul(O, slotCount(buffering.output));
  if (ck.overflowed) {
    r.reasons.push_back(LegalityReason::ByteFootprintOverflow);
    r.isLegal = false;
    return r;
  }

  if (buffering.input == BufferingMode::Double &&
      r.inputRequiredBytes > static_cast<std::uint64_t>(target.inputBufferBytes))
    r.reasons.push_back(LegalityReason::InputDoubleBufferExceedsBuffer);
  if (buffering.weight == BufferingMode::Double &&
      r.weightRequiredBytes > static_cast<std::uint64_t>(target.weightBufferBytes))
    r.reasons.push_back(LegalityReason::WeightDoubleBufferExceedsBuffer);
  if (buffering.output == BufferingMode::Double &&
      r.outputRequiredBytes > static_cast<std::uint64_t>(target.outputBufferBytes))
    r.reasons.push_back(LegalityReason::OutputDoubleBufferExceedsBuffer);

  std::uint64_t agg = ck.add(ck.add(r.inputRequiredBytes, r.weightRequiredBytes), r.outputRequiredBytes);
  if (ck.overflowed) {
    r.reasons.push_back(LegalityReason::ByteFootprintOverflow);
  } else {
    r.aggregateRequiredBytes = agg;
    if (agg > static_cast<std::uint64_t>(target.scratchpadBytes))
      r.reasons.push_back(LegalityReason::AsyncAggregateScratchpadExceeded);
  }

  r.isLegal = r.reasons.empty();
  return r;
}

// ===========================================================================
// operator== implementations
// ===========================================================================
bool AsyncScheduleOperation::operator==(const AsyncScheduleOperation &other) const {
  return operationIndex == other.operationIndex && kind == other.kind &&
         transferKind == other.transferKind && eventId == other.eventId &&
         coordinate.mTile == other.coordinate.mTile && coordinate.nTile == other.coordinate.nTile &&
         coordinate.kTile == other.coordinate.kTile &&
         (input.has_value() == other.input.has_value()) &&
         (!input || *input == *other.input) && (weight.has_value() == other.weight.has_value()) &&
         (!weight || *weight == *other.weight) && (output.has_value() == other.output.has_value()) &&
         (!output || *output == *other.output) && bytes == other.bytes;
}

bool DMAEvent::operator==(const DMAEvent &other) const {
  return eventId == other.eventId && transferKind == other.transferKind &&
         issueOperationIndex == other.issueOperationIndex &&
         waitOperationIndex == other.waitOperationIndex && buffer == other.buffer;
}

bool AsyncCandidateCost::operator==(const AsyncCandidateCost &other) const {
  return candidateId == other.candidateId &&
         synchronousEstimatedCycles == other.synchronousEstimatedCycles &&
         asynchronousEstimatedCycles == other.asynchronousEstimatedCycles &&
         exposedDMACycles == other.exposedDMACycles && hiddenDMACycles == other.hiddenDMACycles &&
         overlapCycles == other.overlapCycles && startupCycles == other.startupCycles &&
         drainCycles == other.drainCycles && dmaOperationCount == other.dmaOperationCount &&
         dmaByteCount == other.dmaByteCount && computeCount == other.computeCount &&
         partialReloadCount == other.partialReloadCount &&
         partialStoreCount == other.partialStoreCount && finalStoreCount == other.finalStoreCount &&
         issuedEventCount == other.issuedEventCount && waitedEventCount == other.waitedEventCount;
}

bool AsyncSchedulePlan::operator==(const AsyncSchedulePlan &other) const {
  if (candidateId != other.candidateId || baseCandidateId != other.baseCandidateId ||
      precision != other.precision || dataflow != other.dataflow || !(peArray == other.peArray) ||
      !(tile == other.tile) || !(buffering == other.buffering) || dmaPolicy != other.dmaPolicy ||
      dmaEngineCount != other.dmaEngineCount || slots.input != other.slots.input ||
      slots.weight != other.slots.weight || slots.output != other.slots.output ||
      !(cost == other.cost))
    return false;
  if (operations.size() != other.operations.size() || events.size() != other.events.size())
    return false;
  for (std::size_t i = 0; i < operations.size(); ++i)
    if (!(operations[i] == other.operations[i]))
      return false;
  for (std::size_t i = 0; i < events.size(); ++i)
    if (!(events[i] == other.events[i]))
      return false;
  return true;
}

// ===========================================================================
// Section 8-13: the builder.
// ===========================================================================
namespace {

BufferedTileRef makeRef(BufferRole role, TileCoordinate coord, BufferSlot slot,
                        std::uint32_t slotIndex) {
  BufferedTileRef ref;
  ref.role = role;
  ref.coordinate = coord;
  ref.slot = slot;
  ref.slotIndex = slotIndex;
  return ref;
}

std::pair<BufferSlot, std::uint32_t> slotForChainPosition(std::size_t pos, BufferingMode mode) {
  if (mode == BufferingMode::Single)
    return {BufferSlot::Single, 0u};
  return (pos % 2 == 0) ? std::make_pair(BufferSlot::Ping, 0u)
                         : std::make_pair(BufferSlot::Pong, 1u);
}

struct BuildResult {
  std::vector<AsyncScheduleOperation> operations;
  std::vector<DMAEvent> events;
  std::size_t coreOpCount = 0;
  AsyncSlotCounts slots;
};

// The one documented scheduling policy (Sections 9-13); see the header
// doc comment for the full rationale. Consumes `lifetimes` (Slice 7's
// analyzeBufferLifetimes output over the REAL Slice 6 operation
// sequence) directly -- never reconstructs from aggregate counts.
llvm::Expected<BuildResult> buildAsyncOperations(const SchedulePlan &schedule,
                                                  const BufferingPolicy &buffering,
                                                  DMASchedulingPolicy policy,
                                                  const std::vector<BufferLifetime> &lifetimes) {
  auto byFirstOp = [&](std::size_t a, std::size_t b) {
    return lifetimes[a].firstOperationIndex < lifetimes[b].firstOperationIndex;
  };

  std::vector<std::size_t> inputChain, weightChain, outputFlat;
  for (std::size_t i = 0; i < lifetimes.size(); ++i) {
    switch (lifetimes[i].role) {
    case BufferRole::InputTile:
      inputChain.push_back(i);
      break;
    case BufferRole::WeightTile:
      weightChain.push_back(i);
      break;
    case BufferRole::OutputAccumulator:
      outputFlat.push_back(i);
      break;
    }
  }
  std::sort(inputChain.begin(), inputChain.end(), byFirstOp);
  std::sort(weightChain.begin(), weightChain.end(), byFirstOp);
  std::sort(outputFlat.begin(), outputFlat.end(), byFirstOp);

  // Group same-(m,n) accumulator lifetimes into one logical output-tile
  // group (Section 10's WS/IS partial-sum exception), keyed by
  // coordinate rather than by adjacency in `outputFlat`. WS/IS's loop
  // nesting places the k-loop OUTSIDE the m-loop (WS) / n-loop (IS), so
  // for KT>1 the K=0..KT-1 lifetimes belonging to one (m,n) accumulator
  // are NOT consecutive in firstOperationIndex order -- MT-1 (WS) or
  // NT-1 (IS) other output tiles' own k=0 lifetimes fall in between.
  // Grouping by a coordinate-keyed map (first-appearance order for the
  // groups themselves, chronological within each group since
  // `outputFlat` is already sorted) handles both the interleaved
  // WS/IS case and OS's naturally-consecutive case uniformly.
  std::vector<std::vector<std::size_t>> outputGroups;
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> outputGroupIndexByCoord;
  for (std::size_t idx : outputFlat) {
    auto key = std::make_pair(lifetimes[idx].coordinate.mTile, lifetimes[idx].coordinate.nTile);
    auto it = outputGroupIndexByCoord.find(key);
    if (it == outputGroupIndexByCoord.end()) {
      outputGroupIndexByCoord[key] = outputGroups.size();
      outputGroups.push_back({idx});
    } else {
      outputGroups[it->second].push_back(idx);
    }
  }

  // Section 12: per-group store-wait placement. The LAST group is
  // deliberately excluded from deferral entirely (left Adjacent) even
  // when output is double-buffered + PrefetchNext: deferring it would
  // place its wait at the very end with nothing else left to overlap
  // with, which is behaviorally identical to waiting immediately --
  // Section 19's "do not add a hardcoded bonus... reconcile to
  // synchronous behavior when overlap is ineffective" applies precisely
  // here. This also keeps the schedule-end drain mechanism meaningful:
  // only genuinely-benefiting deferrals ever produce a drain-appended
  // wait.
  enum class Placement { Adjacent, DeferredToPosition, DeferredToDrain };
  std::vector<Placement> placement(outputGroups.size(), Placement::Adjacent);
  std::map<std::uint64_t, std::vector<std::size_t>> deferredWaitsAt; // anchor opIndex -> group indices
  bool prefetch = (policy == DMASchedulingPolicy::PrefetchNext);
  bool doubleOutput = (buffering.output == BufferingMode::Double);
  for (std::size_t g = 0; g + 1 < outputGroups.size(); ++g) {
    if (doubleOutput && prefetch) {
      std::size_t blocker = g + 2;
      if (blocker < outputGroups.size()) {
        placement[g] = Placement::DeferredToPosition;
        std::size_t firstLtIdx = outputGroups[blocker].front();
        deferredWaitsAt[lifetimes[firstLtIdx].firstOperationIndex].push_back(g);
      } else {
        placement[g] = Placement::DeferredToDrain;
      }
    }
  }

  BuildResult out;
  out.slots.input = slotCount(buffering.input);
  out.slots.weight = slotCount(buffering.weight);
  out.slots.output = slotCount(buffering.output);

  std::uint64_t nextOpIndex = 0;
  std::uint64_t nextEventId = 0;
  std::vector<bool> issuedInput(inputChain.size(), false);
  std::vector<bool> issuedWeight(weightChain.size(), false);
  std::vector<std::uint64_t> outputGroupClosingEventId(outputGroups.size(),
                                                        std::numeric_limits<std::uint64_t>::max());
  std::vector<std::uint64_t> drainEvents;

  // eventId -> a reusable "wait template" (kind flipped to DMAWait, bytes 0).
  std::map<std::uint64_t, AsyncScheduleOperation> waitTemplate;

  auto pushOp = [&](AsyncScheduleOperation op) {
    op.operationIndex = nextOpIndex++;
    out.operations.push_back(op);
  };

  auto emitIssue = [&](DMATransferKind tk, TileCoordinate coord, std::optional<BufferedTileRef> input,
                       std::optional<BufferedTileRef> weight, std::optional<BufferedTileRef> output,
                       std::uint64_t bytes) -> std::uint64_t {
    std::uint64_t eid = nextEventId++;
    AsyncScheduleOperation issueOp;
    issueOp.kind = AsyncOperationKind::DMAIssue;
    issueOp.transferKind = tk;
    issueOp.eventId = eid;
    issueOp.coordinate = coord;
    issueOp.input = input;
    issueOp.weight = weight;
    issueOp.output = output;
    issueOp.bytes = bytes;
    std::uint64_t issuedAt = nextOpIndex;
    pushOp(issueOp);

    BufferedTileRef ref = input ? *input : (weight ? *weight : *output);
    out.events.push_back(DMAEvent{eid, tk, issuedAt, 0, ref});

    AsyncScheduleOperation waitOp = issueOp;
    waitOp.kind = AsyncOperationKind::DMAWait;
    waitOp.bytes = 0;
    waitTemplate[eid] = waitOp;
    return eid;
  };

  auto emitWait = [&](std::uint64_t eid) {
    AsyncScheduleOperation waitOp = waitTemplate.at(eid);
    pushOp(waitOp);
    out.events[eid].waitOperationIndex = out.operations.back().operationIndex;
  };

  std::vector<std::uint64_t> inputEventId(inputChain.size(), std::numeric_limits<std::uint64_t>::max());
  std::vector<std::uint64_t> weightEventId(weightChain.size(),
                                           std::numeric_limits<std::uint64_t>::max());

  // Issues instance `pos` of `chain` (role `role`, buffering `mode`) if not
  // already issued (Section 9: at most one outstanding future tile per
  // role -- `pos+1` is only ever pre-issued once `pos` itself has been
  // waited, see the prefetch-ahead call sites below).
  auto ensureIssued = [&](const std::vector<std::size_t> &chain, std::vector<bool> &issued,
                          std::vector<std::uint64_t> &eventIds, std::size_t pos, BufferRole role,
                          BufferingMode mode, DMATransferKind tk) {
    if (issued[pos])
      return;
    auto [slot, slotIdx] = slotForChainPosition(pos, mode);
    BufferedTileRef ref = makeRef(role, lifetimes[chain[pos]].coordinate, slot, slotIdx);
    std::optional<BufferedTileRef> inputRef = (role == BufferRole::InputTile) ? std::make_optional(ref)
                                                                              : std::nullopt;
    std::optional<BufferedTileRef> weightRef =
        (role == BufferRole::WeightTile) ? std::make_optional(ref) : std::nullopt;
    std::uint64_t eid =
        emitIssue(tk, ref.coordinate, inputRef, weightRef, std::nullopt, lifetimes[chain[pos]].sizeBytes);
    issued[pos] = true;
    eventIds[pos] = eid;
  };

  std::size_t inputCursor = 0, weightCursor = 0;
  std::size_t curInputPos = 0, curWeightPos = 0;
  // Per-(m,n) output-chain position: WS/IS interleave OTHER output
  // tiles' operations between one (m,n) accumulator's own k=0..KT-1
  // touches (see the outputGroupIndexByCoord doc comment above), so
  // "which group/lt-position are we currently at" cannot be a single
  // scalar -- it must be tracked per coordinate, resolved fresh from
  // each operation's own (mTile, nTile).
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> curLtPosByCoord;
  auto coordKeyOf = [](const TileCoordinate &c) { return std::make_pair(c.mTile, c.nTile); };

  for (const ScheduleOperation &op : schedule.operations) {
    // Fire any deferred output-store waits anchored at this original position.
    auto it = deferredWaitsAt.find(op.operationIndex);
    if (it != deferredWaitsAt.end())
      for (std::size_t g : it->second)
        emitWait(outputGroupClosingEventId[g]);

    switch (op.kind) {
    case ScheduleOperationKind::LoadInputTile: {
      std::size_t pos = inputCursor++;
      curInputPos = pos;
      ensureIssued(inputChain, issuedInput, inputEventId, pos, BufferRole::InputTile,
                  buffering.input, DMATransferKind::InputLoad);
      emitWait(inputEventId[pos]);
      if (buffering.input == BufferingMode::Double && policy == DMASchedulingPolicy::PrefetchNext &&
          pos + 1 < inputChain.size())
        ensureIssued(inputChain, issuedInput, inputEventId, pos + 1, BufferRole::InputTile,
                    buffering.input, DMATransferKind::InputLoad);
      break;
    }
    case ScheduleOperationKind::LoadWeightTile: {
      std::size_t pos = weightCursor++;
      curWeightPos = pos;
      ensureIssued(weightChain, issuedWeight, weightEventId, pos, BufferRole::WeightTile,
                  buffering.weight, DMATransferKind::WeightLoad);
      emitWait(weightEventId[pos]);
      if (buffering.weight == BufferingMode::Double && policy == DMASchedulingPolicy::PrefetchNext &&
          pos + 1 < weightChain.size())
        ensureIssued(weightChain, issuedWeight, weightEventId, pos + 1, BufferRole::WeightTile,
                    buffering.weight, DMATransferKind::WeightLoad);
      break;
    }
    case ScheduleOperationKind::ReloadPartialOutput: {
      auto key = coordKeyOf(op.coordinate);
      std::size_t groupPos = outputGroupIndexByCoord.at(key);
      std::size_t ltPos = ++curLtPosByCoord[key];
      std::size_t ltIdx = outputGroups[groupPos][ltPos];
      auto [slot, slotIdx] = slotForChainPosition(groupPos, buffering.output);
      BufferedTileRef ref = makeRef(BufferRole::OutputAccumulator, lifetimes[ltIdx].coordinate, slot, slotIdx);
      std::uint64_t eid = emitIssue(DMATransferKind::PartialOutputReload, ref.coordinate, std::nullopt,
                                    std::nullopt, ref, op.bytes);
      emitWait(eid);
      break;
    }
    case ScheduleOperationKind::ComputeTile: {
      auto key = coordKeyOf(op.coordinate);
      std::size_t groupPos = outputGroupIndexByCoord.at(key);
      if (op.coordinate.kTile == 0)
        curLtPosByCoord[key] = 0;
      std::size_t ltPos = curLtPosByCoord.at(key);
      auto [inSlot, inIdx] = slotForChainPosition(curInputPos, buffering.input);
      auto [wSlot, wIdx] = slotForChainPosition(curWeightPos, buffering.weight);
      auto [oSlot, oIdx] = slotForChainPosition(groupPos, buffering.output);
      AsyncScheduleOperation cop;
      cop.kind = AsyncOperationKind::Compute;
      cop.coordinate = op.coordinate;
      cop.input = makeRef(BufferRole::InputTile, lifetimes[inputChain[curInputPos]].coordinate, inSlot, inIdx);
      cop.weight = makeRef(BufferRole::WeightTile, lifetimes[weightChain[curWeightPos]].coordinate, wSlot, wIdx);
      cop.output = makeRef(BufferRole::OutputAccumulator,
                            lifetimes[outputGroups[groupPos][ltPos]].coordinate, oSlot, oIdx);
      cop.bytes = 0;
      pushOp(cop);
      break;
    }
    case ScheduleOperationKind::StorePartialOutput:
    case ScheduleOperationKind::StoreFinalOutput: {
      auto key = coordKeyOf(op.coordinate);
      std::size_t groupPos = outputGroupIndexByCoord.at(key);
      std::size_t ltPos = curLtPosByCoord.at(key);
      std::size_t ltIdx = outputGroups[groupPos][ltPos];
      bool isClosing = (ltPos + 1 == outputGroups[groupPos].size());
      auto [slot, slotIdx] = slotForChainPosition(groupPos, buffering.output);
      BufferedTileRef ref = makeRef(BufferRole::OutputAccumulator, lifetimes[ltIdx].coordinate, slot, slotIdx);
      DMATransferKind tk = (op.kind == ScheduleOperationKind::StoreFinalOutput)
                                ? DMATransferKind::FinalOutputStore
                                : DMATransferKind::PartialOutputStore;
      std::uint64_t eid =
          emitIssue(tk, ref.coordinate, std::nullopt, std::nullopt, ref, op.bytes);
      if (!isClosing) {
        emitWait(eid);
      } else {
        outputGroupClosingEventId[groupPos] = eid;
        switch (placement[groupPos]) {
        case Placement::Adjacent:
          emitWait(eid);
          break;
        case Placement::DeferredToPosition:
          break; // waited later, via deferredWaitsAt
        case Placement::DeferredToDrain:
          drainEvents.push_back(eid);
          break;
        }
      }
      break;
    }
    case ScheduleOperationKind::Synchronize:
      break;
    }
  }

  out.coreOpCount = out.operations.size();
  for (std::uint64_t eid : drainEvents)
    emitWait(eid);

  return out;
}

// Reuses Slice 3's own per-(m,n,k)-tile compute-cycle formula (see
// CandidateCostModel.h Section 8: arrayWavesPerTile * reductionCyclesPerKTile
// -- the per-single-ComputeTile-operation cost, since Slice 3's
// computeCyclesPerMNTile already folds in numReductionTiles as a loop
// count over exactly this many ComputeTile operations).
llvm::Expected<std::uint64_t> perComputeOpCycles(const SchedulePlan &schedule,
                                                  const NPUTargetConfig &target) {
  if (schedule.peArray.rows <= 0 || schedule.peArray.cols <= 0)
    return makeError("perComputeOpCycles: non-positive PE array dimension");
  std::uint64_t macsPerPEPerCycle = (schedule.precision == Precision::INT8)
                                        ? target.int8MacsPerPEPerCycle
                                        : target.fp16MacsPerPEPerCycle;
  if (macsPerPEPerCycle == 0)
    return makeError("perComputeOpCycles: target's precision throughput is zero");

  CheckedArith ck;
  std::uint64_t tileM = ck.mul(static_cast<std::uint64_t>(schedule.tile.height),
                               static_cast<std::uint64_t>(schedule.tile.width));
  std::uint64_t tileN = static_cast<std::uint64_t>(schedule.tile.outputChannels);
  std::uint64_t tileK = static_cast<std::uint64_t>(schedule.tile.reductionDepth);
  std::uint64_t mWaves = ck.ceilDiv(tileM, static_cast<std::uint64_t>(schedule.peArray.rows));
  std::uint64_t nWaves = ck.ceilDiv(tileN, static_cast<std::uint64_t>(schedule.peArray.cols));
  std::uint64_t arrayWavesPerTile = ck.mul(mWaves, nWaves);
  std::uint64_t reductionCyclesPerKTile = ck.ceilDiv(tileK, macsPerPEPerCycle);
  std::uint64_t result = ck.mul(arrayWavesPerTile, reductionCyclesPerKTile);
  if (ck.overflowed)
    return makeError("perComputeOpCycles: integer overflow");
  return result;
}

// Section 16-18: two-timeline (compute engine + single DMA engine)
// discrete simulation. `coreOpCount` marks the boundary before any
// schedule-end drain waits were appended (Section 25) so drainCycles can
// be isolated.
struct SimResult {
  std::uint64_t finalCycles = 0;
  std::uint64_t hiddenDMACycles = 0;
  std::uint64_t exposedDMACycles = 0;
  std::uint64_t startupCycles = 0;
  std::uint64_t drainCycles = 0;
  bool overflowed = false;
};

SimResult simulateCycles(const std::vector<AsyncScheduleOperation> &ops,
                          std::uint64_t perComputeCycles, std::size_t coreOpCount,
                          const NPUTargetConfig &target) {
  SimResult sim;
  sim.startupCycles = target.kernelLaunchCycles;
  CheckedArith ck;
  std::uint64_t computeClock = target.kernelLaunchCycles;
  std::uint64_t dmaClock = target.kernelLaunchCycles;
  std::uint64_t clockAtCoreBoundary = computeClock;

  std::map<std::uint64_t, std::uint64_t> eventFinish;
  std::map<std::uint64_t, std::uint64_t> eventCost;
  std::map<std::uint64_t, std::uint64_t> eventIssueComputeTime;

  for (std::size_t i = 0; i < ops.size(); ++i) {
    if (i == coreOpCount)
      clockAtCoreBoundary = computeClock;
    const AsyncScheduleOperation &op = ops[i];
    switch (op.kind) {
    case AsyncOperationKind::DMAIssue: {
      std::uint64_t transfer = 0;
      if (!transferCyclesChecked(op.bytes, target.clockHz, target.offChipBandwidthBytesPerSecond,
                                 transfer)) {
        sim.overflowed = true;
        break;
      }
      std::uint64_t cost = ck.add(static_cast<std::uint64_t>(target.dmaSetupCycles), transfer);
      dmaClock = ck.add(dmaClock, cost);
      std::uint64_t eid = *op.eventId;
      eventFinish[eid] = dmaClock;
      eventCost[eid] = cost;
      eventIssueComputeTime[eid] = computeClock;
      break;
    }
    case AsyncOperationKind::DMAWait: {
      std::uint64_t eid = *op.eventId;
      std::uint64_t issuedAt = eventIssueComputeTime.count(eid) ? eventIssueComputeTime[eid] : computeClock;
      std::uint64_t available = computeClock >= issuedAt ? computeClock - issuedAt : 0;
      std::uint64_t cost = eventCost.count(eid) ? eventCost[eid] : 0;
      std::uint64_t hidden = std::min(cost, available);
      std::uint64_t exposed = cost - hidden;
      sim.hiddenDMACycles = ck.add(sim.hiddenDMACycles, hidden);
      sim.exposedDMACycles = ck.add(sim.exposedDMACycles, exposed);
      std::uint64_t finish = eventFinish.count(eid) ? eventFinish[eid] : computeClock;
      if (computeClock < finish)
        computeClock = ck.add(finish, static_cast<std::uint64_t>(target.tileSynchronizationCycles));
      break;
    }
    case AsyncOperationKind::Compute:
      computeClock = ck.add(computeClock, perComputeCycles);
      break;
    case AsyncOperationKind::ComputeBarrier:
      break;
    }
    if (ck.overflowed)
      sim.overflowed = true;
    if (sim.overflowed)
      break;
  }
  if (coreOpCount >= ops.size())
    clockAtCoreBoundary = computeClock; // no drain ops appended

  sim.finalCycles = computeClock;
  sim.drainCycles = (computeClock >= clockAtCoreBoundary) ? (computeClock - clockAtCoreBoundary) : 0;
  return sim;
}

void fillReconciliationCounts(AsyncCandidateCost &cost, const AsyncSchedulePlan &plan) {
  std::set<std::uint64_t> seenIssued, seenWaited;
  for (const AsyncScheduleOperation &op : plan.operations) {
    switch (op.kind) {
    case AsyncOperationKind::Compute:
      ++cost.computeCount;
      break;
    case AsyncOperationKind::DMAIssue: {
      seenIssued.insert(*op.eventId);
      cost.dmaByteCount += op.bytes;
      switch (*op.transferKind) {
      case DMATransferKind::InputLoad:
      case DMATransferKind::WeightLoad:
        break;
      case DMATransferKind::PartialOutputReload:
        ++cost.partialReloadCount;
        break;
      case DMATransferKind::PartialOutputStore:
        ++cost.partialStoreCount;
        break;
      case DMATransferKind::FinalOutputStore:
        ++cost.finalStoreCount;
        break;
      }
      ++cost.dmaOperationCount;
      break;
    }
    case AsyncOperationKind::DMAWait:
      seenWaited.insert(*op.eventId);
      break;
    case AsyncOperationKind::ComputeBarrier:
      break;
    }
  }
  cost.issuedEventCount = seenIssued.size();
  cost.waitedEventCount = seenWaited.size();
}

} // namespace

llvm::Expected<AsyncSchedulePlan>
materializeAsyncSchedule(const SchedulePlan &synchronousSchedule, const BufferingPolicy &buffering,
                         DMASchedulingPolicy schedulingPolicy, const NPUTargetConfig &target) {
  if (target.dmaEngineCount == 0)
    return makeError("materializeAsyncSchedule: target.dmaEngineCount is zero -- at least one "
                      "logical DMA engine is required (Section 17)");
  if (target.clockHz == 0 || target.offChipBandwidthBytesPerSecond == 0)
    return makeError("materializeAsyncSchedule: target clock or off-chip bandwidth is zero");

  llvm::Expected<std::vector<BufferLifetime>> lifetimesOrErr =
      analyzeBufferLifetimes(synchronousSchedule, target);
  if (!lifetimesOrErr)
    return lifetimesOrErr.takeError();

  llvm::Expected<BuildResult> requestedOrErr =
      buildAsyncOperations(synchronousSchedule, buffering, schedulingPolicy, *lifetimesOrErr);
  if (!requestedOrErr)
    return requestedOrErr.takeError();

  BuildResult syncBaseline;
  if (schedulingPolicy == DMASchedulingPolicy::Synchronous) {
    syncBaseline = *requestedOrErr;
  } else {
    llvm::Expected<BuildResult> syncOrErr = buildAsyncOperations(
        synchronousSchedule, buffering, DMASchedulingPolicy::Synchronous, *lifetimesOrErr);
    if (!syncOrErr)
      return syncOrErr.takeError();
    syncBaseline = *syncOrErr;
  }

  llvm::Expected<std::uint64_t> perOpCyclesOrErr = perComputeOpCycles(synchronousSchedule, target);
  if (!perOpCyclesOrErr)
    return perOpCyclesOrErr.takeError();

  SimResult reqSim =
      simulateCycles(requestedOrErr->operations, *perOpCyclesOrErr, requestedOrErr->coreOpCount, target);
  SimResult syncSim =
      simulateCycles(syncBaseline.operations, *perOpCyclesOrErr, syncBaseline.coreOpCount, target);
  if (reqSim.overflowed || syncSim.overflowed)
    return makeError("materializeAsyncSchedule: integer overflow in the cycle simulation");

  AsyncSchedulePlan plan;
  plan.baseCandidateId = synchronousSchedule.candidateId;
  plan.candidateId = synchronousSchedule.candidateId + extendedCandidateIdSuffix(buffering, schedulingPolicy);
  plan.precision = synchronousSchedule.precision;
  plan.dataflow = synchronousSchedule.dataflow;
  plan.peArray = synchronousSchedule.peArray;
  plan.tile = synchronousSchedule.tile;
  plan.buffering = buffering;
  plan.dmaPolicy = schedulingPolicy;
  plan.dmaEngineCount = target.dmaEngineCount;
  plan.slots = requestedOrErr->slots;
  plan.operations = std::move(requestedOrErr->operations);
  plan.events = std::move(requestedOrErr->events);

  AsyncCandidateCost cost;
  cost.candidateId = plan.candidateId;
  cost.synchronousEstimatedCycles = syncSim.finalCycles;
  cost.asynchronousEstimatedCycles = reqSim.finalCycles;
  cost.hiddenDMACycles = reqSim.hiddenDMACycles;
  cost.exposedDMACycles = reqSim.exposedDMACycles;
  cost.overlapCycles = reqSim.hiddenDMACycles;
  cost.startupCycles = reqSim.startupCycles;
  cost.drainCycles = reqSim.drainCycles;
  fillReconciliationCounts(cost, plan);
  plan.cost = cost;

  if (llvm::Error err = validateAsyncCostAgainstSchedule(plan, plan.cost))
    return std::move(err);
  if (llvm::Error err = validateAsyncSchedule(plan, target))
    return std::move(err);

  return plan;
}

// ===========================================================================
// Section 24: structural/dependency validation.
// ===========================================================================
llvm::Error validateAsyncSchedule(const AsyncSchedulePlan &plan, const NPUTargetConfig &target) {
  for (std::size_t i = 0; i < plan.operations.size(); ++i)
    if (plan.operations[i].operationIndex != i)
      return makeError("validateAsyncSchedule: operation indices are not dense and monotonic");

  if (target.dmaEngineCount == 0)
    return makeError("validateAsyncSchedule: target.dmaEngineCount is zero");

  if (plan.candidateId != plan.baseCandidateId + extendedCandidateIdSuffix(plan.buffering, plan.dmaPolicy))
    return makeError("validateAsyncSchedule: candidateId does not match baseCandidateId + the "
                      "buffering/dma-policy suffix");

  std::map<std::uint64_t, std::uint64_t> issuePos;
  std::set<std::uint64_t> waited;
  for (const AsyncScheduleOperation &op : plan.operations) {
    if (op.kind == AsyncOperationKind::DMAIssue) {
      if (!op.eventId)
        return makeError("validateAsyncSchedule: DMAIssue with no eventId");
      if (issuePos.count(*op.eventId))
        return makeError("validateAsyncSchedule: duplicate event id " + llvm::Twine(*op.eventId));
      issuePos[*op.eventId] = op.operationIndex;
    } else if (op.kind == AsyncOperationKind::DMAWait) {
      if (!op.eventId)
        return makeError("validateAsyncSchedule: DMAWait with no eventId");
      auto it = issuePos.find(*op.eventId);
      if (it == issuePos.end())
        return makeError("validateAsyncSchedule: wait for unknown event " + llvm::Twine(*op.eventId));
      if (it->second > op.operationIndex)
        return makeError("validateAsyncSchedule: wait occurs before its own issue");
      if (!waited.insert(*op.eventId).second)
        return makeError("validateAsyncSchedule: event " + llvm::Twine(*op.eventId) + " waited twice");
    }
  }
  for (const auto &[eid, pos] : issuePos) {
    (void)pos;
    if (!waited.count(eid))
      return makeError("validateAsyncSchedule: event " + llvm::Twine(eid) +
                        " is never waited/drained before schedule end");
  }

  if (plan.events.size() != issuePos.size())
    return makeError("validateAsyncSchedule: events vector size does not match the number of "
                      "distinct issued event ids");
  for (const DMAEvent &ev : plan.events) {
    auto it = issuePos.find(ev.eventId);
    if (it == issuePos.end() || it->second != ev.issueOperationIndex)
      return makeError("validateAsyncSchedule: DMAEvent issueOperationIndex does not match its "
                        "own DMAIssue operation");
    if (!waited.count(ev.eventId))
      return makeError("validateAsyncSchedule: DMAEvent has no matching wait");
  }

  // Slot-index range checks.
  auto checkRef = [&](const std::optional<BufferedTileRef> &ref) -> llvm::Error {
    if (!ref)
      return llvm::Error::success();
    if (ref->slot == BufferSlot::Single && ref->slotIndex != 0)
      return makeError("validateAsyncSchedule: a Single-buffered role used a nonzero slot index");
    if (ref->slot != BufferSlot::Single && ref->slotIndex > 1)
      return makeError("validateAsyncSchedule: a double-buffered role used a slot index outside {0,1}");
    return llvm::Error::success();
  };
  for (const AsyncScheduleOperation &op : plan.operations) {
    if (llvm::Error e = checkRef(op.input))
      return e;
    if (llvm::Error e = checkRef(op.weight))
      return e;
    if (llvm::Error e = checkRef(op.output))
      return e;
  }

  // Synchronous policy contains no prefetched future transfer: every
  // issue is immediately followed by its own wait.
  if (plan.dmaPolicy == DMASchedulingPolicy::Synchronous) {
    for (std::size_t i = 0; i < plan.operations.size(); ++i) {
      if (plan.operations[i].kind != AsyncOperationKind::DMAIssue)
        continue;
      if (i + 1 >= plan.operations.size() ||
          plan.operations[i + 1].kind != AsyncOperationKind::DMAWait ||
          plan.operations[i + 1].eventId != plan.operations[i].eventId)
        return makeError("validateAsyncSchedule: Synchronous schedule contains a prefetched "
                          "(non-adjacent) transfer");
    }
  }

  // Section 6-7 dependency hazard check: a physical (role, slot) may not
  // be overwritten (re-issued) or consumed (Compute) while its current
  // occupant is outstanding (issued but not yet waited).
  auto slotKey = [](const BufferedTileRef &ref) {
    std::uint32_t effectiveSlot = (ref.slot == BufferSlot::Single) ? 0u : ref.slotIndex;
    return std::make_pair(static_cast<int>(ref.role), effectiveSlot);
  };
  std::map<std::pair<int, std::uint32_t>, std::uint64_t> occupant;
  std::set<std::uint64_t> occupantWaited;
  for (const AsyncScheduleOperation &op : plan.operations) {
    if (op.kind == AsyncOperationKind::DMAIssue) {
      std::optional<BufferedTileRef> ref = op.input ? op.input : (op.weight ? op.weight : op.output);
      if (!ref)
        return makeError("validateAsyncSchedule: DMAIssue with no associated buffer");
      auto key = slotKey(*ref);
      auto it = occupant.find(key);
      if (it != occupant.end() && !occupantWaited.count(it->second))
        return makeError("validateAsyncSchedule: a slot was overwritten while its prior "
                          "DMA/store was still outstanding");
      occupant[key] = *op.eventId;
    } else if (op.kind == AsyncOperationKind::DMAWait) {
      occupantWaited.insert(*op.eventId);
    } else if (op.kind == AsyncOperationKind::Compute) {
      auto checkLive = [&](const std::optional<BufferedTileRef> &ref) -> llvm::Error {
        if (!ref)
          return llvm::Error::success();
        auto key = slotKey(*ref);
        auto it = occupant.find(key);
        if (it != occupant.end() && !occupantWaited.count(it->second))
          return makeError("validateAsyncSchedule: Compute consumed an incomplete DMA destination");
        return llvm::Error::success();
      };
      if (llvm::Error e = checkLive(op.input))
        return e;
      if (llvm::Error e = checkLive(op.weight))
        return e;
      if (llvm::Error e = checkLive(op.output))
        return e;
    }
  }

  return llvm::Error::success();
}

// ===========================================================================
// Section 25: cost/schedule reconciliation.
// ===========================================================================
llvm::Error validateAsyncCostAgainstSchedule(const AsyncSchedulePlan &schedule,
                                             const AsyncCandidateCost &cost) {
  if (schedule.candidateId != cost.candidateId)
    return makeError("validateAsyncCostAgainstSchedule: candidateId mismatch");

  AsyncCandidateCost recomputed;
  fillReconciliationCounts(recomputed, schedule);

  if (recomputed.computeCount != cost.computeCount)
    return makeError("validateAsyncCostAgainstSchedule: computeCount mismatch");
  if (recomputed.dmaOperationCount != cost.dmaOperationCount)
    return makeError("validateAsyncCostAgainstSchedule: dmaOperationCount mismatch");
  if (recomputed.dmaByteCount != cost.dmaByteCount)
    return makeError("validateAsyncCostAgainstSchedule: dmaByteCount mismatch");
  if (recomputed.partialReloadCount != cost.partialReloadCount)
    return makeError("validateAsyncCostAgainstSchedule: partialReloadCount mismatch");
  if (recomputed.partialStoreCount != cost.partialStoreCount)
    return makeError("validateAsyncCostAgainstSchedule: partialStoreCount mismatch");
  if (recomputed.finalStoreCount != cost.finalStoreCount)
    return makeError("validateAsyncCostAgainstSchedule: finalStoreCount mismatch");
  if (recomputed.issuedEventCount != cost.issuedEventCount)
    return makeError("validateAsyncCostAgainstSchedule: issuedEventCount mismatch");
  if (recomputed.waitedEventCount != cost.waitedEventCount)
    return makeError("validateAsyncCostAgainstSchedule: waitedEventCount mismatch");
  if (cost.issuedEventCount != cost.waitedEventCount)
    return makeError("validateAsyncCostAgainstSchedule: not every issued event was waited/drained");

  return llvm::Error::success();
}

// ===========================================================================
// Section 21: ping/pong scratchpad allocation.
// ===========================================================================
llvm::Expected<AsyncScratchpadAllocationPlan>
allocateAsyncScratchpad(const AsyncSchedulePlan &plan, const NPUTargetConfig &target) {
  if (target.dmaAlignmentBytes <= 0)
    return makeError("allocateAsyncScratchpad: target.dmaAlignmentBytes must be positive");
  std::uint64_t alignment = static_cast<std::uint64_t>(target.dmaAlignmentBytes);

  std::uint64_t maxInputBytes = 0, maxWeightBytes = 0, maxOutputBytes = 0;
  for (const AsyncScheduleOperation &op : plan.operations) {
    if (op.kind != AsyncOperationKind::DMAIssue || !op.transferKind)
      continue;
    switch (*op.transferKind) {
    case DMATransferKind::InputLoad:
      maxInputBytes = std::max(maxInputBytes, op.bytes);
      break;
    case DMATransferKind::WeightLoad:
      maxWeightBytes = std::max(maxWeightBytes, op.bytes);
      break;
    case DMATransferKind::PartialOutputReload:
    case DMATransferKind::PartialOutputStore:
    case DMATransferKind::FinalOutputStore:
      maxOutputBytes = std::max(maxOutputBytes, op.bytes);
      break;
    }
  }

  auto buildRegion = [&](BufferRole role, MemoryRegionKind region, std::uint32_t slots,
                        std::uint64_t tileBytes, std::vector<AsyncBufferAllocation> &out,
                        std::uint64_t &peak) -> llvm::Error {
    if (tileBytes == 0) {
      peak = 0;
      return llvm::Error::success();
    }
    AsyncBufferAllocation ping;
    ping.role = role;
    ping.region = region;
    ping.slot = (slots >= 2) ? BufferSlot::Ping : BufferSlot::Single;
    ping.offsetBytes = 0;
    ping.sizeBytes = tileBytes;
    out.push_back(ping);
    std::uint64_t total = tileBytes;

    if (slots >= 2) {
      std::uint64_t pongOffset = 0;
      if (!alignUpChecked(tileBytes, alignment, pongOffset))
        return makeError("allocateAsyncScratchpad: address overflow aligning the Pong offset");
      std::uint64_t pongEnd = 0;
      if (__builtin_add_overflow(pongOffset, tileBytes, &pongEnd))
        return makeError("allocateAsyncScratchpad: address overflow computing the Pong end");
      AsyncBufferAllocation pong;
      pong.role = role;
      pong.region = region;
      pong.slot = BufferSlot::Pong;
      pong.offsetBytes = pongOffset;
      pong.sizeBytes = tileBytes;
      out.push_back(pong);
      total = pongEnd;
    }
    peak = total;
    return llvm::Error::success();
  };

  AsyncScratchpadAllocationPlan result;
  result.candidateId = plan.candidateId;
  if (llvm::Error err = buildRegion(BufferRole::InputTile, MemoryRegionKind::InputBuffer,
                                    plan.slots.input, maxInputBytes, result.allocations,
                                    result.peakInputBufferBytes))
    return std::move(err);
  if (llvm::Error err = buildRegion(BufferRole::WeightTile, MemoryRegionKind::WeightBuffer,
                                    plan.slots.weight, maxWeightBytes, result.allocations,
                                    result.peakWeightBufferBytes))
    return std::move(err);
  if (llvm::Error err = buildRegion(BufferRole::OutputAccumulator, MemoryRegionKind::OutputBuffer,
                                    plan.slots.output, maxOutputBytes, result.allocations,
                                    result.peakOutputBufferBytes))
    return std::move(err);

  CheckedArith ck;
  result.peakAggregateScratchpadBytes =
      ck.add(ck.add(result.peakInputBufferBytes, result.peakWeightBufferBytes),
             result.peakOutputBufferBytes);
  if (ck.overflowed)
    return makeError("allocateAsyncScratchpad: overflow computing the aggregate peak");

  return result;
}

llvm::Error validateAsyncAllocationAgainstBufferingLegality(
    const AsyncScratchpadAllocationPlan &allocation, const BufferingLegalityResult &bufferingLegality,
    const NPUTargetConfig &target) {
  if (allocation.candidateId != bufferingLegality.candidateId)
    return makeError("validateAsyncAllocationAgainstBufferingLegality: candidateId mismatch");
  if (allocation.peakInputBufferBytes > bufferingLegality.inputRequiredBytes)
    return makeError("validateAsyncAllocationAgainstBufferingLegality: concrete input peak "
                      "exceeds the Section 14 conservative estimate");
  if (allocation.peakWeightBufferBytes > bufferingLegality.weightRequiredBytes)
    return makeError("validateAsyncAllocationAgainstBufferingLegality: concrete weight peak "
                      "exceeds the Section 14 conservative estimate");
  if (allocation.peakOutputBufferBytes > bufferingLegality.outputRequiredBytes)
    return makeError("validateAsyncAllocationAgainstBufferingLegality: concrete output peak "
                      "exceeds the Section 14 conservative estimate");
  if (allocation.peakAggregateScratchpadBytes > bufferingLegality.aggregateRequiredBytes)
    return makeError("validateAsyncAllocationAgainstBufferingLegality: concrete aggregate peak "
                      "exceeds the Section 14 conservative estimate");
  if (allocation.peakInputBufferBytes > static_cast<std::uint64_t>(target.inputBufferBytes))
    return makeError("validateAsyncAllocationAgainstBufferingLegality: input peak exceeds target "
                      "capacity");
  if (allocation.peakWeightBufferBytes > static_cast<std::uint64_t>(target.weightBufferBytes))
    return makeError("validateAsyncAllocationAgainstBufferingLegality: weight peak exceeds "
                      "target capacity");
  if (allocation.peakOutputBufferBytes > static_cast<std::uint64_t>(target.outputBufferBytes))
    return makeError("validateAsyncAllocationAgainstBufferingLegality: output peak exceeds "
                      "target capacity");
  if (allocation.peakAggregateScratchpadBytes > static_cast<std::uint64_t>(target.scratchpadBytes))
    return makeError("validateAsyncAllocationAgainstBufferingLegality: aggregate peak exceeds "
                      "target capacity");
  return llvm::Error::success();
}

// ===========================================================================
// Section 27: stable, deterministic JSON serialization.
// ===========================================================================
std::string serializeAsyncSchedulePlanToJson(const AsyncSchedulePlan &plan) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "{\n";
  os << "  \"candidate_id\": \"" << plan.candidateId << "\",\n";
  os << "  \"buffering\": {\"input\": \"" << toString(plan.buffering.input) << "\", \"weight\": \""
     << toString(plan.buffering.weight) << "\", \"output\": \"" << toString(plan.buffering.output)
     << "\"},\n";
  os << "  \"dma_policy\": \"" << toString(plan.dmaPolicy) << "\",\n";
  os << "  \"estimated_cycles\": {\"synchronous\": " << plan.cost.synchronousEstimatedCycles
     << ", \"asynchronous\": " << plan.cost.asynchronousEstimatedCycles
     << ", \"hidden_dma\": " << plan.cost.hiddenDMACycles
     << ", \"exposed_dma\": " << plan.cost.exposedDMACycles << "},\n";
  os << "  \"slots\": {\"input\": " << plan.slots.input << ", \"weight\": " << plan.slots.weight
     << ", \"output\": " << plan.slots.output << "},\n";
  os << "  \"operations\": [\n";
  for (std::size_t i = 0; i < plan.operations.size(); ++i) {
    const AsyncScheduleOperation &op = plan.operations[i];
    os << "    {\"index\": " << op.operationIndex << ", \"kind\": \"" << toString(op.kind) << "\"";
    if (op.transferKind)
      os << ", \"transfer\": \"" << toString(*op.transferKind) << "\"";
    if (op.eventId)
      os << ", \"event_id\": " << *op.eventId;
    std::optional<BufferedTileRef> ref = op.input ? op.input : (op.weight ? op.weight : op.output);
    if (ref && op.kind != AsyncOperationKind::Compute)
      os << ", \"slot\": \"" << toString(ref->slot) << "\"";
    os << ", \"bytes\": " << op.bytes << "}";
    if (i + 1 < plan.operations.size())
      os << ",";
    os << "\n";
  }
  os << "  ]\n";
  os << "}";
  os.flush();
  return out;
}

// ===========================================================================
// Section 20: deterministic ranking.
// ===========================================================================
std::vector<RankedAsyncCandidate>
rankAsyncCandidates(llvm::ArrayRef<AsyncSchedulePlan> plans,
                    llvm::ArrayRef<AsyncScratchpadAllocationPlan> allocations) {
  std::vector<std::size_t> idx(plans.size());
  std::iota(idx.begin(), idx.end(), 0);

  auto numDoubleBuffered = [](const BufferingPolicy &b) {
    std::uint32_t n = 0;
    if (b.input == BufferingMode::Double)
      ++n;
    if (b.weight == BufferingMode::Double)
      ++n;
    if (b.output == BufferingMode::Double)
      ++n;
    return n;
  };

  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
    const AsyncSchedulePlan &pa = plans[a];
    const AsyncSchedulePlan &pb = plans[b];
    if (pa.cost.asynchronousEstimatedCycles != pb.cost.asynchronousEstimatedCycles)
      return pa.cost.asynchronousEstimatedCycles < pb.cost.asynchronousEstimatedCycles;
    if (pa.cost.exposedDMACycles != pb.cost.exposedDMACycles)
      return pa.cost.exposedDMACycles < pb.cost.exposedDMACycles;
    std::uint64_t peakA = a < allocations.size() ? allocations[a].peakAggregateScratchpadBytes : 0;
    std::uint64_t peakB = b < allocations.size() ? allocations[b].peakAggregateScratchpadBytes : 0;
    if (peakA != peakB)
      return peakA < peakB;
    std::uint32_t da = numDoubleBuffered(pa.buffering), db = numDoubleBuffered(pb.buffering);
    if (da != db)
      return da < db;
    return a < b;
  });

  std::vector<RankedAsyncCandidate> result;
  result.reserve(idx.size());
  for (std::size_t r = 0; r < idx.size(); ++r) {
    RankedAsyncCandidate rc;
    rc.candidateId = plans[idx[r]].candidateId;
    rc.originalIndex = idx[r];
    rc.rank = r;
    result.push_back(rc);
  }
  return result;
}

} // namespace mlir::costmodel
