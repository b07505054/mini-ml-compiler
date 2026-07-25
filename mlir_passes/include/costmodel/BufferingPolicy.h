#pragma once

// Cost Model Slice 8 (Double Buffering and Asynchronous DMA Scheduling):
// buffering-policy axis. Small, standalone, cohesive header (Section 30)
// so Slice 8's candidate-identity axis has one obvious source of truth,
// separate from the async scheduling/event/allocation logic in
// AsyncDMASchedule.h.
//
// A BufferingPolicy is a per-role (input/weight/output) choice between
// single- and double-buffering. It says nothing about WHEN a transfer is
// issued (see DMASchedulingPolicy in AsyncDMASchedule.h) -- only how many
// physical slots that role's data occupies. Eight policies are legal
// candidate points (SSS..DDD); none is assumed to be legal or profitable
// a priori -- see costmodel::checkBufferingLegality in AsyncDMASchedule.h.

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::costmodel {

enum class BufferingMode { Single, Double };

llvm::StringRef toString(BufferingMode mode); // "single" / "double"
char toChar(BufferingMode mode);              // 'S' / 'D'

// Number of physical slots a role occupies under this mode: 1 for
// Single, 2 for Double. Never any other value in this Slice (no triple
// buffering).
std::uint32_t slotCount(BufferingMode mode);

struct BufferingPolicy {
  BufferingMode input = BufferingMode::Single;
  BufferingMode weight = BufferingMode::Single;
  BufferingMode output = BufferingMode::Single;

  bool operator==(const BufferingPolicy &other) const {
    return input == other.input && weight == other.weight && output == other.output;
  }
};

// "SSS", "DSS", ..., "DDD" -- position order is always Input/Weight/Output
// (Section 2), matching this header's own field order.
std::string policyLabel(const BufferingPolicy &policy);

// The fixed 8-entry enumeration, in EXACTLY this order (Section 2):
//   SSS, DSS, SDS, SSD, DDS, DSD, SDD, DDD
// This is part of Slice 8's determinism contract: generateExtendedCandidates
// (AsyncDMASchedule.h) iterates buffering policies in exactly this order.
// Full DDD is never assumed legal or optimal merely by being last.
const std::vector<BufferingPolicy> &allBufferingPolicies();

} // namespace mlir::costmodel
