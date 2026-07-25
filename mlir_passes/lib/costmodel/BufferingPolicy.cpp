#include "costmodel/BufferingPolicy.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir::costmodel {

llvm::StringRef toString(BufferingMode mode) {
  switch (mode) {
  case BufferingMode::Single:
    return "single";
  case BufferingMode::Double:
    return "double";
  }
  llvm_unreachable("unhandled BufferingMode");
}

char toChar(BufferingMode mode) {
  switch (mode) {
  case BufferingMode::Single:
    return 'S';
  case BufferingMode::Double:
    return 'D';
  }
  llvm_unreachable("unhandled BufferingMode");
}

std::uint32_t slotCount(BufferingMode mode) {
  switch (mode) {
  case BufferingMode::Single:
    return 1;
  case BufferingMode::Double:
    return 2;
  }
  llvm_unreachable("unhandled BufferingMode");
}

std::string policyLabel(const BufferingPolicy &policy) {
  std::string label;
  label.push_back(toChar(policy.input));
  label.push_back(toChar(policy.weight));
  label.push_back(toChar(policy.output));
  return label;
}

const std::vector<BufferingPolicy> &allBufferingPolicies() {
  static const std::vector<BufferingPolicy> kPolicies = [] {
    using M = BufferingMode;
    // Fixed order (Section 2): SSS, DSS, SDS, SSD, DDS, DSD, SDD, DDD.
    return std::vector<BufferingPolicy>{
        BufferingPolicy{M::Single, M::Single, M::Single},
        BufferingPolicy{M::Double, M::Single, M::Single},
        BufferingPolicy{M::Single, M::Double, M::Single},
        BufferingPolicy{M::Single, M::Single, M::Double},
        BufferingPolicy{M::Double, M::Double, M::Single},
        BufferingPolicy{M::Double, M::Single, M::Double},
        BufferingPolicy{M::Single, M::Double, M::Double},
        BufferingPolicy{M::Double, M::Double, M::Double},
    };
  }();
  return kPolicies;
}

} // namespace mlir::costmodel
