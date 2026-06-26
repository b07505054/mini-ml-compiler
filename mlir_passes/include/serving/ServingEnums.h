#pragma once

namespace mlir::hir {

enum class ServingPhase  { Prefill, Decode, Unknown };
enum class ExecutionMode { Colocated, PDSplit, Unknown };
enum class KVLayout      { Paged, Contiguous, Unknown };
enum class Confidence    { Low, Medium, High };

} // namespace mlir::hir
