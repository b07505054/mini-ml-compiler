#pragma once

// Compiler-side typed contract for target.* MLIR module attributes.
//
// A TargetLowering step (outside compiler core) converts an external
// TargetDeviceProfile (from PocketChef, desktop config, or CI fixture) into
// target.* MLIR module attributes.  MLIR passes consume those attrs directly
// via getAttrOfType<>; this struct provides a typed C++ view used by tools
// and tests that need to inject or round-trip the constraint set without
// hand-writing MLIR attribute syntax.
//
// Runtime-dynamic fields (thermal_state, power_policy, live memory footprint)
// are NOT in this struct — they belong in the runtime layer.

#include "mlir/IR/BuiltinOps.h"

#include <string>
#include <vector>

namespace mlir::hir {

struct TargetConstraints {
  // ---- Compile-time constraint fields ------------------------------------
  // Absent in module → field stays at its default (unconstrained).
  // Check the corresponding has_* flag for numeric/bool fields where the
  // default value (0.0 / true) is ambiguous with "absent".

  std::string profile_id;               // target.profile_id  — "" if absent
  double      memory_budget_mb    = 0.0; // target.memory_budget_mb
  bool        static_shape_support = true; // target.static_shape_support
  double      frame_latency_budget_ms = 0.0; // target.frame_latency_budget_ms
  std::string preferred_backend;        // target.preferred_backend — "" if absent
  std::vector<std::string> allowed_backends;            // target.allowed_backends (ArrayAttr)
  std::vector<std::string> supported_precisions;        // target.supported_precisions (ArrayAttr)
  std::vector<std::string> paged_kv_compatible_backends; // target.paged_kv_compatible_backends (ArrayAttr)
                                                         // Empty = no backend supports paged KV on this target.

  // Presence flags: set only when the corresponding attr was found in the
  // module.  Needed for fields whose zero/false value is a valid constraint.
  bool has_memory_budget        = false;
  bool has_static_shape_support = false;
  bool has_frame_latency_budget = false;

  // Read target.* attrs from a ModuleOp.  Absent attrs leave fields at their
  // defaults and leave presence flags false.
  static TargetConstraints fromModule(mlir::ModuleOp module);

  // Write non-absent fields as target.* attrs on the ModuleOp.  Used by
  // TargetLowering tools and tests that build a constraint profile in C++
  // and need it reflected in the MLIR module before compilation begins.
  void attachToModule(mlir::ModuleOp module, mlir::MLIRContext *ctx) const;
};

} // namespace mlir::hir
