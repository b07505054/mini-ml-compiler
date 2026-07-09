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

#include <optional>
#include <string>
#include <vector>

namespace mlir::hir {

// ---------------------------------------------------------------------------
// BackendCapability
//
// Declared hardware capability for one named backend (e.g. "coreml", "cuda",
// "arm_compute", "amx", "mychip_npu").  Populated from a target profile JSON
// and lowered into target.backend_capabilities.{backend}.* MLIR module attrs.
//
// source_level labels:
//   "public_docs"       — fact appears in publicly released technical docs
//   "declared_profile"  — declared in this project's profile JSON, not measured
//   "estimated"         — derived from formula/assumption, not measured
//   "unknown"           — not publicly documented at required granularity
//   "measured"          — obtained from a benchmark on actual hardware
//
// Cost fields and alignment fields use std::optional:
//   std::nullopt  → unknown; attachToModule() does NOT emit the attr.
//   Absent in MLIR attrs means unknown.  0.0 would mean free; 0 alignment would
//   mean no requirement.  Missing and zero are distinct.
// ---------------------------------------------------------------------------
struct BackendCapability {
  std::string backend_name;  // "cpu" | "gpu" | "tensor_core" | "amx" | "coreml" | ...
  std::string backend_api;   // "coreml" | "metal" | "cuda" | "arm_compute" | "iree_hal" | ...

  std::vector<std::string> supported_ops;              // op families this backend accelerates
  std::vector<std::string> supported_dtypes;           // e.g. ["fp16", "int8"]
  std::vector<std::string> accumulation_dtypes;        // e.g. ["fp32"]
  std::vector<std::string> supported_quant_modes;      // "static_int8" | "weight_only" | ...
  std::vector<std::string> preferred_activation_layouts; // ["NHWC"] | ["abstracted_by_coreml"]
  std::vector<std::string> preferred_weight_layouts;     // ["HWIO"] | ["blocked_kc"]
  std::vector<std::string> layout_agnostic_ops;        // ops that propagate input layout unchanged

  bool supports_layout_transform = false;
  bool supports_cast             = false;
  bool supports_dequant_boundary = false;
  bool supports_requantize       = false;

  std::vector<std::string> supports_fusion_patterns;  // ["matmul_bias_relu", ...]

  // ---- Level 2: Hardware Constraints ------------------------------------
  // nullopt / empty string = not declared in profile (do not emit attr).

  std::optional<int64_t> required_k_alignment;         // nullopt = no documented requirement
  std::optional<int64_t> required_n_alignment;
  std::optional<int64_t> required_m_alignment;

  std::vector<std::string> allowed_quant_granularity;  // ["per_channel", "per_tensor"]
  std::string required_activation_quant_mode;          // empty = no requirement
  std::string required_weight_quant_mode;
  std::optional<int64_t>  max_supported_rank;          // nullopt = no declared limit
  std::optional<bool>     requires_static_shape;       // nullopt = not declared
  std::optional<bool>     requires_constant_weight;    // nullopt = not declared
  std::vector<std::string> unsupported_ops;
  std::string fallback_backend;                        // empty = no fallback declared

  // ---- Level 3: Hardware Preferences ------------------------------------

  std::vector<std::string> acceptable_activation_layouts;
  std::vector<std::string> acceptable_weight_layouts;
  std::vector<std::string> preferred_dtypes;
  std::vector<std::string> acceptable_dtypes;
  std::vector<std::string> preferred_fusion_patterns;

  // ---- Level 4: Runtime Model (planning-only; do not use in passes) -----
  // Stored as descriptive strings from public documentation.
  // Empty = unknown; do not emit attr.

  std::string runtime_model_launch_overhead;
  std::string runtime_model_dma_transfer;
  std::string runtime_model_memory_hierarchy;
  std::string runtime_model_occupancy;
  std::string runtime_model_register_pressure;
  std::string runtime_model_shared_memory;
  std::string runtime_model_source_level;
  std::string runtime_model_truth_boundary;

  // ---- Level 5: Cost Model Placeholder (planning-only; do not evaluate) --

  std::string cost_model_kind;
  std::string cost_model_inputs;
  std::string cost_model_source_level;
  std::string cost_model_truth_boundary;

  // ---- Legacy cost params -----------------------------------------------
  // nullopt = unknown (do not emit attr). 0.0 = free. Distinct.
  std::optional<double> layout_transform_cost_ms;
  std::optional<double> cast_cost_ms;
  std::optional<double> quantize_cost_ms;
  std::optional<double> dequantize_cost_ms;
  std::optional<double> requantize_cost_ms;
  std::optional<double> backend_transfer_cost_ms;

  std::string source_level;
  std::string truth_boundary;
};

// ---------------------------------------------------------------------------
// KernelLibraryCapability  (Kernel Layer — Layer 3)
//
// Describes whether an actual kernel exists for a specific
// (op_type, backend, dtype, layout, quant_mode) combination.
// Separates three layers:
//   HardwareCapability  : hardware theoretically supports a feature.
//   BackendCapability   : backend/compiler API exposes it.
//   KernelLibraryCapability : an actual kernel exists for the specific tuple.
//
// source_level labels use the same convention as BackendCapability.
// truth_boundary must acknowledge that kernel availability is declared from
// public docs or profile — not verified at runtime.
// ---------------------------------------------------------------------------
struct KernelLibraryCapability {
  std::string op_type;          // "matmul" | "conv2d" | "softmax" | ...
  std::string kernel_name;      // e.g. "coreml_matmul", "cublaslt_matmul"
  std::string backend;          // "coreml" | "tensor_core" | "arm_compute" | ...
  std::string kernel_library;   // "coreml_builtin" | "cublasLt" | "TensorRT" | ...

  std::vector<std::string> supported_dtypes;       // empty = any
  std::vector<std::string> supported_layouts;      // empty = any / layout-agnostic
  std::vector<std::string> supported_quant_modes;  // empty = any

  std::optional<int64_t> required_m_alignment;
  std::optional<int64_t> required_n_alignment;
  std::optional<int64_t> required_k_alignment;

  bool supports_dynamic_shape    = true;
  bool requires_constant_weight  = false;
  bool supports_fusion           = false;

  std::vector<std::string> fusion_patterns;   // declared fusion patterns
  std::vector<std::string> rewrite_patterns;  // patterns that rewrite to a matchable form

  std::string fallback_kernel;   // empty = no named fallback kernel
  std::string fallback_backend;  // empty = no fallback backend

  std::string source_level;    // "public_docs" | "declared_profile" | "estimated"
  std::string truth_boundary;  // must name this kernel layer claim
};

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

  // Serving cost constants (target.prefill_ms_per_token, etc.).
  // Absent → ServingPhaseAnalysisPass falls back to built-in formula defaults.
  double prefill_ms_per_token   = 0.0; // target.prefill_ms_per_token
  double decode_ms_per_token    = 0.0; // target.decode_ms_per_token
  double pd_bandwidth_mb_per_ms = 0.0; // target.pd_bandwidth_mb_per_ms

  // Static cost profile (target.static_cost_profile.*): declared theoretical
  // peak numbers consumed by shape_cost_model_v2 in ServingCostModelPass.
  // 0 = not declared → the cost model emits shape facts only, no time
  // estimates. These are public-docs/declared-profile peaks, never measured.
  double static_cost_peak_flops_fp32             = 0.0;
  double static_cost_peak_flops_fp16             = 0.0;
  double static_cost_peak_flops_int8             = 0.0;
  double static_cost_memory_bandwidth_bytes_per_sec = 0.0;
  // Memory hierarchy (declared capacities/capabilities, never measured):
  // local memory = SRAM / shared memory / scratchpad per compute unit.
  // 0 = not declared → TilePlanningPass is inert.
  int64_t static_cost_local_memory_bytes         = 0;
  int64_t static_cost_cache_line_bytes           = 0;
  bool static_cost_supports_async_copy           = false;
  bool static_cost_supports_dma                  = false;
  bool has_static_cost_supports_async_copy       = false; // declared in profile
  bool has_static_cost_supports_dma              = false; // declared in profile
  std::string static_cost_profile_truth_boundary;  // "" when absent

  // Presence flags: set only when the corresponding attr was found in the
  // module.  Needed for fields whose zero/false value is a valid constraint.
  bool has_memory_budget           = false;
  bool has_static_shape_support    = false;
  bool has_frame_latency_budget    = false;
  bool has_prefill_ms_per_token    = false;
  bool has_decode_ms_per_token     = false;
  bool has_pd_bandwidth_mb_per_ms  = false;

  // Per-backend hardware capability declarations.
  // Lowered from target profile JSON; consumed by compiler passes via
  // target.backend_capabilities.{backend}.* MLIR module attrs.
  // Empty when the profile predates the backendCapabilities schema.
  std::vector<BackendCapability> backend_capabilities;

  // Per-op kernel library capability declarations (Kernel Layer — Layer 3).
  // Describes actual kernel existence for op + dtype + layout + quant mode.
  // Lowered from target profile JSON kernelLibraries array; stored in module
  // as target.kernel_libraries.{backend} ArrayAttr of DictionaryAttr.
  // Empty when the profile predates the kernelLibraries schema.
  std::vector<KernelLibraryCapability> kernel_library_capabilities;

  // Read target.* attrs from a ModuleOp.  Absent attrs leave fields at their
  // defaults and leave presence flags false.
  static TargetConstraints fromModule(mlir::ModuleOp module);

  // Write non-absent fields as target.* attrs on the ModuleOp.  Used by
  // TargetLowering tools and tests that build a constraint profile in C++
  // and need it reflected in the MLIR module before compilation begins.
  void attachToModule(mlir::ModuleOp module, mlir::MLIRContext *ctx) const;
};

} // namespace mlir::hir
