#pragma once

// CapabilityBundle.h — six-layer capability context for compiler planning.
//
// CapabilityBundle aggregates all static capability information available at
// compile time.  It is assembled once at the tool boundary before any pass
// runs; no pass writes to it.
//
// BackendCapability and KernelLibraryCapability are defined in
// serving/TargetConstraints.h and reused here without duplication.
// New layers (HardwareCapability, ModelCapability, WorkloadConstraints,
// DeploymentTarget) are defined in this file.
//
// Profile loading is NOT in this header; loaders live in lib/capability/ and
// tools/.  This file defines the typed contract only.
//
// Note: this header transitively includes mlir/IR/BuiltinOps.h via
// serving/TargetConstraints.h.  Build targets that use CapabilityBundle must
// include the MLIR/LLVM include path.

#include "serving/TargetConstraints.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::hir {

// Hardware layer — theoretical device capability from public docs or
// declared device profiles.  Neither this repo nor any compiler pass may
// claim measured hardware performance via this struct.
// truth_boundary: "public_docs | declared_profile"
struct HardwareCapability {
  std::string hardware_id;        // "nvidia_gtx1650_maxq", "apple_m1_pro"
  std::string vendor;             // "nvidia", "apple", "amd"
  std::string compute_family;     // "turing" (cc 7.5), "ampere", "apple_silicon"
  int64_t     memory_bytes = 0;   // total device memory — not the model budget
  bool        fp16_native  = false;
  bool        bf16_native  = false;
  bool        int8_native  = false;
  bool        int4_native  = false;
  bool        multi_device_supported = false;
  int64_t     device_count = 1;
  std::string truth_boundary;     // "public_docs | declared_profile"
};

// Model layer — architecture constants from the model being compiled.
// Populated from configs/models/{spec}.json; never from model weights.
// truth_boundary: "declared_model_config_not_full_graph_import"
struct ModelCapability {
  std::string model_id;                    // "qwen2.5-0.5b"
  std::string model_family;                // "transformer_decoder", "vision_transformer"
  int64_t     num_layers              = 0;
  int64_t     hidden_size             = 0;
  int64_t     num_attention_heads     = 0;
  int64_t     num_kv_heads            = 0; // non-zero implies GQA
  int64_t     intermediate_size       = 0;
  int64_t     vocab_size              = 0;
  int64_t     max_position_embeddings = 0;
  std::string native_dtype;                // "fp16", "bf16", "fp32"
  // attention_mechanism: "gqa" | "mha" | "mqa"
  std::string attention_mechanism;
  // positional_encoding: "rope" | "alibi" | "absolute" | "none"
  std::string positional_encoding;
  std::string truth_boundary;
};

// Workload layer — expected request characteristics that constrain decisions.
// truth_boundary: "declared_phase1_target" | "measured_production_traffic"
struct WorkloadConstraints {
  int64_t     max_prompt_tokens              = 0;
  int64_t     max_output_tokens              = 0;
  int64_t     expected_concurrency           = 1;
  bool        shared_prefix_expected         = false;
  // prompt_length_distribution: "short" | "short_to_medium" | "long"
  std::string prompt_length_distribution;
  std::string truth_boundary;
};

// Static cost weight parameters for ServingCostModelPass.
// Weights are tunable per deployment target; defaults match the V0 constants
// so behavior is unchanged when no profile specifies them.
// truth_boundary: all weights are declared_profile, not measured_latency.
struct StaticCostWeights {
  // Compute: algebraic rewrite adds work proportional to decomposition depth.
  int64_t compute_decomposition           =  5; // algebraic_decomposition overhead
  // Memory: dtype/format mismatch requires extra read+write passes.
  int64_t memory_repr_conv               =  3; // representation_conversion base
  // Boundary operation costs.
  int64_t dequant_cost                   =  3; // weight dequant boundary (double bandwidth vs cast)
  int64_t requant_cost                   =  2; // requant boundary
  int64_t layout_transform_boundary      =  2; // layout_transform boundary op (co-fused, lighter)
  int64_t layout_transform_standalone    =  4; // standalone layout_conversion candidate
  int64_t cast_cost                      =  2; // cast boundary or standalone cast_conversion
  // Routing costs.
  int64_t backend_switch_cost            = 20; // cross-backend fallback (deliberately high)
  int64_t launch_overhead_per_boundary   =  1; // per extra boundary kernel launch
  int64_t launch_overhead_fallback       =  2; // fallback routing overhead beyond per-boundary
  // Transfer costs (conditional on attrs).
  int64_t transfer_cost_fallback         =  5; // conservative static estimate for device crossing
};

// Deployment target layer — compile-session-specific constraints.
// Populated from configs/target_profiles/{profile}.json.
// truth_boundary: "declared_profile"
struct DeploymentTarget {
  std::string target_profile_id;
  double      memory_budget_fraction           = 0.75; // fraction of hardware memory for model+kv
  int64_t     parallelism_degree               = 1;
  // parallelism_kind: "none" | "tensor_parallel" | "pipeline_parallel"
  std::string parallelism_kind;
  bool        static_shape_required            = false;
  double      latency_budget_ms                = 0.0;  // 0 = unconstrained
  // preferred_serving_topology: "colocated" | "prefill_decode_split" | "any"
  std::string preferred_serving_topology;
  // Cost constants for compiler planning formulas — not measured latency.
  double      prefill_ms_per_token             = 0.0;
  double      decode_ms_per_token              = 0.0;
  double      inter_device_bandwidth_mb_per_ms = 0.0;
  StaticCostWeights static_cost_weights;        // tunable cost model weights
  std::string truth_boundary;
};

// CapabilityBundle — the complete read-only capability context for one
// compilation session.  Assembled from multiple profile sources before
// any pass runs.
//
// Source mapping:
//   hardware   <- ml-platform-capabilities/profiles/hardware/{id}.json
//   backends[] <- ml-platform-capabilities/profiles/backend/{id}.json
//   kernels[]  <- ml-platform-capabilities/profiles/kernels/{id}.json
//   workload   <- ml-platform-capabilities/profiles/workloads/{id}.json
//   model      <- configs/models/{spec}.json              (compiler-local)
//   deployment <- configs/target_profiles/{profile}.json  (compiler-local)
//
// BackendCapability and KernelLibraryCapability are the types from
// serving/TargetConstraints.h — the same types already used by compiler
// passes.  No duplication; same struct, new home.
struct CapabilityBundle {
  HardwareCapability                   hardware;
  std::vector<BackendCapability>       backends;
  std::vector<KernelLibraryCapability> kernels;
  ModelCapability                      model;
  WorkloadConstraints                  workload;
  DeploymentTarget                     deployment;
};

} // namespace mlir::hir
