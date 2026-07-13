#include "serving/TargetConstraints.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::hir {

// ---------------------------------------------------------------------------
// Helpers for flat-prefix backend capability attr names.
// Attr pattern: "target.backend_capabilities.{backend}.{field}"
// ---------------------------------------------------------------------------

static std::string bcAttr(llvm::StringRef backend, llvm::StringRef field) {
  return ("target.backend_capabilities." + backend + "." + field).str();
}

static std::vector<std::string>
readStringArray(mlir::Operation *op, llvm::StringRef attrName) {
  std::vector<std::string> result;
  if (auto a = op->getAttrOfType<mlir::ArrayAttr>(attrName))
    for (mlir::Attribute elem : a)
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
        result.push_back(s.getValue().str());
  return result;
}

TargetConstraints TargetConstraints::fromModule(mlir::ModuleOp module) {
  TargetConstraints tc;
  mlir::Operation *op = module.getOperation();

  if (auto a = op->getAttrOfType<mlir::StringAttr>("target.profile_id"))
    tc.profile_id = a.getValue().str();
  if (auto a = op->getAttrOfType<mlir::StringAttr>("target.profile_kind"))
    tc.profile_kind = a.getValue().str();

  if (auto a = op->getAttrOfType<mlir::FloatAttr>("target.memory_budget_mb")) {
    tc.memory_budget_mb  = a.getValueAsDouble();
    tc.has_memory_budget = true;
  }

  if (auto a = op->getAttrOfType<mlir::BoolAttr>("target.static_shape_support")) {
    tc.static_shape_support  = a.getValue();
    tc.has_static_shape_support = true;
  }

  if (auto a = op->getAttrOfType<mlir::FloatAttr>("target.frame_latency_budget_ms")) {
    tc.frame_latency_budget_ms = a.getValueAsDouble();
    tc.has_frame_latency_budget = true;
  }

  if (auto a = op->getAttrOfType<mlir::StringAttr>("target.preferred_backend"))
    tc.preferred_backend = a.getValue().str();

  if (auto a = op->getAttrOfType<mlir::ArrayAttr>("target.allowed_backends")) {
    for (mlir::Attribute elem : a)
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
        tc.allowed_backends.push_back(s.getValue().str());
  }

  if (auto a = op->getAttrOfType<mlir::ArrayAttr>("target.supported_precisions")) {
    for (mlir::Attribute elem : a)
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
        tc.supported_precisions.push_back(s.getValue().str());
  }

  if (auto a = op->getAttrOfType<mlir::ArrayAttr>("target.paged_kv_compatible_backends")) {
    for (mlir::Attribute elem : a)
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
        tc.paged_kv_compatible_backends.push_back(s.getValue().str());
  }

  if (auto a = op->getAttrOfType<mlir::FloatAttr>("target.prefill_ms_per_token")) {
    tc.prefill_ms_per_token    = a.getValueAsDouble();
    tc.has_prefill_ms_per_token = true;
  }
  if (auto a = op->getAttrOfType<mlir::FloatAttr>("target.decode_ms_per_token")) {
    tc.decode_ms_per_token    = a.getValueAsDouble();
    tc.has_decode_ms_per_token = true;
  }
  if (auto a = op->getAttrOfType<mlir::FloatAttr>("target.pd_bandwidth_mb_per_ms")) {
    tc.pd_bandwidth_mb_per_ms    = a.getValueAsDouble();
    tc.has_pd_bandwidth_mb_per_ms = true;
  }

  if (auto a = op->getAttrOfType<mlir::FloatAttr>(
          "target.static_cost_profile.peak_flops_fp32"))
    tc.static_cost_peak_flops_fp32 = a.getValueAsDouble();
  if (auto a = op->getAttrOfType<mlir::FloatAttr>(
          "target.static_cost_profile.peak_flops_fp16"))
    tc.static_cost_peak_flops_fp16 = a.getValueAsDouble();
  if (auto a = op->getAttrOfType<mlir::FloatAttr>(
          "target.static_cost_profile.peak_flops_int8"))
    tc.static_cost_peak_flops_int8 = a.getValueAsDouble();
  if (auto a = op->getAttrOfType<mlir::FloatAttr>(
          "target.static_cost_profile.memory_bandwidth_bytes_per_sec"))
    tc.static_cost_memory_bandwidth_bytes_per_sec = a.getValueAsDouble();
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>(
          "target.static_cost_profile.local_memory_bytes"))
    tc.static_cost_local_memory_bytes = a.getInt();
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>(
          "target.static_cost_profile.cache_line_bytes"))
    tc.static_cost_cache_line_bytes = a.getInt();
  if (auto a = op->getAttrOfType<mlir::BoolAttr>(
          "target.static_cost_profile.supports_async_copy")) {
    tc.static_cost_supports_async_copy = a.getValue();
    tc.has_static_cost_supports_async_copy = true;
  }
  if (auto a = op->getAttrOfType<mlir::BoolAttr>(
          "target.static_cost_profile.supports_dma")) {
    tc.static_cost_supports_dma = a.getValue();
    tc.has_static_cost_supports_dma = true;
  }
  if (auto a = op->getAttrOfType<mlir::StringAttr>(
          "target.static_cost_profile.truth_boundary"))
    tc.static_cost_profile_truth_boundary = a.getValue().str();

  if (auto a = op->getAttrOfType<mlir::IntegerAttr>(
          "target.hardware.physical_compute_units"))
    tc.hardware_execution_profile.physical_compute_units = a.getInt();
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>(
          "target.hardware.effective_compute_units"))
    tc.hardware_execution_profile.effective_compute_units = a.getInt();
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>(
          "target.hardware.max_concurrent_work_items_per_unit"))
    tc.hardware_execution_profile.max_concurrent_work_items_per_unit =
        a.getInt();
  if (auto a = op->getAttrOfType<mlir::BoolAttr>(
          "target.hardware.supports_latency_hiding"))
    tc.hardware_execution_profile.supports_latency_hiding = a.getValue();
  if (auto a = op->getAttrOfType<mlir::StringAttr>(
          "target.hardware.local_memory_kind"))
    tc.hardware_execution_profile.local_memory_kind = a.getValue().str();

  // Read per-backend capability entries.  The index attr lists backend names;
  // per-backend fields use flat prefix "target.backend_capabilities.{name}.*".
  // Cost and alignment fields use std::optional: absent attr → std::nullopt.
  if (auto names = op->getAttrOfType<mlir::ArrayAttr>(
          "target.backend_capability_names")) {
    for (mlir::Attribute elem : names) {
      auto s = mlir::dyn_cast<mlir::StringAttr>(elem);
      if (!s)
        continue;
      std::string n = s.getValue().str();

      BackendCapability cap;
      cap.backend_name = n;

      auto readStr = [&](llvm::StringRef field) -> std::string {
        if (auto a = op->getAttrOfType<mlir::StringAttr>(bcAttr(n, field)))
          return a.getValue().str();
        return "";
      };
      auto readBool = [&](llvm::StringRef field) -> bool {
        if (auto a = op->getAttrOfType<mlir::BoolAttr>(bcAttr(n, field)))
          return a.getValue();
        return false;
      };
      auto readOptInt = [&](llvm::StringRef field) -> std::optional<int64_t> {
        if (auto a = op->getAttrOfType<mlir::IntegerAttr>(bcAttr(n, field)))
          return a.getInt();
        return std::nullopt;
      };
      auto readOptCost = [&](llvm::StringRef field) -> std::optional<double> {
        if (auto a = op->getAttrOfType<mlir::FloatAttr>(bcAttr(n, field)))
          return a.getValueAsDouble();
        return std::nullopt; // absent attr → unknown
      };

      cap.backend_api                  = readStr("backend_api");
      cap.supported_ops                = readStringArray(op, bcAttr(n, "supported_ops"));
      cap.supported_dtypes             = readStringArray(op, bcAttr(n, "supported_dtypes"));
      cap.accumulation_dtypes          = readStringArray(op, bcAttr(n, "accumulation_dtypes"));
      cap.supported_quant_modes        = readStringArray(op, bcAttr(n, "supported_quant_modes"));
      cap.preferred_activation_layouts = readStringArray(op, bcAttr(n, "preferred_activation_layouts"));
      cap.preferred_weight_layouts     = readStringArray(op, bcAttr(n, "preferred_weight_layouts"));
      cap.layout_agnostic_ops          = readStringArray(op, bcAttr(n, "layout_agnostic_ops"));
      cap.supports_layout_transform    = readBool("supports_layout_transform");
      cap.supports_cast                = readBool("supports_cast");
      cap.supports_dequant_boundary    = readBool("supports_dequant_boundary");
      cap.supports_requantize          = readBool("supports_requantize");
      cap.supports_fusion_patterns     = readStringArray(op, bcAttr(n, "supports_fusion_patterns"));

      // Level 2: Hardware Constraints
      cap.required_k_alignment         = readOptInt("required_k_alignment");
      cap.required_n_alignment         = readOptInt("required_n_alignment");
      cap.required_m_alignment         = readOptInt("required_m_alignment");
      cap.allowed_quant_granularity    = readStringArray(op, bcAttr(n, "allowed_quant_granularity"));
      cap.required_activation_quant_mode = readStr("required_activation_quant_mode");
      cap.required_weight_quant_mode   = readStr("required_weight_quant_mode");
      cap.max_supported_rank           = readOptInt("max_supported_rank");
      if (auto a = op->getAttrOfType<mlir::BoolAttr>(bcAttr(n, "requires_static_shape")))
        cap.requires_static_shape = a.getValue();
      if (auto a = op->getAttrOfType<mlir::BoolAttr>(bcAttr(n, "requires_constant_weight")))
        cap.requires_constant_weight = a.getValue();
      cap.unsupported_ops              = readStringArray(op, bcAttr(n, "unsupported_ops"));
      cap.fallback_backend             = readStr("fallback_backend");

      // Level 3: Hardware Preferences
      cap.acceptable_activation_layouts = readStringArray(op, bcAttr(n, "acceptable_activation_layouts"));
      cap.acceptable_weight_layouts    = readStringArray(op, bcAttr(n, "acceptable_weight_layouts"));
      cap.preferred_dtypes             = readStringArray(op, bcAttr(n, "preferred_dtypes"));
      cap.acceptable_dtypes            = readStringArray(op, bcAttr(n, "acceptable_dtypes"));
      cap.preferred_fusion_patterns    = readStringArray(op, bcAttr(n, "preferred_fusion_patterns"));

      // Level 4: Runtime Model
      cap.runtime_model_launch_overhead   = readStr("runtime_model_launch_overhead");
      cap.runtime_model_dma_transfer      = readStr("runtime_model_dma_transfer");
      cap.runtime_model_memory_hierarchy  = readStr("runtime_model_memory_hierarchy");
      cap.runtime_model_occupancy         = readStr("runtime_model_occupancy");
      cap.runtime_model_register_pressure = readStr("runtime_model_register_pressure");
      cap.runtime_model_shared_memory     = readStr("runtime_model_shared_memory");
      cap.runtime_model_source_level      = readStr("runtime_model_source_level");
      cap.runtime_model_truth_boundary    = readStr("runtime_model_truth_boundary");

      // Level 5: Cost Model Placeholder
      cap.cost_model_kind            = readStr("cost_model_kind");
      cap.cost_model_inputs          = readStr("cost_model_inputs");
      cap.cost_model_source_level    = readStr("cost_model_source_level");
      cap.cost_model_truth_boundary  = readStr("cost_model_truth_boundary");

      // Legacy cost params
      cap.layout_transform_cost_ms     = readOptCost("layout_transform_cost_ms");
      cap.cast_cost_ms                 = readOptCost("cast_cost_ms");
      cap.quantize_cost_ms             = readOptCost("quantize_cost_ms");
      cap.dequantize_cost_ms           = readOptCost("dequantize_cost_ms");
      cap.requantize_cost_ms           = readOptCost("requantize_cost_ms");
      cap.backend_transfer_cost_ms     = readOptCost("backend_transfer_cost_ms");
      cap.source_level                 = readStr("source_level");
      cap.truth_boundary               = readStr("truth_boundary");

      tc.backend_capabilities.push_back(std::move(cap));
    }
  }

  // Read kernel library capabilities.
  // Index: target.kernel_library_backends lists unique backend names.
  // Per-backend entries: target.kernel_libraries.{backend} = ArrayAttr of DictionaryAttr.
  if (auto names = op->getAttrOfType<mlir::ArrayAttr>("target.kernel_library_backends")) {
    for (mlir::Attribute nameElem : names) {
      auto ns = mlir::dyn_cast<mlir::StringAttr>(nameElem);
      if (!ns) continue;
      std::string backendName = ns.getValue().str();
      auto attrKey = "target.kernel_libraries." + backendName;
      auto arr = op->getAttrOfType<mlir::ArrayAttr>(attrKey);
      if (!arr) continue;
      for (mlir::Attribute elem : arr) {
        auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(elem);
        if (!dict) continue;
        KernelLibraryCapability ke;
        auto rStr = [&](llvm::StringRef k) -> std::string {
          if (auto a = dict.get(k))
            if (auto s = mlir::dyn_cast<mlir::StringAttr>(a)) return s.getValue().str();
          return {};
        };
        auto rStrs = [&](llvm::StringRef k) -> std::vector<std::string> {
          std::vector<std::string> r;
          if (auto a = dict.get(k))
            if (auto arr2 = mlir::dyn_cast<mlir::ArrayAttr>(a))
              for (auto e : arr2)
                if (auto s = mlir::dyn_cast<mlir::StringAttr>(e)) r.push_back(s.getValue().str());
          return r;
        };
        auto rBool = [&](llvm::StringRef k, bool def) -> bool {
          if (auto a = dict.get(k))
            if (auto b = mlir::dyn_cast<mlir::BoolAttr>(a)) return b.getValue();
          return def;
        };
        auto rOptInt = [&](llvm::StringRef k) -> std::optional<int64_t> {
          if (auto a = dict.get(k))
            if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(a)) return i.getInt();
          return std::nullopt;
        };
        ke.op_type                  = rStr("op_type");
        ke.kernel_name              = rStr("kernel_name");
        ke.backend                  = backendName;
        ke.kernel_library           = rStr("kernel_library");
        ke.supported_dtypes         = rStrs("supported_dtypes");
        ke.supported_layouts        = rStrs("supported_layouts");
        ke.supported_quant_modes    = rStrs("supported_quant_modes");
        ke.required_m_alignment     = rOptInt("required_m_alignment");
        ke.required_n_alignment     = rOptInt("required_n_alignment");
        ke.required_k_alignment     = rOptInt("required_k_alignment");
        ke.supports_dynamic_shape   = rBool("supports_dynamic_shape", true);
        ke.requires_constant_weight = rBool("requires_constant_weight", false);
        ke.supports_fusion          = rBool("supports_fusion", false);
        ke.fusion_patterns          = rStrs("fusion_patterns");
        ke.rewrite_patterns         = rStrs("rewrite_patterns");
        ke.fallback_kernel          = rStr("fallback_kernel");
        ke.fallback_backend         = rStr("fallback_backend");
        ke.source_level             = rStr("source_level");
        ke.truth_boundary           = rStr("truth_boundary");
        tc.kernel_library_capabilities.push_back(std::move(ke));
      }
    }
  }

  return tc;
}

void TargetConstraints::attachToModule(mlir::ModuleOp module,
                                       mlir::MLIRContext *ctx) const {
  mlir::Operation *op = module.getOperation();
  mlir::Type f64 = mlir::Float64Type::get(ctx);

  if (!profile_id.empty())
    op->setAttr("target.profile_id", mlir::StringAttr::get(ctx, profile_id));
  if (!profile_kind.empty())
    op->setAttr("target.profile_kind", mlir::StringAttr::get(ctx, profile_kind));

  if (has_memory_budget)
    op->setAttr("target.memory_budget_mb",
                mlir::FloatAttr::get(f64, memory_budget_mb));

  if (has_static_shape_support)
    op->setAttr("target.static_shape_support",
                mlir::BoolAttr::get(ctx, static_shape_support));

  if (has_frame_latency_budget)
    op->setAttr("target.frame_latency_budget_ms",
                mlir::FloatAttr::get(f64, frame_latency_budget_ms));

  if (!preferred_backend.empty())
    op->setAttr("target.preferred_backend",
                mlir::StringAttr::get(ctx, preferred_backend));

  if (!allowed_backends.empty()) {
    llvm::SmallVector<mlir::Attribute> elems;
    for (const auto &b : allowed_backends)
      elems.push_back(mlir::StringAttr::get(ctx, b));
    op->setAttr("target.allowed_backends", mlir::ArrayAttr::get(ctx, elems));
  }

  if (!supported_precisions.empty()) {
    llvm::SmallVector<mlir::Attribute> elems;
    for (const auto &p : supported_precisions)
      elems.push_back(mlir::StringAttr::get(ctx, p));
    op->setAttr("target.supported_precisions",
                mlir::ArrayAttr::get(ctx, elems));
  }

  // Always emit target.paged_kv_compatible_backends — even when empty — so
  // passes can distinguish "profile says no paged-KV backends" from
  // "profile was not lowered" (which leaves the attr absent).
  {
    llvm::SmallVector<mlir::Attribute> elems;
    for (const auto &b : paged_kv_compatible_backends)
      elems.push_back(mlir::StringAttr::get(ctx, b));
    op->setAttr("target.paged_kv_compatible_backends",
                mlir::ArrayAttr::get(ctx, elems));
  }

  if (has_prefill_ms_per_token)
    op->setAttr("target.prefill_ms_per_token",
                mlir::FloatAttr::get(f64, prefill_ms_per_token));
  if (has_decode_ms_per_token)
    op->setAttr("target.decode_ms_per_token",
                mlir::FloatAttr::get(f64, decode_ms_per_token));
  if (has_pd_bandwidth_mb_per_ms)
    op->setAttr("target.pd_bandwidth_mb_per_ms",
                mlir::FloatAttr::get(f64, pd_bandwidth_mb_per_ms));

  // Static cost profile attrs for shape_cost_model_v2. Only declared (> 0)
  // values are emitted; absent attrs mean "not declared", and the cost model
  // then produces shape facts without time estimates.
  if (static_cost_peak_flops_fp32 > 0.0)
    op->setAttr("target.static_cost_profile.peak_flops_fp32",
                mlir::FloatAttr::get(f64, static_cost_peak_flops_fp32));
  if (static_cost_peak_flops_fp16 > 0.0)
    op->setAttr("target.static_cost_profile.peak_flops_fp16",
                mlir::FloatAttr::get(f64, static_cost_peak_flops_fp16));
  if (static_cost_peak_flops_int8 > 0.0)
    op->setAttr("target.static_cost_profile.peak_flops_int8",
                mlir::FloatAttr::get(f64, static_cost_peak_flops_int8));
  if (static_cost_memory_bandwidth_bytes_per_sec > 0.0)
    op->setAttr("target.static_cost_profile.memory_bandwidth_bytes_per_sec",
                mlir::FloatAttr::get(
                    f64, static_cost_memory_bandwidth_bytes_per_sec));
  if (static_cost_local_memory_bytes > 0)
    op->setAttr("target.static_cost_profile.local_memory_bytes",
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       static_cost_local_memory_bytes));
  if (static_cost_cache_line_bytes > 0)
    op->setAttr("target.static_cost_profile.cache_line_bytes",
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       static_cost_cache_line_bytes));
  if (has_static_cost_supports_async_copy)
    op->setAttr("target.static_cost_profile.supports_async_copy",
                mlir::BoolAttr::get(ctx, static_cost_supports_async_copy));
  if (has_static_cost_supports_dma)
    op->setAttr("target.static_cost_profile.supports_dma",
                mlir::BoolAttr::get(ctx, static_cost_supports_dma));
  if (!static_cost_profile_truth_boundary.empty())
    op->setAttr("target.static_cost_profile.truth_boundary",
                mlir::StringAttr::get(ctx,
                                      static_cost_profile_truth_boundary));

  if (hardware_execution_profile.physical_compute_units.has_value())
    op->setAttr("target.hardware.physical_compute_units",
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       *hardware_execution_profile
                                            .physical_compute_units));
  if (hardware_execution_profile.effective_compute_units.has_value())
    op->setAttr("target.hardware.effective_compute_units",
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                       *hardware_execution_profile
                                            .effective_compute_units));
  if (hardware_execution_profile.max_concurrent_work_items_per_unit
          .has_value())
    op->setAttr("target.hardware.max_concurrent_work_items_per_unit",
                mlir::IntegerAttr::get(
                    mlir::IntegerType::get(ctx, 64),
                    *hardware_execution_profile
                         .max_concurrent_work_items_per_unit));
  if (hardware_execution_profile.supports_latency_hiding.has_value())
    op->setAttr("target.hardware.supports_latency_hiding",
                mlir::BoolAttr::get(
                    ctx,
                    *hardware_execution_profile.supports_latency_hiding));
  if (hardware_execution_profile.local_memory_kind.has_value())
    op->setAttr("target.hardware.local_memory_kind",
                mlir::StringAttr::get(
                    ctx, *hardware_execution_profile.local_memory_kind));

  // Phase P1D.1 offline thread-schedule policy. These attrs are policy
  // provenance and selection inputs, not hardware capability and not raw
  // measured evidence.
  if (offline_thread_schedule_policy.active) {
    const auto &p = offline_thread_schedule_policy;
    auto setS = [&](llvm::StringRef key, const std::string &value) {
      if (!value.empty())
        op->setAttr(("target.thread_schedule_policy." + key).str(),
                    mlir::StringAttr::get(ctx, value));
    };
    auto setI = [&](llvm::StringRef key, int64_t value) {
      op->setAttr(("target.thread_schedule_policy." + key).str(),
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                         value));
    };
    auto setSchedule = [&](llvm::StringRef prefix,
                           const RuntimeThreadScheduleOption &ts) {
      setI((prefix + ".thread_count").str(), ts.thread_count);
      setS((prefix + ".partition_axis").str(), ts.partition_axis);
      setS((prefix + ".partition_strategy").str(), ts.partition_strategy);
    };
    setS("policy_id", p.policy_id);
    setS("policy_version", p.policy_version);
    setS("target_profile_id", p.target_profile_id);
    setS("fused_region_identity", p.fused_region_identity);
    setS("dtype", p.dtype);
    setS("kernel_id", p.kernel_id);
    setS("metric", p.metric);
    setI("threshold", p.threshold);
    setS("boundary_rule", p.boundary_rule);
    setSchedule("below_threshold", p.below_threshold_schedule);
    setSchedule("at_or_above_threshold", p.at_or_above_threshold_schedule);
    setS("calibration_evidence_ref", p.calibration_evidence_ref);
    setS("evidence_sha256", p.evidence_sha256);
    setS("calibration_workload_scope", p.calibration_workload_scope);
    setS("generated_at", p.generated_at);
    setS("calibration_compiler_commit", p.calibration_compiler_commit);
    setS("calibration_runtime_commit", p.calibration_runtime_commit);
    setS("truth_boundary", p.truth_boundary);
    setS("artifact_ref", p.artifact_ref);
  }

  // Emit per-backend capability attrs.
  // Index: target.backend_capability_names lists the backend names.
  // Per-backend fields use flat prefix target.backend_capabilities.{name}.*
  //
  // Optional cost and alignment fields are absent (not emitted) when nullopt.
  // Absent means unknown. 0.0 means free. 0 alignment means no requirement.
  // These are distinct values and must not be conflated.
  //
  // Source citations embedded in profiles:
  //   Apple Core ML — developer.apple.com/documentation/coreml (compute units, dtypes)
  //   NVIDIA Tensor Cores — H100/A100 Architecture Whitepapers, cuBLAS docs (precision families)
  //   CUDA memory model — CUDA Programming Guide (shared memory, global memory hierarchy)
  //   Arm Compute Library — github.com/ARM-software/ComputeLibrary (dtypes, NHWC preference)
  //   Intel AMX — Intel Architecture ISA Extensions Reference (TMUL, INT8, BF16)
  //   IREE — openxla.org/iree (target backends, device/architecture/performance profiles)
  if (!backend_capabilities.empty()) {
    mlir::Type i64 = mlir::IntegerType::get(ctx, 64);

    llvm::SmallVector<mlir::Attribute> nameAttrs;
    for (const auto &cap : backend_capabilities)
      nameAttrs.push_back(mlir::StringAttr::get(ctx, cap.backend_name));
    op->setAttr("target.backend_capability_names",
                mlir::ArrayAttr::get(ctx, nameAttrs));

    for (const auto &cap : backend_capabilities) {
      const std::string &n = cap.backend_name;

      auto setStr = [&](llvm::StringRef field, llvm::StringRef val) {
        op->setAttr(bcAttr(n, field), mlir::StringAttr::get(ctx, val));
      };
      auto setArr = [&](llvm::StringRef field,
                        const std::vector<std::string> &vals) {
        llvm::SmallVector<mlir::Attribute> elems;
        for (const auto &v : vals)
          elems.push_back(mlir::StringAttr::get(ctx, v));
        op->setAttr(bcAttr(n, field), mlir::ArrayAttr::get(ctx, elems));
      };
      auto setBool = [&](llvm::StringRef field, bool val) {
        op->setAttr(bcAttr(n, field), mlir::BoolAttr::get(ctx, val));
      };
      auto setOptInt = [&](llvm::StringRef field,
                           const std::optional<int64_t> &val) {
        if (val.has_value())
          op->setAttr(bcAttr(n, field), mlir::IntegerAttr::get(i64, *val));
        // nullopt → do not emit attr.
      };
      auto setOptCost = [&](llvm::StringRef field,
                            const std::optional<double> &val) {
        if (val.has_value())
          op->setAttr(bcAttr(n, field), mlir::FloatAttr::get(f64, *val));
        // nullopt → do not emit attr.  Absent ≠ 0.0 (free).
      };

      setStr("backend_api",                  cap.backend_api);
      setArr("supported_ops",                cap.supported_ops);
      setArr("supported_dtypes",             cap.supported_dtypes);
      setArr("accumulation_dtypes",          cap.accumulation_dtypes);
      setArr("supported_quant_modes",        cap.supported_quant_modes);
      setArr("preferred_activation_layouts", cap.preferred_activation_layouts);
      setArr("preferred_weight_layouts",     cap.preferred_weight_layouts);
      setArr("layout_agnostic_ops",          cap.layout_agnostic_ops);
      setBool("supports_layout_transform",   cap.supports_layout_transform);
      setBool("supports_cast",               cap.supports_cast);
      setBool("supports_dequant_boundary",   cap.supports_dequant_boundary);
      setBool("supports_requantize",         cap.supports_requantize);
      setArr("supports_fusion_patterns",     cap.supports_fusion_patterns);

      // Level 2: Hardware Constraints
      setOptInt("required_k_alignment",      cap.required_k_alignment);
      setOptInt("required_n_alignment",      cap.required_n_alignment);
      setOptInt("required_m_alignment",      cap.required_m_alignment);
      if (!cap.allowed_quant_granularity.empty())
        setArr("allowed_quant_granularity",  cap.allowed_quant_granularity);
      if (!cap.required_activation_quant_mode.empty())
        setStr("required_activation_quant_mode", cap.required_activation_quant_mode);
      if (!cap.required_weight_quant_mode.empty())
        setStr("required_weight_quant_mode", cap.required_weight_quant_mode);
      setOptInt("max_supported_rank",        cap.max_supported_rank);
      if (cap.requires_static_shape.has_value())
        setBool("requires_static_shape",     *cap.requires_static_shape);
      if (cap.requires_constant_weight.has_value())
        setBool("requires_constant_weight",  *cap.requires_constant_weight);
      if (!cap.unsupported_ops.empty())
        setArr("unsupported_ops",            cap.unsupported_ops);
      if (!cap.fallback_backend.empty())
        setStr("fallback_backend",           cap.fallback_backend);

      // Level 3: Hardware Preferences
      if (!cap.acceptable_activation_layouts.empty())
        setArr("acceptable_activation_layouts", cap.acceptable_activation_layouts);
      if (!cap.acceptable_weight_layouts.empty())
        setArr("acceptable_weight_layouts",  cap.acceptable_weight_layouts);
      if (!cap.preferred_dtypes.empty())
        setArr("preferred_dtypes",           cap.preferred_dtypes);
      if (!cap.acceptable_dtypes.empty())
        setArr("acceptable_dtypes",          cap.acceptable_dtypes);
      if (!cap.preferred_fusion_patterns.empty())
        setArr("preferred_fusion_patterns",  cap.preferred_fusion_patterns);

      // Level 4: Runtime Model (planning-only)
      if (!cap.runtime_model_launch_overhead.empty())
        setStr("runtime_model_launch_overhead",   cap.runtime_model_launch_overhead);
      if (!cap.runtime_model_dma_transfer.empty())
        setStr("runtime_model_dma_transfer",      cap.runtime_model_dma_transfer);
      if (!cap.runtime_model_memory_hierarchy.empty())
        setStr("runtime_model_memory_hierarchy",  cap.runtime_model_memory_hierarchy);
      if (!cap.runtime_model_occupancy.empty())
        setStr("runtime_model_occupancy",         cap.runtime_model_occupancy);
      if (!cap.runtime_model_register_pressure.empty())
        setStr("runtime_model_register_pressure", cap.runtime_model_register_pressure);
      if (!cap.runtime_model_shared_memory.empty())
        setStr("runtime_model_shared_memory",     cap.runtime_model_shared_memory);
      if (!cap.runtime_model_source_level.empty())
        setStr("runtime_model_source_level",      cap.runtime_model_source_level);
      if (!cap.runtime_model_truth_boundary.empty())
        setStr("runtime_model_truth_boundary",    cap.runtime_model_truth_boundary);

      // Level 5: Cost Model Placeholder (planning-only)
      if (!cap.cost_model_kind.empty())
        setStr("cost_model_kind",           cap.cost_model_kind);
      if (!cap.cost_model_inputs.empty())
        setStr("cost_model_inputs",         cap.cost_model_inputs);
      if (!cap.cost_model_source_level.empty())
        setStr("cost_model_source_level",   cap.cost_model_source_level);
      if (!cap.cost_model_truth_boundary.empty())
        setStr("cost_model_truth_boundary", cap.cost_model_truth_boundary);

      setStr("source_level",                 cap.source_level);
      setStr("truth_boundary",               cap.truth_boundary);

      // Legacy cost params
      setOptCost("layout_transform_cost_ms", cap.layout_transform_cost_ms);
      setOptCost("cast_cost_ms",             cap.cast_cost_ms);
      setOptCost("quantize_cost_ms",         cap.quantize_cost_ms);
      setOptCost("dequantize_cost_ms",       cap.dequantize_cost_ms);
      setOptCost("requantize_cost_ms",       cap.requantize_cost_ms);
      setOptCost("backend_transfer_cost_ms", cap.backend_transfer_cost_ms);
    }
  }

  // Emit kernel library capabilities.
  // Groups entries by backend; emits:
  //   target.kernel_library_backends = [name, ...]   (index)
  //   target.kernel_libraries.{backend} = [{...}, ...] (ArrayAttr of DictionaryAttr)
  if (!kernel_library_capabilities.empty()) {
    mlir::Type i64 = mlir::IntegerType::get(ctx, 64);

    // Collect unique backends in declaration order.
    llvm::SmallVector<std::string> backends;
    {
      llvm::SmallSet<std::string, 8> seen;
      for (const auto &ke : kernel_library_capabilities) {
        if (seen.insert(ke.backend).second)
          backends.push_back(ke.backend);
      }
    }

    // Index attr.
    {
      llvm::SmallVector<mlir::Attribute> nameAttrs;
      for (const auto &b : backends)
        nameAttrs.push_back(mlir::StringAttr::get(ctx, b));
      op->setAttr("target.kernel_library_backends",
                  mlir::ArrayAttr::get(ctx, nameAttrs));
    }

    // Per-backend ArrayAttr of DictionaryAttr.
    for (const auto &backend : backends) {
      llvm::SmallVector<mlir::Attribute> entries;
      for (const auto &ke : kernel_library_capabilities) {
        if (ke.backend != backend) continue;

        // Build DictionaryAttr in sorted key order (MLIR requirement).
        auto S = [&](llvm::StringRef s) -> mlir::Attribute {
          return mlir::StringAttr::get(ctx, s);
        };
        auto A = [&](const std::vector<std::string> &v) -> mlir::Attribute {
          llvm::SmallVector<mlir::Attribute> elems;
          for (const auto &s : v) elems.push_back(mlir::StringAttr::get(ctx, s));
          return mlir::ArrayAttr::get(ctx, elems);
        };
        auto B = [&](bool v) -> mlir::Attribute { return mlir::BoolAttr::get(ctx, v); };

        llvm::SmallVector<mlir::NamedAttribute> fields;
        // Keys must be emitted in sorted order for DictionaryAttr validity.
        auto addS = [&](llvm::StringRef k, llvm::StringRef v) {
          fields.emplace_back(mlir::StringAttr::get(ctx, k), S(v));
        };
        auto addA = [&](llvm::StringRef k, const std::vector<std::string> &v) {
          fields.emplace_back(mlir::StringAttr::get(ctx, k), A(v));
        };
        auto addB = [&](llvm::StringRef k, bool v) {
          fields.emplace_back(mlir::StringAttr::get(ctx, k), B(v));
        };
        auto addOptI = [&](llvm::StringRef k, const std::optional<int64_t> &v) {
          if (v.has_value())
            fields.emplace_back(mlir::StringAttr::get(ctx, k),
                                mlir::IntegerAttr::get(i64, *v));
        };

        // Sorted alphabetically:
        addS("fallback_backend",        ke.fallback_backend);
        addS("fallback_kernel",         ke.fallback_kernel);
        addA("fusion_patterns",         ke.fusion_patterns);
        addS("kernel_library",          ke.kernel_library);
        addS("kernel_name",             ke.kernel_name);
        addS("op_type",                 ke.op_type);
        addOptI("required_k_alignment", ke.required_k_alignment);
        addOptI("required_m_alignment", ke.required_m_alignment);
        addOptI("required_n_alignment", ke.required_n_alignment);
        addB("requires_constant_weight", ke.requires_constant_weight);
        addA("rewrite_patterns",        ke.rewrite_patterns);
        addS("source_level",            ke.source_level);
        addA("supported_dtypes",        ke.supported_dtypes);
        addA("supported_layouts",       ke.supported_layouts);
        addA("supported_quant_modes",   ke.supported_quant_modes);
        addB("supports_dynamic_shape",  ke.supports_dynamic_shape);
        addB("supports_fusion",         ke.supports_fusion);
        addS("truth_boundary",          ke.truth_boundary);

        entries.push_back(mlir::DictionaryAttr::get(ctx, fields));
      }
      op->setAttr("target.kernel_libraries." + backend,
                  mlir::ArrayAttr::get(ctx, entries));
    }
  }

  // Concrete runtime kernel descriptors (kernel_selection_contract_v1):
  //   target.runtime_kernels = [{...}, ...] (ArrayAttr of DictionaryAttr)
  // A flat list — each descriptor names its backend. Absent attr = no
  // runtime kernels declared; KernelSelectionPass records the deferral.
  if (!runtime_kernels.empty()) {
    mlir::Type i64 = mlir::IntegerType::get(ctx, 64);
    llvm::SmallVector<mlir::Attribute> entries;
    for (const auto &rk : runtime_kernels) {
      auto S = [&](llvm::StringRef s) -> mlir::Attribute {
        return mlir::StringAttr::get(ctx, s);
      };
      auto A = [&](const std::vector<std::string> &v) -> mlir::Attribute {
        llvm::SmallVector<mlir::Attribute> elems;
        for (const auto &s : v) elems.push_back(mlir::StringAttr::get(ctx, s));
        return mlir::ArrayAttr::get(ctx, elems);
      };
      llvm::SmallVector<mlir::NamedAttribute> fields;
      auto add = [&](llvm::StringRef k, mlir::Attribute v) {
        fields.emplace_back(mlir::StringAttr::get(ctx, k), v);
      };
      // Thread-decomposition options (Phase P1D, thread_schedule_contract_v1):
      // an ArrayAttr of small DictionaryAttrs, same nesting style as the
      // outer runtime_kernels list itself. Empty when the kernel declares
      // no thread schedules (pre-P1D profiles) — absent capacity, never
      // invented.
      llvm::SmallVector<mlir::Attribute> threadScheduleEntries;
      for (const auto &ts : rk.supported_thread_schedules) {
        llvm::SmallVector<mlir::NamedAttribute> tsFields;
        tsFields.emplace_back(mlir::StringAttr::get(ctx, "thread_count"),
                               mlir::IntegerAttr::get(i64, ts.thread_count));
        tsFields.emplace_back(mlir::StringAttr::get(ctx, "partition_axis"),
                               S(ts.partition_axis));
        tsFields.emplace_back(mlir::StringAttr::get(ctx, "partition_strategy"),
                               S(ts.partition_strategy));
        threadScheduleEntries.push_back(mlir::DictionaryAttr::get(ctx, tsFields));
      }
      // Sorted alphabetically (DictionaryAttr requirement):
      add("backend",                     S(rk.backend));
      add("kernel_id",                   S(rk.kernel_id));
      add("op_name",                     S(rk.op_name));
      add("requires_local_memory_bytes",
          mlir::IntegerAttr::get(i64, rk.requires_local_memory_bytes));
      add("requires_static_shape",
          mlir::BoolAttr::get(ctx, rk.requires_static_shape));
      add("source",                      S(rk.source));
      add("supported_dtypes",            A(rk.supported_dtypes));
      add("supported_layouts",           A(rk.supported_layouts));
      add("supported_quant_modes",       A(rk.supported_quant_modes));
      add("supported_thread_schedules",
          mlir::ArrayAttr::get(ctx, threadScheduleEntries));
      add("supported_tile_shapes",       A(rk.supported_tile_shapes));
      add("truth_boundary",              S(rk.truth_boundary));
      entries.push_back(mlir::DictionaryAttr::get(ctx, fields));
    }
    op->setAttr("target.runtime_kernels", mlir::ArrayAttr::get(ctx, entries));
  }
}

} // namespace mlir::hir
