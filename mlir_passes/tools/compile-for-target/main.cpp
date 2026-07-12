// compile-for-target: compiler driver for serving artifact generation.
//
// Pipeline:
//   TargetDeviceProfile JSON
//     → TargetDeviceProfile (parsed here, tool boundary)
//     → TargetProfileLowering (tool boundary)
//     → TargetConstraints + CapabilityBundle
//     → TargetConstraints::attachToModule()
//     → serving passes (15-pass pipeline)
//     → ExecutionPlanBuilder
//     → ExecutionPlan
//     → ExecutionPlanExporter
//         → canonical artifact  (execution_plan.json)
//
// JSON construction is fully owned by ExecutionPlanExporter.
// This file never touches llvm::json types directly.

#include "FusionPasses.h"
#include "serving/ExecutionPlan.h"
#include "serving/ExecutionPlanBuilder.h"
#include "serving/ExecutionPlanExporter.h"
#include "serving/TargetConstraints.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// CLI flags
// ---------------------------------------------------------------------------

static llvm::cl::opt<std::string> DeviceProfilePath(
    "device-profile",
    llvm::cl::desc("Path to TargetDeviceProfile JSON (e.g. configs/target_profiles/apple_a17pro_mobile.json)"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> MlirPath(
    "mlir",
    llvm::cl::desc("Path to input MLIR file"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> OutPath(
    "out",
    llvm::cl::desc("Output path for canonical serving_execution_plan artifact"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> DumpAnnotatedMlir(
    "dump-annotated-mlir",
    llvm::cl::desc("(Optional) write pass-annotated MLIR to this path"),
    llvm::cl::init(""));

static llvm::cl::opt<std::string> DispatchUnitReportPath(
    "dispatch-unit-report",
    llvm::cl::desc("(Optional) write Phase 26 dispatch-unit reconciliation "
                   "report JSON to this path"),
    llvm::cl::init(""));

// ---------------------------------------------------------------------------
// TargetDeviceProfile — tool-boundary struct
// Only the fields consumed by TargetProfileLowering are populated.
// chipName, totalRAMBytes, thermalState, lowPowerMode, modelIdentifier,
// and CPU count are parsed for provenance only and not forwarded to the compiler.
// ---------------------------------------------------------------------------

// Per-kernel library entry as declared in the profile JSON.
// Maps 1:1 to KernelLibraryCapability in TargetConstraints.h.
struct KernelLibraryProfile {
  std::string op_type;
  std::string kernel_name;
  std::string backend;
  std::string kernel_library;
  std::vector<std::string> supported_dtypes;
  std::vector<std::string> supported_layouts;
  std::vector<std::string> supported_quant_modes;
  std::optional<int64_t>   required_m_alignment;
  std::optional<int64_t>   required_n_alignment;
  std::optional<int64_t>   required_k_alignment;
  bool supports_dynamic_shape    = true;
  bool requires_constant_weight  = false;
  bool supports_fusion           = false;
  std::vector<std::string> fusion_patterns;
  std::vector<std::string> rewrite_patterns;
  std::string fallback_kernel;
  std::string fallback_backend;
  std::string source_level;
  std::string truth_boundary;
};

// Per-backend capability as declared in the profile JSON.
// Maps 1:1 to BackendCapability in TargetConstraints.h.
// camelCase field names match the JSON schema; snake_case is used in TargetConstraints.
struct BackendCapabilityProfile {
  std::string backend_name;
  std::string backend_api;
  std::vector<std::string> supported_ops;
  std::vector<std::string> supported_dtypes;
  std::vector<std::string> accumulation_dtypes;
  std::vector<std::string> supported_quant_modes;
  std::vector<std::string> preferred_activation_layouts;
  std::vector<std::string> preferred_weight_layouts;
  std::vector<std::string> layout_agnostic_ops;
  bool supports_layout_transform = false;
  bool supports_cast             = false;
  bool supports_dequant_boundary = false;
  bool supports_requantize       = false;
  std::vector<std::string> supports_fusion_patterns;

  // Level 2: Hardware Constraints
  std::optional<int64_t> required_k_alignment;
  std::optional<int64_t> required_n_alignment;
  std::optional<int64_t> required_m_alignment;
  std::vector<std::string> allowed_quant_granularity;
  std::string required_activation_quant_mode;
  std::string required_weight_quant_mode;
  std::optional<int64_t>  max_supported_rank;
  std::optional<bool>     requires_static_shape;
  std::optional<bool>     requires_constant_weight;
  std::vector<std::string> unsupported_ops;
  std::string fallback_backend;

  // Level 3: Hardware Preferences
  std::vector<std::string> acceptable_activation_layouts;
  std::vector<std::string> acceptable_weight_layouts;
  std::vector<std::string> preferred_dtypes;
  std::vector<std::string> acceptable_dtypes;
  std::vector<std::string> preferred_fusion_patterns;

  // Level 4: Runtime Model (planning-only)
  std::string runtime_model_launch_overhead;
  std::string runtime_model_dma_transfer;
  std::string runtime_model_memory_hierarchy;
  std::string runtime_model_occupancy;
  std::string runtime_model_register_pressure;
  std::string runtime_model_shared_memory;
  std::string runtime_model_source_level;
  std::string runtime_model_truth_boundary;

  // Level 5: Cost Model Placeholder (planning-only)
  std::string cost_model_kind;
  std::string cost_model_inputs;
  std::string cost_model_source_level;
  std::string cost_model_truth_boundary;

  // Legacy cost params: std::nullopt = unknown. 0.0 = free. Distinct.
  std::optional<double> layout_transform_cost_ms;
  std::optional<double> cast_cost_ms;
  std::optional<double> quantize_cost_ms;
  std::optional<double> dequantize_cost_ms;
  std::optional<double> requantize_cost_ms;
  std::optional<double> backend_transfer_cost_ms;
  std::string source_level;
  std::string truth_boundary;
};

struct TargetDeviceProfile {
  std::string profileId;
  std::string profileKind;
  double      metalMaxWorkingSetMB = 0.0;
  std::string configuredComputeUnits;
  bool        staticShapeSupport = true;
  std::vector<std::string> supportedPrecisions;
  std::vector<std::string> pagedKVCompatibleBackends;
  // Formula-calibration cost estimates forwarded to TargetConstraints.
  double prefillMsPerToken    = 0.0;
  double decodeMsPerToken     = 0.0;
  double pdBandwidthMbPerMs   = 0.0;
  bool   hasPrefillMsPerToken  = false;
  bool   hasDecodeMsPerToken   = false;
  bool   hasPdBandwidthMbPerMs = false;
  std::string truthBoundary;
  // Per-backend hardware capability declarations.
  std::vector<BackendCapabilityProfile> backendCapabilities;
  // Kernel library capability declarations (Layer 3: actual kernel availability).
  std::vector<KernelLibraryProfile> kernelLibraries;
  // Concrete runtime kernel descriptors (kernel_selection_contract_v1).
  // Small and honest by design: only kernels with a known dispatchable
  // implementation are declared here.
  std::vector<mlir::hir::RuntimeKernelDescriptor> runtimeKernels;
  // Optional quantization co-design policy; "" = co-design pass inert.
  std::string quantizationCoDesignPolicy;

  // Optional static cost profile for shape_cost_model_v2: declared
  // theoretical peak numbers (public docs or declared profile — never
  // measured). 0 = not declared; the cost model then emits shape facts
  // without time estimates.
  double      staticCostPeakFlopsFp32 = 0.0;
  double      staticCostPeakFlopsFp16 = 0.0;
  double      staticCostPeakFlopsInt8 = 0.0;
  double      staticCostMemoryBandwidthBytesPerSec = 0.0;
  // Memory hierarchy (declared capacities/capabilities, never measured).
  int64_t     staticCostLocalMemoryBytes = 0;
  int64_t     staticCostCacheLineBytes   = 0;
  std::optional<bool> staticCostSupportsAsyncCopy;
  std::optional<bool> staticCostSupportsDma;
  std::string staticCostTruthBoundary;
  std::optional<int64_t> hardwarePhysicalComputeUnits;
  std::optional<int64_t> hardwareEffectiveComputeUnits;
  std::optional<int64_t> hardwareMaxConcurrentWorkItemsPerUnit;
  std::optional<bool> hardwareSupportsLatencyHiding;
  std::optional<std::string> hardwareLocalMemoryKind;

  // Optional experimental global quantization override (Phase C minimal AWQ
  // support). When absent, quantization planning is unchanged: no
  // quantization.* module attrs are set by this driver, exactly as before
  // this field existed. When present, it forces a GLOBAL quantization
  // decision (global_decisions.quantization in the exported plan) that is
  // independent of per-op backendCapabilities.supportedQuantModes above --
  // it does not claim the declared backend/kernel capabilities changed.
  bool        hasForcedQuantization = false;
  std::string forcedQuantStrategy;         // e.g. "weight_only_int4"
  std::string forcedQuantAlgorithm;        // e.g. "awq"
  std::string forcedQuantArtifactRef;      // e.g. "artifacts/qwen_awq"
  std::string forcedQuantTruthBoundary;
};

// ---------------------------------------------------------------------------
// TargetProfileLowering — tool-boundary mapping
// ---------------------------------------------------------------------------

static mlir::hir::TargetConstraints
lowerToTargetConstraints(const TargetDeviceProfile &prof) {
  mlir::hir::TargetConstraints tc;

  tc.profile_id = prof.profileId;
  tc.profile_kind = prof.profileKind;

  if (prof.metalMaxWorkingSetMB > 0.0) {
    tc.memory_budget_mb  = prof.metalMaxWorkingSetMB;
    tc.has_memory_budget = true;
  }

  tc.static_shape_support     = prof.staticShapeSupport;
  tc.has_static_shape_support = true;

  const auto &cu = prof.configuredComputeUnits;
  if (cu == "CPU+GPU+ANE") {
    tc.preferred_backend = "coreml";
    tc.allowed_backends  = {"coreml", "metal", "cpu"};
  } else if (cu == "CPU+GPU") {
    tc.preferred_backend = "metal";
    tc.allowed_backends  = {"metal", "cpu"};
  } else if (cu == "CPU") {
    tc.preferred_backend = "cpu";
    tc.allowed_backends  = {"cpu"};
  } else if (cu == "CUDA") {
    tc.preferred_backend = "cuda";
    tc.allowed_backends  = {"cuda", "cpu"};
  } else if (cu == "CPU+GPU-ARM") {
    tc.preferred_backend = "arm_compute";
    tc.allowed_backends  = {"arm_compute", "cpu"};
  } else if (cu == "AMX") {
    tc.preferred_backend = "amx";
    tc.allowed_backends  = {"amx", "cpu"};
  } else if (cu == "CUSTOM-NPU") {
    tc.preferred_backend = "iree_hal";
    tc.allowed_backends  = {"iree_hal", "cpu"};
  }
  // Unknown/absent configuredComputeUnits: leave unconstrained.

  if (!prof.supportedPrecisions.empty()) {
    tc.supported_precisions = prof.supportedPrecisions;
  } else {
    // Default: fp16 for GPU/ANE targets, fp32+fp16 for CPU-only.
    if (cu == "CPU") {
      tc.supported_precisions = {"fp32", "fp16"};
    } else {
      tc.supported_precisions = {"fp16"};
    }
  }

  // Paged KV compatibility is a target property, not a compiler invariant.
  // Absent field in profile → empty list → no backend supports paged KV on
  // this target → constraint_conflict for any prefill (producer) function.
  tc.paged_kv_compatible_backends = prof.pagedKVCompatibleBackends;

  // Serving cost constants: forward from profile when present so
  // ServingPhaseAnalysisPass uses hardware-calibrated estimates instead of
  // built-in formula defaults.
  if (prof.hasPrefillMsPerToken) {
    tc.prefill_ms_per_token    = prof.prefillMsPerToken;
    tc.has_prefill_ms_per_token = true;
  }
  if (prof.hasDecodeMsPerToken) {
    tc.decode_ms_per_token    = prof.decodeMsPerToken;
    tc.has_decode_ms_per_token = true;
  }
  if (prof.hasPdBandwidthMbPerMs) {
    tc.pd_bandwidth_mb_per_ms    = prof.pdBandwidthMbPerMs;
    tc.has_pd_bandwidth_mb_per_ms = true;
  }

  // Static cost profile numbers for shape_cost_model_v2 (declared peaks).
  tc.static_cost_peak_flops_fp32 = prof.staticCostPeakFlopsFp32;
  tc.static_cost_peak_flops_fp16 = prof.staticCostPeakFlopsFp16;
  tc.static_cost_peak_flops_int8 = prof.staticCostPeakFlopsInt8;
  tc.static_cost_memory_bandwidth_bytes_per_sec =
      prof.staticCostMemoryBandwidthBytesPerSec;
  tc.static_cost_local_memory_bytes = prof.staticCostLocalMemoryBytes;
  tc.static_cost_cache_line_bytes   = prof.staticCostCacheLineBytes;
  if (prof.staticCostSupportsAsyncCopy) {
    tc.static_cost_supports_async_copy = *prof.staticCostSupportsAsyncCopy;
    tc.has_static_cost_supports_async_copy = true;
  }
  if (prof.staticCostSupportsDma) {
    tc.static_cost_supports_dma = *prof.staticCostSupportsDma;
    tc.has_static_cost_supports_dma = true;
  }
  tc.static_cost_profile_truth_boundary = prof.staticCostTruthBoundary;

  tc.hardware_execution_profile.physical_compute_units =
      prof.hardwarePhysicalComputeUnits;
  tc.hardware_execution_profile.effective_compute_units =
      prof.hardwareEffectiveComputeUnits;
  tc.hardware_execution_profile.max_concurrent_work_items_per_unit =
      prof.hardwareMaxConcurrentWorkItemsPerUnit;
  tc.hardware_execution_profile.supports_latency_hiding =
      prof.hardwareSupportsLatencyHiding;
  tc.hardware_execution_profile.local_memory_kind =
      prof.hardwareLocalMemoryKind;

  // Lower per-backend capability declarations.
  // Cost fields and alignment fields preserve nullopt → absent in MLIR attrs.
  for (const auto &bp : prof.backendCapabilities) {
    mlir::hir::BackendCapability cap;
    cap.backend_name                   = bp.backend_name;
    cap.backend_api                    = bp.backend_api;
    cap.supported_ops                  = bp.supported_ops;
    cap.supported_dtypes               = bp.supported_dtypes;
    cap.accumulation_dtypes            = bp.accumulation_dtypes;
    cap.supported_quant_modes          = bp.supported_quant_modes;
    cap.preferred_activation_layouts   = bp.preferred_activation_layouts;
    cap.preferred_weight_layouts       = bp.preferred_weight_layouts;
    cap.layout_agnostic_ops            = bp.layout_agnostic_ops;
    cap.supports_layout_transform      = bp.supports_layout_transform;
    cap.supports_cast                  = bp.supports_cast;
    cap.supports_dequant_boundary      = bp.supports_dequant_boundary;
    cap.supports_requantize            = bp.supports_requantize;
    cap.supports_fusion_patterns       = bp.supports_fusion_patterns;

    // Level 2
    cap.required_k_alignment           = bp.required_k_alignment;
    cap.required_n_alignment           = bp.required_n_alignment;
    cap.required_m_alignment           = bp.required_m_alignment;
    cap.allowed_quant_granularity      = bp.allowed_quant_granularity;
    cap.required_activation_quant_mode = bp.required_activation_quant_mode;
    cap.required_weight_quant_mode     = bp.required_weight_quant_mode;
    cap.max_supported_rank             = bp.max_supported_rank;
    cap.requires_static_shape          = bp.requires_static_shape;
    cap.requires_constant_weight       = bp.requires_constant_weight;
    cap.unsupported_ops                = bp.unsupported_ops;
    cap.fallback_backend               = bp.fallback_backend;

    // Level 3
    cap.acceptable_activation_layouts  = bp.acceptable_activation_layouts;
    cap.acceptable_weight_layouts      = bp.acceptable_weight_layouts;
    cap.preferred_dtypes               = bp.preferred_dtypes;
    cap.acceptable_dtypes              = bp.acceptable_dtypes;
    cap.preferred_fusion_patterns      = bp.preferred_fusion_patterns;

    // Level 4
    cap.runtime_model_launch_overhead   = bp.runtime_model_launch_overhead;
    cap.runtime_model_dma_transfer      = bp.runtime_model_dma_transfer;
    cap.runtime_model_memory_hierarchy  = bp.runtime_model_memory_hierarchy;
    cap.runtime_model_occupancy         = bp.runtime_model_occupancy;
    cap.runtime_model_register_pressure = bp.runtime_model_register_pressure;
    cap.runtime_model_shared_memory     = bp.runtime_model_shared_memory;
    cap.runtime_model_source_level      = bp.runtime_model_source_level;
    cap.runtime_model_truth_boundary    = bp.runtime_model_truth_boundary;

    // Level 5
    cap.cost_model_kind            = bp.cost_model_kind;
    cap.cost_model_inputs          = bp.cost_model_inputs;
    cap.cost_model_source_level    = bp.cost_model_source_level;
    cap.cost_model_truth_boundary  = bp.cost_model_truth_boundary;

    // Legacy cost params
    cap.layout_transform_cost_ms       = bp.layout_transform_cost_ms;
    cap.cast_cost_ms                   = bp.cast_cost_ms;
    cap.quantize_cost_ms               = bp.quantize_cost_ms;
    cap.dequantize_cost_ms             = bp.dequantize_cost_ms;
    cap.requantize_cost_ms             = bp.requantize_cost_ms;
    cap.backend_transfer_cost_ms       = bp.backend_transfer_cost_ms;
    cap.source_level                   = bp.source_level;
    cap.truth_boundary                 = bp.truth_boundary;
    tc.backend_capabilities.push_back(std::move(cap));
  }

  // Lower kernel library declarations.
  for (const auto &kp : prof.kernelLibraries) {
    mlir::hir::KernelLibraryCapability ke;
    ke.op_type                  = kp.op_type;
    ke.kernel_name              = kp.kernel_name;
    ke.backend                  = kp.backend;
    ke.kernel_library           = kp.kernel_library;
    ke.supported_dtypes         = kp.supported_dtypes;
    ke.supported_layouts        = kp.supported_layouts;
    ke.supported_quant_modes    = kp.supported_quant_modes;
    ke.required_m_alignment     = kp.required_m_alignment;
    ke.required_n_alignment     = kp.required_n_alignment;
    ke.required_k_alignment     = kp.required_k_alignment;
    ke.supports_dynamic_shape   = kp.supports_dynamic_shape;
    ke.requires_constant_weight = kp.requires_constant_weight;
    ke.supports_fusion          = kp.supports_fusion;
    ke.fusion_patterns          = kp.fusion_patterns;
    ke.rewrite_patterns         = kp.rewrite_patterns;
    ke.fallback_kernel          = kp.fallback_kernel;
    ke.fallback_backend         = kp.fallback_backend;
    ke.source_level             = kp.source_level;
    ke.truth_boundary           = kp.truth_boundary;
    tc.kernel_library_capabilities.push_back(std::move(ke));
  }

  // Runtime kernel descriptors pass through unchanged (already the
  // compiler-side type).
  tc.runtime_kernels = prof.runtimeKernels;

  return tc;
}

// ---------------------------------------------------------------------------
// parseDeviceProfile — reads and validates the JSON device profile
// ---------------------------------------------------------------------------

static llvm::Expected<TargetDeviceProfile>
parseDeviceProfile(llvm::StringRef path) {
  auto buf = llvm::MemoryBuffer::getFile(path);
  if (!buf)
    return llvm::make_error<llvm::StringError>(
        "cannot read device profile '" + path.str() + "': " +
            buf.getError().message(),
        llvm::inconvertibleErrorCode());

  auto json = llvm::json::parse((*buf)->getBuffer());
  if (!json)
    return llvm::make_error<llvm::StringError>(
        "JSON parse error in device profile '" + path.str() + "': " +
            llvm::toString(json.takeError()),
        llvm::inconvertibleErrorCode());

  const llvm::json::Object *obj = json->getAsObject();
  if (!obj)
    return llvm::make_error<llvm::StringError>(
        "device profile must be a JSON object",
        llvm::inconvertibleErrorCode());

  TargetDeviceProfile prof;

  if (auto v = obj->getString("profileId"))
    prof.profileId = v->str();
  else
    return llvm::make_error<llvm::StringError>(
        "device profile missing required field 'profileId'",
        llvm::inconvertibleErrorCode());

  if (auto v = obj->getString("profileKind"))
    prof.profileKind = v->str();

  if (auto v = obj->getNumber("metalMaxWorkingSetMB"))
    prof.metalMaxWorkingSetMB = *v;

  if (auto v = obj->getString("configuredComputeUnits"))
    prof.configuredComputeUnits = v->str();

  if (auto v = obj->getBoolean("staticShapeSupport"))
    prof.staticShapeSupport = *v;

  if (auto *arr = obj->getArray("supportedPrecisions")) {
    for (const auto &elem : *arr) {
      if (auto s = elem.getAsString())
        prof.supportedPrecisions.push_back(s->str());
    }
  }

  if (auto *arr = obj->getArray("pagedKVCompatibleBackends")) {
    for (const auto &elem : *arr) {
      if (auto s = elem.getAsString())
        prof.pagedKVCompatibleBackends.push_back(s->str());
    }
  }

  if (auto v = obj->getNumber("prefillMsPerToken")) {
    prof.prefillMsPerToken    = *v;
    prof.hasPrefillMsPerToken = true;
  }
  if (auto v = obj->getNumber("decodeMsPerToken")) {
    prof.decodeMsPerToken    = *v;
    prof.hasDecodeMsPerToken = true;
  }
  if (auto v = obj->getNumber("pdBandwidthMbPerMs")) {
    prof.pdBandwidthMbPerMs    = *v;
    prof.hasPdBandwidthMbPerMs = true;
  }

  if (auto v = obj->getString("truthBoundary"))
    prof.truthBoundary = v->str();

  // Parse optional staticCostProfile block (shape_cost_model_v2 declared
  // theoretical peaks). Absent in most profiles; the cost model then emits
  // shape facts only, without time estimates.
  if (auto *scp = obj->getObject("staticCostProfile")) {
    if (auto v = scp->getNumber("peakFlopsFp32"))
      prof.staticCostPeakFlopsFp32 = *v;
    if (auto v = scp->getNumber("peakFlopsFp16"))
      prof.staticCostPeakFlopsFp16 = *v;
    if (auto v = scp->getNumber("peakFlopsInt8"))
      prof.staticCostPeakFlopsInt8 = *v;
    if (auto v = scp->getNumber("memoryBandwidthBytesPerSec"))
      prof.staticCostMemoryBandwidthBytesPerSec = *v;
    if (auto v = scp->getInteger("localMemoryBytes"))
      prof.staticCostLocalMemoryBytes = static_cast<int64_t>(*v);
    if (auto v = scp->getInteger("cacheLineBytes"))
      prof.staticCostCacheLineBytes = static_cast<int64_t>(*v);
    if (auto v = scp->getBoolean("supportsAsyncCopy"))
      prof.staticCostSupportsAsyncCopy = *v;
    if (auto v = scp->getBoolean("supportsDma"))
      prof.staticCostSupportsDma = *v;
    if (auto v = scp->getString("truthBoundary"))
      prof.staticCostTruthBoundary = v->str();
  }

  if (auto *hep = obj->getObject("hardwareExecutionProfile")) {
    if (auto v = hep->getInteger("physicalComputeUnits"))
      prof.hardwarePhysicalComputeUnits = static_cast<int64_t>(*v);
    if (auto v = hep->getInteger("effectiveComputeUnits"))
      prof.hardwareEffectiveComputeUnits = static_cast<int64_t>(*v);
    if (auto v = hep->getInteger("maxConcurrentWorkItemsPerUnit"))
      prof.hardwareMaxConcurrentWorkItemsPerUnit =
          static_cast<int64_t>(*v);
    if (auto v = hep->getBoolean("supportsLatencyHiding"))
      prof.hardwareSupportsLatencyHiding = *v;
    if (auto v = hep->getString("localMemoryKind"))
      prof.hardwareLocalMemoryKind = v->str();
  }

  // Optional quantization co-design policy (quantization_codesign_contract_v1).
  // Absent in every existing profile — the co-design pass is then inert and
  // existing artifacts stay byte-identical.
  if (auto v = obj->getString("quantizationCoDesignPolicy"))
    prof.quantizationCoDesignPolicy = v->str();

  // Parse optional forcedQuantization block (Phase C minimal AWQ support).
  // Absent in every existing profile; only present in profiles that opt in
  // to an experimental forced global quantization override.
  if (auto *fq = obj->getObject("forcedQuantization")) {
    prof.hasForcedQuantization = true;
    if (auto v = fq->getString("strategy"))
      prof.forcedQuantStrategy = v->str();
    if (auto v = fq->getString("algorithm"))
      prof.forcedQuantAlgorithm = v->str();
    if (auto v = fq->getString("quantizedModelArtifactRef"))
      prof.forcedQuantArtifactRef = v->str();
    if (auto v = fq->getString("truthBoundary"))
      prof.forcedQuantTruthBoundary = v->str();
  }

  // Parse backendCapabilities array.
  // Missing cost and alignment fields → std::nullopt (unknown, not zero).
  if (auto *caps = obj->getArray("backendCapabilities")) {
    auto readStrings = [](const llvm::json::Object *o,
                          llvm::StringRef key) -> std::vector<std::string> {
      std::vector<std::string> result;
      if (auto *arr = o->getArray(key))
        for (const auto &elem : *arr)
          if (auto s = elem.getAsString())
            result.push_back(s->str());
      return result;
    };

    for (const auto &capElem : *caps) {
      const llvm::json::Object *co = capElem.getAsObject();
      if (!co)
        continue;

      BackendCapabilityProfile bp;
      if (auto v = co->getString("backendName"))  bp.backend_name = v->str();
      if (auto v = co->getString("backendApi"))   bp.backend_api  = v->str();

      bp.supported_ops                = readStrings(co, "supportedOps");
      bp.supported_dtypes             = readStrings(co, "supportedDtypes");
      bp.accumulation_dtypes          = readStrings(co, "accumulationDtypes");
      bp.supported_quant_modes        = readStrings(co, "supportedQuantModes");
      bp.preferred_activation_layouts = readStrings(co, "preferredActivationLayouts");
      bp.preferred_weight_layouts     = readStrings(co, "preferredWeightLayouts");
      bp.layout_agnostic_ops          = readStrings(co, "layoutAgnosticOps");
      bp.supports_fusion_patterns     = readStrings(co, "supportsFusionPatterns");

      if (auto v = co->getBoolean("supportsLayoutTransform"))
        bp.supports_layout_transform = *v;
      if (auto v = co->getBoolean("supportsCast"))
        bp.supports_cast = *v;
      if (auto v = co->getBoolean("supportsDequantBoundary"))
        bp.supports_dequant_boundary = *v;
      if (auto v = co->getBoolean("supportsRequantize"))
        bp.supports_requantize = *v;

      // Level 2: Hardware Constraints
      if (auto v = co->getInteger("requiredKAlignment"))
        bp.required_k_alignment = static_cast<int64_t>(*v);
      if (auto v = co->getInteger("requiredNAlignment"))
        bp.required_n_alignment = static_cast<int64_t>(*v);
      if (auto v = co->getInteger("requiredMAlignment"))
        bp.required_m_alignment = static_cast<int64_t>(*v);
      bp.allowed_quant_granularity = readStrings(co, "allowedQuantGranularity");
      if (auto v = co->getString("requiredActivationQuantMode"))
        bp.required_activation_quant_mode = v->str();
      if (auto v = co->getString("requiredWeightQuantMode"))
        bp.required_weight_quant_mode = v->str();
      if (auto v = co->getInteger("maxSupportedRank"))
        bp.max_supported_rank = static_cast<int64_t>(*v);
      if (auto v = co->getBoolean("requiresStaticShape"))
        bp.requires_static_shape = *v;
      if (auto v = co->getBoolean("requiresConstantWeight"))
        bp.requires_constant_weight = *v;
      bp.unsupported_ops = readStrings(co, "unsupportedOps");
      if (auto v = co->getString("fallbackBackend"))
        bp.fallback_backend = v->str();

      // Level 3: Hardware Preferences
      bp.acceptable_activation_layouts = readStrings(co, "acceptableActivationLayouts");
      bp.acceptable_weight_layouts     = readStrings(co, "acceptableWeightLayouts");
      bp.preferred_dtypes              = readStrings(co, "preferredDtypes");
      bp.acceptable_dtypes             = readStrings(co, "acceptableDtypes");
      bp.preferred_fusion_patterns     = readStrings(co, "preferredFusionPatterns");

      // Level 4: Runtime Model (planning-only; stored as descriptive strings)
      if (auto v = co->getString("runtimeModelLaunchOverhead"))
        bp.runtime_model_launch_overhead = v->str();
      if (auto v = co->getString("runtimeModelDmaTransfer"))
        bp.runtime_model_dma_transfer = v->str();
      if (auto v = co->getString("runtimeModelMemoryHierarchy"))
        bp.runtime_model_memory_hierarchy = v->str();
      if (auto v = co->getString("runtimeModelOccupancy"))
        bp.runtime_model_occupancy = v->str();
      if (auto v = co->getString("runtimeModelRegisterPressure"))
        bp.runtime_model_register_pressure = v->str();
      if (auto v = co->getString("runtimeModelSharedMemory"))
        bp.runtime_model_shared_memory = v->str();
      if (auto v = co->getString("runtimeModelSourceLevel"))
        bp.runtime_model_source_level = v->str();
      if (auto v = co->getString("runtimeModelTruthBoundary"))
        bp.runtime_model_truth_boundary = v->str();

      // Level 5: Cost Model Placeholder (planning-only)
      if (auto v = co->getString("costModelKind"))
        bp.cost_model_kind = v->str();
      if (auto v = co->getString("costModelInputs"))
        bp.cost_model_inputs = v->str();
      if (auto v = co->getString("costModelSourceLevel"))
        bp.cost_model_source_level = v->str();
      if (auto v = co->getString("costModelTruthBoundary"))
        bp.cost_model_truth_boundary = v->str();

      // Legacy cost params: present → has value; absent → nullopt (unknown, not free).
      if (auto v = co->getNumber("layoutTransformCostMs"))
        bp.layout_transform_cost_ms = *v;
      if (auto v = co->getNumber("castCostMs"))
        bp.cast_cost_ms = *v;
      if (auto v = co->getNumber("quantizeCostMs"))
        bp.quantize_cost_ms = *v;
      if (auto v = co->getNumber("dequantizeCostMs"))
        bp.dequantize_cost_ms = *v;
      if (auto v = co->getNumber("requantizeCostMs"))
        bp.requantize_cost_ms = *v;
      if (auto v = co->getNumber("backendTransferCostMs"))
        bp.backend_transfer_cost_ms = *v;

      if (auto v = co->getString("sourceLevel"))   bp.source_level   = v->str();
      if (auto v = co->getString("truthBoundary")) bp.truth_boundary = v->str();

      prof.backendCapabilities.push_back(std::move(bp));
    }
  }

  // Parse kernelLibraries array (Layer 3: actual kernel availability).
  if (auto *klibs = obj->getArray("kernelLibraries")) {
    auto readStrings = [](const llvm::json::Object *o,
                          llvm::StringRef key) -> std::vector<std::string> {
      std::vector<std::string> result;
      if (auto *arr = o->getArray(key))
        for (const auto &elem : *arr)
          if (auto s = elem.getAsString()) result.push_back(s->str());
      return result;
    };

    for (const auto &kelem : *klibs) {
      const llvm::json::Object *ko = kelem.getAsObject();
      if (!ko) continue;

      KernelLibraryProfile kp;
      if (auto v = ko->getString("opType"))        kp.op_type        = v->str();
      if (auto v = ko->getString("kernelName"))    kp.kernel_name    = v->str();
      if (auto v = ko->getString("backend"))       kp.backend        = v->str();
      if (auto v = ko->getString("kernelLibrary")) kp.kernel_library = v->str();

      kp.supported_dtypes       = readStrings(ko, "supportedDtypes");
      kp.supported_layouts      = readStrings(ko, "supportedLayouts");
      kp.supported_quant_modes  = readStrings(ko, "supportedQuantModes");

      if (auto v = ko->getInteger("requiredMAlignment"))
        kp.required_m_alignment = static_cast<int64_t>(*v);
      if (auto v = ko->getInteger("requiredNAlignment"))
        kp.required_n_alignment = static_cast<int64_t>(*v);
      if (auto v = ko->getInteger("requiredKAlignment"))
        kp.required_k_alignment = static_cast<int64_t>(*v);

      if (auto v = ko->getBoolean("supportsDynamicShape"))   kp.supports_dynamic_shape   = *v;
      if (auto v = ko->getBoolean("requiresConstantWeight")) kp.requires_constant_weight = *v;
      if (auto v = ko->getBoolean("supportsFusion"))         kp.supports_fusion          = *v;

      kp.fusion_patterns   = readStrings(ko, "fusionPatterns");
      kp.rewrite_patterns  = readStrings(ko, "rewritePatterns");

      if (auto v = ko->getString("fallbackKernel"))  kp.fallback_kernel  = v->str();
      if (auto v = ko->getString("fallbackBackend")) kp.fallback_backend = v->str();
      if (auto v = ko->getString("sourceLevel"))     kp.source_level     = v->str();
      if (auto v = ko->getString("truthBoundary"))   kp.truth_boundary   = v->str();

      prof.kernelLibraries.push_back(std::move(kp));
    }
  }

  // Parse runtimeKernels array (kernel_selection_contract_v1). Absent in
  // most profiles; KernelSelectionPass then records
  // deferred_no_kernel_library_declared per op — never a silent no-op.
  if (auto *rks = obj->getArray("runtimeKernels")) {
    auto readStrings = [](const llvm::json::Object *o,
                          llvm::StringRef key) -> std::vector<std::string> {
      std::vector<std::string> result;
      if (auto *arr = o->getArray(key))
        for (const auto &elem : *arr)
          if (auto s = elem.getAsString()) result.push_back(s->str());
      return result;
    };
    for (const auto &relem : *rks) {
      const llvm::json::Object *ro = relem.getAsObject();
      if (!ro) continue;
      mlir::hir::RuntimeKernelDescriptor rk;
      if (auto v = ro->getString("kernelId"))       rk.kernel_id = v->str();
      if (auto v = ro->getString("opName"))         rk.op_name = v->str();
      if (auto v = ro->getString("backend"))        rk.backend = v->str();
      rk.supported_dtypes      = readStrings(ro, "supportedDtypes");
      rk.supported_quant_modes = readStrings(ro, "supportedQuantModes");
      rk.supported_layouts     = readStrings(ro, "supportedLayouts");
      rk.supported_tile_shapes = readStrings(ro, "supportedTileShapes");
      if (auto v = ro->getBoolean("requiresStaticShape"))
        rk.requires_static_shape = *v;
      if (auto v = ro->getInteger("requiresLocalMemoryBytes"))
        rk.requires_local_memory_bytes = static_cast<int64_t>(*v);
      if (auto v = ro->getString("source"))         rk.source = v->str();
      if (auto v = ro->getString("truthBoundary"))  rk.truth_boundary = v->str();
      prof.runtimeKernels.push_back(std::move(rk));
    }
  }

  return prof;
}

// ---------------------------------------------------------------------------
// lowerToCapabilityBundle — tool-boundary mapping for ExecutionPlanBuilder.
//
// Populates only the fields read by ExecutionPlanBuilder::build:
//   hardware.hardware_id        → provenance.capability_bundle.hardware_profile_ref
//   backends[*].backend_name    → provenance.capability_bundle.backend_profile_refs
//   kernels[*].kernel_library   → provenance.capability_bundle.kernel_profile_refs
//   deployment.memory_budget_fraction → global_decisions.memory.memory_budget_fraction
// ---------------------------------------------------------------------------

static mlir::hir::CapabilityBundle
lowerToCapabilityBundle(const TargetDeviceProfile &prof) {
  mlir::hir::CapabilityBundle bundle;

  bundle.hardware.hardware_id   = prof.profileId;
  bundle.hardware.truth_boundary = prof.truthBoundary;

  for (const auto &bp : prof.backendCapabilities) {
    mlir::hir::BackendCapability cap;
    cap.backend_name = bp.backend_name;
    bundle.backends.push_back(std::move(cap));
  }

  for (const auto &kp : prof.kernelLibraries) {
    mlir::hir::KernelLibraryCapability ke;
    ke.kernel_library = kp.kernel_library;
    bundle.kernels.push_back(std::move(ke));
  }

  bundle.deployment.target_profile_id      = prof.profileId;
  bundle.deployment.memory_budget_fraction  = 0.75;
  bundle.deployment.truth_boundary          = prof.truthBoundary;

  if (prof.hasPrefillMsPerToken)
    bundle.deployment.prefill_ms_per_token = prof.prefillMsPerToken;
  if (prof.hasDecodeMsPerToken)
    bundle.deployment.decode_ms_per_token  = prof.decodeMsPerToken;

  return bundle;
}

// ---------------------------------------------------------------------------
// printTerminalSummary
// ---------------------------------------------------------------------------

static void printTerminalSummary(const TargetDeviceProfile &prof,
                                  const mlir::hir::ExecutionPlan &plan,
                                  llvm::StringRef canonicalPath) {
  llvm::outs() << "\n";
  llvm::outs() << "compile-for-target: " << prof.profileId
               << " \xe2\x86\x92 " << plan.model_identity.model_id << "\n";
  llvm::outs() << "  canonical:      " << canonicalPath << "\n";
  llvm::outs() << "  function plans: " << plan.function_plans.size() << "\n";

  for (const auto &fp : plan.function_plans) {
    llvm::outs() << "\n";
    llvm::outs() << "  " << fp.function_name << ":\n";
    llvm::outs() << "    serving_phase:    "
                 << (fp.serving_phase == mlir::hir::ServingPhase::Prefill
                         ? "prefill"
                         : fp.serving_phase == mlir::hir::ServingPhase::Decode
                               ? "decode"
                               : "unknown")
                 << "\n";
    llvm::outs() << "    selected_backend: "
                 << fp.backend.selected_backend << "\n";
    llvm::outs() << "    decision_source:  "
                 << fp.backend.meta.source_pass << "\n";
  }

  if (plan.global_decisions.memory) {
    const auto &mem = *plan.global_decisions.memory;
    llvm::outs() << "\n";
    llvm::outs() << "  global.memory:\n";
    llvm::outs() << "    kv_cache_layout:      " << mem.kv_cache_layout << "\n";
    llvm::outs() << "    estimated_kv_peak_mb: " << mem.estimated_kv_peak_mb << "\n";
  }

  if (plan.global_decisions.serving) {
    const auto &srv = *plan.global_decisions.serving;
    llvm::outs() << "  global.serving:\n";
    llvm::outs() << "    topology:         " << srv.topology << "\n";
    llvm::outs() << "    replay_eligible:  "
                 << (srv.replay_eligible ? "true" : "false") << "\n";
  }

  llvm::outs() << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "compile-for-target: compile MLIR for a target device profile\n");

  // 1. Parse device profile.
  auto profOrErr = parseDeviceProfile(DeviceProfilePath);
  if (!profOrErr) {
    llvm::errs() << "error: " << llvm::toString(profOrErr.takeError()) << "\n";
    return 1;
  }
  TargetDeviceProfile prof = std::move(*profOrErr);

  // 2. Lower profile → TargetConstraints + CapabilityBundle.
  mlir::hir::TargetConstraints constraints = lowerToTargetConstraints(prof);
  mlir::hir::CapabilityBundle   capabilities = lowerToCapabilityBundle(prof);

  // 3. Parse MLIR module.
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);
  ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect,
                  mlir::linalg::LinalgDialect, mlir::arith::ArithDialect,
                  mlir::math::MathDialect>();

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(MlirPath, &ctx);
  if (!module) {
    llvm::errs() << "error: failed to parse MLIR file: " << MlirPath << "\n";
    return 1;
  }

  // 4. Attach TargetConstraints as module attrs.
  constraints.attachToModule(module.get(), &ctx);

  // 4a. Optional quantization co-design policy: attach only when the
  // profile declares one; the co-design pass is inert otherwise.
  if (!prof.quantizationCoDesignPolicy.empty())
    module.get()->setAttr(
        "quant.codesign.policy",
        mlir::StringAttr::get(&ctx, prof.quantizationCoDesignPolicy));

  // 4b. Attach an experimental forced global quantization override, only
  // when the profile explicitly opts in via forcedQuantization. This is a
  // driver-level attribute set, not a compiler pass: it does not run
  // QuantizationPlanningPass and it does not touch any other profile's
  // behavior. attrToGlobalQuantDecision() (ExecutionPlanBuilder.cpp) reads
  // these same attr names for both this driver-set path and (if ever run)
  // QuantizationPlanningPass's own output.
  if (prof.hasForcedQuantization) {
    auto *ctxPtr = &ctx;
    mlir::Operation *moduleOp = module.get();
    moduleOp->setAttr("quantization.plan_dtype",
                       mlir::StringAttr::get(ctxPtr, "int4"));
    moduleOp->setAttr("quantization.plan_source",
                       mlir::StringAttr::get(ctxPtr, "forced_quant_profile"));
    moduleOp->setAttr("quantization.truth_boundary",
                       mlir::StringAttr::get(ctxPtr, prof.forcedQuantTruthBoundary));
    moduleOp->setAttr("quantization.strategy",
                       mlir::StringAttr::get(ctxPtr, prof.forcedQuantStrategy));
    moduleOp->setAttr("quantization.algorithm",
                       mlir::StringAttr::get(ctxPtr, prof.forcedQuantAlgorithm));
    moduleOp->setAttr("quantization.quantized_model_artifact_ref",
                       mlir::StringAttr::get(ctxPtr, prof.forcedQuantArtifactRef));
  }

  // 5. Run the serving-optimization-pipeline (16 planning passes), then
  //    boundary materialization (the first IR-transforming stage: inserts
  //    hir.cast where the selected plan requires a cast boundary, before
  //    ExecutionPlanBuilder collects the annotated module).
  mlir::PassManager pm(&ctx);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createServingPhaseAnalysisPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createKVLayoutPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createReplayEligibilityPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createExecutionProviderPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createRepresentationPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createLayoutPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createBoundaryPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createWeightClassificationPlanningPass());  // pass 8
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createQuantizationStrategyPlanningPass());
  // Real upstream CV graphs use cv.* semantic attrs rather than llm.* serving
  // ops. This skip-safe pass only annotates functions already marked by
  // cv-semantic-annotation, then the generic kernel/lowering/selection passes
  // below can refine those attrs from the same target profile infrastructure.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createCVExecutionPlanAttrsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createKernelAvailabilityPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createLoweringDecisionPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createQuantizedBoundaryRefinementPass());
  // Memory-hierarchy-aware tile planning (inert unless the profile declares
  // staticCostProfile.localMemoryBytes). Runs after quantization strategy so
  // quant dtypes shape the tile footprint, before candidate evaluation so
  // the cost model can annotate tiled traffic.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createTilePlanningPass());
  // Concrete runtime-kernel contract selection (kernel_selection_contract_v1):
  // matches ops against target.runtime_kernels descriptors; defers with an
  // explicit reason when no registry is declared. Runs after tile planning
  // so tile constraints are visible.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createKernelSelectionPass());
  // Quantization co-design evidence (quantization_codesign_contract_v1):
  // inert unless the profile/module opts in via quant.codesign.policy, so
  // existing plans are byte-identical by default.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createQuantizationCoDesignPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createAlternativeLoweringPlanningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createCandidateGenerationPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createCandidateEvaluationPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createPlanSelectionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hir::createBoundaryMaterializationPass());

  if (pm.run(module.get()).failed()) {
    llvm::errs() << "error: serving pass pipeline failed\n";
    return 1;
  }

  // 6. Optionally dump annotated MLIR.
  if (!DumpAnnotatedMlir.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream os(DumpAnnotatedMlir, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "warning: cannot write annotated MLIR to '"
                   << DumpAnnotatedMlir << "': " << ec.message() << "\n";
    } else {
      module->print(os);
    }
  }

  // 7. Build ExecutionPlan from annotated module.
  std::string plan_id = prof.profileId + "_serving_plan";
  mlir::hir::ExecutionPlan plan =
      mlir::hir::ExecutionPlanBuilder::build(module.get(), capabilities,
                                               plan_id);

  // 8. Export canonical artifact.
  if (auto err = mlir::hir::ExecutionPlanExporter::exportToFile(plan,
                                                                    OutPath)) {
    llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }

  // 8b. Optional Phase 26 dispatch-unit reconciliation report.
  if (!DispatchUnitReportPath.empty()) {
    if (auto err = mlir::hir::ExecutionPlanExporter::exportDispatchUnitReport(
            plan, DispatchUnitReportPath)) {
      llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
      return 1;
    }
  }

  // 9. Print terminal summary.
  printTerminalSummary(prof, plan, OutPath);

  return 0;
}
