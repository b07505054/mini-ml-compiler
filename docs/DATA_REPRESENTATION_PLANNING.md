# Data Representation Planning — Architecture Plan

## Purpose

This document defines the architecture for the Data Representation Planning layer: the
compiler-side schema, pass design, and source-grounded hardware description model for
handling Cast (dtype coercion), Dequant (integer-domain boundary management), and Layout
Transform (memory address order) decisions.

**Status:** Design plan. No code exists for this layer yet. See audit results in the
summary below before implementation begins.

---

## Source Labeling Rules

Every capability or cost value in hardware profiles must carry one of these labels.
Fabricating specific numbers to fill fields is explicitly prohibited (see CLAUDE.md).

| Label | Meaning |
|---|---|
| `public_docs` | Fact appears in publicly released technical documentation, whitepaper, or official SDK reference |
| `declared_profile` | Value is a declaration in this project's profile JSON, not a measured result |
| `estimated` | Derived from a formula or assumption, not from measurement on the described hardware |
| `unknown` | Not publicly documented at the required granularity — field must be left null or absent |
| `measured` | Obtained by running a benchmark on the actual hardware and recording the result with provenance |

**Cost values must never be listed as a specific number without `truth_boundary`.**
If hardware internal latency is unknown, the field is null, not a plausible guess.

---

## Schema: `HardwareTargetProfile` (JSON, tool-boundary input)

This is the profile file read by the `compile-for-target` tool. It extends the existing
`TargetDeviceProfile` fields with per-backend capability declarations.

```json
{
  "profileId":         "string — unique across all profile files",
  "vendor":            "string — apple | nvidia | arm | intel | qualcomm | unknown",
  "deviceClass":       "string — mobile_soc | datacenter_gpu | cpu_amx | edge_npu | custom_asic",
  "sourceLevel":       "string — public_docs | declared_profile | measured_profile | unknown",
  "truthBoundary":     "string",

  "backendCapabilities": [
    {
      "backendName":   "string — cpu | gpu | neural_engine | tensor_core | amx | custom_npu",
      "backendApi":    "string — coreml | metal | cuda | arm_compute | iree_hal | custom_runtime",

      "supportedDtypes":          ["fp32", "fp16", "bf16", "int8", "int4", "fp8", "uint8"],
      "accumulationDtypes":       ["fp32", "fp16"],
      "supportedQuantModes":      ["weight_only", "static_int8", "dynamic_int8", "int4_weight_only"],
      "preferredActivationLayouts": ["NHWC", "NCHW", "any"],
      "preferredWeightLayouts":     ["HWIO", "OHWI", "blocked_kc", "KN", "any"],
      "layoutAgnosticOps":          ["relu", "add", "gelu", "softmax", "reshape", "gather"],
      "requiresAlignedShape":       true,
      "requiredKAlignment":         32,
      "requiredNAlignment":         32,
      "supportsLayoutTransform":    true,
      "supportsCast":               true,
      "supportsDequantBoundary":    true,
      "supportsRequantize":         false,
      "supportsFusionPatterns":     ["matmul_bias_relu", "conv_bn_relu"],

      "memoryHierarchy": {
        "memorySpaces": [
          {
            "name":                    "string",
            "role":                    "string",
            "knownCapacityBytesOrUnknown": null,
            "bandwidthOrUnknown":         null,
            "latencyOrUnknown":           null,
            "sourceLevel":                "string"
          }
        ],
        "notes": "string"
      },

      "costModelParams": {
        "launchOverheadMs":          null,
        "layoutTransformCostMs":     null,
        "castCostMs":                null,
        "quantizeCostMs":            null,
        "dequantizeCostMs":          null,
        "requantizeCostMs":          null,
        "backendTransferCostMs":     null,
        "computeThroughputOrUnknown": null,
        "memoryBandwidthOrUnknown":   null,
        "sourceLevel":               "declared_profile",
        "truthBoundary":             "declared_or_estimated_not_measured_on_hardware"
      },

      "sourceLevel":   "string",
      "truthBoundary": "string"
    }
  ]
}
```

---

## Schema: `BackendCapability` (C++ struct, lives in `TargetConstraints.h`)

This is the C++ view of one `backendCapabilities` entry after parsing and lowering. It is
stored in `TargetConstraints` and round-trips through `target.backend_capabilities.*` MLIR
module attrs (flat prefix form, not nested DictionaryAttr).

```cpp
struct BackendLayoutCapability {
  std::string backend_name;              // "cpu", "gpu", "neural_engine"
  std::string backend_api;               // "metal", "coreml", "cuda", "arm_compute"
  std::vector<std::string> supported_dtypes;
  std::vector<std::string> accumulation_dtypes;
  std::vector<std::string> supported_quant_modes;
  std::vector<std::string> preferred_activation_layouts;
  std::vector<std::string> preferred_weight_layouts;
  std::vector<std::string> layout_agnostic_ops;
  bool requires_aligned_shape  = false;
  int  required_k_alignment    = 0;
  int  required_n_alignment    = 0;
  bool supports_layout_transform = false;
  bool supports_cast             = false;
  bool supports_dequant_boundary = false;
  bool supports_requantize       = false;
  std::vector<std::string> supports_fusion_patterns;

  // Cost params — null (negative sentinel) means unknown/not declared.
  double layout_transform_cost_ms = -1.0;
  double cast_cost_ms             = -1.0;
  double quantize_cost_ms         = -1.0;
  double dequantize_cost_ms       = -1.0;
  double requantize_cost_ms       = -1.0;
  double backend_transfer_cost_ms = -1.0;

  std::string source_level;
  std::string truth_boundary;
};
```

`TargetConstraints` gains:
```cpp
std::vector<BackendLayoutCapability> backend_capabilities;
```

MLIR attr encoding (flat prefix, one attr per field):
```
target.backend_capabilities.{backend}.preferred_activation_layouts  ArrayAttr(StringAttr)
target.backend_capabilities.{backend}.preferred_weight_layouts       ArrayAttr(StringAttr)
target.backend_capabilities.{backend}.supported_dtypes               ArrayAttr(StringAttr)
target.backend_capabilities.{backend}.supported_quant_modes          ArrayAttr(StringAttr)
target.backend_capabilities.{backend}.layout_agnostic_ops            ArrayAttr(StringAttr)
target.backend_capabilities.{backend}.requires_aligned_shape         BoolAttr
target.backend_capabilities.{backend}.required_k_alignment           IntegerAttr
target.backend_capabilities.{backend}.required_n_alignment           IntegerAttr
target.backend_capabilities.{backend}.supports_layout_transform      BoolAttr
target.backend_capabilities.{backend}.supports_cast                  BoolAttr
target.backend_capabilities.{backend}.supports_dequant_boundary      BoolAttr
target.backend_capabilities.{backend}.supports_fusion_patterns       ArrayAttr(StringAttr)
target.backend_capabilities.{backend}.layout_transform_cost_ms       FloatAttr  (absent if unknown)
target.backend_capabilities.{backend}.cast_cost_ms                   FloatAttr  (absent if unknown)
target.backend_capabilities.{backend}.source_level                   StringAttr
target.backend_capabilities.{backend}.truth_boundary                 StringAttr
```

Flat attrs are easier to write FileCheck patterns against than nested
`DictionaryAttr`. The `{backend}` placeholder in the attr name is the literal backend
string — e.g. `target.backend_capabilities.metal.preferred_activation_layouts`.

---

## Concrete Profiles: Source-Grounded Examples

### Apple A17 Pro Mobile (`apple-a17pro-mobile`)

Extends `configs/target_profiles/apple_a17pro_mobile.json`.

#### GPU backend (Metal API)

```json
{
  "backendName": "gpu",
  "backendApi": "metal",
  "supportedDtypes": ["fp32", "fp16"],
  "accumulationDtypes": ["fp32"],
  "supportedQuantModes": [],
  "preferredActivationLayouts": ["NHWC"],
  "preferredWeightLayouts": ["HWIO", "OHWI"],
  "layoutAgnosticOps": ["relu", "add", "softmax", "reshape"],
  "requiresAlignedShape": false,
  "requiredKAlignment": 0,
  "requiredNAlignment": 0,
  "supportsLayoutTransform": true,
  "supportsCast": true,
  "supportsDequantBoundary": false,
  "supportsRequantize": false,
  "supportsFusionPatterns": ["matmul_bias_relu", "conv_bn_relu"],
  "memoryHierarchy": {
    "memorySpaces": [
      {
        "name": "unified_memory",
        "role": "cpu_gpu_shared_dram",
        "knownCapacityBytesOrUnknown": null,
        "bandwidthOrUnknown": null,
        "latencyOrUnknown": null,
        "sourceLevel": "unknown"
      }
    ],
    "notes": "Apple Silicon uses a unified memory architecture shared between CPU and GPU cores. Internal GPU L1/L2 cache sizes and bandwidth are not publicly documented at the required granularity."
  },
  "costModelParams": {
    "launchOverheadMs": null,
    "layoutTransformCostMs": null,
    "castCostMs": null,
    "quantizeCostMs": null,
    "dequantizeCostMs": null,
    "requantizeCostMs": null,
    "backendTransferCostMs": null,
    "computeThroughputOrUnknown": null,
    "memoryBandwidthOrUnknown": null,
    "sourceLevel": "unknown",
    "truthBoundary": "declared_or_estimated_not_measured_on_hardware"
  },
  "sourceLevel": "public_docs",
  "truthBoundary": "layout_preference_from_metal_performance_shaders_documentation_not_measured_kernel_throughput"
}
```

**Source notes for GPU backend:**
- `supportedDtypes: ["fp32", "fp16"]` — Metal Shading Language spec and Metal Performance
  Shaders public documentation. Source: `public_docs`.
- `preferredActivationLayouts: ["NHWC"]` — Metal Performance Shaders image ops (MPSImage)
  use NHWC (feature channels last). Documented in MPSNNGraph API reference. Source:
  `public_docs`.
- `supportsDequantBoundary: false` — Metal does not natively execute INT8 matrix ops
  outside of Core ML delegation. Metal Shading Language has no dedicated INT8 matmul
  intrinsic in public docs. Source: `public_docs`.
- Memory hierarchy: Apple does not publish GPU L1/L2 or tile memory sizes. All cost
  params: `unknown`.

#### Neural Engine backend (Core ML API)

```json
{
  "backendName": "neural_engine",
  "backendApi": "coreml",
  "supportedDtypes": ["fp16", "int8"],
  "accumulationDtypes": ["fp32"],
  "supportedQuantModes": ["static_int8", "weight_only"],
  "preferredActivationLayouts": ["any"],
  "preferredWeightLayouts": ["any"],
  "layoutAgnosticOps": ["relu", "add", "softmax", "reshape", "matmul", "conv2d"],
  "requiresAlignedShape": false,
  "requiredKAlignment": 0,
  "requiredNAlignment": 0,
  "supportsLayoutTransform": false,
  "supportsCast": true,
  "supportsDequantBoundary": true,
  "supportsRequantize": false,
  "supportsFusionPatterns": [],
  "memoryHierarchy": {
    "memorySpaces": [
      {
        "name": "ane_sram",
        "role": "npu_on_chip_buffer",
        "knownCapacityBytesOrUnknown": null,
        "bandwidthOrUnknown": null,
        "latencyOrUnknown": null,
        "sourceLevel": "unknown"
      }
    ],
    "notes": "Neural Engine SRAM size and topology are not publicly documented by Apple. Internal buffer layout is managed by Core ML runtime, not exposed to the compiler."
  },
  "costModelParams": {
    "launchOverheadMs": null,
    "layoutTransformCostMs": null,
    "castCostMs": null,
    "quantizeCostMs": null,
    "dequantizeCostMs": null,
    "sourceLevel": "unknown",
    "truthBoundary": "declared_or_estimated_not_measured_on_hardware"
  },
  "sourceLevel": "public_docs",
  "truthBoundary": "dtype_support_from_coreml_documentation_layout_managed_by_coreml_runtime"
}
```

**Source notes for Neural Engine backend:**
- `supportedDtypes: ["fp16", "int8"]` — Core ML Framework documentation and Apple WWDC
  session materials document FP16 and INT8 as supported precisions for on-device inference
  on Neural Engine. Source: `public_docs`.
- `preferredActivationLayouts: ["any"]` — Core ML manages layout internally. The compiler
  does not control tensor layout passed to the Neural Engine; Core ML converts as needed.
  Source: `public_docs`.
- `supportsLayoutTransform: false` — The Neural Engine is accessed through the Core ML API,
  not through direct buffer layout ops. Layout transforms are irrelevant at this interface
  level. Source: `public_docs`.
- ANE SRAM / internal bandwidth: not publicly documented. All cost params: `unknown`.

#### CPU backend (Arm CPU, Apple Clang)

```json
{
  "backendName": "cpu",
  "backendApi": "arm_compute",
  "supportedDtypes": ["fp32", "fp16", "int8", "uint8"],
  "accumulationDtypes": ["fp32"],
  "supportedQuantModes": ["static_int8", "dynamic_int8"],
  "preferredActivationLayouts": ["NHWC"],
  "preferredWeightLayouts": ["OHWI"],
  "layoutAgnosticOps": ["relu", "add", "reshape", "gather", "softmax"],
  "requiresAlignedShape": true,
  "requiredKAlignment": 32,
  "requiredNAlignment": 32,
  "supportsLayoutTransform": true,
  "supportsCast": true,
  "supportsDequantBoundary": true,
  "supportsRequantize": true,
  "supportsFusionPatterns": ["matmul_bias_relu", "conv_bn_relu"],
  "memoryHierarchy": {
    "memorySpaces": [
      {
        "name": "l1_cache",
        "role": "per_core_instruction_and_data",
        "knownCapacityBytesOrUnknown": null,
        "bandwidthOrUnknown": null,
        "latencyOrUnknown": null,
        "sourceLevel": "unknown"
      },
      {
        "name": "unified_memory",
        "role": "system_dram",
        "knownCapacityBytesOrUnknown": null,
        "bandwidthOrUnknown": null,
        "latencyOrUnknown": null,
        "sourceLevel": "unknown"
      }
    ],
    "notes": "Arm CPU cache hierarchy is architecture-specific and not published for Apple's custom Arm cores. Arm Compute Library recommends NHWC for convolutions as documented in its public reference."
  },
  "costModelParams": {
    "sourceLevel": "unknown",
    "truthBoundary": "declared_or_estimated_not_measured_on_hardware"
  },
  "sourceLevel": "public_docs",
  "truthBoundary": "layout_from_arm_compute_library_docs_alignment_from_neon_kernel_requirements_not_measured"
}
```

**Source notes for CPU backend:**
- `preferredActivationLayouts: ["NHWC"]` — Arm Compute Library documentation explicitly
  recommends NHWC for CNN convolution operators. The library's `NEConvolutionLayer` and
  related operators document NHWC as the preferred data layout for performance on Arm cores.
  Source: `public_docs`.
- `supportedDtypes: ["fp32", "fp16", "int8", "uint8"]` — Arm Compute Library public
  GitHub documentation and README list FP32, FP16, INT8, UINT8, and BF16 as supported
  element types. Source: `public_docs`.
- `requiredKAlignment: 32`, `requiredNAlignment: 32` — derived from NEON SIMD register
  width (128-bit = 16 fp8 or 32 int8 elements). This is a `declared_profile` choice for
  the project's existing `rankedMatmulShape()` check, not a measured constraint.
- Cache sizes: Apple does not publish L1/L2 cache sizes for custom Arm cores. `unknown`.

---

### Generic NVIDIA Datacenter GPU Target (`nvidia-hopper-datacenter`)

This profile does not exist in this project today. It is included here as a
source-grounded reference example for the schema.

#### Tensor Core backend (CUDA)

```json
{
  "profileId": "nvidia-hopper-datacenter",
  "vendor": "nvidia",
  "deviceClass": "datacenter_gpu",
  "sourceLevel": "public_docs",
  "truthBoundary": "declared_or_estimated_not_measured_on_hardware",
  "backendCapabilities": [
    {
      "backendName": "tensor_core",
      "backendApi": "cuda",
      "supportedDtypes": ["fp32", "fp16", "bf16", "tf32", "int8", "fp8"],
      "accumulationDtypes": ["fp32", "fp16"],
      "supportedQuantModes": ["static_int8", "dynamic_int8", "int4_weight_only"],
      "preferredActivationLayouts": ["NHWC", "NCHW"],
      "preferredWeightLayouts": ["OHWI", "OIHW", "KN"],
      "layoutAgnosticOps": ["relu", "add", "gelu", "reshape", "gather"],
      "requiresAlignedShape": true,
      "requiredKAlignment": 16,
      "requiredNAlignment": 16,
      "supportsLayoutTransform": true,
      "supportsCast": true,
      "supportsDequantBoundary": true,
      "supportsRequantize": true,
      "supportsFusionPatterns": ["matmul_bias_relu", "conv_bias", "flash_attention"],
      "memoryHierarchy": {
        "memorySpaces": [
          {
            "name": "registers",
            "role": "per_thread_register_file",
            "knownCapacityBytesOrUnknown": null,
            "bandwidthOrUnknown": null,
            "latencyOrUnknown": null,
            "sourceLevel": "unknown"
          },
          {
            "name": "shared_memory",
            "role": "per_sm_programmable_scratchpad",
            "knownCapacityBytesOrUnknown": 233472,
            "bandwidthOrUnknown": null,
            "latencyOrUnknown": null,
            "sourceLevel": "public_docs"
          },
          {
            "name": "l2_cache",
            "role": "device_level_unified_cache",
            "knownCapacityBytesOrUnknown": 52428800,
            "bandwidthOrUnknown": null,
            "latencyOrUnknown": null,
            "sourceLevel": "public_docs"
          },
          {
            "name": "hbm3",
            "role": "device_dram",
            "knownCapacityBytesOrUnknown": null,
            "bandwidthOrUnknown": null,
            "latencyOrUnknown": null,
            "sourceLevel": "unknown"
          }
        ],
        "notes": "NVIDIA H100 SXM shared memory per SM (228KB) and L2 cache (50MB) documented in H100 Tensor Core GPU Architecture whitepaper. HBM3 bandwidth varies by GPU variant; per-access latency not published."
      },
      "costModelParams": {
        "sourceLevel": "unknown",
        "truthBoundary": "declared_or_estimated_not_measured_on_hardware"
      },
      "sourceLevel": "public_docs",
      "truthBoundary": "dtype_and_alignment_from_nvidia_tensor_core_architecture_whitepaper"
    }
  ]
}
```

**Source notes for NVIDIA Tensor Core backend:**
- `supportedDtypes: [..., "fp8"]` — H100 Tensor Core GPU Architecture whitepaper (2022)
  and cuBLAS documentation. FP8 (E4M3 and E5M2) introduced in Hopper architecture.
  Source: `public_docs`.
- `supportedDtypes: ["bf16"]` — A100 whitepaper and cuBLAS 11+ documentation.
  Source: `public_docs`.
- `supportedDtypes: ["tf32"]` — A100 whitepaper. TF32 is NVIDIA-proprietary; 19-bit
  effective representation, FP32 accumulation. Source: `public_docs`.
- `requiredKAlignment: 16` — CUDA Programming Guide and cuBLAS documentation state that
  Tensor Core operations require dimensions divisible by 16 for INT8/FP16, and 8 for
  TF32. Source: `public_docs`. The value here uses the stricter INT8/FP16 requirement.
- `memoryHierarchy.shared_memory: 233472` — H100 SXM SM has 228KB shared memory
  (228 × 1024 = 233472 bytes). Documented in H100 Architecture whitepaper.
  Source: `public_docs`.
- `memoryHierarchy.l2_cache: 52428800` — H100 SXM has 50MB L2 cache
  (50 × 1024 × 1024 = 52428800 bytes). Documented in H100 Architecture whitepaper.
  Source: `public_docs`.
- All cost params (`layoutTransformCostMs`, `castCostMs`, etc.): `null`. Latency on
  hardware depends on problem size, data locality, and driver version. Not published
  as fixed values. Must be measured if needed.

---

### Generic Intel CPU with AMX (`intel-sapphire-rapids-amx`)

```json
{
  "backendName": "amx",
  "backendApi": "custom_runtime",
  "supportedDtypes": ["bf16", "int8"],
  "accumulationDtypes": ["fp32"],
  "supportedQuantModes": ["static_int8"],
  "preferredActivationLayouts": ["NCHW"],
  "preferredWeightLayouts": ["OIHW", "blocked_kc"],
  "layoutAgnosticOps": ["relu", "add", "reshape"],
  "requiresAlignedShape": true,
  "requiredKAlignment": 32,
  "requiredNAlignment": 16,
  "supportsLayoutTransform": true,
  "supportsCast": false,
  "supportsDequantBoundary": true,
  "supportsRequantize": false,
  "supportsFusionPatterns": ["matmul_bias_relu"],
  "costModelParams": {
    "sourceLevel": "unknown",
    "truthBoundary": "declared_or_estimated_not_measured_on_hardware"
  },
  "sourceLevel": "public_docs",
  "truthBoundary": "dtype_from_intel_amx_isa_extension_documentation_alignment_from_tile_config_spec"
}
```

**Source notes for Intel AMX backend:**
- `supportedDtypes: ["bf16", "int8"]` — Intel Architecture Instruction Set Extensions
  Programming Reference documents AMX-BF16 and AMX-INT8 as separate ISA extensions
  introduced with Sapphire Rapids. FP32 is not supported as an AMX tile dtype.
  Source: `public_docs`.
- `accumulationDtypes: ["fp32"]` — The TDPBF16PS instruction (BF16 tiles → FP32
  accumulator) and TDPBSSD/TDPBSUD/TDPBUSD/TDPBUUD instructions (INT8 tiles → INT32
  accumulator, cast to FP32) are documented. Source: `public_docs`.
- `requiredKAlignment: 32` — AMX tile configuration requires K dimension aligned to
  32 (VNNI INT8 4-byte packing × 8 columns). Documented in Intel AMX Programming
  Model. Source: `public_docs`.

---

## Pass Architecture

### Pass Ordering

```
QuantizationPlanningPass         (module scope — unchanged)
  ↓ emits: quantization.plan_dtype
ExecutionProviderPlanningPass    (func scope — unchanged)
  ↓ emits: execution_provider.primary
RepresentationPlanningPass       (func scope — new)
  reads: quantization.plan_dtype, execution_provider.primary,
         target.backend_capabilities.{backend}.*
  ↓ emits: representation.effective_dtype,
            representation.preferred_activation_layout,
            representation.preferred_weight_layout
LayoutPlanningPass               (func scope — new, planning only)
  reads: representation.preferred_activation_layout,
         representation.preferred_weight_layout
  ↓ emits per-op: layout.effective_layout,
                   layout.required_input_layout,
                   layout.transform_required,
                   layout.transform_cost_estimate_ms (absent if cost unknown)
[BoundaryMaterializationPass]    (deferred — inserts hir.cast, hir.layout_transform)
```

### `RepresentationPlanningPass` — Attr Output Spec

On `func.func`:
```
representation.effective_dtype              StringAttr  "fp32" | "fp16" | "bf16" | "int8"
representation.effective_dtype_source       StringAttr  "target_profile" | "quantization_plan" | "default"
representation.preferred_activation_layout  StringAttr  from backend_capabilities
representation.preferred_weight_layout      StringAttr  from backend_capabilities
representation.truth_boundary               StringAttr
  = "representation_plan_static_not_validated_against_kernel_performance"
```

If `target.backend_capabilities.{backend}.preferred_activation_layouts` is absent
(profile predates the new schema), the pass emits `representation.preferred_activation_layout = "any"`
and sets `representation.truth_boundary = "representation_plan_no_capability_data_layout_unconstrained"`.

### `LayoutPlanningPass` — Attr Output Spec

On each op (except those in `layout_agnostic_ops` for the chosen backend):
```
layout.effective_layout          StringAttr  layout of this op's primary output
layout.required_input_layout     StringAttr  layout this op needs from its inputs
layout.transform_required        BoolAttr    true iff producer layout ≠ required input layout
layout.transform_cost_estimate_ms FloatAttr  absent if backend cost is unknown
layout.truth_boundary            StringAttr
  = "layout_planning_static_cost_model_not_measured_kernel_performance"
```

**Key rule:** ops in `layout_agnostic_ops` propagate their input's `layout.effective_layout`
to their output without marking `layout.transform_required`. This prevents redundant
transforms between a conv and the relu that follows it on the same backend.

**Key rule:** `layout.transform_cost_estimate_ms` is omitted (not zero) when the backend
`costModelParams.layoutTransformCostMs` is null. A zero cost and an absent cost have
different meanings: zero means free, absent means unknown.

### Conflict Annotation

If the chosen backend (`execution_provider.primary`) does not support the dtype from
`quantization.plan_dtype`:

```
representation.conflict = "backend_lacks_dtype"
representation.conflict_fallback_dtype = "fp32"
```

This is an annotation, not an error. It surfaces the mismatch for downstream review
without re-running backend selection.

---

## IR Changes Required

### New ops in `HIROps.td` (deferred until `BoundaryMaterializationPass`)

```tablegen
def HIR_CastOp : HIR_Op<"cast", [Pure]> {
  // Floating-point precision cast. No scale/zero_point. Not lossy in the
  // integer sense; values are rounded to target dtype precision.
  let arguments = (ins AnyRankedTensor:$input,
                       StrAttr:$source_dtype,
                       StrAttr:$target_dtype,
                       StrAttr:$rounding_mode,
                       StrAttr:$truth_boundary);
  let results = (outs AnyRankedTensor:$output);
}

def HIR_LayoutTransformOp : HIR_Op<"layout_transform", [Pure]> {
  // Memory layout permutation. Same dtype, same values, different address order.
  // source_layout and target_layout are symbolic strings (NHWC, NCHW, blocked_kc, etc.)
  let arguments = (ins AnyRankedTensor:$input,
                       StrAttr:$source_layout,
                       StrAttr:$target_layout,
                       StrAttr:$truth_boundary);
  let results = (outs AnyRankedTensor:$output);
}
```

These ops must NOT be added until `BoundaryMaterializationPass` is ready to insert them.
Planning-only attrs are sufficient for the first two commits.

### Canonicalization for new ops (deferred)

`hir.layout_transform(NHWC→NCHW) ∘ hir.layout_transform(NCHW→NHWC) → identity`

This follows the same pattern as `HIRQuantCanonicalizationPass`. The elimination condition
is that `target_layout` of the inner op equals `source_layout` of the outer op.

---

## Impact on Existing Files

### Files modified in Commit A (schema only)

| File | Change |
|---|---|
| `configs/target_profiles/apple_a17pro_mobile.json` | Add `backendCapabilities` array with GPU and ANE entries |
| `configs/target_profiles/apple_iphone15_5_a16.json` | Same |
| `mlir_passes/include/serving/TargetConstraints.h` | Add `BackendLayoutCapability` struct and `backend_capabilities` field |
| `mlir_passes/lib/serving/TargetConstraints.cpp` | Extend `fromModule()`/`attachToModule()` |
| `mlir_passes/tools/compile-for-target/main.cpp` | Extend `TargetDeviceProfile` struct, extend `lowerToTargetConstraints()` |
| `mlir_passes/test/serving/TargetConstraintsTest.cpp` | Add round-trip test for new fields |
| `mlir_passes/test/serving/target_constraints.mlir` | Add FileCheck patterns for new attrs |

### Files NOT changed in Commit A

`kCapabilities[]` in `QuantizationCompilerPasses.cpp` is NOT changed until Commit C
(operator selection refactor). The existing table continues to function. The new
`BackendLayoutCapability` structs from `TargetConstraints` replace the hardcoded layout
strings only after `LayoutPlanningPass` is wired in.

The hardcoded `"NHWC"` / `"blocked_kc"` strings in `MatMulBiasReluFusionPass.cpp:610-613`
remain as-is through Commit B. They are replaced in Commit C when `HIRINT8OperatorSelectionPass`
is updated to read layout preferences from `target.backend_capabilities.cpu.*` attrs.

### Files modified in Commit B (pass only)

| File | Change |
|---|---|
| `mlir_passes/lib/serving/RepresentationPlanningPass.cpp` | New file |
| `mlir_passes/include/FusionPasses.td` | Add `RepresentationPlanning` pass def |
| `mlir_passes/include/FusionPasses.h` | Declare `createRepresentationPlanningPass()` |
| `mlir_passes/lib/MatMulBiasReluFusionPass.cpp` | Register new pass |
| `mlir_passes/test/serving/representation_planning.mlir` | New FileCheck test |

---

## Truth Boundary Vocabulary

These strings must be used consistently. Do not invent new synonyms per-pass.

| Context | Truth boundary string |
|---|---|
| Backend capability declared from profile, not measured | `"declared_backend_layout_preference_not_measured_kernel_throughput"` |
| Layout planning attr, cost from capability table | `"layout_planning_static_cost_model_not_measured_kernel_performance"` |
| Layout transform planned, op not yet materialized | `"layout_transform_planned_not_materialized_no_actual_memory_permutation"` |
| Cast planned, numerical accuracy not validated | `"precision_cast_planned_not_numerical_accuracy_validated"` |
| Dequant boundary planned, no calibration data | `"dequant_boundary_planned_not_calibrated_scale_execution"` |
| Representation planning (dtype + layout combined) | `"representation_plan_static_not_validated_against_kernel_performance"` |
| Cost param is null because hardware internals are not public | `"declared_or_estimated_not_measured_on_hardware"` |
| Capability data absent, layout unconstrained | `"representation_plan_no_capability_data_layout_unconstrained"` |

---

## What This Plan Does NOT Claim

- Specific kernel throughput or latency for any backend. These are `unknown` unless
  measured by a benchmark on the actual hardware with recorded provenance.
- That the Apple Neural Engine prefers any specific tensor layout. Core ML manages
  layout internally; the compiler does not observe it. Layout is `"any"` for ANE.
- That CUDA shared memory latency is a fixed value. Latency depends on bank conflicts,
  access pattern, and SM occupancy — not a single constant.
- That `layout_transform_cost_ms: 0.05` (or any specific value) is correct. All cost
  params in this plan are `null` / `unknown` until measured.
- That the `requiredKAlignment` values are the only legal alignment. They match the
  project's existing `rankedMatmulShape()` check and are labeled `declared_profile`.
