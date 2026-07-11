# Runtime Kernel Contract (kernel_selection_contract_v1)

## What this is

A compiler-side **kernel selection framework** — not broad kernel coverage.
The compiler selects a `selected_kernel` for an op **only when a concrete
`RuntimeKernelDescriptor` exists and fully matches the planned op**:

```text
op name × backend × dtype × quant mode × layout (when both sides state one)
        × static/dynamic shape × tile plan (when the kernel is
        tile-constrained) × declared local memory (when required)
```

Everything else is rejected or deferred with an explicit reason. Coverage is
never inferred from the third-party library layer, from op names, or from
wishful thinking.

## Truth boundary

- A selection is a **contract handed to the runtime**, not an execution
  claim. `truth_boundary =
  kernel_selection_static_descriptor_match_not_runtime_execution`.
- The compiler never executes, dispatches, or benchmarks kernels.
- Measured performance may only enter a descriptor as
  `source: "measured_runtime"` after a real benchmark exists (see the
  checklist below); nothing in this repo claims that today.

## Two layers, deliberately distinct

| Layer | Models | Pass | Question |
|---|---|---|---|
| `kernelLibraries` (`KernelLibraryCapability`) | Declared third-party library coverage (cuBLAS, Triton, CoreML public APIs) | `KernelAvailabilityPlanningPass` | "Does the declared library claim coverage for this tuple?" |
| `runtimeKernels` (`RuntimeKernelDescriptor`) | Concrete kernels with a known dispatchable implementation | `KernelSelectionPass` | "Which specific kernel is the compiler/runtime contract for this op?" |

A library claiming matmul coverage does **not** create a runtime kernel
contract; only a descriptor does.

## Kernels actually declared today

Exactly **one**:

| kernel_id | op | backend | dtype | source | implementation |
|---|---|---|---|---|---|
| `metal_rmsnorm_f32_v1` | rmsnorm | metal | fp32 | `handwritten_runtime` | `metal/rmsnorm.metal` (`rmsnorm_f32`) + `src/runtime/metal_rmsnorm_executor.mm` |

Declared in `configs/target_profiles/apple_a17pro_mobile.json` under
`runtimeKernels`, with `truth_boundary =
handwritten_kernel_source_in_repo_dispatch_validated_not_benchmarked` (the
dispatch path is CTest-validated when the MLIR pipeline has produced the
required trace; no throughput/latency is claimed).

Everything else — matmul, attention, MLP, embeddings, all CV ops — is
**rejected** (`rejected_no_kernel_for_op`, `rejected_dtype_unsupported`, …)
or **deferred** (`deferred_no_kernel_library_declared`,
`deferred_dynamic_shape`, `deferred_missing_tile_plan`, …). Note that even
RMSNorm honestly rejects with `backend_mismatch` on the A17 Pro plan,
because that plan selects the CoreML backend as primary while the kernel
dispatches on Metal. The demo-harness CPU kernels (`src/kernels/`) and the
other Metal shaders are deliberately **not** declared: they are toy-IR demo
code without a serving-plan dispatch path.

## Descriptor schema (profile JSON `runtimeKernels[]`)

```json
{
  "kernelId": "metal_rmsnorm_f32_v1",
  "opName": "rmsnorm",
  "backend": "metal",
  "supportedDtypes": ["fp32"],
  "supportedQuantModes": ["none"],
  "supportedLayouts": [],
  "supportedTileShapes": [],
  "requiresStaticShape": true,
  "requiresLocalMemoryBytes": 0,
  "source": "handwritten_runtime",
  "implementationRef": "path/to/implementation",
  "truthBoundary": "..."
}
```

Empty lists mean *unconstrained* (`supportedLayouts: []` = layout-agnostic).
A non-empty `supportedTileShapes` requires a planned `tile_plan` to match —
absence of a tile plan defers, a mismatching plan rejects.
`source` ∈ `handwritten_runtime` | `declared_profile` | `measured_runtime` |
`fixture` (test-only descriptors must be labeled `fixture`).

C++ type: `RuntimeKernelDescriptor` in
`mlir_passes/include/serving/TargetConstraints.h`. Module attr:
`target.runtime_kernels`. Pass:
`mlir_passes/lib/serving/KernelSelectionPass.cpp`
(`kernel-selection-pipeline`). Export: per-op `kernel_selection` object in
`execution_plan.json` (status, `selected_kernel` + `source` when selected,
`rejection_reasons`, `contract_version`, `truth_boundary`).

## How to add a new handwritten runtime kernel (checklist)

1. **Implement the kernel** in the runtime (shader/C++), with a unit or
   dispatch-validation test that actually runs it.
2. **Register it in the runtime's dispatch path** so the runtime can map
   the exported `selected_kernel` id to the implementation
   (runtime-repo responsibility; this repo only exports the contract).
3. **Add the compiler `RuntimeKernelDescriptor`**: a `runtimeKernels` entry
   in the relevant target profile(s), stating only what the implementation
   truly supports (dtypes, quant modes, layouts, tile shapes, shape
   staticness, local-memory needs), `source: "handwritten_runtime"`, an
   `implementationRef`, and an honest `truthBoundary`.
4. **Add a FileCheck case** to
   `mlir_passes/test/serving/kernel_selection.mlir` showing the kernel is
   selected under its real constraints and rejected outside them.
5. **Add/extend a schema/CTest assertion** if the export surface changes
   (see `RunCompileForTargetTest.cmake`).
6. **Only after a real benchmark exists**, add measured evidence and flip
   the descriptor `source` to `measured_runtime` with a
   `measured_profile`-class truth boundary. Never before.

## Non-goals

- No runtime execution, scheduling, or dispatch in this repo.
- No coverage claims beyond declared descriptors.
- No performance claims without a measured profile.
