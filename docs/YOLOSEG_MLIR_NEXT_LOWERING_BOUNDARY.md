# YOLO-Seg MLIR Next Lowering Boundary

Phase 21 defines the first post-emission lowering boundary for the full real
YOLO-Seg graph. It does not add runtime execution, backend code generation,
numerical validation, or `ExecutionPlan` integration.

## Current Input

The input artifact is:

```text
artifacts/yoloseg_generic_frontend/yoloseg.generic.mlir
```

It verifies with `mlir-opt` and contains the complete YOLO-Seg graph:

- emitted nodes: 268 / 268
- unsupported nodes: 0
- emitted dialects: `func`, `tensor`, `linalg`, `arith`, `math`
- function: `@main_graph`
- function arguments: 158 static tensor arguments
- model input arguments: 1
- initializer/weight/constant arguments: 157
- returns: `(tensor<1x116x8400xf32>, tensor<1x32x160x160xf32>)`

The text file is about 395 KB and 3,285 lines. The largest expansion concern
is not textual size; it is the number of whole-tensor temporaries produced by
the current SSA tensor emitter.

## Operation Histogram

Full YOLO-Seg emitted MLIR operation histogram:

| Op | Count |
| --- | ---: |
| `arith.addf` | 152 |
| `arith.constant` | 217 |
| `arith.divf` | 69 |
| `arith.divui` | 4 |
| `arith.maximumf` | 1 |
| `arith.mulf` | 68 |
| `arith.subf` | 70 |
| `func.func` | 1 |
| `linalg.conv_2d_nchw_fchw` | 76 |
| `linalg.fill` | 81 |
| `linalg.generic` | 226 |
| `linalg.pooling_nchw_max` | 3 |
| `linalg.transpose` | 1 |
| `linalg.yield` | 226 |
| `math.exp` | 68 |
| `tensor.collapse_shape` | 10 |
| `tensor.empty` | 326 |
| `tensor.expand_shape` | 1 |
| `tensor.extract` | 2 |
| `tensor.extract_slice` | 18 |
| `tensor.generate` | 2 |
| `tensor.insert_slice` | 52 |
| `tensor.pad` | 50 |
| `tensor.yield` | 52 |

Selected counts:

- `tensor.empty`: 326
- `tensor.pad`: 50
- `tensor.generate`: 2
- `tensor.extract`: 2
- `tensor.extract_slice`: 18
- `tensor.insert_slice`: 52
- `linalg.generic`: 226
- named linalg ops:
  - `linalg.conv_2d_nchw_fchw`: 76
  - `linalg.fill`: 81
  - `linalg.pooling_nchw_max`: 3
  - `linalg.transpose`: 1
- arith ops: 581
- math ops: 68

## Available Pass Surface

The installed `/opt/homebrew/opt/llvm/bin/mlir-opt` exposes the required
upstream dialects and passes for the next lowering experiments.

Verified available dialects include:

- `affine`
- `arith`
- `bufferization`
- `cf`
- `func`
- `linalg`
- `llvm`
- `math`
- `memref`
- `scf`
- `tensor`
- `vector`

Verified available passes include:

| Pass | Availability |
| --- | --- |
| `one-shot-bufferize` | available |
| `bufferization` dialect | available |
| `buffer-deallocation-pipeline` | available |
| `buffer-deallocation-simplification` | available |
| `bufferization-lower-deallocations` | available |
| `ownership-based-buffer-deallocation` | available |
| `convert-linalg-to-loops` | available |
| `convert-linalg-to-parallel-loops` | available |
| `convert-linalg-to-affine-loops` | available |
| `lower-affine` | available |
| `convert-scf-to-cf` | available |
| `convert-math-to-llvm` | available |
| `convert-arith-to-llvm` | available |
| `finalize-memref-to-llvm` | available |
| `convert-func-to-llvm` | available |
| `convert-cf-to-llvm` | available |
| `convert-index-to-llvm` | available |
| `reconcile-unrealized-casts` | available |
| LLVM dialect | available |

LLVM dialect translation was not selected as a Phase 21 target. The pass
surface is present, but this phase stops before machine code, runtime calls,
or ABI design for executable invocation.

## Candidate Paths

### A. Tensor/Linalg To Bufferized Memref/Linalg To Loops

Pipeline shape:

```text
tensor/linalg
  -> one-shot-bufferize
  -> buffer-deallocation-pipeline
  -> memref/linalg
  -> convert-linalg-to-loops
  -> scf/memref
  -> lower-level backend IR
```

Prerequisites:

- static ranked tensors
- bufferizable tensor producers/consumers
- explicit ownership/deallocation policy
- a later ABI decision for returned result buffers

Available passes:

- all required Phase 21 passes are available locally
- `convert-linalg-to-loops` verifies on the full graph after bufferization

Missing build integration:

- no CMake target or compiler driver wraps this pipeline yet
- no runtime ABI or result-buffer ownership contract
- no backend-specific lowering beyond generic SCF/memref validation

Correctness risk:

- bufferization success is structural only
- no numerical equivalence is proven
- returned buffers currently escape as memref returns
- memory pressure can be high because the tensor emitter materializes many
  temporaries before later optimization

Optimization opportunity:

- clear CPU path through SCF loops
- later loop fusion, tiling, vectorization, and memory reuse can be introduced
  around a concrete memref IR

CPU relevance:

- high; this is the shortest path toward an executable CPU lowering plan

GPU/Metal/CUDA relevance:

- moderate; SCF/memref is useful as a correctness baseline, but GPU backends
  will likely need tiling, layout, vector, or GPU dialect mapping before codegen

### B. Tensor/Linalg Optimization Before Bufferization

Pipeline shape:

```text
tensor/linalg
  -> linalg tiling/fusion/vectorization
  -> one-shot-bufferize
  -> memref/vector/linalg
  -> backend IR
```

Prerequisites:

- a target-independent tiling/fusion policy
- cost model or target profile constraints for tile sizes
- legality checks for reductions, convolutions, transposes, and slice/concat
  materialization

Available passes:

- linalg/vector/affine/scf dialects and conversion passes are available
- this phase did not validate a full graph optimization pipeline before
  bufferization

Missing build integration:

- no generic YOLO-Seg tiling or fusion driver exists
- no target-aware transform pipeline is wired to the generic emitter

Correctness risk:

- higher than Path A because transformations can reorder reductions and expose
  aliasing or layout assumptions
- requires more precise validation per op family

Optimization opportunity:

- highest; this is where conv/elementwise fusion, vectorization, and memory
  traffic reduction should eventually happen

CPU relevance:

- high after a baseline bufferization path exists

GPU/Metal/CUDA relevance:

- high; this is likely the right path for backend-quality code generation

### C. Keep Tensor/Linalg As The Compiler Artifact

Pipeline shape:

```text
tensor/linalg
  -> runtime/backend adapter consumes tensor/linalg later
```

Prerequisites:

- a runtime/backend adapter that understands tensor/linalg MLIR
- external buffer ownership and initializer loading policy
- backend-specific interpretation of high-level linalg ops

Available passes:

- not pass-driven; current tensor/linalg artifact already verifies

Missing build integration:

- no runtime adapter consumes this artifact today
- no execution ABI, initializer materialization, or result handling exists

Correctness risk:

- low for preserving semantics in a compiler artifact
- high if treated as executable without a defined adapter contract

Optimization opportunity:

- deferred; backend-specific adapters can choose their own lowering strategy

CPU relevance:

- moderate as an interchange artifact, low as an executable boundary

GPU/Metal/CUDA relevance:

- moderate; useful as a common semantic source, but still requires backend
  lowering work

## Selected Phase 21 Boundary

The selected minimal prototype boundary is Path A, stopped at bufferized
memref/linalg:

```text
tensor/linalg MLIR
  -> one-shot-bufferize{bufferize-function-boundaries}
  -> buffer-deallocation-pipeline
  -> verified memref/linalg MLIR
```

The exact pass pipeline is:

```text
builtin.module(one-shot-bufferize{bufferize-function-boundaries},buffer-deallocation-pipeline)
```

The prototype script is:

```bash
scripts/lower_yoloseg_mlir_to_bufferized.sh
```

It emits:

- `artifacts/yoloseg_generic_frontend/yoloseg.bufferized.mlir`
- `artifacts/yoloseg_generic_frontend/yoloseg.bufferization_report.json`

## Full Graph Bufferization Result

Full YOLO-Seg bufferization succeeds.

Post-boundary report:

- verification status: `verified_with_mlir_opt_after_bufferization`
- remaining tensor ops: 0
- remaining linalg ops: 723
- introduced memref ops: 989
- `memref.alloc`: 378
- `memref.dealloc`: 376
- `memref.copy`: 102
- `memref.subview`: 120
- `memref.collapse_shape`: 10
- `memref.expand_shape`: 1
- `memref.load`: 2
- bufferized function arguments: 158 memref arguments
- bufferized returns: `(memref<1x116x8400xf32>, memref<1x32x160x160xf32>)`

All source tensor operations bufferize:

- `tensor.empty`: 326 -> 0
- `tensor.pad`: 50 -> 0
- `tensor.generate`: 2 -> 0
- `tensor.extract`: 2 -> 0
- `tensor.extract_slice`: 18 -> 0
- `tensor.insert_slice`: 52 -> 0

The deallocation pipeline inserts 376 deallocations. The two remaining
allocations intentionally escape as function results at this boundary.

## Optional Loop Lowering Probe

After bufferization, this additional command also verifies:

```bash
mlir-opt yoloseg.bufferized.mlir --convert-linalg-to-loops
```

Measured result:

- remaining tensor ops: 0
- remaining linalg ops: 0
- `scf.for`: 1,982
- `memref.load`: 622
- `memref.store`: 439
- `affine.apply`: 162

This is useful evidence for Phase 22, but it is not the selected Phase 21
boundary. Loop IR needs separate ABI, memory, optimization, and backend
policy work.

## Truth Boundary

The Phase 21 truth boundary is:

```text
full_graph_bufferization_verified_no_machine_codegen_no_runtime_execution_no_numerical_equivalence_validation_no_execution_plan_generation
```

Successful bufferization means the full emitted graph structurally lowers to
verified memref/linalg MLIR. It does not prove numerical correctness and does
not imply executable code generation.

## Recommended Phase 22

Phase 22 should define the next boundary after memref/linalg:

1. Choose whether the compiler artifact should stop at memref/linalg or lower
   to SCF loops for a CPU baseline.
2. Define the function ABI for inputs, initializers, result buffers, and
   ownership of returned allocations.
3. Add a loop-lowering script only after the ABI decision.
4. Start memory-pressure reduction with generic linalg/tensor optimization
   passes before targeting backend-specific codegen.
