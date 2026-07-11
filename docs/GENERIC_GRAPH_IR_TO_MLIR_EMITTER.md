# GenericGraphIR-to-MLIR Emitter v0

`tools/generic_graph_ir_to_mlir.py` is the first automatic emitter from
shape-annotated `GenericGraphIR` into existing upstream MLIR dialects.

It does not introduce a custom generic `nn` MLIR dialect. The v0 emitter uses:

- `func`
- `tensor`
- `linalg`
- `arith`
- `math`

It performs no domain recognition, no YOLO/CV recognition, no LLM recognition,
and no `ExecutionPlan` generation.

## Input Contract

Input must be shape-annotated `GenericGraphIR` JSON and must pass:

```bash
.venv/bin/python tools/check_generic_lowering_contract.py \
  --in path/to/shape_generic_graph_ir.json \
  --out path/to/lowering_contract.json
```

The emitter then applies its own smaller v0 support gate. A graph can be
`ready_for_existing_mlir_lowering` at the contract level while still containing
ops that this first emitter intentionally does not emit yet.

## Module ABI

The current ABI is intentionally simple:

- one `func.func` per GenericGraphIR graph
- graph inputs become function arguments
- external initializer operands consumed by emitted ops become additional
  function arguments when literal tensor data is not embedded in GenericGraphIR
- graph outputs become function returns
- tensor types are ranked static tensors derived from value metadata
- supported v0 data type is `f32`
- source node metadata is preserved as comments near emitted operations

Example shape:

```mlir
module {
  func.func @graph(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
    ...
    return %out : tensor<2x3xf32>
  }
}
```

## Supported Ops

The v0 emitter supports:

| GenericGraphIR op | MLIR lowering |
| --- | --- |
| `nn.constant` | `arith.constant` dense tensor constant |
| `nn.identity` | SSA value forwarding |
| `nn.add` | `linalg.generic` + `arith.addf` |
| `nn.sub` | `linalg.generic` + `arith.subf` |
| `nn.mul` | `linalg.generic` + `arith.mulf` |
| `nn.div` | `linalg.generic` + `arith.divf` |
| `nn.conv2d` | `tensor.pad` when needed + `linalg.fill` + `linalg.conv_2d_nchw_fchw` + optional bias `linalg.generic` |
| `nn.conv_transpose2d` | selected non-overlapping stride-2 `linalg.generic` + `arith` |
| `nn.relu` | `linalg.generic` + `arith.maximumf` |
| `nn.sigmoid` | `linalg.generic` + `math.exp` + `arith` |
| `nn.reshape` | `tensor.collapse_shape`, `tensor.expand_shape`, or `tensor.cast` |
| `nn.maxpool2d` | `tensor.pad` + `linalg.fill` + `linalg.pooling_nchw_max` |
| `nn.softmax` | stable `linalg.generic` reductions + `math.exp` + `arith.divf` |
| `nn.transpose` | `linalg.transpose` |
| `nn.resize` | `tensor.generate` + `tensor.extract` + `arith.divui` for selected nearest 2x NCHW resize |
| `nn.slice` | `tensor.extract_slice` |
| `nn.split` | one `tensor.extract_slice` per output |
| `nn.concat` | `tensor.empty` + repeated `tensor.insert_slice` |

Elementwise ops support identical static tensor shapes, rank-aligned static
broadcasting, and scalar tensor broadcasting. Unsupported broadcast relations
fail before MLIR is emitted.

`nn.conv2d` support is the Phase 19 static standard-convolution subset:

- static rank-4 `f32` input/output in NCHW layout
- static rank-4 `f32` weights in FCHW/OIHW layout
- `groups = 1`
- known positive `kernel_shape`, `strides`, and `dilations`
- known non-negative ONNX `pads = [top, left, bottom, right]`
- optional rank-1 `f32` bias whose length equals the output channel count
- output shape must exactly match the ONNX convolution formula

Non-zero padding is emitted as `tensor.pad` over spatial dimensions only, with
zero `f32` padding. The convolution output is initialized with zero through
`linalg.fill`. Bias, when present, is added with a `linalg.generic` broadcast
over N/H/W.

`nn.conv_transpose2d` support is the Phase 20 verified YOLO-Seg subset:

- static rank-4 `f32` input/output in NCHW layout
- static rank-4 `f32` ONNX ConvTranspose weights in `[input_channels,
  output_channels, kernel_h, kernel_w]` layout for `groups = 1`
- `kernel_shape = [2,2]`
- `strides = [2,2]`
- `dilations = [1,1]`
- `pads = [0,0,0,0]`
- `output_padding = [0,0]`
- no explicit `output_shape`
- optional rank-1 `f32` bias whose length equals the output channel count

The lowering initializes the output tensor with either the broadcast bias or
zero, then emits a specialized `linalg.generic` with affine/index expressions
`oh floordiv 2`, `ow floordiv 2`, `oh mod 2`, and `ow mod 2`. This preserves
the non-overlapping ONNX ConvTranspose placement for the selected
`kernel_shape == strides` form.

`nn.reshape` support is intentionally semantic-subset based: static ranked
input/output shapes, equal element count, `allowzero` and `target_shape`
validation, identity casts only for shape-compatible types, and row-major
contiguous reassociation expressible by `tensor.collapse_shape` or
`tensor.expand_shape`. The ONNX-style two-input form is accepted when the
second input is a static shape operand already reflected in canonical attrs
and output metadata. Reshapes that require non-contiguous remapping or general
data movement are rejected.

`nn.maxpool2d` support is the static `f32` NCHW subset used by the current
YOLO-Seg graph:

- rank-4 input/output tensors
- known `kernel_shape`, `pads`, `strides`, and `dilations`
- non-negative pads and positive kernel/stride/dilation values
- `ceil_mode = 0`
- output shape exactly matches floor-mode NCHW pooling
- padding and output initialization use negative infinity

`nn.softmax` support requires static ranked `f32` tensors and a canonical
static axis. The lowering computes max reduction, subtracts the max, applies
`math.exp`, computes the sum reduction, and divides by the sum.

`nn.resize` support is exactly the Phase 14B selected subset:

- rank-4 NCHW tensors
- `mode = nearest`
- `coordinate_transformation_mode = asymmetric`
- `nearest_mode = floor`
- static `scales = [1, 1, 2, 2]`

`nn.slice` support requires static `starts`, `ends`, `axes`, and `steps`,
with normalized axes and `steps = 1`. Negative or non-unit steps are rejected.

`nn.split` support requires a static axis and either explicit static split
sizes or an equal split derivable from the input dimension and output count.
Each output receives a distinct SSA value.

`nn.concat` support requires static ranked inputs and output, a static axis,
matching non-axis dimensions, and an accumulated concat-axis size equal to the
declared output shape.

## Unsupported Ops

Unsupported ops fail before MLIR is written unless `--allow-partial` is passed.
The default behavior avoids producing partial invalid MLIR.

Examples of contract-supported but emitter-unsupported ops in v0 include:

- `nn.matmul`
- `nn.gemm`

Unsupported semantic forms for otherwise supported ops include:

- non-static shapes
- non-`f32` tensor element types
- grouped/depthwise conv2d forms (`groups != 1`)
- conv2d channel, kernel, padding, stride, dilation, output-shape, or bias
  inconsistencies
- conv_transpose2d forms outside the exact static `f32` non-overlapping
  stride-2 subset above
- reshape forms without valid contiguous Tensor dialect reassociation
- maxpool layouts/ranks/dtypes outside static `f32` NCHW, invalid attrs, or `ceil_mode != 0`
- softmax invalid/dynamic axes or non-floating dtypes
- invalid transpose permutations
- resize modes other than nearest/asymmetric/floor static NCHW 2x
- slice forms with negative or non-unit steps
- split forms with unresolved or inconsistent split sizes
- concat forms with rank, non-axis dimension, or accumulated output mismatch

## Constants

`nn.constant` requires small `literal_values` metadata on the output value and
currently supports only dense `f32` tensor constants. Large initializer storage
and external weight materialization remain future emitter work.

## Relation To Phase 14B Prototypes

Phase 14B verified isolated existing-dialect prototypes for selected
`nn.resize` and `nn.conv_transpose2d` forms:

- nearest 2x NCHW resize using `tensor.generate`, `tensor.extract`, and `arith`
- selected stride-2 transposed convolution using `linalg.generic` and `arith`

Those prototypes remain structural proof points. Phase 16 wires the resize
prototype subset into the automatic emitter. Phase 20 wires the selected
transposed-convolution subset into the automatic emitter.

## Truth Boundary

The emitted MLIR contains a truth-boundary comment:

```text
generic_graph_ir_to_existing_mlir_no_domain_recognition_no_execution_plan_generation
```

This marks the output as existing-dialect MLIR only, with no domain lowering,
runtime planning, or execution-plan generation.

## Post-Emission Bufferization Boundary

Phase 21 validates the next structural lowering boundary for the full
YOLO-Seg artifact without changing emitter semantics:

```text
tensor/linalg MLIR
  -> one-shot-bufferize{bufferize-function-boundaries}
  -> buffer-deallocation-pipeline
  -> verified memref/linalg MLIR
```

The prototype script is:

```bash
scripts/lower_yoloseg_mlir_to_bufferized.sh
```

It emits:

- `artifacts/yoloseg_generic_frontend/yoloseg.bufferized.mlir`
- `artifacts/yoloseg_generic_frontend/yoloseg.bufferization_report.json`

For the current YOLO-Seg graph, this removes all remaining `tensor.*` ops and
keeps `linalg.*` as the compute dialect over memrefs. The truth boundary is:

```text
full_graph_bufferization_verified_no_machine_codegen_no_runtime_execution_no_numerical_equivalence_validation_no_execution_plan_generation
```

See `docs/YOLOSEG_MLIR_NEXT_LOWERING_BOUNDARY.md` for the measured operation
histogram, available pass surface, and candidate path comparison.
