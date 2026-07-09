# ConvTranspose2D and Resize Lowering Decision

## Decision

Use the upstream dialects already integrated by the project:

- lower the selected `nn.resize` subset to `tensor.generate`,
  `tensor.extract`, and `arith`
- lower the selected `nn.conv_transpose2d` subset to `linalg.generic` and
  `arith`

Do not add a custom dialect. Do not add TOSA to the Phase 14B dependency
surface.

Both selected forms were parsed and verified with the installed `mlir-opt`
before updating the lowering contract. This is a semantic and structural
decision only; the general GenericGraphIR emitter is not implemented.

## Available Dialects

The configured `/opt/homebrew/opt/llvm` installation contains:

- `libMLIRTosaDialect.a`
- `libMLIRTosaTransforms.a`
- `libMLIRTosaToLinalg.a`
- `libMLIRTosaToArith.a`
- `libMLIRTosaToTensor.a`
- `libMLIRTosaToSCF.a`

The installed TOSA dialect defines both:

- `tosa.transpose_conv2d`
- `tosa.resize`

Adding `MLIRTosaDialect` and registering `mlir::tosa::TosaDialect` would be a
small CMake/source change. That makes TOSA linkable at low build-system risk,
but it does not make the semantic integration free:

- the current project does not link or register TOSA
- TOSA image operations use TOSA layout and operand conventions, requiring
  explicit conversion from the frontend's NCHW and ONNX weight layouts
- `tosa.resize` has a conversion pattern in the installed
  `libMLIRTosaToLinalg.a`
- no `tosa.transpose_conv2d` conversion pattern was found in the installed
  `libMLIRTosaToLinalg.a`

Using TOSA for only these three nodes would therefore add a dialect dependency,
layout conversions, and an incomplete downstream conversion path.

## Actual Model Requirements

Source model: `models/yolo-seg.onnx`, ONNX opset 20.

### Resize Nodes

Nodes 99 (`/model.10/Resize`) and 115 (`/model.13/Resize`) have identical
semantics:

- dtype: `float`
- rank/layout: NCHW
- `mode = nearest`
- `coordinate_transformation_mode = asymmetric`
- `nearest_mode = floor`
- `scales = [1.0, 1.0, 2.0, 2.0]`
- node 99: `[1,256,20,20] -> [1,256,40,40]`
- node 115: `[1,128,40,40] -> [1,128,80,80]`

`cubic_coeff_a = -0.75` is preserved as a source attribute but is irrelevant
to nearest-neighbor mode.

For these attributes, ONNX indexing reduces to:

```text
output[n,c,oh,ow] = input[n,c,floor(oh/2),floor(ow/2)]
```

### Transposed Convolution Node

Node 150 (`/model.22/proto/upsample/ConvTranspose`) requires:

- dtype: `float`
- input: `[1,64,80,80]`
- weight: `[64,64,2,2]` in ONNX ConvTranspose layout
- bias: `[64]`
- output: `[1,64,160,160]`
- `kernel_shape = [2,2]`
- `strides = [2,2]`
- `dilations = [1,1]`
- `groups = 1`
- `pads = [0,0,0,0]`
- `output_padding = [0,0]`
- no explicit `output_shape`

Because kernel size equals stride and padding is zero, output positions map
without overlapping spatial kernel windows:

```text
input index  = [n, ic, oh floordiv 2, ow floordiv 2]
weight index = [ic, oc, oh mod 2, ow mod 2]
```

A `linalg.generic` reduction over `ic`, with a bias-initialized output tensor,
expresses this exact operation without zero insertion or control flow.

## Candidate Evaluation

### TOSA

Advantages:

- both source-level operation names exist upstream
- TOSA provides explicit image-operation semantics
- installed TOSA resize can be converted toward Linalg

Disadvantages:

- new project dialect registration and link dependency
- NCHW and weight-layout conversion overhead
- no installed TOSA-to-Linalg conversion was found for transpose convolution
- the final lowering path would remain incomplete for one of the two blockers

Decision: do not integrate TOSA in Phase 14B.

### Linalg/Tensor/Arith Decomposition

Advantages:

- uses dialects already exercised by this project
- preserves NCHW directly
- both exact model cases verify as legal upstream MLIR
- no runtime ABI or custom operation is required

Risks:

- these are selected semantic subsets, not general implementations
- direct transposed-convolution indexing applies only when kernel equals
  stride, with unit dilation, zero padding, zero output padding, and group 1
- resize lowering currently applies only to rank-4 integer spatial upscales,
  unchanged batch/channels, nearest/asymmetric/floor semantics
- naive `tensor.generate` resize and `linalg.generic` transposed convolution
  need later performance lowering and bufferization validation

Decision: use this strategy for Phase 14B.

### Runtime Boundary

A runtime boundary would preserve unsupported variants but requires a defined
ABI, capability query, and execution semantics. None exists for these generic
frontend operations.

Decision: keep unmatched variants blocked by the lowering contract rather than
inventing a runtime boundary prematurely.

## Selected Contract Predicates

`nn.resize` is structurally supported only when:

- mode is `nearest`
- coordinate transformation is `asymmetric`
- nearest mode is `floor`
- scales are rank-4
- batch and channel scales are one
- spatial scales are exactly `[2,2]`

`nn.conv_transpose2d` is structurally supported only when:

- `groups == 1`
- dilations are `[1,1]`
- pads are all zero
- output padding is `[0,0]`
- kernel shape equals strides
- explicit output shape is absent

All other variants remain `needs_lowering_support`.

## Why No Custom Dialect

The selected semantics are expressible using existing upstream `tensor`,
`arith`, and `linalg` operations. A custom operation would not preserve
additional domain meaning, enable a required analysis, or solve a missing
runtime boundary. It would only defer decomposition.

## Phase 14B Prototype

Phase 14B adds isolated verified MLIR forms, not a general emitter:

1. `mlir/generic_resize_nearest_2x_prototype.mlir` uses `tensor.generate`,
   `arith.divui`, and `tensor.extract`.
2. `mlir/generic_conv_transpose2d_stride2_prototype.mlir` builds a
   bias-initialized output and applies the specialized `linalg.generic`
   reduction.
3. Both files carry FileCheck assertions and are registered as CTests.
4. Contract tests reject non-nearest, non-2x, unsupported coordinate,
   grouped, padded, output-padded, and overlapping variants.
5. Unsupported variants remain rejected before MLIR emission.
