# GenericGraphIR v0

`GenericGraphIR` is the compiler-owned, model-agnostic graph layer produced
from `ImportedGraphIR`.

`ImportedGraphIR` is source-faithful: it preserves ONNX node names, op types,
attributes, values, initializers, shapes, dtypes, and provenance with minimal
interpretation.

`GenericGraphIR` is normalized: source op types are mapped into a small
compiler-owned `nn.*` vocabulary while preserving source mapping back to the
imported graph and ONNX names.

`GenericGraphIR` is canonicalized before domain recognition. Canonicalization
preserves source attributes and writes compiler-owned `canonical_attrs`.

Shape/type inference may then annotate the canonical graph with conservative,
model-agnostic tensor metadata.

Diagnostics may be generated over shape/type annotated `GenericGraphIR` to
summarize frontend readiness without changing the graph.

Domain recognition happens after `GenericGraphIR` canonicalization and
shape/type consistency analysis.

## Pipeline Role

```text
ONNX
  -> ImportedGraphIR
  -> GenericGraphIR
  -> Canonical GenericGraphIR
  -> Shape/Type Annotated GenericGraphIR
  -> Diagnostics Report
  -> Domain Recognition
      -> LLM dialect
      -> CV dialect
  -> Planning
  -> ExecutionPlan
```

Phase 2 implements only `ImportedGraphIR -> GenericGraphIR`. It does not add
domain recognition or change the existing Qwen GraphFacts path.

Phase 4 adds `tools/canonicalize_generic_graph_ir.py`, which normalizes selected
`nn.*` attributes into compiler-owned `canonical_attrs`.

Phase 5 adds `tools/infer_generic_graph_shapes.py`, which performs conservative
model-agnostic shape/type consistency analysis over canonicalized
`GenericGraphIR`.

Phase 6 adds `tools/run_generic_onnx_frontend.py`, which runs the generic ONNX
frontend pipeline end to end through shape/type annotation and writes all
intermediate artifacts plus a frontend report.

Phase 7 adds `tools/diagnose_generic_graph_ir.py`, which reads shape-annotated
`GenericGraphIR` and emits a diagnostics/readiness report. It is reporting
only; it does not perform domain recognition, lowering, or planning.

## Schema

Top-level fields:

- `schema`: fixed string, `generic_graph_ir`
- `schema_version`: current version, `0.1.0`
- `graph`: graph metadata
- `nodes`: normalized node list
- `values`: tensor/value metadata
- `initializers`: initializer metadata
- `provenance`: source schema metadata and truth boundary

`graph` fields:

- `name`: graph name carried from `ImportedGraphIR`
- `source_name`: source graph name
- `inputs`: graph input value names
- `outputs`: graph output value names
- `opset_version`: source default ONNX opset version when available
- `opset_imports`: source opset imports
- `source_graph`: `{schema, schema_version, graph_name}` reference

`nodes[]` fields:

- `id`: stable node id, currently copied from ImportedGraphIR node id
- `name`: source node name
- `op`: compiler-owned generic op name
- `inputs`: input value names
- `outputs`: output value names
- `attributes`: source attributes preserved as imported
- `source_node_id`: ImportedGraphIR node id
- `source_op_type`: original ONNX op type
- `source_name`: original ONNX node name
- `source_domain`: original ONNX operator domain

Canonicalized nodes also include:

- `source_attributes`: original imported/source attributes, preserved exactly
  as JSON records
- `canonicalized`: `true`
- `canonicalization_version`: canonicalization pass version
- `canonical_attrs`: compiler-owned normalized attributes

Shape/type annotated nodes also include:

- `shape_inference_status`: `inferred`, `partially_inferred`, `unknown`, or
  `error`
- `inferred_outputs`: output tensor metadata records
- `shape_inference_notes`: explanatory notes or error details

`values[]` fields:

- `name`
- `source_name`
- `dtype`
- `shape`
- `literal_values`, optional small numeric tensor contents preserved from
  `ImportedGraphIR`

`initializers[]` fields:

- `name`
- `source_name`
- `dtype`
- `shape`
- `data_location`, when present in source
- `raw_data_bytes`, when present in source
- `literal_values`, optional small numeric tensor contents preserved from
  `ImportedGraphIR`

`provenance` fields:

- `source_schema`
- `source_schema_version`
- `source_truth_boundary`
- `truth_boundary`: `imported_graph_ir_normalized_no_domain_recognition`

## Generic Op Vocabulary

Initial v0 vocabulary:

- `nn.conv2d`
- `nn.conv_transpose2d`
- `nn.add`
- `nn.sub`
- `nn.mul`
- `nn.div`
- `nn.matmul`
- `nn.gemm`
- `nn.reshape`
- `nn.transpose`
- `nn.concat`
- `nn.split`
- `nn.slice`
- `nn.resize`
- `nn.maxpool2d`
- `nn.sigmoid`
- `nn.relu`
- `nn.softmax`
- `nn.identity`
- `nn.constant`
- `nn.unknown`

## ONNX Mapping

Current mapping:

- `Conv` -> `nn.conv2d`
- `ConvTranspose` -> `nn.conv_transpose2d`
- `Add` -> `nn.add`
- `Sub` -> `nn.sub`
- `Mul` -> `nn.mul`
- `Div` -> `nn.div`
- `MatMul` -> `nn.matmul`
- `Gemm` -> `nn.gemm`
- `Reshape` -> `nn.reshape`
- `Transpose` -> `nn.transpose`
- `Concat` -> `nn.concat`
- `Split` -> `nn.split`
- `Slice` -> `nn.slice`
- `Resize` -> `nn.resize`
- `MaxPool` -> `nn.maxpool2d`
- `Sigmoid` -> `nn.sigmoid`
- `Relu` -> `nn.relu`
- `Softmax` -> `nn.softmax`
- `Identity` -> `nn.identity`
- `Constant` -> `nn.constant`
- any unmapped op -> `nn.unknown`

## Canonical Attributes

Canonicalization v0 normalizes:

- `nn.conv2d`: `pads`, `strides`, `dilations`, `groups`, `kernel_shape`
- `nn.conv_transpose2d`: `pads`, `strides`, `dilations`, `groups`,
  `kernel_shape`, `output_padding`, `output_shape`
- `nn.maxpool2d`: `kernel_shape`, `pads`, `strides`, `dilations`,
  `ceil_mode`
- `nn.transpose`: `perm`
- `nn.concat`: `axis`
- `nn.split`: `axis`, `split`
- `nn.slice`: `axes`, `starts`, `ends`, `steps`
- `nn.resize`: `mode`, `coordinate_transformation_mode`, `nearest_mode`
- `nn.softmax`: `axis`
- `nn.gemm`: `alpha`, `beta`, `transA`, `transB`
- `nn.reshape`: `allowzero`, plus `target_shape` when its shape input is static

Defaults are compiler-owned and explicit:

- `nn.conv2d`: `pads=[0,0,0,0]`, `strides=[1,1]`, `dilations=[1,1]`,
  `groups=1`, `kernel_shape=[]`
- `nn.conv_transpose2d`: `pads=[0,0,0,0]`, `strides=[1,1]`,
  `dilations=[1,1]`, `groups=1`, `kernel_shape=[]`,
  `output_padding=[0,0]`, `output_shape=[]`
- `nn.maxpool2d`: `kernel_shape=[]`, `pads=[0,0,0,0]`, `strides=[1,1]`,
  `dilations=[1,1]`, `ceil_mode=0`
- `nn.transpose`: `perm=[]`
- `nn.concat`: `axis=0`
- `nn.split`: `axis=0`; `split` is present only when provided by source attrs
- `nn.slice`: optional `axes`, `starts`, `ends`, `steps` when provided by
  source attrs or static shape-bearing inputs
- `nn.resize`: `mode="nearest"`,
  `coordinate_transformation_mode="half_pixel"`,
  `nearest_mode="round_prefer_floor"`
- `nn.softmax`: `axis=-1`
- `nn.gemm`: `alpha=1.0`, `beta=1.0`, `transA=0`, `transB=0`
- `nn.reshape`: `allowzero=0`

Unknown or unhandled ops keep `op` unchanged and receive empty
`canonical_attrs`; source attributes are still preserved.

Canonicalization also reads bounded `literal_values` metadata from values and
initializers to recover static shape-bearing operands without recognizing a
model family:

- `nn.reshape`: `target_shape` from the second input when static
- `nn.slice`: `starts`, `ends`, `axes`, and `steps` from static inputs
- `nn.resize`: `scales` and `sizes` from static inputs
- `nn.split`: `split` from a static split-size input when not already present
  as an attribute

Large tensors are never required to be inlined for canonicalization.

## Shape/Type Inference

Shape/type inference v0 supports static integer dimensions and unknown or
symbolic dimensions. It does not solve full symbolic equations. Unknown shapes
do not fail the graph; they are recorded as `unknown` or
`partially_inferred`.

Implemented checks and inference:

- `nn.conv2d`: NCHW rank-4 input/weight checks, channel/group consistency,
  output shape from canonical pads/strides/dilations/kernel shape
- `nn.conv_transpose2d`: NCHW rank-4 input/weight checks and transposed
  convolution output shape from canonical attributes
- `nn.maxpool2d`: NCHW rank-4 pooling output shape from canonical attributes
- `nn.add`, `nn.sub`, `nn.mul`, `nn.div`: NumPy-style broadcast shape
  inference and obvious incompatibility detection
- `nn.matmul`: rank >= 2 shape inference, batch broadcasting, inner-dimension
  compatibility checks
- `nn.gemm`: rank-2 matrix shape inference using `transA`/`transB`
- `nn.reshape`: static target shape with ONNX-compatible `allowzero`, copied
  `0` dimensions, and single `-1` dimension inference when element counts are
  statically known; unresolved inferred dimensions remain unknown
- `nn.transpose`: output permutation and rank compatibility
- `nn.concat`: rank/axis compatibility and axis-size accumulation
- `nn.split`: multiple output shapes when split sizes or equal static splits
  are known; otherwise partial axis-size inference
- `nn.slice`: static output shape when starts/ends/axes/steps are available
- `nn.resize`: output `sizes`/`scales` when present, otherwise conservative
  rank-preserving partial inference
- `nn.softmax`, `nn.sigmoid`, `nn.relu`, `nn.identity`: unary shape and dtype
  propagation
- `nn.constant`: output metadata propagation from existing value metadata
- `nn.unknown`: `unknown` status with source metadata preserved

## Diagnostics

Diagnostics v0 reports:

- graph name
- node count
- value count
- initializer count
- op histogram
- unknown op count
- unknown source op types
- shape inference status histogram
- nodes with `error`, `unknown`, or `partially_inferred` shape status
- missing or unresolved dtype/shape metadata counts
- top initializer sizes by `raw_data_bytes`
- verifier status
- frontend readiness status
- truth boundary:
  `diagnostics_only_no_domain_recognition_no_mlir_lowering_no_execution_plan_generation`

Readiness values are:

- `ready_for_generic_lowering`
- `needs_op_support`
- `needs_shape_support`
- `invalid_ir`

## Verifier Invariants

`tools/verify_graph_ir.py` validates `GenericGraphIR` before domain recognition
or lowering. The verifier checks:

- Required top-level fields exist: `schema`, `schema_version`, `graph`,
  `nodes`, `values`, `initializers`, `provenance`.
- `schema == "generic_graph_ir"` and `schema_version == "0.1.0"`.
- Every node op is in the supported `nn.*` vocabulary or is `nn.unknown`.
- Every node preserves source mapping fields: `source_node_id`,
  `source_op_type`, and `source_name`.
- Every node input resolves to a graph input, value metadata entry,
  initializer, or prior node output.
- Every graph output is produced or declared.
- Value and initializer records carry consistent names, dtypes, and shapes.
- Each initializer has corresponding value metadata.
- Schema field names do not contain domain-specific terms such as `qwen`,
  `llm`, `yolo`, `cv`, `kv_cache`, `attention`, `backbone`, `neck`, or
  `head`.

## Non-Goals

Phase 2 does not implement:

- Domain recognition
- Transformer or LLM recognition
- CV recognition
- ExecutionPlan schema changes
- MLIR lowering
- Operator legality or target planning
