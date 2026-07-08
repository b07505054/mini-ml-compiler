# GenericGraphIR v0

`GenericGraphIR` is the compiler-owned, model-agnostic graph layer produced
from `ImportedGraphIR`.

`ImportedGraphIR` is source-faithful: it preserves ONNX node names, op types,
attributes, values, initializers, shapes, dtypes, and provenance with minimal
interpretation.

`GenericGraphIR` is normalized: source op types are mapped into a small
compiler-owned `nn.*` vocabulary while preserving source mapping back to the
imported graph and ONNX names.

Domain recognition happens after `GenericGraphIR`.

## Pipeline Role

```text
ONNX
  -> ImportedGraphIR
  -> GenericGraphIR
  -> Domain Recognition
      -> LLM dialect
      -> CV dialect
  -> Planning
  -> ExecutionPlan
```

Phase 2 implements only `ImportedGraphIR -> GenericGraphIR`. It does not add
domain recognition or change the existing Qwen GraphFacts path.

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

`values[]` fields:

- `name`
- `source_name`
- `dtype`
- `shape`

`initializers[]` fields:

- `name`
- `source_name`
- `dtype`
- `shape`
- `data_location`, when present in source
- `raw_data_bytes`, when present in source

`provenance` fields:

- `source_schema`
- `source_schema_version`
- `source_truth_boundary`
- `truth_boundary`: `imported_graph_ir_normalized_no_domain_recognition`

## Generic Op Vocabulary

Initial v0 vocabulary:

- `nn.conv2d`
- `nn.add`
- `nn.mul`
- `nn.matmul`
- `nn.gemm`
- `nn.reshape`
- `nn.transpose`
- `nn.concat`
- `nn.resize`
- `nn.sigmoid`
- `nn.relu`
- `nn.softmax`
- `nn.identity`
- `nn.unknown`

## ONNX Mapping

Current mapping:

- `Conv` -> `nn.conv2d`
- `Add` -> `nn.add`
- `Mul` -> `nn.mul`
- `MatMul` -> `nn.matmul`
- `Gemm` -> `nn.gemm`
- `Reshape` -> `nn.reshape`
- `Transpose` -> `nn.transpose`
- `Concat` -> `nn.concat`
- `Resize` -> `nn.resize`
- `Sigmoid` -> `nn.sigmoid`
- `Relu` -> `nn.relu`
- `Softmax` -> `nn.softmax`
- `Identity` -> `nn.identity`
- any unmapped op -> `nn.unknown`

## Non-Goals

Phase 2 does not implement:

- Domain recognition
- Transformer or LLM recognition
- CV recognition
- ExecutionPlan schema changes
- MLIR lowering
- Shape inference beyond imported metadata
- Operator legality or target planning
