# ImportedGraphIR v0

`ImportedGraphIR` is the generic ONNX importer boundary. It preserves ONNX graph
structure and metadata before any semantic recognition, domain lowering, or
execution planning.

It is intentionally not a model-family schema. It must not contain terminology
from any downstream domain dialect.

## Pipeline Role

Legacy path, still supported:

```text
Qwen ONNX
  -> Qwen GraphFacts
  -> qwen-onnx-to-serving-mlir
  -> LLM dialect
  -> ExecutionPlan
```

New target path:

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

Phase 1 implemented only the first arrow: ONNX to `ImportedGraphIR`.
Phase 2 adds the next boundary: `ImportedGraphIR` to `GenericGraphIR`.

## Schema

Top-level fields:

- `schema`: fixed string, `imported_graph_ir`
- `schema_version`: current version, `0.1.0`
- `graph`: imported graph payload
- `provenance`: source metadata and truth boundary

`graph` fields:

- `name`: ONNX graph name
- `source_name`: original ONNX graph name
- `opset_version`: default ONNX domain opset version, or null
- `opset_imports`: list of `{domain, version}` records
- `inputs`: graph input value names
- `outputs`: graph output value names
- `nodes`: ordered imported nodes
- `values`: known value/tensor metadata from graph inputs, outputs, value_info,
  and initializers
- `initializers`: initializer tensor metadata

`nodes[]` fields:

- `id`: stable import-order integer
- `name`: ONNX node name, possibly empty if the source node is unnamed
- `source_name`: original ONNX node name
- `op_type`: ONNX op type
- `domain`: ONNX operator domain
- `inputs`: input value names
- `outputs`: output value names
- `attributes`: ONNX attributes as `{name, type, value}` records

`values[]` and `initializers[]` fields:

- `name`: value or initializer name
- `source_name`: original ONNX name
- `dtype`: ONNX tensor element type as a lower-case string where available
- `shape`: dimension records, each one `{kind, value}` where `kind` is
  `static`, `symbolic`, or `unknown`

Initializer records also include:

- `data_location`: ONNX data location enum as an integer
- `raw_data_bytes`: number of bytes stored in `raw_data`

`provenance` fields:

- `source_format`: `onnx`
- `source_file`: basename of the imported source file
- `producer_name`
- `producer_version`
- `ir_version`
- `truth_boundary`:
  `onnx_protobuf_metadata_preserved_no_domain_recognition`

## Non-Goals

Phase 1 does not implement:

- Domain recognition
- Transformer or CV pattern recognition
- ExecutionPlan schema changes
- Lowering into MLIR dialects
- Shape inference beyond metadata already present in the ONNX file
- Weight loading beyond initializer metadata
