# YOLO-Seg Generic Frontend Gap Report

## Summary

`models/yolo-seg.onnx` was available locally and the generic frontend pipeline
completed successfully.

Phase 12A reduced YOLO-Seg `nn.unknown` coverage gaps to zero by adding
model-agnostic mappings and shape rules for:

- `ConvTranspose`
- `Div`
- `MaxPool`
- `Slice`
- `Split`
- `Sub`

Phase 12B added bounded, model-agnostic constant/initializer extraction for
shape-bearing operands. Phase 12C fixed generic ONNX Reshape semantics for
copied `0` dimensions and inferred `-1` dimensions. Current readiness is
`ready_for_generic_lowering`: generic op and static shape coverage are
complete for this model.

Expected model path:

```text
models/yolo-seg.onnx
```

Command:

```bash
scripts/run_yoloseg_generic_frontend.sh
```

Equivalent direct command:

```bash
.venv/bin/python tools/run_generic_onnx_frontend.py models/yolo-seg.onnx artifacts/yoloseg_generic_frontend --prefix yoloseg
```

Setup details: `docs/YOLOSEG_MODEL_SETUP.md`.

Project ONNX inventory: `docs/ONNX_MODEL_INVENTORY.md`.

This report is diagnostics-only. No YOLO/CV recognition, Transformer/LLM
recognition, MLIR lowering, ExecutionPlan generation, or Qwen GraphFacts
changes were added.

## Model Discovery

Searches performed:

- `*yolo*seg*.onnx`
- `*yoloseg*.onnx`
- `*yolo*.onnx`
- all `.onnx` files under the repo root to depth 4

Project ONNX files found:

- `models/bert_tiny.onnx`
- `models/tiny_mlp.onnx`
- `models/matmul_add_relu.onnx`
- `models/yolo-seg.onnx` locally present for this run

No `models/yolo-seg.onnx.data` sidecar was present; this run used the single
ONNX protobuf file.

See `docs/ONNX_MODEL_INVENTORY.md` for the current project ONNX inventory and
the dependency ONNX files that are intentionally ignored.

## Frontend Execution

Status: completed.

Artifacts emitted under `artifacts/yoloseg_generic_frontend/`:

- `yoloseg.imported_graph_ir.json`
- `yoloseg.generic_graph_ir.json`
- `yoloseg.canonical_generic_graph_ir.json`
- `yoloseg.shape_generic_graph_ir.json`
- `yoloseg.diagnostics_report.json`
- `yoloseg.frontend_report.json`

## Diagnostics

Source: `artifacts/yoloseg_generic_frontend/yoloseg.diagnostics_report.json`.

- `frontend_readiness_status`: `ready_for_generic_lowering`
- `node_count`: 268
- `unknown_op_count`: 0
- `unknown_source_op_types`: []
- `shape_inference_status_histogram`:
  - `inferred`: 268

Generic op histogram:

- `nn.conv2d`: 76
- `nn.sigmoid`: 67
- `nn.mul`: 67
- `nn.concat`: 18
- `nn.reshape`: 11
- `nn.split`: 8
- `nn.add`: 8
- `nn.maxpool2d`: 3
- `nn.resize`: 2
- `nn.slice`: 2
- `nn.sub`: 2
- `nn.conv_transpose2d`: 1
- `nn.div`: 1
- `nn.softmax`: 1
- `nn.transpose`: 1

Remaining shape errors:

- None.

Remaining unknown shape nodes:

- None.

Remaining partial shape nodes:

- None.

Missing generic op support:

- None for the current YOLO-Seg graph.

Missing shape inference support:

- None for the current static model.

## Resolved Final Concat Error

The failing node was:

- node id/name: `267`, `/model.22/Concat_4`
- source op: `Concat`
- canonical attrs: `axis=1`
- inputs:
  - `/model.22/Mul_2_output_0`, produced by node 266
    `/model.22/Mul_2` (`Mul`): `[1,4,8400]`
  - `/model.22/Sigmoid_output_0`, produced by node 253
    `/model.22/Sigmoid` (`Sigmoid`): previously `[1,80,-3]`
  - `/model.22/Concat_2_output_0`, produced by node 251
    `/model.22/Concat_2` (`Concat`, `axis=2`): previously `[1,32,-3]`

The concat axis and ranks were correct. Upstream `nn.reshape` nodes had
canonical targets such as `[1,80,-1]` and `[1,32,-1]`, but shape inference
incorrectly emitted `-1` as a static output dimension. Concatenating three
branches on axis 2 then summed those invalid dimensions to `-3`.

The generic fix resolves a single `-1` from the input element count, applies
ONNX `allowzero=0` copy semantics for `0`, validates invalid target shapes,
and leaves the inferred dimension unknown when static resolution is not
possible. The corrected producer shapes have trailing dimension `8400`; the
final concat output is `[1,116,8400]`.

## Existing-MLIR Lowering Contract

Phase 13 checks the shape-annotated graph without emitting MLIR:

```bash
.venv/bin/python tools/check_generic_lowering_contract.py \
  --in artifacts/yoloseg_generic_frontend/yoloseg.shape_generic_graph_ir.json \
  --out artifacts/yoloseg_generic_frontend/yoloseg.lowering_contract.json
```

Current result:

- `contract_status`: `ready_for_existing_mlir_lowering`
- unsupported strategies: none for the current static model
- missing required canonical attrs: none
- missing shapes: none
- missing dtypes: none

Phase 14A selected narrow existing-dialect strategies for the model's exact
semantics. Broader transposed-convolution and resize variants remain blocked.
See `docs/CONV_TRANSPOSE_RESIZE_LOWERING_DECISION.md`.

## Gap Analysis Procedure

Once `models/yolo-seg.onnx` is added, run:

```bash
scripts/run_yoloseg_generic_frontend.sh
```

Then inspect:

- `artifacts/yoloseg_generic_frontend/yoloseg.frontend_report.json`
- `artifacts/yoloseg_generic_frontend/yoloseg.diagnostics_report.json`
- `artifacts/yoloseg_generic_frontend/yoloseg.shape_generic_graph_ir.json`

Use `yoloseg.diagnostics_report.json` as the primary source for:

- generic op coverage gaps via `unknown_source_op_types`
- shape/type gaps via `shape_inference_status_histogram`
- failing nodes via `shape_error_nodes`
- incomplete nodes via `shape_unknown_nodes` and
  `shape_partially_inferred_nodes`
- model size pressure via `top_initializers_by_raw_data_bytes`
- overall readiness via `frontend_readiness_status`

## Recommended Next Implementation Phases

1. Implement the selected `tensor`/`arith` resize and `linalg`/`arith`
   transposed-convolution forms as isolated Phase 14B prototypes.
2. Verify both through upstream MLIR verification and the existing
   bufferization/lowering pipeline before building a general emitter.
3. Preserve emitted frontend diagnostics as checked-in sample outputs only if
   they are small enough for the repo.
4. Keep domain recognition as a separate phase over `GenericGraphIR`.
