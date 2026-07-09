# GenericGraphIR to Existing MLIR Lowering Contract

## Purpose

This contract defines structural preconditions for a future
`GenericGraphIR`-to-MLIR emitter. It does not emit MLIR, perform domain
recognition, select a runtime, or generate an `ExecutionPlan`.

Existing upstream MLIR dialects are the default lowering targets. No generic
`nn` MLIR dialect is proposed.

## Current MLIR Availability

The configured MLIR installation is `/opt/homebrew/opt/llvm`. Availability has
two distinct meanings:

| Dialect | Installed library | Current project use |
|---|---:|---|
| `func` | yes | explicitly linked by MLIR executables |
| `tensor` | yes | registered and used by the pass plugin; Apple plugin uses dynamic lookup |
| `arith` | yes | registered and used by the pass plugin; Apple plugin uses dynamic lookup |
| `math` | yes | registered and used by the pass plugin; Apple plugin uses dynamic lookup |
| `linalg` | yes | registered and used by the pass plugin; Apple plugin uses dynamic lookup |
| `memref` | yes | installed, not explicitly linked or registered by current project targets |
| `scf` | yes | installed, not explicitly linked or registered by current project targets |
| `affine` | yes | installed; affine maps are used, but the Affine dialect is not explicitly linked or registered |
| `vector` | yes | installed, not explicitly linked or registered by current project targets |
| `tosa` | yes | installed, not explicitly linked or registered by current project targets |
| `stablehlo` / `mhlo` | no | no project dependency or installed library found |

The Phase 13 contract therefore uses `func`, `tensor`, `arith`, `math`, and
`linalg`. Merely finding another dialect in the LLVM installation is not
treated as project lowering support.

## Graph Preconditions

`tools/check_generic_lowering_contract.py` requires:

- valid GenericGraphIR v0 according to `tools/verify_graph_ir.py`
- no `nn.unknown` nodes
- a selected lowering strategy for every node
- required `canonical_attrs` for each operation
- `shape_inference_status == "inferred"` for every node
- ranked shape metadata for every non-empty node input and output
- no unknown dimensions; symbolic dimensions preserve a known dynamic rank
- a known dtype for every non-empty node input and output

An empty shape is a valid rank-0 tensor shape, not missing metadata.

## Operation Contract

“Direct” means one primary upstream operation or SSA forwarding. “Decompose”
means a deterministic expansion into existing upstream operations. A listed
strategy is a contract decision, not evidence that an emitter exists.

| Generic op | Preferred existing MLIR target | Fallback | Required canonical attrs | Shape/dtype requirements | Treatment | Difficulty | Notes |
|---|---|---|---|---|---|---|---|
| `nn.conv2d` | `linalg.conv_2d_nchw_fchw` | `linalg.generic` | `pads`, `strides`, `dilations`, `groups`, `kernel_shape` | ranked input, weight, output; known dtypes | direct or decompose | medium | Grouped/depthwise cases may require generic indexing or decomposition. |
| `nn.conv_transpose2d` | `linalg.generic` + `arith` for the selected non-overlapping subset | TOSA only after complete layout and downstream conversion support | `pads`, `strides`, `dilations`, `groups`, `kernel_shape`, `output_padding`, `output_shape` | ranked input, weight, output; known dtypes | direct specialized | medium | Selected for group 1, unit dilation, zero pads/output padding, kernel equal to stride, and no explicit output shape. Other variants block. |
| `nn.maxpool2d` | `linalg.pooling_nchw_max` | `linalg.generic` | `kernel_shape`, `pads`, `strides`, `dilations`, `ceil_mode` | ranked input/output; known dtype | direct or decompose | medium | Padding identity and ceil semantics must be preserved. |
| `nn.add` | `linalg.generic` + `arith.addf/addi` | TOSA add after TOSA integration | none | broadcast-compatible ranked tensors; known dtypes | direct | low | Index maps encode broadcasting. |
| `nn.sub` | `linalg.generic` + `arith.subf/subi` | TOSA sub after TOSA integration | none | broadcast-compatible ranked tensors; known dtypes | direct | low | Signedness selects the integer operation. |
| `nn.mul` | `linalg.generic` + `arith.mulf/muli` | TOSA mul after TOSA integration | none | broadcast-compatible ranked tensors; known dtypes | direct | low | Index maps encode broadcasting. |
| `nn.div` | `linalg.generic` + `arith.divf/divsi/divui` | TOSA arithmetic after TOSA integration | none | broadcast-compatible ranked tensors; known dtypes | direct | medium | Integer signedness and division semantics must be explicit. |
| `nn.matmul` | `linalg.matmul` or `linalg.batch_matmul` | `linalg.generic` | none | ranked operands/output; compatible inner dimensions; known dtypes | direct | low | Batch broadcasting may need explicit materialization. |
| `nn.gemm` | `linalg.matmul` + `linalg.generic` + `arith` | `linalg.generic` | `alpha`, `beta`, `transA`, `transB` | ranked operands/output; known dtypes | decompose | medium | Decompose transpose, scaling, bias, and matrix multiplication. |
| `nn.reshape` | `tensor.expand_shape`, `tensor.collapse_shape`, or `tensor.reshape` | TOSA reshape after TOSA integration | `allowzero`, `target_shape` | ranked input/output; equal element count when static; known dtype | direct or decompose | medium | Reassociation legality determines the Tensor operation. |
| `nn.transpose` | `linalg.transpose` | `linalg.generic` | `perm` | equal input/output rank; known dtype | direct | low | `perm` must be a complete permutation. |
| `nn.concat` | repeated `tensor.insert_slice` | `linalg.generic` | `axis` | equal rank and compatible non-axis dimensions; known dtype | decompose | medium | Negative axis must be rank-normalizable. |
| `nn.resize` | `tensor.generate` + `tensor.extract` + `arith` for selected nearest 2x upscale | TOSA after complete layout/conversion integration | `mode`, `coordinate_transformation_mode`, `nearest_mode`; static `scales` | ranked input/output; known dtype | direct specialized | medium | Selected for rank-4 nearest/asymmetric/floor scale `[1,1,2,2]`. Other variants block. |
| `nn.softmax` | Linalg reductions + `arith` + `math.exp` | TOSA softmax after TOSA integration | `axis` | ranked input/output; floating dtype | decompose | medium | Use max-subtract-exp-sum-div. |
| `nn.sigmoid` | `linalg.generic` + `arith` + `math.exp` | TOSA sigmoid after TOSA integration | none | same ranked input/output shape; floating dtype | decompose | low | Compute `1 / (1 + exp(-x))`. |
| `nn.relu` | `linalg.generic` + `arith.maximumf/maxsi/maxui` | TOSA clamp after TOSA integration | none | same ranked input/output shape; known dtype | direct | low | Integer signedness must be preserved. |
| `nn.split` | repeated `tensor.extract_slice` | `linalg.generic` | `axis`; `split` when unequal sizes cannot be derived from outputs | ranked input/outputs; known dtype | decompose | low | Output shapes can supply static split sizes. |
| `nn.slice` | `tensor.extract_slice` | `linalg.generic` | `starts`, `ends`; `axes` defaults by rank and `steps` defaults to ones | ranked input/output; known dtype | direct or decompose | medium | Non-unit or negative steps need decomposition. |
| `nn.constant` | `arith.constant` | `tensor.generate` for non-dense construction | none | ranked output and known dtype; literal or initializer data available to emitter | direct | low | Large initializer storage remains an emitter concern. |
| `nn.identity` | SSA value forwarding | `tensor.cast` for compatible type refinement | none | compatible input/output types | direct | trivial | Emit no operation when types are identical. |

## Status Semantics

- `ready_for_existing_mlir_lowering`: every node satisfies this structural
  contract and has a selected existing-dialect strategy.
- `needs_lowering_support`: GenericGraphIR is valid, but at least one node has
  no selected strategy or lacks required attrs, shape status, shape metadata,
  or dtype metadata.
- `invalid_generic_graph_ir`: the GenericGraphIR verifier fails.

The report truth boundary is:

```text
lowering_contract_only_no_mlir_emission_no_domain_recognition_no_execution_plan_generation
```

## Verified Prototypes

- `mlir/generic_resize_nearest_2x_prototype.mlir`
- `mlir/generic_conv_transpose2d_stride2_prototype.mlir`

These files verify the selected upstream forms with `mlir-opt` and FileCheck.
They are isolated prototypes, not a GenericGraphIR emitter.

## CLI

```bash
.venv/bin/python tools/check_generic_lowering_contract.py \
  --in artifacts/yoloseg_generic_frontend/yoloseg.shape_generic_graph_ir.json \
  --out artifacts/yoloseg_generic_frontend/yoloseg.lowering_contract.json
```
