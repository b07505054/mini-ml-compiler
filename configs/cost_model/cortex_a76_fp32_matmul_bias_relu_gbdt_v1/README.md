# Cortex-A76 FP32 MatMul–Bias–ReLU GBDT V1

This model predicts latency for candidates that have already passed deterministic
compiler legality, lowering-completeness, target, code-size, and spill checks.
It does not generate candidates or make illegal implementations executable.

## Candidate semantic decomposition

| Candidate | Tiling | Vectorization | Padding | M-tail | N-tail | K-tail | Full-tile constraints |
|---|---|---|---|---|---|---|---|
| FusedScalar | none | none | none | scalar schedule | scalar schedule | scalar schedule | none |
| WholeShapeVectorNoPadding | whole shape | whole-shape vector | none | represented by whole shape | represented by whole shape | represented by whole shape | code-size/resource contract, not 8-alignment |
| WholeShapeVectorMaterializedPadding | whole shape | whole-shape vector | whole-shape materialized | materialized | materialized | materialized | none after padded shape is formed |
| TiledVectorFullTiles | 8×8×8 | tiled vector, multiple dimensions | none | none | none | none | M, N, and K must be full tiles |
| TiledVectorMaterializedTail | 8×8×8 | tiled vector, multiple dimensions | tile-local materialized | materialized when present | materialized when present | materialized when present | none; temporary/copy/zero-fill handles remainders |
| TiledVectorDirectCleanup | 8×8×8 | tiled vector, multiple dimensions | none | unsupported | unsupported | direct vector cleanup | full M and N tiles; K remainder required |

`DirectVector` K-tail is direct work on the original A/B tensors. It does not
mean hardware padding: complete K tiles retain the fixed vector microkernel,
while the K remainder updates the same accumulator without padded source
temporaries, zero-fill, or source copies.

Strategy decomposition is descriptive input to the latency model. It is not
permission to synthesize combinations absent from the production candidate
registry.
