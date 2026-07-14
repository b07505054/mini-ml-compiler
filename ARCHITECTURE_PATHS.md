# Architecture Paths

Last verified: 2026-07-14.

Status taxonomy: `production_canonical`, `evaluation_canonical`, `measured`, `calibrated`, `derived`, `predicted`, `shadow`, `experimental`, `historical`, `invalid`, `planned`, `missing`.

| Path | Status | Owner | Truth boundary |
|---|---|---|---|
| Portable Pi CPU fused MatMul + Bias + ReLU | production_canonical + measured | compiler/runtime | fixed FP32 op, Pi 5, portable native kernel |
| P1D.1 thread policy | production_canonical + calibrated + measured | compiler | threshold `262144` valid only for declared Pi/kernel/dtype/region |
| Portable provider/feasibility/policy/materialization | production_canonical | compiler | complete for current portable serial/X4 candidates only |
| XNNPACK provider and E3 contract | evaluation_canonical + measured | compiler/runtime evaluation | same-XNNPACK comparison path, not portable production fallback |
| E3 same-stack comparison | measured | runtime evidence | narrow Pi 5 FP32 fused-op comparison |
| Triton shadow provider | shadow + measured/predicted artifacts | compiler | unresolved IR bridge; no production ExecutionPlan effect |
| AWQ/vLLM | executable_parallel + measured serving | compiler/runtime | real artifact/materialization; no accuracy/perplexity calibration |
| Capability DB | declared_profile + partial | capabilities | intended source for declared facts, not sole source today |
| E1 ExecuTorch | historical measured smoke | runtime | bring-up only, not formal comparison |
| E2 ExecuTorch | historical invalid | runtime | correctness predicate invalidated comparison |
| E2.1 ExecuTorch | historical implementation-stack comparison | runtime | valid corrected correctness; not live-Compiler or same-stack comparison |
| NPU | missing/planned | compiler/capabilities | no executable NPU path |
| DMA/local memory | missing/planned | compiler | no mature IR or execution evidence |
| KV-cache compiler planning | future | runtime/compiler docs | do not present as Epoch 1 compiler path |
| Speculative planning | future | runtime docs | not part of Epoch 1 compiler publication |
