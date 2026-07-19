# Publication Status

Last verified: 2026-07-14.

| Item | Category | Public wording |
|---|---|---|
| Portable Pi CPU execution | production/canonical + measured | Real Pi 5 FP32 fused-op path through compiler-selected portable native kernel. |
| P1D.1 thread policy | production/canonical + calibrated + measured | Offline-calibrated static threshold policy for one Pi/kernel/dtype/region contract. |
| XNNPACK provider | evaluation/canonical comparison path + measured | Live Compiler can enumerate X1/X4 and emit a validated same-stack comparison contract. |
| E3 same-stack comparison | measured narrow scope | Static X1 was no worse than default and faster on two held-out workloads under the preregistered tie rule. |
| Triton provider | shadow + measured/predicted artifacts | Real artifact-backed shadow records; unresolved IR bridge; no production dispatch. |
| AWQ serving | executable parallel path + measured serving | Real AWQ artifact and vLLM materialization/serving traces exist. |
| AWQ accuracy | missing | No perplexity/task accuracy calibration. |
| E2 | historical invalid | Correctness predicate invalidated comparison. |
| E2.1 | historical implementation-stack comparison | Valid corrected correctness; not live-Compiler or same-stack comparison. |
| Capability DB | declared-profile partial | Intended source for declared facts; not sole source yet. |
| NPU | missing | Planning-only concepts, no executable path. |
| DMA/local memory | missing | No mature compiler IR or runtime evidence. |
| KV-cache compiler planning | future | Not part of Epoch 1 publication claim. |
| Speculative planning | future | Not part of Epoch 1 publication claim. |
