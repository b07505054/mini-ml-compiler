# Project Maturity

Last verified: 2026-07-14.

## Pillar 1 - Hardware Abstraction

Maturity: `ADVANCED_PARTIAL`.

Verified: Pi 5 CPU target profile, compute units, thread capability, kernel/runtime descriptors, target profile consumption, XNNPACK software/artifact requirements, GPU/Triton evidence paths.

Partial: Capability DB is not fully canonical/consumed; memory hierarchy is incomplete; software and hardware capability are only partially unified.

Missing: NPU, DMA, SRAM/local memory, bandwidth model, transfer model, heterogeneous partition capability.

## Pillar 2 - Decision-making Compiler

Maturity: `STRONGEST_PILLAR / ADVANCED_PARTIAL`.

Verified: Semantic IR, complete `ImplementationCandidate`, providers, feasibility, `PolicyResult`, materialization, portable threshold policy, XNNPACK static calibrated policy, Execution Contract, regret analysis, live Compiler same-stack comparison.

Partial: Many decisions remain outside one universal policy engine; Triton remains shadow; Implementation IR is incomplete; cross-backend ranking is incomplete.

Do not call the entire compiler universally evidence-driven. P1D.1 and E3 are evidence-driven; other decisions may be declared-profile, rule-based, shadow, or experimental.

## Pillar 3 - Quantization Co-design

Maturity: `EARLY_PARTIAL`.

Verified: real AutoAWQ artifact, real vLLM materialization, real serving evidence.

Partial: compiler metadata/planning exists; executable path exists outside the canonical candidate architecture.

Missing/broken: accuracy/perplexity calibration, unified FP16/INT8/INT4 candidates, canonical feasibility/policy, known AWQ plan inconsistency, NPU quantization.

## Pillar 4 - Hardware-Compiler Co-design

Maturity: `EARLY_PARTIAL`.

Verified: physical compute units constrain thread schedules; Pi measurements influence portable policy; Pi/XNNPACK measurements influence X1 policy.

Partial: target-specific software/hardware co-optimization exists.

Missing: hardware parameter sweep, SRAM sensitivity, bandwidth sensitivity, DMA, NPU architecture, precision-unit sensitivity, feedback to hardware design.
