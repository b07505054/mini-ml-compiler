# Architecture Status

Last verified: 2026-07-14.

This table is a maturity assessment, not a roadmap promise and not a percentage score.

| Area | Status | Evidence | Limits |
|---|---|---|---|
| Portable Pi CPU execution | production/canonical + measured | P1B/P1C/P1D/P1D.1 reports, Runtime P1D evidence | fixed FP32 fused MatMul + Bias + ReLU scope |
| ImplementationCandidate architecture | canonical for active paths | A1-A5 tests/reports, E3 XNNPACK provider | not universal across every old decision system |
| Provider/feasibility/policy separation | production for portable path, evaluation-canonical for E3 | `PortableCPUProvider`, `XNNPACKCandidateProvider` | Triton/AWQ not production-integrated |
| E3 same-XNNPACK comparison | measured narrow comparison | runtime `results/executorch_e3` | static X1 winner, not complex learned policy |
| Triton provider | shadow | A6 report/tests | unresolved IR bridge, no Runtime dispatch |
| AWQ/vLLM | executable parallel path + measured serving | AWQ artifact, vLLM materialization, runtime traces | no accuracy/perplexity calibration, contradictory per-op/global plan details |
| Capability DB | partial declared source | `ml-platform-capabilities` profiles | not sole source of truth; compiler-local profiles are richer for Pi paths |
| Implementation IR | partial | HIR and limited boundary materialization | memory spaces, DMA, synchronization, NPU command regions incomplete |
| Runtime boundary | strong for canonical paths | strict adapter validation and E3 contract validation | older simulations/evaluation paths must stay scoped |

See `PROJECT_MATURITY.md` for the four-pillar assessment.
