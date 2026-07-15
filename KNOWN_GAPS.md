# Known Gaps

Last verified: 2026-07-14.

| Gap | Status | Publication wording |
|---|---|---|
| Universal policy engine | missing | P1D.1 and E3 are evidence-driven loops; many other decisions remain declared-profile, rule-based, shadow, or experimental. |
| Triton production integration | missing | Triton is shadow/provider-shaped with real artifacts, but not production Runtime dispatch. |
| Quantization scope | advanced partial | Static INT8 is materialized and executed for the validated fused Pi operator. Full-model accuracy, graph-wide mixed precision, INT4/AWQ/GPTQ, and NPU quantization remain absent. |
| AWQ plan consistency | known inconsistency | The positive AWQ plan has global AWQ/int4 intent while per-op entries contain fp16 fallback strategy text and int4 dtype fields. Do not claim complete INT4 support. |
| Capability DB canonicality | partial | Capability profiles are intended ownership, but compiler-local profiles remain richer and synchronization is incomplete. |
| Memory hierarchy / DMA / transfer model | missing | No mature memory-space/DMA/synchronization IR or bandwidth/transfer model. |
| NPU execution | missing | NPU profiles/plans are planning-only. |
| E2 correctness | invalid | Preserve invalid verdict. |
| E2.1 compiler-only interpretation | incorrect | Reclassify as implementation-stack comparison. |
| Fake `5.7 ms` ExecuTorch number | placeholder only | Must not appear as measured evidence. |
| General superiority claims | forbidden | No “beats ExecuTorch” or universal project/runtime superiority claim. |
