# Decode Attention KV-Cache Bandwidth Model

This model is a GEMM-adjacent attention case study for serving-time decode.
It focuses on KV-cache read bandwidth rather than implementing a full FlashAttention kernel.

| Context | Heads | Head dim | KV bytes/token | FLOPs/token | AI FLOPs/byte | Blocks | Bottleneck |
|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | 16 | 64 | 524288 | 524288 | 0.992 | 8 | kv_cache_memory_bandwidth |
| 512 | 16 | 64 | 2097152 | 2097152 | 0.998 | 32 | kv_cache_memory_bandwidth |
| 2048 | 16 | 64 | 8388608 | 8388608 | 1.000 | 128 | kv_cache_memory_bandwidth |
| 4096 | 16 | 64 | 16777216 | 16777216 | 1.000 | 256 | kv_cache_memory_bandwidth |

## Compiler/Runtime Relevance

- Compiler lowering can select layouts and tile sizes that preserve coalesced KV reads.
- Runtime scheduling must account for KV-cache memory pressure during decode.
- Paged KV improves reuse/admission behavior but introduces block lookup and partial-block overhead.
- This model explains why attention decode is often bandwidth-bound even when compute kernels are optimized.
