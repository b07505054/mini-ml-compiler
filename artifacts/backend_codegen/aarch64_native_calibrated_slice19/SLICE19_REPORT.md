# Slice 19: fresh calibrated AArch64 schedule selection

Scope is exactly `hir.fused_matmul_bias_relu`, f32 32x32x32, Cortex-A76,
tile 8x8x8, 128-bit vectors, and UK 1/2/4.  This is not general autotuning.

## Result

Fresh exact-target calibration changed selection from `tile8x8x8_uk1` to
`tile8x8x8_uk4`.

| candidate | median session p50 (ms) | median session p95 (ms) | mean session mean (ms) | CI95 of session mean (ms) |
|---|---:|---:|---:|---:|
| uk1 | 0.004373420 | 0.004390645 | 0.004376332 | [0.004375416, 0.004377282] |
| uk2 | 0.004148885 | 0.004166665 | 0.004151494 | [0.004148125, 0.004153772] |
| uk4 | 0.003856660 | 0.003872310 | 0.003858395 | [0.003856528, 0.003860261] |

UK4 reduced median p50 latency by 11.816% versus uk1 (1.1340x speedup) and
7.043% versus uk2 (1.0758x speedup).  The session-level intervals do not
overlap and the differences exceed the serialized 3% equivalence threshold.

All 18 calibration executions passed numerical correctness, repeated-call
correctness, guard-buffer checks, object/candidate/entry-point identity, and
zero Runtime redecision.  Each candidate received six independent sessions,
each with 30 warmup samples and 1,000 measured samples of 100 calls: 600,000
measured calls per candidate.

The independent confirmation of uk4 used another 100,000 calls and reported
p50 0.003851290 ms and p95 0.003868700 ms, with zero maximum error, intact
guards, exact object/entry-point identity, and zero redecision.  Its p50 is
0.139% below the calibration median and therefore consistent.

Static evidence still records uk4's larger 3,128-byte `.text`, 11 spill
stores, 12 reload loads, and 176 spill-slot bytes.  The measured result is
retained honestly: those costs did not prevent uk4 from winning this fixed
workload on this exact Raspberry Pi target.
