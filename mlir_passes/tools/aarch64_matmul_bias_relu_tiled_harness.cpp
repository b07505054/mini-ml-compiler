// aarch64_matmul_bias_relu_tiled_harness.cpp
//
// Correctness + benchmark harness for the tiled-vectorized AArch64
// microkernel slice. Compares up to five implementations of the same fused
// MatMul-Bias-ReLU computation, for one shape per process (see the note on
// process isolation below, inherited from the Stage 1/Stage 2 harnesses):
//
//   1. scalar C++ reference (embedded in this file)
//   2. generic compiler-generated kernel (--variant generic)
//   3. fully-unrolled vectorized compiler-generated kernel (--variant
//      vectorized) -- ONLY available for 8x8x8/16x16x16/32x32x32; this
//      variant was never generated for 64x64x64/32x64x32/64x32x64 because
//      its object size is impractical there (see
//      artifacts/backend_codegen/aarch64_matmul_bias_relu_tiled/README.md).
//      Reported as "available": false for the other three shapes rather
//      than fabricating a number.
//   4. tiled-vectorized compiler-generated kernel (--variant tiled-vectorized)
//   5. handwritten kernel (src/kernels/cpu_kernels.cpp fused_matmul_add_relu)
//
// ABI: identical MemRef2D / _mlir_ciface_<fn> convention as every prior
// harness in this series -- read off the generated LLVM IR, not guessed.
//
// Process isolation: each shape's full correctness+benchmark run happens in
// its own process (see run_backend_codegen_tiled_pi_integration.sh, which
// invokes this binary once per shape) -- the same hardware-validated
// requirement discovered in the Stage 1 slice.
//
// No NEON intrinsics are hand-written anywhere in this file. All generated
// kernels' NEON code is entirely LLVM-generated from MLIR IR; this harness
// only calls it through the plain C ABI described above.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kernels/cpu_kernels.h"  // handwritten reference: fused_matmul_add_relu
#include "ir/tensor.h"

extern "C" {
struct MemRef2D {
  float* allocated;
  float* aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

// Generic variant -- all six shapes.
void _mlir_ciface_matmul_bias_relu_8x8x8(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_16x16x16(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_32x32x32(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_64x64x64(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_32x64x32(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_64x32x64(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);

// Fully-unrolled vectorized variant -- only the three original square shapes.
void _mlir_ciface_matmul_bias_relu_vectorized_8x8x8(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_vectorized_16x16x16(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_vectorized_32x32x32(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);

// Tiled-vectorized variant -- all six shapes.
void _mlir_ciface_matmul_bias_relu_tiled_8x8x8(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_tiled_16x16x16(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_tiled_32x32x32(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_tiled_64x64x64(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_tiled_32x64x32(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_tiled_64x32x64(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
}

using GeneratedFn = void (*)(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);

namespace {

constexpr int64_t kMaxTimedIterations = 20000;
double g_scalarTimesBuf[kMaxTimedIterations];
double g_genericTimesBuf[kMaxTimedIterations];
double g_vectorizedTimesBuf[kMaxTimedIterations];
double g_tiledTimesBuf[kMaxTimedIterations];
double g_handwrittenTimesBuf[kMaxTimedIterations];

MemRef2D makeDescriptor(std::vector<float>& buf, int64_t rows, int64_t cols) {
  MemRef2D d{};
  d.allocated = buf.data();
  d.aligned = buf.data();
  d.offset = 0;
  d.sizes[0] = rows;
  d.sizes[1] = cols;
  d.strides[0] = cols;
  d.strides[1] = 1;
  return d;
}

void fillDeterministic(std::vector<float>& v, uint32_t seed) {
  uint32_t state = seed;
  for (auto& x : v) {
    state = state * 1664525u + 1013904223u;
    x = static_cast<float>((state >> 8) & 0xFFFFu) / 65536.0f - 0.5f;
  }
}

void scalarReferenceMatMulBiasRelu(const std::vector<float>& lhs,
                                    const std::vector<float>& rhs,
                                    const std::vector<float>& bias,
                                    std::vector<float>& out, int64_t M,
                                    int64_t N, int64_t K) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        acc += lhs[i * K + k] * rhs[k * N + j];
      }
      float v = acc + bias[i * N + j];
      out[i * N + j] = v > 0.0f ? v : 0.0f;
    }
  }
}

uint64_t checksumBits(const std::vector<float>& v) {
  uint64_t h = 1469598103934665603ull;
  for (float x : v) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    h ^= bits;
    h *= 1099511628211ull;
  }
  return h;
}

double percentile(const double* buf, int64_t count, double p) {
  std::vector<double> v(buf, buf + count);
  std::sort(v.begin(), v.end());
  size_t idx = static_cast<size_t>(p * static_cast<double>(v.size() - 1));
  return v[idx];
}

struct Stats {
  double medianMs = 0.0;
  double p95Ms = 0.0;
};

Stats summarize(const double* buf, int64_t count) {
  Stats s;
  s.medianMs = percentile(buf, count, 0.5);
  s.p95Ms = percentile(buf, count, 0.95);
  return s;
}

Stats timeFn(GeneratedFn fn, MemRef2D& lhsDesc, MemRef2D& rhsDesc,
             MemRef2D& biasDesc, int64_t warmup, int64_t iterations,
             double* buf) {
  for (int64_t i = 0; i < warmup; ++i) {
    MemRef2D o{};
    fn(&o, &lhsDesc, &rhsDesc, &biasDesc);
    std::free(o.allocated);
  }
  for (int64_t i = 0; i < iterations; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    MemRef2D o{};
    fn(&o, &lhsDesc, &rhsDesc, &biasDesc);
    auto t1 = std::chrono::steady_clock::now();
    buf[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::free(o.allocated);
  }
  return summarize(buf, iterations);
}

}  // namespace

int main(int argc, char** argv) {
  int64_t iterations = 2000;
  int64_t warmup = 200;
  std::string shapeName = "32x32x32";
  if (argc > 1) iterations = std::atoll(argv[1]);
  if (argc > 2) warmup = std::atoll(argv[2]);
  if (argc > 3) shapeName = argv[3];

  if (iterations > kMaxTimedIterations) {
    std::fprintf(stderr, "warning: clamping iterations to %lld\n",
                 static_cast<long long>(kMaxTimedIterations));
    iterations = kMaxTimedIterations;
  }

  int64_t M, N, K;
  GeneratedFn genericFn = nullptr;
  GeneratedFn vectorizedFn = nullptr;  // may stay null (not available for this shape)
  GeneratedFn tiledFn = nullptr;

  if (shapeName == "8x8x8") {
    M = N = K = 8;
    genericFn = _mlir_ciface_matmul_bias_relu_8x8x8;
    vectorizedFn = _mlir_ciface_matmul_bias_relu_vectorized_8x8x8;
    tiledFn = _mlir_ciface_matmul_bias_relu_tiled_8x8x8;
  } else if (shapeName == "16x16x16") {
    M = N = K = 16;
    genericFn = _mlir_ciface_matmul_bias_relu_16x16x16;
    vectorizedFn = _mlir_ciface_matmul_bias_relu_vectorized_16x16x16;
    tiledFn = _mlir_ciface_matmul_bias_relu_tiled_16x16x16;
  } else if (shapeName == "32x32x32") {
    M = N = K = 32;
    genericFn = _mlir_ciface_matmul_bias_relu_32x32x32;
    vectorizedFn = _mlir_ciface_matmul_bias_relu_vectorized_32x32x32;
    tiledFn = _mlir_ciface_matmul_bias_relu_tiled_32x32x32;
  } else if (shapeName == "64x64x64") {
    M = N = K = 64;
    genericFn = _mlir_ciface_matmul_bias_relu_64x64x64;
    tiledFn = _mlir_ciface_matmul_bias_relu_tiled_64x64x64;
  } else if (shapeName == "32x64x32") {
    M = 32; N = 64; K = 32;
    genericFn = _mlir_ciface_matmul_bias_relu_32x64x32;
    tiledFn = _mlir_ciface_matmul_bias_relu_tiled_32x64x32;
  } else if (shapeName == "64x32x64") {
    M = 64; N = 32; K = 64;
    genericFn = _mlir_ciface_matmul_bias_relu_64x32x64;
    tiledFn = _mlir_ciface_matmul_bias_relu_tiled_64x32x64;
  } else {
    std::fprintf(stderr, "error: unknown shape '%s'\n", shapeName.c_str());
    return 2;
  }

  std::vector<float> lhs(static_cast<size_t>(M * K));
  std::vector<float> rhs(static_cast<size_t>(K * N));
  std::vector<float> bias(static_cast<size_t>(M * N));
  std::vector<float> refOut(static_cast<size_t>(M * N), 0.0f);

  fillDeterministic(lhs, 0x1234u + static_cast<uint32_t>(M));
  fillDeterministic(rhs, 0x5678u + static_cast<uint32_t>(N));
  fillDeterministic(bias, 0x9abcu + static_cast<uint32_t>(K));

  scalarReferenceMatMulBiasRelu(lhs, rhs, bias, refOut, M, N, K);
  uint64_t refChecksum = checksumBits(refOut);

  MemRef2D lhsDesc = makeDescriptor(lhs, M, K);
  MemRef2D rhsDesc = makeDescriptor(rhs, K, N);
  MemRef2D biasDesc = makeDescriptor(bias, M, N);

  auto runOnce = [&](GeneratedFn fn) -> std::vector<float> {
    MemRef2D out{};
    fn(&out, &lhsDesc, &rhsDesc, &biasDesc);
    std::vector<float> result(out.aligned, out.aligned + M * N);
    std::free(out.allocated);
    return result;
  };

  auto maxAbsError = [&](const std::vector<float>& a) {
    double maxErr = 0.0;
    for (int64_t i = 0; i < M * N; ++i) {
      maxErr = std::max(maxErr, static_cast<double>(std::fabs(
                                     a[static_cast<size_t>(i)] -
                                     refOut[static_cast<size_t>(i)])));
    }
    return maxErr;
  };

  std::vector<float> genericOut = runOnce(genericFn);
  double genericErr = maxAbsError(genericOut);
  bool genericCorrect = genericErr < 1e-3;

  bool vectorizedAvailable = vectorizedFn != nullptr;
  std::vector<float> vectorizedOut;
  double vectorizedErr = 0.0;
  bool vectorizedCorrect = true;  // vacuously true when not available
  if (vectorizedAvailable) {
    vectorizedOut = runOnce(vectorizedFn);
    vectorizedErr = maxAbsError(vectorizedOut);
    vectorizedCorrect = vectorizedErr < 1e-3;
  }

  std::vector<float> tiledOut = runOnce(tiledFn);
  double tiledErr = maxAbsError(tiledOut);
  bool tiledCorrect = tiledErr < 1e-3;

  Tensor hwA("A", {static_cast<int>(M), static_cast<int>(K)});
  Tensor hwB("B", {static_cast<int>(K), static_cast<int>(N)});
  Tensor hwBias("Bias", {static_cast<int>(M), static_cast<int>(N)});
  Tensor hwOut("Out", {static_cast<int>(M), static_cast<int>(N)});
  hwA.data = lhs;
  hwB.data = rhs;
  hwBias.data = bias;
  fused_matmul_add_relu(hwA, hwB, hwBias, hwOut);
  double handwrittenErr = maxAbsError(hwOut.data);
  bool handwrittenCorrect = handwrittenErr < 1e-3;

  bool allCorrect = genericCorrect && vectorizedCorrect && tiledCorrect && handwrittenCorrect;

  // ---- Benchmark: warmup + timed, static storage throughout ----
  Stats scalarStats, genericStats, vectorizedStats, tiledStats, handwrittenStats;

  genericStats = timeFn(genericFn, lhsDesc, rhsDesc, biasDesc, warmup, iterations, g_genericTimesBuf);

  if (vectorizedAvailable) {
    vectorizedStats = timeFn(vectorizedFn, lhsDesc, rhsDesc, biasDesc, warmup, iterations, g_vectorizedTimesBuf);
  }

  tiledStats = timeFn(tiledFn, lhsDesc, rhsDesc, biasDesc, warmup, iterations, g_tiledTimesBuf);

  std::vector<float> scratch(static_cast<size_t>(M * N));
  for (int64_t i = 0; i < warmup; ++i) {
    scalarReferenceMatMulBiasRelu(lhs, rhs, bias, scratch, M, N, K);
  }
  for (int64_t i = 0; i < iterations; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    scalarReferenceMatMulBiasRelu(lhs, rhs, bias, scratch, M, N, K);
    auto t1 = std::chrono::steady_clock::now();
    g_scalarTimesBuf[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  scalarStats = summarize(g_scalarTimesBuf, iterations);

  Tensor hwScratchOut("HwScratch", {static_cast<int>(M), static_cast<int>(N)});
  for (int64_t i = 0; i < warmup; ++i) {
    fused_matmul_add_relu(hwA, hwB, hwBias, hwScratchOut);
  }
  for (int64_t i = 0; i < iterations; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    fused_matmul_add_relu(hwA, hwB, hwBias, hwScratchOut);
    auto t1 = std::chrono::steady_clock::now();
    g_handwrittenTimesBuf[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  handwrittenStats = summarize(g_handwrittenTimesBuf, iterations);

  std::printf("{\n");
  std::printf("  \"shape\": \"%s\",\n", shapeName.c_str());
  std::printf("  \"M\": %lld, \"N\": %lld, \"K\": %lld,\n",
              static_cast<long long>(M), static_cast<long long>(N),
              static_cast<long long>(K));
  std::printf("  \"iterations\": %lld,\n", static_cast<long long>(iterations));
  std::printf("  \"correctness\": {\n");
  std::printf("    \"reference_checksum\": \"0x%016llx\",\n",
              static_cast<unsigned long long>(refChecksum));
  std::printf("    \"generic\": {\"correct\": %s, \"max_abs_error\": %.8f, \"checksum\": \"0x%016llx\"},\n",
              genericCorrect ? "true" : "false", genericErr,
              static_cast<unsigned long long>(checksumBits(genericOut)));
  if (vectorizedAvailable) {
    std::printf("    \"fully_unrolled_vectorized\": {\"available\": true, \"correct\": %s, \"max_abs_error\": %.8f, \"checksum\": \"0x%016llx\"},\n",
                vectorizedCorrect ? "true" : "false", vectorizedErr,
                static_cast<unsigned long long>(checksumBits(vectorizedOut)));
  } else {
    std::printf("    \"fully_unrolled_vectorized\": {\"available\": false},\n");
  }
  std::printf("    \"tiled_vectorized\": {\"correct\": %s, \"max_abs_error\": %.8f, \"checksum\": \"0x%016llx\"},\n",
              tiledCorrect ? "true" : "false", tiledErr,
              static_cast<unsigned long long>(checksumBits(tiledOut)));
  std::printf("    \"handwritten\": {\"correct\": %s, \"max_abs_error\": %.8f, \"checksum\": \"0x%016llx\"},\n",
              handwrittenCorrect ? "true" : "false", handwrittenErr,
              static_cast<unsigned long long>(checksumBits(hwOut.data)));
  std::printf("    \"all_correct\": %s\n", allCorrect ? "true" : "false");
  std::printf("  },\n");
  std::printf("  \"benchmark_ms\": {\n");
  std::printf("    \"scalar\":                  {\"median\": %.6f, \"p95\": %.6f},\n", scalarStats.medianMs, scalarStats.p95Ms);
  std::printf("    \"generic\":                 {\"median\": %.6f, \"p95\": %.6f},\n", genericStats.medianMs, genericStats.p95Ms);
  if (vectorizedAvailable) {
    std::printf("    \"fully_unrolled_vectorized\": {\"median\": %.6f, \"p95\": %.6f},\n", vectorizedStats.medianMs, vectorizedStats.p95Ms);
  } else {
    std::printf("    \"fully_unrolled_vectorized\": null,\n");
  }
  std::printf("    \"tiled_vectorized\":        {\"median\": %.6f, \"p95\": %.6f},\n", tiledStats.medianMs, tiledStats.p95Ms);
  std::printf("    \"handwritten\":             {\"median\": %.6f, \"p95\": %.6f}\n", handwrittenStats.medianMs, handwrittenStats.p95Ms);
  std::printf("  },\n");
  std::printf("  \"tiled_over_generic_median_speedup\": %.6f,\n",
              genericStats.medianMs / tiledStats.medianMs);
  if (vectorizedAvailable) {
    std::printf("  \"tiled_over_fully_unrolled_median_ratio\": %.6f,\n",
                tiledStats.medianMs / vectorizedStats.medianMs);
  } else {
    std::printf("  \"tiled_over_fully_unrolled_median_ratio\": null,\n");
  }
  std::printf("  \"tiled_over_scalar_median_ratio\": %.6f,\n",
              tiledStats.medianMs / scalarStats.medianMs);
  std::printf("  \"tiled_over_handwritten_median_ratio\": %.6f\n",
              tiledStats.medianMs / handwrittenStats.medianMs);
  std::printf("}\n");

  return allCorrect ? 0 : 1;
}
