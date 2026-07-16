// aarch64_matmul_bias_relu_vectorized_harness.cpp
//
// Correctness + benchmark harness for the vectorized AArch64 backend slice.
// Compares four implementations of the same fused MatMul-Bias-ReLU
// computation, for one shape per process (see the note on process isolation
// below):
//
//   1. scalar C++ reference (embedded in this file)
//   2. generic compiler-generated kernel (mlir_passes/tools/
//      compile_hir_matmul_bias_relu_aarch64.sh --variant generic)
//   3. vectorized compiler-generated kernel (--variant vectorized)
//   4. handwritten kernel (src/kernels/cpu_kernels.cpp fused_matmul_add_relu)
//
// ABI for the two compiler-generated kernels: identical to the Stage 1
// harness (aarch64_matmul_bias_relu_harness.cpp) -- read off the generated
// LLVM IR for a function carrying llvm.emit_c_interface, not guessed. Both
// the generic and vectorized variants produce byte-identical
// _mlir_ciface_<fn>(MemRef2D* out, MemRef2D* lhs, MemRef2D* rhs,
// MemRef2D* bias) signatures (verified: the vectorized variant's
// function-boundary bufferization option changes only the argument memref's
// internal layout map, not the ABI-visible descriptor shape), so this
// harness calls both through the same MemRef2D struct with no special-casing.
//
// Process isolation: like the Stage 1 harness, each shape's full
// correctness+benchmark run happens in its own process (see
// run_backend_codegen_vectorized_pi_integration.sh, which invokes this
// binary once per shape). This was a real, hardware-validated requirement
// discovered in Stage 1 -- see that harness's header comment.
//
// No NEON intrinsics are hand-written anywhere in this file. The vectorized
// kernel's NEON code is entirely LLVM-generated from MLIR vector-dialect IR;
// this harness only calls it through the plain C ABI described above.

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

// Generic variant (matches Stage 1 symbol names exactly).
void _mlir_ciface_matmul_bias_relu_8x8x8(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_16x16x16(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_32x32x32(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);

// Vectorized variant.
void _mlir_ciface_matmul_bias_relu_vectorized_8x8x8(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_vectorized_16x16x16(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_vectorized_32x32x32(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
}

using GeneratedFn = void (*)(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);

namespace {

// Static (non-heap) timing storage -- see Stage 1 harness header comment for
// why this must not be a heap allocation that coexists with the benchmarked
// loop's own malloc/free traffic.
constexpr int64_t kMaxTimedIterations = 20000;
double g_scalarTimesBuf[kMaxTimedIterations];
double g_genericTimesBuf[kMaxTimedIterations];
double g_vectorizedTimesBuf[kMaxTimedIterations];
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

}  // namespace

int main(int argc, char** argv) {
  int64_t iterations = 2000;
  int64_t warmup = 200;
  std::string shapeName = "16x16x16";
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
  GeneratedFn vectorizedFn = nullptr;
  if (shapeName == "8x8x8") {
    M = N = K = 8;
    genericFn = _mlir_ciface_matmul_bias_relu_8x8x8;
    vectorizedFn = _mlir_ciface_matmul_bias_relu_vectorized_8x8x8;
  } else if (shapeName == "16x16x16") {
    M = N = K = 16;
    genericFn = _mlir_ciface_matmul_bias_relu_16x16x16;
    vectorizedFn = _mlir_ciface_matmul_bias_relu_vectorized_16x16x16;
  } else if (shapeName == "32x32x32") {
    M = N = K = 32;
    genericFn = _mlir_ciface_matmul_bias_relu_32x32x32;
    vectorizedFn = _mlir_ciface_matmul_bias_relu_vectorized_32x32x32;
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

  // ---- Correctness: single call per implementation ----
  auto runOnce = [&](GeneratedFn fn) -> std::vector<float> {
    MemRef2D out{};
    fn(&out, &lhsDesc, &rhsDesc, &biasDesc);
    std::vector<float> result(out.aligned, out.aligned + M * N);
    std::free(out.allocated);
    return result;
  };

  std::vector<float> genericOut = runOnce(genericFn);
  std::vector<float> vectorizedOut = runOnce(vectorizedFn);

  // Handwritten reference via the project's own Tensor type.
  Tensor hwA("A", {static_cast<int>(M), static_cast<int>(K)});
  Tensor hwB("B", {static_cast<int>(K), static_cast<int>(N)});
  Tensor hwBias("Bias", {static_cast<int>(M), static_cast<int>(N)});
  Tensor hwOut("Out", {static_cast<int>(M), static_cast<int>(N)});
  hwA.data = lhs;
  hwB.data = rhs;
  hwBias.data = bias;
  fused_matmul_add_relu(hwA, hwB, hwBias, hwOut);

  auto maxAbsError = [&](const std::vector<float>& a) {
    double maxErr = 0.0;
    for (int64_t i = 0; i < M * N; ++i) {
      maxErr = std::max(maxErr, static_cast<double>(std::fabs(
                                     a[static_cast<size_t>(i)] -
                                     refOut[static_cast<size_t>(i)])));
    }
    return maxErr;
  };

  double genericErr = maxAbsError(genericOut);
  double vectorizedErr = maxAbsError(vectorizedOut);
  double handwrittenErr = maxAbsError(hwOut.data);
  bool genericCorrect = genericErr < 1e-3;
  bool vectorizedCorrect = vectorizedErr < 1e-3;
  bool handwrittenCorrect = handwrittenErr < 1e-3;
  bool allCorrect = genericCorrect && vectorizedCorrect && handwrittenCorrect;

  // ---- Benchmark: warmup + timed, static storage throughout ----
  for (int64_t i = 0; i < warmup; ++i) {
    MemRef2D o{};
    genericFn(&o, &lhsDesc, &rhsDesc, &biasDesc);
    std::free(o.allocated);
  }
  for (int64_t i = 0; i < iterations; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    MemRef2D o{};
    genericFn(&o, &lhsDesc, &rhsDesc, &biasDesc);
    auto t1 = std::chrono::steady_clock::now();
    g_genericTimesBuf[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::free(o.allocated);
  }

  for (int64_t i = 0; i < warmup; ++i) {
    MemRef2D o{};
    vectorizedFn(&o, &lhsDesc, &rhsDesc, &biasDesc);
    std::free(o.allocated);
  }
  for (int64_t i = 0; i < iterations; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    MemRef2D o{};
    vectorizedFn(&o, &lhsDesc, &rhsDesc, &biasDesc);
    auto t1 = std::chrono::steady_clock::now();
    g_vectorizedTimesBuf[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::free(o.allocated);
  }

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

  Stats scalarStats = summarize(g_scalarTimesBuf, iterations);
  Stats genericStats = summarize(g_genericTimesBuf, iterations);
  Stats vectorizedStats = summarize(g_vectorizedTimesBuf, iterations);
  Stats handwrittenStats = summarize(g_handwrittenTimesBuf, iterations);

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
  std::printf("    \"vectorized\": {\"correct\": %s, \"max_abs_error\": %.8f, \"checksum\": \"0x%016llx\"},\n",
              vectorizedCorrect ? "true" : "false", vectorizedErr,
              static_cast<unsigned long long>(checksumBits(vectorizedOut)));
  std::printf("    \"handwritten\": {\"correct\": %s, \"max_abs_error\": %.8f, \"checksum\": \"0x%016llx\"},\n",
              handwrittenCorrect ? "true" : "false", handwrittenErr,
              static_cast<unsigned long long>(checksumBits(hwOut.data)));
  std::printf("    \"all_correct\": %s\n", allCorrect ? "true" : "false");
  std::printf("  },\n");
  std::printf("  \"benchmark_ms\": {\n");
  std::printf("    \"scalar\":      {\"median\": %.6f, \"p95\": %.6f},\n", scalarStats.medianMs, scalarStats.p95Ms);
  std::printf("    \"generic\":     {\"median\": %.6f, \"p95\": %.6f},\n", genericStats.medianMs, genericStats.p95Ms);
  std::printf("    \"vectorized\":  {\"median\": %.6f, \"p95\": %.6f},\n", vectorizedStats.medianMs, vectorizedStats.p95Ms);
  std::printf("    \"handwritten\": {\"median\": %.6f, \"p95\": %.6f}\n", handwrittenStats.medianMs, handwrittenStats.p95Ms);
  std::printf("  },\n");
  std::printf("  \"vectorized_over_generic_median_speedup\": %.6f,\n",
              genericStats.medianMs / vectorizedStats.medianMs);
  std::printf("  \"vectorized_over_scalar_median_ratio\": %.6f,\n",
              vectorizedStats.medianMs / scalarStats.medianMs);
  std::printf("  \"vectorized_over_handwritten_median_ratio\": %.6f\n",
              vectorizedStats.medianMs / handwrittenStats.medianMs);
  std::printf("}\n");

  return allCorrect ? 0 : 1;
}
