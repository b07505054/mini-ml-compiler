// aarch64_matmul_bias_relu_repeated_call_test.cpp
//
// Repeated-call correctness regression test for the vectorized AArch64
// backend variant. Added after a real bug was found and fixed: the shared
// hir-matmul-bias-relu-to-linalg lowering fed linalg.matmul an
// un-zero-initialized tensor.empty() accumulator, so on heap-address reuse
// across repeated calls, the matmul result silently accumulated across
// calls instead of starting fresh (output on call N equaled (N+1) x the
// correct value for affected elements). Fixed by explicit linalg.fill.
// This test guards against that class of regression recurring, for both
// the generic and vectorized variants -- see
// aarch64_matmul_bias_relu_mixed_shape_test.cpp for the same-process,
// mixed-shape/mixed-variant companion stress test.
//
// Usage: ./aarch64_matmul_bias_relu_repeated_call_test <shape> <num_calls>
//
// Every caller-owned input buffer (lhs/rhs/bias) is allocated with a guard
// prefix and suffix filled with a deterministic sentinel float pattern.
// After every call, guards are checked for corruption, the output is
// checked against a scalar reference, and the returned descriptor fields
// are sanity-checked (sizes/strides/offset/allocated-vs-aligned).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

extern "C" {
struct MemRef2D {
  float* allocated;
  float* aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};
void _mlir_ciface_matmul_bias_relu_vectorized_8x8x8(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_vectorized_16x16x16(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_vectorized_32x32x32(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_8x8x8(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_16x16x16(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_32x32x32(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
}

using GeneratedFn = void (*)(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);

namespace {

constexpr int64_t kGuardElems = 64;         // 256 bytes of guard on each side
constexpr uint32_t kSentinelBits = 0xDEADBEEFu;

float sentinelValue(int index) {
  uint32_t bits = kSentinelBits ^ static_cast<uint32_t>(index * 2654435761u);
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// A guarded buffer: [prefix guard][payload][suffix guard], all one
// contiguous allocation so any overrun in either direction is directly
// adjacent to payload data (no intervening allocator metadata to obscure
// small overruns).
struct GuardedBuffer {
  std::vector<float> storage;
  int64_t payloadElems;

  explicit GuardedBuffer(int64_t elems) : storage(static_cast<size_t>(elems + 2 * kGuardElems)), payloadElems(elems) {
    for (int64_t i = 0; i < kGuardElems; ++i) {
      storage[static_cast<size_t>(i)] = sentinelValue(static_cast<int>(i));
      storage[static_cast<size_t>(kGuardElems + payloadElems + i)] = sentinelValue(static_cast<int>(1000 + i));
    }
  }

  float* payload() { return storage.data() + kGuardElems; }

  // Bit-exact comparison -- sentinel values may legitimately be NaN bit
  // patterns, and IEEE 754 NaN != NaN even when the bits are identical, so
  // a plain float `!=` produces false positives. Compare raw bits instead.
  static bool bitsEqual(float a, float b) {
    uint32_t ba, bb;
    std::memcpy(&ba, &a, sizeof(ba));
    std::memcpy(&bb, &b, sizeof(bb));
    return ba == bb;
  }

  // Returns index of first corrupted guard element, or -1 if clean.
  int checkGuards(const char* label) {
    for (int64_t i = 0; i < kGuardElems; ++i) {
      float expected = sentinelValue(static_cast<int>(i));
      if (!bitsEqual(storage[static_cast<size_t>(i)], expected)) {
        std::fprintf(stderr, "GUARD CORRUPTION: %s prefix[%lld] expected_bits=%08x got_bits=%08x\n",
                     label, static_cast<long long>(i), bitsOf(expected), bitsOf(storage[static_cast<size_t>(i)]));
        return static_cast<int>(i - kGuardElems);
      }
    }
    for (int64_t i = 0; i < kGuardElems; ++i) {
      float expected = sentinelValue(static_cast<int>(1000 + i));
      size_t idx = static_cast<size_t>(kGuardElems + payloadElems + i);
      if (!bitsEqual(storage[idx], expected)) {
        std::fprintf(stderr, "GUARD CORRUPTION: %s suffix[%lld] expected_bits=%08x got_bits=%08x\n",
                     label, static_cast<long long>(i), bitsOf(expected), bitsOf(storage[idx]));
        return static_cast<int>(payloadElems + i);
      }
    }
    return -1;
  }

  static uint32_t bitsOf(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
  }
};

void fillDeterministic(float* p, int64_t n, uint32_t seed) {
  uint32_t state = seed;
  for (int64_t i = 0; i < n; ++i) {
    state = state * 1664525u + 1013904223u;
    p[i] = static_cast<float>((state >> 8) & 0xFFFFu) / 65536.0f - 0.5f;
  }
}

void scalarRef(const float* lhs, const float* rhs, const float* bias, float* out,
               int64_t M, int64_t N, int64_t K) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) acc += lhs[i * K + k] * rhs[k * N + j];
      float v = acc + bias[i * N + j];
      out[i * N + j] = v > 0.0f ? v : 0.0f;
    }
  }
}

MemRef2D makeDescriptor(float* payload, int64_t rows, int64_t cols) {
  MemRef2D d{};
  d.allocated = payload;
  d.aligned = payload;
  d.offset = 0;
  d.sizes[0] = rows;
  d.sizes[1] = cols;
  d.strides[0] = cols;
  d.strides[1] = 1;
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <shape> <num_calls> [variant: generic|vectorized]\n", argv[0]);
    return 2;
  }
  std::string shape = argv[1];
  int64_t numCalls = std::atoll(argv[2]);
  std::string variant = argc > 3 ? argv[3] : "vectorized";

  int64_t M, N, K;
  GeneratedFn fn = nullptr;
  if (variant == "vectorized") {
    if (shape == "8x8x8") { M = N = K = 8; fn = _mlir_ciface_matmul_bias_relu_vectorized_8x8x8; }
    else if (shape == "16x16x16") { M = N = K = 16; fn = _mlir_ciface_matmul_bias_relu_vectorized_16x16x16; }
    else if (shape == "32x32x32") { M = N = K = 32; fn = _mlir_ciface_matmul_bias_relu_vectorized_32x32x32; }
    else { std::fprintf(stderr, "unknown shape %s\n", shape.c_str()); return 2; }
  } else if (variant == "generic") {
    if (shape == "8x8x8") { M = N = K = 8; fn = _mlir_ciface_matmul_bias_relu_8x8x8; }
    else if (shape == "16x16x16") { M = N = K = 16; fn = _mlir_ciface_matmul_bias_relu_16x16x16; }
    else if (shape == "32x32x32") { M = N = K = 32; fn = _mlir_ciface_matmul_bias_relu_32x32x32; }
    else { std::fprintf(stderr, "unknown shape %s\n", shape.c_str()); return 2; }
  } else {
    std::fprintf(stderr, "unknown variant %s (expected generic|vectorized)\n", variant.c_str());
    return 2;
  }

  GuardedBuffer lhsBuf(M * K), rhsBuf(K * N), biasBuf(M * N);
  fillDeterministic(lhsBuf.payload(), M * K, 0x1234u + static_cast<uint32_t>(M));
  fillDeterministic(rhsBuf.payload(), K * N, 0x5678u + static_cast<uint32_t>(N));
  fillDeterministic(biasBuf.payload(), M * N, 0x9abcu + static_cast<uint32_t>(K));

  std::vector<float> ref(static_cast<size_t>(M * N));
  scalarRef(lhsBuf.payload(), rhsBuf.payload(), biasBuf.payload(), ref.data(), M, N, K);

  MemRef2D lhsDesc = makeDescriptor(lhsBuf.payload(), M, K);
  MemRef2D rhsDesc = makeDescriptor(rhsBuf.payload(), K, N);
  MemRef2D biasDesc = makeDescriptor(biasBuf.payload(), M, N);

  int64_t expectedAllocBytes = M * N * 4 + 64;  // matches the project's own
                                                  // malloc(size+64) convention

  int64_t firstFailingIteration = -1;
  std::string failureReason;

  for (int64_t call = 0; call < numCalls; ++call) {
    MemRef2D out{};
    fn(&out, &lhsDesc, &rhsDesc, &biasDesc);

    // 1. Descriptor sanity.
    bool descOk = true;
    if (out.sizes[0] != M || out.sizes[1] != N) { descOk = false; failureReason = "sizes mismatch"; }
    if (out.strides[0] != N || out.strides[1] != 1) { descOk = false; failureReason = "strides mismatch"; }
    if (out.offset != 0) { descOk = false; failureReason = "offset nonzero"; }
    if (out.allocated == nullptr || out.aligned == nullptr) { descOk = false; failureReason = "null pointer"; }

    // 2. Compare output against reference.
    double maxErr = 0.0;
    for (int64_t i = 0; i < M * N; ++i) {
      maxErr = std::max(maxErr, static_cast<double>(std::fabs(out.aligned[i] - ref[static_cast<size_t>(i)])));
    }
    bool outputOk = maxErr < 1e-3;

    // 3. Guard regions on inputs.
    int lhsGuard = lhsBuf.checkGuards("lhs");
    int rhsGuard = rhsBuf.checkGuards("rhs");
    int biasGuard = biasBuf.checkGuards("bias");
    bool guardsOk = (lhsGuard == -1 && rhsGuard == -1 && biasGuard == -1);

    if (!descOk || !outputOk || !guardsOk) {
      firstFailingIteration = call;
      std::printf("FIRST FAILURE at call %lld:\n", static_cast<long long>(call));
      std::printf("  descOk=%d (%s)\n", descOk, descOk ? "" : failureReason.c_str());
      std::printf("  outputOk=%d maxErr=%.6f\n", outputOk, maxErr);
      std::printf("  guardsOk=%d lhsGuard=%d rhsGuard=%d biasGuard=%d\n", guardsOk, lhsGuard, rhsGuard, biasGuard);
      std::printf("  allocated=%p aligned=%p offset=%lld sizes=[%lld,%lld] strides=[%lld,%lld]\n",
                   (void*)out.allocated, (void*)out.aligned, (long long)out.offset,
                   (long long)out.sizes[0], (long long)out.sizes[1],
                   (long long)out.strides[0], (long long)out.strides[1]);
      std::printf("  expected malloc size (M*N*4+64) = %lld\n", (long long)expectedAllocBytes);
      std::free(out.allocated);
      break;
    }

    std::free(out.allocated);
  }

  if (firstFailingIteration == -1) {
    std::printf("PASS: shape=%s variant=%s all %lld calls correct, guards clean\n",
                shape.c_str(), variant.c_str(), (long long)numCalls);
    return 0;
  } else {
    std::printf("FAIL: shape=%s variant=%s first failure at call %lld / %lld\n",
                shape.c_str(), variant.c_str(),
                (long long)firstFailingIteration, (long long)numCalls);
    return 1;
  }
}
