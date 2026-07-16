// aarch64_matmul_bias_relu_mixed_shape_test.cpp
//
// Mixed-shape, mixed-variant, same-process correctness regression test,
// plus unrelated heap allocations interleaved between calls (allocator
// stress). Added alongside aarch64_matmul_bias_relu_repeated_call_test.cpp
// after finding and fixing a real un-zero-initialized matmul accumulator
// bug in the shared hir-matmul-bias-relu-to-linalg lowering (see that
// file's header for the full explanation).
//
// Repeats at least 100 cycles of: 8x8x8, 16x16x16, 32x32x32, 32x32x32,
// 16x16x16, 8x8x8 -- for both the generic and vectorized variants, called
// from the SAME process without any per-call process isolation. This is a
// correctness regression test, not a benchmark-isolation workaround: the
// underlying accumulator bug is fixed at the lowering level, so this test
// asserts real cross-call/cross-shape/cross-variant safety rather than
// relying on process isolation to avoid triggering it.

// Usage: ./aarch64_matmul_bias_relu_mixed_shape_test [cycles]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

extern "C" {
struct MemRef2D { float* allocated; float* aligned; int64_t offset; int64_t sizes[2]; int64_t strides[2]; };
void _mlir_ciface_matmul_bias_relu_8x8x8(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_16x16x16(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_32x32x32(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_vectorized_8x8x8(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_vectorized_16x16x16(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
void _mlir_ciface_matmul_bias_relu_vectorized_32x32x32(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);
}
using GeneratedFn = void (*)(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*);

namespace {

void fillDeterministic(float* p, int64_t n, uint32_t seed) {
  uint32_t state = seed;
  for (int64_t i = 0; i < n; ++i) {
    state = state * 1664525u + 1013904223u;
    p[i] = static_cast<float>((state >> 8) & 0xFFFFu) / 65536.0f - 0.5f;
  }
}
void scalarRef(const float* lhs, const float* rhs, const float* bias, float* out, int64_t M, int64_t N, int64_t K) {
  for (int64_t i=0;i<M;i++) for(int64_t j=0;j<N;j++){
    float acc=0; for(int64_t k=0;k<K;k++) acc += lhs[i*K+k]*rhs[k*N+j];
    float v = acc + bias[i*N+j]; out[i*N+j] = v>0?v:0;
  }
}

struct ShapeCtx {
  std::string name;
  int64_t M, N, K;
  std::vector<float> lhs, rhs, bias, ref;
  MemRef2D lhsDesc, rhsDesc, biasDesc;
  GeneratedFn genericFn, vectorizedFn;
};

ShapeCtx makeShape(const std::string& name, int64_t M, int64_t N, int64_t K,
                    GeneratedFn genericFn, GeneratedFn vectorizedFn) {
  ShapeCtx c;
  c.name = name; c.M = M; c.N = N; c.K = K;
  c.lhs.resize(M * K); c.rhs.resize(K * N); c.bias.resize(M * N); c.ref.resize(M * N);
  fillDeterministic(c.lhs.data(), M * K, 0x1234u + (uint32_t)M);
  fillDeterministic(c.rhs.data(), K * N, 0x5678u + (uint32_t)N);
  fillDeterministic(c.bias.data(), M * N, 0x9abcu + (uint32_t)K);
  scalarRef(c.lhs.data(), c.rhs.data(), c.bias.data(), c.ref.data(), M, N, K);
  c.lhsDesc = MemRef2D{c.lhs.data(), c.lhs.data(), 0, {M, K}, {K, 1}};
  c.rhsDesc = MemRef2D{c.rhs.data(), c.rhs.data(), 0, {K, N}, {N, 1}};
  c.biasDesc = MemRef2D{c.bias.data(), c.bias.data(), 0, {M, N}, {N, 1}};
  c.genericFn = genericFn; c.vectorizedFn = vectorizedFn;
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  int64_t cycles = argc > 1 ? std::atoll(argv[1]) : 100;

  std::vector<ShapeCtx> shapes;
  shapes.push_back(makeShape("8x8x8", 8, 8, 8,
                              _mlir_ciface_matmul_bias_relu_8x8x8,
                              _mlir_ciface_matmul_bias_relu_vectorized_8x8x8));
  shapes.push_back(makeShape("16x16x16", 16, 16, 16,
                              _mlir_ciface_matmul_bias_relu_16x16x16,
                              _mlir_ciface_matmul_bias_relu_vectorized_16x16x16));
  shapes.push_back(makeShape("32x32x32", 32, 32, 32,
                              _mlir_ciface_matmul_bias_relu_32x32x32,
                              _mlir_ciface_matmul_bias_relu_vectorized_32x32x32));

  // Cycle order: 0(8),1(16),2(32),2(32),1(16),0(8)
  int order[] = {0, 1, 2, 2, 1, 0};

  int64_t totalCalls = 0;
  int64_t firstFailCycle = -1, firstFailStep = -1;
  std::string firstFailShape, firstFailVariant;
  double firstFailErr = 0.0;

  for (int64_t cycle = 0; cycle < cycles && firstFailCycle == -1; ++cycle) {
    for (int step = 0; step < 6; ++step) {
      ShapeCtx& c = shapes[order[step]];

      // Unrelated heap allocation stress between calls, varying size.
      size_t noise = static_cast<size_t>(16 + (cycle * 7 + step * 13) % 512);
      std::vector<char> churn(noise, static_cast<char>(cycle + step));

      for (int variant = 0; variant < 2; ++variant) {
        GeneratedFn fn = variant == 0 ? c.genericFn : c.vectorizedFn;
        MemRef2D out{};
        fn(&out, &c.lhsDesc, &c.rhsDesc, &c.biasDesc);
        double maxErr = 0.0;
        for (int64_t i = 0; i < c.M * c.N; ++i) {
          maxErr = std::max(maxErr, static_cast<double>(std::fabs(
              out.aligned[i] - c.ref[static_cast<size_t>(i)])));
        }
        totalCalls++;
        if (maxErr >= 1e-3 && firstFailCycle == -1) {
          firstFailCycle = cycle;
          firstFailStep = step;
          firstFailShape = c.name;
          firstFailVariant = variant == 0 ? "generic" : "vectorized";
          firstFailErr = maxErr;
        }
        std::free(out.allocated);
      }
      (void)churn;
    }
  }

  if (firstFailCycle == -1) {
    std::printf("PASS: %lld cycles (%lld total calls) all correct\n",
                (long long)cycles, (long long)totalCalls);
    return 0;
  } else {
    std::printf("FAIL: first failure at cycle=%lld step=%d shape=%s variant=%s maxErr=%.6f (call #%lld)\n",
                (long long)firstFailCycle, firstFailStep, firstFailShape.c_str(),
                firstFailVariant.c_str(), firstFailErr, (long long)totalCalls);
    return 1;
  }
}
