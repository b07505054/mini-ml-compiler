#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// K-tail experiment for one 8x8 output tile.  All paths read A/B directly
// except materialized_k_tail, which intentionally preserves the old
// zero-fill+copy baseline.  The direct paths never form padded source tiles.
// A vector.transfer_read padding operand has the same semantic role as the
// explicit zero used by guarded_k_tail: an out-of-range lane evaluates to
// zero.  It does not promise a masked machine load.  Likewise, in_bounds may
// only be true for dimensions proven in range; permutation maps only describe
// dimension mapping and never provide bounds protection.

static inline void epilogue(const float *bias, float *out,
                            float32x4_t acc[8][2]) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  for (int m = 0; m < 8; ++m)
    for (int n = 0; n < 2; ++n)
      vst1q_f32(out + m * 8 + n * 4,
                vmaxq_f32(vaddq_f32(acc[m][n],
                                    vld1q_f32(bias + m * 8 + n * 4)),
                          zero));
}

extern "C" __attribute__((noinline))
void materialized_k_tail(const float *a, const float *b, const float *bias,
                         float *out, int kTail) {
  alignas(64) float ap[64] = {};
  alignas(64) float bp[64] = {};
  for (int m = 0; m < 8; ++m)
    for (int k = 0; k < kTail; ++k)
      ap[m * 8 + k] = a[m * kTail + k];
  for (int k = 0; k < kTail; ++k)
    for (int n = 0; n < 8; ++n)
      bp[k * 8 + n] = b[k * 8 + n];
  float32x4_t acc[8][2];
  for (auto &row : acc)
    for (auto &v : row) v = vdupq_n_f32(0.0f);
  for (int k = 0; k < 8; ++k) {
    float32x4_t b0 = vld1q_f32(bp + k * 8);
    float32x4_t b1 = vld1q_f32(bp + k * 8 + 4);
    for (int m = 0; m < 8; ++m) {
      acc[m][0] = vfmaq_n_f32(acc[m][0], b0, ap[m * 8 + k]);
      acc[m][1] = vfmaq_n_f32(acc[m][1], b1, ap[m * 8 + k]);
    }
  }
  epilogue(bias, out, acc);
}

extern "C" __attribute__((noinline))
void direct_vector_k_tail(const float *a, const float *b, const float *bias,
                          float *out, int kTail) {
  float32x4_t acc[8][2];
  for (auto &row : acc)
    for (auto &v : row) v = vdupq_n_f32(0.0f);
  for (int k = 0; k < kTail; ++k) {
    const float32x4_t b0 = vld1q_f32(b + k * 8);
    const float32x4_t b1 = vld1q_f32(b + k * 8 + 4);
    for (int m = 0; m < 8; ++m) {
      const float av = a[m * kTail + k];
      acc[m][0] = vfmaq_n_f32(acc[m][0], b0, av);
      acc[m][1] = vfmaq_n_f32(acc[m][1], b1, av);
    }
  }
  epilogue(bias, out, acc);
}

extern "C" __attribute__((noinline))
void guarded_transfer_k_tail(const float *a, const float *b,
                             const float *bias, float *out, int kTail) {
  float32x4_t acc[8][2];
  for (auto &row : acc)
    for (auto &v : row) v = vdupq_n_f32(0.0f);
  // Fixed eight-lane representation of transfer-read padding semantics.
  // The guard, not the vector lane mapping, proves that A/B are in bounds.
  for (int k = 0; k < 8; ++k) {
    if (k >= kTail) continue;
    const float32x4_t b0 = vld1q_f32(b + k * 8);
    const float32x4_t b1 = vld1q_f32(b + k * 8 + 4);
    for (int m = 0; m < 8; ++m) {
      const float av = a[m * kTail + k];
      acc[m][0] = vfmaq_n_f32(acc[m][0], b0, av);
      acc[m][1] = vfmaq_n_f32(acc[m][1], b1, av);
    }
  }
  epilogue(bias, out, acc);
}

extern "C" __attribute__((noinline))
void full8_main_direct_tail(const float *a, const float *b, const float *bias,
                            float *out, int kSize) {
  float32x4_t acc[8][2];
  for (auto &row : acc)
    for (auto &v : row) v = vdupq_n_f32(0.0f);
  const int full = kSize / 8 * 8;
  for (int kb = 0; kb < full; kb += 8)
    for (int k = kb; k < kb + 8; ++k) {
      const float32x4_t b0 = vld1q_f32(b + k * 8);
      const float32x4_t b1 = vld1q_f32(b + k * 8 + 4);
      for (int m = 0; m < 8; ++m) {
        const float av = a[m * kSize + k];
        acc[m][0] = vfmaq_n_f32(acc[m][0], b0, av);
        acc[m][1] = vfmaq_n_f32(acc[m][1], b1, av);
      }
    }
  for (int k = full; k < kSize; ++k) {
    const float32x4_t b0 = vld1q_f32(b + k * 8);
    const float32x4_t b1 = vld1q_f32(b + k * 8 + 4);
    for (int m = 0; m < 8; ++m) {
      const float av = a[m * kSize + k];
      acc[m][0] = vfmaq_n_f32(acc[m][0], b0, av);
      acc[m][1] = vfmaq_n_f32(acc[m][1], b1, av);
    }
  }
  epilogue(bias, out, acc);
}

using Kernel = void (*)(const float *, const float *, const float *, float *,
                        int);

static void fill(std::vector<float> &v, uint32_t seed) {
  for (float &x : v) {
    seed = seed * 1664525u + 1013904223u;
    x = float((seed >> 8) & 65535) / 32768.0f - 1.0f;
  }
}

static void reference(const float *a, const float *b, const float *bias,
                      float *out, int kSize) {
  for (int m = 0; m < 8; ++m)
    for (int n = 0; n < 8; ++n) {
      float x = bias[m * 8 + n];
      for (int k = 0; k < kSize; ++k)
        x += a[m * kSize + k] * b[k * 8 + n];
      out[m * 8 + n] = std::max(x, 0.0f);
    }
}

static double percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[static_cast<size_t>(p * (v.size() - 1))];
}

static void runOne(const char *name, Kernel fn, int kSize) {
  std::vector<float> a(8 * kSize), b(kSize * 8), bias(64), ref(64);
  fill(a, 0x1234u + kSize);
  fill(b, 0x5678u + kSize);
  fill(bias, 0x9abcu + kSize);
  reference(a.data(), b.data(), bias.data(), ref.data(), kSize);
  std::vector<float> guarded(80, 12345.0f);
  float *out = guarded.data() + 8;
  double maxAbs = 0.0, maxRel = 0.0;
  for (int q = 0; q < 1000; ++q) {
    fn(a.data(), b.data(), bias.data(), out, kSize);
    for (int i = 0; i < 64; ++i) {
      double e = std::abs(double(out[i]) - ref[i]);
      maxAbs = std::max(maxAbs, e);
      maxRel = std::max(maxRel,
                        e / std::max(std::abs(double(ref[i])), 1e-6));
    }
  }
  for (int i = 0; i < 8; ++i)
    if (guarded[i] != 12345.0f || guarded[72 + i] != 12345.0f)
      std::abort();
  for (int q = 0; q < 1000; ++q)
    fn(a.data(), b.data(), bias.data(), out, kSize);
  std::vector<double> samples;
  samples.reserve(20000);
  for (int q = 0; q < 20000; ++q) {
    auto begin = std::chrono::steady_clock::now();
    fn(a.data(), b.data(), bias.data(), out, kSize);
    auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::nano>(end - begin).count());
  }
  std::printf(
      "RESULT K=%d strategy=%s max_abs=%.9g max_rel=%.9g median_ns=%.9g "
      "p95_ns=%.9g min_ns=%.9g max_ns=%.9g\n",
      kSize, name, maxAbs, maxRel, percentile(samples, 0.5),
      percentile(samples, 0.95), percentile(samples, 0.0),
      percentile(samples, 1.0));
}

int main() {
  for (int k = 1; k <= 7; ++k) {
    runOne("materialized", materialized_k_tail, k);
    runOne("direct_vector", direct_vector_k_tail, k);
    runOne("guarded_transfer", guarded_transfer_k_tail, k);
  }
  for (int k : {8, 9, 15, 16, 17, 31, 32, 33, 63, 65})
    runOne("full8_plus_direct_tail", full8_main_direct_tail, k);
}
