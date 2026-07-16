// Empirical timer/harness-overhead probe for Stage 17's timing-quality
// requirement. Measures:
//   1. Back-to-back steady_clock::now() call overhead (clock-read cost).
//   2. Overhead of a near-empty function call through the SAME call
//      pattern the schedule harness uses (call + malloc + free, since the
//      generated kernels are sret-return-via-malloc), to bound realistic
//      per-timed-iteration floor.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

int main() {
  constexpr int64_t N = 100000;

  // 1. Raw clock-read overhead: time N consecutive now() calls.
  std::vector<double> clock_overhead_ns(N);
  for (int64_t i = 0; i < N; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    auto t1 = std::chrono::steady_clock::now();
    clock_overhead_ns[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
  }
  std::sort(clock_overhead_ns.begin(), clock_overhead_ns.end());
  double clock_median = clock_overhead_ns[N / 2];
  double clock_p95 = clock_overhead_ns[static_cast<size_t>(N * 0.95)];

  // 2. malloc+free overhead (the generated kernel's sret allocation
  // pattern -- a floor for what any generated kernel call must at least
  // cost, independent of its actual compute).
  constexpr int64_t M = 50000;
  std::vector<double> malloc_free_ns(M);
  for (int64_t i = 0; i < M; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    void* p = malloc(4160);  // matches the generated kernel's allocation size (verified: 4160 bytes, seen in earlier disassembly, w0=#4160)
    auto t1 = std::chrono::steady_clock::now();
    free(p);
    malloc_free_ns[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
  }
  std::sort(malloc_free_ns.begin(), malloc_free_ns.end());
  double malloc_median = malloc_free_ns[M / 2];
  double malloc_p95 = malloc_free_ns[static_cast<size_t>(M * 0.95)];

  std::printf("{\n");
  std::printf("  \"clock_read_overhead_ns\": {\"median\": %.2f, \"p95\": %.2f, \"samples\": %lld},\n",
              clock_median, clock_p95, static_cast<long long>(N));
  std::printf("  \"malloc_free_overhead_ns\": {\"median\": %.2f, \"p95\": %.2f, \"samples\": %lld}\n",
              malloc_median, malloc_p95, static_cast<long long>(M));
  std::printf("}\n");
  return 0;
}
