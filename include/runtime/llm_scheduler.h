#pragma once

#include "runtime/llm_request.h"
#include "runtime/paged_kv_cache.h"
#include "runtime/serving_profiler.h"
#include <chrono>
#include <deque>
#include <vector>
#include "runtime/continuous_batcher.h"
#include "runtime/prefill_decode_executor.h"

class LLMScheduler {
public:
    void submit(LLMRequest request);
    void run();

private:
    using Clock = std::chrono::high_resolution_clock;
    PrefillDecodeExecutor executor;
    std::deque<LLMRequest> waiting_queue;
    std::vector<LLMRequest> active_prefill;
    std::vector<LLMRequest> active_decode;
    ServingProfiler profiler;
    ContinuousBatcher batcher;
    PagedKVCache kv_cache{16, 16};

    void schedule_prefill();
    void schedule_decode();
    bool all_finished() const;

    double now_ms() const;
};