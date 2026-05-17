#include "runtime/llm_scheduler.h"

#include <chrono>
#include <iostream>

void LLMScheduler::submit(
    LLMRequest request
) {
    waiting_queue.push_back(request);
}

bool LLMScheduler::all_finished() const {
    return waiting_queue.empty()
        && active_prefill.empty()
        && active_decode.empty();
}

void LLMScheduler::schedule_prefill() {
    while (!waiting_queue.empty()) {
        auto req = waiting_queue.front();

        waiting_queue.pop_front();

        req.state = RequestState::Prefill;
        req.start_time_ms = now_ms();

        profiler.add_trace_event(
            req.request_id,
            "prefill",
            0,
            req.start_time_ms
        );
        req.kv_blocks =
            kv_cache.allocate_blocks(
                req.request_id,
                static_cast<int>(req.prompt_tokens.size())
            );

        std::cout << "[Scheduler] Allocated KV blocks: ";

        for (int b : req.kv_blocks) {
            std::cout << b << " ";
        }

        std::cout << "\n";

        executor.run_prefill(req);

        std::cout << "[Scheduler] PREFILL request "
                  << req.request_id
                  << " prompt_tokens="
                  << req.prompt_tokens.size()
                  << "\n";

        active_prefill.push_back(req);
    }

    for (auto& req : active_prefill) {
        req.state = RequestState::Decode;

        active_decode.push_back(req);
    }

    active_prefill.clear();
}

void LLMScheduler::schedule_decode() {
    auto batch =
        batcher.build_decode_batch(
            active_decode
        );

    std::vector<LLMRequest> next_decode;

    for (auto req : batch) {
        if (req.generated_count >= req.max_new_tokens) {
            req.state = RequestState::Finished;
            req.finish_time_ms = now_ms();

            profiler.add_trace_event(
            req.request_id,
            "finish",
            req.generated_count + 1,
            req.finish_time_ms
        );

            profiler.add_finished_request(req);

            std::cout << "[Scheduler] FINISHED request "
                      << req.request_id
                      << "\n";

            kv_cache.free_request(req.request_id);

            std::cout << "[Scheduler] Freed KV cache for request "
                      << req.request_id
                      << "\n";

            continue;
        }

        req.state = RequestState::Decode;

        profiler.add_trace_event(
            req.request_id,
            "decode",
            req.generated_count,
            now_ms()
        );

        executor.run_decode_step(req);

        std::cout << "[Scheduler] Reusing KV blocks: ";

        for (int b : req.kv_blocks) {
            std::cout << b << " ";
        }

        std::cout << "\n";

        next_decode.push_back(req);
    }

    active_decode = next_decode;
}

void LLMScheduler::run() {
    std::cout << "=== LLM Serving Scheduler ===\n";

    while (!all_finished()) {
        schedule_prefill();

        if (!active_decode.empty()) {
            std::cout << "\n[Scheduler] CONTINUOUS BATCH STEP\n";

            schedule_decode();
        }

        kv_cache.dump();

        std::cout << "\n";
    }

    profiler.print_metrics();

    profiler.export_trace_json(
        "../trace/serving_trace.json"
    );
}

double LLMScheduler::now_ms() const {
    auto t =
        Clock::now().time_since_epoch();

    return std::chrono::duration<double, std::milli>(t).count();
}