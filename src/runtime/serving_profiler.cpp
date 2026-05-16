#include "runtime/serving_profiler.h"

#include <iostream>

void ServingProfiler::add_finished_request(
    const LLMRequest& request
) {
    finished_requests.push_back(request);
}

void ServingProfiler::print_metrics() const {
    std::cout << "\n=== Serving Metrics ===\n";

    int total_tokens = 0;

    double total_latency = 0.0;

    for (const auto& req : finished_requests) {
        double latency =
            req.finish_time_ms
            - req.start_time_ms;

        total_latency += latency;

        total_tokens += req.generated_count;

        double tok_per_sec =
            req.generated_count
            / (latency / 1000.0);

        std::cout << "Request "
                  << req.request_id
                  << "\n";

        std::cout << "  latency_ms: "
                  << latency
                  << "\n";

        std::cout << "  generated_tokens: "
                  << req.generated_count
                  << "\n";

        std::cout << "  tokens/sec: "
                  << tok_per_sec
                  << "\n";
    }

    double avg_latency =
        total_latency
        / finished_requests.size();

    std::cout << "\nTotal generated tokens: "
              << total_tokens
              << "\n";

    std::cout << "Average request latency: "
              << avg_latency
              << " ms\n";
}