#include "runtime/serving_profiler.h"
#include <fstream>
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
void ServingProfiler::add_trace_event(
    int request_id,
    const std::string& phase,
    int step,
    double timestamp_ms
) {
    trace_events.push_back({
        request_id,
        phase,
        step,
        timestamp_ms
    });
}
void ServingProfiler::export_trace_json(
    const std::string& path
) const {
    std::ofstream out(path);

    out << "[\n";

    for (size_t i = 0; i < trace_events.size(); ++i) {
        const auto& e = trace_events[i];

        out << "  {\n";
        out << "    \"request_id\": "
            << e.request_id << ",\n";

        out << "    \"phase\": \""
            << e.phase << "\",\n";

        out << "    \"step\": "
            << e.step << ",\n";

        out << "    \"timestamp_ms\": "
            << e.timestamp_ms << "\n";

        out << "  }";

        if (i + 1 < trace_events.size()) {
            out << ",";
        }

        out << "\n";
    }

    out << "]\n";
}