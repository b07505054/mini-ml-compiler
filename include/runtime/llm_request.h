#pragma once

#include <string>
#include <vector>

enum class RequestState {
    Waiting,
    Prefill,
    Decode,
    Finished
};

struct LLMRequest {
    int request_id;

    std::string prompt;

    std::vector<int> prompt_tokens;

    std::vector<int> generated_tokens;

    std::vector<int> kv_blocks;

    int max_new_tokens = 0;

    int generated_count = 0;

    double start_time_ms = 0.0;

    double finish_time_ms = 0.0;

    RequestState state = RequestState::Waiting;
};