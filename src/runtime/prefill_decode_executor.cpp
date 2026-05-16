#include "runtime/prefill_decode_executor.h"

#include <iostream>

void PrefillDecodeExecutor::run_prefill(
    LLMRequest& request
) const {
    std::cout << "[Executor] PREFILL request "
              << request.request_id
              << "\n";
}

bool PrefillDecodeExecutor::run_decode_step(
    LLMRequest& request
) const {
    if (request.generated_count
        >= request.max_new_tokens) {
        return false;
    }

    int fake_token =
        100 + request.generated_count;

    request.generated_tokens.push_back(
        fake_token
    );

    request.generated_count++;

    std::cout << "[Executor] DECODE request "
              << request.request_id
              << " generated token "
              << fake_token
              << "\n";

    return true;
}