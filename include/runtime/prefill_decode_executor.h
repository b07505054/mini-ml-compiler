#pragma once

#include "runtime/llm_request.h"

class PrefillDecodeExecutor {
public:
    void run_prefill(
        LLMRequest& request
    ) const;

    bool run_decode_step(
        LLMRequest& request
    ) const;
};