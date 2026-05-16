#pragma once

#include "runtime/llm_request.h"

#include <vector>

class ContinuousBatcher {
public:
    std::vector<LLMRequest> build_decode_batch(
        const std::vector<LLMRequest>& active_decode
    ) const;
};