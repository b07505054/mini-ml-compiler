#include "runtime/continuous_batcher.h"

#include <iostream>

std::vector<LLMRequest>
ContinuousBatcher::build_decode_batch(
    const std::vector<LLMRequest>& active_decode
) const {
    std::vector<LLMRequest> batch;

    std::cout << "[ContinuousBatcher] Building decode batch\n";

    for (const auto& req : active_decode) {
        if (req.state == RequestState::Decode) {
            batch.push_back(req);

            std::cout << "  request "
                      << req.request_id
                      << "\n";
        }
    }

    std::cout << "[ContinuousBatcher] Batch size: "
              << batch.size()
              << "\n";

    return batch;
}