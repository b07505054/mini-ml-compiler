#include "runtime/paged_kv_cache.h"

#include <cmath>
#include <stdexcept>

PagedKVCache::PagedKVCache(
    int num_blocks,
    int block_size_tokens
)
    : block_size_tokens(block_size_tokens)
{
    for (int i = 0; i < num_blocks; ++i) {
        blocks.push_back({i, false});
    }
}

std::vector<int> PagedKVCache::allocate_blocks(
    int request_id,
    int num_tokens
) {
    int needed_blocks =
        static_cast<int>(
            std::ceil(
                static_cast<float>(num_tokens)
                / block_size_tokens
            )
        );

    std::vector<int> allocated;

    for (auto& block : blocks) {
        if (!block.used) {
            block.used = true;

            allocated.push_back(block.block_id);

            if (allocated.size() == needed_blocks) {
                break;
            }
        }
    }

    if (allocated.size() != needed_blocks) {
        throw std::runtime_error(
            "PagedKVCache: out of KV blocks"
        );
    }

    request_table[request_id] = allocated;

    return allocated;
}

void PagedKVCache::free_request(
    int request_id
) {
    auto it = request_table.find(request_id);

    if (it == request_table.end()) {
        return;
    }

    for (int block_id : it->second) {
        blocks[block_id].used = false;
    }

    request_table.erase(it);
}

void PagedKVCache::dump() const {
    std::cout << "=== Paged KV Cache ===\n";

    int used = 0;

    for (const auto& block : blocks) {
        if (block.used) {
            used++;
        }
    }

    std::cout << "Used blocks: "
              << used
              << "/"
              << blocks.size()
              << "\n\n";

    for (const auto& kv : request_table) {
        std::cout << "Request "
                  << kv.first
                  << " blocks: ";

        for (int b : kv.second) {
            std::cout << b << " ";
        }

        std::cout << "\n";
    }

    std::cout << "\n";
}