#include "runtime/paged_kv_cache.h"

#include <iostream>

int main() {
    PagedKVCache kv_cache(
        8,   // total blocks
        16   // tokens per block
    );

    auto req1 =
        kv_cache.allocate_blocks(
            1,
            20
        );

    auto req2 =
        kv_cache.allocate_blocks(
            2,
            40
        );

    std::cout << "After allocations:\n";

    kv_cache.dump();

    kv_cache.free_request(1);

    std::cout << "After freeing request 1:\n";

    kv_cache.dump();

    auto req3 =
        kv_cache.allocate_blocks(
            3,
            32
        );

    std::cout << "After allocating request 3:\n";

    kv_cache.dump();

    return 0;
}