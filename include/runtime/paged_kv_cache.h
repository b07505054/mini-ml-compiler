#pragma once

#include <unordered_map>
#include <vector>
#include <iostream>

struct KVBlock {
    int block_id;

    bool used = false;
};

class PagedKVCache {
public:
    PagedKVCache(
        int num_blocks,
        int block_size_tokens
    );

    std::vector<int> allocate_blocks(
        int request_id,
        int num_tokens
    );

    void free_request(
        int request_id
    );

    void dump() const;

private:
    int block_size_tokens;

    std::vector<KVBlock> blocks;

    std::unordered_map<
        int,
        std::vector<int>
    > request_table;
};