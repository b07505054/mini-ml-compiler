#pragma once

#include <cstddef>
#include <vector>
#include <iostream>

struct MemoryBlock {
    size_t offset;
    size_t size;
    bool free;
};

class FreeListAllocator {
public:
    explicit FreeListAllocator(size_t total_size);

    size_t allocate(size_t size);
    void free(size_t offset);

    void dump() const;

    size_t used_memory() const;
    size_t free_memory() const;
    size_t largest_free_block() const;

private:
    std::vector<MemoryBlock> blocks;

    void coalesce();
};