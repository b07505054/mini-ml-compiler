#include "runtime/free_list_allocator.h"

#include <limits>
#include <stdexcept>

FreeListAllocator::FreeListAllocator(size_t total_size) {
    blocks.push_back({0, total_size, true});
}

size_t FreeListAllocator::allocate(size_t size) {
    int best_index = -1;
    size_t best_size = std::numeric_limits<size_t>::max();

    for (int i = 0; i < static_cast<int>(blocks.size()); ++i) {
        const auto& block = blocks[i];

        if (block.free && block.size >= size && block.size < best_size) {
            best_index = i;
            best_size = block.size;
        }
    }

    if (best_index == -1) {
        throw std::runtime_error("FreeListAllocator: out of memory");
    }

    MemoryBlock& block = blocks[best_index];

    size_t allocated_offset = block.offset;

    if (block.size == size) {
        block.free = false;
    } else {
        MemoryBlock remaining{
            block.offset + size,
            block.size - size,
            true
        };

        block.size = size;
        block.free = false;

        blocks.insert(blocks.begin() + best_index + 1, remaining);
    }

    return allocated_offset;
}

void FreeListAllocator::free(size_t offset) {
    for (auto& block : blocks) {
        if (block.offset == offset) {
            block.free = true;
            coalesce();
            return;
        }
    }

    throw std::runtime_error("FreeListAllocator: invalid free offset");
}

void FreeListAllocator::coalesce() {
    for (size_t i = 0; i + 1 < blocks.size();) {
        if (blocks[i].free && blocks[i + 1].free) {
            blocks[i].size += blocks[i + 1].size;
            blocks.erase(blocks.begin() + i + 1);
        } else {
            ++i;
        }
    }
}

size_t FreeListAllocator::used_memory() const {
    size_t used = 0;

    for (const auto& block : blocks) {
        if (!block.free) {
            used += block.size;
        }
    }

    return used;
}

size_t FreeListAllocator::free_memory() const {
    size_t free_total = 0;

    for (const auto& block : blocks) {
        if (block.free) {
            free_total += block.size;
        }
    }

    return free_total;
}

size_t FreeListAllocator::largest_free_block() const {
    size_t largest = 0;

    for (const auto& block : blocks) {
        if (block.free && block.size > largest) {
            largest = block.size;
        }
    }

    return largest;
}

void FreeListAllocator::dump() const {
    std::cout << "=== FreeListAllocator Blocks ===\n";

    for (const auto& block : blocks) {
        std::cout << "offset=" << block.offset
                  << " size=" << block.size
                  << " free=" << (block.free ? "true" : "false")
                  << "\n";
    }

    std::cout << "Used memory: " << used_memory() << "\n";
    std::cout << "Free memory: " << free_memory() << "\n";
    std::cout << "Largest free block: " << largest_free_block() << "\n";
}