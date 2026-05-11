#include "runtime/arena_allocator.h"

void ArenaAllocator::allocate(size_t size) {
    arena.resize(size);
}

float* ArenaAllocator::get_ptr(size_t offset) {
    return arena.data() + offset;
}

size_t ArenaAllocator::size() const {
    return arena.size();
}