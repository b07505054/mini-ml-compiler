#include "runtime/free_list_allocator.h"

#include <iostream>

int main() {
    std::cout << "=== FreeList Allocator v2 Demo ===\n";

    FreeListAllocator allocator(64);

    size_t a = allocator.allocate(16);
    size_t b = allocator.allocate(8);
    size_t c = allocator.allocate(16);

    std::cout << "\nAfter allocations A=16, B=8, C=16:\n";
    allocator.dump();

    allocator.free(b);

    std::cout << "\nAfter freeing B:\n";
    allocator.dump();

    size_t d = allocator.allocate(6);

    std::cout << "\nAfter allocating D=6 using best-fit reuse:\n";
    std::cout << "D offset: " << d << "\n";
    allocator.dump();

    allocator.free(a);
    allocator.free(c);
    allocator.free(d);

    std::cout << "\nAfter freeing all blocks and coalescing:\n";
    allocator.dump();

    return 0;
}