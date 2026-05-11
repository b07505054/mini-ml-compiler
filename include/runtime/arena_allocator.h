#pragma once

#include <vector>
#include <cstddef>

class ArenaAllocator {
public:
    void allocate(size_t size);

    float* get_ptr(size_t offset);

    size_t size() const;

private:
    std::vector<float> arena;
};