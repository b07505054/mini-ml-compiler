#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct QuantTensor {
    std::string name;
    std::vector<int> shape;
    std::vector<int8_t> data;

    float scale = 1.0f;
    int zero_point = 0;

    QuantTensor() = default;

    QuantTensor(
        const std::string& name,
        const std::vector<int>& shape,
        float scale,
        int zero_point
    )
        : name(name),
          shape(shape),
          scale(scale),
          zero_point(zero_point) {
        int size = 1;
        for (int d : shape) {
            size *= d;
        }
        data.resize(size);
    }

    int numel() const {
        int size = 1;
        for (int d : shape) {
            size *= d;
        }
        return size;
    }
};