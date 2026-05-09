#pragma once

#include "runtime/backend_type.h"

inline const char* backend_name(BackendType backend) {
    switch (backend) {
        case BackendType::CPU:
            return "CPU";
        case BackendType::MockGPU:
            return "MockGPU";
        default:
            return "Unknown";
    }
}