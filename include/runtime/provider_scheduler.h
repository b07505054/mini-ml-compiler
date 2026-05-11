#pragma once

#include "ir/node.h"
#include "runtime/backend_type.h"
#include "runtime/cpu_execution_provider.h"
#include "runtime/mock_gpu_execution_provider.h"

#include <vector>

class ProviderScheduler {
public:
    ProviderScheduler();

    BackendType select_backend(const Node& node) const;
    const char* selected_provider_name(const Node& node) const;

private:
    MockGPUExecutionProvider mock_gpu_provider;
    CPUExecutionProvider cpu_provider;
};