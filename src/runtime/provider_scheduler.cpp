#include "runtime/provider_scheduler.h"

ProviderScheduler::ProviderScheduler() = default;

BackendType ProviderScheduler::select_backend(const Node& node) const {
    if (mock_gpu_provider.can_execute(node)) {
        return mock_gpu_provider.backend_type();
    }

    return cpu_provider.backend_type();
}

const char* ProviderScheduler::selected_provider_name(const Node& node) const {
    if (mock_gpu_provider.can_execute(node)) {
        return mock_gpu_provider.name();
    }

    return cpu_provider.name();
}