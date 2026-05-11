#include "runtime/cpu_execution_provider.h"

bool CPUExecutionProvider::can_execute(const Node&) const {
    // CPU is the fallback provider and supports all registered runtime ops.
    return true;
}

BackendType CPUExecutionProvider::backend_type() const {
    return BackendType::CPU;
}

const char* CPUExecutionProvider::name() const {
    return "CPUExecutionProvider";
}