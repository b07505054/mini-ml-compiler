#pragma once

#include "runtime/execution_provider.h"

class MockGPUExecutionProvider : public ExecutionProvider {
public:
    bool can_execute(const Node& node) const override;
    BackendType backend_type() const override;
    const char* name() const override;
};