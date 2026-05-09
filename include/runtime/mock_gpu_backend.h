#pragma once

#include "runtime/backend.h"
#include "runtime/op_registry.h"

class MockGPUBackend : public Backend {
public:
    MockGPUBackend();

    void execute(Graph& graph, const Node& node) override;
    const char* name() const override;

private:
    OpRegistry registry;
};