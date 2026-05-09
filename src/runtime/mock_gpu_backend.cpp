#include "runtime/mock_gpu_backend.h"

#include <iostream>

MockGPUBackend::MockGPUBackend()
    : registry(create_default_registry()) {}

void MockGPUBackend::execute(Graph& graph, const Node& node) {
    std::cout << "[MockGPUBackend] Executing node on simulated GPU: "
              << node.name << "\n";

    // For now, MockGPU uses the same CPU kernels.
    // This models backend dispatch without requiring real GPU code.
    registry.dispatch(node.op, graph, node);
}

const char* MockGPUBackend::name() const {
    return "MockGPU";
}