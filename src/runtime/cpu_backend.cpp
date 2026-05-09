#include "runtime/cpu_backend.h"

CPUBackend::CPUBackend()
    : registry(create_default_registry()) {}

void CPUBackend::execute(Graph& graph, const Node& node) {
    registry.dispatch(node.op, graph, node);
}

const char* CPUBackend::name() const {
    return "CPU";
}