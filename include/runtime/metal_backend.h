#pragma once

#include "runtime/backend.h"

class MetalBackend : public Backend {
public:
    MetalBackend();

    const char* name() const override;

    void execute(
        Graph& graph,
        const Node& node
    ) override;
};