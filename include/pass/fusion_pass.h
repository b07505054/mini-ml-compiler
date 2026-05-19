#pragma once

#include "pass/pass.h"

class FusionPass : public Pass {
public:
    void run(Graph& graph) override;
    const char* name() const override {
        return "FusionPass";
    }
};