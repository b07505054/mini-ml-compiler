#pragma once

#include "pass/pass.h"

class CanonicalizationPass : public Pass {
public:
    const char* name() const override {
        return "CanonicalizationPass";
    }

    void run(Graph& graph) override;
};