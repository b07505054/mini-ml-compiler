#pragma once

#include "pass/pass.h"

#include <memory>
#include <vector>

class PassManager {
public:
    void add_pass(std::unique_ptr<Pass> pass) {
        passes.push_back(std::move(pass));
    }

    void run(Graph& graph) {
        for (auto& pass : passes) {
            pass->run(graph);
        }
    }

private:
    std::vector<std::unique_ptr<Pass>> passes;
};