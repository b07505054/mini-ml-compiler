#pragma once

#include "pass/pass.h"

#include <iostream>
#include <memory>
#include <vector>

class PassManager {
public:
    void add_pass(
        std::unique_ptr<Pass> pass
    ) {
        passes.push_back(
            std::move(pass)
        );
    }

    void run(Graph& graph) {
        std::cout
            << "=== Compiler Pass Pipeline ===\n";

        for (auto& pass : passes) {
            std::cout
                << "[PassManager] Running "
                << pass->name()
                << "\n";

            pass->run(graph);
        }

        std::cout
            << "[PassManager] Pipeline complete\n";
    }

private:
    std::vector<std::unique_ptr<Pass>> passes;
};