#pragma once

#include "pass/pass.h"

class DeadNodeEliminationPass : public Pass {
public:
    void run(Graph& graph) override;
};