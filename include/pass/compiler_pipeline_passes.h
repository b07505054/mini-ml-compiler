#pragma once
#include "pass/pass.h"
#include "pass/compiler_pass.h"

class ShapeInferencePass : public Pass {
public:
    const char* name() const override;
    void run(Graph& graph) override;
};

class DTypePropagationPass : public Pass {
public:
    const char* name() const override;
    void run(Graph& graph) override;
};

class FusionCandidatePass : public Pass {
public:
    const char* name() const override;
    void run(Graph& graph) override;
};

class MemoryPlanningPass : public Pass {
public:
    const char* name() const override;
    void run(Graph& graph) override;
};

class BackendPlacementPass : public Pass {
public:
    const char* name() const override;
    void run(Graph& graph) override;
};

class SchedulingPass : public Pass {
public:
    const char* name() const override;
    void run(Graph& graph) override;
};