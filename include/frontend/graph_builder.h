#pragma once

#include "ir/graph.h"

// Abstract base for all frontend graph builders.
// Subclasses produce a Graph IR from a domain-specific model descriptor.
// The Graph IR is the compiler's canonical input representation; all
// downstream passes (shape inference, fusion, memory planning, lowering)
// operate on Graph regardless of which builder produced it.
class GraphBuilder {
public:
    virtual ~GraphBuilder() = default;

    // Build and return a fully constructed Graph.
    // The returned Graph is independent of the builder — callers may
    // modify it freely after build() returns.
    virtual Graph build() = 0;
};
