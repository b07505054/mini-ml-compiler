#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "runtime/metal_backend.h"

#include <iostream>

MetalBackend::MetalBackend() {}

const char* MetalBackend::name() const {
    return "Metal";
}

void MetalBackend::execute(
    Graph& graph,
    const Node& node
) {
    std::cout
        << "[MetalBackend] Executing node: "
        << node.name
        << "\n";

    id<MTLDevice> device =
        MTLCreateSystemDefaultDevice();

    if (!device) {
        std::cout
            << "[MetalBackend] No Metal device\n";
        return;
    }

    std::cout
        << "[MetalBackend] Device: "
        << [[device name] UTF8String]
        << "\n";

    // Phase 1:
    // runtime dispatch simulation only

    // Next:
    // real Metal kernel execution
}