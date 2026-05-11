#pragma once

#include "ir/node.h"
#include "runtime/backend_type.h"

class ExecutionProvider {
public:
    virtual ~ExecutionProvider() = default;

    virtual bool can_execute(const Node& node) const = 0;
    virtual BackendType backend_type() const = 0;
    virtual const char* name() const = 0;
};