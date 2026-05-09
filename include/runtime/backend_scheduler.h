#pragma once

#include "ir/node.h"
#include "runtime/backend_type.h"

class BackendScheduler {
public:
    BackendType select_backend(const Node& node) const;
};