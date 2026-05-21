#include <stdexcept>

class MetalBackend {
public:
    MetalBackend() {
        throw std::runtime_error("Metal backend is only supported on macOS");
    }

    virtual ~MetalBackend() = default;
};
