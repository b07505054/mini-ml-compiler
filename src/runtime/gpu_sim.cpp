#include "runtime/gpu_sim.h"

void GPUSimulator::launch(
    Dim3 gridDim,
    Dim3 blockDim,
    KernelFn kernel
) {
    for (int by = 0; by < gridDim.y; ++by) {
        for (int bx = 0; bx < gridDim.x; ++bx) {

            for (int ty = 0; ty < blockDim.y; ++ty) {
                for (int tx = 0; tx < blockDim.x; ++tx) {

                    kernel(bx, by, tx, ty);

                }
            }
        }
    }
}