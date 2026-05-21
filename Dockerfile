FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN rm -rf build && mkdir build
WORKDIR /workspace/build

RUN cmake .. -DCMAKE_CUDA_ARCHITECTURES=75

RUN cmake --build . --target run_cuda_vector_add_profile -j

CMD ["./run_cuda_vector_add_profile"]
