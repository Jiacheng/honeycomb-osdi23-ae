#include <hip/hip_runtime.h>

__global__ extern "C" void fill(int *base, int b) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    base[gid] = b;
}
