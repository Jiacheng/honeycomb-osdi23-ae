#include <hip/hip_runtime.h>

extern "C" {
__global__ void VectorAdd(int *c, const int *a, const int *b) {
    unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;
    c[gid] = a[gid] + b[gid];
}

}