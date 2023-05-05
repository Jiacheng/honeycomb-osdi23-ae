#include <hip/hip_runtime.h>

extern "C" {
__launch_bounds__(256)
__global__ void clear_cp(int *a) {
    a[threadIdx.x + 256 * blockIdx.x] = 42;
}

__global__ void clear_any_blocksize(int *a) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    a[gid] = 42;
}

__global__ void clear_general(int *a, int N) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid < N) {
        a[gid] = 42;
    }
}

}