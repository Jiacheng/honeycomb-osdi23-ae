#include <hip/hip_runtime.h>

extern "C" {
__global__ void f(int *a) {
#pragma unroll 1
    for (int i = 0; i < 5; i++) {
#pragma unroll 1
        for (int j = 0; j < 6; j++) {
                a[i*6 + j] = 42;
        }
    }
}
}