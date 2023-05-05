#include <hip/hip_runtime.h>

extern "C" {
__global__ void gpu_matrix_mult(int *a, int *b, int *c, int m, int n, int k, int _align) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}
}