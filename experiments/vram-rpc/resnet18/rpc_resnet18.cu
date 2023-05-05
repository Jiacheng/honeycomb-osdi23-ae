#include "rpc/mem_capability.h"
#include "rpc/ring_queue.h"
#include "rpc_resnet18_params.h"
#include <hip/hip_runtime.h>

using namespace gpumpc::rpc;
using namespace gpumpc::experiment;

template <unsigned kBlockSize, unsigned kSize>
__device__ static inline void Copy(unsigned idx, unsigned long *dst,
                                   const unsigned long *src) {
    static_assert(kSize % sizeof(unsigned long) == 0, "");
    for (int i = 0;
         i < (kSize / sizeof(unsigned long) + kBlockSize - 1) / kBlockSize;
         i++) {
        unsigned j = idx + i * kBlockSize;
        if (j < kSize / sizeof(unsigned long)) {
            dst[j] = src[j];
        }
    }
}

extern "C" {

__global__ void GetClock(unsigned long *v) { *v = __clock64(); }

//
// Raw RPC CPU performance, no capability
__global__ void PingServerCpu(unsigned long *dummy) {
}

__launch_bounds__(kResNet18RPCClientBlockSize) __global__
    void PingClientCpu(unsigned long *base, unsigned long *request,
                       const unsigned long *image) {
    enum { kBlockSize = kResNet18RPCClientBlockSize };
    unsigned idx = threadIdx.x;
    unsigned long start = __clock64();

    Copy<kBlockSize, kResNet18InputImageSize>(idx, request, image);
    base[idx] = start;
}

__launch_bounds__(kResNet18RPCClientBlockSize) __global__
void PingClientCpuCollect(unsigned long *base, unsigned long *request,
                          unsigned long *result,
                          unsigned long timestamp) {
    enum { kBlockSize = kResNet18RPCClientBlockSize };
    unsigned idx = threadIdx.x;
    Copy<kBlockSize, kResNet18ResultSize>(idx, result, request);
    base[idx] = __clock64() - timestamp;
}

//
// Raw RPC performance, no capability
__launch_bounds__(kResNet18RPCServerBlockSize) __global__
    void PingServerRaw(LockFreeQueueView q, unsigned long *base,
                       unsigned long *image) {
    enum { kBlockSize = kResNet18RPCServerBlockSize };
    LockFreeQueueView::Entry e;
    unsigned idx = threadIdx.x;

    if (!idx) {
        while (!q.DequeueSingleReader(&e)) {
            asm("s_sleep 2" ::: "memory");
        }
    }
    __syncthreads();

    Copy<kBlockSize, kResNet18InputImageSize>(idx, image, base);
}

//
// Raw RPC performance, no capability
__launch_bounds__(kResNet18RPCServerBlockSize) __global__
    void RespondResult(unsigned long *base, const unsigned long *result) {
    enum { kBlockSize = kResNet18RPCServerBlockSize };
    unsigned idx = threadIdx.x;
    Copy<kBlockSize, kResNet18ResultSize>(idx, base + kResNet18RPCResultOffset,
                                          result);

    __atomic_store_n(base + kRPCSynIndicatorOffset, 1, __ATOMIC_RELEASE);
}

__launch_bounds__(kResNet18RPCClientBlockSize) __global__
    void PingClientRaw(LockFreeQueueView q, unsigned long *base,
                       const unsigned long *image) {
    enum { kBlockSize = kResNet18RPCClientBlockSize };
    LockFreeQueueView::Entry e;
    unsigned idx = threadIdx.x;
    unsigned long start = __clock64();

    Copy<kBlockSize, kResNet18InputImageSize>(idx, base, image);
    base[kRPCSynIndicatorOffset] = 0;
    __syncthreads();
    if (idx == 0) {
        q.Enqueue(e);
    }
    unsigned v;
    do {
        v = __atomic_load_n(base + kRPCSynIndicatorOffset, __ATOMIC_ACQUIRE);
        asm("s_sleep 2" ::: "memory");
    } while (v == 0);
    __syncthreads();
    unsigned long end = __clock64();
    base[kRPCTimeOffset] = end - start;
}

__launch_bounds__(kResNet18RPCServerBlockSize) __global__
    void PingServerCap(LockFreeQueueView q, unsigned long *base,
                       unsigned long *image) {
    enum { kBlockSize = kResNet18RPCServerBlockSize };
    LockFreeQueueView::Entry e;
    unsigned idx = threadIdx.x;

    if (!idx) {
        while (!q.DequeueSingleReader(&e)) {
            asm("s_sleep 2" ::: "memory");
        }
        if ((!IsValidCapability(e.read_cap) ||
             !IsValidCapability(e.write_cap))) {
            asm("s_trap 2" ::: "memory");
        }
    }

    __syncthreads();
    Copy<kBlockSize, kResNet18InputImageSize>(idx, image, base);
}

__launch_bounds__(kResNet18RPCClientBlockSize) __global__
    void PingClientCap(LockFreeQueueView q, unsigned long *base,
                       const unsigned long *image) {
    enum { kBlockSize = kResNet18RPCClientBlockSize };
    LockFreeQueueView::Entry e;
    unsigned idx = threadIdx.x;
    unsigned long start = __clock64();

    Copy<kBlockSize, kResNet18InputImageSize>(idx, base, image);
    base[kRPCSynIndicatorOffset] = 0;
    __syncthreads();
    if (idx == 0) {
        ToCapability(1, 0, 0, 0, e.read_cap);
        ToCapability(1, 0, 0, 0, e.write_cap);
        q.Enqueue(e);
    }
    unsigned v;
    do {
        v = __atomic_load_n(base + kRPCSynIndicatorOffset, __ATOMIC_ACQUIRE);
        asm("s_sleep 2" ::: "memory");
    } while (v == 0);
    __syncthreads();
    unsigned long end = __clock64();
    base[kRPCTimeOffset] = end - start;
}

}
