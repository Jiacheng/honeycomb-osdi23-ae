#include "rpc/mem_capability.h"
#include "rpc/ring_queue.h"
#include <hip/hip_runtime.h>

extern "C" {
using namespace gpumpc::rpc;

__global__ void GetClock(unsigned long *v) { *v = __clock64(); }

//
// Raw RPC CPU performance, no capability

__global__ void PingServerCpu(unsigned long *base, unsigned long timestamp) {
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    base[idx] = timestamp;
}

__global__ void PingClientCpu(unsigned long *base) {
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    base[idx] = __clock64();
}

__global__ void PingClientCpuCollect(unsigned long *base, unsigned long timestamp) {
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    base[idx] = __clock64() - timestamp;
}

//
// Raw RPC performance, no capability
__global__ void PingServerRaw(LockFreeQueueView q, unsigned long *base) {
    LockFreeQueueView::Entry e;

    while (true) {
        while (!q.DequeueSingleReader(&e)) {
            asm("s_sleep 2" ::: "memory");
        }

        auto idx = e.write_cap[0];
        base[idx] = e.read_cap[0];
    }
}

__global__ void PingClientRaw(LockFreeQueueView q, unsigned long *base) {
    LockFreeQueueView::Entry e;
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long start = __clock64();
    e.read_cap[0] = 1;
    e.write_cap[0] = idx;
    base[idx] = 0;
    q.Enqueue(e);
    unsigned v;
    do {
        v = __atomic_load_n(base + idx, __ATOMIC_ACQUIRE);
        asm("s_sleep 2" ::: "memory");
    } while (v == 0);
    unsigned long end = __clock64();
    base[idx] = end - start;
}

//
// RPC with capability
__global__ void PingServerCap(LockFreeQueueView q, unsigned long *base) {
    LockFreeQueueView::Entry e;

    while (true) {
        while (!q.DequeueSingleReader(&e)) {
            asm("s_sleep 2" ::: "memory");
        }

        if (IsValidCapability(e.read_cap) && IsValidCapability(e.write_cap)) {
            auto idx = e.write_cap[2];
            base[idx] = 1;
        }
    }
}

__global__ void PingClientCap(LockFreeQueueView q, unsigned long *base) {
    LockFreeQueueView::Entry e;
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long start = __clock64();
    ToCapability(1, 0, idx, 0, e.read_cap);
    ToCapability(1, 0, idx, 0, e.write_cap);
    base[idx] = 0;
    q.Enqueue(e);
    unsigned v;
    do {
        v = __atomic_load_n(base + idx, __ATOMIC_ACQUIRE);
        asm("s_sleep 2" ::: "memory");
    } while (v == 0);
    unsigned long end = __clock64();
    base[idx] = end - start;
}
}
