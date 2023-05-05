#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace gpumpc {
namespace rpc {

class LockFreeQueueView {
  public:
    struct Descriptor {
        unsigned rptr;
        unsigned wptr;
        unsigned doorbell;
        unsigned padding;
    };
    struct Entry {
        unsigned read_cap[8];
        unsigned write_cap[8];
    };
    enum {
        kSize = 32768 / sizeof(Entry),
    };

    __device__ inline void Enqueue(const Entry &v);
    __device__ inline bool DequeueSingleReader(Entry *value);

    Descriptor *desc_;
    Entry *entries_;
};

#ifdef __HIP_DEVICE_COMPILE__

__device__ inline void LockFreeQueueView::Enqueue(const Entry &v) {
    unsigned rptr = __atomic_load_n(&desc_->rptr, __ATOMIC_ACQUIRE);
    unsigned wptr = atomicAdd(&desc_->wptr, 1);
    while (wptr >= rptr + kSize) {
        asm("s_sleep 2" ::: "memory");
        rptr = __atomic_load_n(&desc_->rptr, __ATOMIC_ACQUIRE);
    }
    entries_[wptr & (kSize - 1)] = v;
    while (atomicCAS(&desc_->doorbell, wptr, wptr + 1) == wptr) {
        asm("s_sleep 2" ::: "memory");
    }
}

__device__ inline bool LockFreeQueueView::DequeueSingleReader(Entry *value) {
    unsigned rptr = __atomic_load_n(&desc_->rptr, __ATOMIC_ACQUIRE);
    unsigned doorbell = __atomic_load_n(&desc_->doorbell, __ATOMIC_ACQUIRE);
    if (doorbell == rptr) {
        return false;
    }

    *value = entries_[rptr & (kSize - 1)];
    atomicAdd(&desc_->rptr, 1);
    return true;
}
#endif

} // namespace rpc
} // namespace gpumpc