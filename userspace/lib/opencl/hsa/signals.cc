#include "signals.h"
#include "kfd_event.h"
#include "memory_manager.h"
#include "platform.h"
#include "types.h"
#include "utils.h"

#include <atomic>
#include <hsa/amd_hsa_signal.h>
#include <hsa/kfd_ioctl.h>
#include <stdexcept>

namespace ocl::hsa {

// Unlike the SDMA queue, there is no explicit reference of the value in the
// signal. So just reuse the layout of the amd_signal_s struct.
Signal::Signal(SignalPool *parent) : parent_(parent) {
    event_ = Platform::Instance().NewSignalEvent();
    if (!event_) {
        throw std::invalid_argument("Cannot create event");
    }
    data_ = parent->slab_.Allocate();
    data_->value = 1;
    data_->event_id = event_->GetEventID();
    data_->event_mailbox_ptr = event_->GetEventMailboxPtr();
    data_->kind = AMD_SIGNAL_KIND_USER;
}

Signal::~Signal() {
    parent_->slab_.Free(data_);
    auto _ = event_->Destroy();
}

gpu_addr_t Signal::GetHandle() const {
    const uint64_t handle =
        static_cast<gpu_addr_t>(reinterpret_cast<uintptr_t>(data_));
    return handle;
}

bool Signal::Barrier() {
    auto p =
        reinterpret_cast<std::atomic_long *>(const_cast<long *>(&data_->value));
    while (true) {
        auto value = p->load(std::memory_order_relaxed);
        if (!value) {
            /* ROCclr will call hsa_signal_wait_scacquire when
             * releaseGpuMemoryFence in
             * https://github.com/ROCm-Developer-Tools/ROCclr/blob/a4432d9bf978998d06d70490f552288eeb0241d7/device/rocm/rocvirtual.hpp#L65
             * then the busy-waiting signal loads the value with memory order
             * relaxed then issues a acquire fence. see
             * https://github.com/RadeonOpenCompute/ROCR-Runtime/blob/9f759e76bca886b76ab158b4c21ef6ad99bbc018/src/core/runtime/default_signal.cpp#L146
             */
            std::atomic_thread_fence(std::memory_order_acquire);
            return false;
        }
        /*
         * BusyWaitSignal won't issue AMDKFD_IOC_WAIT_EVENTS ioctl
         * Only InterruptSignal will do so
         */
    }
    return true;
}

void Signal::Set(int64_t value) {
    auto p =
        reinterpret_cast<std::atomic_long *>(const_cast<long *>(&data_->value));
    p->store(value, std::memory_order_relaxed);
}

gpu_addr_t Signal::ValueLocation() const { return (gpu_addr_t)&data_->value; }

gpu_addr_t Signal::GetEventMailboxPtr() const {
    return event_->GetEventMailboxPtr();
}

unsigned Signal::GetEventId() const { return event_->GetEventID(); }

SignalPool::SignalPool(MemoryManager *mm) : mm_(mm) {
    // Bound the number of the active signals to KFD_SIGNAL_EVENT_LIMIT
    enum {
        kSignalPoolBytesSize = KFD_SIGNAL_EVENT_LIMIT * sizeof(amd_signal_t)
    };
    gtt_ = mm_->NewGTTMemory(kSignalPoolBytesSize, true);
    slab_.Initialize(absl::Span<char>(
        reinterpret_cast<char *>(gtt_->GetBuffer()), gtt_->GetSize()));
}

SignalPool::~SignalPool() { Destroy(); }

Signal *SignalPool::GetSignal() { return new Signal(this); }

void SignalPool::PutSignal(Signal *signal) { delete signal; }

void SignalPool::Destroy() {
    if (gtt_) {
        gtt_.reset();
    }
}

} // namespace ocl::hsa
