#pragma once

#include "memory_manager.h"
#include "slab_allocator.h"
#include "types.h"

struct amd_signal_s;

namespace ocl::hsa {

class Event;
class SignalPool;
class MemoryManager;

//
// A signal is essentially an atomic variable that maps into the address space
// of both GPU and CPU. The application can poll on it or the kernel can
// optionally wait on a signal via interrupts, providing a underlying mechanism
// to implement synchronization barriers.
//
class Signal {
  public:
    friend class SignalPool;
    gpu_addr_t GetHandle() const;
    Signal(const Signal &) = delete;
    Signal &operator=(const Signal &) = delete;

    // Block until the value of signal becomes zero.
    bool Barrier();
    // Compare to ROCR, the caller needs to call SetEvent() to explicitly notify
    // the device
    void Set(int64_t value);

    gpu_addr_t ValueLocation() const;
    gpu_addr_t GetEventMailboxPtr() const;
    unsigned GetEventId() const;

    ~Signal();

  private:
    explicit Signal(SignalPool *parent);
    absl::Status CreateSignalEvent();
    SignalPool *parent_;
    std::unique_ptr<Event> event_;
    amd_signal_s *data_;
};

//
// A pool of signals to prevent memory fragmentation.
class SignalPool {
  public:
    friend class Signal;
    explicit SignalPool(MemoryManager *mm);
    SignalPool(const SignalPool &) = delete;
    ~SignalPool();
    SignalPool &operator=(const SignalPool &) = delete;
    Signal *GetSignal();
    void PutSignal(Signal *);
    void Destroy();

  private:
    MemoryManager *mm_;
    std::unique_ptr<Memory> gtt_;
    SlabAllocator<amd_signal_s> slab_;
};

} // namespace ocl::hsa
