#pragma once

#include "types.h"
#include <absl/status/status.h>

namespace ocl::hsa {
class Device;

//
// The Memory class represents a contiguous address range on the
// device. In general there are three types of memory:
//
// (1) User-level buffer that only resides on the device (i.e., VRAM).
// (2) User-level buffer that is backed by the main memory, but is mapped into
// both the user-level process and the device (i.e., GTT).
// (3) Shared, system-wide buffer that is mapped into all processes, like
// events, signals and doorbells, which might require special handling on
// destructions.
//
// The caller is expected to manage the virtual address space and the lifecycles
// of the memory.
//
class Memory {
  public:
    enum : size_t {
        kGPUHugePageSize = 2 << 20,
        kPageSize = 4096,
    };
    enum Domain {
        kDomainVRAM,
        kDomainGTT,
    };

    enum ResourceScope {
        kNone = 0,
        // The event page is expected to be automatically freed when the process
        // is terminated.
        kUnmanagedBO = 1,
        // The enclave host agent manages its address space itself thus the
        // mapping should not be freed.
        kUnmanagedMmap = 2,
    };

    virtual ~Memory() = default;
    virtual size_t GetSize() const = 0;
    //
    // Return the underlying handle for the GPU memory.
    virtual uint64_t GetHandle() const = 0;
    //
    // Return the CPU address if the memory is a piece of GTT memory and is
    // mapped into the user-level process.
    virtual void *GetBuffer() const = 0;

    // Return the GPU address of the memory.
    //
    // The ROCM driver uses mmap() to manage the virtual address spaces for all
    // GPUs and the user-level process, thus (1) a piece of memory that is
    // shared across multiple GPUs has a unique address, and (2) a piece of GTT
    // memory has identical a CPU and GPU address.
    //
    // We introduce a separate API here to support a fixed memory layout for GPU
    // programs. The memory layout can help simplify the proofs on showing all
    // memory accesses are safe and not leaking information.
    virtual gpu_addr_t GetGPUAddress() const = 0;

    virtual absl::Status Destroy() = 0;

    Memory(const Memory &) = delete;
    Memory &operator=(const Memory &) = delete;

  protected:
    Memory() = default;
};

class MemoryManager;

class MemoryManager {
  public:
    enum AllocateFlag {
        kAllocateFlagClearHost = 1,
        kAllocateFlagUncached = 1 << 1,
    };
    virtual ~MemoryManager() = default;
    MemoryManager(const MemoryManager &) = delete;
    MemoryManager &operator=(const MemoryManager &) = delete;

    //
    // Prefer 2MB thunks to alleviate the pressures of TLB.
    virtual std::unique_ptr<Memory> NewGTTMemory(size_t size,
                                                 bool uncached) = 0;
    virtual std::unique_ptr<Memory> NewDeviceMemory(size_t size) = 0;
    virtual std::unique_ptr<Memory> NewRingBuffer(size_t size) = 0;
    virtual std::unique_ptr<Memory> NewEventPage() = 0;

    Device *GetDevice() const { return dev_; }
    static void *Mmap(void *addr, size_t size, size_t alignment, int flags);

  protected:
    enum MmapFlag {
        kMapIntoHostAddressSpace = 1,
        kClearHost = 1 << 1,
        kFixed = 1 << 2,
    };

    explicit MemoryManager(Device *dev);
    Device *dev_;
    bool strict_layout_;
};

} // namespace ocl::hsa
