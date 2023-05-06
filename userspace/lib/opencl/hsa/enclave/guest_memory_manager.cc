#include "guest_memory.h"

#include "opencl/hsa/assert.h"
#include "opencl/hsa/bump_allocator.h"
#include "opencl/hsa/memory_manager.h"
#include "opencl/hsa/platform.h"

#include "opencl/hsa/runtime_options.h"
#include "utils/align.h"

#include <hsa/kfd_ioctl.h>
#include <memory>
#include <sys/mman.h>

namespace ocl::hsa::enclave {

static const uint32_t kIOCFlagsBase =
    KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
    KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;

/*
 * Note that the event and doorbell pages only reside in the host side.
 * VRAM is expected to be allocated for once.
 */
class EnclaveGuestMemoryManager : public MemoryManager {
  public:
    explicit EnclaveGuestMemoryManager(Device *dev, absl::Span<char> gtt_space,
                                       absl::Span<char> vram_space);
    virtual std::unique_ptr<Memory> NewGTTMemory(size_t size,
                                                 bool uncached) override;
    virtual std::unique_ptr<Memory> NewDeviceMemory(size_t size) override;
    virtual std::unique_ptr<Memory> NewRingBuffer(size_t size) override;
    virtual std::unique_ptr<Memory> NewEventPage() override;
    std::unique_ptr<Memory> NewDoorbell(size_t size, uint64_t mmap_offset);

  protected:
    enum AddressSpace {
        kGTT,
        kVRAM,
    };
    std::unique_ptr<Memory> NewMemoryImpl(size_t size, size_t alignment,
                                          AddressSpace space, int mmap_flag,
                                          unsigned ioc_flags);
    BumpAllocator gtt_alloc_;
    BumpAllocator vram_alloc_;
};

EnclaveGuestMemoryManager::EnclaveGuestMemoryManager(
    Device *dev, absl::Span<char> gtt_space, absl::Span<char> vram_space)
    : MemoryManager(dev) {
    auto gtt_base = reinterpret_cast<uintptr_t>(gtt_space.data());
    auto vram_base = reinterpret_cast<uintptr_t>(vram_space.data());
    HSA_ASSERT(gtt_base % Memory::kGPUHugePageSize == 0);
    HSA_ASSERT(vram_base % Memory::kGPUHugePageSize == 0);

    gtt_alloc_.Initialize(gtt_base, gtt_space.size());
    vram_alloc_.Initialize(vram_base, vram_space.size());
}

std::unique_ptr<Memory>
EnclaveGuestMemoryManager::NewMemoryImpl(size_t size, size_t alignment,
                                         AddressSpace space, int mmap_flag,
                                         unsigned ioc_flags) {
    size = gpumpc::AlignUp(size, Device::kPageSize);
    HSA_ASSERT(size % Device::kPageSize == 0);
    unsigned gpu_id = dev_->GetGPUID();
    auto alloc = space == kGTT ? &gtt_alloc_ : &vram_alloc_;
    void *buf = reinterpret_cast<void *>(alloc->AllocateAlign(size, alignment));
    HSA_ASSERT(buf);

    auto ret = std::make_unique<EnclaveGuestMemory>(
        static_cast<EnclaveGuestDevice *>(dev_), buf, size);
    bool map_remote_pages = GetRuntimeOptions()->MapRemotePhysicalPage() && space == kGTT;
    auto stat = ret->AllocGPUMemory(map_remote_pages, gpu_id, ioc_flags,
                                    (uintptr_t)buf);
    if (!stat.ok()) {
        return nullptr;
    }

    // The caller might just allocate from the address space when it does not
    // pass in the kMapIntoHostAddressSpace flag. Don't care for now.
    if (mmap_flag & kClearHost) {
        memset(buf, 0, size);
    }

    stat = ret->MapGPUMemory();
    if (!stat.ok()) {
        return nullptr;
    }
    return ret;
}

std::unique_ptr<Memory>
EnclaveGuestMemoryManager::NewDeviceMemory(size_t size) {
    auto ioc_flags = kIOCFlagsBase | KFD_IOC_ALLOC_MEM_FLAGS_VRAM;
    return NewMemoryImpl(size, Memory::kGPUHugePageSize, AddressSpace::kVRAM, 0,
                         ioc_flags);
}

std::unique_ptr<Memory> EnclaveGuestMemoryManager::NewGTTMemory(size_t size,
                                                                bool uncached) {
    auto ioc_flags = kIOCFlagsBase | KFD_IOC_ALLOC_MEM_FLAGS_USERPTR;
    if (uncached) {
        ioc_flags |= KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED;
    }
    return NewMemoryImpl(size, Memory::kGPUHugePageSize, AddressSpace::kGTT,
                         kMapIntoHostAddressSpace, ioc_flags);
}

std::unique_ptr<Memory> EnclaveGuestMemoryManager::NewRingBuffer(size_t size) {
    static const uint32_t kIOCFlag = kIOCFlagsBase |
                                     KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                                     KFD_IOC_ALLOC_MEM_FLAGS_USERPTR;
    return NewMemoryImpl(size, Memory::kGPUHugePageSize, AddressSpace::kGTT,
                         kMapIntoHostAddressSpace | kClearHost, kIOCFlag);
}

std::unique_ptr<Memory> EnclaveGuestMemoryManager::NewEventPage() {
    enum {
        kEventPageSize = KFD_SIGNAL_EVENT_LIMIT * sizeof(uint64_t),
        kEventNumPage = kEventPageSize / Device::kPageSize,
    };
    static const unsigned kEventPageIOCFlag = kIOCFlagsBase |
                                              KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                                              KFD_IOC_ALLOC_MEM_FLAGS_GTT;
    auto ret = NewMemoryImpl(kEventPageSize, Memory::kGPUHugePageSize,
                             AddressSpace::kGTT, 0, kEventPageIOCFlag);
    if (!ret) {
        return ret;
    }
    static_cast<EnclaveGuestMemory *>(ret.get())->SetResourceScopeFlag(
        EnclaveGuestMemory::kUnmanagedBO);
    return ret;
}

std::unique_ptr<Memory>
EnclaveGuestMemoryManager::NewDoorbell(size_t size, uint64_t mmap_offset) {
    HSA_ASSERT(0 && "Unreachable");
    return std::unique_ptr<Memory>();
}

std::unique_ptr<MemoryManager>
NewEnclaveGuestMemoryManager(Device *dev, absl::Span<char> gtt_space,
                             absl::Span<char> vram_space) {
    return std::unique_ptr<MemoryManager>(
        new EnclaveGuestMemoryManager(dev, gtt_space, vram_space));
}

} // namespace ocl::hsa::enclave
