#include "kfd_memory_manager.h"
#include "kfd_memory.h"
#include "opencl/hsa/assert.h"
#include "opencl/hsa/platform.h"
#include "utils/align.h"

#include <hsa/kfd_ioctl.h>
#include <sys/mman.h>

namespace ocl::hsa {

static const uint32_t kIOCFlagsBase =
    KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
    KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;

KFDMemoryManager::KFDMemoryManager(Device *dev) : MemoryManager(dev) {
    sys_ptr_ = kSysStart;
}

std::unique_ptr<Memory> KFDMemoryManager::NewMemoryImpl(void *addr, size_t size,
                                                        size_t alignment,
                                                        int mmap_flag,
                                                        unsigned ioc_flags) {
    size = gpumpc::AlignUp(size, Device::kPageSize);
    HSA_ASSERT(size % Device::kPageSize == 0);
    unsigned gpu_id = dev_->GetGPUID();
    auto buf = Mmap(addr, size, alignment, mmap_flag);
    auto ret = std::make_unique<KFDMemory>(buf, size);
    auto stat = ret->AllocGPUMemory(gpu_id, ioc_flags, (uintptr_t)buf);
    if (!stat.ok()) {
        return nullptr;
    }

    stat = ret->MapGPUMemory();
    if (!stat.ok()) {
        return nullptr;
    }
    return ret;
}

std::unique_ptr<Memory> KFDMemoryManager::NewDeviceMemory(size_t size) {
    auto ioc_flags = kIOCFlagsBase | KFD_IOC_ALLOC_MEM_FLAGS_VRAM;
    return NewMemoryImpl(nullptr, size, Memory::kGPUHugePageSize, 0, ioc_flags);
}

std::unique_ptr<Memory> KFDMemoryManager::NewGTTMemory(size_t size,
                                                       bool uncached) {
    auto ioc_flags = kIOCFlagsBase | KFD_IOC_ALLOC_MEM_FLAGS_USERPTR;
    if (uncached) {
        ioc_flags |= KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED;
    }
    return NewMemoryImpl(nullptr, size, Memory::kGPUHugePageSize,
                         kMapIntoHostAddressSpace, ioc_flags);
}

std::unique_ptr<Memory> KFDMemoryManager::NewSystemMemory(size_t size,
                                                          size_t alignment,
                                                          int mmap_flags,
                                                          uint32_t ioc_flags) {
    void *addr = nullptr;

    if (strict_layout_) {
        addr = reinterpret_cast<void *>(gpumpc::AlignUp(sys_ptr_, alignment));

        // update sys_ptr for next alloc
        const auto map_size = gpumpc::AlignUp(size, alignment);
        sys_ptr_ = reinterpret_cast<uintptr_t>(addr) + map_size;

        mmap_flags |= kFixed;
    }

    return NewMemoryImpl(addr, size, alignment, mmap_flags, ioc_flags);
}

std::unique_ptr<Memory> KFDMemoryManager::NewRingBuffer(size_t size) {
    static const uint32_t kIOCFlag = kIOCFlagsBase |
                                     KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                                     KFD_IOC_ALLOC_MEM_FLAGS_USERPTR;
    return NewSystemMemory(size, Memory::kGPUHugePageSize,
                           kMapIntoHostAddressSpace | kClearHost, kIOCFlag);
}

std::unique_ptr<Memory> KFDMemoryManager::NewEventPage() {
    enum {
        kEventPageSize = KFD_SIGNAL_EVENT_LIMIT * sizeof(uint64_t),
        kEventNumPage = kEventPageSize / Device::kPageSize,
    };
    static const unsigned kEventPageIOCFlag = kIOCFlagsBase |
                                              KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                                              KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
                                              KFD_IOC_ALLOC_MEM_FLAGS_GTT;

    auto ret = NewSystemMemory(kEventPageSize, Memory::kGPUHugePageSize, 0,
                               kEventPageIOCFlag);
    if (!ret) {
        return ret;
    }
    static_cast<KFDMemory *>(ret.get())->SetResourceScopeFlag(
        KFDMemory::kUnmanagedBO);
    return ret;
}

std::unique_ptr<Memory> KFDMemoryManager::NewDoorbell(size_t size,
                                                      uint64_t mmap_offset) {
    static const unsigned kDoorbellIOCFlag =
        KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
        KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE |
        KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL;

    auto doorbell = NewSystemMemory(size, Memory::kGPUHugePageSize,
                                    kMapIntoHostAddressSpace, kDoorbellIOCFlag);
    if (!doorbell) {
        return nullptr;
    }

    int kfd_fd = Platform::Instance().GetKFDFD();
    // Remap the doorbell page to the user space
    auto buf = mmap(doorbell->GetBuffer(), size, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_FIXED, kfd_fd, mmap_offset);
    if (buf == MAP_FAILED) {
        return nullptr;
    }
    return doorbell;
}

} // namespace ocl::hsa
