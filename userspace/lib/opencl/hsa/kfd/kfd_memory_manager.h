#pragma once

#include "opencl/hsa/memory_manager.h"

namespace ocl::hsa {
class Device;

class KFDMemoryManager : public MemoryManager {
  public:
    explicit KFDMemoryManager(Device *dev);
    virtual std::unique_ptr<Memory> NewGTTMemory(size_t size,
                                                 bool uncached) override;
    virtual std::unique_ptr<Memory> NewDeviceMemory(size_t size) override;
    virtual std::unique_ptr<Memory> NewRingBuffer(size_t size) override;
    virtual std::unique_ptr<Memory> NewEventPage() override;
    std::unique_ptr<Memory> NewDoorbell(size_t size, uint64_t mmap_offset);

  protected:
    std::unique_ptr<Memory> NewMemoryImpl(void *addr, size_t size,
                                          size_t alignment, int mmap_flag,
                                          unsigned ioc_flags);
    std::unique_ptr<Memory> NewSystemMemory(size_t size, size_t alignment,
                                            int mmap_flags, uint32_t ioc_flags);

  private:
    enum RegionConstant {
        // userspace program can not use kernel space va (e.g. 1UL << 63)
        kSysStart = (1UL << 47) - (1UL << 31),
    };
    uintptr_t sys_ptr_;
};

} // namespace ocl::hsa
