#pragma once

#include "opencl/hsa/bump_allocator.h"
#include "opencl/hsa/memory_manager.h"

namespace ocl::hsa {
class G6Device;

class G6MemoryManager : public MemoryManager {
  public:
    friend class G6Device;
    explicit G6MemoryManager(Device *dev);
    virtual std::unique_ptr<Memory> NewGTTMemory(size_t size,
                                                 bool uncached) override;
    virtual std::unique_ptr<Memory> NewDeviceMemory(size_t size) override;
    virtual std::unique_ptr<Memory> NewRingBuffer(size_t size) override;
    virtual std::unique_ptr<Memory> NewEventPage() override;

  protected:
    size_t GetGTTBumpAllocOffset() const;
    BumpAllocator gtt_alloc_;
    BumpAllocator vram_alloc_;
};

} // namespace ocl::hsa
