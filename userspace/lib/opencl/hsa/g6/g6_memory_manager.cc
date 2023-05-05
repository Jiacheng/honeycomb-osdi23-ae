#include "g6_memory_manager.h"
#include "g6_device.h"
#include "g6_memory.h"
#include "page_table.h"

#include "utils/align.h"

#include "opencl/hsa/assert.h"
#include "opencl/hsa/platform.h"
#include "opencl/hsa/utils.h"

#include "hsa/amdgpu_vm.h"

namespace ocl::hsa {

G6MemoryManager::G6MemoryManager(Device *dev) : MemoryManager(dev) {
    auto gdev = static_cast<G6Device *>(dev_);
    auto gtt_region = gdev->GetGTTRegion();
    gtt_alloc_.Initialize((uintptr_t)gdev->GetGTTVABase(),
                          gtt_region->size() * Device::kPageSize);
    vram_alloc_.Initialize(gdev->GetUserVRAMPhysAddr(),
                           gdev->GetUserVRAMSize());
}

size_t G6MemoryManager::GetGTTBumpAllocOffset() const {
    return gtt_alloc_.GetOffset();
}

std::unique_ptr<Memory> G6MemoryManager::NewDeviceMemory(size_t size) {
    HSA_ASSERT(size % Device::kPageSize == 0);
    auto va = reinterpret_cast<gpu_addr_t>(
        Mmap(nullptr, size, Memory::kGPUHugePageSize, 0));
    if (!va) {
        return nullptr;
    }
    auto phys = vram_alloc_.Allocate(size);
    if (!phys) {
        return nullptr;
    }
    auto gdev = static_cast<G6Device *>(dev_);
    auto ptm = gdev->GetPageTableManager();
    ptm->MapVRAM(va, phys, size,
                 AMDGPU_PTE_EXECUTABLE | AMDGPU_PTE_READABLE |
                     AMDGPU_PTE_WRITEABLE | AMDGPU_PTE_VALID);

    return std::unique_ptr<Memory>(new G6Memory(va, size));
}

std::unique_ptr<Memory> G6MemoryManager::NewGTTMemory(size_t size,
                                                      bool uncached) {
    auto gdev = static_cast<G6Device *>(dev_);
    auto gtt_region = gdev->GetGTTRegion();
    auto ptm = gdev->GetPageTableManager();

    auto aligned_size = gpumpc::AlignUp(size, Device::kPageSize);
    auto va = gtt_alloc_.Allocate(aligned_size);
    if (!va) {
        return nullptr;
    }
    auto num_pages = aligned_size / Device::kPageSize;

    ptm->MapGTT(va, num_pages, gtt_region->data(),
                AMDGPU_PTE_VALID | AMDGPU_PTE_SYSTEM | AMDGPU_PTE_SNOOPED |
                    AMDGPU_PTE_READABLE | AMDGPU_PTE_WRITEABLE |
                    AMDGPU_PTE_EXECUTABLE);
    auto mem = std::unique_ptr<Memory>(new G6Memory(va, aligned_size));
    gtt_region->erase(gtt_region->begin(), gtt_region->begin() + num_pages);
    return mem;
}

std::unique_ptr<Memory> G6MemoryManager::NewRingBuffer(size_t size) {
    return NewGTTMemory(size, true);
}

std::unique_ptr<Memory> G6MemoryManager::NewEventPage() {
    enum {
        kEventPageSize = KFD_SIGNAL_EVENT_LIMIT * sizeof(uint64_t),
        kEventNumPage = kEventPageSize / Device::kPageSize,
    };
    return NewGTTMemory(kEventPageSize, true);
}

} // namespace ocl::hsa
