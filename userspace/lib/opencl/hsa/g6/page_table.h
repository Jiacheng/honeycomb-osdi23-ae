#pragma once

#include "opencl/hsa/types.h"
#include <vector>
#include <cstddef>

namespace ocl::g6 {

using ocl::hsa::gpu_addr_t;

class PageAllocator {
  public:
    struct Page {
        unsigned long *cpu_base;
        gpu_addr_t gpu_base;
    };
    enum { kPageSize = 4096, kPageShift = 12 };
    Page *AllocatePage();
    void Initialize(void *cpu_addr_base, gpu_addr_t gpu_base, size_t num_pages);
    void Initialize(std::vector<Page> &&pages);
    Page *LookupPageByGPUAddr(gpu_addr_t phys_addr);
    unsigned GetNextFreePageIndex() const { return idx_; }
    const std::vector<Page> &GetPages() const { return pages_; }

    PageAllocator() = default;
    PageAllocator(PageAllocator &&) = default;
    PageAllocator(const PageAllocator &) = delete;
    PageAllocator &operator=(const PageAllocator &) = delete;

  private:
    std::vector<Page> pages_;
    unsigned idx_;
};

class PageTableManager {
  public:
    enum PageLevel {
        kVMLevelPDB2,
        kVMLevelPDB1,
        kVMLevelPDB0,
        kVMLevelPTB,
    };
    // Map VRAM into the address space
    void MapVRAM(uintptr_t va_start, gpu_addr_t phys_addr, size_t size,
                 uint64_t flags);
    // Map a number of GTT pages in the host memory to the GPU
    void MapGTT(uintptr_t va_start, size_t count, const uintptr_t *pages,
                uint64_t flags);
    explicit PageTableManager();
    PageAllocator *GetAllocator() { return &alloc_; }

    PageTableManager(const PageTableManager &) = delete;
    PageTableManager &operator=(const PageTableManager &) = delete;

  private:
    static int GetPageLevelShift(int level);
    static void ClearPageTable(int level, unsigned long *page);
    PageAllocator::Page *GetOrInitializeRootPageTable();

    PageAllocator alloc_;
    PageAllocator::Page *root_;
};

} // namespace ocl::g6
