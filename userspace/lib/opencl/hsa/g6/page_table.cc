#include "page_table.h"
#include "../assert.h"
#include <stdexcept>
#include <cstring>
#include <algorithm>

#define AMDGPU_PTE_VALID (1ULL << 0)
#define AMDGPU_PTE_SYSTEM (1ULL << 1)
#define AMDGPU_PTE_SNOOPED (1ULL << 2)
/* VI only */
#define AMDGPU_PTE_EXECUTABLE	(1ULL << 4)

#define AMDGPU_PTE_READABLE	(1ULL << 5)
#define AMDGPU_PTE_WRITEABLE	(1ULL << 6)

#define AMDGPU_PTE_FRAG(x)	((x & 0x1fULL) << 7)

/* TILED for VEGA10, reserved for older ASICs  */
#define AMDGPU_PTE_PRT		(1ULL << 51)

/* PDE is handled as PTE for VEGA10 */
#define AMDGPU_PDE_PTE		(1ULL << 54)

#define AMDGPU_MTYPE_NC 0
#define AMDGPU_MTYPE_CC 2

/* gfx10 */
#define AMDGPU_PTE_MTYPE_NV10(a)       ((uint64_t)(a) << 48)
#define AMDGPU_PTE_MTYPE_NV10_MASK     AMDGPU_PTE_MTYPE_NV10(7ULL)


namespace ocl::g6 {

using Page = PageAllocator::Page;

enum {
	kPageTableEntries = 512,
};

class PageTableEntry {
  public:
    enum : size_t {
        kPhysAddrMask = ((1ull << 48) - 1) & ~((1ull << 12) - 1),
	kAccessFlagsMask = (1 << 7) - 1,
    };
    explicit PageTableEntry(unsigned long v) : v_(v) {}

    static PageTableEntry NewPDE(gpu_addr_t phys_addr) {
	    auto v = phys_addr & kPhysAddrMask;
	    v |= AMDGPU_PTE_VALID | AMDGPU_PTE_SYSTEM;
	    return PageTableEntry(v);
    };

    static PageTableEntry NewVRAM2MEntry(gpu_addr_t phys_addr, uint64_t flags) {
	    auto v = phys_addr & kPhysAddrMask;
	    v |= flags & kAccessFlagsMask;
	    v |= AMDGPU_PDE_PTE;
	    v |= AMDGPU_PTE_FRAG(9); 
	    return PageTableEntry(v);
    };

    static PageTableEntry NewGTTEntry(gpu_addr_t phys_addr, uint64_t flags) {
	    auto v = phys_addr & kPhysAddrMask;
	    v |= flags & kAccessFlagsMask;
            v |= 3ull << 48; 
	    return PageTableEntry(v);
    };

    gpu_addr_t GetPhysAddr() const {
        return v_ & kPhysAddrMask;
    }
    bool IsValid() const { return v_ & AMDGPU_PTE_VALID; }
    unsigned long Get() const { return v_; }

  private:
    unsigned long v_;
};


Page *PageAllocator::AllocatePage() {
    if (idx_ >= pages_.size()) {
        throw std::invalid_argument("No page");
    }
    auto id = idx_++;
    return &pages_[id];
}

void PageAllocator::Initialize(void *cpu_addr_base, gpu_addr_t gpu_base,
                               size_t num_pages) {
    idx_ = 0;
    pages_.resize(num_pages);
    for (size_t i = 0; i < num_pages; ++i) {
        Page *p = &pages_[i];
        p->cpu_base = reinterpret_cast<unsigned long *>(cpu_addr_base) +
                      i * kPageSize / sizeof(unsigned long);
        p->gpu_base = gpu_base + i * kPageSize;
    }
}

void PageAllocator::Initialize(std::vector<Page> &&pages) {
    idx_ = 0;
    pages_ = std::move(pages);
}

Page * PageAllocator::LookupPageByGPUAddr(gpu_addr_t phys_addr) {
    auto it = std::find_if(pages_.begin(), pages_.end(), [phys_addr](const auto &p) {
        return p.gpu_base == phys_addr;
    });
    return it == pages_.end() ? nullptr : &*it;
}

PageTableManager::PageTableManager()
    : root_(nullptr) {}

int PageTableManager::GetPageLevelShift(int level) {
    switch (level) {
    case kVMLevelPDB2:
    case kVMLevelPDB1:
    case kVMLevelPDB0:
        return 9 * (kVMLevelPDB0 - level) + 9;
    case kVMLevelPTB:
        return 0;
    default:
        return ~0;
    }
}

void PageTableManager::ClearPageTable(int level, unsigned long *page) {
	unsigned long flags = 0;
	switch (level) {
		case kVMLevelPDB2:
		case kVMLevelPDB1:
			flags |= AMDGPU_PDE_PTE;
			break;
		case kVMLevelPDB0:
			flags &= ~AMDGPU_PDE_PTE;
			break;
		case kVMLevelPTB:
			flags = AMDGPU_PTE_EXECUTABLE;
			break;
		default:
			break;
	}
	for (int i = 0; i < 0x200; i++) {
		page[i] = flags;
	}
}

void PageTableManager::MapVRAM(uintptr_t va_start, gpu_addr_t phys_addr,
                               size_t size, uint64_t flags) {
    HSA_ASSERT(va_start % (2 << 20) == 0);
    HSA_ASSERT(size == (2 << 20));
    auto pgt = GetOrInitializeRootPageTable();
    auto level = (int)kVMLevelPDB2;
    while (level < kVMLevelPDB0) {
        auto idx = (va_start >>
                    (GetPageLevelShift(level) + PageAllocator::kPageShift)) &
                   0x1ff;
        Page *next = nullptr;
        PageTableEntry e(pgt->cpu_base[idx]);
        if (e.IsValid()) {
            next = alloc_.LookupPageByGPUAddr(e.GetPhysAddr());
        } else {
            next = alloc_.AllocatePage();
	    ClearPageTable(level + 1, next->cpu_base);
            PageTableEntry pde = PageTableEntry::NewPDE(next->gpu_base);
            pgt->cpu_base[idx] = pde.Get();
        }
        pgt = next;
        level++;
    }

    // Insert into PDB0, as it is a 2MB chunk
    auto idx = (va_start >>
		    (GetPageLevelShift(kVMLevelPDB0) + PageAllocator::kPageShift)) &
	    0x1ff;
    pgt->cpu_base[idx] = PageTableEntry::NewVRAM2MEntry(phys_addr, flags).Get(); 
}

// Map a number of GTT pages in the host memory to the GPU
void PageTableManager::MapGTT(uintptr_t va_start, size_t count,
                              const uintptr_t *pages, uint64_t flags) {
    auto remaining = count;
    auto start = va_start;
    auto pgt = GetOrInitializeRootPageTable();
    auto level = (int)kVMLevelPDB2;
    size_t page_idx = 0;

    while (remaining) {
	    while (level < kVMLevelPTB) {
		    auto idx = (start >>
				    (GetPageLevelShift(level) + PageAllocator::kPageShift)) &
			    0x1ff;
		    Page *next = nullptr;
		    PageTableEntry e(pgt->cpu_base[idx]);
		    if (e.IsValid()) {
			    next = alloc_.LookupPageByGPUAddr(e.GetPhysAddr());
		    } else {
			    next = alloc_.AllocatePage();
			    ClearPageTable(level + 1, next->cpu_base);
			    PageTableEntry pde = PageTableEntry::NewPDE(next->gpu_base);
			    pgt->cpu_base[idx] = pde.Get();
		    }
		    pgt = next;
		    level++;
	    }
	    auto idx = (start >>
			    (GetPageLevelShift(kVMLevelPTB) + PageAllocator::kPageShift)) &
		    0x1ff;
	    auto batch_size = std::min(kPageTableEntries - idx, remaining); 
	    for (unsigned i = 0; i < batch_size; i++) {
		    pgt->cpu_base[idx + i] = PageTableEntry::NewGTTEntry(pages[page_idx + i], flags).Get(); 
	    }
	    page_idx += batch_size;
	    remaining -= batch_size;
	    start += batch_size * PageAllocator::kPageSize;
    }
}

PageAllocator::Page *PageTableManager::GetOrInitializeRootPageTable() {
    if (root_) {
        return root_;
    }
    root_ = alloc_.AllocatePage();
    ClearPageTable((int)kVMLevelPDB2, root_->cpu_base);
    return root_;
}
} // namespace ocl::g6
