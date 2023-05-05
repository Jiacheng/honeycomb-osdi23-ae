#include "g6_device.h"
#include "g6_ioctl.h"
#include "g6_memory_manager.h"
#include "g6_queue.h"
#include "hsa/amdgpu_vm.h"
#include "opencl/hsa/types.h"
#include "opencl/hsa/utils.h"

#include <algorithm>

#include <sys/mman.h>

namespace ocl::hsa {

G6Device::G6Device(unsigned node_id, unsigned gpu_id)
    : KFDDevice(node_id, gpu_id), user_vram_phys_addr_(0),
      doorbell_phys_addr_(0), doorbell_slice_size_(0), doorbell_va_(0) {
    std::fill_n(pgt_region_gpu_addr_,
                sizeof(pgt_region_gpu_addr_) / sizeof(gpu_addr_t), 0);
}

absl::Status G6Device::AcquireVMG6() {
    using namespace g6;
    int kfd_fd = Platform::Instance().GetKFDFD();
    kfd_ioctl_acquire_vm_g6_args args = {
        .gpu_id = GetGPUID(),
    };
    int r = kmtIoctl(kfd_fd, AMDKFD_IOC_ACQUIRE_VM_G6, &args);
    if (r) {
        return absl::InvalidArgumentError(
            "Cannot acquire information for the G6 VM");
    }

    user_vram_phys_addr_ = args.user_vram_gpu_addr;
    std::copy(args.pgt_region_gpu_addr,
              args.pgt_region_gpu_addr + kReservedPGTPage,
              pgt_region_gpu_addr_);
    doorbell_phys_addr_ = args.doorbell_gpu_addr;
    doorbell_slice_size_ = args.doorbell_slice_size;
    // FIXME: We assume that the region is continous.
    gtt_phys_addr_ = args.gtt_region_gpu_addr;
    gtt_region_size_ = args.gtt_region_size;

    pgt_buf_.resize(kReservedPGTPage * kPageSize);
    auto alloc = ptm_.GetAllocator();
    std::vector<PageAllocator::Page> pages;
    for (int i = 0; i < kReservedPGTPage; ++i) {
        PageAllocator::Page p = {
            .cpu_base = reinterpret_cast<unsigned long *>(pgt_buf_.data() +
                                                          i * kPageSize),
            .gpu_base = pgt_region_gpu_addr_[i],
        };
        pages.push_back(p);
    }
    alloc->Initialize(std::move(pages));
    auto stat = InitializeDoorbell();
    if (!stat.ok()) {
        return stat;
    }
    stat = MapGTTMemoryToHost();
    SetMemoryManager(std::unique_ptr<MemoryManager>(new G6MemoryManager(this)));
    return stat;
}

std::unique_ptr<DeviceQueue> G6Device::CreateSDMAQueue() {
    return std::unique_ptr<DeviceQueue>(new G6SDMAQueue(this));
}

std::unique_ptr<DeviceQueue> G6Device::CreateAQLQueue() {
    return std::unique_ptr<DeviceQueue>(new G6AQLQueue(this));
}

absl::Status G6Device::InitializeDoorbell() {
    int kfd_fd = Platform::Instance().GetKFDFD();
    // The KFD device allocates and maps the doorbell into the userspace
    // lazily, i.e., at the time when the first queue is created.
    // Here we eagerly maps it in.
    doorbell_va_ = reinterpret_cast<uintptr_t>(GetMemoryManager()->Mmap(
        nullptr, GetDoorbellPageSize(), Memory::kGPUHugePageSize, 0));
    std::vector<uintptr_t> doorbell_pages(GetDoorbellPageSize() / 4096);
    for (size_t i = 0; i < doorbell_pages.size(); ++i) {
        doorbell_pages[i] = doorbell_phys_addr_ + i * 4096;
    }
    ptm_.MapGTT(doorbell_va_, doorbell_pages.size(), doorbell_pages.data(),
                AMDGPU_PTE_VALID | AMDGPU_PTE_SYSTEM | AMDGPU_PTE_SNOOPED |
                    AMDGPU_PTE_READABLE | AMDGPU_PTE_WRITEABLE);

    auto mmap_offset = KFD_MMAP_GPU_ID(GetGPUID()) | KFD_MMAP_TYPE_DOORBELL;

    // Remap the doorbell page to the user space
    auto buf = mmap(reinterpret_cast<void *>(doorbell_va_),
                    GetDoorbellPageSize(), PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_FIXED, kfd_fd, mmap_offset);
    if (buf == MAP_FAILED) {
        return absl::ResourceExhaustedError(
            "Cannot map the doorbell into the userspace");
    }
    return absl::OkStatus();
}

absl::Status G6Device::MapGTTMemoryToHost() {
    // Map in the whole GTT memory segment to the host
    int kfd_fd = Platform::Instance().GetKFDFD();
    auto mmap_offset = KFD_MMAP_GPU_ID(GetGPUID()) | KFD_MMAP_TYPE_MMIO;
    gtt_va_ = mm_->Mmap(nullptr, gtt_region_size_, kHugeGPUPageSize, 1);
    if (!gtt_va_) {
        return absl::ResourceExhaustedError(
            "cannot map gtt region into the host");
    }

    // Remap the GTT page to the user space
    gtt_va_ = mmap(gtt_va_, gtt_region_size_, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_FIXED, kfd_fd, mmap_offset);
    if (gtt_va_ == MAP_FAILED) {
        return absl::ResourceExhaustedError(
            "cannot map gtt region into the host");
    }

    for (unsigned i = 0; i < gtt_region_size_ / kPageSize; i++) {
        auto p = gtt_phys_addr_ + i * kPageSize;
        gtt_regions_.push_back(p);
    }

    return absl::OkStatus();
}

absl::Status G6Device::Initialize() {
    auto stat = KFDDevice::Initialize();
    if (!stat.ok()) {
        return stat;
    }
    return AcquireVMG6();
}

absl::Span<const gpu_addr_t> G6Device::GetPageTableRegionPhysAddr() const {
    return absl::MakeConstSpan(pgt_region_gpu_addr_, kReservedPGTPage);
}

absl::Status
G6Device::DumpPageTableBO(std::vector<kfd_ioctl_dump_page_table_metadata> *md,
                          std::vector<char> *page_tables) {
    enum { kMaxPageTableNum = 16 };
    page_tables->resize(kMaxPageTableNum * kPageSize);
    md->resize(kMaxPageTableNum);
    int kfd_fd = Platform::Instance().GetKFDFD();
    struct kfd_ioctl_dump_page_table_args arg = {
        .gpu_id = GetGPUID(),
        .entry = kMaxPageTableNum,
        .metadata_ptr = reinterpret_cast<uintptr_t>(md->data()),
        .pagetable_ptr = reinterpret_cast<uintptr_t>(page_tables->data()),
    };

    int r = kmtIoctl(kfd_fd, AMDKFD_IOC_DUMP_PAGE_TABLE, &arg);
    if (r < 0) {
        return absl::InvalidArgumentError("Cannot dump page table");
    }
    md->resize(arg.entry);
    page_tables->resize(arg.entry * kPageSize);
    return absl::OkStatus();
}

absl::Status G6Device::DumpAddressSpace(
    std::vector<kfd_ioctl_dump_addr_space_map_info> *mappings,
    std::vector<uintptr_t> *dma_addr) {
    enum { kMaxPageTableNum = 16, kMaxDMAAddresses = 4096 };

    mappings->resize(kMaxPageTableNum);
    dma_addr->resize(kMaxDMAAddresses);
    int kfd_fd = Platform::Instance().GetKFDFD();
    struct kfd_ioctl_dump_addr_space_args arg = {
        .gpu_id = GetGPUID(),
        .mapping_entries = kMaxPageTableNum,
        .dma_addr_entries = kMaxDMAAddresses,
        .mapping_info_ptr = (unsigned long)mappings->data(),
        .dma_addr_ptr = (unsigned long)dma_addr->data(),
    };

    int r = kmtIoctl(kfd_fd, AMDKFD_IOC_DUMP_ADDR_SPACE, &arg);
    if (r) {
        return absl::InvalidArgumentError("Cannot dump address space");
    }
    mappings->resize(arg.mapping_entries);
    dma_addr->resize(arg.dma_addr_entries);

    return absl::OkStatus();
}

absl::Status G6Device::FlushPageTable() {
    int kfd_fd = Platform::Instance().GetKFDFD();
    auto alloc = ptm_.GetAllocator();
    const auto &pages = ptm_.GetAllocator()->GetPages();

    for (int i = alloc->GetNextFreePageIndex(); i--;) {
        struct kfd_ioctl_set_pgt_args arg = {
            .gpu_id = GetGPUID(),
            .pgt_id = (unsigned)i,
            .src = (unsigned long)pages[i].cpu_base,
        };
        auto r = kmtIoctl(kfd_fd, AMDKFD_IOC_SET_PGT, &arg);
        if (r) {
            return absl::InvalidArgumentError("Cannot set page table");
        }
    }

    struct kfd_ioctl_set_pdb_args arg = {
        .gpu_id = GetGPUID(),
        .pdb_phys_addr = (unsigned long)pages[0].gpu_base | 0x3,
    };

    auto r = kmtIoctl(kfd_fd, AMDKFD_IOC_SET_PDB, &arg);
    if (r) {
        return absl::InvalidArgumentError("Cannot set page table");
    }
    return absl::OkStatus();
}

absl::Status G6Device::InitializeSignalBO(std::unique_ptr<Memory> *signal_bo) {
    enum {
        kSignalBOSize = KFD_SIGNAL_EVENT_LIMIT * sizeof(uint64_t),
        kSignalBOPage = kSignalBOSize / kPageSize,
    };

    int kfd_fd = Platform::Instance().GetKFDFD();
    auto mm = static_cast<G6MemoryManager *>(mm_.get());
    auto offset = mm->GetGTTBumpAllocOffset();
    *signal_bo = mm->NewEventPage();
    if (!signal_bo->get()) {
        return absl::ResourceExhaustedError("Cannot allow signal BO");
    }

    kfd_ioctl_set_event_page_args arg = {
        .gpu_id = GetGPUID(),
        .gtt_region_offset = offset,
    };
    auto r = kmtIoctl(kfd_fd, AMDKFD_IOC_SET_EVENT_PAGE, &arg);
    if (r) {
        return absl::InvalidArgumentError("Cannot set signal BO");
    }
    return absl::OkStatus();
}

void *G6Device::GetOrInitializeDoorbell(uint64_t doorbell_mmap_offset) {
    return reinterpret_cast<void *>(doorbell_va_);
}

} // namespace ocl::hsa
