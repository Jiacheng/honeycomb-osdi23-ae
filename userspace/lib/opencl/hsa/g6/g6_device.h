#pragma once

#include "g6_ioctl.h"
#include "opencl/hsa/kfd/kfd_device.h"
#include "opencl/hsa/memory_manager.h"
#include "opencl/hsa/platform.h"
#include "opencl/hsa/types.h"
#include "page_table.h"
#include <absl/types/span.h>

namespace ocl::hsa {

class G6Platform;

class G6Device : public KFDDevice {
  public:
    friend class G6Platform;
    enum {
        kReservedPGTPage = KFD_IOCTL_G6_RESERVED_PGT_REGION_PAGES,
    };
    enum { kReservedVRAMSize = 2 << 20 };
    virtual absl::Status Initialize() override;
    virtual void *
    GetOrInitializeDoorbell(uint64_t doorbell_mmap_offset) override;
    virtual std::unique_ptr<DeviceQueue> CreateSDMAQueue() override;
    virtual std::unique_ptr<DeviceQueue> CreateAQLQueue() override;

    gpu_addr_t GetUserVRAMPhysAddr() const { return user_vram_phys_addr_; }
    size_t GetUserVRAMSize() const { return kReservedVRAMSize; }
    gpu_addr_t GetDoorbellPhysAddr() const { return doorbell_phys_addr_; }
    absl::Span<const gpu_addr_t> GetPageTableRegionPhysAddr() const;
    size_t GetDoorbellSliceSize() const { return doorbell_slice_size_; }
    std::vector<phys_addr_t> *GetGTTRegion() { return &gtt_regions_; }
    void *GetGTTVABase() const { return gtt_va_; }
    phys_addr_t GetGTTPhysAddr() const { return gtt_phys_addr_; }

    g6::PageTableManager *GetPageTableManager() { return &ptm_; }
    //
    // Update the GPU page table for the process
    absl::Status FlushPageTable();

    //
    // For debugging purposes
    //
    absl::Status
    DumpPageTableBO(std::vector<kfd_ioctl_dump_page_table_metadata> *md,
                    std::vector<char> *page_tables);
    absl::Status
    DumpAddressSpace(std::vector<kfd_ioctl_dump_addr_space_map_info> *mappings,
                     std::vector<uintptr_t> *dma_addr);

  protected:
    explicit G6Device(unsigned node_id, unsigned gpu_id);
    absl::Status AcquireVMG6();
    absl::Status InitializeDoorbell();
    absl::Status InitializeSignalBO(std::unique_ptr<Memory> *signal_bo);
    absl::Status MapGTTMemoryToHost();

    g6::PageTableManager ptm_;
    std::vector<char> pgt_buf_;
    std::vector<phys_addr_t> gtt_regions_;
    gpu_addr_t user_vram_phys_addr_;
    gpu_addr_t pgt_region_gpu_addr_[kReservedPGTPage];
    gpu_addr_t doorbell_phys_addr_;
    size_t doorbell_slice_size_;
    gpu_addr_t gtt_phys_addr_;
    gpu_addr_t doorbell_va_;
    size_t gtt_region_size_;
    void *gtt_va_;
};

} // namespace ocl::hsa