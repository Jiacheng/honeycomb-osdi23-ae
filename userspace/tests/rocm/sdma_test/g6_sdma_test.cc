#include "hsa/amdgpu_vm.h"
#include "opencl/hsa/g6/g6_device.h"
#include "opencl/hsa/g6/page_table.h"
#include "opencl/hsa/memory_manager.h"
#include "opencl/hsa/platform.h"
#include "opencl/hsa/queue.h"
#include "opencl/hsa/sdma_ops.h"
#include "opencl/hsa/signals.h"
#include "opencl/hsa/utils.h"
#include "sdma_test_base.h"

#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <stdexcept>

using namespace ocl::hsa;

class G6UAPITest : public ::testing::Test, protected SDMATestBase {
  protected:
    virtual void SetUp() override;
    virtual void TearDown() override;

    void DumpMemoryInfo();
    void DumpPageTableManager();

    G6Device *dev_;

    MemoryManager *mm_;
    std::unique_ptr<SDMAQueue> queue_;

    std::unique_ptr<Memory> scratch_;
    std::unique_ptr<Memory> scratch_vram_;

    enum { kReservedVRAMSize = 2 << 20 };
};

void G6UAPITest::SetUp() {
    Platform::ChooseVariant(Platform::Variant::kPlatformG6);
    auto &plat = Platform::Instance();
    auto stat = plat.Initialize();
    ASSERT_TRUE(stat.ok());
    ASSERT_GT(plat.GetDevices().size(), 0);
    dev_ = static_cast<G6Device *>(plat.GetDevices()[0]);

    mm_ = dev_->GetMemoryManager();
    signals_ = std::make_unique<SignalPool>(mm_);
    scratch_vram_ = mm_->NewDeviceMemory(kReservedVRAMSize);
    signal_ = signals_->GetSignal();
}

void G6UAPITest::TearDown() {
    signals_->PutSignal(signal_);
    ops_queue_.reset();
    auto stat = queue_->Destroy();
    ASSERT_TRUE(stat.ok());
    queue_.reset();
    signals_.reset();
    stat = Platform::Instance().Close();
    ASSERT_TRUE(stat.ok());
}

void G6UAPITest::DumpMemoryInfo() {
    std::vector<kfd_ioctl_dump_addr_space_map_info> mappings;
    std::vector<uintptr_t> dma_addresses;
    auto stat = dev_->DumpAddressSpace(&mappings, &dma_addresses);
    ASSERT_TRUE(stat.ok());
    std::ofstream ofs_addr_md("addrspace_md.bin", std::ios::binary);
    std::ofstream ofs_dma("addrspace_dma.bin", std::ios::binary);
    ofs_addr_md.write((const char *)mappings.data(),
                      mappings.size() *
                          sizeof(kfd_ioctl_dump_addr_space_map_info));
    ofs_dma.write((const char *)dma_addresses.data(),
                  dma_addresses.size() * sizeof(uint64_t));

    std::vector<kfd_ioctl_dump_page_table_metadata> md;
    std::vector<char> pgt;
    stat = dev_->DumpPageTableBO(&md, &pgt);
    ASSERT_TRUE(stat.ok());

    std::ofstream ofs("metadata.bin", std::ios::binary);
    std::ofstream ofs_pgt("pagetable.bin", std::ios::binary);
    ofs.write((const char *)md.data(),
              md.size() * sizeof(kfd_ioctl_dump_page_table_metadata));
    ofs_pgt.write(pgt.data(), pgt.size());
}

void G6UAPITest::DumpPageTableManager() {
    using namespace ocl::g6;
    PageTableManager *ptm = dev_->GetPageTableManager();
    auto alloc = ptm->GetAllocator();
    const auto &pages = ptm->GetAllocator()->GetPages();
    std::ofstream ofs_md("metadata.bin", std::ios::binary);
    std::ofstream ofs_pgt("pagetable.bin", std::ios::binary);
    for (unsigned i = 0; i < alloc->GetNextFreePageIndex(); i++) {
        ofs_md.write((const char *)&pages[i].gpu_base, sizeof(unsigned long));
        ofs_pgt.write((const char *)pages[i].cpu_base, 4096);
    }
}

TEST_F(G6UAPITest, TestG6SDMA) {
    using namespace ocl::g6;
    absl::Status stat;
    queue_ = SDMAQueue::Create(dev_, &stat);
    ASSERT_TRUE(stat.ok());
    ops_queue_ = std::make_unique<SDMAOpsQueue>(queue_.get());

    scratch_ = mm_->NewGTTMemory(Device::kPageSize, false);

    stat = dev_->FlushPageTable();
    ASSERT_TRUE(stat.ok());
    stat = queue_->Register();
    ASSERT_TRUE(stat.ok());

    auto scratch_addr = reinterpret_cast<uint64_t *>(scratch_->GetBuffer());
    TestGetTimestamp(scratch_addr);
    TestMemcpy(scratch_addr, scratch_->GetGPUAddress(),
               scratch_vram_->GetGPUAddress());
}
