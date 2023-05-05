#include "opencl/hsa/memory_manager.h"
#include "opencl/hsa/queue.h"
#include "opencl/hsa/sdma_ops.h"
#include "opencl/hsa/signals.h"
#include "opencl/hsa/types.h"
#include "sdma_test_base.h"

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

using namespace ocl::hsa;

class SDMATest : public ::testing::Test, protected SDMATestBase {
  protected:
    virtual void SetUp() override;
    virtual void TearDown() override;

    Device *dev_;

    MemoryManager *mm_;
    std::unique_ptr<Memory> scratch_;
    std::unique_ptr<Memory> scratch_vram_;
    std::unique_ptr<SDMAQueue> queue_;
};

void SDMATest::SetUp() {
    auto &plat = Platform::Instance();
    auto stat = plat.Initialize();
    ASSERT_TRUE(stat.ok());
    ASSERT_GT(plat.GetDevices().size(), 0);
    dev_ = plat.GetDevices()[0];

    // Allocate queue buffer.
    queue_ = SDMAQueue::Create(dev_, &stat);
    ASSERT_TRUE(stat.ok());
    stat = queue_->Register();
    ASSERT_TRUE(stat.ok());
    ops_queue_ = std::make_unique<SDMAOpsQueue>(queue_.get());
    mm_ = dev_->GetMemoryManager();
    signals_ = std::make_unique<SignalPool>(mm_);
    scratch_ = mm_->NewGTTMemory(2 << 20, false);
    scratch_vram_ = mm_->NewDeviceMemory(2 << 20);
    signal_ = signals_->GetSignal();
}

void SDMATest::TearDown() {
    signals_->PutSignal(signal_);
    auto stat = scratch_->Destroy();
    ASSERT_TRUE(stat.ok());
    stat = scratch_vram_->Destroy();
    ASSERT_TRUE(stat.ok());

    ops_queue_.reset();
    stat = queue_->Destroy();
    ASSERT_TRUE(stat.ok());
    queue_.reset();
    signals_.reset();
    stat = Platform::Instance().Close();
    ASSERT_TRUE(stat.ok());
}

TEST_F(SDMATest, TestSDMAOperations) {
    TestGetTimestamp(reinterpret_cast<uint64_t *>(scratch_->GetBuffer()));
    TestMemcpy(reinterpret_cast<uint64_t *>(scratch_->GetBuffer()),
               scratch_->GetGPUAddress(), scratch_vram_->GetGPUAddress());
}