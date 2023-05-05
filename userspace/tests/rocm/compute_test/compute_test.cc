#include <absl/types/span.h>
#include <algorithm>
#include <gtest/gtest.h>
#include <hsa/hsa.h>

#include "compute_test_base.h"

#include <cstring>
#include <fstream>

using namespace ocl::hsa;
using absl::Status;

class ROCmComputeTest : public ::testing::Test, protected ComputeTestBase {
  protected:
    virtual void SetUp() override;
    virtual void TearDown() override;
};

void ROCmComputeTest::SetUp() {
    auto &plat = Platform::Instance();
    auto stat = plat.Initialize();
    ASSERT_TRUE(stat.ok());
    ASSERT_GT(plat.GetDevices().size(), 0);
    dev_ = plat.GetDevices()[0];

    mm_ = dev_->GetMemoryManager();
    signals_ = std::make_unique<SignalPool>(mm_);
    queue_ = AQLQueue::Create(dev_, &stat);
    ASSERT_TRUE(stat.ok());
    stat = queue_->Register();
    ASSERT_TRUE(stat.ok());
}

void ROCmComputeTest::TearDown() {
    auto stat = queue_->Destroy();
    ASSERT_TRUE(stat.ok());
    signals_->Destroy();
    stat = Platform::Instance().Close();
    ASSERT_TRUE(stat.ok());
}

TEST_F(ROCmComputeTest, TestCompute) {
    TestLaunchKernel();
    TestSharedMemory();
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
