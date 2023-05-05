#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include "utils/hip_helper.h"
#include "utils/filesystem.h"

static const char *sKernelFile;

class HipLaunchTest : public ::testing::Test {
  protected:
    enum { kScratchSize = 1 << 20 };
    virtual void SetUp() override;
    virtual void TearDown() override;

    hipModule_t module_;
    hipDeviceptr_t scratch_;
};

void HipLaunchTest::SetUp() {
    absl::Status stat;
    auto data = gpumpc::ReadAll(sKernelFile, &stat);
    ASSERT_TRUE(stat.ok());
    ASSERT_EQ(hipSuccess, hipModuleLoadData(&module_, data.data()));
    ASSERT_EQ(hipSuccess, hipMalloc(&scratch_, kScratchSize));
}

void HipLaunchTest::TearDown() {
    ASSERT_EQ(hipSuccess, hipModuleUnload(module_));
    ASSERT_EQ(hipSuccess, hipFree(scratch_));
}

TEST_F(HipLaunchTest, TestLaunchKernel) {
    using gpumpc::experiment::LaunchKernel;

    hipFunction_t func;
    ASSERT_EQ(hipSuccess, hipModuleGetFunction(&func, module_, "fill"));

    struct {
        hipDeviceptr_t base;
        unsigned val;
    } args = {
        .base = scratch_,
        .val = 42,
    };

    ASSERT_TRUE(LaunchKernel(func, 1, 1, args).ok());
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

    unsigned val;
    ASSERT_EQ(hipSuccess, hipMemcpyDtoH(&val, scratch_, sizeof(unsigned)));
    ASSERT_EQ(42, val);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel>\n";
        return -1;
    }
    sKernelFile = argv[1];
    return RUN_ALL_TESTS();
}
