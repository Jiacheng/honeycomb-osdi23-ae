#include "opencl/hip/usm/secure_memcpy.h"
#include "utils/align.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

class HipMemcpyTest : public ::testing::Test {
  protected:
    enum { kLargeBlockSize = 128UL * 1024 * 1024 };
    virtual void SetUp() override;
    virtual void TearDown() override;

    void TestMemcpy();

    hipDeviceptr_t mem_;
};

void HipMemcpyTest::SetUp() {
    ASSERT_EQ(hipSuccess, hipMalloc((void **)&mem_, kLargeBlockSize));
}

void HipMemcpyTest::TearDown() { ASSERT_EQ(hipSuccess, hipFree(mem_)); }

TEST_F(HipMemcpyTest, TestSmallUnalignedMemcpy) {
    unsigned src = std::rand();
    unsigned dst = 0;
    ASSERT_EQ(hipSuccess, hipMemcpyHtoD(mem_, &src, sizeof(src)));
    ASSERT_EQ(hipSuccess, hipMemcpyDtoH(&dst, mem_, sizeof(src)));
    ASSERT_EQ(src, dst);
}

TEST_F(HipMemcpyTest, TestUnalignedMemcpy) {
    using ocl::hip::kAESBlockSize;
    enum {
        kTransferSize = 16 * kAESBlockSize,
        kOffset = 1, // to make mem_ unaligned
    };

    auto mem_aligned = reinterpret_cast<hipDeviceptr_t>(
        (gpumpc::AlignUp(reinterpret_cast<uintptr_t>(mem_), kAESBlockSize)));
    auto align_offset = reinterpret_cast<uintptr_t>(mem_aligned) -
                        reinterpret_cast<uintptr_t>(mem_);
    auto mem_unaligned = reinterpret_cast<hipDeviceptr_t>(
        reinterpret_cast<uintptr_t>(mem_aligned) + kOffset);

    ASSERT_LE(align_offset + kOffset + kTransferSize, kLargeBlockSize);

    std::vector<unsigned> src(kTransferSize / sizeof(unsigned));
    std::vector<unsigned> dst(kTransferSize / sizeof(unsigned));
    std::uniform_int_distribution<unsigned> dist(
        std::numeric_limits<unsigned>::min(),
        std::numeric_limits<unsigned>::max());
    static std::default_random_engine gen;
    std::generate(src.begin(), src.end(), [&]() { return dist(gen); });

    // original buffer should be aligned
    ASSERT_EQ(hipSuccess,
              hipMemcpyHtoD(mem_unaligned, src.data(), kTransferSize));
    ASSERT_EQ(hipSuccess,
              hipMemcpyDtoH(dst.data(), mem_unaligned, kTransferSize));
    ASSERT_EQ(src, dst);
}

TEST_F(HipMemcpyTest, TestLargeMemcpy) {
    std::vector<unsigned> src(kLargeBlockSize / sizeof(unsigned));
    std::vector<unsigned> dst(kLargeBlockSize / sizeof(unsigned));
    std::uniform_int_distribution<unsigned> dist(
        std::numeric_limits<unsigned>::min(),
        std::numeric_limits<unsigned>::max());
    static std::default_random_engine gen;
    std::generate(src.begin(), src.end(), [&]() { return dist(gen); });

    ASSERT_EQ(hipSuccess,
              hipMemcpyHtoD(mem_, src.data(), src.size() * sizeof(unsigned)));
    ASSERT_EQ(hipSuccess,
              hipMemcpyDtoH(dst.data(), mem_, src.size() * sizeof(unsigned)));
    ASSERT_EQ(src, dst);
}
