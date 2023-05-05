#include <gtest/gtest.h>

#include "opencl/hsa/ring_allocator.h"

using namespace ocl::hsa;

class RingAllocatorTest : public ::testing::Test {
  protected:
    virtual void SetUp() override;
    virtual void TearDown() override;

    const uintptr_t kBase = 0x80000000UL;
    const size_t kLength = 1 << 10;
    const size_t kAlign = 4;
    const size_t kRounds = 1024;
};

void RingAllocatorTest::SetUp() {}

void RingAllocatorTest::TearDown() {}

TEST_F(RingAllocatorTest, TestFullAlloc) {
    RingAllocator alloc;
    auto length = alloc.Initialize(kBase, kLength, kAlign);
    for (size_t i = 0; i != kRounds; ++i) {
        ASSERT_TRUE(alloc.AllocateAlign(nullptr, length, kAlign).ok());
        ASSERT_TRUE(alloc.Free().ok());
    }
}

TEST_F(RingAllocatorTest, TestHalfAlloc) {
    RingAllocator alloc;
    auto length = alloc.Initialize(kBase, kLength, kAlign);
    for (size_t i = 0; i != kRounds; ++i) {
        ASSERT_TRUE(alloc.AllocateAlign(nullptr, length / 2, kAlign).ok());
        ASSERT_TRUE(alloc.AllocateAlign(nullptr, length / 2, kAlign).ok());
        ASSERT_TRUE(alloc.Free().ok());
        ASSERT_TRUE(alloc.Free().ok());
    }
}

TEST_F(RingAllocatorTest, TestHalfAlloc2) {
    RingAllocator alloc;
    auto length = alloc.Initialize(kBase, kLength, kAlign);
    ASSERT_TRUE(alloc.AllocateAlign(nullptr, length / 2, kAlign).ok());
    for (size_t i = 0; i != kRounds; ++i) {
        ASSERT_TRUE(alloc.AllocateAlign(nullptr, length / 2, kAlign).ok());
        ASSERT_TRUE(alloc.Free().ok());
        ASSERT_TRUE(alloc.AllocateAlign(nullptr, length / 2, kAlign).ok());
        ASSERT_TRUE(alloc.Free().ok());
    }
}

TEST_F(RingAllocatorTest, TestQuarterAlloc) {
    RingAllocator alloc;
    auto length = alloc.Initialize(kBase, kLength, kAlign);
    for (size_t i = 0; i != kRounds; ++i) {
        ASSERT_TRUE(alloc.AllocateAlign(nullptr, length / 4, kAlign).ok());
        ASSERT_TRUE(alloc.AllocateAlign(nullptr, length / 2, kAlign).ok());
        ASSERT_TRUE(alloc.AllocateAlign(nullptr, length / 4, kAlign).ok());
        ASSERT_TRUE(alloc.Free().ok());
        ASSERT_TRUE(alloc.Free().ok());
        ASSERT_TRUE(alloc.Free().ok());
    }
}

TEST_F(RingAllocatorTest, TestQuarterAllocIncremental) {
    RingAllocator alloc;
    auto length = alloc.Initialize(kBase, kLength, kAlign);
    ASSERT_TRUE(alloc.AllocateAlign(nullptr, length / 4, kAlign).ok());
    ASSERT_TRUE(alloc.AllocateAlign(nullptr, length / 2, kAlign).ok());
    ASSERT_TRUE(alloc.AllocateAlign(nullptr, length / 4, kAlign).ok());
    ASSERT_TRUE(alloc.Free().ok());
    ASSERT_TRUE(alloc.Free().ok());
    ASSERT_TRUE(alloc.Free().ok());
    // advance internal ptr by a kAlign
    ASSERT_TRUE(alloc.AllocateAlign(nullptr, 1, kAlign).ok());
    ASSERT_TRUE(alloc.Free().ok());
    // the first two alloc should be fine, but the last one can not
    // fit in the remaining slot
    ASSERT_TRUE(alloc.AllocateAlign(nullptr, length / 4, kAlign).ok());
    ASSERT_TRUE(alloc.AllocateAlign(nullptr, length / 2, kAlign).ok());
    ASSERT_FALSE(alloc.AllocateAlign(nullptr, length / 4, kAlign).ok());
    // should be able to alloc length/4 + kAlign at base
    ASSERT_TRUE(alloc.Free().ok());
    // at this moment there are length/4 + kAlign spare space in the begining
    // of the ring buffer and length/4 - kAlign spare space in the tail of the
    // ring buffer so we can allocate at the head
    uintptr_t target;
    ASSERT_TRUE(alloc.AllocateAlign(&target, length / 4 + kAlign, kAlign).ok());
    ASSERT_EQ(target, kBase);
}

TEST_F(RingAllocatorTest, TestFullAllocIncremental) {
    RingAllocator alloc;
    auto length = alloc.Initialize(kBase, kLength, kAlign);
    for (size_t i = 0; i != length / kAlign; ++i) {
        ASSERT_TRUE(alloc.AllocateAlign(nullptr, length, kAlign).ok());
        ASSERT_FALSE(alloc.AvailableAlign(1, kAlign));
        ASSERT_TRUE(alloc.Free().ok());
        // advance internal ptr by a kAlign
        ASSERT_TRUE(alloc.AllocateAlign(nullptr, 1, kAlign).ok());
        ASSERT_TRUE(alloc.Free().ok());
        // should still be able to allocate in the next round
    }
}

TEST_F(RingAllocatorTest, TestSmallAlloc) {
    RingAllocator alloc;
    auto length = alloc.Initialize(kBase, kLength, kAlign);
    for (size_t i = 0; i != length / kAlign; ++i) {
        ASSERT_LE(1, kAlign);
        ASSERT_TRUE(alloc.AllocateAlign(nullptr, 1, kAlign).ok());
    }
    // Should be not avaiable
    ASSERT_FALSE(alloc.AvailableAlign(1, kAlign));
    // Should be avaiable
    ASSERT_TRUE(alloc.Free().ok());
    ASSERT_TRUE(alloc.AvailableAlign(1, kAlign));
}

TEST_F(RingAllocatorTest, TestUnalignedBase) {
    RingAllocator alloc;
    auto length = alloc.Initialize(kBase - 1, kLength + 2, kAlign);
    // actually usable length: kLength
    ASSERT_EQ(length, kLength);
    for (size_t i = 0; i != length / kAlign; ++i) {
        ASSERT_LE(1, kAlign);
        ASSERT_TRUE(alloc.AllocateAlign(nullptr, 1, kAlign).ok());
    }
    // Should be not avaiable
    ASSERT_FALSE(alloc.AvailableAlign(1, kAlign));
    // Should be avaiable
    ASSERT_TRUE(alloc.Free().ok());
    ASSERT_TRUE(alloc.AvailableAlign(1, kAlign));
}

TEST_F(RingAllocatorTest, TestAllocAlign) {
    // user has a stricter alignment requirement on the allocated address
    const size_t kUserAlign = 8 * kAlign;

    RingAllocator alloc;
    auto length = alloc.Initialize(kBase, kLength, kAlign);
    for (size_t i = 0; i != length / kAlign; ++i) {
        uintptr_t target;
        ASSERT_TRUE(alloc.AllocateAlign(&target, 1, kUserAlign).ok());
        ASSERT_EQ(target % kUserAlign, 0);
        ASSERT_TRUE(alloc.Free().ok());
        // advance internal ptr by a kAlign
        ASSERT_TRUE(alloc.AllocateAlign(nullptr, 1, kAlign).ok());
        ASSERT_TRUE(alloc.Free().ok());
        // should still be able to allocate in the next round
    }
}
