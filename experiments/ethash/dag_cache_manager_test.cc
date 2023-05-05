#include "dag_cache_manager.h"

#include <gtest/gtest.h>

using namespace gpumpc::experiment::ethash;

TEST(TestDagCacheManager, TestComputeEpoch) {
    static const unsigned char seed_bytes[] = {
        0x4e, 0x99, 0xa3, 0x0e, 0x99, 0x71, 0x2c, 0x8c, 0x6e, 0x29, 0x2f,
        0xe7, 0xba, 0x6b, 0x27, 0xa3, 0x7c, 0x7c, 0xed, 0x12, 0xe2, 0xec,
        0x78, 0x62, 0xf3, 0x1f, 0xb6, 0x76, 0x72, 0x4c, 0xb4, 0x04};
    std::string seed(reinterpret_cast<const char *>(seed_bytes),
                     sizeof(seed_bytes));
    auto epoch = DagCacheManager::Instance().ComputeEpoch(seed);
    ASSERT_EQ(170, epoch);
}

TEST(TestDagCacheManager, TestComputeSize) {
    ASSERT_EQ(39059264UL, DagCacheManager::Instance().GetCacheSizeByEpoch(170));
    ASSERT_EQ(2499803776UL, DagCacheManager::Instance().GetDAGSizeByEpoch(170));
}
