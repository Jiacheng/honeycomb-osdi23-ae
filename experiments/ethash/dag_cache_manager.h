#pragma once

#include "dag_cache_manager_base.h"

namespace gpumpc::experiment::ethash {

// Compute the DAG for Ethereum
// See https://github.com/ethereum/wiki/wiki/Ethash for more details.
class DagCacheManager : public DagCacheManagerBase {
  public:
    static const size_t kEthSeedLength = 32;
    static DagCacheManager &Instance();
    virtual size_t GetCacheSizeByEpoch(unsigned epoch) override;
    virtual size_t GetDAGSizeByEpoch(unsigned epoch) override;

  protected:
    DagCacheManager() = default;
    DagCacheManager(const DagCacheManager &) = delete;
    DagCacheManager &operator=(const DagCacheManager &) = delete;
};

} // namespace bminer
