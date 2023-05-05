#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gpumpc::experiment::ethash {

class DagCacheManagerBase {
  public:
    enum {
        kMaxEpoch = 2048,
    };
    virtual ~DagCacheManagerBase() = default;
    std::string ComputeSeed(int epoch) const;
    int ComputeEpoch(const std::string &seed) const;

    std::shared_ptr<std::vector<char>> GetCurrentCache() const;
    std::shared_ptr<std::vector<char>> GetCache(int seed_epoch);
    const std::map<std::string, int> &GetSeedToEpochMap() const;
    virtual size_t GetCacheSizeByEpoch(unsigned epoch) = 0;
    virtual size_t GetDAGSizeByEpoch(unsigned epoch) = 0;

  protected:
    enum : unsigned {
        kSha3256DigestLength = 256 / 8,
        kCacheRound = 3,
        kHashBytes = 64,
    };
    DagCacheManagerBase();

    void RecomputeCacheIfNecessary(int seed);
    void MakeCache();

    int seed_;
    std::shared_ptr<std::vector<char>> cache_;
    std::map<std::string, int> seed_epoch_map_;
    std::map<int, std::string> epoch_seed_map_;
    std::mutex mutex_;
};

} // namespace bminer