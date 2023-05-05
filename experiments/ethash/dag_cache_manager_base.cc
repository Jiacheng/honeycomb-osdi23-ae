#include "dag_cache_manager_base.h"
#include "crypto/sha3.h"

#include <atomic>
#include <cassert>

namespace gpumpc::experiment::ethash {

DagCacheManagerBase::DagCacheManagerBase() : seed_(-1) {
    char hash[kSha3256DigestLength] = {
        0,
    };
    for (unsigned i = 0; i < kMaxEpoch; ++i) {
        auto key = std::string(hash, sizeof(hash));
        seed_epoch_map_[key] = i;
        epoch_seed_map_[i] = key;
        keccak_256(reinterpret_cast<unsigned char *>(hash),
                   reinterpret_cast<unsigned char *>(hash),
                   kSha3256DigestLength);
    }
}

const std::map<std::string, int> &
DagCacheManagerBase::GetSeedToEpochMap() const {
    return seed_epoch_map_;
}

std::shared_ptr<std::vector<char>>
DagCacheManagerBase::GetCurrentCache() const {
    return std::atomic_load(&cache_);
}

std::shared_ptr<std::vector<char>> DagCacheManagerBase::GetCache(int seed) {
    std::lock_guard<std::mutex> lock(mutex_);
    RecomputeCacheIfNecessary(seed);
    return GetCurrentCache();
}

std::string DagCacheManagerBase::ComputeSeed(int seed_epoch) const {
    auto it = epoch_seed_map_.find(seed_epoch);
    return it == epoch_seed_map_.end() ? "" : it->second;
}

int DagCacheManagerBase::ComputeEpoch(const std::string &seed) const {
    auto it = seed_epoch_map_.find(seed);
    return it == seed_epoch_map_.end() ? 0 : it->second;
}

void DagCacheManagerBase::RecomputeCacheIfNecessary(int seed) {
    if (!cache_ || !cache_->size() || seed != seed_) {
        seed_ = seed;
        MakeCache();
    }
}

void DagCacheManagerBase::MakeCache() {
    const auto cache_size = GetCacheSizeByEpoch(seed_);
    std::shared_ptr<std::vector<char>> ret(new std::vector<char>(cache_size));
    unsigned n = unsigned(cache_size / kHashBytes);
    assert(n > 0 && "DAG is too small");

    typedef union node {
        uint8_t bytes[16 * 4];
        uint32_t words[16];
    } node;

    node *cache_nodes = reinterpret_cast<node *>(&ret->front());
    const auto &seed_str = epoch_seed_map_[seed_];
    keccak_512(cache_nodes[0].bytes,
               reinterpret_cast<const unsigned char *>(seed_str.c_str()),
               seed_str.size());
    for (unsigned i = 1; i < n; ++i) {
        keccak_512(cache_nodes[i].bytes, cache_nodes[i - 1].bytes, kHashBytes);
    }

    for (unsigned j = 0; j < kCacheRound; j++) {
        for (unsigned i = 0; i < n; i++) {
            uint32_t const idx = cache_nodes[i].words[0] % n;
            node data;
            data = cache_nodes[(n - 1 + i) % n];
            for (uint32_t w = 0; w < 16; ++w) {
                data.words[w] ^= cache_nodes[idx].words[w];
            }
            keccak_512(cache_nodes[i].bytes, data.bytes, sizeof(data));
        }
    }
    std::atomic_store(&cache_, ret);
}

} // namespace gpumpc::experiment::ethash