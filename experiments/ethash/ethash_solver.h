#pragma once

#include "dag_cache_manager_base.h"
#include "dualsolver_types.h"
#include <absl/status/status.h>
#include <absl/types/span.h>
#include <hip/hip_runtime_api.h>

namespace gpumpc::experiment::ethash {

class DagCacheManagerBase;
class EthashSolver {
  public:
    enum : unsigned {
        kEthashMixBytes = 128,
        kEthashHashBytes = 64,
        kEthashHeaderBytes = 32,
    };

    explicit EthashSolver();
    ~EthashSolver();
    absl::Status Initialize();
    void Close();

    void InitDAG(int seed_epoch);
    void SetJob(const uint64_t header[4]);
    absl::Status Solve(uint64_t nonce, uint64_t target);
    unsigned Fetch(absl::Span<unsigned> *ethash, size_t *scanned_nonce);

  protected:
    void ReallocateDAG(size_t dag_size, const std::vector<char> &light_data);
    void GenerateDAG(size_t dag_size, const std::vector<char> &light_data);

    DagCacheManagerBase *dag_cache_manager_;
    hipDeviceptr_t db_dag_;
    hipDeviceptr_t db_light_;
    hipDeviceptr_t db_output_;

    hipModule_t module_;
    hipFunction_t f_dag_;
    hipFunction_t f_search_;

    uint64_t h_header_[4];
    unsigned dag_items_;
    unsigned sft_;
    unsigned mult_;
};

} // namespace gpumpc::experiment::ethash
