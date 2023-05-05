#include "ethash_solver.h"
#include "dag_cache_manager.h"
#include "experiments/platform.h"
#include "utils/hip_helper.h"
#include "utils/monad_runner.h"

#include <absl/status/status.h>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <hip/hip_runtime.h>

namespace gpumpc::experiment::ethash {
static inline int32_t CountLeadingZero(uint32_t val) {
    return __builtin_clz(val);
}

// The "round-up" approach described in
// https://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html
// Using the generic 33-bit fix-up approach.
static std::pair<unsigned, unsigned>
DivisionToMultiplicationRoundUp(uint32_t d) {
    uint32_t floor_log_2_d = 31 - CountLeadingZero(d);

    uint8_t more;
    unsigned proposed_m = (1ull << (32 + floor_log_2_d)) / d;
    unsigned rem = (1ull << (32 + floor_log_2_d)) % d;

    proposed_m += proposed_m;
    const uint32_t twice_rem = rem + rem;
    if (twice_rem >= d || twice_rem < rem) {
        proposed_m += 1;
    }
    more = floor_log_2_d;
    return std::make_pair(1 + proposed_m, more);
}

EthashSolver::EthashSolver()
    : dag_cache_manager_(&DagCacheManager::Instance()), db_dag_(nullptr),
      db_light_(nullptr), db_output_(nullptr), module_(nullptr),
      f_dag_(nullptr), f_search_(nullptr), sft_(0), mult_(0) {
    std::fill_n(h_header_, 4, 0);
}

EthashSolver::~EthashSolver() { Close(); }

#define checkCuErrors checkHipErrors

void EthashSolver::InitDAG(int seed) {
    auto dag_size = dag_cache_manager_->GetDAGSizeByEpoch(seed);
    const auto light_data = dag_cache_manager_->GetCache(seed);
    ReallocateDAG(dag_size, *light_data);
    GenerateDAG(dag_size, *light_data);
}

absl::Status EthashSolver::Initialize() {
    auto &plat = ExperimentPlatform::Instance();
    std::vector<char> data;
    auto stat = plat.LoadResource("ethash/ethash_gfx1030.bin", &data);
    if (!stat.ok()) {
        return stat;
    }
    MonadRunner<hipError_t> runner(hipSuccess);
    runner.Run([&]() { return hipModuleLoadData(&module_, data.data()); })
        .Run([&]() {
            return hipModuleGetFunction(&f_dag_, module_,
                                        "InitializeEthashDAG");
        })
        .Run([&]() {
            return hipModuleGetFunction(&f_search_, module_,
                                        "EthashSingleSearch");
        })
        .Run([&]() {
            return hipMalloc(&db_output_, sizeof(EthashSolutionsOnDevice));
        });
    return HipErrToStatus(runner.code());
}

void EthashSolver::Close() {
    if (!module_) {
        return;
    }
    checkCuErrors(hipDeviceSynchronize());
    if (db_dag_) {
        auto _ = hipFree(db_dag_);
        (void)_;
        db_dag_ = nullptr;
    }
    if (db_light_) {
        auto _ = hipFree(db_light_);
        (void)_;
        db_light_ = nullptr;
    }
    checkCuErrors(hipFree(db_output_));
    db_output_ = nullptr;

    auto _ = hipModuleUnload(module_);
    (void)_;
    module_ = nullptr;
}

void EthashSolver::ReallocateDAG(size_t dag_size,
                                 const std::vector<char> &light_data) {
    if (db_dag_) {
        checkHipErrors(hipFree(db_dag_));
        db_dag_ = nullptr;
    }

    if (db_light_) {
        checkCuErrors(hipFree(db_light_));
        db_light_ = nullptr;
    }

    // size_t dag_size_mix = dag_size / kEthashMixBytes;
    // Reallocate buffer and copy light data into the buffer
    checkCuErrors(hipMalloc(&db_dag_, dag_size));
    checkCuErrors(hipMalloc(&db_light_, light_data.size()));
    checkCuErrors(hipMemcpyHtoD(
        db_light_, const_cast<char *>(light_data.data()), light_data.size()));
    dag_items_ = (unsigned)(dag_size / kEthashMixBytes);
    std::tie(mult_, sft_) = DivisionToMultiplicationRoundUp(dag_items_);
    checkCuErrors(hipDeviceSynchronize());
}

void EthashSolver::GenerateDAG(size_t dag_size,
                               const std::vector<char> &light_data) {
    uint32_t const work = dag_size / kEthashHashBytes;
    static const uint32_t kRun = kEthashBlockSizeDag * kEthashGridSizeDag;
#pragma pack(push, 1)
    struct {
        hipDeviceptr_t d_dag;
        hipDeviceptr_t d_light;
        unsigned base;
        unsigned dag_entries;
        unsigned light_entries;
    } p = {
        .d_dag = db_dag_,
        .d_light = db_light_,
        .base = 0,
        .dag_entries = (unsigned)(dag_size / kEthashMixBytes),
        .light_entries = (unsigned)(light_data.size() / kEthashHashBytes),
    };
#pragma pack(pop)
    for (unsigned base = 0; base < work; base += kRun) {
        p.base = base;
        auto grid =
            (work - base + kEthashBlockSizeDag - 1) / kEthashBlockSizeDag;
        auto grid_size = std::min<unsigned>(kEthashGridSizeDag, grid);
        auto stat = LaunchKernel(f_dag_, grid_size, kEthashBlockSizeDag, p);
        assert(stat.ok());
    }
    checkCuErrors(hipDeviceSynchronize());
}

void EthashSolver::SetJob(const uint64_t header[4]) {
    std::copy(header, header + 4, h_header_);
}

absl::Status EthashSolver::Solve(uint64_t nonce, uint64_t target) {
    static const unsigned kZero = 0;

#pragma pack(push, 1)
    struct SearchKernelParam {
        uint64_t header[4];
        hipDeviceptr_t d_dag;
        hipDeviceptr_t d_output;
        uint64_t start_nonce;
        uint64_t target;
        unsigned dag_items;
        unsigned sft;
        unsigned mult;
    } p = {
        .d_dag = db_dag_,
        .d_output = db_output_,
        .start_nonce = nonce,
        .target = target,
        .dag_items = dag_items_,
        .sft = sft_,
        .mult = mult_,
    };
#pragma pack(pop)
    std::copy(h_header_, h_header_ + 4, p.header);

    auto stat = HipErrToStatus(hipMemcpyHtoD(
        db_output_, const_cast<unsigned *>(&kZero), sizeof(kZero)));
    if (!stat.ok()) {
        return stat;
    }
    stat = LaunchKernel(f_search_, kEthashGridSize, kEthashBlockSize, p);
    return stat;
}

unsigned EthashSolver::Fetch(absl::Span<unsigned> *ethash,
                             size_t *scanned_nonce) {
    EthashSolutionsOnDevice sol = {
        0,
    };
    checkCuErrors(hipMemcpyDtoH(&sol, db_output_, sizeof(sol)));
    if (!sol.count) {
        return 0;
    }
    auto length = std::min<unsigned>(sol.count, ethash->length());
    std::copy(sol.gid, sol.gid + length, ethash->data());
    *scanned_nonce = kEthashGridSize * kEthashBlockSize;
    return sol.count;
}

} // namespace gpumpc::experiment::ethash