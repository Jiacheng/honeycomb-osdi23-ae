#include "experiments/vram-rpc/gpu_direct_dispatch.h"
#include "rpc/ipc_shm.h"
#include "rpc/ring_queue.h"
#include "utils/hip_helper.h"
#include "utils/monad_runner.h"
#include <absl/status/status.h>
#include <chrono>
#include <fstream>
#include <gflags/gflags.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <unistd.h>
#include <vector>

static const char *kModule = "/data/rpc_pingpong.bin";

using namespace gpumpc::rpc;
using namespace gpumpc::experiment;

class PingPong : public GPUDirectRPC {
  public:
    virtual absl::Status LaunchServer() override;
    virtual absl::Status LaunchClient() override;
};

absl::Status PingPong::LaunchServer() {
#pragma pack(push, 1)
    struct {
        LockFreeQueueView q;
        unsigned *base;
    } param = {
        .q = queue_,
        .base = reinterpret_cast<unsigned *>(mem_->GetBuffer()),
    };
#pragma pack(pop)
    // It is a busy polling loop
    auto stat = LaunchKernel(server_, 1, 1, param);
    if (!stat.ok()) {
        return stat;
    }
    auto err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        return absl::InvalidArgumentError("Cannot sync");
    }
    return absl::OkStatus();
}

absl::Status PingPong::LaunchClient() {
    enum {
        kGridSize = 1,
    };
    std::vector<unsigned long> values(kGridSize * options_.block_size);
    while (true) {
#pragma pack(push, 1)
        struct {
            LockFreeQueueView q;
            unsigned long *base;
        } param = {
            .q = queue_,
            .base = reinterpret_cast<unsigned long *>(mem_->GetBuffer()),
        };
#pragma pack(pop)

        auto stat = LaunchKernel(client_, 1, options_.block_size, param);
        if (!stat.ok()) {
            return stat;
        }

        gpumpc::MonadRunner<hipError_t> runner(hipSuccess);
        runner
            .Run([&]() {
                return hipMemcpyDtoH(values.data(), param.base,
                                     sizeof(unsigned long) * values.size());
            })
            .Run([&]() { return hipDeviceSynchronize(); });
        if (runner.code() != hipSuccess) {
            return absl::InvalidArgumentError("Cannot sync");
        }
        unsigned long v = std::accumulate(
            values.begin(), values.end(), 0,
            [](unsigned long a, unsigned long b) { return a + b; });
        auto avg_cycles = (double)v / values.size();
        auto avg_ns = avg_cycles * ns_per_tsc_;
        printf("Average latency: %.04f cycles / %.04f ns\n", avg_cycles,
               avg_ns);
        sleep(1);
    }
    return absl::OkStatus();
}

DEFINE_string(mode, "server", "Act as server or client");
DEFINE_string(handle, "", "The name of the file that store the IPC handle");
DEFINE_uint32(blocksize, 1, "Block size when enqueuing IPC request");
DEFINE_string(variant, "raw",
              "Variant to be benchmarked (raw / cap). Raw RPC performance or "
              "wrap things with capability");

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    auto resource_dir = getenv("GPUMPC_RESOURCE_DIR");
    if (!resource_dir) {
        std::cerr << "GPUMPC_RESOURCE_DIR has not been set\n";
        return -1;
    }

    PingPong::Options options = {
        .variant = FLAGS_variant == "raw"
                       ? PingPong::Variant::kRawRPCPerformance
                       : PingPong::Variant::kRPCWithCapability,
        .module_path = std::string(resource_dir) + kModule,
        .ipc_mem_handle_file = FLAGS_handle,
        .mode = FLAGS_mode == "server" ? PingPong::Mode::kServer
                                       : PingPong::Mode::kClient,
        .block_size = FLAGS_blocksize,
    };

    if (options.block_size != 1) {
        std::cerr
            << "The current implementation only works when block size = 1\n";
        return -1;
    }

    PingPong test;
    gpumpc::MonadRunner<absl::Status> runner(absl::OkStatus());
    runner.Run([&]() { return test.Initialize(options); })
        .Run([&]() { return test.CalibrateTime(); })
        .Run([&]() { return test.Launch(); });
    auto stat = runner.code();
    if (!stat.ok()) {
        std::cerr << "Failed to run the benchmark: " << stat.ToString() << "\n";
        return -1;
    }

    return 0;
}
