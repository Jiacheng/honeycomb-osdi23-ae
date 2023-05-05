#include "experiments/vram-rpc/cpu_dispatch.h"
#include "opencl/hip/usm/secure_memcpy.h"
#include "utils/hip_helper.h"
#include "utils/monad_runner.h"
#include <absl/status/status.h>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <gflags/gflags.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <openssl/evp.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

static const char *kModule = "/data/rpc_pingpong.bin";

using namespace gpumpc::experiment;
using namespace ocl::hip;

class PingPong : public CPURPC {
  public:
    PingPong();

  protected:
    virtual absl::Status LaunchServer();
    virtual absl::Status LaunchClient();
};

PingPong::PingPong() {
    // dummy payload
    payload_size_ = kAESBlockSize;
}

absl::Status PingPong::LaunchServer() {
    while (true) {
#pragma pack(push, 1)
        struct {
            unsigned long *base;
            unsigned long timestamp;
        } param = {
            .base = reinterpret_cast<unsigned long *>(base_),
            .timestamp = 0,
        };
#pragma pack(pop)
        gpumpc::MonadRunner<absl::Status> runner(absl::OkStatus());
        runner
            .Run([&]() {
                Wait(client_signal_);
                return absl::OkStatus();
            })
            .Run([&]() { return DecryptFromPayload(); })
            .Run([&]() {
                param.timestamp = timestamp_;
                return LaunchKernel(server_, 1, 1, param);
            })
            .Run([&]() { return GetTimestamp(); })
            .Run([&]() { return EncryptToPayload(); })
            .Run([&]() {
                Signal(server_signal_);
                return absl::OkStatus();
            });

        if (!runner.code().ok()) {
            return runner.code();
        }
    }
    return absl::OkStatus();
}

absl::Status PingPong::LaunchClient() {
    while (true) {
#pragma pack(push, 1)
        struct {
            unsigned long *base;
        } param = {
            .base = reinterpret_cast<unsigned long *>(base_),
        };
        struct {
            unsigned long *base;
            unsigned long timestamp;
        } param_collect = {
            .base = reinterpret_cast<unsigned long *>(base_),
            .timestamp = 0,
        };
#pragma pack(pop)
        gpumpc::MonadRunner<absl::Status> runner(absl::OkStatus());
        runner.Run([&]() { return LaunchKernel(client_, 1, 1, param); })
            .Run([&]() { return GetTimestamp(); })
            .Run([&]() { return EncryptToPayload(); })
            .Run([&]() {
                Signal(client_signal_);
                Wait(server_signal_);
                return absl::OkStatus();
            })
            .Run([&]() { return DecryptFromPayload(); })
            .Run([&]() {
                param_collect.timestamp = timestamp_;
                return LaunchKernel(client_collect_, 1, 1, param_collect);
            })
            .Run([&]() { return GetTimestamp(); });

        if (!runner.code().ok()) {
            return runner.code();
        }
        auto cycles = (double)timestamp_;
        auto ns = cycles * ns_per_tsc_;
        printf("latency: %.04f cycles / %.04f ns\n", cycles, ns);
        sleep(1);
    }
    return absl::OkStatus();
}

DEFINE_string(mode, "server", "Act as server or client");
DEFINE_string(shm, "", "The name of the file for the IPC shared memory");

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    auto resource_dir = getenv("GPUMPC_RESOURCE_DIR");
    if (!resource_dir) {
        std::cerr << "GPUMPC_RESOURCE_DIR has not been set\n";
        return -1;
    }

    PingPong::Options options = {
        .module_path = std::string(resource_dir) + kModule,
        .ipc_mem_file = FLAGS_shm,
        .mode = FLAGS_mode == "server" ? PingPong::Mode::kServer
                                       : PingPong::Mode::kClient,
    };

    if (options.ipc_mem_file == "") {
        std::cerr << "Name of ipc mem file is not provided" << std::endl;
        return -1;
    }

    PingPong test;
    gpumpc::MonadRunner<absl::Status> runner(absl::OkStatus());
    runner.Run([&]() { return test.Initialize(options); })
        .Run([&]() { return test.CalibrateTime(); })
        .Run([&]() { return test.Launch(); })
        .Run([&]() { return test.Destroy(); });
    auto stat = runner.code();
    if (!stat.ok()) {
        std::cerr << "Failed to run the benchmark: " << stat.ToString() << "\n";
        return -1;
    }

    return 0;
}
