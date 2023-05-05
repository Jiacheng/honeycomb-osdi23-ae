#include "experiments/platform.h"
#include "experiments/vram-rpc/cpu_dispatch.h"
#include "experiments/resnet/resnet_inference.h"
#include "rpc_resnet18_params.h"
#include "rpc/ipc_shm.h"
#include "rpc/ring_queue.h"
#include "utils/align.h"
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

static const char *kModule = "/data/rpc_resnet18.bin";
static const char kImagePath[] = "data/resnet/image/image_1.dat";

using namespace gpumpc::rpc;
using namespace gpumpc::experiment;

class ResNet18Serving : public CPURPC {
  public:
    ResNet18Serving();
    virtual absl::Status Initialize(const Options &options) override;
    virtual absl::Status Destroy();
  protected:
    virtual absl::Status LaunchServer() override;
    virtual absl::Status LaunchClient() override;

    std::unique_ptr<ResNetInference> model_;
    void *client_data_;
    void *client_result_;
};

ResNet18Serving::ResNet18Serving() {
    payload_size_ = gpumpc::AlignUp(std::max(kResNet18InputImageSize, kResNet18ResultSize), ocl::hip::kAESBlockSize);
}

absl::Status ResNet18Serving::Initialize(const Options &options) {
    auto stat = CPURPC::Initialize(options);
    if (!stat.ok()) {
        return stat;
    }
    client_data_ = nullptr;
    client_result_ = nullptr;

    if (options_.mode == Mode::kClient) {
        std::vector<char> image;
        auto &plat = gpumpc::experiment::ExperimentPlatform::Instance();
        auto stat = plat.LoadResource(kImagePath, &image);
        if (!stat.ok()) {
            return stat;
        } else if (image.size() != kResNet18InputImageSize) {
            return absl::InvalidArgumentError("Invalid image");
        }

        gpumpc::MonadRunner<hipError_t> runner(hipSuccess);
        runner
            .Run([&]() {
                return hipMalloc(&client_data_, kResNet18InputImageSize);
            })
            .Run([&]() {
                return hipMalloc(&client_result_, kResNet18ResultSize);
            })
            .Run([&]() {
                return hipMemcpyHtoD(client_data_, image.data(), image.size());
            });
        if (runner.code()) {
            return absl::InvalidArgumentError(
                "Cannot copy the image to the GPU");
        }
        return absl::OkStatus();
    } else {
        model_ = NewResNet18();
        auto stat = model_->Initialize();
        if (!stat.ok()) {
            return stat;
        }
        return absl::OkStatus();
    }
}

absl::Status ResNet18Serving::Destroy() {
    if (client_data_) {
        auto hipError = hipFree(client_data_);
        if (hipError != hipSuccess) {
            return absl::InvalidArgumentError("cannot free mem");
        }
    }
    if (client_result_) {
        auto hipError = hipFree(client_result_);
        if (hipError != hipSuccess) {
            return absl::InvalidArgumentError("cannot free mem");
        }
    }
    return CPURPC::Destroy();
}

absl::Status ResNet18Serving::LaunchServer() {
    while (true) {
        gpumpc::MonadRunner<absl::Status> runner(absl::OkStatus());
        runner
            .Run([&]() {
                Wait(client_signal_);
                return absl::OkStatus();
            })
            .Run([&]() {
                return DecryptFromPayload();
            })
            .Run([&]() {
                auto start = std::chrono::high_resolution_clock::now();
                absl::Span<const char> request(buf_.data(), kResNet18InputImageSize);
                auto stat = model_->Run(request);
                absl::Span<char> result(buf_.data(), kResNet18ResultSize);
                stat = model_->Fetch(result);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::micro> elapsed = end - start;
                double iteration_us = elapsed.count();
                printf("passed %lf us\n", iteration_us);
                return stat;
            })
            .Run([&]() {
                return EncryptToPayload();
            })
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

absl::Status ResNet18Serving::LaunchClient() {
    enum {
        kGridSize = 1,
        kBlockSize = 512,
    };
    while (true) {
#pragma pack(push, 1)
        struct {
            unsigned long *base;
            unsigned long *request;
            unsigned long *image;
        } param = {
            .base = reinterpret_cast<unsigned long *>(base_),
            .request = reinterpret_cast<unsigned long *>(request_),
            .image = reinterpret_cast<unsigned long *>(client_data_),
        };
        struct {
            unsigned long *base;
            unsigned long *request;
            unsigned long *result;
            unsigned long timestamp;
        } param_collect = {
            .base = reinterpret_cast<unsigned long *>(base_),
            .request = reinterpret_cast<unsigned long *>(request_),
            .result = reinterpret_cast<unsigned long *>(client_result_),
            .timestamp = 0,
        };
#pragma pack(pop)
        auto start = std::chrono::high_resolution_clock::now();
        gpumpc::MonadRunner<absl::Status> runner(absl::OkStatus());
        runner
            .Run([&]() {
                return LaunchKernel(client_, 1, kResNet18RPCClientBlockSize, param);
            })
            .Run([&]() {
                return GetTimestamp();
            })
            .Run([&]() {
                return FetchRequest();
            })
            .Run([&]() {
                return EncryptToPayload();
            })
            .Run([&]() {
                Signal(client_signal_);
                Wait(server_signal_);
                return absl::OkStatus();
            })
            .Run([&]() {
                return DecryptFromPayload();
            })
            .Run([&]() {
                return PutResponse();
            })
            .Run([&]() {
                param_collect.timestamp = timestamp_;
                return LaunchKernel(client_collect_, 1, kResNet18RPCClientBlockSize, param_collect);
            })
            .Run([&]() {
                return GetTimestamp();
            });
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;
        double iteration_us = elapsed.count();
        printf("passed %lf us\n", iteration_us);

        if (!runner.code().ok()) {
            return runner.code();
        }
        auto cycles = (double)timestamp_;
        auto ns = cycles * ns_per_tsc_;
        printf("latency: %.04f cycles / %.04f ns\n", cycles, ns);
        //sleep(1);
    }
    return absl::OkStatus();
}

DEFINE_string(mode, "server", "Act as server or client");
DEFINE_string(shm, "", "The name of the file for the IPC shared memory");

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    auto &plat = ExperimentPlatform::Instance();
    auto stat = plat.Initialize();
    if (!stat.ok()) {
        std::cerr << "Failed to initialize experiment platform: " << stat.ToString() << "\n";
        return -1;
    }

    auto resource_dir = getenv("GPUMPC_RESOURCE_DIR");
    ResNet18Serving::Options options = {
        .module_path = std::string(resource_dir) + kModule,
        .ipc_mem_file = FLAGS_shm,
        .mode = FLAGS_mode == "server" ? ResNet18Serving::Mode::kServer
                                       : ResNet18Serving::Mode::kClient,
    };

    ResNet18Serving test;
    gpumpc::MonadRunner<absl::Status> runner(absl::OkStatus());
    runner.Run([&]() { return test.Initialize(options); })
        .Run([&]() { return test.CalibrateTime(); })
        .Run([&]() { return test.Launch(); });
    stat = runner.code();
    if (!stat.ok()) {
        std::cerr << "Failed to run the benchmark: " << stat.ToString() << "\n";
        return -1;
    }

    return 0;
}
