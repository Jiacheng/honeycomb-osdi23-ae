#include "experiments/platform.h"
#include "experiments/vram-rpc/gpu_direct_dispatch.h"
#include "experiments/resnet/resnet_inference.h"
#include "rpc_resnet18_params.h"
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

static const char *kModule = "/data/rpc_resnet18.bin";
static const char kImagePath[] = "data/resnet/image/image_1.dat";

using namespace gpumpc::rpc;
using namespace gpumpc::experiment;

class ResNet18Serving : public GPUDirectRPC {
  public:
    virtual absl::Status Initialize(const Options &options) override;
    virtual absl::Status LaunchServer() override;
    virtual absl::Status LaunchClient() override;
    void Close();

    absl::Status RespondResult();
    hipFunction_t f_respond_result_;
    std::unique_ptr<ResNetInference> model_;
    void *client_data_;
};

absl::Status ResNet18Serving::Initialize(const Options &options) {
    auto stat = GPUDirectRPC::Initialize(options);
    if (!stat.ok()) {
        return stat;
    }
    client_data_ = nullptr;

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
        auto err = hipModuleGetFunction(&f_respond_result_, module_, "RespondResult");
        if (err) {
            return absl::InvalidArgumentError("Cannot get the RespondResult function");
        }
        return absl::OkStatus();
    }
}

absl::Status ResNet18Serving::LaunchServer() {
#pragma pack(push, 1)
    struct {
        LockFreeQueueView q;
        unsigned long *base;
        unsigned long *image;
    } param = {
        .q = queue_,
        .base = reinterpret_cast<unsigned long *>(mem_->GetBuffer()),
        .image = reinterpret_cast<unsigned long *>(
            model_->GetImageSourceBufferAddr()),
    };
#pragma pack(pop)
    while (true) {
        gpumpc::MonadRunner<absl::Status> runner(absl::OkStatus());
        runner
            .Run([&]() {
                return LaunchKernel(server_, 1, kResNet18RPCServerBlockSize,
                                    param);
            })
            .Run([&]() { return model_->RunDirect(); })
            .Run([&]() { return RespondResult(); });
        auto stat = runner.code();
        if (!stat.ok()) {
            return stat;
        }
        auto err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            return absl::InvalidArgumentError("Cannot sync");
        }
    }
    return absl::OkStatus();
}

absl::Status ResNet18Serving::RespondResult() {
#pragma pack(push, 1)
    struct {
        unsigned long *base;
        unsigned long *image;
    } param = {
        .base = reinterpret_cast<unsigned long *>(mem_->GetBuffer()),
        .image = reinterpret_cast<unsigned long *>(
            model_->GetImageSourceBufferAddr()),
    };
#pragma pack(pop)
    return LaunchKernel(f_respond_result_, 1, kResNet18RPCServerBlockSize,
                        param);
}

absl::Status ResNet18Serving::LaunchClient() {
    enum {
        kGridSize = 1,
        kBlockSize = 512,
    };
#pragma pack(push, 1)
        struct {
            LockFreeQueueView q;
            unsigned long *base;
            const unsigned long *image;
        } param = {
            .q = queue_,
            .base = reinterpret_cast<unsigned long *>(mem_->GetBuffer()),
            .image = reinterpret_cast<unsigned long *>(client_data_), 
        };
#pragma pack(pop)

    std::vector<unsigned long> results(kResNet18ResultSize + sizeof(unsigned long));
    while (true) {
        auto stat = LaunchKernel(client_, 1, kResNet18RPCClientBlockSize, param);
        if (!stat.ok()) {
            return stat;
        }

        gpumpc::MonadRunner<hipError_t> runner(hipSuccess);
        runner
            .Run([&]() {
                return hipMemcpyDtoH(results.data(),
                                     param.base + kRPCTimeOffset,
                                     sizeof(unsigned long) * results.size());
            })
            .Run([&]() { return hipDeviceSynchronize(); });
        if (runner.code() != hipSuccess) {
            return absl::InvalidArgumentError("Cannot sync");
        }
        auto avg_cycles = results[0];
        auto avg_ns = avg_cycles * ns_per_tsc_;
        printf("Average latency: %ld cycles / %.04f ns\n", avg_cycles, avg_ns);
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
    auto &plat = ExperimentPlatform::Instance();
    auto stat = plat.Initialize();
    if (!stat.ok()) {
        std::cerr << "Failed to initialize experiment platform: " << stat.ToString() << "\n";
        return -1;
    }

    auto resource_dir = getenv("GPUMPC_RESOURCE_DIR");
    ResNet18Serving::Options options = {
        .variant = FLAGS_variant == "raw"
                       ? ResNet18Serving::Variant::kRawRPCPerformance
                       : ResNet18Serving::Variant::kRPCWithCapability,
        .module_path = std::string(resource_dir) + kModule,
        .ipc_mem_handle_file = FLAGS_handle,
        .mode = FLAGS_mode == "server" ? ResNet18Serving::Mode::kServer
                                       : ResNet18Serving::Mode::kClient,
        .block_size = FLAGS_blocksize,
    };

    if (options.block_size != 1) {
        std::cerr
            << "The current implementation only works when block size = 1\n";
        return -1;
    }

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
