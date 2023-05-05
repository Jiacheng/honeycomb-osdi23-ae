#include "gpu_direct_dispatch.h"
#include "utils/hip_helper.h"
#include "utils/monad_runner.h"
#include <fstream>
#include <numeric>

namespace gpumpc::experiment {

using namespace ::gpumpc::rpc;

absl::Status GPUDirectRPC::Initialize(const Options &options) {
    factory_ = CreateHIPSharedMemoryFactory();
    options_ = options;
    auto stat = LoadBinary();
    if (!stat.ok()) {
        return stat;
    }

    if (options_.mode == Mode::kServer) {
        hipIpcMemHandle_t handle;
        auto stat = InitializeServer(&handle);
        if (!stat.ok()) {
            return stat;
        }
        std::ofstream ofs(options_.ipc_mem_handle_file, std::ios::binary);
        ofs.write((const char *)&handle, sizeof(handle));
        ofs.close();
    } else {
        std::ifstream ifs(options_.ipc_mem_handle_file, std::ios::binary);
        std::vector<char> handle((std::istreambuf_iterator<char>(ifs)),
                                 std::istreambuf_iterator<char>());
        hipIpcMemHandle_t h = *(hipIpcMemHandle_t *)handle.data();
        return InitializeClient(h);
    }
    return absl::OkStatus();
}

absl::Status GPUDirectRPC::LoadModule(hipModule_t *module,
                                      const std::string &filepath) {
    std::ifstream elf_file(filepath, std::ios::binary);
    if (!elf_file.is_open()) {
        return absl::InvalidArgumentError("can not open file");
    }

    std::string str((std::istreambuf_iterator<char>(elf_file)),
                    std::istreambuf_iterator<char>());
    elf_file.close();

    auto hipError = hipModuleLoadData(module, str.c_str());
    if (hipError != hipSuccess) {
        return absl::InvalidArgumentError("can not load data");
    }
    return absl::OkStatus();
}

absl::Status GPUDirectRPC::LoadBinary() {
    static const char *kRawRPCFunctions[] = {"PingServerRaw", "PingClientRaw",
                                             "GetClock"};
    static const char *kCapRPCFunctions[] = {"PingServerCap", "PingClientCap",
                                             "GetClock"};
    auto stat = LoadModule(&module_, options_.module_path);
    if (!stat.ok()) {
        return absl::InvalidArgumentError("Cannot load kernel");
    }

    gpumpc::MonadRunner<hipError_t> runner(hipSuccess);
    auto f = options_.variant == Variant::kRawRPCPerformance ? kRawRPCFunctions
                                                             : kCapRPCFunctions;
    runner.Run([&]() { return hipModuleGetFunction(&server_, module_, f[0]); })
        .Run([&]() { return hipModuleGetFunction(&client_, module_, f[1]); })
        .Run(
            [&]() { return hipModuleGetFunction(&get_clock_, module_, f[2]); });

    auto err = runner.code();
    if (err != hipSuccess) {
        return absl::InvalidArgumentError("Cannot load function");
    }
    return absl::OkStatus();
}

absl::Status GPUDirectRPC::CalibrateTime() {
    using clock = std::chrono::high_resolution_clock;
    enum {
        kMeasureTime = 10,
        kIntervalMs = 100,
    };

    struct {
        clock::time_point tp;
        unsigned long tsc;
    } measurements[kMeasureTime];

    void *buf = mem_->GetBuffer();
    for (int i = 0; i < kMeasureTime; i++) {
        measurements[i].tp = std::chrono::high_resolution_clock::now();
        auto stat = LaunchKernel(get_clock_, 1, 1, buf);
        if (!stat.ok()) {
            return stat;
        }

        gpumpc::MonadRunner<hipError_t> runner(hipSuccess);
        runner.Run([&]() { return hipDeviceSynchronize(); }).Run([&]() {
            return hipMemcpyDtoH(&measurements[i].tsc, buf,
                                 sizeof(unsigned long));
        });
        usleep(kIntervalMs * 1000);
        if (runner.code()) {
            return absl::InvalidArgumentError("Cannot calibrate time");
        }
    }
    auto err = hipDeviceSynchronize();
    if (err) {
        return absl::InvalidArgumentError("Cannot calibrate time");
    }

    double r = 0;
    for (int i = 1; i < kMeasureTime; i++) {
        auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         measurements[i].tp - measurements[i - 1].tp)
                         .count();
        auto delta_tsc = measurements[i].tsc - measurements[i - 1].tsc;
        r += (double)delta / delta_tsc;
    }
    ns_per_tsc_ = r / (kMeasureTime - 1);
    printf("Calibrated counter: %.4f ns per GPU clock counter\n", ns_per_tsc_);
    return absl::OkStatus();
}

absl::Status GPUDirectRPC::Launch() {
    return options_.mode == Mode::kServer ? LaunchServer() : LaunchClient();
}

absl::Status GPUDirectRPC::InitializeServer(hipIpcMemHandle_t *handle) {
    auto stat = factory_->CreateSharedMemory(kSharedMemorySize, &mem_);
    if (stat.ok()) {
        *handle =
            *reinterpret_cast<const hipIpcMemHandle_t *>(mem_->GetHandle());
    }
    InitializeQueueView();
    auto err = hipMemset((void *)queue_.desc_, 0, 4096);
    if (err != hipSuccess) {
        return absl::InvalidArgumentError(
            "Cannot clear initialize the descriptor");
    }
    return stat;
}

absl::Status GPUDirectRPC::InitializeClient(hipIpcMemHandle_t handle) {
    auto stat = factory_->AttachSharedMemory(handle, &mem_);
    if (stat.ok()) {
        InitializeQueueView();
    }
    return stat;
}

void GPUDirectRPC::InitializeQueueView() {
    queue_.desc_ = reinterpret_cast<LockFreeQueueView::Descriptor *>(
        reinterpret_cast<char *>(mem_->GetBuffer()) + kSharedMemorySize - 4096);
    queue_.entries_ = reinterpret_cast<LockFreeQueueView::Entry *>(
        reinterpret_cast<char *>((void *)queue_.desc_) -
        LockFreeQueueView::kSize * sizeof(LockFreeQueueView::Entry));
}

absl::Status GPUDirectRPC::LaunchServer() {
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

absl::Status GPUDirectRPC::LaunchClient() {
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

} // namespace gpumpc::experiment