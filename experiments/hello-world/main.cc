#include "experiments/resnet/resnet_inference.h"
#include "utils/filesystem.h"
#include "utils/hip_helper.h"
#include "utils/monad_runner.h"
#include <absl/status/status.h>
#include <algorithm>
#include <gflags/gflags.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

DEFINE_string(module_file, "", "Name of the binary");

using namespace gpumpc;

class Context {
  public:
    enum { kElements = 1024 };
    absl::Status Initialize(const std::string &fn);
    absl::Status Run();
    Context();
    ~Context();
    hipModule_t module_;
    hipFunction_t func_;
    void *data_;
};

int main(int argc, char *argv[]) {
    gflags::SetUsageMessage("Hello world");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    Context ctx;
    auto stat = ctx.Initialize(FLAGS_module_file);
    if (!stat.ok()) {
        std::cerr << "Cannot intialize the HIP module\n";
        return -1;
    }
    stat = ctx.Run();
    if (!stat.ok()) {
        std::cerr << "Failed to execute:" << stat.ToString() << "\n";
        return -1;
    }

    std::cout << "Success\n";
    return 0;
}

absl::Status Context::Initialize(const std::string &fn) {
    absl::Status stat;
    auto mod = ReadAll(FLAGS_module_file, &stat);
    if (!stat.ok()) {
        return stat;
    }
    MonadRunner<hipError_t> runner(hipSuccess);
    runner.Run([&]() { return hipModuleLoadData(&module_, mod.data()); })
        .Run([&]() {
            return hipModuleGetFunction(&func_, module_, "VectorAdd");
        })
        .Run([&]() { return hipMalloc(&data_, kElements * sizeof(int) * 3); });
    if (runner.code() == hipSuccess) {
        return absl::OkStatus();
    } else {
        return absl::InvalidArgumentError("Cannot initialize the context");
    }
}

absl::Status Context::Run() {
    std::vector<int> src(kElements * 2);
    int count = 0;
    std::generate_n(src.begin(), src.size(), [&count]() { return count++; });
    std::vector<unsigned> result(kElements);
    unsigned *d = reinterpret_cast<unsigned *>(data_);

    MonadRunner<hipError_t> runner(hipSuccess);
    runner
        .Run([&]() {
            return hipMemcpyHtoD(data_, src.data(),
                                 sizeof(int) * kElements * 2);
        })
        .Run([&]() {
#pragma pack(push, 1)
            struct {
                void *c;
                void *a;
                void *b;
            } p = {
                .c = d + 2 * kElements,
                .a = d,
                .b = d + kElements,
            };
#pragma pack(pop)
            auto stat = experiment::LaunchKernel(func_, 64, kElements / 64, p);
            return stat.ok() ? hipSuccess : hipErrorInvalidValue;
        })
        .Run([&]() {
            return hipMemcpyDtoH(result.data(), d + 2 * kElements,
                                 sizeof(int) * kElements);
        })
        .Run([&]() { return hipDeviceSynchronize(); });

    if (runner.code() != hipSuccess) {
        return absl::InvalidArgumentError("Failed to run the kernel");
    }

    for (unsigned i = 0; i < kElements; i++) {
        if (result[i] != i + kElements + i) {
            return absl::InvalidArgumentError("Incorrect result");
        }
    }
    return absl::OkStatus();
}

Context::Context() : module_(nullptr), func_(nullptr), data_(nullptr) {}

Context::~Context() {
    if (data_) {
        auto _ = hipFree(data_);
        (void)_;
    }
    if (module_) {
        auto _ = hipModuleUnload(module_);
        (void)_;
    }
}