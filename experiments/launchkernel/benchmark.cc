#include <absl/status/status.h>
#include <chrono>
#include <fstream>
#include <gflags/gflags.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <string>
#include <utils/hip_helper.h>
#include <utils/monad_runner.h>
#include <vector>

using namespace std;
using namespace gpumpc::experiment;

// approximately 10us for one run, so 10s / 10us = 1000000
DEFINE_int32(warmup, 1000000, "warmup times for benchmark");
DEFINE_int32(loop, 3000000, "loop run times for benchmark");

static const char *kModule = "/data/pingpong.bin";
static const char *kFunction = "pingpong";

class LaunchNoOpKernel {
  public:
    absl::Status Initialize();
    absl::Status Close();
    absl::Status RunOnce();
    absl::Status Benchmark(int warm_ups, int loop);

  protected:
    absl::Status LoadModule(hipModule_t *module, std::string module_path);
    absl::Status LoadBinary();

    hipModule_t module_;
    hipFunction_t func_;
};

absl::Status LaunchNoOpKernel::Initialize() { return LoadBinary(); }

absl::Status LaunchNoOpKernel::LoadModule(hipModule_t *module,
                                          std::string module_path) {
    auto resource_dir = getenv("GPUMPC_RESOURCE_DIR");
    if (!resource_dir) {
        return absl::InvalidArgumentError("env var not set");
    }

    auto filepath = string(resource_dir) + module_path;

    std::ifstream elf_file(filepath, ios::binary);
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

absl::Status LaunchNoOpKernel::LoadBinary() {
    auto stat = LoadModule(&module_, kModule);
    if (!stat.ok()) {
        return absl::InvalidArgumentError("Cannot load kernel");
    }

    auto err = hipModuleGetFunction(&func_, module_, kFunction);
    if (err != hipSuccess) {
        return absl::InvalidArgumentError("Cannot load function");
    }
    return absl::OkStatus();
}

absl::Status LaunchNoOpKernel::Close() {
    auto _ = hipModuleUnload(module_);
    (void)_;
    return absl::OkStatus();
}

absl::Status LaunchNoOpKernel::RunOnce() {
    char dummy[0];
    static_assert(sizeof(dummy) == 0, "");
    auto stat = LaunchKernel(func_, 1, 1, dummy);
    if (!stat.ok()) {
        return stat;
    }
    auto err = hipDeviceSynchronize();
    if (err) {
        return absl::InvalidArgumentError("Failed in synchronizing the queue");
    }
    return absl::OkStatus();
}

absl::Status LaunchNoOpKernel::Benchmark(int warm_ups, int loop) {
    double min_us = std::numeric_limits<double>::max();
    double max_us = std::numeric_limits<double>::min();
    double total_us = 0.0;

    // warm up
    for (int i = 0; i < warm_ups; i++) {
        auto stat = RunOnce();
        if (!stat.ok()) {
            return stat;
        }
    }

    auto begintime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loop; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto stat = RunOnce();
        auto end = std::chrono::high_resolution_clock::now();
        if (!stat.ok()) {
            return stat;
        }
        std::chrono::duration<double, std::micro> elapsed = end - start;
        double iteration_us = elapsed.count();
        min_us = std::min(iteration_us, min_us);
        max_us = std::max(iteration_us, max_us);
        total_us = total_us + iteration_us;
    }
    auto endtime = std::chrono::high_resolution_clock::now();
    auto mean = total_us / (double)loop;
    std::cout << std::fixed << "Summary: [min, max, mean] = [" << min_us << ", "
              << max_us << ", " << mean << "] us\n";
    std::chrono::duration<double, std::milli> elapsed = endtime - begintime;
    double total_ms = elapsed.count();
    std::cout << "Total: " << total_ms << "ms\n";

    return absl::OkStatus();
}

int main(int argc, char *argv[]) {
    gflags::SetUsageMessage("Benchmark driver for launch no-op kernel.");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    LaunchNoOpKernel noop;
    gpumpc::MonadRunner<absl::Status> runner(absl::OkStatus());
    runner.Run([&]() { return noop.Initialize(); })
        .Run([&]() {
            std::cout << "Running pingpong benchmark\n";
            return noop.Benchmark(FLAGS_warmup, FLAGS_loop);
        })
        .Run([&]() { return noop.Close(); });

    auto stat = runner.code();
    if (!stat.ok()) {
        std::cerr << "error: failed run the benchmark: " << stat << "\n";
        return -1;
    }

    return 0;
}
