#include <absl/types/span.h>
#include <chrono>
#include <fstream>
#include <gflags/gflags.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <string>
#include <vector>

namespace {

DEFINE_int32(warmup, 5, "warmup time (in second) for benchmark");
DEFINE_int32(loop, 100, "loop run times for benchmark");

template <typename T> std::string sizeToHuman(T length) {
    std::vector<std::string> array = {"B", "KB", "MB", "GB"};
    int index = 0;
    while (length >= 1024) {
        index += 1;
        length /= 1024;
    }
    return std::to_string(length) + array[index];
}

void RunMemcpyBenchmark(hipDeviceptr_t dst, void *src, unsigned long length) {
    const int kWarmUp = FLAGS_warmup;
    const int kLoop = FLAGS_loop;
    double min_us = std::numeric_limits<double>::max();
    double max_us = std::numeric_limits<double>::min();
    double total_us = 0.0;

    // warm up
    auto warmup_start = std::chrono::high_resolution_clock::now();
    while (1) {
        auto _ = hipMemcpyHtoD(dst, src, length);
        _ = hipDeviceSynchronize();
        (void)_;
        auto warmup_now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = warmup_now - warmup_start;
        double warmup_second = elapsed.count();
        if (warmup_second > kWarmUp) {
            break;
        }
    }

    auto begintime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kLoop; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        auto _ = hipMemcpyHtoD(dst, src, length);
        _ = hipDeviceSynchronize();
        (void)_;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;
        double iteration_us = elapsed.count();
        min_us = std::min(iteration_us, min_us);
        max_us = std::max(iteration_us, max_us);
        total_us = total_us + iteration_us;
    }
    auto endtime = std::chrono::high_resolution_clock::now();

    auto mean = (total_us - min_us - max_us) / (kLoop - 2);
    std::cout << std::fixed << "Summary for " << sizeToHuman(length)
              << ": [min, max, mean] = [" << min_us << ", " << max_us << ", "
              << mean << "] us\n";

    std::chrono::duration<double, std::milli> elapsed = endtime - begintime;
    double total_ms = elapsed.count();
    std::cout << "Total: " << total_ms << "ms\n";

    double bandwidth = length / mean * 1e6;
    uint64_t bandwidth_integer = std::round(bandwidth);
    std::cout << "Bandwidth for " << sizeToHuman(length) << " : "
              << sizeToHuman(bandwidth) << "/s (Human readable)\n";
    std::cout << "Bandwidth for " << length << " : " << bandwidth_integer
              << " B/s (Machine readable)\n";
}

} // namespace

void RunBenchmark() {
    const unsigned long kStart = 4;                      // 4B
    const unsigned long kEnd = 1UL * 1024 * 1024 * 1024; // 1GB

    hipDeviceptr_t mem;

    auto ret = hipMalloc((void **)&mem, kEnd);
    if (ret != hipSuccess) {
        std::cout << "error: Can not allocate device memory\n";
    }

    std::vector<unsigned char> host;
    host.reserve(kEnd);

    /* sanity check */
    ret = hipMemcpyHtoD(mem, host.data(), kEnd);
    if (ret != hipSuccess) {
        std::cout << "error: Can not memcpy\n";
    }
    ret = hipDeviceSynchronize();
    (void)ret;

    for (auto length = kStart; length <= kEnd; length *= 2) {
        RunMemcpyBenchmark(mem, host.data(), length);
    }

    auto _ = hipFree(mem);
    (void)_;
}

int main(int argc, char *argv[]) {
    gflags::SetUsageMessage("Benchmark driver for Memcpy.");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    RunBenchmark();
    return 0;
}
