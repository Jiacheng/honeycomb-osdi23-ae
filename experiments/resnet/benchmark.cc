#include "experiments/platform.h"
#include "experiments/resnet/resnet_inference.h"
#include "utils/monad_runner.h"

#include <absl/status/status.h>
#include <absl/types/span.h>
#include <chrono>
#include <fstream>
#include <gflags/gflags.h>
#include <map>

namespace {
static const std::map<
    std::string,
    std::function<std::unique_ptr<gpumpc::experiment::ResNetInference>()>>
    kMapInference = {
        {"resnet1", gpumpc::experiment::NewResNet1},
        {"resnet1-baseline", gpumpc::experiment::NewResNet1Baseline},
        {"resnet18", gpumpc::experiment::NewResNet18},
};
static const std::map<std::string, std::pair<std::string, int>> kMapFiles = {
    {"resnet1", {"data/resnet/image/image.dat", 262144}},
    {"resnet1-baseline", {"data/resnet/image/image.dat", 262144}},
    {"resnet18", {"data/resnet/image/image_1.dat", 4000}},
};

DEFINE_string(bench, "resnet18", "target deep learning network name");
DEFINE_string(warmup, "10", "warmup times for benchmark");
DEFINE_string(loop, "100", "loop run times for benchmark");

static bool ValidateBench(const char *flagname, const std::string &name) {
    if (kMapInference.count(name)) {
        return true;
    } else {
        return false;
    }
}
DEFINE_validator(bench, &ValidateBench);

static std::unique_ptr<gpumpc::experiment::ResNetInference>
CreateInferenceNetwork() {
    return kMapInference.find(FLAGS_bench)->second();
}

absl::Status RunInferenceBenchmark() {
    const int kWarmUp = std::stoi(FLAGS_warmup);
    const int kLoop = std::stoi(FLAGS_loop);
    auto model = CreateInferenceNetwork();
    std::vector<char> result;
    result.resize(kMapFiles.find(FLAGS_bench)->second.second);
    std::vector<char> image;
    auto &plat = gpumpc::experiment::ExperimentPlatform::Instance();
    auto stat = model->Initialize();
    double min_us = std::numeric_limits<double>::max();
    double max_us = std::numeric_limits<double>::min();
    double total_us = 0.0;

    // Warm up
    gpumpc::MonadRunner<absl::Status> runner(absl::OkStatus());
    for (int i = 0; i < kWarmUp; i++) {
        runner
            .Run([&] {
                return plat.LoadResource(
                    kMapFiles.find(FLAGS_bench)->second.first.c_str(), &image);
            })
            .Run([&] { return model->Run(image); })
            .Run([&] {
                return model->Fetch(
                    absl::MakeSpan(result.data(), result.size()));
            });
    }

    if (!runner.code().ok()) {
        return runner.code();
    }

    auto begintime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kLoop; i++) {
        std::chrono::high_resolution_clock::time_point start;
        runner
            .Run([&] {
                start = std::chrono::high_resolution_clock::now();
                return absl::OkStatus();
            })
            .Run([&] {
                return plat.LoadResource(
                    kMapFiles.find(FLAGS_bench)->second.first.c_str(), &image);
            })
            .Run([&] { return model->Run(image); })
            .Run([&] {
                return model->Fetch(
                    absl::MakeSpan(result.data(), result.size()));
            })
            .Run([&] {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::micro> elapsed = end - start;
                double iteration_us = elapsed.count();
                total_us += iteration_us;
                min_us = std::min(iteration_us, min_us);
                max_us = std::max(iteration_us, max_us);
                if (i % 128 == 0) {
                    std::cout << "[" << i << " / " << kLoop
                              << "]: " << iteration_us << "us\n";
                }
                return absl::OkStatus();
            });
    }

    if (!runner.code().ok()) {
        return runner.code();
    }

    auto endtime = std::chrono::high_resolution_clock::now();
    std::cout << "Summary: [min, max, mean] = [" << min_us << ", " << max_us
              << ", " << (total_us - min_us - max_us) / (kLoop - 2) << "] us\n";
    std::chrono::duration<double, std::milli> elapsed = endtime - begintime;
    double total_ms = elapsed.count();
    std::cout << "Total: " << total_ms << "ms\n";
    return model->Close();
}

absl::Status RunBenchmark() {
    if (!kMapInference.count(FLAGS_bench)) {
        return absl::InvalidArgumentError("Benchmark unavailable");
    }
    return RunInferenceBenchmark();
}

} // namespace

int main(int argc, char *argv[]) {
    gflags::SetUsageMessage("Benchmark driver for ResNet.");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    auto stat = gpumpc::experiment::ExperimentPlatform::Instance().Initialize();
    if (!stat.ok()) {
        std::cerr << "Failed to initialize the experiment platform\n";
        return -1;
    }
    stat = RunBenchmark();
    if (!stat.ok()) {
        std::cerr << "Failed to run the benchmark: " << stat.ToString() << "\n";
        return -1;
    }
    return 0;
}
